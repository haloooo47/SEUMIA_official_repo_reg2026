"""Slot-aware bounded final-report candidate selector.

This is a fixed, GT-free rule layer:
  1. build the same online candidate pool as the report OOF selector;
  2. choose a conservative path-score-consistent report-clf candidate;
  3. optionally switch to a candidate whose structured report slots agree with
     visual slot-head predictions.

Only the final-report answer is changed. Non-final workflow steps are untouched.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any

import numpy as np

from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_report_selector_fused import select_final_report_fused
from src.reg2_report_selector_oof import (
    _build_candidates,
    _final_report_index,
    _fused_clf,
    _fused_probs,
    _steps_to_pred_steps,
)
from src.reg2_report_slot_heads import predict_slots
from src.reg2_report_slot_patch import extract_breast_nottingham, extract_prostate_slots
from src.reg2_text_calibrator import LabelContext, TextCalibrator
from src.reg2_v11_pipeline import _sanitize_steps, _truthy

DISABLE_ENV = "REG2_DISABLE_SLOT_CANDIDATE_SELECTOR"
ENABLE_ENV = "REG2_SLOT_CANDIDATE_SELECTOR"
MAX_RANK_ENV = "REG2_SLOT_CAND_MAX_RANK"
MIN_CAND_PROB_ENV = "REG2_SLOT_CAND_MIN_PROB"
MIN_JACCARD_ENV = "REG2_SLOT_CAND_MIN_JACCARD"
MIN_GLEASON_ENV = "REG2_SLOT_CAND_MIN_PROB_GLEASON"
MIN_BREAST_ENV = "REG2_SLOT_CAND_MIN_PROB_BREAST"
DISABLE_SECONDARY_MIL_ENV = "REG2_DISABLE_SECONDARY_MIL_CANDIDATE"
SECONDARY_DCIS_TH_ENV = "REG2_SECONDARY_MIL_BREAST_DCIS_TH"


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _enabled() -> bool:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return False
    raw = os.environ.get(ENABLE_ENV, "1")
    return _truthy(raw) or raw == "1"


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(t) > 1}


def _recall(needle: str, haystack: str) -> float:
    a = _token_set(needle)
    b = _token_set(haystack)
    return len(a & b) / max(1, len(a))


def _jaccard(a_text: str, b_text: str) -> float:
    a = _token_set(a_text)
    b = _token_set(b_text)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _path_score_answer(pred_steps: list[dict[str, str]], question_fragment: str) -> str:
    for step in pred_steps:
        q = str(step.get("question", "") or "").lower()
        if question_fragment in q:
            m = re.search(r"\b([123])\b", str(step.get("answer", "") or ""))
            if m:
                return m.group(1)
    return ""


def _path_score_consistent(row: dict[str, Any], cand: dict[str, Any]) -> bool:
    text = str(cand.get("report", "") or "").lower()
    checks = (
        ("score for tubular differentiation", "tubule formation"),
        ("score for nuclear pleomorphism", "nuclear grade"),
        ("score for mitotic rate", "mitoses"),
    )
    for q_fragment, report_label in checks:
        expected = _path_score_answer(row.get("pred_steps") or [], q_fragment)
        if not expected or report_label not in text:
            continue
        m = re.search(rf"{re.escape(report_label)}\s*[:=]\s*([123])", text)
        if m and m.group(1) != expected:
            return False
    return True


def _candidate_features(row: dict[str, Any], cand: dict[str, Any]) -> dict[str, float]:
    report = str(cand.get("report", "") or "")
    base = str(row.get("baseline_report", "") or "")
    return {
        "primary_recall": _recall(str(row.get("primary", "") or ""), report),
        "jaccard": _jaccard(base, report),
        "same_header": float(str(cand.get("header", "")) == str(row.get("baseline_header", ""))),
        "prob_ratio": float(cand.get("prob", 0.0)) / max(float(row.get("baseline_model_prob", 0.0)), 1e-6),
    }


def _has_clf(cand: dict[str, Any]) -> bool:
    src = str(cand.get("source", ""))
    return bool(cand.get("has_clf")) or "clf" in src


def _passes_pathscore(row: dict[str, Any], cand: dict[str, Any]) -> bool:
    if not _has_clf(cand):
        return False
    if int(cand.get("rank", 99)) > 1:
        return False
    if float(cand.get("prob", 0.0)) < 0.70:
        return False
    feat = _candidate_features(row, cand)
    if feat["prob_ratio"] < 3.0:
        return False
    if feat["primary_recall"] < 1.0:
        return False
    if feat["jaccard"] < 0.5:
        return False
    return _path_score_consistent(row, cand)


def _pick_pathscore(row: dict[str, Any]) -> dict[str, Any]:
    picked = row["candidates"][0]
    best_key: tuple[float, ...] | None = None
    baseline = str(row.get("baseline_report", "") or "")
    for cand in row.get("candidates", [])[1:]:
        if str(cand.get("report", "") or "") == baseline:
            continue
        if not _passes_pathscore(row, cand):
            continue
        key = (
            float(cand.get("prob", 0.0)),
            float(cand.get("dx_score", 0.0)),
            -float(cand.get("rank", 99)),
        )
        if best_key is None or key > best_key:
            picked = cand
            best_key = key
    return picked


def _slot_confident(slots: dict[str, Any], key: str, min_prob: float) -> bool:
    rec = slots.get(key) or {}
    pred = str(rec.get("pred", ""))
    return bool(pred and pred != "<other>" and float(rec.get("prob", 0.0)) >= min_prob)


def _gleason_matches(report: str, pred: str) -> bool:
    got = extract_prostate_slots(report).get("gleason_score", "")
    return got.replace(" ", "") == str(pred or "").replace(" ", "")


def _nottingham_matches(report: str, slots: dict[str, dict[str, Any]]) -> bool:
    got = extract_breast_nottingham(report)
    if not got.get("tubule"):
        return False
    want = {
        "tubule": str((slots.get("breast_tubule_score") or {}).get("pred", "")),
        "nuclear": str((slots.get("breast_nuclear_score") or {}).get("pred", "")),
        "mitoses": str((slots.get("breast_mitotic_score") or {}).get("pred", "")),
    }
    return all(want[k] == got.get(k) for k in want if want[k] and want[k] != "<other>")


def _slot_reason(report: str, organ: str, slots: dict[str, dict[str, Any]]) -> str:
    organ_l = str(organ or "").lower()
    if organ_l == "prostate" and _slot_confident(slots, "gleason_score", _f(MIN_GLEASON_ENV, 0.45)):
        pred = str((slots.get("gleason_score") or {}).get("pred", ""))
        if _gleason_matches(report, pred):
            return "prostate_gleason"
    if organ_l == "breast":
        keys = ("breast_tubule_score", "breast_nuclear_score", "breast_mitotic_score")
        min_prob = _f(MIN_BREAST_ENV, 0.75)
        if all(_slot_confident(slots, k, min_prob) for k in keys) and _nottingham_matches(report, slots):
            return "breast_nottingham"
    return ""


def _passes_slot_candidate(
    row: dict[str, Any],
    cand: dict[str, Any],
    *,
    min_prob: float | None = None,
) -> bool:
    if not _has_clf(cand):
        return False
    if int(cand.get("rank", 99)) > _i(MAX_RANK_ENV, 3):
        return False
    threshold = _f(MIN_CAND_PROB_ENV, 0.05) if min_prob is None else min_prob
    if float(cand.get("prob", 0.0)) < threshold:
        return False
    feat = _candidate_features(row, cand)
    if feat["primary_recall"] < 1.0:
        return False
    if feat["jaccard"] < _f(MIN_JACCARD_ENV, 0.30):
        return False
    return _path_score_consistent(row, cand)


def _pick_slot_aligned(
    row: dict[str, Any],
    slots: dict[str, dict[str, Any]],
    fallback: dict[str, Any],
    *,
    min_prob: float | None = None,
) -> tuple[dict[str, Any], str]:
    aligned: list[tuple[tuple[float, ...], dict[str, Any], str]] = []
    for cand in row.get("candidates", []):
        if not _passes_slot_candidate(row, cand, min_prob=min_prob):
            continue
        reason = _slot_reason(str(cand.get("report", "") or ""), str(row.get("organ", "")), slots)
        if not reason:
            continue
        feat = _candidate_features(row, cand)
        key = (
            float(cand.get("prob", 0.0)),
            float(cand.get("dx_score", 0.0)),
            -float(cand.get("rank", 99)),
            feat["jaccard"],
        )
        aligned.append((key, cand, reason))
    if not aligned:
        return fallback, "pathscore"
    _key, cand, reason = max(aligned, key=lambda item: item[0])
    return cand, reason


def _norm_finding(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    text = re.sub(r"[\s,.;:()/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _append_secondary_finding_report(report: str, finding: str) -> str:
    """Append a single secondary diagnosis while preserving the current primary.

    Reports in the submission intentionally use literal ``\\n`` separators.
    The offline eligibility-board candidate for secondary DCIS is exactly a
    final-stage transformation of the current report, not a re-render from the
    original visual LabelContext; preserving the current primary keeps this
    online overlay aligned with the frozen replay.
    """
    text = str(report or "").strip()
    finding_s = str(finding or "").strip()
    if not text or not finding_s:
        return text
    if _norm_finding(finding_s) in _norm_finding(text):
        return text
    sep = "\\n" if "\\n" in text else "\n"
    if sep not in text:
        return f"1. {text}{sep} 2. {finding_s}"
    header, body = text.split(sep, 1)
    body = body.strip()
    if not body:
        return text
    nums = [int(m.group(1)) for m in re.finditer(r"(?:^|\\n|\n)\s*(\d+)\.", text)]
    if nums:
        next_num = max(nums) + 1
        return f"{text}{sep} {next_num}. {finding_s}"
    return f"{header}{sep}  1. {body}{sep}  2. {finding_s}"


def select_final_report_secondary_mil_candidate(
    *,
    steps: list[ChainOfThoughtStep],
    context: LabelContext,
    dx_scores: dict[str, list[tuple[str, float]]],
    fused_feature: np.ndarray,
    calib: TextCalibrator,
    secondary_scores: dict[str, dict[str, Any]],
) -> list[ChainOfThoughtStep]:
    """Add a secondary DCIS report candidate only when visual MIL supports it.

    This is intentionally narrower than the generic eligibility prior: only
    Breast + additional ``Ductal carcinoma in situ`` is allowed, no pairs.
    """
    if _truthy(os.environ.get(DISABLE_SECONDARY_MIL_ENV)):
        return steps
    if str(context.organ or "").strip() != "Breast":
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline_report = str(steps[idx].get("answer", "") or "").strip()
    if not baseline_report:
        return steps

    dcis_key = _norm_finding("Ductal carcinoma in situ")
    support = float((secondary_scores.get(dcis_key) or {}).get("prob", 0.0))
    threshold = _f(SECONDARY_DCIS_TH_ENV, 0.85)
    if support < threshold:
        return steps

    try:
        if _norm_finding(context.primary_dx()) == dcis_key:
            return steps
        picked_report = _append_secondary_finding_report(
            baseline_report,
            "Ductal carcinoma in situ",
        )
        if not picked_report or picked_report == baseline_report:
            return steps

        out = copy.deepcopy(steps)
        out[idx]["answer"] = picked_report
        print(
            "[secondary-mil-candidate] replaced final report "
            f"value=DCIS support={support:.4f} threshold={threshold:.4f} "
            f"mode=append"
        )
        return _sanitize_steps(out)
    except Exception as exc:
        print(f"[secondary-mil-candidate] failed: {type(exc).__name__}: {exc}")
        return steps


def _row_for_candidates(
    *,
    baseline_report: str,
    context: LabelContext,
    steps: list[ChainOfThoughtStep],
    candidates: list[dict[str, Any]],
    dx_topk: list[tuple[str, float]],
) -> dict[str, Any]:
    base_lines = [ln for ln in baseline_report.splitlines() if ln.strip()]
    return {
        "organ": context.organ,
        "procedure": context.procedure,
        "primary": context.primary_dx(),
        "top1_dx_score": dx_topk[0][1] if dx_topk else 0.0,
        "top2_dx_score": dx_topk[1][1] if len(dx_topk) > 1 else 0.0,
        "baseline_report": baseline_report,
        "baseline_header": candidates[0].get("header", "") if candidates else "",
        "baseline_word_count": len(baseline_report.split()),
        "baseline_line_count": len(base_lines),
        "baseline_train_freq": 0,
        "baseline_model_prob": float(candidates[0].get("prob", 0.0)) if candidates else 0.0,
        "pred_steps": _steps_to_pred_steps(steps),
        "candidates": candidates,
    }


def select_final_report_slot_candidate(
    *,
    steps: list[ChainOfThoughtStep],
    context: LabelContext,
    dx_scores: dict[str, list[tuple[str, float]]],
    fused_feature: np.ndarray,
    calib: TextCalibrator,
) -> list[ChainOfThoughtStep]:
    if not _enabled():
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline_report = str(steps[idx].get("answer", "") or "").strip()
    if not baseline_report:
        return steps

    try:
        clf = _fused_clf()
        probs = _fused_probs(fused_feature) if clf is not None else None
        classes = list(clf["classes"]) if clf is not None else []
        dx_topk = [(str(l), float(p)) for l, p in (dx_scores.get("primary_dx") or []) if str(l).strip()]
        candidates = _build_candidates(
            baseline_report=baseline_report,
            context=context,
            calib=calib,
            dx_topk=dx_topk,
            clf_probs=probs,
            clf_classes=classes,
        )
        if len(candidates) <= 1:
            return steps
        row = _row_for_candidates(
            baseline_report=baseline_report,
            context=context,
            steps=steps,
            candidates=candidates,
            dx_topk=dx_topk,
        )
        path_pick = _pick_pathscore(row)
        slots = predict_slots(fused_feature, organ=context.organ) or {}
        picked, reason = _pick_slot_aligned(row, slots, path_pick)
        picked_report = str(picked.get("report", "") or "").strip()
        if not picked_report or picked_report == baseline_report:
            return steps

        out = copy.deepcopy(steps)
        out[idx]["answer"] = picked_report
        print(
            "[slot-candidate] replaced final report "
            f"reason={reason} source={picked.get('source')} "
            f"rank={picked.get('rank')} prob={float(picked.get('prob', 0.0)):.4f}"
        )
        return _sanitize_steps(out)
    except Exception as exc:
        print(f"[slot-candidate] failed; falling back to fused selector: {type(exc).__name__}: {exc}")
        return select_final_report_fused(steps=steps, fused_feature=fused_feature)


def select_final_report_pathscore_candidate(
    *,
    steps: list[ChainOfThoughtStep],
    context: LabelContext,
    dx_scores: dict[str, list[tuple[str, float]]],
    fused_feature: np.ndarray,
    calib: TextCalibrator,
) -> list[ChainOfThoughtStep]:
    """Apply only the conservative path-score candidate rule.

    This is intentionally narrower than ``select_final_report_slot_candidate``:
    it allows high-confidence report-clf candidates whose primary diagnosis and
    path-score constraints agree, but it does not use visual slot heads to pick
    lower-probability Gleason/Nottingham variants.
    """

    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline_report = str(steps[idx].get("answer", "") or "").strip()
    if not baseline_report:
        return steps

    try:
        clf = _fused_clf()
        probs = _fused_probs(fused_feature) if clf is not None else None
        classes = list(clf["classes"]) if clf is not None else []
        dx_topk = [(str(l), float(p)) for l, p in (dx_scores.get("primary_dx") or []) if str(l).strip()]
        candidates = _build_candidates(
            baseline_report=baseline_report,
            context=context,
            calib=calib,
            dx_topk=dx_topk,
            clf_probs=probs,
            clf_classes=classes,
        )
        if len(candidates) <= 1:
            return steps
        row = _row_for_candidates(
            baseline_report=baseline_report,
            context=context,
            steps=steps,
            candidates=candidates,
            dx_topk=dx_topk,
        )
        picked = _pick_pathscore(row)
        picked_report = str(picked.get("report", "") or "").strip()
        if not picked_report or picked_report == baseline_report:
            return steps

        out = copy.deepcopy(steps)
        out[idx]["answer"] = picked_report
        print(
            "[pathscore-candidate] replaced final report "
            f"source={picked.get('source')} rank={picked.get('rank')} "
            f"prob={float(picked.get('prob', 0.0)):.4f}"
        )
        return _sanitize_steps(out)
    except Exception as exc:
        print(f"[pathscore-candidate] failed: {type(exc).__name__}: {exc}")
        return steps

