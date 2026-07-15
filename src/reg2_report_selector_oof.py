"""Report-first OOF selector (online, GT-free).

Loads a train-fit HGB+ExtraTrees regression ensemble saved by
``train_report_oof_selector.py``. At inference it rebuilds the same candidate
pool shape as offline (baseline + dx_topk renders + fused clf templates),
scores each candidate, and replaces ONLY the final report.

Modes (env):
  REG2_REPORT_OOF_SELECTOR=1     enable this selector (after v14 dx render)
  REPORT_FIRST_AGGRESSIVE=1      skip path-score veto; always take model argmax
  REG2_REPORT_OOF_MIN_GAIN       min predicted score gain over baseline (default from bundle)

Conservative mode keeps the path-score veto (Nottingham components must match
workflow answers). Aggressive mode is for debug/leaderboard chasing.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_text_calibrator import (
    FINAL_REPORT_QUESTION_CANONICAL,
    LabelContext,
    TextCalibrator,
    canonicalize_question,
    normalize_whitespace,
)
from src.reg2_v11_pipeline import _find_asset_dir, _sanitize_steps, _truthy

ENABLE_ENV = "REG2_REPORT_OOF_SELECTOR"
AGGRESSIVE_ENV = "REPORT_FIRST_AGGRESSIVE"
MIN_GAIN_ENV = "REG2_REPORT_OOF_MIN_GAIN"
DISABLE_ENV = "REG2_DISABLE_REPORT_OOF_SELECTOR"
OTHER_REPORT = "<other_report>"
NONE_LABELS = {"", "<none>", "<other>", "none", "null", "unknown", "n/a"}


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _report_header(report: str) -> str:
    first = report.splitlines()[0] if report.splitlines() else report
    return normalize_whitespace(first.split(";")[0]).lower()


def _final_report_index(steps: list[ChainOfThoughtStep]) -> int | None:
    for i, step in enumerate(steps):
        if canonicalize_question(step.get("question", "")) == FINAL_REPORT_QUESTION_CANONICAL:
            return i
    return None


def _steps_to_pred_steps(steps: list[ChainOfThoughtStep]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for step in steps:
        out.append(
            {
                "question": str(step.get("question", "") or ""),
                "answer": str(step.get("answer", "") or ""),
                "next_question": str(step.get("next_question", "") or ""),
            }
        )
    return out


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
        ("tubule formation", "tubule formation"),
        ("nuclear grade", "nuclear grade"),
        ("mitoses", "mitoses"),
    )
    pred_steps = row.get("pred_steps") or []
    for q_fragment, label in checks:
        if label not in text:
            continue
        expected = _path_score_answer(pred_steps, q_fragment)
        if not expected:
            continue
        m = re.search(rf"\b{re.escape(label)}\s*[:=]?\s*([123])\b", text)
        if m and m.group(1) != expected:
            return False
    return True


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any] | None:
    try:
        model_dir = _find_asset_dir("reg2_report_oof_selector", "report_oof_selector.pkl")
        with (model_dir / "report_oof_selector.pkl").open("rb") as fh:
            obj = pickle.load(fh)
        return obj
    except Exception as exc:
        print(f"[oof-report] unavailable: {type(exc).__name__}: {exc}")
        return None


@lru_cache(maxsize=1)
def _fused_clf() -> dict[str, Any] | None:
    try:
        import torch
        import torch.nn as nn

        model_dir = _find_asset_dir("reg2_report_clf_fused", "report_clf.pt")
        ck = torch.load(model_dir / "report_clf.pt", map_location="cpu", weights_only=False)
        classes = list(ck["classes"])
        mean = np.asarray(ck["scaler_mean"], np.float32)
        std = np.asarray(ck["scaler_std"], np.float32)

        class Head(nn.Module):
            def __init__(self, d: int, h: int, c: int, dp: float):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dp), nn.Linear(h, c))

            def forward(self, x):
                return self.net(x)

        model = Head(mean.shape[0], int(ck["hidden"]), len(classes), float(ck.get("dropout", 0.2)))
        model.load_state_dict(ck["model_state"])
        model.eval()
        return {"torch": torch, "model": model, "classes": classes, "class_to_idx": {r: i for i, r in enumerate(classes)}, "mean": mean, "std": std}
    except Exception as exc:
        print(f"[oof-report] fused clf unavailable: {type(exc).__name__}: {exc}")
        return None


def _fused_probs(fused_feature: np.ndarray) -> np.ndarray | None:
    clf = _fused_clf()
    if clf is None:
        return None
    feat = np.asarray(fused_feature, np.float32).ravel()
    if feat.shape != clf["mean"].shape:
        return None
    torch = clf["torch"]
    with torch.no_grad():
        x = ((feat - clf["mean"]) / clf["std"]).astype(np.float32)
        logits = clf["model"](torch.from_numpy(x[None]))
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def _softmax_pairs(pairs: list[tuple[str, float]]) -> dict[str, float]:
    if not pairs:
        return {}
    xs = np.asarray([float(p) for _, p in pairs], np.float64)
    xs = xs - xs.max()
    e = np.exp(xs)
    probs = e / max(float(e.sum()), 1e-8)
    return {str(l): float(probs[i]) for i, (l, _) in enumerate(pairs)}


def _enrich_row_online(row: dict[str, Any], dx_topk: list[tuple[str, float]]) -> None:
    sm = _softmax_pairs(dx_topk)
    if not sm:
        row["_ranker"] = {}
        return
    labels = list(sm.keys())
    probs = np.asarray([sm[l] for l in labels], np.float32)
    order = np.argsort(-probs)
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    q = np.clip(probs, 1e-8, 1.0)
    row["_ranker"] = {
        "ranker_top1_prob": top1,
        "ranker_top12_margin": top1 - top2,
        "ranker_entropy": float(-(q * np.log(q)).sum() / max(1e-8, np.log(len(q)))),
    }
    top1_i = int(order[0])
    for cand in row["candidates"]:
        dx = str(cand.get("candidate_dx") or row.get("primary") or "")
        p = float(sm.get(dx, 0.0))
        cand["_ranker_prob"] = p
        if dx in labels:
            ci = labels.index(dx)
            cand["_ranker_rank"] = int(np.where(order == ci)[0][0] + 1)
            cand["_ranker_is_top1"] = int(ci == top1_i)
        else:
            cand["_ranker_rank"] = 999
            cand["_ranker_is_top1"] = 0
        cand["_conf_fwd"] = 0.0
        cand["_conf_rev"] = 0.0


def _candidate_features(row: dict[str, Any], cand: dict[str, Any], *, enhanced: bool) -> dict[str, Any]:
    """Mirror exp_report_no_gpu_sweep.feature_dict without importing scripts/."""

    def token_set(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(t) > 1}

    def jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    base_tokens = token_set(row["baseline_report"])
    cand_tokens = token_set(cand["report"])
    primary_tokens = token_set(str(cand.get("candidate_dx") or row.get("primary") or ""))
    organ_tokens = token_set(row.get("organ", ""))
    proc_tokens = token_set(row.get("procedure", ""))
    prob = float(cand.get("prob", 0.0))
    dx_score = float(cand.get("dx_score", 0.0))
    base_prob = float(row.get("baseline_model_prob", 0.0))
    feat = {
        "source": cand.get("source", ""),
        "is_baseline": int(cand.get("source") == "baseline"),
        "has_clf": int(cand.get("has_clf", 0)),
        "has_dx": int(cand.get("has_dx", 0)),
        "rank": int(cand.get("rank", 99)),
        "raw_rank": int(cand.get("raw_rank", 999)),
        "dx_rank": int(cand.get("dx_rank", 99)),
        "prob": prob,
        "log_prob": math.log(max(prob, 1e-8)),
        "dx_score": dx_score,
        "log_dx_score": math.log(max(dx_score, 1e-8)),
        "dx_margin_top1": dx_score - float(row.get("top1_dx_score", 0.0)),
        "top1_dx_score": float(row.get("top1_dx_score", 0.0)),
        "top2_dx_score": float(row.get("top2_dx_score", 0.0)),
        "top12_margin": float(row.get("top1_dx_score", 0.0)) - float(row.get("top2_dx_score", 0.0)),
        "bucket": cand.get("bucket", "unknown"),
        "organ": row.get("organ", ""),
        "procedure": row.get("procedure", ""),
        "primary": row.get("primary", ""),
        "candidate_dx": cand.get("candidate_dx", ""),
        "same_header": int(cand.get("header") == row.get("baseline_header")),
        "same_as_baseline": int(cand["report"] == row["baseline_report"]),
        "word_count": int(cand.get("word_count", 0)),
        "line_count": int(cand.get("line_count", 0)),
        "word_delta_baseline": int(cand.get("word_count", 0)) - int(row.get("baseline_word_count", 0)),
        "line_delta_baseline": int(cand.get("line_count", 0)) - int(row.get("baseline_line_count", 0)),
        "has_note": int(cand.get("has_note", 0)),
        "has_dash": int(cand.get("has_dash", 0)),
        "candidate_train_freq": int(cand.get("train_freq", 0)),
        "candidate_log_train_freq": math.log1p(float(cand.get("train_freq", 0))),
        "baseline_train_freq": int(row.get("baseline_train_freq", 0)),
        "baseline_log_train_freq": math.log1p(float(row.get("baseline_train_freq", 0))),
        "baseline_model_prob": base_prob,
        "prob_minus_baseline": prob - base_prob,
        "prob_ratio_baseline": prob / max(base_prob, 1e-6),
        "token_jaccard_baseline": jaccard(base_tokens, cand_tokens),
        "primary_token_recall": len(primary_tokens & cand_tokens) / max(1, len(primary_tokens)),
        "organ_token_recall": len(organ_tokens & cand_tokens) / max(1, len(organ_tokens)),
        "procedure_token_recall": len(proc_tokens & cand_tokens) / max(1, len(proc_tokens)),
        "source_has_clf": int(bool(cand.get("has_clf")) or "clf" in str(cand.get("source", ""))),
        "source_has_dx": int(bool(cand.get("has_dx")) or "dx_topk" in str(cand.get("source", ""))),
        "is_top1_clf": int((bool(cand.get("has_clf")) or "clf" in str(cand.get("source", ""))) and int(cand.get("rank", 99)) == 1),
        "is_top1_dx": int((bool(cand.get("has_dx")) or "dx_topk" in str(cand.get("source", ""))) and int(cand.get("dx_rank", 99)) == 1),
        "rank_inv": 1.0 / max(1.0, float(cand.get("rank", 99))),
        "dx_rank_inv": 1.0 / max(1.0, float(cand.get("dx_rank", 99))),
        "prob_x_same_header": prob * int(cand.get("header") == row.get("baseline_header")),
        "dx_score_x_same_header": dx_score * int(cand.get("header") == row.get("baseline_header")),
    }
    if enhanced:
        rk = row.get("_ranker") or {}
        feat.update(
            {
                "ranker_top1_prob": float(rk.get("ranker_top1_prob", 0.0)),
                "ranker_top12_margin": float(rk.get("ranker_top12_margin", 0.0)),
                "ranker_entropy": float(rk.get("ranker_entropy", 0.0)),
                "cand_ranker_prob": float(cand.get("_ranker_prob", 0.0)),
                "cand_ranker_rank": int(cand.get("_ranker_rank", 999)),
                "cand_ranker_is_top1": int(cand.get("_ranker_is_top1", 0)),
                "cand_conf_fwd": float(cand.get("_conf_fwd", 0.0)),
                "cand_conf_rev": float(cand.get("_conf_rev", 0.0)),
                "cand_conf_max": max(float(cand.get("_conf_fwd", 0.0)), float(cand.get("_conf_rev", 0.0))),
                "n_candidates": len(row.get("candidates", [])),
            }
        )
    return feat


def _build_candidates(
    *,
    baseline_report: str,
    context: LabelContext,
    calib: TextCalibrator,
    dx_topk: list[tuple[str, float]],
    clf_probs: np.ndarray | None,
    clf_classes: list[str],
    dx_topk_n: int = 12,
    clf_topk_n: int = 12,
    raw_topk_n: int = 96,
) -> list[dict[str, Any]]:
    candidates_by_report: dict[str, dict[str, Any]] = {}

    def add_candidate(**kwargs: Any) -> None:
        report = str(kwargs.pop("report", "") or "")
        source = str(kwargs.pop("source", ""))
        if not report or report == OTHER_REPORT:
            return
        lines = [ln for ln in report.splitlines() if ln.strip()]
        cand = candidates_by_report.get(report)
        if cand is None:
            cand = {
                "source": source,
                "report": report,
                "rank": int(kwargs.get("rank", 99)),
                "raw_rank": int(kwargs.get("raw_rank", 999)),
                "prob": float(kwargs.get("prob", 0.0)),
                "dx_rank": int(kwargs.get("dx_rank", 99)),
                "dx_score": float(kwargs.get("dx_score", 0.0)),
                "candidate_dx": str(kwargs.get("candidate_dx", "") or ""),
                "train_freq": 0,
                "bucket": "unknown",
                "header": _report_header(report),
                "word_count": len(report.split()),
                "line_count": len(lines),
                "has_note": int("note" in report.lower()),
                "has_dash": int("\n -" in report or "\n-" in report),
                "has_clf": int(source == "clf" or "clf" in source),
                "has_dx": int(source == "dx_topk" or "dx_topk" in source),
            }
            candidates_by_report[report] = cand
            return
        if source not in cand["source"]:
            cand["source"] = "+".join([cand["source"], source]) if cand["source"] != source else cand["source"]
        cand["rank"] = min(int(cand["rank"]), int(kwargs.get("rank", 99)))
        cand["prob"] = max(float(cand["prob"]), float(kwargs.get("prob", 0.0)))
        cand["dx_rank"] = min(int(cand["dx_rank"]), int(kwargs.get("dx_rank", 99)))
        cand["dx_score"] = max(float(cand["dx_score"]), float(kwargs.get("dx_score", 0.0)))
        cand["has_clf"] = int(cand["has_clf"] or source == "clf" or "clf" in source)
        cand["has_dx"] = int(cand["has_dx"] or source == "dx_topk" or "dx_topk" in source)

    top1 = dx_topk[0][1] if dx_topk else 0.0
    baseline_model_prob = 0.0
    baseline_model_rank = 9999
    if clf_probs is not None and baseline_report in clf_classes:
        bi = clf_classes.index(baseline_report)
        baseline_model_prob = float(clf_probs[bi])
        order = np.argsort(-clf_probs)
        baseline_model_rank = int(np.where(order == bi)[0][0] + 1)

    add_candidate(
        report=baseline_report,
        source="baseline",
        rank=0,
        raw_rank=baseline_model_rank,
        prob=baseline_model_prob,
        dx_rank=1,
        dx_score=top1,
        candidate_dx=context.primary_dx(),
    )

    for dx_rank, (dx, dx_score) in enumerate(dx_topk[:dx_topk_n], 1):
        if str(dx).strip().lower() in NONE_LABELS:
            continue
        ctx = LabelContext(
            organ=context.organ,
            procedure=context.procedure,
            diagnoses=[dx],
            histologic_type=context.histologic_type,
            grade=context.grade,
            behavior=context.behavior,
        )
        add_candidate(
            report=calib.compose_report(calib.expand_diagnoses(ctx) if hasattr(calib, "expand_diagnoses") else ctx),
            source="dx_topk",
            dx_rank=dx_rank,
            dx_score=float(dx_score),
            candidate_dx=dx,
        )

    if clf_probs is not None:
        order = np.argsort(-clf_probs)
        distinct = 0
        for raw_rank, j in enumerate(order[:raw_topk_n], 1):
            report = str(clf_classes[int(j)])
            if report == OTHER_REPORT or not report:
                continue
            distinct += int(report not in candidates_by_report)
            add_candidate(
                report=report,
                source="clf",
                rank=max(1, distinct),
                raw_rank=raw_rank,
                prob=float(clf_probs[int(j)]),
            )
            if distinct >= clf_topk_n:
                break

    cands = list(candidates_by_report.values())
    cands.sort(key=lambda c: (int(c.get("source") != "baseline"), int(c.get("rank", 99)), int(c.get("dx_rank", 99))))
    return cands


def _predict_scores(bundle: dict[str, Any], row: dict[str, Any]) -> np.ndarray:
    enhanced = bool(bundle.get("enhanced"))
    feats = [_candidate_features(row, c, enhanced=enhanced) for c in row["candidates"]]
    models = bundle["reg_models"]
    preds = [np.asarray(m.predict(feats), np.float32) for m in models.values()]
    return sum(preds) / max(1, len(preds))


def select_final_report_oof(
    *,
    steps: list[ChainOfThoughtStep],
    context: LabelContext,
    dx_scores: dict[str, list[tuple[str, float]]],
    fused_feature: np.ndarray,
    calib: TextCalibrator | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return steps
    if not _truthy(os.environ.get(ENABLE_ENV)):
        return steps
    bundle = _bundle()
    if bundle is None:
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline_report = str(steps[idx].get("answer", "") or "").strip()
    if not baseline_report:
        return steps

    cal = calib or TextCalibrator.from_assets(_find_asset_dir("reg2_calibrator", "answer_calibrator.json"))
    dx_topk = [(str(l), float(p)) for l, p in (dx_scores.get("primary_dx") or []) if str(l).strip()]
    clf = _fused_clf()
    probs = _fused_probs(fused_feature) if clf is not None else None
    classes = list(clf["classes"]) if clf is not None else []

    candidates = _build_candidates(
        baseline_report=baseline_report,
        context=context,
        calib=cal,
        dx_topk=dx_topk,
        clf_probs=probs,
        clf_classes=classes,
    )
    if len(candidates) <= 1:
        return steps

    base_lines = [ln for ln in baseline_report.splitlines() if ln.strip()]
    row = {
        "organ": context.organ,
        "procedure": context.procedure,
        "primary": context.primary_dx(),
        "top1_dx_score": dx_topk[0][1] if dx_topk else 0.0,
        "top2_dx_score": dx_topk[1][1] if len(dx_topk) > 1 else 0.0,
        "baseline_report": baseline_report,
        "baseline_header": _report_header(baseline_report),
        "baseline_word_count": len(baseline_report.split()),
        "baseline_line_count": len(base_lines),
        "baseline_train_freq": 0,
        "baseline_model_prob": float(candidates[0].get("prob", 0.0)),
        "pred_steps": _steps_to_pred_steps(steps),
        "candidates": candidates,
    }
    if bool(bundle.get("enhanced")):
        _enrich_row_online(row, dx_topk)

    pred = _predict_scores(bundle, row)
    best_i = int(np.argmax(pred))
    base_i = 0
    picked = candidates[best_i]
    aggressive = _truthy(os.environ.get(AGGRESSIVE_ENV))
    min_gain = _f(MIN_GAIN_ENV, float(bundle.get("min_gain", 0.0)))

    if best_i != base_i:
        if float(pred[best_i]) < float(pred[base_i]) + min_gain and not aggressive:
            picked = candidates[base_i]
        elif not aggressive and not _path_score_consistent(row, picked):
            order = np.argsort(-pred)
            picked = candidates[base_i]
            for j in order:
                if int(j) == base_i:
                    continue
                cand = candidates[int(j)]
                if _path_score_consistent(row, cand) and float(pred[j]) >= float(pred[base_i]) + min_gain:
                    picked = cand
                    break

    if picked["report"] == baseline_report:
        return steps

    out = copy.deepcopy(steps)
    out[idx]["answer"] = picked["report"]
    mode = "aggressive" if aggressive else "conservative"
    print(
        f"[oof-report] replaced final report mode={mode} "
        f"source={picked.get('source')} pred={float(pred[best_i]):.4f} "
        f"base={float(pred[base_i]):.4f}"
    )
    return _sanitize_steps(out)
