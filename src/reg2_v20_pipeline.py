"""v20 pipeline with a high-confidence diagnosis-answer MIL rescue.

The rescue changes only the canonical primary-diagnosis answer and leaves the
rendered workflow path and report unchanged.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_text_calibrator import LabelContext, SEP
from src.reg2_v11_pipeline import (
    TEXT_CALIBRATOR_DIR_ENV,
    TEXT_CALIBRATOR_V2_DIR_ENV,
    _env_asset_dir,
    _find_asset_dir,
    _sanitize_steps,
    _truthy,
)
from src.reg2_v17_pipeline import predict_v17_chain_of_thought

DISABLE_ENV = "REG2_DISABLE_V20"
DISABLE_DX_RESCUE_ENV = "REG2_DISABLE_V20_MIL_DX_RESCUE"
ALLOW_CASCADE_FALLBACK_ENV = "REG2_ALLOW_CASCADE_FALLBACK"
DX_RESCUE_PROB_ENV = "REG2_V20_MIL_DX_PROB"
ONLINE_DUMP_DIR_ENV = "REG2_ONLINE_DUMP_DIR"
REPORT_DX_GUARD_ENV = "REG2_REPORT_DX_GUARD"
MIL_DX_ORGAN_GUARD_ENV = "REG2_MIL_DX_ORGAN_GUARD"
ORGAN_REPORT_HEADER_VERIFIER_ENV = "REG2_ORGAN_REPORT_HEADER_VERIFIER"
REPORT_PACKAGE_VERIFIER_ENV = "REG2_REPORT_PACKAGE_VERIFIER"
REPORT_STATE_RERENDER_ENV = "REG2_REPORT_STATE_RERENDER"
POST_ANSWER_DX_REPORT_VERIFIER_ENV = "REG2_POST_ANSWER_DX_REPORT_VERIFIER"
REPORT_BREAST_SCORE_VERIFIER_ENV = "REG2_REPORT_BREAST_SCORE_VERIFIER"
REPORT_FORMAT_CANONICALIZER_ENV = "REG2_REPORT_FORMAT_CANONICALIZER"
REPORT_SNAPSHOT_ROLLBACK_ENV = "REG2_REPORT_SNAPSHOT_ROLLBACK"
PROCEDURE_TOOL_HEADER_GUARD_ENV = "REG2_PROCEDURE_TOOL_HEADER_GUARD"
OPENSET_ORGAN_GUARD_ENV = "REG2_OPENSET_ORGAN_GUARD"
OPENSET_ORGAN_GUARD_EVIDENCE_ENV = "REG2_OPENSET_ORGAN_GUARD_EVIDENCE"
OPENSET_ORGAN_GUARD_REPORT_ENV = "REG2_OPENSET_ORGAN_GUARD_REPORT"
OPENSET_ORGAN_GUARD_PATH_ENV = "REG2_OPENSET_ORGAN_GUARD_PATH"
OPENSET_ORGAN_GUARD_PATH_PROCEDURE_ENV = "REG2_OPENSET_ORGAN_GUARD_PATH_PROCEDURE"
OPENSET_ORGAN_GUARD_POLICY_ENV = "REG2_OPENSET_ORGAN_GUARD_POLICY"
OPENSET_ORGAN_GUARD_SCOPE_ENV = "REG2_OPENSET_ORGAN_GUARD_SCOPE"
OPENSET_ORGAN_GUARD_MIN_PROB_ENV = "REG2_OPENSET_ORGAN_GUARD_MIN_PROB"
OPENSET_ORGAN_GUARD_MIN_MARGIN_ENV = "REG2_OPENSET_ORGAN_GUARD_MIN_MARGIN"
OPENSET_ORGAN_GUARD_MIN_VOTES_ENV = "REG2_OPENSET_ORGAN_GUARD_MIN_VOTES"
OPENSET_ORGAN_GUARD_ORGAN_MIN_PROB_ENV = "REG2_OPENSET_ORGAN_GUARD_ORGAN_MIN_PROB"
OPENSET_ORGAN_GUARD_ORGAN_MIN_MARGIN_ENV = "REG2_OPENSET_ORGAN_GUARD_ORGAN_MIN_MARGIN"
OPENSET_ORGAN_GUARD_ORGAN_MIN_VOTES_ENV = "REG2_OPENSET_ORGAN_GUARD_ORGAN_MIN_VOTES"
OPENSET_DENSE_MAX_PATCHES_ENV = "REG2_OPENSET_DENSE_MAX_PATCHES"
OPENSET_DENSE_SCAN_TARGET_ENV = "REG2_OPENSET_DENSE_SCAN_TARGET"
OPENSET_DENSE_SCAN_BUDGET_ENV = "REG2_OPENSET_DENSE_SCAN_BUDGET_S"
OPENSET_DENSE_SCAN_CAP_ENV = "REG2_OPENSET_DENSE_SCAN_CAP"
OPENSET_DENSE_TISSUE_THRESH_ENV = "REG2_OPENSET_DENSE_TISSUE_THRESH"
OPENSET_DENSE_BATCH_ENV = "REG2_OPENSET_DENSE_BATCH"
VISUAL_FLOOR_VETO_ENV = "REG2_VISUAL_FLOOR_VETO"
VISUAL_FLOOR_VETO_MIN_PROB_ENV = "REG2_VISUAL_FLOOR_VETO_MIN_PROB"
DX_RESCUE_ASSET_NAME = "reg2_h1_mil_rescue"
OPENSET_H1_1024_ASSET_NAME = "abmil_hopt1_1024"
DX_Q_CANONICAL = "what is the #1 diagnosis?"
FINAL_REPORT_Q_CANONICAL = "what is the final pathology report?"
ORGAN_Q_CANONICAL = "what is the organ?"
PROCEDURE_Q_CANONICAL = "what is the procedure?"
HISTO_Q_CANONICAL = "what is the histologic type of neoplasm?"
GRADE_Q_CANONICAL = "what is the grade of neoplasm?"
BEHAVIOR_Q_CANONICAL = "what is the behavior of neoplasm?"
KNOWN_ORGAN_HEADERS = {
    "Breast",
    "Colon",
    "Lung",
    "Prostate",
    "Rectum",
    "Stomach",
    "Urinary bladder",
    "Uterine cervix",
}
COLORECTAL_ORGANS = {"Colon", "Rectum"}
COLORECTAL_DX_FAMILIES = {"adenoma_polyp", "adenocarcinoma", "inflammation", "negative"}
VISUAL_FLOOR_VETO_QS = {
    "is there any abnormality present?",
    "is there any neoplasm present?",
    "what is the behavior of neoplasm?",
    "what is the primary of neoplasm?",
    "what is the histologic type of neoplasm?",
    "what is the #1 diagnosis?",
}


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _clean(raw: object) -> str:
    return " ".join(str(raw or "").strip().split())


def _text_has(raw: object, needle: str) -> bool:
    return needle in _clean(raw).lower()


def _dx_exact(raw: object, value: str) -> bool:
    return _clean(raw).lower() == value.lower()


def _question_key(raw: object) -> str:
    return _clean(raw).lower().rstrip("?")


def _norm_label(raw: object) -> str:
    text = str(raw or "").lower().replace("\\n", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _answer_for(steps: list[dict[str, object]], question: str) -> str:
    key = _question_key(question)
    for step in steps:
        if _question_key(step.get("question")) == key:
            return _clean(step.get("answer"))
    return ""


def _raw_answer_for(steps: list[dict[str, object]], question: str) -> str:
    key = _question_key(question)
    for step in steps:
        if _question_key(step.get("question")) == key:
            return str(step.get("answer") or "")
    return ""


def _set_answer_for(
    steps: list[ChainOfThoughtStep],
    question: str,
    answer: str,
) -> tuple[list[ChainOfThoughtStep], bool]:
    key = _question_key(question)
    answer = _clean(answer)
    if not answer:
        return steps, False
    out = [dict(step) for step in steps]
    for step in out:
        if _question_key(step.get("question")) == key:
            before = _clean(step.get("answer"))
            if before == answer:
                return out, False
            step["answer"] = answer
            return out, True
    return out, False


def _set_final_report_for(
    steps: list[ChainOfThoughtStep],
    report: str,
) -> tuple[list[ChainOfThoughtStep], bool]:
    report = str(report or "").strip()
    if not report:
        return steps, False
    out = [dict(step) for step in steps]
    for step in out:
        if _question_key(step.get("question")) == _question_key(FINAL_REPORT_Q_CANONICAL):
            before = str(step.get("answer") or "")
            if before == report:
                return out, False
            step["answer"] = report
            return out, True
    return out, False


def _context_record(context: LabelContext) -> dict[str, Any]:
    return {
        "organ": _clean(getattr(context, "organ", "")),
        "procedure": _clean(getattr(context, "procedure", "")),
        "diagnoses": [_clean(x) for x in getattr(context, "diagnoses", []) if _clean(x)],
        "histologic_type": _clean(getattr(context, "histologic_type", "")),
        "grade": _clean(getattr(context, "grade", "")),
        "behavior": _clean(getattr(context, "behavior", "")),
    }


def _scores_record(scores: dict[str, list[tuple[str, float]]]) -> dict[str, list[list[Any]]]:
    out: dict[str, list[list[Any]]] = {}
    for key, vals in scores.items():
        out[str(key)] = [[_clean(label), float(prob)] for label, prob in vals]
    return out


def _i_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _norm_report_text(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("\\n", " ")
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    compact = re.sub(r"\s+", " ", text).strip()
    return f" {compact} "


def _report_mentions_dx(report: object, dx: object) -> bool:
    dx_norm = _norm_report_text(dx).strip()
    report_norm = _norm_report_text(report)
    if not dx_norm:
        return True
    if f" {dx_norm} " in report_norm:
        return True

    # Common report/header wording variants that should remain valid while
    # still blocking whole-report swaps to a different lesion family.
    relaxed = dx_norm
    for prefix in ("invasive ",):
        if relaxed.startswith(prefix):
            relaxed = relaxed[len(prefix) :]
            if f" {relaxed} " in report_norm:
                return True
    if dx_norm in {"no tumor present", "no evidence of tumor"}:
        return " no tumor " in report_norm or " no evidence of tumor " in report_norm
    if dx_norm.startswith("fibroepithelial tumor"):
        return " fibroepithelial tumor " in report_norm
    if dx_norm.startswith("invasive urothelial carcinoma"):
        return " invasive urothelial carcinoma " in report_norm
    if dx_norm.startswith("non invasive papillary urothelial carcinoma"):
        return " non invasive papillary urothelial carcinoma " in report_norm
    if "squamous cell carcinoma" in dx_norm:
        return " squamous cell carcinoma " in report_norm
    if "adenocarcinoma" in dx_norm:
        return " adenocarcinoma " in report_norm
    return False


def _report_mentions_dx_strict(report: object, dx: object) -> bool:
    dx_norm = _norm_report_text(dx).strip()
    if not dx_norm:
        return True
    return f" {dx_norm} " in _norm_report_text(report)


def _report_header_organ(report: object) -> str:
    text = str(report or "").replace("\\n", "\n").strip()
    if not text:
        return ""
    first = _clean(re.split(r"[,;\n]", text, maxsplit=1)[0])
    for organ in KNOWN_ORGAN_HEADERS:
        if first.lower() == organ.lower():
            return organ
    return ""


def _procedure_tool_header_guard_enabled() -> bool:
    return _truthy(os.environ.get(PROCEDURE_TOOL_HEADER_GUARD_ENV, "0"))


def _procedure_tool_header_guard_reason(
    *,
    before_report: object,
    after_report: object,
    procedure_before: object,
) -> str:
    if not _procedure_tool_header_guard_enabled():
        return ""
    before_header = _report_header_organ(before_report)
    after_header = _report_header_organ(after_report)
    if not before_header or not after_header or before_header == after_header:
        return ""
    proc_key = _clean(procedure_before).lower()
    if after_header in COLORECTAL_ORGANS and "colonoscopic" not in proc_key:
        return f"colorectal_header_without_colonoscopic_proc:{before_header}->{after_header}"
    return ""


def _guard_procedure_tool_header_change(
    before: list[ChainOfThoughtStep],
    after: list[ChainOfThoughtStep],
    *,
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    before_report = _answer_for(before, FINAL_REPORT_Q_CANONICAL)
    after_report = _answer_for(after, FINAL_REPORT_Q_CANONICAL)
    procedure_before = _answer_for(before, PROCEDURE_Q_CANONICAL)
    reason = _procedure_tool_header_guard_reason(
        before_report=before_report,
        after_report=after_report,
        procedure_before=procedure_before,
    )
    if not reason:
        return after, None
    event = {
        "kind": "procedure_tool_header_guard",
        "reason": reason,
        "procedure_before": procedure_before,
        "before_header": _report_header_organ(before_report),
        "after_header": _report_header_organ(after_report),
    }
    print(
        "[procedure-tool-guard] reverted header-changing patch "
        f"case={case_id or '?'} reason={reason}"
    )
    return before, event


def _vocab_map(labels: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for label in labels:
        key = _norm_label(label)
        if key:
            out[key] = str(label)
    return out


@lru_cache(maxsize=1)
def _label_vocabs_for_report_package() -> dict[str, list[str]]:
    try:
        vocab_dir = _find_asset_dir("reg2_titan_head", "label_vocab.json")
        obj = json.loads((vocab_dir / "label_vocab.json").read_text(encoding="utf-8"))
        vocabs = obj.get("vocabs") if isinstance(obj, dict) else {}
        if not isinstance(vocabs, dict):
            return {}
        return {
            str(key): [str(x) for x in value]
            for key, value in vocabs.items()
            if isinstance(value, list)
        }
    except Exception as exc:
        print(f"[report-package-verifier] failed to load label vocab: {type(exc).__name__}: {exc}")
        return {}


def _report_package_verifier_mode() -> str:
    raw = _clean(os.environ.get(REPORT_PACKAGE_VERIFIER_ENV, "0")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"", "0", "false", "off", "none", "disabled"}:
        return "off"
    if raw in {"organ"}:
        return "organ"
    if raw in {"1", "true", "yes", "on", "organ_procedure", "organ+procedure"}:
        return "organ_procedure"
    if raw in {"full", "full_package"}:
        return "full_package"
    return "off"


def _parse_report_package(report: object) -> dict[str, str]:
    text = str(report or "").replace("\\n", "\n").strip()
    if not text:
        return {"organ": "", "procedure": "", "primary_dx": ""}
    vocabs = _label_vocabs_for_report_package()
    if not vocabs:
        return {"organ": "", "procedure": "", "primary_dx": ""}

    organ_map = _vocab_map([x for x in vocabs.get("organ", []) if x not in {"Anus", "Nipple"}])
    proc_map = _vocab_map(vocabs.get("procedure", []))
    dx_labels = [
        str(x)
        for x in vocabs.get("primary_dx", [])
        if str(x) not in {"<none>", "<other>"}
    ]

    header, sep, body = text.partition(";")
    if not sep:
        first, *rest = text.splitlines()
        header = first
        body = "\n".join(rest)
    header_parts = [_clean(x) for x in header.split(",", 1)]
    organ = organ_map.get(_norm_label(header_parts[0]), "") if header_parts else ""
    procedure = ""
    if len(header_parts) > 1:
        procedure = proc_map.get(_norm_label(header_parts[1]), "")

    first_dx_line = ""
    for line in body.splitlines():
        line = _clean(line)
        if line:
            first_dx_line = line.rstrip(".")
            break
    dx_norm = _norm_label(first_dx_line)
    primary_dx = ""
    if dx_norm:
        best = ""
        best_len = -1
        for label in dx_labels:
            key = _norm_label(label)
            if not key:
                continue
            if dx_norm == key or dx_norm.startswith(key + " "):
                if len(key) > best_len:
                    best = label
                    best_len = len(key)
        primary_dx = best
    return {"organ": organ, "procedure": procedure, "primary_dx": primary_dx}


def _apply_report_package_verifier(
    *,
    steps: list[ChainOfThoughtStep],
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    mode = _report_package_verifier_mode()
    if mode == "off":
        return steps, None
    report = _answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    parsed = _parse_report_package(report)
    before = {
        "organ": _answer_for(steps, ORGAN_Q_CANONICAL),
        "procedure": _answer_for(steps, PROCEDURE_Q_CANONICAL),
        "primary_dx": _answer_for(steps, DX_Q_CANONICAL),
    }
    out = steps
    changed: dict[str, str] = {}
    if parsed["organ"]:
        out, touched = _set_answer_for(out, ORGAN_Q_CANONICAL, parsed["organ"])
        if touched:
            changed["organ"] = parsed["organ"]
    if mode in {"organ_procedure", "full_package"} and parsed["procedure"]:
        out, touched = _set_answer_for(out, PROCEDURE_Q_CANONICAL, parsed["procedure"])
        if touched:
            changed["procedure"] = parsed["procedure"]
    if mode == "full_package" and parsed["primary_dx"]:
        out, touched = _set_answer_for(out, DX_Q_CANONICAL, parsed["primary_dx"])
        if touched:
            changed["primary_dx"] = parsed["primary_dx"]
    if not changed:
        return steps, None
    event = {
        "kind": "report_package_verifier",
        "mode": mode,
        "before": before,
        "parsed": parsed,
        "changed": changed,
    }
    print(
        "[report-package-verifier] aligned answer package to final report "
        f"case={case_id or '?'} mode={mode} changed={sorted(changed)}"
    )
    return out, event


def _coarse_dx_family(dx: object) -> str:
    text = _clean(dx).lower()
    if not text:
        return ""
    if "no evidence" in text or "no tumor" in text or "negative for" in text:
        return "negative"
    if "adenocarcinoma" in text:
        return "adenocarcinoma"
    if "gastritis" in text:
        return "gastritis"
    if "inflammation" in text or "cervicitis" in text:
        return "inflammation"
    if "adenoma" in text or "polyp" in text or "serrated" in text:
        return "adenoma_polyp"
    if "urothelial" in text:
        return "urothelial"
    if "ductal carcinoma in situ" in text or "dcis" in text:
        return "breast_dcis"
    if "invasive carcinoma" in text:
        return "breast_invasive"
    if "squamous cell carcinoma" in text:
        return "squamous_carcinoma"
    return text


def _organ_report_header_verifier_mode() -> str:
    raw = _clean(os.environ.get(ORGAN_REPORT_HEADER_VERIFIER_ENV, "0")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"", "0", "false", "off", "none", "disabled"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "colorectal"}:
        return "colorectal"
    if raw in {"supported", "supported_header", "all"}:
        return "supported_header"
    return "off"


def _colorectal_report_header_guard(
    *,
    organ: str,
    header: str,
    procedure: str,
    primary_dx: str,
) -> bool:
    if organ not in COLORECTAL_ORGANS or header not in COLORECTAL_ORGANS:
        return False
    if organ == header:
        return False
    if "colonoscopic" not in procedure.lower():
        return False
    return _coarse_dx_family(primary_dx) in COLORECTAL_DX_FAMILIES


def _supported_report_header_guard(
    *,
    organ: str,
    header: str,
    procedure: str,
    primary_dx: str,
) -> bool:
    if not organ or not header or organ == header:
        return False
    if organ not in KNOWN_ORGAN_HEADERS or header not in KNOWN_ORGAN_HEADERS:
        return False
    if organ in COLORECTAL_ORGANS or header in COLORECTAL_ORGANS:
        return _colorectal_report_header_guard(
            organ=organ,
            header=header,
            procedure=procedure,
            primary_dx=primary_dx,
        )
    return True


def _apply_report_header_organ_verifier(
    *,
    steps: list[ChainOfThoughtStep],
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    mode = _organ_report_header_verifier_mode()
    if mode == "off":
        return steps, None
    organ = _answer_for(steps, ORGAN_Q_CANONICAL)
    procedure = _answer_for(steps, PROCEDURE_Q_CANONICAL)
    primary_dx = _answer_for(steps, DX_Q_CANONICAL)
    report = _answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    header = _report_header_organ(report)
    if mode == "colorectal":
        allowed = _colorectal_report_header_guard(
            organ=organ,
            header=header,
            procedure=procedure,
            primary_dx=primary_dx,
        )
    else:
        allowed = _supported_report_header_guard(
            organ=organ,
            header=header,
            procedure=procedure,
            primary_dx=primary_dx,
        )
    if not allowed:
        return steps, None
    out, changed = _set_answer_for(steps, ORGAN_Q_CANONICAL, header)
    if not changed:
        return steps, None
    event = {
        "kind": "report_header_organ_verifier",
        "mode": mode,
        "organ_before": organ,
        "organ_after": header,
        "procedure": procedure,
        "primary_dx": primary_dx,
        "dx_family": _coarse_dx_family(primary_dx),
        "report_header": header,
    }
    print(
        "[organ-verifier] aligned organ answer to report header "
        f"case={case_id or '?'} mode={mode} {organ!r} -> {header!r} "
        f"dx={primary_dx!r}"
    )
    return out, event


def _guard_report_dx_consistency(
    before: list[ChainOfThoughtStep],
    after: list[ChainOfThoughtStep],
    *,
    context: LabelContext,
    source: str,
) -> list[ChainOfThoughtStep]:
    if not _truthy(os.environ.get(REPORT_DX_GUARD_ENV, "0")):
        return after
    dx = context.primary_dx()
    if not dx:
        return after
    before_report = _answer_for(before, FINAL_REPORT_Q_CANONICAL)
    after_report = _answer_for(after, FINAL_REPORT_Q_CANONICAL)
    if not after_report or after_report == before_report:
        return after
    before_ok = _report_mentions_dx(before_report, dx)
    after_ok = _report_mentions_dx(after_report, dx)
    if before_ok and not after_ok:
        print(
            "[v20-report-guard] reverted inconsistent report "
            f"source={source} dx={dx!r}"
        )
        return before
    return after


def _report_state_rerender_mode() -> str:
    raw = _clean(os.environ.get(REPORT_STATE_RERENDER_ENV, "0")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"", "0", "false", "off", "none", "disabled"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "if_inconsistent", "inconsistent"}:
        return "if_inconsistent"
    if raw in {"always", "all"}:
        return "always"
    return "off"


def _report_header_matches_state(report: object, organ: str, procedure: str) -> bool:
    text = str(report or "").replace("\\n", "\n").strip()
    if not text:
        return False
    header, sep, _body = text.partition(";")
    if not sep:
        header = text.splitlines()[0] if text.splitlines() else ""
    expected = ", ".join([x for x in (_clean(organ), _clean(procedure)) if x])
    if not expected:
        return True
    return _norm_label(header) == _norm_label(expected)


def _committed_context_from_steps(
    steps: list[ChainOfThoughtStep],
    fallback: LabelContext,
) -> LabelContext:
    organ = _answer_for(steps, ORGAN_Q_CANONICAL) or _clean(getattr(fallback, "organ", ""))
    procedure = _answer_for(steps, PROCEDURE_Q_CANONICAL) or _clean(getattr(fallback, "procedure", ""))
    dx = _answer_for(steps, DX_Q_CANONICAL) or fallback.primary_dx()
    histologic_type = _answer_for(steps, HISTO_Q_CANONICAL) or _clean(
        getattr(fallback, "histologic_type", "")
    )
    grade = _answer_for(steps, GRADE_Q_CANONICAL) or _clean(getattr(fallback, "grade", ""))
    behavior = _answer_for(steps, BEHAVIOR_Q_CANONICAL) or _clean(getattr(fallback, "behavior", ""))
    return LabelContext(
        organ=organ,
        procedure=procedure,
        diagnoses=[dx] if _clean(dx) else [],
        histologic_type=histologic_type,
        grade=grade,
        behavior=behavior,
    )


def _report_state_inconsistent(steps: list[ChainOfThoughtStep]) -> bool:
    report = _answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    organ = _answer_for(steps, ORGAN_Q_CANONICAL)
    procedure = _answer_for(steps, PROCEDURE_Q_CANONICAL)
    dx = _answer_for(steps, DX_Q_CANONICAL)
    if not report:
        return True
    if not _report_header_matches_state(report, organ, procedure):
        return True
    if dx and not _report_mentions_dx(report, dx):
        return True
    return False


def _report_snapshot_state_key(steps: list[ChainOfThoughtStep]) -> tuple[str, str, str]:
    return (
        _norm_label(_answer_for(steps, ORGAN_Q_CANONICAL)),
        _norm_label(_answer_for(steps, PROCEDURE_Q_CANONICAL)),
        _norm_label(_answer_for(steps, DX_Q_CANONICAL)),
    )


def _remember_consistent_report_snapshot(
    snapshots: dict[tuple[str, str, str], str],
    steps: list[ChainOfThoughtStep],
) -> None:
    if not _truthy(os.environ.get(REPORT_SNAPSHOT_ROLLBACK_ENV, "0")):
        return
    if _report_state_inconsistent(steps):
        return
    key = _report_snapshot_state_key(steps)
    report = _raw_answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    if all(key) and report:
        snapshots[key] = report


def _restore_consistent_report_snapshot(
    *,
    snapshots: dict[tuple[str, str, str], str],
    steps: list[ChainOfThoughtStep],
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    if not _truthy(os.environ.get(REPORT_SNAPSHOT_ROLLBACK_ENV, "0")):
        return steps, None
    if not _report_state_inconsistent(steps):
        return steps, None
    key = _report_snapshot_state_key(steps)
    report = snapshots.get(key)
    if not report:
        return steps, None
    before_report = _raw_answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    out, changed = _set_final_report_for(steps, report)
    if not changed or _report_state_inconsistent(out):
        return steps, None
    event = {
        "kind": "report_snapshot_rollback",
        "state_key": list(key),
        "before_report": before_report,
        "after_report": report,
    }
    print(
        "[report-snapshot-rollback] restored last consistent report "
        f"case={case_id or '?'} state={key!r}"
    )
    return out, event


def _apply_report_state_rerender(
    *,
    steps: list[ChainOfThoughtStep],
    context: LabelContext,
    calib: Any,
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    mode = _report_state_rerender_mode()
    if mode == "off":
        return steps, None
    inconsistent = _report_state_inconsistent(steps)
    if mode == "if_inconsistent" and not inconsistent:
        return steps, None
    committed = _committed_context_from_steps(steps, context)
    before_report = _answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    new_report = calib.compose_report(committed)
    if not new_report:
        return steps, None
    dx = committed.primary_dx()
    if dx and _report_mentions_dx(before_report, dx) and not _report_mentions_dx(new_report, dx):
        return steps, None
    out, changed = _set_final_report_for(steps, new_report)
    if not changed:
        return steps, None
    event = {
        "kind": "report_state_rerender",
        "mode": mode,
        "inconsistent_before": inconsistent,
        "committed_context": _context_record(committed),
        "before_report": before_report,
        "after_report": new_report,
    }
    print(
        "[report-state-rerender] rerendered final report "
        f"case={case_id or '?'} mode={mode} inconsistent={inconsistent}"
    )
    return out, event


def _diagnosis_answers_from_steps(steps: list[ChainOfThoughtStep]) -> list[str]:
    pairs: list[tuple[int, str]] = []
    for step in steps:
        key = _question_key(step.get("question"))
        match = re.match(r"^what is the #(\d+) diagnosis$", key)
        if not match:
            continue
        answer = _clean(step.get("answer"))
        if answer:
            pairs.append((int(match.group(1)), answer))
    out: list[str] = []
    seen: set[str] = set()
    for _idx, answer in sorted(pairs):
        norm = _norm_label(answer)
        if norm and norm not in seen:
            out.append(answer)
            seen.add(norm)
    return out


def _report_header_line_or_expected(report: object, organ: str, procedure: str) -> str:
    text = str(report or "").replace("\\n", "\n").strip()
    header = ""
    if text:
        header = _clean(text.split(";", 1)[0] if ";" in text else text.splitlines()[0])
    if header and _report_header_organ(header):
        return header
    return ", ".join(x for x in (_clean(organ), _clean(procedure)) if x) or "Pathology report"


def _first_report_body_line(report: object) -> str:
    text = str(report or "").replace("\\n", "\n").strip()
    if not text:
        return ""
    _header, sep, body = text.partition(";")
    if not sep:
        lines = text.splitlines()[1:]
    else:
        lines = body.splitlines()
    for line in lines:
        line = _clean(line).rstrip(".")
        if line:
            return line
    return ""


def _dx_with_compatible_report_detail(dx: str, before_report: object) -> str:
    dx = _clean(dx)
    if not dx:
        return ""
    dx_key = _norm_label(dx)
    if "dysplasia" in dx_key:
        return dx
    adenoma_like = (
        "adenoma" in dx_key
        or "serrated lesion" in dx_key
        or "serrated adenoma" in dx_key
    )
    if not adenoma_like:
        return dx
    before_key = _norm_label(_first_report_body_line(before_report))
    for phrase in ("with high grade dysplasia", "with low grade dysplasia"):
        if _norm_label(phrase) in before_key:
            return f"{dx} {phrase}"
    return dx


def _render_report_from_answer_package(steps: list[ChainOfThoughtStep], before_report: object) -> str:
    diagnoses = _diagnosis_answers_from_steps(steps)
    if not diagnoses:
        return ""
    diagnoses = [
        _dx_with_compatible_report_detail(dx, before_report) if idx == 0 else dx
        for idx, dx in enumerate(diagnoses)
    ]
    organ = _answer_for(steps, ORGAN_Q_CANONICAL)
    procedure = _answer_for(steps, PROCEDURE_Q_CANONICAL)
    header = _report_header_line_or_expected(before_report, organ, procedure)
    if len(diagnoses) == 1:
        body = diagnoses[0]
    else:
        body = "\\n".join(f"{idx}. {dx}" for idx, dx in enumerate(diagnoses, start=1))
    return f"{header};\\n  {body}"


def _apply_post_answer_dx_report_verifier(
    *,
    steps: list[ChainOfThoughtStep],
    answer_only_dx_rescue_event: dict[str, Any] | None,
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    if not _truthy(os.environ.get(POST_ANSWER_DX_REPORT_VERIFIER_ENV, "0")):
        return steps, None
    if answer_only_dx_rescue_event is None:
        return steps, None
    final_dx = _answer_for(steps, DX_Q_CANONICAL)
    candidate_dx = _clean(answer_only_dx_rescue_event.get("candidate_dx"))
    if not final_dx or _norm_label(final_dx) != _norm_label(candidate_dx):
        return steps, None
    before_report = _raw_answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    if _report_mentions_dx_strict(before_report, final_dx):
        return steps, None
    after_report = _render_report_from_answer_package(steps, before_report)
    if not after_report or _report_mentions_dx_strict(after_report, final_dx) is False:
        return steps, None
    out, changed = _set_final_report_for(steps, after_report)
    if not changed:
        return steps, None
    event = {
        "kind": "post_answer_dx_report_verifier",
        "trigger": "answer_only_dx_rescue",
        "base_dx": _clean(answer_only_dx_rescue_event.get("base_dx")),
        "candidate_dx": candidate_dx,
        "final_dx": final_dx,
        "before_report": before_report,
        "after_report": after_report,
    }
    print(
        "[post-answer-dx-report-verifier] rerendered rescued dx report "
        f"case={case_id or '?'} dx={final_dx!r}"
    )
    return out, event


def _parse_breast_nottingham_from_report(report: object) -> dict[str, str]:
    text = str(report or "").replace("\\n", "\n")
    if _report_header_organ(text) != "Breast":
        return {}
    norm = _norm_report_text(text)
    if " invasive carcinoma " not in norm:
        return {}
    scores: dict[str, str] = {}
    patterns = {
        "tubule": r"tubule formation\s*:\s*([123])",
        "nuclear": r"nuclear grade\s*:\s*([123])",
        "mitotic": r"mitoses\s*:\s*([123])",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return {}
        scores[key] = match.group(1)
    total = sum(int(scores[key]) for key in ("tubule", "nuclear", "mitotic"))
    scores["overall"] = str(total)
    grade_match = re.search(r"\bgrade\s*(I{1,3}|1|2|3)\b", text, flags=re.IGNORECASE)
    if grade_match:
        raw_grade = grade_match.group(1).upper()
        roman = {"1": "I", "2": "II", "3": "III"}.get(raw_grade, raw_grade)
    elif total <= 5:
        roman = "I"
    elif total <= 7:
        roman = "II"
    else:
        roman = "III"
    scores["grade"] = f"Grade {roman}"
    return scores


def _apply_report_breast_score_verifier(
    *,
    steps: list[ChainOfThoughtStep],
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    if not _truthy(os.environ.get(REPORT_BREAST_SCORE_VERIFIER_ENV, "0")):
        return steps, None
    report = _raw_answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    parsed = _parse_breast_nottingham_from_report(report)
    if not parsed:
        return steps, None
    updates = {
        "What is the score for tubular differentiation?": parsed["tubule"],
        "What is the score for nuclear pleomorphism?": parsed["nuclear"],
        "What is the score for mitotic rate?": parsed["mitotic"],
        "What is the overall score?": parsed["overall"],
        GRADE_Q_CANONICAL: parsed["grade"],
    }
    out = steps
    changed: dict[str, dict[str, str]] = {}
    for question, value in updates.items():
        before = _answer_for(out, question)
        out, touched = _set_answer_for(out, question, value)
        if touched:
            changed[question] = {"before": before, "after": value}
    if not changed:
        return steps, None
    event = {
        "kind": "report_breast_score_verifier",
        "parsed": parsed,
        "changed": changed,
        "final_report": report,
    }
    print(
        "[report-breast-score-verifier] aligned breast score slots to report "
        f"case={case_id or '?'} changed={len(changed)}"
    )
    return out, event


def _report_format_canonicalizer_enabled() -> bool:
    return _truthy(os.environ.get(REPORT_FORMAT_CANONICALIZER_ENV, "0"))


def _canonicalize_final_report_format(report: object) -> str:
    text = str(report or "").strip()
    if not text:
        return ""
    text = re.sub(r";\\n *", r";\\n  ", text)
    text = re.sub(r";\n *", ";\n  ", text)
    return text


def _apply_report_format_canonicalizer(
    *,
    steps: list[ChainOfThoughtStep],
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    if not _report_format_canonicalizer_enabled():
        return steps, None
    before_report = _raw_answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    after_report = _canonicalize_final_report_format(before_report)
    if not after_report or after_report == before_report:
        return steps, None
    out, changed = _set_final_report_for(steps, after_report)
    if not changed:
        return steps, None
    event = {
        "kind": "report_format_canonicalizer",
        "mode": "semicolon_newline_double_space",
        "before_report": before_report,
        "after_report": after_report,
    }
    print(
        "[report-format] canonicalized final report spacing "
        f"case={case_id or '?'}"
    )
    return out, event


def _dump_online_state(case_id: str | None, stage: str, payload: dict[str, Any]) -> None:
    root = os.environ.get(ONLINE_DUMP_DIR_ENV, "").strip()
    if not root:
        return
    cid = _clean(case_id) or "unknown"
    safe_cid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in cid)
    try:
        out_dir = Path(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{safe_cid}.{stage}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[v20-dump] failed stage={stage}: {type(exc).__name__}: {exc}")


def _dump_report_transition(
    case_id: str | None,
    stage: str,
    before: list[ChainOfThoughtStep],
    after: list[ChainOfThoughtStep],
    *,
    context: LabelContext,
    extra: dict[str, Any] | None = None,
) -> None:
    before_report = _answer_for(before, FINAL_REPORT_Q_CANONICAL)
    after_report = _answer_for(after, FINAL_REPORT_Q_CANONICAL)
    payload: dict[str, Any] = {
        "case_id": case_id,
        "stage": stage,
        "context": _context_record(context),
        "before_dx_answer": _answer_for(before, DX_Q_CANONICAL),
        "after_dx_answer": _answer_for(after, DX_Q_CANONICAL),
        "before_report": before_report,
        "after_report": after_report,
        "changed": before_report != after_report,
    }
    if extra:
        payload["extra"] = extra
    _dump_online_state(case_id, f"report_{stage}", payload)


def _dump_emergency_fallback(
    case_id: str | None,
    exc: Exception,
    *,
    fallback: str,
    cascade_allowed: bool,
) -> None:
    _dump_online_state(
        case_id,
        "emergency_fallback",
        {
            "case_id": case_id,
            "stage": "emergency_fallback",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "fallback": fallback,
            "cascade_allowed": bool(cascade_allowed),
        },
    )


def _broad_family(dx: str) -> str:
    d = dx.lower()
    if "no tumor" in d or "nonspecific inflammation" in d or "gastritis" in d:
        return "benign_inflammatory"
    if "adenocarcinoma" in d or "carcinoma" in d or "squamous cell carcinoma" in d:
        return "malignant"
    if "adenoma" in d or "polyp" in d or "papilloma" in d:
        return "benign_neoplasm"
    if "intraepithelial lesion" in d or "dysplasia" in d or "carcinoma in situ" in d:
        return "preinvasive"
    return "other"


@lru_cache(maxsize=1)
def _dx_to_organs_from_calibrator() -> dict[str, set[str]]:
    try:
        from src.reg2_v17_pipeline import _text_calibrator as _v17_text_calibrator

        report_modes = getattr(_v17_text_calibrator(), "report_modes", {}) or {}
    except Exception as exc:
        print(f"[v20-organ-guard] could not load calibrator map: {type(exc).__name__}: {exc}")
        return {}
    out: dict[str, set[str]] = {}
    for key in report_modes:
        parts = str(key).split(SEP)
        organ = ""
        dx = ""
        if len(parts) >= 4 and parts[0] == "L3":
            organ, dx = _clean(parts[1]), _clean(parts[3])
        elif len(parts) >= 3 and parts[0] == "L4":
            organ, dx = _clean(parts[1]), _clean(parts[2])
        if organ and dx:
            out.setdefault(dx, set()).add(organ)
    return out


def _dx_allowed_for_context_organ(base_organ: str, cand_dx: str) -> bool:
    base = _clean(base_organ)
    dx = _clean(cand_dx)
    if not base or not dx:
        return True
    inferred = _dx_to_organs_from_calibrator().get(dx, set())
    if not inferred:
        return True
    if base in inferred:
        return True
    return bool(base in {"Colon", "Rectum"} and {"Colon", "Rectum"} & inferred)


class H1MilDxRescue:
    """Deploy-format H1 ABMIL dx head used only as a high-confidence rescue."""

    def __init__(self, model_dir: str | Path):
        import json
        import torch
        import torch.nn as nn

        model_dir = Path(model_dir)
        ck = torch.load(model_dir / "abmil_heads.pt", map_location="cpu", weights_only=False)
        self.torch = torch
        vocab_obj = json.loads((model_dir / "label_vocab.json").read_text(encoding="utf-8"))[
            "vocabs"
        ]
        self.vocabs = {str(k): [str(x) for x in v] for k, v in vocab_obj.items()}
        self.vocab = self.vocabs.get("primary_dx", [])
        self.mean = np.asarray(ck["scaler_mean"], np.float32)
        self.std = np.asarray(ck["scaler_std"], np.float32)
        self.max_patches = int(ck.get("max_patches", 256))

        class GatedABMIL(nn.Module):
            def __init__(self, d: int, h: int, head_sizes: dict[str, int]):
                super().__init__()
                self.fc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.25))
                self.att_V = nn.Linear(h, h)
                self.att_U = nn.Linear(h, h)
                self.att_w = nn.Linear(h, 1)
                self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in head_sizes.items()})

            def forward(self, x, mask):
                h = self.fc(x)
                a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))
                a = a.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
                a = torch.softmax(a, dim=1)
                z = (a * h).sum(1)
                return {k: layer(z) for k, layer in self.heads.items()}

        self.model = GatedABMIL(self.mean.shape[0], int(ck["hidden"]), ck["head_sizes"])
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

    def predict(self, patch_features: np.ndarray) -> tuple[str, float, float]:
        pf = np.asarray(patch_features, np.float32)[: self.max_patches]
        return self._predict_head_array("primary_dx", pf)

    def predict_budget(self, patch_features: np.ndarray, max_patches: int = 0) -> tuple[str, float, float]:
        pf = np.asarray(patch_features, np.float32)
        if max_patches > 0:
            pf = pf[:max_patches]
        return self._predict_head_array("primary_dx", pf)

    def predict_head_budget(
        self,
        head: str,
        patch_features: np.ndarray,
        max_patches: int = 0,
    ) -> tuple[str, float, float]:
        pf = np.asarray(patch_features, np.float32)
        if max_patches > 0:
            pf = pf[:max_patches]
        return self._predict_head_array(head, pf)

    def predict_heads_budget(
        self,
        heads: list[str] | tuple[str, ...],
        patch_features: np.ndarray,
        max_patches: int = 0,
    ) -> dict[str, tuple[str, float, float]]:
        pf = np.asarray(patch_features, np.float32)
        if max_patches > 0:
            pf = pf[:max_patches]
        if pf.ndim != 2 or pf.shape[0] == 0 or pf.shape[1] != self.mean.shape[0]:
            return {head: ("", 0.0, 0.0) for head in heads}
        x = ((pf - self.mean) / self.std)[None].astype(np.float32)
        mask = np.ones((1, pf.shape[0]), np.float32)
        with self.torch.no_grad():
            logits_by_head = self.model(
                self.torch.from_numpy(x),
                self.torch.from_numpy(mask),
            )
        out: dict[str, tuple[str, float, float]] = {}
        for head in heads:
            if head not in self.vocabs or head not in logits_by_head:
                out[head] = ("", 0.0, 0.0)
                continue
            probs = self.torch.softmax(logits_by_head[head], dim=1).cpu().numpy()[0]
            order = np.argsort(-probs)
            i0 = int(order[0])
            i1 = int(order[1]) if len(order) > 1 else i0
            out[head] = (self.vocabs[head][i0], float(probs[i0]), float(probs[i0] - probs[i1]))
        return out

    def _predict_head_array(self, head: str, pf: np.ndarray) -> tuple[str, float, float]:
        if head not in self.vocabs:
            return "", 0.0, 0.0
        if pf.ndim != 2 or pf.shape[0] == 0 or pf.shape[1] != self.mean.shape[0]:
            return "", 0.0, 0.0
        x = ((pf - self.mean) / self.std)[None].astype(np.float32)
        mask = np.ones((1, pf.shape[0]), np.float32)
        with self.torch.no_grad():
            logits = self.model(
                self.torch.from_numpy(x),
                self.torch.from_numpy(mask),
            )[head]
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]
        order = np.argsort(-probs)
        i0 = int(order[0])
        i1 = int(order[1]) if len(order) > 1 else i0
        return self.vocabs[head][i0], float(probs[i0]), float(probs[i0] - probs[i1])


@lru_cache(maxsize=1)
def _mil_dx_rescue() -> H1MilDxRescue:
    model_dir = _find_asset_dir(DX_RESCUE_ASSET_NAME, "abmil_heads.pt")
    return H1MilDxRescue(model_dir)


@lru_cache(maxsize=1)
def _openset_h1_1024_critic() -> H1MilDxRescue:
    model_dir = _find_asset_dir(OPENSET_H1_1024_ASSET_NAME, "abmil_heads.pt")
    return H1MilDxRescue(model_dir)


def _openset_guard_enabled() -> bool:
    return _truthy(os.environ.get(OPENSET_ORGAN_GUARD_ENV, "0"))


def _openset_guard_evidence() -> str:
    raw = _clean(os.environ.get(OPENSET_ORGAN_GUARD_EVIDENCE_ENV, "online")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"dense", "dense_h1", "multibudget", "multi_budget"}:
        return "dense"
    return "online"


def _openset_guard_scope() -> str:
    raw = _clean(os.environ.get(OPENSET_ORGAN_GUARD_SCOPE_ENV, "squamous_preinvasive")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"all", "all_cervix", "cervix"}:
        return "all_cervix"
    if raw in {"squamous", "squamous_preinvasive", "preinvasive", "lsil_hsil"}:
        return "squamous_preinvasive"
    return "squamous_preinvasive"


def _openset_report_mode() -> str:
    raw = _clean(os.environ.get(OPENSET_ORGAN_GUARD_REPORT_ENV, "family_compatible")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"preserve", "preserve_existing", "path_only"}:
        return "preserve_existing"
    if raw in {"family", "family_fallback", "typed"}:
        return "family"
    if raw in {"compatible", "family_compatible"}:
        return "family_compatible"
    if raw in {"generic", "coarse", "uterine_body", "1", "true", "yes", "on"}:
        return "generic"
    if raw in {"off", "none", "disabled", "0", "false"}:
        return "off"
    return "generic"


def _openset_path_mode() -> str:
    raw = _clean(os.environ.get(OPENSET_ORGAN_GUARD_PATH_ENV, "off")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"1", "true", "yes", "on", "all", "all_family", "graph", "full"}:
        return "all_family"
    if raw in {"adeno", "adenocarcinoma", "adeno_only"}:
        return "adeno_only"
    if raw in {"adeno_plus_non_neoplasm", "adeno_non_neoplasm", "non_neoplasm"}:
        return "adeno_plus_non_neoplasm"
    if raw in {"off", "none", "disabled", "0", "false", "report_only"}:
        return "off"
    return "off"


def _openset_path_procedure() -> str:
    raw = _clean(os.environ.get(OPENSET_ORGAN_GUARD_PATH_PROCEDURE_ENV, "Biopsy"))
    if not raw:
        return "Biopsy"
    if raw.lower().replace(" ", "_") in {"biopsy", "sample_biopsy"}:
        return "Biopsy"
    if raw.lower().replace(" ", "_") in {"curette_biopsy"}:
        return "Curette biopsy"
    if raw.lower().replace(" ", "_") in {"curettage"}:
        return "Curettage"
    return raw


def _openset_guard_policy() -> str:
    raw = _clean(os.environ.get(OPENSET_ORGAN_GUARD_POLICY_ENV, "family")).lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"train_organ", "train_organ_consensus", "organ_consensus", "organ_head"}:
        return "train_organ_consensus"
    if raw in {"family", "dx_family", "legacy", "0", "false", "off"}:
        return "family"
    return "family"


def _openset_family(dx: object) -> str:
    text = _clean(dx).lower()
    if not text:
        return ""
    if (
        "low-grade squamous intraepithelial lesion" in text
        or "high-grade squamous intraepithelial lesion" in text
        or " lsil" in f" {text}"
        or " hsil" in f" {text}"
        or "cin " in text
        or " cin" in text
        or "endocervical" in text
        or "cervicitis" in text
    ):
        return "cervix"
    if "squamous cell carcinoma" in text:
        return "cervix_compatible_squamous"
    if "endometrioid" in text:
        return "adenocarcinoma"
    if "adenocarcinoma" in text or "serous carcinoma" in text or "carcinosarcoma" in text:
        return "adenocarcinoma"
    if "adenoma" in text or "polyp" in text or "serrated" in text or "papilloma" in text:
        return "adenoma_polyp"
    if "leiomyoma" in text or "smooth muscle" in text:
        return "smooth_muscle"
    if (
        "proliferative endometrium" in text
        or "secretory endometrium" in text
        or "decidualized endometrium" in text
        or "negative for malignancy" in text
        or "no evidence" in text
        or "no tumor" in text
    ):
        return "benign_endometrium"
    if "columnar cell lesion" in text or "usual ductal hyperplasia" in text:
        return "benign_epithelial"
    if "ductal" in text or "lobular" in text or "fibroepithelial" in text or "breast" in text:
        return "breast"
    if "non-small cell" in text or "small cell carcinoma" in text or "pulmonary" in text:
        return "lung"
    if "urothelial" in text:
        return "urothelial"
    if "gastritis" in text:
        return "gastritis"
    if "inflammation" in text or "colitis" in text or "granuloma" in text:
        return "inflammation"
    if "lymphoma" in text:
        return "lymphoma"
    return text


def _openset_family_is_cervix_compatible(family: str) -> bool:
    return family in {"cervix", "cervix_compatible_squamous"}


def _openset_family_is_uterine_compatible(family: str) -> bool:
    return family in {
        "adenocarcinoma",
        "adenoma_polyp",
        "smooth_muscle",
        "benign_endometrium",
        "benign_epithelial",
        "inflammation",
        "lymphoma",
    }


def _openset_base_in_scope(organ: str, dx: str) -> bool:
    if _clean(organ) != "Uterine cervix":
        return False
    scope = _openset_guard_scope()
    if scope == "all_cervix":
        return True
    family = _openset_family(dx)
    return family == "cervix"


def _openset_family_report(family: str, procedure: str = "biopsy") -> str:
    mode = _openset_report_mode()
    if mode in {"off", "preserve_existing"}:
        return ""
    if mode == "family_compatible" and not _openset_family_is_uterine_compatible(family):
        return ""
    report_procedure = (_clean(procedure) or "biopsy").lower()
    if mode in {"family", "family_compatible"}:
        if family == "adenocarcinoma":
            return f"Endometrium, {report_procedure};\\n  Endometrioid adenocarcinoma"
        if family == "adenoma_polyp":
            return f"Endometrium, {report_procedure};\\n  Endometrial polyp"
        if family == "smooth_muscle":
            return f"Uterus, {report_procedure};\\n  Smooth muscle lesion"
        if family in {"benign_endometrium", "benign_epithelial", "inflammation"}:
            return f"Endometrium, {report_procedure};\\n  Proliferative endometrium, negative for malignancy"
        if family == "lymphoma":
            return f"Uterus/endometrium, {report_procedure};\\n  Malignant lymphoma"
    return f"Uterus/endometrium, {report_procedure};\\n  Endometrial tissue"


def _cot_step(question: str, answer: str, next_question: str) -> ChainOfThoughtStep:
    return {"question": question, "answer": answer, "next_question": next_question}


def _openset_endometrial_malignant_steps(report: str, procedure: str) -> list[ChainOfThoughtStep]:
    return [
        _cot_step("What is the organ?", "Endometrium", "Is there any abnormality present?"),
        _cot_step("What is the procedure?", procedure, "Is there any abnormality present?"),
        _cot_step(
            "Is there any abnormality present?",
            "Yes, there is an abnormality.",
            "Is there any neoplasm present?",
        ),
        _cot_step(
            "Is there any neoplasm present?",
            "Yes, there is a neoplasm.",
            "What is the behavior of neoplasm?",
        ),
        _cot_step(
            "Is there any neoplasm present?",
            "Yes, there is a neoplasm.",
            "What is the primary of neoplasm?",
        ),
        _cot_step("What is the behavior of neoplasm?", "Malignant", "What is the histologic type of neoplasm?"),
        _cot_step("What is the primary of neoplasm?", "Endometrial", "What is the histologic type of neoplasm?"),
        _cot_step(
            "What is the histologic type of neoplasm?",
            "Endometrioid adenocarcinoma",
            "Is there any additional finding present?",
        ),
        _cot_step(
            "Is there any additional finding present?",
            "No, there is no additional finding.",
            "What is the number of diagnoses to includes?",
        ),
        _cot_step("What is the number of diagnoses to includes?", "1", "What is the #1 diagnosis?"),
        _cot_step("What is the #1 diagnosis?", "Endometrioid adenocarcinoma", "What is the final pathology report?"),
        _cot_step("What is the final pathology report?", report, ""),
    ]


def _openset_endometrial_polyp_steps(report: str, procedure: str) -> list[ChainOfThoughtStep]:
    return [
        _cot_step("What is the organ?", "Endometrium", "Is there any abnormality present?"),
        _cot_step("What is the procedure?", procedure, "Is there any abnormality present?"),
        _cot_step(
            "Is there any abnormality present?",
            "Yes, there is an abnormality.",
            "Is there any neoplasm present?",
        ),
        _cot_step(
            "Is there any neoplasm present?",
            "Yes, there is a neoplasm.",
            "What is the behavior of neoplasm?",
        ),
        _cot_step(
            "Is there any neoplasm present?",
            "Yes, there is a neoplasm.",
            "What is the primary of neoplasm?",
        ),
        _cot_step("What is the behavior of neoplasm?", "Benign", "What is the histologic type of neoplasm?"),
        _cot_step("What is the primary of neoplasm?", "Endometrial", "What is the histologic type of neoplasm?"),
        _cot_step(
            "What is the histologic type of neoplasm?",
            "Endometrial polyp",
            "Is there any additional finding present?",
        ),
        _cot_step(
            "Is there any additional finding present?",
            "No, there is no additional finding.",
            "What is the number of diagnoses to includes?",
        ),
        _cot_step("What is the number of diagnoses to includes?", "1", "What is the #1 diagnosis?"),
        _cot_step("What is the #1 diagnosis?", "Endometrial polyp", "What is the final pathology report?"),
        _cot_step("What is the final pathology report?", report, ""),
    ]


def _openset_endometrial_pattern_steps(report: str, procedure: str) -> list[ChainOfThoughtStep]:
    return [
        _cot_step("What is the organ?", "Endometrium", "Is there any abnormality present?"),
        _cot_step("What is the procedure?", procedure, "Is there any abnormality present?"),
        _cot_step(
            "Is there any abnormality present?",
            "No, there is no abnormality.",
            "What is the histologic pattern of endometrium?",
        ),
        _cot_step(
            "What is the histologic pattern of endometrium?",
            "Proliferative pattern",
            "What is the number of diagnoses to includes?",
        ),
        _cot_step("What is the number of diagnoses to includes?", "1", "What is the #1 diagnosis?"),
        _cot_step("What is the #1 diagnosis?", "Proliferative endometrium", "What is the final pathology report?"),
        _cot_step("What is the final pathology report?", report, ""),
    ]


def _openset_family_steps(family: str, report: str) -> list[ChainOfThoughtStep] | None:
    mode = _openset_path_mode()
    if mode == "off":
        return None
    procedure = _openset_path_procedure()
    if family == "adenocarcinoma":
        return _openset_endometrial_malignant_steps(report, procedure)
    if mode == "all_family" and family == "adenoma_polyp":
        return _openset_endometrial_polyp_steps(report, procedure)
    if mode in {"all_family", "adeno_plus_non_neoplasm"} and family in {
        "benign_endometrium",
        "benign_epithelial",
        "inflammation",
    }:
        return _openset_endometrial_pattern_steps(report, procedure)
    return None


def _openset_critic_vote(
    *,
    source: str,
    dx: str,
    prob: float,
    margin: float,
    organ: str = "",
    organ_prob: float = 0.0,
    organ_margin: float = 0.0,
    require_confidence: bool = True,
) -> dict[str, Any] | None:
    family = _openset_family(dx)
    if not family or _openset_family_is_cervix_compatible(family):
        return None
    if require_confidence:
        min_prob = _f(OPENSET_ORGAN_GUARD_MIN_PROB_ENV, 0.70)
        min_margin = _f(OPENSET_ORGAN_GUARD_MIN_MARGIN_ENV, 0.10)
        if prob < min_prob or margin < min_margin:
            return None
    return {
        "source": source,
        "dx": _clean(dx),
        "family": family,
        "prob": float(prob),
        "margin": float(margin),
        "organ": _clean(organ),
        "organ_prob": float(organ_prob),
        "organ_margin": float(organ_margin),
    }


def _extract_openset_dense_h1_features(wsi_path: Path) -> np.ndarray:
    import tiffslide

    from src.reg2_hoptimus_online import _encode_tiles
    from src.reg2_hoptimus_online import load_hoptimus
    from src.reg2_titan_online import collect_patches
    from src.reg2_titan_online import patch_px_for_mpp
    from src.reg2_titan_online import slide_mpp

    max_patches = _i_env(OPENSET_DENSE_MAX_PATCHES_ENV, 1024)
    scan_target = _i_env(OPENSET_DENSE_SCAN_TARGET_ENV, 4096)
    scan_cap = _i_env(OPENSET_DENSE_SCAN_CAP_ENV, 0)
    scan_budget_s = _f(OPENSET_DENSE_SCAN_BUDGET_ENV, 80.0)
    tissue_thresh = _f(OPENSET_DENSE_TISSUE_THRESH_ENV, 0.10)
    batch = _i_env(OPENSET_DENSE_BATCH_ENV, 16)
    with tiffslide.TiffSlide(str(wsi_path)) as slide:
        mpp = slide_mpp(slide)
        patch_px = patch_px_for_mpp(mpp)
        tiles, _coords, scanned = collect_patches(
            slide,
            max_patches=max_patches,
            tissue_thresh=tissue_thresh,
            patch_px=patch_px,
            scan_target=scan_target,
            scan_cap=scan_cap,
            scan_budget_s=scan_budget_s,
        )
    if not tiles:
        return np.zeros((0, 1536), np.float32)
    model, device = load_hoptimus()
    feats = _encode_tiles(model, device, tiles, batch)
    print(
        "[openset-organ-guard] dense H1 evidence "
        f"patches={feats.shape[0]} scanned={scanned} patch_px={patch_px}"
    )
    return np.asarray(feats, np.float32)


def _openset_dense_guard_votes(wsi_path: Path) -> list[dict[str, Any]]:
    try:
        patch_features = _extract_openset_dense_h1_features(wsi_path)
    except Exception as exc:
        print(f"[openset-organ-guard] dense evidence unavailable: {type(exc).__name__}: {exc}")
        return []
    if patch_features.ndim != 2 or patch_features.shape[0] == 0:
        return []
    votes: list[dict[str, Any]] = []
    budget_sources = [
        ("h1_rescue_b128", _mil_dx_rescue, 128),
        ("h1_rescue_b256", _mil_dx_rescue, 256),
        ("h1_rescue_b512", _mil_dx_rescue, 512),
        ("h1_rescue_b1024", _mil_dx_rescue, 1024),
        ("h1_rescue_ball", _mil_dx_rescue, 0),
        ("h1_abmil1024_b128", _openset_h1_1024_critic, 128),
        ("h1_abmil1024_b256", _openset_h1_1024_critic, 256),
        ("h1_abmil1024_b512", _openset_h1_1024_critic, 512),
        ("h1_abmil1024_b1024", _openset_h1_1024_critic, 1024),
        ("h1_abmil1024_ball", _openset_h1_1024_critic, 0),
    ]
    for source, loader, budget in budget_sources:
        try:
            pred = loader().predict_heads_budget(["primary_dx", "organ"], patch_features, budget)
            cand_dx, prob, margin = pred.get("primary_dx", ("", 0.0, 0.0))
            cand_organ, organ_prob, organ_margin = pred.get("organ", ("", 0.0, 0.0))
        except Exception as exc:
            print(f"[openset-organ-guard] dense critic unavailable source={source}: {type(exc).__name__}: {exc}")
            continue
        vote = _openset_critic_vote(
            source=source,
            dx=cand_dx,
            prob=prob,
            margin=margin,
            organ=cand_organ,
            organ_prob=organ_prob,
            organ_margin=organ_margin,
            require_confidence=False,
        )
        if vote is not None:
            votes.append(vote)
    return votes


def _openset_guard_votes(
    *,
    patch_features: np.ndarray,
    wsi_path: Path | None = None,
) -> list[dict[str, Any]]:
    if _openset_guard_evidence() == "dense" and wsi_path is not None:
        votes = _openset_dense_guard_votes(wsi_path)
        if votes:
            return votes
    votes: list[dict[str, Any]] = []
    critics = [
        ("h1_abmil_256", _mil_dx_rescue),
        ("h1_abmil_1024", _openset_h1_1024_critic),
    ]
    for source, loader in critics:
        try:
            pred = loader().predict_heads_budget(["primary_dx", "organ"], patch_features)
            cand_dx, prob, margin = pred.get("primary_dx", ("", 0.0, 0.0))
            cand_organ, organ_prob, organ_margin = pred.get("organ", ("", 0.0, 0.0))
        except Exception as exc:
            print(f"[openset-organ-guard] critic unavailable source={source}: {type(exc).__name__}: {exc}")
            continue
        vote = _openset_critic_vote(
            source=source,
            dx=cand_dx,
            prob=prob,
            margin=margin,
            organ=cand_organ,
            organ_prob=organ_prob,
            organ_margin=organ_margin,
        )
        if vote is not None:
            votes.append(vote)
    return votes


def _visual_floor_veto_enabled() -> bool:
    return _truthy(os.environ.get(VISUAL_FLOOR_VETO_ENV, "0"))


def _visual_floor_superfamily(raw: object) -> str:
    fam = _coarse_dx_family(raw)
    if fam in {"negative", "inflammation", "gastritis"}:
        return "non_neoplastic"
    if fam == "adenoma_polyp":
        return "benign_neoplasm"
    if fam in {"breast_dcis"}:
        return "preinvasive"
    text = _clean(raw).lower()
    if "intraepithelial" in text or "dysplasia" in text or "carcinoma in situ" in text:
        return "preinvasive"
    if (
        "carcinoma" in text
        or "adenocarcinoma" in text
        or "sarcoma" in text
        or "lymphoma" in text
        or "gastrointestinal stromal tumor" in text
    ):
        return "malignant"
    return fam


def _visual_floor_core_diffs(
    floor_steps: list[ChainOfThoughtStep],
    candidate_steps: list[ChainOfThoughtStep],
) -> dict[str, tuple[str, str]]:
    diffs: dict[str, tuple[str, str]] = {}
    for question in sorted(VISUAL_FLOOR_VETO_QS):
        before = _answer_for(floor_steps, question)
        after = _answer_for(candidate_steps, question)
        if before != after:
            diffs[_question_key(question)] = (before, after)
    return diffs


def _visual_floor_path_slot_drift_is_bad(
    *,
    floor_steps: list[ChainOfThoughtStep],
    candidate_steps: list[ChainOfThoughtStep],
    floor_dx: str,
) -> bool:
    floor_dx_key = _norm_label(floor_dx)
    floor_histo = _answer_for(floor_steps, HISTO_Q_CANONICAL)
    cand_histo = _answer_for(candidate_steps, HISTO_Q_CANONICAL)
    if floor_histo and cand_histo and _norm_label(floor_histo) != _norm_label(cand_histo):
        if _norm_label(cand_histo) and _norm_label(cand_histo) not in floor_dx_key:
            return True
    cand_behavior = _clean(_answer_for(candidate_steps, BEHAVIOR_Q_CANONICAL)).lower()
    floor_super = _visual_floor_superfamily(floor_dx)
    if cand_behavior == "malignant" and floor_super not in {"malignant", "preinvasive"}:
        return True
    if cand_behavior == "benign" and floor_super == "malignant":
        return True
    return False


def _visual_floor_prediction(patch_features: np.ndarray) -> dict[str, Any] | None:
    try:
        pred = _openset_h1_1024_critic().predict_heads_budget(
            ["primary_dx", "organ"],
            patch_features,
            0,
        )
    except Exception as exc:
        print(f"[visual-floor-veto] critic unavailable: {type(exc).__name__}: {exc}")
        return None
    dx, prob, margin = pred.get("primary_dx", ("", 0.0, 0.0))
    organ, organ_prob, organ_margin = pred.get("organ", ("", 0.0, 0.0))
    return {
        "pred_dx": _clean(dx),
        "pred_prob": float(prob),
        "pred_margin": float(margin),
        "pred_family": _coarse_dx_family(dx),
        "pred_superfamily": _visual_floor_superfamily(dx),
        "organ": _clean(organ),
        "organ_prob": float(organ_prob),
        "organ_margin": float(organ_margin),
    }


def _apply_visual_floor_veto(
    *,
    floor_steps: list[ChainOfThoughtStep],
    steps: list[ChainOfThoughtStep],
    patch_features: np.ndarray,
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    if not _visual_floor_veto_enabled():
        return steps, None
    diffs = _visual_floor_core_diffs(floor_steps, steps)
    if not diffs:
        return steps, None
    visual = _visual_floor_prediction(patch_features)
    if not visual:
        return steps, None
    min_prob = _f(VISUAL_FLOOR_VETO_MIN_PROB_ENV, 0.95)
    if float(visual.get("pred_prob") or 0.0) < min_prob:
        return steps, None

    floor_dx = _answer_for(floor_steps, DX_Q_CANONICAL)
    cand_dx = _answer_for(steps, DX_Q_CANONICAL)
    floor_key = _norm_label(floor_dx)
    cand_key = _norm_label(cand_dx)
    visual_key = _norm_label(visual.get("pred_dx"))
    floor_family = _coarse_dx_family(floor_dx)
    cand_family = _coarse_dx_family(cand_dx)
    visual_family = _coarse_dx_family(visual.get("pred_dx"))
    floor_super = _visual_floor_superfamily(floor_dx)
    cand_super = _visual_floor_superfamily(cand_dx)
    visual_super = _visual_floor_superfamily(visual.get("pred_dx"))
    dx_changed = bool(floor_key and cand_key and floor_key != cand_key)

    reason = ""
    if floor_key and visual_key == floor_key:
        if dx_changed:
            reason = "visual_exact_floor_dx_changed"
        elif _visual_floor_path_slot_drift_is_bad(
            floor_steps=floor_steps,
            candidate_steps=steps,
            floor_dx=floor_dx,
        ):
            reason = "visual_exact_floor_bad_path_slot_drift"
    elif dx_changed and floor_family and visual_family == floor_family and cand_family != floor_family:
        reason = "visual_family_floor_cross_family"
    elif (
        dx_changed
        and floor_super
        and visual_super == floor_super
        and cand_super
        and cand_super != floor_super
    ):
        reason = f"visual_super_floor_cross_super:{floor_super}->{cand_super}"
    if not reason:
        return steps, None
    event = {
        "kind": "visual_floor_veto",
        "reason": reason,
        "min_prob": float(min_prob),
        "floor_dx": floor_dx,
        "candidate_dx": cand_dx,
        "visual": visual,
        "diff_keys": sorted(diffs),
    }
    print(
        "[visual-floor-veto] reverted late core drift "
        f"case={case_id or '?'} reason={reason} floor={floor_dx!r} cand={cand_dx!r} "
        f"visual={visual.get('pred_dx')!r} p={float(visual.get('pred_prob') or 0.0):.3f}"
    )
    return [dict(step) for step in floor_steps], event


def _openset_same_train_organ(a: object, b: object) -> bool:
    aa = _clean(a)
    bb = _clean(b)
    if not aa or not bb:
        return False
    if aa == bb:
        return True
    return bool({aa, bb} <= {"Colon", "Rectum"})


def _openset_train_organ_contradiction(
    *,
    votes: list[dict[str, Any]],
    base_organ: str,
    evidence: str,
) -> tuple[bool, dict[str, Any]]:
    if _openset_guard_policy() != "train_organ_consensus":
        return True, {"policy": _openset_guard_policy(), "required": False}
    min_prob = _f(OPENSET_ORGAN_GUARD_ORGAN_MIN_PROB_ENV, 0.55)
    min_margin = _f(OPENSET_ORGAN_GUARD_ORGAN_MIN_MARGIN_ENV, 0.05)
    default_min_votes = 3 if evidence == "dense" else 1
    min_votes = max(1, _i_env(OPENSET_ORGAN_GUARD_ORGAN_MIN_VOTES_ENV, default_min_votes))
    organ_votes: list[dict[str, Any]] = []
    base = _clean(base_organ)
    for vote in votes:
        organ = _clean(vote.get("organ"))
        if not organ or _openset_same_train_organ(base, organ):
            continue
        if float(vote.get("organ_prob") or 0.0) < min_prob:
            continue
        if float(vote.get("organ_margin") or 0.0) < min_margin:
            continue
        organ_votes.append(vote)
    counts: dict[str, int] = {}
    for vote in organ_votes:
        organ = _clean(vote.get("organ"))
        counts[organ] = counts.get(organ, 0) + 1
    detail = {
        "policy": _openset_guard_policy(),
        "required": True,
        "base_organ": base,
        "min_prob": float(min_prob),
        "min_margin": float(min_margin),
        "min_votes": int(min_votes),
        "non_base_organ_vote_count": len(organ_votes),
        "non_base_organ_counts": counts,
        "non_base_organ_votes": organ_votes,
    }
    return len(organ_votes) >= min_votes, detail


def _apply_open_set_organ_guard(
    *,
    steps: list[ChainOfThoughtStep],
    context: LabelContext,
    patch_features: np.ndarray,
    wsi_path: Path | None = None,
    case_id: str | None,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    if not _openset_guard_enabled():
        return steps, None
    report_mode = _openset_report_mode()
    if report_mode == "off":
        return steps, None
    organ = _answer_for(steps, ORGAN_Q_CANONICAL) or _clean(getattr(context, "organ", ""))
    dx = _answer_for(steps, DX_Q_CANONICAL) or context.primary_dx()
    if not _openset_base_in_scope(organ, dx):
        return steps, None
    evidence = _openset_guard_evidence()
    votes = _openset_guard_votes(patch_features=patch_features, wsi_path=wsi_path)
    default_min_votes = 3 if evidence == "dense" else 2
    min_votes = max(1, _i_env(OPENSET_ORGAN_GUARD_MIN_VOTES_ENV, default_min_votes))
    by_family: dict[str, list[dict[str, Any]]] = {}
    for vote in votes:
        by_family.setdefault(str(vote["family"]), []).append(vote)
    family = ""
    family_votes: list[dict[str, Any]] = []
    for cand_family, cand_votes in sorted(
        by_family.items(),
        key=lambda item: (len(item[1]), max(float(v["prob"]) for v in item[1])),
        reverse=True,
    ):
        if len(cand_votes) >= min_votes:
            family = cand_family
            family_votes = cand_votes
            break
    if not family:
        return steps, None
    organ_ok, organ_detail = _openset_train_organ_contradiction(
        votes=family_votes,
        base_organ=organ,
        evidence=evidence,
    )
    if not organ_ok:
        print(
            "[openset-organ-guard] abstained; train-organ evidence insufficient "
            f"case={case_id or '?'} organ={organ!r} dx={dx!r} "
            f"family={family} family_votes={len(family_votes)} "
            f"non_base_organ_votes={organ_detail.get('non_base_organ_vote_count', 0)}"
        )
        return steps, None
    path_mode = _openset_path_mode()
    before_report = _raw_answer_for(steps, FINAL_REPORT_Q_CANONICAL)
    if report_mode == "preserve_existing":
        report = before_report
    else:
        report = _openset_family_report(
            family,
            procedure=_openset_path_procedure() if path_mode != "off" else "biopsy",
        )
    if not report:
        return steps, None
    path_steps = _openset_family_steps(family, report)
    if path_steps is not None:
        event = {
            "kind": "open_set_organ_guard",
            "mode": "path_graph",
            "path_mode": path_mode,
            "path_procedure": _openset_path_procedure(),
            "evidence": evidence,
            "scope": _openset_guard_scope(),
            "report_mode": report_mode,
            "policy": _openset_guard_policy(),
            "organ_evidence": organ_detail,
            "organ": organ,
            "base_dx": dx,
            "base_family": _openset_family(dx),
            "selected_family": family,
            "votes": family_votes,
            "all_votes": votes,
            "before_report": before_report,
            "after_report": report,
        }
        print(
            "[openset-organ-guard] path-graph fallback "
            f"case={case_id or '?'} organ={organ!r} dx={dx!r} "
            f"family={family} votes={len(family_votes)} mode={path_mode}"
        )
        return path_steps, event
    out, changed = _set_final_report_for(steps, report)
    if not changed:
        return steps, None
    event = {
        "kind": "open_set_organ_guard",
        "mode": "report_only",
        "evidence": evidence,
        "scope": _openset_guard_scope(),
        "report_mode": report_mode,
        "policy": _openset_guard_policy(),
        "organ_evidence": organ_detail,
        "organ": organ,
        "base_dx": dx,
        "base_family": _openset_family(dx),
        "selected_family": family,
        "votes": family_votes,
        "all_votes": votes,
        "before_report": before_report,
        "after_report": report,
    }
    print(
        "[openset-organ-guard] report-only fallback "
        f"case={case_id or '?'} organ={organ!r} dx={dx!r} "
        f"family={family} votes={len(family_votes)} mode={report_mode}"
    )
    return out, event


def _apply_mil_dx_rescue_to_steps(
    steps: list[ChainOfThoughtStep],
    context: LabelContext,
    *,
    patch_features: np.ndarray,
) -> tuple[list[ChainOfThoughtStep], dict[str, Any] | None]:
    """Change only the primary-diagnosis answer after report rendering."""
    if _truthy(os.environ.get(DISABLE_DX_RESCUE_ENV)):
        return steps, None
    base_dx = context.primary_dx()
    if not base_dx:
        return steps, None
    try:
        cand_dx, prob, margin = _mil_dx_rescue().predict(patch_features)
    except Exception as exc:
        print(f"[v20-mil] rescue failed: {type(exc).__name__}: {exc}")
        return steps, None

    threshold = _f(DX_RESCUE_PROB_ENV, 0.85)
    if not cand_dx or cand_dx == base_dx or prob < threshold:
        return steps, None
    if _truthy(os.environ.get(MIL_DX_ORGAN_GUARD_ENV, "0")) and not _dx_allowed_for_context_organ(context.organ, cand_dx):
        print(
            "[v20-mil] rejected cross-organ dx-only rescue "
            f"organ={context.organ or '?'} base={base_dx!r} cand={cand_dx!r} prob={prob:.3f}"
        )
        return steps, None
    if _broad_family(cand_dx) != _broad_family(base_dx):
        return steps, None

    out: list[ChainOfThoughtStep] = []
    changed = False
    for step in steps:
        item = dict(step)
        if _question_key(item.get("question")) == _question_key(DX_Q_CANONICAL):
            item["answer"] = cand_dx
            changed = True
        out.append(item)  # type: ignore[arg-type]
    if not changed:
        return steps, None

    print(
        "[v20-mil] dx-only rescue "
        f"organ={context.organ or '?'} {base_dx!r} -> {cand_dx!r} "
        f"prob={prob:.3f} margin={margin:.3f}"
    )
    event = {
        "kind": "answer_only_dx_rescue",
        "base_dx": base_dx,
        "candidate_dx": cand_dx,
        "prob": float(prob),
        "margin": float(margin),
        "threshold": float(threshold),
    }
    return out, event


def predict_v20_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    case_id = case_id or Path(wsi_path).stem
    if _truthy(os.environ.get(DISABLE_ENV)):
        return predict_v17_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    try:
        # Keep v17 behavior bit-for-bit up to the final optional dx-only rescue.
        from src.reg2_hoptimus_online import extract_from_tiles
        from src.reg2_report_hard_heads import predict_hard_heads
        from src.reg2_report_selector_fused import select_final_report_fused
        from src.reg2_report_slot_candidate_selector import select_final_report_secondary_mil_candidate
        from src.reg2_report_slot_candidate_selector import select_final_report_slot_candidate
        from src.reg2_report_slot_heads import apply_slot_heads_to_steps
        from src.reg2_secondary_mil_heads import predict_secondary_findings
        from src.reg2_titan_online import extract_titan_with_tiles
        from src.reg2_v17_pipeline import (
            DX_TOPK_ENV,
            _apply_profile_runtime_env,
            _apply_labelgraph_patch,
            _apply_procedure_tool_patch,
            _dual_predictor,
            _dx_voter_needs_fused,
            _dx_voter_needs_v2,
            _i,
            _text_calibrator,
        )

        _apply_profile_runtime_env()
        titan_feats, tiles, coords, patch_px = extract_titan_with_tiles(wsi_path)
        hopt = extract_from_tiles(tiles, coords, patch_px)
        features = {
            "slide_embedding": titan_feats["slide_embedding"],
            "patch_features_conch": titan_feats["patch_features"],
            "patch_features": hopt["patch_features"],
            "pooled_mean": hopt["pooled_mean"],
            "coords": hopt["coords"],
            "n_patches": hopt["n_patches"],
            "patch_px": hopt["patch_px"],
            "scanned": titan_feats.get("scanned", hopt.get("scanned", np.asarray([0], np.int32))),
        }
        v2_feats = None
        if _dx_voter_needs_v2():
            from src.reg2_virchow2_online import extract_v2_features_from_tiles

            v2_feats = extract_v2_features_from_tiles(tiles)
            features["patch_features_v2"] = v2_feats["patch_features_v2"]
            if _dx_voter_needs_fused():
                features["fused"] = np.concatenate(
                    [
                        np.asarray(titan_feats["slide_embedding"], np.float32).ravel(),
                        np.asarray(hopt["pooled_mean"], np.float32).ravel(),
                        np.asarray(v2_feats["pooled_mean"], np.float32).ravel(),
                    ]
                ).astype(np.float32)

        dx_topk = _i(DX_TOPK_ENV, 15)
        context, scores = _dual_predictor().predict_with_scores(features, dx_topk=dx_topk)
        _dump_online_state(
            case_id,
            "context_pre_commit",
            {
                "case_id": case_id,
                "context": _context_record(context),
                "scores": _scores_record(scores),
                "n_patches": int(hopt.get("n_patches", 0) or 0),
                "patch_px": int(hopt.get("patch_px", 0) or 0),
                "scanned": int(titan_feats.get("scanned", [0])[0]) if "scanned" in titan_feats else 0,
                "feature_shapes": {
                    str(k): list(np.asarray(v).shape)
                    for k, v in features.items()
                    if isinstance(v, np.ndarray)
                },
            },
        )
        _dump_online_state(
            case_id,
            "context_committed",
            {
                "case_id": case_id,
                "context": _context_record(context),
                "scores": _scores_record(scores),
                "n_patches": int(hopt.get("n_patches", 0) or 0),
                "patch_px": int(hopt.get("patch_px", 0) or 0),
                "scanned": int(titan_feats.get("scanned", [0])[0]) if "scanned" in titan_feats else 0,
            },
        )
        steps = _sanitize_steps(_text_calibrator().render_from_labels(context, True))
        if not steps:
            raise RuntimeError("calibrator produced an empty chain")
        visual_floor_steps = [dict(step) for step in steps]
        report_snapshots: dict[tuple[str, str, str], str] = {}
        _remember_consistent_report_snapshot(report_snapshots, steps)

        from src.reg2_virchow2_online import extract_v2_features_from_tiles

        if v2_feats is None:
            v2_feats = extract_v2_features_from_tiles(tiles)
        v2_pooled = v2_feats["pooled_mean"]
        fused = np.concatenate(
            [
                np.asarray(titan_feats["slide_embedding"], np.float32).ravel(),
                np.asarray(hopt["pooled_mean"], np.float32).ravel(),
                np.asarray(v2_pooled, np.float32).ravel(),
            ]
        ).astype(np.float32)
        hard = predict_hard_heads(fused) or {}
        secondary_scores = predict_secondary_findings(
            v2_feats.get("patch_features_v2"),
            asset_name=os.environ.get(
                "REG2_SECONDARY_MIL_ASSET",
                "reg2_secondary_mil_heads_virchow2_pos20_s2049",
            ),
        )

        before_steps = steps
        steps = _sanitize_steps(
            select_final_report_slot_candidate(
                steps=steps,
                context=context,
                dx_scores=scores,
                fused_feature=fused,
                calib=_text_calibrator(),
            )
        )
        steps = _guard_report_dx_consistency(
            before_steps,
            steps,
            context=context,
            source="slot_candidate",
        )
        _dump_report_transition(
            case_id,
            "01_slot_candidate",
            before_steps,
            steps,
            context=context,
        )
        before_steps = steps
        steps = _sanitize_steps(
            apply_slot_heads_to_steps(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
            )
        )
        _dump_report_transition(
            case_id,
            "02_slot_heads",
            before_steps,
            steps,
            context=context,
        )
        before_steps = steps
        steps = _sanitize_steps(
            _apply_labelgraph_patch(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
                hard_heads=hard,
            )
        )
        _dump_report_transition(
            case_id,
            "03_labelgraph_pre",
            before_steps,
            steps,
            context=context,
        )
        _remember_consistent_report_snapshot(report_snapshots, steps)

        before_steps = steps
        steps = _sanitize_steps(
            select_final_report_fused(
                steps=steps,
                fused_feature=fused,
            )
        )
        steps = _guard_report_dx_consistency(
            before_steps,
            steps,
            context=context,
            source="fused_selector",
        )
        _dump_report_transition(
            case_id,
            "04_fused_selector",
            before_steps,
            steps,
            context=context,
        )
        _remember_consistent_report_snapshot(report_snapshots, steps)

        before_steps = steps
        steps = _sanitize_steps(
            select_final_report_secondary_mil_candidate(
                steps=steps,
                context=context,
                dx_scores=scores,
                fused_feature=fused,
                calib=_text_calibrator(),
                secondary_scores=secondary_scores,
            )
        )
        steps = _guard_report_dx_consistency(
            before_steps,
            steps,
            context=context,
            source="secondary_mil_candidate",
        )
        _dump_report_transition(
            case_id,
            "05_secondary_mil",
            before_steps,
            steps,
            context=context,
            extra={"secondary_scores": secondary_scores},
        )
        before_steps = steps
        steps = _sanitize_steps(
            _apply_labelgraph_patch(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
                hard_heads=hard,
            )
        )
        _dump_report_transition(
            case_id,
            "06_labelgraph_post",
            before_steps,
            steps,
            context=context,
        )
        before_steps = steps
        steps = _sanitize_steps(
            _apply_procedure_tool_patch(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
                hard_heads=hard,
            )
        )
        steps, procedure_tool_guard_event = _guard_procedure_tool_header_change(
            before_steps,
            steps,
            case_id=case_id,
        )
        _dump_report_transition(
            case_id,
            "07_procedure_tool",
            before_steps,
            steps,
            context=context,
            extra=procedure_tool_guard_event,
        )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        before_steps = steps
        steps, answer_only_dx_rescue_event = _apply_mil_dx_rescue_to_steps(
            steps,
            context,
            patch_features=np.asarray(hopt["patch_features"], np.float32),
        )
        if answer_only_dx_rescue_event is not None:
            _dump_online_state(
                case_id,
                "answer_only_dx_rescue",
                {
                    "case_id": case_id,
                    "context": _context_record(context),
                    "event": answer_only_dx_rescue_event,
                    "before_dx_answer": _answer_for(before_steps, DX_Q_CANONICAL),
                    "after_dx_answer": _answer_for(steps, DX_Q_CANONICAL),
                    "report_unchanged": _answer_for(before_steps, FINAL_REPORT_Q_CANONICAL)
                    == _answer_for(steps, FINAL_REPORT_Q_CANONICAL),
                },
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        before_steps = steps
        steps, report_state_event = _apply_report_state_rerender(
            steps=steps,
            context=context,
            calib=_text_calibrator(),
            case_id=case_id,
        )
        if report_state_event is not None:
            _dump_report_transition(
                case_id,
                "10_report_state_rerender",
                before_steps,
                steps,
                context=context,
                extra=report_state_event,
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        before_steps = steps
        steps, post_answer_dx_report_event = _apply_post_answer_dx_report_verifier(
            steps=steps,
            answer_only_dx_rescue_event=answer_only_dx_rescue_event,
            case_id=case_id,
        )
        if post_answer_dx_report_event is not None:
            _dump_report_transition(
                case_id,
                "11_post_answer_dx_report_verifier",
                before_steps,
                steps,
                context=context,
                extra=post_answer_dx_report_event,
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        before_steps = steps
        steps, breast_score_event = _apply_report_breast_score_verifier(
            steps=steps,
            case_id=case_id,
        )
        if breast_score_event is not None:
            _dump_report_transition(
                case_id,
                "12_report_breast_score_verifier",
                before_steps,
                steps,
                context=context,
                extra=breast_score_event,
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        steps, organ_verifier_event = _apply_report_header_organ_verifier(
            steps=steps,
            case_id=case_id,
        )
        if organ_verifier_event is not None:
            _dump_online_state(
                case_id,
                "organ_report_header_verifier",
                {
                    "case_id": case_id,
                    "context": _context_record(context),
                    "event": organ_verifier_event,
                    "final_organ_answer": _answer_for(steps, ORGAN_Q_CANONICAL),
                    "final_report_header": _report_header_organ(
                        _answer_for(steps, FINAL_REPORT_Q_CANONICAL)
                    ),
                },
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        steps, report_package_event = _apply_report_package_verifier(
            steps=steps,
            case_id=case_id,
        )
        if report_package_event is not None:
            _dump_online_state(
                case_id,
                "report_package_verifier",
                {
                    "case_id": case_id,
                    "context": _context_record(context),
                    "event": report_package_event,
                    "final_answers": {
                        "organ": _answer_for(steps, ORGAN_Q_CANONICAL),
                        "procedure": _answer_for(steps, PROCEDURE_Q_CANONICAL),
                        "primary_dx": _answer_for(steps, DX_Q_CANONICAL),
                    },
                    "final_report_header": _report_header_organ(
                        _answer_for(steps, FINAL_REPORT_Q_CANONICAL)
                    ),
                },
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        before_steps = steps
        steps, open_set_guard_event = _apply_open_set_organ_guard(
            steps=steps,
            context=context,
            patch_features=np.asarray(hopt["patch_features"], np.float32),
            wsi_path=wsi_path,
            case_id=case_id,
        )
        if open_set_guard_event is not None:
            _dump_report_transition(
                case_id,
                "13_open_set_organ_guard",
                before_steps,
                steps,
                context=context,
                extra=open_set_guard_event,
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        before_steps = steps
        steps, visual_floor_veto_event = _apply_visual_floor_veto(
            floor_steps=visual_floor_steps,
            steps=steps,
            patch_features=np.asarray(hopt["patch_features"], np.float32),
            case_id=case_id,
        )
        if visual_floor_veto_event is not None:
            _dump_report_transition(
                case_id,
                "13b_visual_floor_veto",
                before_steps,
                steps,
                context=context,
                extra=visual_floor_veto_event,
            )
        _remember_consistent_report_snapshot(report_snapshots, steps)
        before_steps = steps
        steps, report_snapshot_event = _restore_consistent_report_snapshot(
            snapshots=report_snapshots,
            steps=steps,
            case_id=case_id,
        )
        if report_snapshot_event is not None:
            _dump_report_transition(
                case_id,
                "13c_report_snapshot_rollback",
                before_steps,
                steps,
                context=context,
                extra=report_snapshot_event,
            )
        steps, report_format_event = _apply_report_format_canonicalizer(
            steps=steps,
            case_id=case_id,
        )
        if report_format_event is not None:
            _dump_online_state(
                case_id,
                "report_format_canonicalizer",
                {
                    "case_id": case_id,
                    "event": report_format_event,
                },
            )
        if not steps:
            raise RuntimeError("v20 produced an empty chain")

        top_dx = scores.get("primary_dx", [("", 0.0)])[0]
        final_dx = _answer_for(steps, DX_Q_CANONICAL) or context.primary_dx()
        print(
            f"[v20] v17-floor+answer-only-mil-rescue "
            f"organ={context.organ or '?'} dx={final_dx or '?'} "
            f"dx_topk={len(scores.get('primary_dx') or [])} "
            f"dx_score={top_dx[1]:.3f} fused_dim={fused.shape[0]} steps={len(steps)}"
        )
        return steps
    except Exception as exc:
        if _truthy(os.environ.get(ALLOW_CASCADE_FALLBACK_ENV)):
            print(f"[v20] failed; falling back to v17: {type(exc).__name__}: {exc}")
            _dump_emergency_fallback(case_id, exc, fallback="v17", cascade_allowed=True)
            return predict_v17_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
        print(f"[v20] failed; using single emergency fallback: {type(exc).__name__}: {exc}")
        _dump_emergency_fallback(case_id, exc, fallback="v02", cascade_allowed=False)
        from src.reg2_v02_pipeline import predict_v02_chain_of_thought

        return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
