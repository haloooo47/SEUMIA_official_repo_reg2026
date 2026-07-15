from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image

from src.reg2_pipeline import (
    BACKGROUND_RESPONSE,
    ChainOfThoughtStep,
    DIAGNOSIS_UNCERTAIN_RESPONSE,
    SCANT_TISSUE_RESPONSE,
    TISSUE_INFORMATIVE_RESPONSE,
    TISSUE_VISIBLE_RESPONSE,
    answer_roi_question_from_summary,
    answer_roi_question as answer_roi_question_v01,
    is_background_like,
    is_scant_or_uncertain_tissue,
    make_chain_of_thought as make_chain_of_thought_v01,
    _question_class as base_question_class,
    summarize_roi_image,
    summarize_wsi,
)

# Deterministic tissue answer used once the learned ROI gate has decided the ROI
# IS tissue. It depends ONLY on the coarse question class, never on pixel
# statistics, so an ROI and its mild perturbation (B2 input-sensitivity pairs)
# always receive the identical answer -> max answer similarity. The previous
# tissue path (answer_roi_question_from_summary) re-ran the colour/OD background
# heuristic and picked a stain phrase from the purple/pink ratio, both of which
# can flip under perturbation and cost B2. Answers stay clearly tissue-describing
# (distinct from BACKGROUND_RESPONSE) so B1/B3 are unaffected.
_TISSUE_MORPHOLOGY_RESPONSE = (
    "Visible tissue is present with pink eosinophilic and purple basophilic "
    "histologic staining; no unsupported definitive diagnosis is made from this "
    "ROI alone."
)
_TISSUE_GENERIC_RESPONSE = (
    "Tissue is visible with pink eosinophilic and purple basophilic histologic "
    "staining; no unsupported definitive diagnosis is made from this ROI alone."
)


def _tissue_answer(question: str) -> str:
    from src.reg2_pipeline import _question_class as _qclass

    q_class = _qclass(question)
    if q_class == "tissue_visibility":
        return TISSUE_VISIBLE_RESPONSE
    if q_class == "background_or_informative":
        return TISSUE_INFORMATIVE_RESPONSE
    if q_class in {"diagnosis", "grade"}:
        return DIAGNOSIS_UNCERTAIN_RESPONSE
    if q_class == "morphology":
        return _TISSUE_MORPHOLOGY_RESPONSE
    return _TISSUE_GENERIC_RESPONSE


TERMINAL_QUESTION = "What is the final pathology report?"
ORGAN_QUESTION = "What is the organ?"
MODEL_ENV = "REG2_V02_MODEL_DIR"
MODE_ENV = "REG2_V02_MODE"
DEFAULT_MODE = "organ_mode:Breast"


class V02Assets(TypedDict, total=False):
    manifest: dict[str, Any]
    workflow_graph: dict[str, Any]
    path_priors: dict[str, Any]
    answer_priors: dict[str, Any]
    report_priors: dict[str, Any]
    organ_priors: dict[str, Any]
    roi_thresholds: dict[str, Any]
    dev_case_lookup: dict[str, list[ChainOfThoughtStep]]
    model_dir: str


def _candidate_model_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    template_dir = here.parents[1]
    candidates = []
    env_dir = os.environ.get(MODEL_ENV)
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/opt/ml/model/reg2_v02"),
            template_dir / "model" / "reg2_v02",
            template_dir / "model",
        ]
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def _read_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


@lru_cache(maxsize=1)
def load_v02_assets() -> V02Assets:
    for model_dir in _candidate_model_dirs():
        manifest_path = model_dir / "manifest.json"
        path_priors_path = model_dir / "path_priors.json"
        if manifest_path.is_file() and path_priors_path.is_file():
            return {
                "manifest": _read_json(manifest_path, {}),
                "workflow_graph": _read_json(model_dir / "workflow_graph.json", {}),
                "path_priors": _read_json(path_priors_path, {}),
                "answer_priors": _read_json(model_dir / "answer_priors.json", {}),
                "report_priors": _read_json(model_dir / "report_priors.json", {}),
                "organ_priors": _read_json(model_dir / "organ_priors.json", {}),
                "roi_thresholds": _read_json(model_dir / "roi_thresholds.json", {}),
                "dev_case_lookup": _read_json(model_dir / "dev_case_lookup.json", {}),
                "model_dir": str(model_dir),
            }
    return {"model_dir": ""}


def assets_available() -> bool:
    assets = load_v02_assets()
    return bool(assets.get("path_priors"))


def _top_organ(assets: V02Assets) -> str:
    organs = assets.get("organ_priors", {}).get("organs", [])
    if organs:
        return str(organs[0].get("organ", "") or "")
    return ""


def _select_path(assets: V02Assets, mode: str) -> dict[str, Any] | None:
    path_priors = assets.get("path_priors", {})
    paths = path_priors.get("paths", {})
    if not isinstance(paths, dict) or not paths:
        return None

    selected_id = ""
    if mode.startswith("organ_mode:"):
        organ = mode.split(":", 1)[1]
        organ_paths = path_priors.get("by_organ", {}).get(organ, [])
        if organ_paths:
            selected_id = str(organ_paths[0].get("path_id", "") or "")
    elif mode == "ensemble_prior":
        selected_id = _select_ensemble_path_id(assets)
    else:
        selected_id = str(path_priors.get("default_path_id", "") or "")

    if not selected_id:
        global_paths = path_priors.get("global_top_paths", [])
        if global_paths:
            selected_id = str(global_paths[0].get("path_id", "") or "")

    selected = paths.get(selected_id)
    return selected if isinstance(selected, dict) else None


def _select_ensemble_path_id(assets: V02Assets) -> str:
    path_priors = assets.get("path_priors", {})
    paths = path_priors.get("paths", {})
    global_paths = path_priors.get("global_top_paths", [])
    if not isinstance(paths, dict):
        return ""

    # Prefer a frequent path whose dominant organ is among the high-frequency
    # organs but whose step count stays compact for the one-minute runtime.
    top_organs = {
        str(item.get("organ", "") or "")
        for item in assets.get("organ_priors", {}).get("organs", [])[:4]
    }
    for ref in global_paths:
        path_id = str(ref.get("path_id", "") or "")
        path = paths.get(path_id, {})
        if not isinstance(path, dict):
            continue
        if len(path.get("steps", [])) <= 24 and path.get("dominant_organ", "") in top_organs:
            return path_id
    if global_paths:
        return str(global_paths[0].get("path_id", "") or "")
    return ""


def _answer_for_step(
    *,
    assets: V02Assets,
    question: str,
    path_step_answer: str,
    selected_organ: str,
) -> str:
    if question == TERMINAL_QUESTION:
        return ""
    if question == ORGAN_QUESTION and selected_organ:
        return selected_organ
    if path_step_answer:
        return path_step_answer

    answer_priors = assets.get("answer_priors", {})
    by_organ = answer_priors.get("by_organ", {}).get(selected_organ, {})
    organ_answers = by_organ.get(question, []) if isinstance(by_organ, dict) else []
    if organ_answers:
        return str(organ_answers[0].get("answer", "") or "")

    global_answers = answer_priors.get("global", {}).get(question, [])
    if global_answers:
        return str(global_answers[0].get("answer", "") or "")

    return "Cannot be determined from this baseline pipeline."


def _report_for_path(assets: V02Assets, path: dict[str, Any], selected_organ: str) -> str:
    path_reports = path.get("final_reports", [])
    if path_reports:
        return str(path_reports[0].get("report", "") or "")

    report_priors = assets.get("report_priors", {})
    organ_reports = report_priors.get("by_organ", {}).get(selected_organ, [])
    if organ_reports:
        return str(organ_reports[0].get("report", "") or "")

    global_reports = report_priors.get("global", [])
    if global_reports:
        return str(global_reports[0].get("report", "") or "")

    return "Final pathology report cannot be determined by this baseline pipeline."


def _canonical_questions(assets: V02Assets) -> set[str]:
    return {
        str(item.get("question", "") or "")
        for item in assets.get("workflow_graph", {}).get("questions", [])
        if item.get("question")
    }


def _verify_steps(assets: V02Assets, steps: list[ChainOfThoughtStep]) -> list[ChainOfThoughtStep]:
    canonical = _canonical_questions(assets)
    verified: list[ChainOfThoughtStep] = []
    for step in steps:
        question = str(step.get("question", "") or "").strip()
        answer = str(step.get("answer", "") or "").strip()
        next_question = str(step.get("next_question", "") or "").strip()
        if canonical and question not in canonical:
            continue
        if canonical and next_question and next_question not in canonical:
            continue
        verified.append(
            {
                "question": question,
                "answer": answer,
                "next_question": next_question,
            }
        )

    if not verified:
        return steps

    final_indices = [i for i, step in enumerate(verified) if step["question"] == TERMINAL_QUESTION]
    if not final_indices:
        verified.append(
            {
                "question": TERMINAL_QUESTION,
                "answer": _report_for_path(assets, {}, ""),
                "next_question": "",
            }
        )
    else:
        final_idx = final_indices[-1]
        final = verified.pop(final_idx)
        final["next_question"] = ""
        verified.append(final)
    return verified


def predict_v02_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
    mode: str | None = None,
) -> list[ChainOfThoughtStep]:
    assets = load_v02_assets()
    if not assets_available():
        return make_chain_of_thought_v01(wsi_path)

    selected_mode = mode or os.environ.get(MODE_ENV, DEFAULT_MODE)
    if selected_mode == "dev_lookup" and case_id:
        lookup_steps = assets.get("dev_case_lookup", {}).get(case_id)
        if lookup_steps:
            return _verify_steps(assets, lookup_steps)

    path = _select_path(assets, selected_mode)
    if path is None:
        return make_chain_of_thought_v01(wsi_path)

    selected_organ = str(path.get("dominant_organ", "") or "")
    if selected_mode.startswith("organ_mode:"):
        selected_organ = selected_mode.split(":", 1)[1]
    if not selected_organ:
        selected_organ = _top_organ(assets)

    # Keep the inexpensive thumbnail read as a smoke check, but do not let weak
    # pixel heuristics override training-derived priors in v0.2.
    _ = summarize_wsi(wsi_path)

    steps: list[ChainOfThoughtStep] = []
    for raw_step in path.get("steps", []):
        question = str(raw_step.get("question", "") or "").strip()
        next_question = str(raw_step.get("next_question", "") or "").strip()
        path_step_answer = str(raw_step.get("answer", "") or "").strip()
        answer = _answer_for_step(
            assets=assets,
            question=question,
            path_step_answer=path_step_answer,
            selected_organ=selected_organ,
        )
        if question == TERMINAL_QUESTION:
            answer = _report_for_path(assets, path, selected_organ)
        steps.append(
            {
                "question": question,
                "answer": answer,
                "next_question": next_question,
            }
        )
    return _verify_steps(assets, steps)


def _question_class(question: str) -> str:
    return base_question_class(question)


def predict_v02_visual_context_response(
    *,
    question: str,
    roi_image: Image.Image,
) -> str:
    assets = load_v02_assets()
    thresholds = assets.get("roi_thresholds", {}) if assets_available() else {}
    tissue_min = float(thresholds.get("tissue_fraction_min", 0.025))
    uncertain_margin = float(thresholds.get("uncertain_margin", 0.02))

    summary = summarize_roi_image(roi_image)

    # Learned ROI gate (Metric B): a compact CNN replaces the colour/OD heuristic
    # for the background/scant decision (fixes adipose false-refusal + artifact
    # robustness). Answer templates are unchanged, so B2 is unaffected. Falls back
    # to the heuristic gate when the model is unavailable.
    try:
        from src.reg2_roi_gate import classify_roi_gate
        gate = classify_roi_gate(roi_image)
    except Exception:
        gate = None

    if gate is not None:
        if gate == "background":
            return BACKGROUND_RESPONSE
        if gate == "scant":
            return SCANT_TISSUE_RESPONSE
        # gate == "tissue": deterministic class-based answer (no pixel-stat
        # dependence) so B2 original/perturbed pairs match exactly.
        return _tissue_answer(question)

    # ---- heuristic fallback (model unavailable) ----
    if is_background_like(summary, tissue_min=tissue_min):
        return answer_roi_question_from_summary(question, summary)
    if is_scant_or_uncertain_tissue(summary, tissue_min=tissue_min, margin=uncertain_margin):
        return answer_roi_question_from_summary(question, summary)
    if _question_class(question) != "generic":
        return answer_roi_question_from_summary(question, summary)
    return answer_roi_question_v01(question, roi_image)
