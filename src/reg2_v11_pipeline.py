from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.reg2_label_predictor import LabelPredictor
from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_text_calibrator import TextCalibrator
from src.reg2_titan_online import extract_titan_slide_embedding
from src.reg2_v10_pipeline import predict_v10_chain_of_thought


DISABLE_ENV = "REG2_DISABLE_TITAN_ONLINE"
MODEL_ROOT_ENV = "REG2_MODEL_ROOT"
TEXT_CALIBRATOR_DIR_ENV = "REG2_TEXT_CALIBRATOR_DIR"
TEXT_CALIBRATOR_V2_DIR_ENV = "REG2_TEXT_CALIBRATOR_V2_DIR"


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_model_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get(MODEL_ROOT_ENV)
    if env_root:
        roots.append(Path(env_root))
    roots.extend([Path("/opt/ml/model"), _template_dir() / "model"])
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen:
            seen.add(resolved)
            out.append(root)
    return out


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _find_asset_dir(name: str, required_file: str) -> Path:
    for root in _candidate_model_roots():
        candidate = root / name
        if (candidate / required_file).is_file():
            return candidate
    checked = ", ".join(str(root / name) for root in _candidate_model_roots())
    raise FileNotFoundError(f"missing {name}/{required_file}; checked: {checked}")


def _env_asset_dir(env_name: str, required_file: str) -> Path | None:
    raw = str(os.environ.get(env_name, "") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if (path / required_file).is_file():
        return path
    raise FileNotFoundError(f"{env_name}={path} missing {required_file}")


def _text_calibrator_asset_dir(*, v2: bool = False) -> Path:
    if v2:
        return _env_asset_dir(TEXT_CALIBRATOR_V2_DIR_ENV, "answer_calibrator.json") or _find_asset_dir(
            "reg2_calibrator_v2_labelgraph",
            "answer_calibrator.json",
        )
    return _env_asset_dir(TEXT_CALIBRATOR_DIR_ENV, "answer_calibrator.json") or _find_asset_dir(
        "reg2_calibrator",
        "answer_calibrator.json",
    )


@lru_cache(maxsize=1)
def _label_predictor() -> LabelPredictor:
    model_dir = _find_asset_dir("reg2_titan_head", "multitask_heads.pt")
    predictor = LabelPredictor(model_dir)
    if not predictor.available:
        raise FileNotFoundError(f"label predictor unavailable at {model_dir}")
    return predictor


@lru_cache(maxsize=1)
def _text_calibrator() -> TextCalibrator:
    assets_dir = _text_calibrator_asset_dir()
    return TextCalibrator.from_assets(assets_dir)


def _sanitize_steps(raw_steps: list[dict[str, Any]]) -> list[ChainOfThoughtStep]:
    steps: list[ChainOfThoughtStep] = []
    for step in raw_steps:
        question = str(step.get("question", "") or "").strip()
        answer = str(step.get("answer", "") or "").strip()
        next_question = str(step.get("next_question", "") or "").strip()
        if not question:
            continue
        steps.append(
            {
                "question": question,
                "answer": answer,
                "next_question": next_question,
            }
        )
    if steps:
        steps[-1]["next_question"] = ""
    return steps


def predict_v11_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return predict_v10_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    try:
        features = extract_titan_slide_embedding(wsi_path)
        context, scores = _label_predictor().predict_with_scores(features, topk=3)
        steps = _sanitize_steps(_text_calibrator().render_from_labels(context, True))
        if not steps:
            raise RuntimeError("calibrator produced an empty chain")
        top_dx = scores.get("primary_dx", [("", 0.0)])[0]
        print(
            "[v11] labels "
            f"organ={context.organ or '?'} dx={context.primary_dx() or '?'} "
            f"primary_dx_top={top_dx[0]}:{top_dx[1]:.3f} steps={len(steps)}"
        )
        return steps
    except Exception as exc:
        print(f"[v11] TITAN-online failed; falling back to v10: {type(exc).__name__}: {exc}")
        return predict_v10_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
