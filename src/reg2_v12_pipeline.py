"""v12 pipeline: TITAN online -> MLP+ABMIL(+kNN+organ-mask) ensemble -> calibrator.

The online CONCH pass yields the patch features consumed by the gated-attention
ABMIL head, so the ensemble adds no extra vision encoder.

Degrades gracefully: if anything in the ensemble path fails (missing ABMIL
asset, bad features, etc.) it falls back to v11 (MLP-only), then v10.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.reg2_ensemble_predictor import EnsemblePredictor
from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_text_calibrator import TextCalibrator
from src.reg2_titan_online import extract_titan_slide_embedding
from src.reg2_v11_pipeline import (
    _candidate_model_roots,
    _find_asset_dir,
    _sanitize_steps,
    _text_calibrator_asset_dir,
    _truthy,
    predict_v11_chain_of_thought,
)

DISABLE_ENV = "REG2_DISABLE_ENSEMBLE"


@lru_cache(maxsize=1)
def _ensemble_predictor() -> EnsemblePredictor:
    mlp_dir = _find_asset_dir("reg2_titan_head", "multitask_heads.pt")
    abmil_dir = _find_asset_dir("reg2_abmil", "abmil_heads.pt")
    predictor = EnsemblePredictor(mlp_dir, abmil_dir)
    if not predictor.available:
        raise FileNotFoundError(
            f"ensemble predictor unavailable (mlp={mlp_dir}, abmil={abmil_dir})"
        )
    return predictor


@lru_cache(maxsize=1)
def _text_calibrator() -> TextCalibrator:
    assets_dir = _text_calibrator_asset_dir()
    return TextCalibrator.from_assets(assets_dir)


def predict_v12_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return predict_v11_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    try:
        features = extract_titan_slide_embedding(wsi_path)
        context, scores = _ensemble_predictor().predict_with_scores(features)
        steps = _sanitize_steps(_text_calibrator().render_from_labels(context, True))
        if not steps:
            raise RuntimeError("calibrator produced an empty chain")
        top_dx = scores.get("primary_dx", [("", 0.0)])[0]
        print(
            "[v12] labels "
            f"organ={context.organ or '?'} dx={context.primary_dx() or '?'} "
            f"dx_score={top_dx[1]:.3f} steps={len(steps)}"
        )
        return steps
    except Exception as exc:
        print(
            f"[v12] ensemble failed; falling back to v11: {type(exc).__name__}: {exc}"
        )
        return predict_v11_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
