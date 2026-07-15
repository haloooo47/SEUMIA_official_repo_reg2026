"""Dual-encoder ensemble with TITAN MLP heads and H-optimus-1 ABMIL diagnosis.

The organ, procedure, grade, behavior, and histologic-type heads use the TITAN
slide embedding. ``primary_dx`` also uses gated attention over H-optimus-1
(ViT-g/14, 1536-d) patch features.

Both encoders share ONE tissue tile scan (the disk-I/O heavy step): TITAN scans +
CONCH-encodes the tiles, then H-optimus-1 re-encodes the SAME tiles. With the
Degrades gracefully: any failure in the dual path falls back to v12 (TITAN-only
MLP+ABMIL on CONCH features), which in turn falls back to v11/v10.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from src.reg2_ensemble_predictor import EnsemblePredictor
from src.reg2_hoptimus_online import extract_from_tiles
from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_report_selector import select_final_report
from src.reg2_text_calibrator import TextCalibrator
from src.reg2_titan_online import extract_titan_with_tiles
from src.reg2_v11_pipeline import _find_asset_dir, _sanitize_steps, _text_calibrator_asset_dir, _truthy
from src.reg2_v12_pipeline import predict_v12_chain_of_thought

DISABLE_ENV = "REG2_DISABLE_DUAL_ENCODER"


@lru_cache(maxsize=1)
def _dual_predictor() -> EnsemblePredictor:
    # MLP / kNN / organ-mask heads on the TITAN slide embedding (768-d).
    mlp_dir = _find_asset_dir("reg2_titan_head", "multitask_heads.pt")
    # primary_dx ABMIL on H-optimus-1 patch features (1536-d).
    abmil_dir = _find_asset_dir("reg2_abmil_h1", "abmil_heads.pt")
    predictor = EnsemblePredictor(mlp_dir, abmil_dir)
    if not predictor.available:
        raise FileNotFoundError(
            f"dual-encoder predictor unavailable (mlp={mlp_dir}, abmil={abmil_dir})"
        )
    return predictor


@lru_cache(maxsize=1)
def _text_calibrator() -> TextCalibrator:
    assets_dir = _text_calibrator_asset_dir()
    return TextCalibrator.from_assets(assets_dir)


def predict_v14_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return predict_v12_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    try:
        # 1) TITAN online: slide embedding (for MLP) + the scanned tiles for reuse.
        titan_feats, tiles, coords, patch_px = extract_titan_with_tiles(wsi_path)
        # 2) H-optimus-1 re-encodes the SAME tiles -> 1536-d patch features (ABMIL).
        hopt = extract_from_tiles(tiles, coords, patch_px)
        # 3) Dual feature dict: TITAN slide_embedding -> MLP, H-opt1 patches -> ABMIL.
        features = {
            "slide_embedding": titan_feats["slide_embedding"],
            "patch_features": hopt["patch_features"],
            "coords": hopt["coords"],
            "n_patches": hopt["n_patches"],
            "patch_px": hopt["patch_px"],
        }
        context, scores = _dual_predictor().predict_with_scores(features)
        steps = _sanitize_steps(_text_calibrator().render_from_labels(context, True))
        steps = _sanitize_steps(select_final_report(steps=steps, patch_features=hopt["patch_features"]))
        if not steps:
            raise RuntimeError("calibrator produced an empty chain")
        top_dx = scores.get("primary_dx", [("", 0.0)])[0]
        print(
            "[v14] dual-encoder labels "
            f"organ={context.organ or '?'} dx={context.primary_dx() or '?'} "
            f"dx_score={top_dx[1]:.3f} steps={len(steps)}"
        )
        return steps
    except Exception as exc:
        print(
            f"[v14] dual-encoder failed; falling back to v12: {type(exc).__name__}: {exc}"
        )
        return predict_v12_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
