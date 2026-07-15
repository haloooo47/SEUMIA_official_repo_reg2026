"""Dual-encoder diagnosis with a Virchow2-fused whole-report selector.

Virchow2 encodes the shared tiles to form
``concat[TITAN(768), H1(1536), V2(2560)] = 4864`` for the whole-report
classifier. The selector changes only the final report when its confidence and
consistency conditions pass.

Degrades gracefully: any failure in the V2 / fused-selector step falls back to
the plain v14 prediction (which itself falls back to v12).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.reg2_ensemble_predictor import EnsemblePredictor
from src.reg2_hoptimus_online import extract_from_tiles
from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_report_selector_oof import select_final_report_oof
from src.reg2_report_slot_candidate_selector import select_final_report_slot_candidate
from src.reg2_report_slot_heads import apply_slot_heads_to_steps
from src.reg2_text_calibrator import TextCalibrator
from src.reg2_titan_online import extract_titan_with_tiles
from src.reg2_v11_pipeline import _find_asset_dir, _sanitize_steps, _text_calibrator_asset_dir, _truthy
from src.reg2_v14_pipeline import predict_v14_chain_of_thought

DISABLE_ENV = "REG2_DISABLE_V15"
OOF_SELECTOR_ENV = "REG2_REPORT_OOF_SELECTOR"


@lru_cache(maxsize=1)
def _dual_predictor() -> EnsemblePredictor:
    mlp_dir = _find_asset_dir("reg2_titan_head", "multitask_heads.pt")
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


def predict_v15_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return predict_v14_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    try:
        # 1) TITAN online: slide embedding (MLP) + scanned tiles for reuse.
        titan_feats, tiles, coords, patch_px = extract_titan_with_tiles(wsi_path)
        # 2) H-optimus-1 re-encodes the SAME tiles -> 1536-d patch features (ABMIL dx).
        hopt = extract_from_tiles(tiles, coords, patch_px)
        features = {
            "slide_embedding": titan_feats["slide_embedding"],
            "patch_features": hopt["patch_features"],
            "coords": hopt["coords"],
            "n_patches": hopt["n_patches"],
            "patch_px": hopt["patch_px"],
        }
        # 3) Proven v14 dual dx + calibrator render (non-final path unchanged).
        context, scores = _dual_predictor().predict_with_scores(features)
        steps = _sanitize_steps(_text_calibrator().render_from_labels(context, True))
        if not steps:
            raise RuntimeError("calibrator produced an empty chain")

        # 4) Virchow2 re-encodes the SAME tiles -> 2560-d pooled mean (fused clf only).
        from src.reg2_virchow2_online import extract_v2_pooled_from_tiles

        v2_pooled = extract_v2_pooled_from_tiles(tiles)
        fused = np.concatenate(
            [
                np.asarray(titan_feats["slide_embedding"], np.float32).ravel(),  # 768
                np.asarray(hopt["pooled_mean"], np.float32).ravel(),             # 1536
                np.asarray(v2_pooled, np.float32).ravel(),                       # 2560
            ]
        ).astype(np.float32)

        # 5) Final report: OOF learned selector (preferred) or fixed fused-clf rule.
        if _truthy(os.environ.get(OOF_SELECTOR_ENV)):
            steps = _sanitize_steps(
                select_final_report_oof(
                    steps=steps,
                    context=context,
                    dx_scores=scores,
                    fused_feature=fused,
                    calib=_text_calibrator(),
                )
            )
        else:
            steps = _sanitize_steps(
                select_final_report_slot_candidate(
                    steps=steps,
                    context=context,
                    dx_scores=scores,
                    fused_feature=fused,
                    calib=_text_calibrator(),
                )
            )
        steps = _sanitize_steps(
            apply_slot_heads_to_steps(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
            )
        )
        if not steps:
            raise RuntimeError("fused selector produced an empty chain")

        top_dx = scores.get("primary_dx", [("", 0.0)])[0]
        print(
            "[v15] dual-dx + v2-fused-selector "
            f"organ={context.organ or '?'} dx={context.primary_dx() or '?'} "
            f"dx_score={top_dx[1]:.3f} fused_dim={fused.shape[0]} steps={len(steps)}"
        )
        return steps
    except Exception as exc:
        print(f"[v15] failed; falling back to v14: {type(exc).__name__}: {exc}")
        return predict_v14_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
