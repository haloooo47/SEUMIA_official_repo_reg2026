"""
Interface 0 — workflow-aligned visual grounding.

Metric B is treated as the ROI-level evidence view of the same pathology
workflow used by Metric A, not as an independent slide diagnosis pipeline.
The default path therefore shares the compact ROI tissue/background gate and
conservative evidence templates used by the earlier visual-evidence layer:

  - Background / scant-tissue ROIs: concise rejection answers.
  - Tissue ROIs: stable H&E evidence answers; no unsupported definitive
    diagnosis, organ, subtype, or grade is inferred from a single ROI.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from core import load_json_file, load_roi_image

# ── constants ───────────────────────────────────────────────────────────

# ── visual grounding entry ──────────────────────────────────────────────

def predict_visual_context_response(
    *,
    question_path: Path,
    roi_image_path: Path,
) -> str:
    """Run Visual Grounding inference for a single ROI.

    Uses the lightweight learned ROI gate and deterministic evidence templates.
    """
    import torch

    question: str = load_json_file(location=question_path)
    roi_image: Image.Image = load_roi_image(location=roi_image_path)

    # ── Fast gate: learned tissue/background CNN ─────────────────────
    gate = None
    if torch.cuda.is_available():
        try:
            from src.reg2_roi_gate import classify_roi_gate
            gate = classify_roi_gate(roi_image)
        except Exception:
            pass

    if gate == "background":
        return (
            "No tissue is visible in this ROI. The region appears to contain "
            "only background, artifact, or non-diagnostic material."
        )
    if gate == "scant":
        return (
            "Only scant tissue is visible in this ROI. The amount of tissue "
            "is insufficient for a reliable histologic assessment."
        )

    # ── deterministic evidence templates ─────────────────────────────
    from src.reg2_v10_pipeline import predict_v10_visual_context_response
    return predict_v10_visual_context_response(question=question, roi_image=roi_image)
