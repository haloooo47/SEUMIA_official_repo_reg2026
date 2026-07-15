"""Online report slot head inference + final-report patching."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.reg2_report_slot_patch import patch_report_for_organ
from src.reg2_text_calibrator import FINAL_REPORT_QUESTION_CANONICAL, canonicalize_question
from src.reg2_v11_pipeline import _find_asset_dir, _truthy

ENABLE_ENV = "REG2_REPORT_SLOT_HEADS"
DISABLE_ENV = "REG2_DISABLE_REPORT_SLOT_HEADS"
MIN_GLEASON_ENV = "REG2_SLOT_MIN_PROB_GLEASON"
MIN_BREAST_ENV = "REG2_SLOT_MIN_PROB_BREAST"
MIN_MARGIN_GLEASON_ENV = "REG2_SLOT_MIN_MARGIN_GLEASON"
DISABLE_DNH_ENV = "REG2_SLOT_DISABLE_DNH"


def _f(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@lru_cache(maxsize=4)
def _bundle(asset_name: str = "reg2_report_slot_heads") -> dict[str, Any] | None:
    try:
        import json
        import torch

        model_dir = _find_asset_dir(asset_name, "slot_heads.pt")
        ckpt = torch.load(model_dir / "slot_heads.pt", map_location="cpu", weights_only=False)
        vocab = json.loads((model_dir / "slot_vocab.json").read_text(encoding="utf-8"))
        return {"ckpt": ckpt, "vocab": vocab, "model_dir": model_dir}
    except Exception as exc:
        print(f"[slot-heads] unavailable {asset_name}: {type(exc).__name__}: {exc}")
        return None


def _asset_for_organ(organ: str) -> str:
    if str(organ or "").strip().lower() == "breast":
        # The breast-only head calibrated better than the all-slot head offline.
        # Fall back silently to the all-slot bundle if this optional asset is
        # absent in an older package.
        return "reg2_report_slot_heads_breast"
    return "reg2_report_slot_heads"


def _load_torch_model(bundle: dict[str, Any]):
    import torch
    import torch.nn as nn

    ckpt = bundle["ckpt"]
    vocab = bundle["vocab"]
    slots = vocab["slots"]
    head_sizes = {s: len(vocab["vocabs"][s]) for s in slots}

    class SlotMLP(nn.Module):
        def __init__(self):
            super().__init__()
            h = int(ckpt["hidden"])
            nl = int(ckpt.get("num_layers", 1))
            dp = float(ckpt.get("dropout", 0.15))
            d = int(np.asarray(ckpt["scaler_mean"]).shape[0])
            layers: list[nn.Module] = []
            for i in range(max(1, nl)):
                layers += [nn.Linear(d if i == 0 else h, h), nn.GELU(), nn.Dropout(dp)]
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in head_sizes.items()})

        def forward(self, x):
            z = self.trunk(x)
            return {k: v(z) for k, v in self.heads.items()}

    model = SlotMLP()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(ckpt["scaler_mean"], np.float32)
    std = np.asarray(ckpt["scaler_std"], np.float32)
    return torch, model, mean, std, slots, vocab["vocabs"]


@lru_cache(maxsize=4)
def _loaded_model(asset_name: str):
    bundle = _bundle(asset_name)
    if bundle is None:
        return None
    return _load_torch_model(bundle)


def predict_slots(
    fused_feature: np.ndarray,
    *,
    organ: str = "",
    asset_name: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    asset_name = asset_name or _asset_for_organ(organ)
    loaded = _loaded_model(asset_name)
    if loaded is None and asset_name != "reg2_report_slot_heads":
        loaded = _loaded_model("reg2_report_slot_heads")
    if loaded is None:
        return None
    feat = np.asarray(fused_feature, np.float32).ravel()
    torch, model, mean, std, slots, vocabs = loaded
    if feat.shape != mean.shape:
        print(f"[slot-heads] feature dim mismatch {feat.shape} vs {mean.shape}")
        return None
    x = ((feat - mean) / std).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(x[None]))
    slots_out: dict[str, dict[str, Any]] = {}
    for slot in slots:
        probs = torch.softmax(out[slot], dim=1).cpu().numpy()[0]
        order = np.argsort(-probs)
        idx = int(order[0])
        top1 = float(probs[idx])
        top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
        slots_out[slot] = {
            "pred": vocabs[slot][idx],
            "prob": top1,
            "margin": top1 - top2,
        }
    return slots_out


def _final_report_index(steps: list[dict[str, Any]]) -> int | None:
    for i, step in enumerate(steps):
        if canonicalize_question(step.get("question", "")) == FINAL_REPORT_QUESTION_CANONICAL:
            return i
    return None


def apply_slot_heads_to_steps(
    steps: list[dict[str, Any]],
    *,
    organ: str,
    fused_feature: np.ndarray,
) -> list[dict[str, Any]]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return steps
    if not _truthy(os.environ.get(ENABLE_ENV)):
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline = str(steps[idx].get("answer", "") or "").strip()
    if not baseline:
        return steps
    slots = predict_slots(fused_feature, organ=organ)
    if not slots:
        return steps
    patched, applied = patch_report_for_organ(
        baseline,
        organ,
        slots,
        min_prob_gleason=_f(MIN_GLEASON_ENV, 0.55),
        min_prob_breast=_f(MIN_BREAST_ENV, 0.70),
        min_margin_gleason=_f(MIN_MARGIN_GLEASON_ENV, 0.15),
        organ_filter="all",
        breast_partial=True,
    )
    if patched == baseline or not applied:
        return steps
    out = [dict(s) for s in steps]
    out[idx] = dict(out[idx])
    out[idx]["answer"] = patched
    print(f"[slot-heads] patched organ={organ} slots={applied}")
    return out
