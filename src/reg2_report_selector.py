"""Lightweight final-report selector for REG2026 Interface 1.

The main v14 pipeline renders a safe baseline report from predicted labels.
This module optionally replaces only that final report with the top whole-report
classifier template when the classifier is extremely confident. Workflow path
and non-final answers are untouched.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from typing import Any

import numpy as np

from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_text_calibrator import FINAL_REPORT_QUESTION_CANONICAL, canonicalize_question
from src.reg2_v11_pipeline import _find_asset_dir, _truthy

DISABLE_ENV = "REG2_DISABLE_REPORT_SELECTOR"
MIN_PROB_ENV = "REG2_REPORT_SELECTOR_MIN_PROB"
DEFAULT_MIN_PROB = 0.998
OTHER_REPORT = "<other_report>"


def _min_prob() -> float:
    try:
        return float(os.environ.get(MIN_PROB_ENV, DEFAULT_MIN_PROB))
    except ValueError:
        return DEFAULT_MIN_PROB


@lru_cache(maxsize=1)
def _report_clf() -> dict[str, Any] | None:
    try:
        import torch
        import torch.nn as nn

        model_dir = _find_asset_dir("reg2_report_clf", "report_clf.pt")
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
        return {"torch": torch, "model": model, "classes": classes, "mean": mean, "std": std}
    except Exception as exc:
        print(f"[report-selector] unavailable: {type(exc).__name__}: {exc}")
        return None


def _final_report_index(steps: list[ChainOfThoughtStep]) -> int | None:
    for i, step in enumerate(steps):
        if canonicalize_question(step.get("question", "")) == FINAL_REPORT_QUESTION_CANONICAL:
            return i
    return None


def select_final_report(
    *,
    steps: list[ChainOfThoughtStep],
    patch_features: np.ndarray,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline = str(steps[idx].get("answer", "") or "").strip()
    if not baseline:
        return steps

    clf = _report_clf()
    if clf is None:
        return steps
    patches = np.asarray(patch_features, np.float32)
    if patches.ndim != 2 or patches.shape[0] == 0:
        return steps
    pooled = patches.mean(axis=0)
    mean = clf["mean"]
    std = clf["std"]
    if pooled.shape != mean.shape:
        print(f"[report-selector] shape mismatch pooled={pooled.shape} expected={mean.shape}")
        return steps

    torch = clf["torch"]
    with torch.no_grad():
        x = ((pooled - mean) / std).astype(np.float32)
        logits = clf["model"](torch.from_numpy(x[None]))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    best_i = int(probs.argmax())
    prob = float(probs[best_i])
    report = str(clf["classes"][best_i])
    if report == OTHER_REPORT or not report or report == baseline or prob < _min_prob():
        return steps

    out = copy.deepcopy(steps)
    out[idx]["answer"] = report
    print(f"[report-selector] replaced final report prob={prob:.4f}")
    return out
