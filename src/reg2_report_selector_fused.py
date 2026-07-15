"""Fused-feature whole-report selector (online, GT-free).

Replaces ONLY the final-report answer when a strong whole-report classifier
(trained on fused TITAN+H1+V2 slide features) is confidently better than the
template baseline. Workflow path and all non-final answers are untouched, so
BPV / Edge-F1 / MESS are unchanged.

Decision rule:

    take the top-1 distinct classifier report R != baseline; replace iff
        prob(R) >= MIN_PROB (0.97)

The selector requires the fused TITAN+H1+V2 feature.
"""

from __future__ import annotations

import copy
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np

from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_text_calibrator import FINAL_REPORT_QUESTION_CANONICAL, canonicalize_question
from src.reg2_v11_pipeline import _find_asset_dir, _truthy

DISABLE_ENV = "REG2_DISABLE_FUSED_SELECTOR"
ASSET_ENV = "REG2_FUSED_SELECTOR_ASSET"
MIN_PROB_ENV = "REG2_FUSED_SELECTOR_MIN_PROB"
MIN_RATIO_ENV = "REG2_FUSED_SELECTOR_MIN_RATIO"
BLOCK_UNSUPPORTED_ENV = "REG2_FUSED_SELECTOR_BLOCK_UNSUPPORTED"
DEFAULT_ASSET = "reg2_report_clf_fused_s101"
FALLBACK_ASSET = "reg2_report_clf_fused"
DEFAULT_MIN_PROB = 0.97
DEFAULT_MIN_RATIO = 0.0
OTHER_REPORT = "<other_report>"


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _asset_dir() -> Any:
    asset_name = os.environ.get(ASSET_ENV, DEFAULT_ASSET).strip() or DEFAULT_ASSET
    try:
        return _find_asset_dir(asset_name, "report_clf.pt")
    except FileNotFoundError:
        if asset_name != DEFAULT_ASSET:
            raise
        return _find_asset_dir(FALLBACK_ASSET, "report_clf.pt")


@lru_cache(maxsize=1)
def _fused_clf() -> dict[str, Any] | None:
    try:
        import torch
        import torch.nn as nn

        model_dir = _asset_dir()
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
        return {
            "torch": torch,
            "model": model,
            "classes": classes,
            "class_to_idx": {r: i for i, r in enumerate(classes)},
            "mean": mean,
            "std": std,
        }
    except Exception as exc:
        print(f"[fused-selector] unavailable: {type(exc).__name__}: {exc}")
        return None


def _final_report_index(steps: list[ChainOfThoughtStep]) -> int | None:
    for i, step in enumerate(steps):
        if canonicalize_question(step.get("question", "")) == FINAL_REPORT_QUESTION_CANONICAL:
            return i
    return None


def _norm_report(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _report_header(text: str) -> str:
    return _norm_report(str(text or "").split("\n", 1)[0]).rstrip(";")


def _same_after_muscle_note_collapse(a: str, b: str) -> bool:
    def collapse(text: str) -> str:
        return _norm_report(
            re.sub(
                r"\bdoes not include muscle proper\b",
                "includes muscle proper",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

    return collapse(a) == collapse(b)


def _same_after_volume_collapse(a: str, b: str) -> bool:
    def collapse(text: str) -> str:
        return _norm_report(
            re.sub(
                r"tumou?r volume:\s*\d+%",
                "tumor volume: X%",
                str(text or ""),
                flags=re.IGNORECASE,
            )
        )

    return collapse(a) == collapse(b)


def _unsupported_isolated_change(baseline: str, candidate: str) -> str:
    """Block report-clf flips in slots this stack does not support visually."""
    if _truthy(os.environ.get(BLOCK_UNSUPPORTED_ENV, "1")) is False:
        return ""
    if _norm_report(baseline) == _norm_report(candidate):
        return ""
    base_header = _report_header(baseline)
    cand_header = _report_header(candidate)
    if (
        base_header.startswith("urinary bladder")
        and cand_header.startswith("urinary bladder")
        and _same_after_muscle_note_collapse(baseline, candidate)
    ):
        return "bladder_muscle_note_only"
    if (
        base_header.startswith("prostate")
        and cand_header.startswith("prostate")
        and _same_after_volume_collapse(baseline, candidate)
    ):
        return "prostate_volume_only"
    return ""


def select_final_report_fused(
    *,
    steps: list[ChainOfThoughtStep],
    fused_feature: np.ndarray,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline = str(steps[idx].get("answer", "") or "").strip()
    if not baseline:
        return steps

    clf = _fused_clf()
    if clf is None:
        return steps
    feat = np.asarray(fused_feature, np.float32).ravel()
    mean = clf["mean"]
    std = clf["std"]
    if feat.shape != mean.shape:
        print(f"[fused-selector] shape mismatch feat={feat.shape} expected={mean.shape}")
        return steps

    torch = clf["torch"]
    with torch.no_grad():
        x = ((feat - mean) / std).astype(np.float32)
        logits = clf["model"](torch.from_numpy(x[None]))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    classes = clf["classes"]
    base_idx = clf["class_to_idx"].get(baseline)
    base_prob = float(probs[base_idx]) if base_idx is not None else 0.0

    # top-1 distinct classifier report that differs from the baseline
    order = np.argsort(-probs)
    top_report = ""
    top_prob = 0.0
    for j in order:
        report = str(classes[int(j)])
        if report == OTHER_REPORT or not report or report == baseline:
            continue
        top_report = report
        top_prob = float(probs[int(j)])
        break
    if not top_report:
        return steps

    min_prob = _f(MIN_PROB_ENV, DEFAULT_MIN_PROB)
    min_ratio = _f(MIN_RATIO_ENV, DEFAULT_MIN_RATIO)
    if top_prob < min_prob or top_prob < min_ratio * max(base_prob, 1e-6):
        return steps
    blocked = _unsupported_isolated_change(baseline, top_report)
    if blocked:
        print(f"[fused-selector] blocked unsupported isolated change reason={blocked} prob={top_prob:.4f}")
        return steps

    out = copy.deepcopy(steps)
    out[idx]["answer"] = top_report
    print(f"[fused-selector] replaced final report prob={top_prob:.4f} base_prob={base_prob:.4f}")
    return out
