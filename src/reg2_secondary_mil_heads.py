"""Patch-level secondary-finding evidence heads for LabelGraph."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.reg2_v11_pipeline import _find_asset_dir

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - online fallback when torch is unavailable.
    torch = None
    nn = None


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_finding(value: Any) -> str:
    text = _clean(value).lower()
    text = re.sub(r"[\s,.;:()/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=2)
def _load(asset_name: str = "reg2_secondary_mil_heads") -> dict[str, Any] | None:
    if torch is None or nn is None:
        return None

    try:
        asset_dir = _find_asset_dir(asset_name, "secondary_mil_heads.pt")
    except Exception:
        return None
    ckpt_path = Path(asset_dir) / "secondary_mil_heads.pt"
    if not ckpt_path.is_file():
        return None

    class ABMILMultiLabel(nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
            self.att_v = nn.Linear(hidden, hidden)
            self.att_u = nn.Linear(hidden, hidden)
            self.att = nn.Linear(hidden, 1)
            self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

        def forward(self, x: Any, mask: Any) -> Any:
            h = self.fc(x)
            a = self.att(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
            a = a.masked_fill(mask == 0, float("-inf"))
            w = torch.softmax(a, dim=1)
            pooled = torch.einsum("bn,bnh->bh", w, h)
            return self.out(pooled)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ABMILMultiLabel(
            int(ckpt["input_dim"]),
            int(ckpt["hidden"]),
            int(ckpt["num_tasks"]),
            float(ckpt.get("dropout", 0.0)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return {
            "torch": torch,
            "model": model,
            "mean": np.asarray(ckpt["mean"], np.float32),
            "std": np.asarray(ckpt["std"], np.float32),
            "tasks": list(ckpt["tasks"]),
        }
    except Exception as exc:
        print(f"[secondary-mil] load failed: {type(exc).__name__}: {exc}")
        return None


def predict_secondary_findings(
    patch_features: np.ndarray | None,
    *,
    asset_name: str = "reg2_secondary_mil_heads",
) -> dict[str, dict[str, Any]]:
    """Return normalized finding -> score records."""
    if patch_features is None:
        return {}
    loaded = _load(asset_name)
    if not loaded:
        return {}
    x = np.asarray(patch_features, np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return {}
    mean = loaded["mean"]
    std = loaded["std"]
    if x.shape[1] != mean.shape[0]:
        return {}
    x = (x - mean) / std
    torch = loaded["torch"]
    model = loaded["model"]
    with torch.no_grad():
        xb = torch.from_numpy(x[None]).float()
        mb = torch.ones((1, x.shape[0]), dtype=torch.float32)
        probs = torch.sigmoid(model(xb, mb)).cpu().numpy()[0]
    out: dict[str, dict[str, Any]] = {}
    for task, prob in zip(loaded["tasks"], probs.tolist()):
        finding = _clean(task.get("finding"))
        if not finding:
            continue
        out[norm_finding(finding)] = {
            "prob": float(prob),
            "finding": finding,
            "task": str(task.get("name") or ""),
            "organ": _clean(task.get("organ")),
        }
    return out


def threshold_for_finding(organ: str, finding: str) -> float:
    """Default evidence thresholds; env vars can loosen/tighten per organ."""
    key = norm_finding(finding)
    organ_s = _clean(organ)
    if organ_s == "Urinary bladder" and "foreign body reaction" in key:
        return float(os.environ.get("REG2_SECONDARY_MIL_BLADDER_FOREIGN_TH", "0.98"))
    if organ_s == "Urinary bladder" and "urothelial carcinoma in situ" in key:
        return float(os.environ.get("REG2_SECONDARY_MIL_BLADDER_CIS_TH", "1.01"))
    if organ_s == "Breast" and "ductal carcinoma in situ" in key:
        return float(os.environ.get("REG2_SECONDARY_MIL_BREAST_DCIS_TH", "1.01"))
    if organ_s == "Breast" and "microcalcification" in key:
        return float(os.environ.get("REG2_SECONDARY_MIL_BREAST_MICRO_TH", "1.01"))
    return float(os.environ.get("REG2_SECONDARY_MIL_GENERIC_TH", "1.01"))
