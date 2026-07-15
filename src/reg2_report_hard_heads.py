"""Online hard-confusion head inference for report-local patches."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.reg2_v11_pipeline import _find_asset_dir


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any] | None:
    try:
        import json
        import torch

        model_dir = _find_asset_dir("reg2_report_hard_heads", "hard_confusion_heads.pt")
        ckpt = torch.load(model_dir / "hard_confusion_heads.pt", map_location="cpu", weights_only=False)
        vocab_path = model_dir / "head_vocab.json"
        vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.is_file() else {}
        return {"ckpt": ckpt, "vocab": vocab, "model_dir": model_dir}
    except Exception as exc:
        print(f"[hard-heads] unavailable: {type(exc).__name__}: {exc}")
        return None


def _load_torch_model(bundle: dict[str, Any]):
    import torch
    import torch.nn as nn

    ckpt = bundle["ckpt"]
    heads = list(ckpt.get("heads") or bundle.get("vocab", {}).get("heads") or [])
    vocabs = ckpt.get("vocabs") or bundle.get("vocab", {}).get("vocabs") or {}
    head_sizes = ckpt.get("head_sizes") or {h: len(vocabs[h]) for h in heads}

    class MultiHeadMLP(nn.Module):
        def __init__(self):
            super().__init__()
            h = int(ckpt["hidden"])
            nl = int(ckpt.get("num_layers", 1))
            dp = float(ckpt.get("dropout", 0.2))
            d = int(np.asarray(ckpt["scaler_mean"]).shape[0])
            layers: list[nn.Module] = []
            for i in range(max(1, nl)):
                layers += [nn.Linear(d if i == 0 else h, h), nn.GELU(), nn.Dropout(dp)]
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({k: nn.Linear(h, int(head_sizes[k])) for k in heads})

        def forward(self, x):
            z = self.trunk(x)
            return {k: v(z) for k, v in self.heads.items()}

    model = MultiHeadMLP()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(ckpt["scaler_mean"], np.float32)
    std = np.asarray(ckpt["scaler_std"], np.float32)
    return torch, model, mean, std, heads, vocabs


@lru_cache(maxsize=1)
def _loaded_model():
    bundle = _bundle()
    if bundle is None:
        return None
    return _load_torch_model(bundle)


def predict_hard_heads(fused_feature: np.ndarray) -> dict[str, dict[str, Any]] | None:
    loaded = _loaded_model()
    if loaded is None:
        return None
    feat = np.asarray(fused_feature, np.float32).ravel()
    torch, model, mean, std, heads, vocabs = loaded
    if feat.shape != mean.shape:
        print(f"[hard-heads] feature dim mismatch {feat.shape} vs {mean.shape}")
        return None
    x = ((feat - mean) / std).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(x[None]))
    heads_out: dict[str, dict[str, Any]] = {}
    for head in heads:
        probs = torch.softmax(out[head], dim=1).cpu().numpy()[0]
        order = np.argsort(-probs)
        idx = int(order[0])
        top1 = float(probs[idx])
        top2 = float(probs[order[1]]) if len(order) > 1 else 0.0
        heads_out[head] = {
            "pred": str(vocabs[head][idx]),
            "prob": top1,
            "margin": top1 - top2,
        }
    return heads_out
