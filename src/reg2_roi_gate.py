"""Learned ROI tissue/background gate for Metric B (interface 0).

A compact CNN (TinyNet, ~2.35 MB) classifies the ROI thumbnail into
background / scant / tissue. It replaces the hand-tuned colour/OD heuristic gate,
which false-refused pale/adipose tissue and let artifact-laden backgrounds through.

Trained on slide-split ROIs with label-preserving artifact augmentation (pen /
blur / blood / bubble / fold / colour-cast) and validated non-circularly on a
held-out-by-slide split, with strong tissue recall, reliable background
rejection, and far more robustness to artifacts than the heuristic.

Only the GATE decision changes; the answer templates in reg2_v02_pipeline are
untouched, so Metric B's B2 (input sensitivity) is unaffected. Degrades gracefully:
on any load/inference failure the caller falls back to the heuristic gate.

Decision rule (FR=0 operating point): refuse-as-background iff argmax == background.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

CLASSES = ["background", "scant", "tissue"]
MODEL_ENV = "REG2_ROI_GATE_DIR"


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_model_dirs() -> list[Path]:
    cands: list[Path] = []
    env = os.environ.get(MODEL_ENV)
    if env:
        cands.append(Path(env))
    cands.extend([
        Path("/opt/ml/model/reg2_roi_gate"),
        _template_dir() / "model" / "reg2_roi_gate",
    ])
    seen, out = set(), []
    for p in cands:
        try:
            r = p.resolve()
        except OSError:
            r = p
        if r not in seen:
            seen.add(r)
            out.append(p)
    return out


def _find_weights() -> Path | None:
    for d in _candidate_model_dirs():
        w = d / "cnn.pt"
        if w.is_file():
            return w
    return None


@lru_cache(maxsize=1)
def _load():
    """Return (model, cfg, torch, F) or None if unavailable."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        w = _find_weights()
        if w is None:
            return None

        class TinyNet(nn.Module):
            def __init__(self, n_classes=3, width=32):
                super().__init__()
                c1, c2, c3, c4 = width, width * 2, width * 4, width * 4

                def block(ci, co):
                    return nn.Sequential(
                        nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                        nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                    )

                self.features = nn.Sequential(block(3, c1), block(c1, c2), block(c2, c3), block(c3, c4))
                self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                          nn.Dropout(0.2), nn.Linear(c4, n_classes))

            def forward(self, x):
                return self.head(self.features(x))

        ck = torch.load(str(w), map_location="cpu")
        classes = ck.get("classes", CLASSES)
        model = TinyNet(n_classes=len(classes), width=ck.get("width", 32)).eval()
        model.load_state_dict(ck["state_dict"])
        cfg = {
            "classes": classes,
            "img_size": int(ck["img_size"]),
            "mean": np.asarray(ck["mean"], dtype=np.float32),
            "std": np.asarray(ck["std"], dtype=np.float32),
        }
        torch.set_grad_enabled(False)
        return model, cfg, torch, F
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[roi-gate] load failed; falling back to heuristic: {type(exc).__name__}: {exc}")
        return None


def classify_roi_gate(roi_image) -> str | None:
    """Return 'background' / 'scant' / 'tissue', or None to signal fallback.

    Uses argmax over the 3-class CNN (conservative tissue-preferring operating point).
    """
    loaded = _load()
    if loaded is None:
        return None
    try:
        from PIL import Image  # noqa: PLC0415

        model, cfg, torch, F = loaded
        im = roi_image.convert("RGB").resize((cfg["img_size"], cfg["img_size"]), Image.Resampling.BILINEAR)
        a = (np.asarray(im, dtype=np.float32) / 255.0 - cfg["mean"]) / cfg["std"]
        x = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float().unsqueeze(0)
        with torch.inference_mode():
            p = F.softmax(model(x), 1)[0].cpu().numpy()
        return cfg["classes"][int(p.argmax())]
    except Exception as exc:
        print(f"[roi-gate] inference failed; falling back to heuristic: {type(exc).__name__}: {exc}")
        return None
