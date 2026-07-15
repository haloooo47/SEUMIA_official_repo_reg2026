#!/usr/bin/env python3
"""TinyNet v3: add Macenko stain perturbation to the v2 artifact augmentation.

Motivation (PathGLS, arXiv 2603.16113): the reference-free pathology-VLM evaluation
that REG2026 Metric B closely mirrors uses **Macenko stain augmentation** as its
Stability/input-sensitivity visual perturbation. To make the ROI tissue/background
gate invariant to exactly that perturbation (so B2 original/perturbed pairs keep the
same gate class -> same canned answer), we add a label-preserving Macenko H&E
stain-concentration jitter (H,E each x U(0.7,1.3)) on top of v2's artifact set.

Stain perturbation is a no-op on background (no H&E to deconvolve -> returns the tile
unchanged), so it never flips a true background label; on tissue it only re-stains.

Same TinyNet arch + objective + slide split as v1/v2. Evaluate with roi_tissue_eval.py.
Macenko deconvolution ported from PathGLS/utils_img.py (CC-BY-NC-ND, academic use).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from roi_tissue_train_cnn import TinyNet, MEAN, STD, CLASSES, CLASS_TO_IDX  # noqa: E402
from roi_tissue_train_cnn_v2 import ARTIFACTS, evaluate  # noqa: E402


def _macenko_stain_matrix(od, beta=0.15, alpha=1):
    """Estimate the 3x2 H&E stain matrix from optical-density pixels (Macenko)."""
    od_hat = od[np.all(od > beta, axis=1)]
    if od_hat.shape[0] < 100:
        return None
    try:
        eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    except np.linalg.LinAlgError:
        return None
    V = eigvecs[:, 1:3]
    proj = od_hat @ V
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    v1 = V @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = V @ np.array([np.cos(max_phi), np.sin(max_phi)])
    HE = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
    norm = np.linalg.norm(HE, axis=0)
    if np.any(norm < 1e-6):
        return None
    return HE / norm


def add_macenko_stain(a, rng, h_range=(0.7, 1.3), e_range=(0.7, 1.3)):
    """Label-preserving Macenko H&E stain jitter on float[0,1] HWC RGB.

    Returns the tile unchanged when stain estimation fails (e.g. background tiles
    with too little tissue), which preserves the background label.
    """
    try:
        h, w, c = a.shape
        I = (np.clip(a, 0, 1).reshape((-1, 3)).astype(np.float64) * 255.0 + 1.0) / 255.0
        od = -np.log10(I)
        HE = _macenko_stain_matrix(od)
        if HE is None:
            return a
        C = od @ np.linalg.pinv(HE.T)
        C[:, 0] *= rng.uniform(*h_range)
        C[:, 1] *= rng.uniform(*e_range)
        od_aug = C @ HE.T
        out = np.power(10.0, -od_aug)  # invert OD = -log10(I)
        out = np.clip(out, 0.0, 1.0).astype(np.float32).reshape((h, w, c))
        return out
    except Exception:
        return a


# v3 augmentation pool = v2 artifacts + Macenko stain jitter
ARTIFACTS_V3 = list(ARTIFACTS) + [add_macenko_stain]


class RoiDatasetV3(Dataset):
    """Same as v2 but: (1) Macenko in the artifact pool, (2) an extra independent
    Macenko draw so the exact B2 perturbation is seen often regardless of which
    artifacts are sampled."""

    def __init__(self, rows, img_size, train, artifact_p, macenko_p=0.5):
        self.rows = rows
        self.img_size = img_size
        self.train = train
        self.artifact_p = artifact_p
        self.macenko_p = macenko_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.Resampling.BILINEAR
        )
        a = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            rng = np.random
            if rng.rand() < 0.5:
                a = a[:, ::-1, :]
            if rng.rand() < 0.5:
                a = a[::-1, :, :]
            k = rng.randint(0, 4)
            if k:
                a = np.rot90(a, k, axes=(0, 1))
            a = np.ascontiguousarray(a)
            # dedicated Macenko stain jitter (the exact PathGLS/Metric-B perturbation)
            if rng.rand() < self.macenko_p:
                a = np.ascontiguousarray(add_macenko_stain(a, rng))
            # general label-preserving artifact augmentation (incl. Macenko in pool)
            if rng.rand() < self.artifact_p:
                for fn in rng.permutation(ARTIFACTS_V3)[: rng.randint(1, 3)]:
                    a = np.ascontiguousarray(fn(a, rng))
            if rng.rand() < 0.5:
                a = np.clip(a * rng.uniform(0.9, 1.1) + rng.uniform(-0.04, 0.04), 0, 1)
        a = (a - MEAN) / STD
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        return torch.from_numpy(a).float(), CLASS_TO_IDX[r["label"]]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("runs/roi_tissue_clf_data"))
    ap.add_argument("--out", type=Path, default=Path("runs/models/roi_tissue_clf_v3"))
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--artifact-p", type=float, default=0.5)
    ap.add_argument("--macenko-p", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    import json
    import time
    import torch.nn.functional as F

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in (args.data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print(f"[v3] train={len(tr)} val={len(va)} artifact_p={args.artifact_p} macenko_p={args.macenko_p} device={device}", flush=True)

    counts = np.array([sum(r["label"] == c for r in tr) for c in CLASSES], dtype=np.float64)
    class_w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)

    tr_ld = DataLoader(RoiDatasetV3(tr, args.img_size, True, args.artifact_p, args.macenko_p), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    va_ld = DataLoader(RoiDatasetV3(va, args.img_size, False, 0.0, 0.0), batch_size=args.batch,
                       shuffle=False, num_workers=args.workers, pin_memory=True)

    model = TinyNet(n_classes=len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_path, hist = -1e9, args.out / "cnn.pt", []
    for ep in range(args.epochs):
        model.train()
        t0, tot, run = time.time(), 0, 0.0
        for x, y in tr_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=class_w)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            tot += x.size(0)
        sched.step()
        bal, tfr, bgc = evaluate(model, va_ld, device)
        score = bal + bgc - 5.0 * tfr
        hist.append({"epoch": ep, "loss": run / max(tot, 1), "val_bal_acc": bal,
                     "val_tissue_false_refusal": tfr, "val_bg_catch": bgc, "score": score})
        flag = ""
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "img_size": args.img_size,
                        "mean": MEAN.tolist(), "std": STD.tolist(), "width": 32, "arch": "TinyNet"}, best_path)
            flag = " *"
        print(f"[v3] ep{ep:02d} loss={run/max(tot,1):.4f} val_bal={bal:.4f} tissue_FR={tfr:.4f} "
              f"bg_catch={bgc:.4f} score={score:.4f}{flag} ({time.time()-t0:.1f}s)", flush=True)

    (args.out / "train_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps({
        "classes": CLASSES, "img_size": args.img_size, "mean": MEAN.tolist(), "std": STD.tolist(),
        "arch": "TinyNet", "width": 32, "weights_file": "cnn.pt", "artifact_p": args.artifact_p,
        "macenko_p": args.macenko_p,
    }, indent=2), encoding="utf-8")
    print(f"[v3] best_score={best:.4f} -> {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
