#!/usr/bin/env python3
"""TinyNet v2: harden the ROI tissue/background gate against artifacts.

Reuses the v1 slide-split dataset (manifest.jsonl + tiles) but adds LABEL-PRESERVING
artifact augmentation during training: pen/marker strokes, out-of-focus blur, blood
wash, air bubbles, tissue folds, scanner color cast, JPEG blocking. Artifacts overlay
content WITHOUT changing the label (a pen mark on tissue is still tissue; on blank is
still background), teaching the gate invariance to artifacts / scanner shift.

Same TinyNet arch + objective as v1 (balanced_acc + bg_catch - 5*tissue_false_refusal).
Evaluate separately with roi_tissue_eval.py --cnn <this out>/cnn.pt for an apples-to-apples
comparison on the SAME held-out test slides.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from roi_tissue_train_cnn import TinyNet, MEAN, STD, CLASSES, CLASS_TO_IDX, BG_IDX, TISSUE_IDX  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/roi_tissue_clf_data"))
    p.add_argument("--out", type=Path, default=Path("runs/models/roi_tissue_clf_v2/cnn"))
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--artifact-p", type=float, default=0.5, help="prob of applying >=1 artifact to a train tile")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------- label-preserving artifact augmentations (operate on float[0,1] HWC) ----------------
def add_pen_strokes(a, rng):
    h, w, _ = a.shape
    colors = [(0.10, 0.45, 0.20), (0.10, 0.20, 0.55), (0.05, 0.05, 0.05), (0.55, 0.10, 0.30)]
    for _ in range(rng.randint(1, 3)):
        col = np.array(colors[rng.randint(0, len(colors))], dtype=np.float32)
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        thick = rng.randint(2, 6)
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for t in np.linspace(0, 1, n):
            cx, cy = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
            yl, yh = max(0, cy - thick), min(h, cy + thick)
            xl, xh = max(0, cx - thick), min(w, cx + thick)
            a[yl:yh, xl:xh] = 0.5 * a[yl:yh, xl:xh] + 0.5 * col
    return a


def add_blur(a, rng):
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 4.0)))
    return np.asarray(img, dtype=np.float32) / 255.0


def add_blood(a, rng):
    h, w, _ = a.shape
    mask = np.zeros((h, w), np.float32)
    cy, cx = rng.randint(0, h), rng.randint(0, w)
    yy, xx = np.ogrid[:h, :w]
    r = rng.randint(h // 5, h // 2)
    mask[((yy - cy) ** 2 + (xx - cx) ** 2) < r * r] = rng.uniform(0.3, 0.7)
    red = np.array([0.55, 0.05, 0.10], np.float32)
    return a * (1 - mask[..., None]) + red * mask[..., None]


def add_bubble(a, rng):
    h, w, _ = a.shape
    cy, cx = rng.randint(0, h), rng.randint(0, w)
    yy, xx = np.ogrid[:h, :w]
    r = rng.randint(h // 6, h // 3)
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = np.clip(1.0 - np.abs(d - r) / 4.0, 0, 1).astype(np.float32)
    return np.clip(a + ring[..., None] * rng.uniform(0.2, 0.5), 0, 1)


def add_fold(a, rng):
    h, w, _ = a.shape
    band = rng.randint(h // 12, h // 5)
    y0 = rng.randint(0, max(1, h - band))
    a[y0:y0 + band] = a[y0:y0 + band] * rng.uniform(0.45, 0.7)
    return a


def add_colorcast(a, rng):
    cast = np.array([rng.uniform(0.9, 1.12) for _ in range(3)], np.float32)
    return np.clip(a * cast, 0, 1)


def add_jpeg(a, rng):
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(15, 40))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


ARTIFACTS = [add_pen_strokes, add_blur, add_blood, add_bubble, add_fold, add_colorcast, add_jpeg]


class RoiDatasetV2(Dataset):
    def __init__(self, rows, img_size, train, artifact_p):
        self.rows = rows
        self.img_size = img_size
        self.train = train
        self.artifact_p = artifact_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB").resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
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
            # label-preserving artifact augmentation
            if rng.rand() < self.artifact_p:
                for fn in rng.permutation(ARTIFACTS)[: rng.randint(1, 3)]:
                    a = np.ascontiguousarray(fn(a, rng))
            if rng.rand() < 0.5:
                a = np.clip(a * rng.uniform(0.9, 1.1) + rng.uniform(-0.04, 0.04), 0, 1)
        a = (a - MEAN) / STD
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        return torch.from_numpy(a).float(), CLASS_TO_IDX[r["label"]]


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    accs = [(p[y == c] == c).mean() for c in range(len(CLASSES)) if (y == c).any()]
    bal = float(np.mean(accs))
    tm = y == TISSUE_IDX
    tfr = float((p[tm] == BG_IDX).mean()) if tm.any() else 0.0
    bm = y == BG_IDX
    bgc = float((p[bm] == BG_IDX).mean()) if bm.any() else 0.0
    return bal, tfr, bgc


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in (args.data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print(f"[v2] train={len(tr)} val={len(va)} artifact_p={args.artifact_p} device={device}", flush=True)

    counts = np.array([sum(r["label"] == c for r in tr) for c in CLASSES], dtype=np.float64)
    class_w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)

    tr_ld = DataLoader(RoiDatasetV2(tr, args.img_size, True, args.artifact_p), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    va_ld = DataLoader(RoiDatasetV2(va, args.img_size, False, 0.0), batch_size=args.batch,
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
        print(f"[v2] ep{ep:02d} loss={run/max(tot,1):.4f} val_bal={bal:.4f} tissue_FR={tfr:.4f} "
              f"bg_catch={bgc:.4f} score={score:.4f}{flag} ({time.time()-t0:.1f}s)", flush=True)

    (args.out / "train_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps({
        "classes": CLASSES, "img_size": args.img_size, "mean": MEAN.tolist(), "std": STD.tolist(),
        "arch": "TinyNet", "width": 32, "weights_file": "cnn.pt", "artifact_p": args.artifact_p,
    }, indent=2), encoding="utf-8")
    print(f"[v2] best_score={best:.4f} -> {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
