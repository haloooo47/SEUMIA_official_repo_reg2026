#!/usr/bin/env python3
"""Train a compact 3-class CNN (tissue / scant / background) on raw ROI tiles.

Primary candidate for the Metric-B learned gate: tiny (few-MB) net that cold-loads
in well under a second and learns adipose / artifacts directly from pixels.

Reads the slide-split manifest from roi_tissue_build_dataset.py. Trains on the
`train` split, early-stops on `val` (balanced accuracy minus tissue false-refusal),
and writes weights + config to the model dir. Evaluation on `test` is done by the
separate roi_tissue_eval.py so training never sees the test slides.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = ["background", "scant", "tissue"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
BG_IDX = CLASS_TO_IDX["background"]
TISSUE_IDX = CLASS_TO_IDX["tissue"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/roi_tissue_clf_data"))
    p.add_argument("--out", type=Path, default=Path("runs/models/roi_tissue_clf/cnn"))
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


MEAN = np.array([0.70, 0.55, 0.68], dtype=np.float32)  # H&E-ish
STD = np.array([0.20, 0.22, 0.18], dtype=np.float32)


class RoiDataset(Dataset):
    def __init__(self, rows, img_size, train):
        self.rows = rows
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB").resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                a = a[:, ::-1, :]
            if np.random.rand() < 0.5:
                a = a[::-1, :, :]
            k = np.random.randint(0, 4)
            if k:
                a = np.rot90(a, k, axes=(0, 1))
            # mild brightness/contrast jitter
            if np.random.rand() < 0.5:
                a = np.clip(a * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.04, 0.04), 0, 1)
        a = (a - MEAN) / STD
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        return torch.from_numpy(a).float(), CLASS_TO_IDX[r["label"]]


class TinyNet(nn.Module):
    """~0.5M param CNN: 4 conv blocks + GAP + linear. Few-MB fp32 weights."""

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


def load_rows(data_dir):
    rows = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return rows


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in loader:
            logit = model(x.to(device))
            pred = logit.argmax(1).cpu().numpy()
            ps.append(pred)
            ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    # balanced accuracy
    accs = []
    for c in range(len(CLASSES)):
        m = y == c
        if m.any():
            accs.append((p[m] == c).mean())
    bal = float(np.mean(accs))
    tissue_mask = y == TISSUE_IDX
    tissue_false_refusal = float((p[tissue_mask] == BG_IDX).mean()) if tissue_mask.any() else 0.0
    bg_mask = y == BG_IDX
    bg_catch = float((p[bg_mask] == BG_IDX).mean()) if bg_mask.any() else 0.0
    return bal, tissue_false_refusal, bg_catch


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.data)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print(f"[cnn] train={len(tr)} val={len(va)} device={device}", flush=True)

    counts = np.array([sum(r["label"] == c for r in tr) for c in CLASSES], dtype=np.float64)
    print(f"[cnn] train class counts {dict(zip(CLASSES, counts.astype(int)))}", flush=True)
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    class_w = torch.tensor(weights, dtype=torch.float32, device=device)

    tr_ds = RoiDataset(tr, args.img_size, train=True)
    va_ds = RoiDataset(va, args.img_size, train=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = TinyNet(n_classes=len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cnn] params={n_params/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = -1e9
    best_path = args.out / "cnn.pt"
    history = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, run = 0, 0.0
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
        # objective: maximize balanced acc + bg catch, hard-penalize tissue false-refusal
        score = bal + bgc - 5.0 * tfr
        history.append({"epoch": ep, "loss": run / max(tot, 1), "val_bal_acc": bal,
                        "val_tissue_false_refusal": tfr, "val_bg_catch": bgc, "score": score})
        flag = ""
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "img_size": args.img_size, "mean": MEAN.tolist(), "std": STD.tolist(),
                        "width": 32, "arch": "TinyNet"}, best_path)
            flag = " *"
        print(f"[cnn] ep{ep:02d} loss={run/max(tot,1):.4f} val_bal={bal:.4f} "
              f"tissue_FR={tfr:.4f} bg_catch={bgc:.4f} score={score:.4f}{flag} ({time.time()-t0:.1f}s)", flush=True)

    (args.out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps({
        "classes": CLASSES, "img_size": args.img_size, "mean": MEAN.tolist(), "std": STD.tolist(),
        "arch": "TinyNet", "width": 32, "n_params": int(n_params), "weights_file": "cnn.pt",
    }, indent=2), encoding="utf-8")
    print(f"[cnn] best_score={best_score:.4f} -> {best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
