#!/usr/bin/env python3
"""Train a single train-only primary_dx DxABMIL and save the checkpoint.

This is the clean (val-excluded) scoring model used by the dense pixel-coverage
gate (``diag_dense_coverage_check.py``). Architecture and preprocessing match
``train_primary_dx_oof_abmil.DxABMIL`` exactly so per-patch class-contribution
``a_p * (W_c . h_p)`` is meaningful. We save model_state + scaler + vocab so the
checker can re-pool arbitrary patch bags without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from train_primary_dx_oof_abmil import (  # noqa: E402
    fit_scaler,
    infer_patch_key,
    label_indices,
    load_json,
    load_patch_features,
)
from train_primary_dx_stacker import ASSETS, COT, load_labels, make_target_vocab  # noqa: E402

DEF_FEATURES = Path("features/hoptimus1_full_256")
DEF_OUT = Path("runs/models/dxabmil_hopt1_fulltrain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEF_FEATURES)
    p.add_argument("--patch-key", default="")
    p.add_argument("--assets", type=Path, default=ASSETS)
    p.add_argument("--cot", type=Path, default=COT)
    p.add_argument("--out", type=Path, default=DEF_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=1)
    p.add_argument("--max-patches", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--class-weight-power", type=float, default=0.35)
    p.add_argument("--effnum-beta", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--train-max-patches",
        type=int,
        default=0,
        help="Randomly subsample each training bag to at most this many patches; 0 keeps all patches.",
    )
    p.add_argument("--patch-dropout", type=float, default=0.0)
    p.add_argument("--feature-noise", type=float, default=0.0)
    p.add_argument("--scaler-sample-patches", type=int, default=240000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def effnum_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    counts = np.maximum(np.asarray(counts, np.float64), 1.0)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    w = w / max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def lr_scale(epoch: int, epochs: int, warmup: int) -> float:
    if warmup > 0 and epoch < warmup:
        return float(epoch + 1) / float(max(1, warmup))
    if epochs <= warmup:
        return 1.0
    progress = float(epoch - warmup) / float(max(1, epochs - warmup))
    return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn as nn

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    labels, _ = load_labels(args.cot)
    split = load_json(args.assets / "split.json")
    train_raw = [cid for cid in split["train_case_ids"] if cid in labels]
    if args.limit:
        train_raw = train_raw[: args.limit]
    target_vocab = make_target_vocab(train_raw, labels)
    patch_key = args.patch_key or infer_patch_key(args.features)
    ids, mats = load_patch_features(args.features, patch_key, train_raw, args.max_patches)
    mats = [np.asarray(m, np.float32) for m in mats]
    y = label_indices(ids, labels, target_vocab)
    dim = mats[0].shape[1]
    C = len(target_vocab)
    print(f"[fulltrain] key={patch_key} train={len(ids)} dim={dim} classes={C} device={device}", flush=True)

    class DxABMIL(nn.Module):
        def __init__(self, in_dim, hidden, classes, dropout, num_heads):
            super().__init__()
            self.num_heads = int(num_heads)
            self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
            self.att_V = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_heads)])
            self.att_U = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_heads)])
            self.att_w = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(self.num_heads)])
            self.heads = nn.ModuleDict({"primary_dx": nn.Linear(hidden * self.num_heads, classes)})

        def forward(self, x, mask):
            h = self.fc(x)
            pooled = []
            for i in range(self.num_heads):
                a = self.att_w[i](torch.tanh(self.att_V[i](h)) * torch.sigmoid(self.att_U[i](h)))
                a = a.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
                a = torch.softmax(a, dim=1)
                pooled.append((a * h).sum(1))
            z = torch.cat(pooled, dim=1) if self.num_heads > 1 else pooled[0]
            return {"primary_dx": self.heads["primary_dx"](z)}

    idx_all = np.arange(len(ids), dtype=np.int64)
    mean, std = fit_scaler(mats, idx_all, args.scaler_sample_patches, args.seed)
    counts = np.maximum(np.bincount(y, minlength=C).astype(np.float32), 1.0)
    if args.effnum_beta > 0.0:
        weights = effnum_weights(counts, args.effnum_beta)
    else:
        weights = (counts.sum() / (C * counts)) ** float(args.class_weight_power)
        weights = weights / max(float(weights.mean()), 1e-6)

    def collate(bi, rng=None):
        prepared = []
        for i in bi:
            raw = mats[int(i)]
            if rng is not None:
                if args.train_max_patches > 0 and raw.shape[0] > args.train_max_patches:
                    rows = rng.choice(raw.shape[0], size=args.train_max_patches, replace=False)
                    raw = raw[rows]
                if args.patch_dropout > 0.0 and raw.shape[0] > 1:
                    keep = rng.random(raw.shape[0]) >= args.patch_dropout
                    if not bool(keep.any()):
                        keep[int(rng.integers(0, raw.shape[0]))] = True
                    raw = raw[keep]
            prepared.append(raw)
        max_len = max(x.shape[0] for x in prepared)
        x = np.zeros((len(bi), max_len, dim), np.float32)
        mask = np.zeros((len(bi), max_len), np.float32)
        for k, raw in enumerate(prepared):
            arr = (raw - mean) / std
            if rng is not None and args.feature_noise > 0.0:
                arr = arr + rng.normal(0.0, args.feature_noise, size=arr.shape).astype(np.float32)
            x[k, : arr.shape[0]] = arr
            mask[k, : raw.shape[0]] = 1.0
        return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)

    model = DxABMIL(dim, args.hidden, C, args.dropout, args.num_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights.astype(np.float32)).to(device),
        label_smoothing=float(args.label_smoothing),
    )
    for epoch in range(args.epochs):
        scale = lr_scale(epoch, args.epochs, args.warmup_epochs)
        for group in opt.param_groups:
            group["lr"] = args.lr * scale
        model.train()
        perm = np.random.permutation(idx_all)
        aug_rng = np.random.default_rng(args.seed + epoch)
        losses = []
        for start in range(0, len(perm), args.batch_size):
            bi = perm[start : start + args.batch_size]
            x, mask = collate(bi, rng=aug_rng)
            yb = torch.from_numpy(y[bi]).to(device)
            loss = ce(model(x, mask)["primary_dx"], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % max(1, args.log_every) == 0 or epoch + 1 == args.epochs:
            print(
                f"[fulltrain] epoch={epoch+1}/{args.epochs} "
                f"lr={args.lr * scale:.2e} loss={np.mean(losses):.4f}",
                flush=True,
            )

    ckpt = args.out / "abmil_heads.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden": args.hidden,
            "num_heads": args.num_heads,
            "head_sizes": {"primary_dx": C},
            "scaler_mean": mean,
            "scaler_std": std,
            "max_patches": args.max_patches,
            "patch_key": patch_key,
        },
        ckpt,
    )
    (args.out / "label_vocab.json").write_text(
        json.dumps({"vocabs": {"primary_dx": target_vocab}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "meta.json").write_text(
        json.dumps(
            {"features": str(args.features), "patch_key": patch_key, "train": len(ids),
             "classes": C, "epochs": args.epochs, "hidden": args.hidden,
             "num_heads": args.num_heads,
             "train_max_patches": args.train_max_patches,
             "patch_dropout": args.patch_dropout,
             "feature_noise": args.feature_noise,
             "label_smoothing": args.label_smoothing},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[fulltrain] saved -> {ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
