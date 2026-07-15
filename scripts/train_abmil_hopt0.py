"""Train a gated-attention MIL (ABMIL) over CONCH-v1.5 patch features.

Replaces TITAN's frozen generic pooling with a task-supervised attention pooler
that can focus on diagnostic patches. Multi-task heads (organ/procedure/primary_dx/
grade/behavior/histologic_type); primary_dx is the target lever. Reports val dx
acc / macro-F1 vs the MLP-on-TITAN-embedding baseline (acc 0.719).

Inputs: per-slide npz with patch_features (N,768 fp16) + coords, from
extract_titan_features.py --save-patch-features.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from train_reg2026_multitask_heads import labels_from_steps, macro_f1, HEADS, NONE, OTHER, clean  # noqa: E402
import eval_reg2026_text_calibrator as ec  # noqa: E402

DEF_FEAT = Path("features/hoptimus0_patch_256")
DEF_COT = Path("data/train_CoT_v01.json")
DEF_SPLIT = Path("runs/calibrator_assets/v1/split.json")
DEF_OUT = Path("runs/models/abmil_hopt0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEF_FEAT)
    p.add_argument("--feat-key", type=str, default="patch_features_h0")
    p.add_argument("--cot", type=Path, default=DEF_COT)
    p.add_argument("--split", type=Path, default=DEF_SPLIT)
    p.add_argument("--out", type=Path, default=DEF_OUT)
    p.add_argument("--dx-topk", type=int, default=120)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--max-patches", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dx-loss-weight", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn as nn

    args.out.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text())
    train_ids, val_ids = set(split["train_case_ids"]), set(split["val_case_ids"])
    cases = json.loads(args.cot.read_text())

    label_by_id: dict[str, dict[str, str]] = {}
    for c in cases:
        cid = ec.normalize_case_id(c.get("id", ""))
        steps = c.get("chain-of-thought") or []
        if not steps:
            continue
        label_by_id[cid] = labels_from_steps(steps, clean(c.get("organ")))

    # Load patch features.
    feats: dict[str, np.ndarray] = {}
    for npz in sorted(args.features.glob("*.npz")):
        stem = npz.stem
        if stem not in label_by_id:
            continue
        d = np.load(npz)
        if args.feat_key not in d.files:
            continue
        pf = np.asarray(d[args.feat_key], dtype=np.float32)
        if pf.shape[0] == 0:
            continue
        feats[stem] = pf[: args.max_patches]
    print(f"[abmil] cases with patch features+labels: {len(feats)}", flush=True)

    train_stems = [s for s in feats if s in train_ids]
    val_stems = [s for s in feats if s in val_ids]
    print(f"[abmil] train={len(train_stems)} val={len(val_stems)}", flush=True)

    vocabs: dict[str, list[str]] = {}
    for head in HEADS:
        vals = [label_by_id[s][head] for s in train_stems]
        if head == "primary_dx":
            common = [v for v, _ in Counter(v for v in vals if v).most_common(args.dx_topk)]
            vocabs[head] = common + [OTHER]
        elif head in ("organ", "procedure"):
            vocabs[head] = sorted({v for v in vals if v})
        else:
            vocabs[head] = sorted({v if v else NONE for v in vals})
    idx_maps = {h: {c: i for i, c in enumerate(v)} for h, v in vocabs.items()}

    def label_index(head: str, raw: str) -> int:
        raw = raw if raw else (NONE if head not in ("organ", "procedure", "primary_dx") else "")
        m = idx_maps[head]
        if raw in m:
            return m[raw]
        if head == "primary_dx" and raw:
            return m[OTHER]
        return -1

    def make_Y(stems):
        return {h: np.array([label_index(h, label_by_id[s][h]) for s in stems], dtype=np.int64) for h in HEADS}

    Ytr, Yva = make_Y(train_stems), make_Y(val_stems)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # feature scaler (per-dim) from train patches
    allp = np.concatenate([feats[s] for s in train_stems], axis=0)
    mean = allp.mean(0).astype(np.float32); std = (allp.std(0) + 1e-6).astype(np.float32)

    def pack(stems):
        mats = [(feats[s] - mean) / std for s in stems]
        return mats

    Xtr, Xva = pack(train_stems), pack(val_stems)

    def collate(mats, ys, idxs):
        m = max(mats[i].shape[0] for i in idxs)
        B = len(idxs)
        x = np.zeros((B, m, mean.shape[0]), np.float32)
        mask = np.zeros((B, m), np.float32)
        for bi, i in enumerate(idxs):
            n = mats[i].shape[0]
            x[bi, :n] = mats[i]; mask[bi, :n] = 1.0
        yb = {h: torch.from_numpy(ys[h][idxs]).to(device) for h in HEADS}
        return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device), yb

    class GatedABMIL(nn.Module):
        def __init__(self, in_dim, hidden, head_sizes):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.25))
            self.att_V = nn.Linear(hidden, hidden)
            self.att_U = nn.Linear(hidden, hidden)
            self.att_w = nn.Linear(hidden, 1)
            self.heads = nn.ModuleDict({h: nn.Linear(hidden, n) for h, n in head_sizes.items()})

        def forward(self, x, mask):
            h = self.fc(x)                                   # (B,N,H)
            a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))  # (B,N,1)
            a = a.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            a = torch.softmax(a, dim=1)
            z = (a * h).sum(1)                               # (B,H)
            return {k: layer(z) for k, layer in self.heads.items()}

    head_sizes = {h: len(vocabs[h]) for h in HEADS}
    model = GatedABMIL(mean.shape[0], args.hidden, head_sizes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    n = len(train_stems)
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, args.batch_size):
            idxs = perm[i:i + args.batch_size]
            x, mask, yb = collate(Xtr, Ytr, idxs)
            out = model(x, mask)
            loss = sum((args.dx_loss_weight if h == "primary_dx" else 1.0) * ce(out[h], yb[h]) for h in HEADS)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 15 == 0 or epoch == args.epochs - 1:
            print(f"[abmil] epoch {epoch+1}/{args.epochs} loss={float(loss):.4f}", flush=True)

    # eval
    model.eval()
    metrics: dict[str, Any] = {"num_train": n, "num_val": len(val_stems)}
    preds = {h: [] for h in HEADS}
    with torch.no_grad():
        for i in range(0, len(val_stems), args.batch_size):
            idxs = list(range(i, min(i + args.batch_size, len(val_stems))))
            x, mask, _ = collate(Xva, Yva, idxs)
            out = model(x, mask)
            for h in HEADS:
                preds[h].append(out[h].argmax(1).cpu().numpy())
    for h in HEADS:
        pred = np.concatenate(preds[h]); gt = Yva[h]; m = gt >= 0
        if m.sum() == 0:
            continue
        metrics[f"{h}_acc"] = float((pred[m] == gt[m]).mean())
        metrics[f"{h}_macro_f1"] = macro_f1(gt[m], pred[m], len(vocabs[h]))

    torch.save({"model_state": model.state_dict(), "hidden": args.hidden,
                "head_sizes": head_sizes, "scaler_mean": mean, "scaler_std": std,
                "max_patches": args.max_patches}, args.out / "abmil_heads.pt")
    (args.out / "label_vocab.json").write_text(json.dumps({"vocabs": vocabs}, ensure_ascii=False, indent=2))
    (args.out / "val_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, indent=2))
    print(f"[abmil] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
