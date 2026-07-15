#!/usr/bin/env python3
"""Train compact multi-task heads: H-optimus slide features -> LabelContext.

Inputs:
  * feature store from scripts/extract_hoptimus_features.py (per-case .npz)
  * labels from train_CoT_v01.json
  * the calibrator split.json (shared train/val for comparable numbers)

Heads (small shared-trunk MLP probes on pooled embeddings):
  organ, procedure, primary_dx (top-K + <other>), grade, behavior, histologic_type.

Outputs (under --out):
  * multitask_heads.pt   (trunk + head weights + feature scaler)
  * label_vocab.json     (per-head class lists + feature config)
  * val_metrics.json     (per-head macro-F1, organ per-class recall)

Run on the 4090 box after Stage 1:
  python scripts/train_reg2026_multitask_heads.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_DIR / "src"))
from reg2_text_calibrator import canonicalize_question  # noqa: E402

DEFAULT_FEATURES = Path("features/hoptimus0_v2")
DEFAULT_COT = Path("data/train_CoT_v01.json")
DEFAULT_SPLIT = Path("runs/calibrator_assets/v1/split.json")
DEFAULT_OUT = Path("runs/models/multitask_v2")

ORGAN_Q = "what is the organ"
PROCEDURE_Q = "what is the procedure"
HISTOLOGIC_TYPE_Q = "what is the histologic type of neoplasm"
GRADE_Q = "what is the grade of neoplasm"
BEHAVIOR_Q = "what is the behavior of neoplasm"
DIAGNOSIS_Q_RE = re.compile(r"^what is the #(\d+) diagnosis$")

NONE = "<none>"
OTHER = "<other>"
HEADS = ["organ", "procedure", "primary_dx", "grade", "behavior", "histologic_type"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--cot", type=Path, default=DEFAULT_COT)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--input-keys", nargs="+", default=["pooled_mean", "pooled_topk"])
    p.add_argument("--dx-topk", type=int, default=60, help="Keep top-K primary diagnoses + <other>.")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=1, help="Number of hidden MLP layers in the shared trunk.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--primary-dx-loss-weight", type=float, default=1.0)
    p.add_argument(
        "--primary-dx-class-weight-power",
        type=float,
        default=0.0,
        help="0 disables class weighting; 0.5 is inverse-sqrt frequency; 1.0 is inverse frequency.",
    )
    p.add_argument("--primary-dx-focal-gamma", type=float, default=0.0)
    return p.parse_args()


def clean(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def labels_from_steps(steps: list[dict], organ_hint: str = "") -> dict[str, str]:
    organ = procedure = ht = grade = beh = ""
    diagnoses: dict[int, str] = {}
    for s in steps:
        cq = canonicalize_question(s.get("question", ""))
        a = clean(s.get("answer"))
        if cq == ORGAN_Q and not organ:
            organ = a
        elif cq == PROCEDURE_Q and not procedure:
            procedure = a
        elif cq == HISTOLOGIC_TYPE_Q and not ht:
            ht = a
        elif cq == GRADE_Q and not grade:
            grade = a
        elif cq == BEHAVIOR_Q and not beh:
            beh = a
        else:
            m = DIAGNOSIS_Q_RE.match(cq)
            if m:
                diagnoses.setdefault(int(m.group(1)), a)
    primary = diagnoses[min(diagnoses)] if diagnoses else ""
    return {
        "organ": organ or organ_hint,
        "procedure": procedure,
        "primary_dx": primary,
        "grade": grade,
        "behavior": beh,
        "histologic_type": ht,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if (y_true == c).sum() == 0:
            continue  # class absent in gt
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn as nn

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids, val_ids = set(split["train_case_ids"]), set(split["val_case_ids"])

    cases = json.loads(args.cot.read_text(encoding="utf-8"))
    label_by_id: dict[str, dict[str, str]] = {}
    for c in cases:
        cid = str(c.get("id", "")).strip()
        stem = cid[:-5] if cid.lower().endswith(".tiff") else cid
        steps = c.get("chain-of-thought") or []
        if not steps:
            continue
        label_by_id[stem] = labels_from_steps(steps, clean(c.get("organ")))

    feats: dict[str, np.ndarray] = {}
    for npz in sorted(args.features.glob("*.npz")):
        stem = npz.stem
        if stem not in label_by_id:
            continue
        data = np.load(npz)
        vec = np.concatenate([np.asarray(data[k], dtype=np.float32).ravel() for k in args.input_keys])
        feats[stem] = vec
    print(f"[train] feature store cases with labels: {len(feats)}")
    if not feats:
        print("[train] no features found yet (run Stage 1 first). Exiting.")
        return 0

    # Build per-head vocabularies from TRAIN cases that have features.
    train_stems = [s for s in feats if s in train_ids]
    val_stems = [s for s in feats if s in val_ids]
    print(f"[train] train={len(train_stems)} val={len(val_stems)}")

    vocabs: dict[str, list[str]] = {}
    for head in HEADS:
        vals = [label_by_id[s][head] for s in train_stems]
        if head == "primary_dx":
            common = [v for v, _ in Counter(v for v in vals if v).most_common(args.dx_topk)]
            vocabs[head] = common + [OTHER]
        elif head in ("organ", "procedure"):
            vocabs[head] = sorted({v for v in vals if v})
        else:  # grade / behavior / histologic_type: empty is a real "<none>" class
            vocabs[head] = sorted({v if v else NONE for v in vals})
    idx_maps = {h: {c: i for i, c in enumerate(v)} for h, v in vocabs.items()}

    def label_index(head: str, raw: str) -> int:
        raw = raw if raw else (NONE if head not in ("organ", "procedure", "primary_dx") else "")
        m = idx_maps[head]
        if raw in m:
            return m[raw]
        if head == "primary_dx" and raw:
            return m[OTHER]
        return -1  # masked (unknown organ/procedure or empty primary_dx)

    def make_xy(stems):
        X = np.stack([feats[s] for s in stems]).astype(np.float32)
        Y = {h: np.array([label_index(h, label_by_id[s][h]) for s in stems], dtype=np.int64) for h in HEADS}
        return X, Y

    Xtr, Ytr = make_xy(train_stems)
    Xva, Yva = make_xy(val_stems) if val_stems else (np.zeros((0, Xtr.shape[1]), np.float32), {h: np.zeros(0, np.int64) for h in HEADS})

    scaler_mean = Xtr.mean(axis=0)
    scaler_std = Xtr.std(axis=0) + 1e-6
    Xtr_n = (Xtr - scaler_mean) / scaler_std
    Xva_n = (Xva - scaler_mean) / scaler_std if len(Xva) else Xva

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} input_dim={Xtr.shape[1]}")

    class MultiHead(nn.Module):
        def __init__(self, in_dim, hidden, head_sizes, num_layers=1, dropout=0.1):
            super().__init__()
            layers: list[nn.Module] = []
            for layer_idx in range(max(1, int(num_layers))):
                layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden, hidden))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({h: nn.Linear(hidden, n) for h, n in head_sizes.items()})

        def forward(self, x):
            z = self.trunk(x)
            return {h: layer(z) for h, layer in self.heads.items()}

    head_sizes = {h: len(vocabs[h]) for h in HEADS}
    model = MultiHead(Xtr.shape[1], args.hidden, head_sizes, args.num_layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_by_head: dict[str, nn.Module] = {}
    for head in HEADS:
        weight = None
        if head == "primary_dx" and args.primary_dx_class_weight_power > 0:
            counts = np.bincount(Ytr[head][Ytr[head] >= 0], minlength=head_sizes[head]).astype(np.float32)
            counts = np.maximum(counts, 1.0)
            weight_np = (counts.sum() / (len(counts) * counts)) ** args.primary_dx_class_weight_power
            weight_np = weight_np / weight_np.mean()
            weight = torch.from_numpy(weight_np.astype(np.float32)).to(device)
        ce_by_head[head] = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=weight,
            label_smoothing=args.label_smoothing,
            reduction="none" if (head == "primary_dx" and args.primary_dx_focal_gamma > 0) else "mean",
        )

    def head_loss(head: str, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = ce_by_head[head](logits, target)
        if head == "primary_dx" and args.primary_dx_focal_gamma > 0:
            valid = target >= 0
            if not torch.any(valid):
                return loss.sum() * 0.0
            probs = torch.softmax(logits[valid], dim=1)
            pt = probs.gather(1, target[valid, None]).squeeze(1).clamp_min(1e-6)
            loss_valid = loss[valid] * ((1.0 - pt) ** args.primary_dx_focal_gamma)
            return loss_valid.mean()
        return loss

    xtr = torch.from_numpy(Xtr_n).to(device)
    ytr = {h: torch.from_numpy(Ytr[h]).to(device) for h in HEADS}
    n = xtr.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            bi = perm[i:i + args.batch_size]
            out = model(xtr[bi])
            loss = sum(
                (args.primary_dx_loss_weight if h == "primary_dx" else 1.0) * head_loss(h, out[h], ytr[h][bi])
                for h in HEADS
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            print(f"[train] epoch {epoch + 1}/{args.epochs} loss={float(loss):.4f}", flush=True)

    metrics: dict[str, Any] = {"num_train": len(train_stems), "num_val": len(val_stems)}
    if len(Xva):
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(Xva_n).to(device))
        for h in HEADS:
            pred = out[h].argmax(1).cpu().numpy()
            gt = Yva[h]
            mask = gt >= 0
            if mask.sum() == 0:
                continue
            metrics[f"{h}_macro_f1"] = macro_f1(gt[mask], pred[mask], len(vocabs[h]))
            metrics[f"{h}_acc"] = float((pred[mask] == gt[mask]).mean())
            metrics[f"{h}_n_val"] = int(mask.sum())
        # organ per-class recall
        if "organ" in vocabs:
            pred = out["organ"].argmax(1).cpu().numpy()
            gt = Yva["organ"]
            rec = {}
            for c, name in enumerate(vocabs["organ"]):
                m = gt == c
                if m.sum():
                    rec[name] = float((pred[m] == c).mean())
            metrics["organ_per_class_recall"] = rec

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_keys": args.input_keys,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "head_sizes": head_sizes,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        },
        args.out / "multitask_heads.pt",
    )
    (args.out / "label_vocab.json").write_text(
        json.dumps({
            "vocabs": vocabs,
            "input_keys": args.input_keys,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "dx_topk": args.dx_topk,
            "primary_dx_loss_weight": args.primary_dx_loss_weight,
            "primary_dx_class_weight_power": args.primary_dx_class_weight_power,
            "primary_dx_focal_gamma": args.primary_dx_focal_gamma,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
        },
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "val_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2, ensure_ascii=False))
    print(f"[train] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
