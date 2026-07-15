#!/usr/bin/env python3
"""Train report-structured slot heads for numeric pathology report fields.

This is deliberately narrower than whole-report classification. It targets the
measured residual buckets that current report templates cannot control well:

* prostate Gleason score / grade group / pattern-4 percent / tumor volume;
* breast Nottingham component scores / overall score;
* DCIS nuclear grade / necrosis slots.

The output contains slot_heads.pt, slot_vocab.json, validation metrics, and
validation predictions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from reg2_text_calibrator import canonicalize_question  # noqa: E402

DEF_FEATURES = Path("features/report_fused_titan_h1_v2_256")
DEF_COT = Path("data/train_CoT_v01.json")
DEF_SPLIT = Path("runs/calibrator_assets/v1/split.json")
DEF_OUT = Path("runs/models/report_slot_heads_fused")

NONE = "<none>"
OTHER = "<other>"

SLOT_QUESTIONS: dict[str, str] = {
    "gleason_score": "what is the gleason score",
    "gleason_pattern3_present": "is there any gleason pattern 3 present",
    "gleason_pattern4_present": "is there any gleason pattern 4 present",
    "gleason_pattern5_present": "is there any gleason pattern 5 present",
    "secondary_pattern_gt5": "what is the secondary pattern constituting more than 5% of tumor",
    "grade_group": "what is the grade group",
    "tumor_volume": "what is the tumor volume",
    "pattern4_percent": "what is the percentage of gleason pattern 4",
    "breast_tubule_score": "what is the score for tubular differentiation",
    "breast_nuclear_score": "what is the score for nuclear pleomorphism",
    "breast_mitotic_score": "what is the score for mitotic rate",
    "breast_overall_score": "what is the overall score",
    "dcis_architectural_pattern": "what is the architectural pattern of lesion",
    "dcis_nuclear_grade": "what is the nuclear grade of lesion",
    "dcis_necrosis_present": "is there any necrosis present",
    "dcis_necrosis_type": "what is the type of necrosis",
}
PROSTATE_SLOTS = [
    "gleason_score",
    "gleason_pattern3_present",
    "gleason_pattern4_present",
    "gleason_pattern5_present",
    "secondary_pattern_gt5",
    "grade_group",
    "tumor_volume",
    "pattern4_percent",
]
BREAST_SLOTS = [
    "breast_tubule_score",
    "breast_nuclear_score",
    "breast_mitotic_score",
    "breast_overall_score",
    "dcis_architectural_pattern",
    "dcis_nuclear_grade",
    "dcis_necrosis_present",
    "dcis_necrosis_type",
]
SLOTS = list(SLOT_QUESTIONS)


def clean(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def norm_case_id(raw: Any) -> str:
    cid = clean(raw)
    for suf in (".tiff", ".svs"):
        if cid.lower().endswith(suf):
            return cid[: -len(suf)]
    return cid


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> float:
    vals = []
    for c in range(n_cls):
        if not np.any(y_true == c):
            continue
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        vals.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def labels_from_steps(steps: list[dict[str, Any]]) -> dict[str, str]:
    out = {slot: "" for slot in SLOTS}
    q_to_slot = {q: slot for slot, q in SLOT_QUESTIONS.items()}
    for step in steps:
        slot = q_to_slot.get(canonicalize_question(step.get("question", "")))
        if slot and not out[slot]:
            out[slot] = clean(step.get("answer"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEF_FEATURES)
    p.add_argument("--feat-key", default="fused", help="Single npz key (legacy).")
    p.add_argument(
        "--feat-keys",
        nargs="+",
        default=None,
        help="Concatenate multiple npz keys (overrides --feat-key).",
    )
    p.add_argument("--cot", type=Path, default=DEF_COT)
    p.add_argument("--split", type=Path, default=DEF_SPLIT)
    p.add_argument("--valref", type=Path, default=None, help="Optional valref id list for extra metrics.")
    p.add_argument("--out", type=Path, default=DEF_OUT)
    p.add_argument(
        "--slots",
        nargs="+",
        default=None,
        choices=SLOTS,
        help="Train/eval only these slot heads (default: all).",
    )
    p.add_argument(
        "--organ",
        choices=("all", "prostate", "breast"),
        default="all",
        help="Keep cases whose CoT organ matches (before slot masking).",
    )
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    return p.parse_args()


def load_feature_vector(data: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray:
    parts = []
    for key in keys:
        if key not in data.files:
            raise KeyError(key)
        parts.append(np.asarray(data[key], dtype=np.float32).ravel())
    return np.concatenate(parts).astype(np.float32)


def case_organ_by_id(cot_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for case in json.loads(cot_path.read_text(encoding="utf-8")):
        cid = norm_case_id(case.get("id", ""))
        if cid:
            out[cid] = clean(case.get("organ", ""))
    return out


def active_slots(args: argparse.Namespace) -> list[str]:
    if args.slots:
        return list(args.slots)
    if args.organ == "prostate":
        return list(PROSTATE_SLOTS)
    if args.organ == "breast":
        return list(BREAST_SLOTS)
    return list(SLOTS)


def main() -> int:
    import torch
    import torch.nn as nn

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    slots = active_slots(args)
    feat_keys = list(args.feat_keys or [args.feat_key])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids = set(split["train_case_ids"])
    val_ids = set(split["val_case_ids"])
    valref_ids: set[str] = set()
    if args.valref and args.valref.is_file():
        raw = json.loads(args.valref.read_text(encoding="utf-8"))
        valref_ids = {norm_case_id(x) for x in raw}

    organ_by_id = case_organ_by_id(args.cot)

    labels: dict[str, dict[str, str]] = {}
    for case in json.loads(args.cot.read_text(encoding="utf-8")):
        cid = norm_case_id(case.get("id", ""))
        steps = case.get("chain-of-thought") or case.get("chain_of_thought") or []
        if cid and isinstance(steps, list):
            labels[cid] = labels_from_steps(steps)

    def organ_ok(cid: str) -> bool:
        if args.organ == "all":
            return True
        organ = organ_by_id.get(cid, "").lower()
        return organ == args.organ

    feats: dict[str, np.ndarray] = {}
    for npz in sorted(args.features.glob("*.npz")):
        cid = npz.stem
        if cid not in labels or not organ_ok(cid):
            continue
        data = np.load(npz)
        try:
            feats[cid] = load_feature_vector(data, feat_keys)
        except KeyError:
            continue

    train = [cid for cid in feats if cid in train_ids]
    val = [cid for cid in feats if cid in val_ids]
    print(
        f"[slot] features={args.features} keys={feat_keys} organ={args.organ} "
        f"slots={slots} train={len(train)} val={len(val)} gpu={args.gpu}",
        flush=True,
    )
    if not train or not val:
        raise SystemExit("[slot] missing train/val features")

    vocabs: dict[str, list[str]] = {}
    for slot in slots:
        cnt = Counter(labels[cid][slot] for cid in train if labels[cid][slot])
        vals = [v for v, n in cnt.most_common() if n >= args.min_count]
        vocabs[slot] = (vals + [OTHER]) if vals else [OTHER]
    idx = {s: {v: i for i, v in enumerate(vs)} for s, vs in vocabs.items()}

    def label_idx(slot: str, value: str) -> int:
        if not value:
            return -1
        return idx[slot].get(value, idx[slot][OTHER])

    Xtr = np.stack([feats[cid] for cid in train]).astype(np.float32)
    Xva = np.stack([feats[cid] for cid in val]).astype(np.float32)
    Ytr = {s: np.array([label_idx(s, labels[cid][s]) for cid in train], np.int64) for s in slots}
    Yva = {s: np.array([label_idx(s, labels[cid][s]) for cid in val], np.int64) for s in slots}

    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    class SlotMLP(nn.Module):
        def __init__(self, d: int, h: int, heads: dict[str, int], nl: int, dp: float):
            super().__init__()
            layers: list[nn.Module] = []
            for i in range(max(1, nl)):
                layers += [nn.Linear(d if i == 0 else h, h), nn.GELU(), nn.Dropout(dp)]
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in heads.items()})

        def forward(self, x):
            z = self.trunk(x)
            return {k: v(z) for k, v in self.heads.items()}

    head_sizes = {s: len(v) for s, v in vocabs.items()}
    model = SlotMLP(Xtr.shape[1], args.hidden, head_sizes, args.num_layers, args.dropout).to(dev)
    losses: dict[str, nn.Module] = {}
    for slot in slots:
        y = Ytr[slot]
        valid = y >= 0
        weight = None
        if valid.any() and args.class_weight_power > 0:
            counts = np.bincount(y[valid], minlength=head_sizes[slot]).astype(np.float32)
            counts = np.maximum(counts, 1.0)
            w = (counts.sum() / (len(counts) * counts)) ** args.class_weight_power
            w = w / w.mean()
            weight = torch.from_numpy(w.astype(np.float32)).to(dev)
        losses[slot] = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xtr = torch.from_numpy(Xtr).to(dev)
    ytr = {s: torch.from_numpy(Ytr[s]).to(dev) for s in slots}
    n = len(train)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        last = 0.0
        for i in range(0, n, args.batch_size):
            bi = perm[i : i + args.batch_size]
            out = model(xtr[bi])
            parts = []
            for slot in slots:
                if torch.any(ytr[slot][bi] >= 0):
                    parts.append(losses[slot](out[slot], ytr[slot][bi]))
            loss = sum(parts) / max(1, len(parts))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss)
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            print(f"[slot] ep{ep+1}/{args.epochs} loss={last:.4f}", flush=True)

    model.eval()
    xva = torch.from_numpy(Xva).to(dev)
    metrics: dict[str, Any] = {
        "num_train": len(train),
        "num_val": len(val),
        "features": str(args.features),
        "feat_keys": feat_keys,
        "organ": args.organ,
        "slots": slots,
        "gpu": args.gpu,
    }
    val_preds: dict[str, dict[str, Any]] = {cid: {} for cid in val}
    valref_mask = np.array([cid in valref_ids for cid in val], dtype=bool) if valref_ids else None

    with torch.no_grad():
        out = model(xva)
        for slot in slots:
            probs_t = torch.softmax(out[slot], dim=1)
            pred = probs_t.argmax(1).cpu().numpy()
            conf = probs_t.max(1).values.cpu().numpy()
            gt = Yva[slot]
            mask = gt >= 0
            metrics[f"{slot}_n_val"] = int(mask.sum())
            metrics[f"{slot}_classes"] = int(len(vocabs[slot]))
            if mask.any():
                metrics[f"{slot}_acc"] = float((pred[mask] == gt[mask]).mean())
                metrics[f"{slot}_macro_f1"] = macro_f1(gt[mask], pred[mask], len(vocabs[slot]))
                metrics[f"{slot}_majority_acc"] = float(
                    Counter(gt[mask].tolist()).most_common(1)[0][1] / int(mask.sum())
                )
                metrics[f"{slot}_beats_majority"] = bool(
                    metrics[f"{slot}_acc"] > metrics[f"{slot}_majority_acc"] + 1e-9
                )
                if valref_mask is not None:
                    mref = mask & valref_mask
                    if mref.any():
                        metrics[f"{slot}_n_valref"] = int(mref.sum())
                        metrics[f"{slot}_acc_valref"] = float((pred[mref] == gt[mref]).mean())
                        metrics[f"{slot}_majority_acc_valref"] = float(
                            Counter(gt[mref].tolist()).most_common(1)[0][1] / int(mref.sum())
                        )
                        metrics[f"{slot}_beats_majority_valref"] = bool(
                            metrics[f"{slot}_acc_valref"] > metrics[f"{slot}_majority_acc_valref"] + 1e-9
                        )
            for vi, cid in enumerate(val):
                pi = int(pred[vi])
                gi = int(gt[vi])
                val_preds[cid][slot] = {
                    "pred": vocabs[slot][pi],
                    "prob": float(conf[vi]),
                    "gt": vocabs[slot][gi] if gi >= 0 else "",
                }

    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "head_sizes": head_sizes,
            "scaler_mean": mean,
            "scaler_std": std,
            "feat_keys": feat_keys,
            "organ": args.organ,
            "slots": slots,
        },
        args.out / "slot_heads.pt",
    )
    (args.out / "slot_vocab.json").write_text(
        json.dumps(
            {
                "slots": slots,
                "slot_questions": {k: SLOT_QUESTIONS[k] for k in slots},
                "vocabs": vocabs,
                "features": str(args.features),
                "feat_keys": feat_keys,
                "organ": args.organ,
                "hidden": args.hidden,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out / "val_predictions.json").write_text(
        json.dumps(val_preds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[slot] saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
