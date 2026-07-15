#!/usr/bin/env python3
"""Whole-report classification head (decoupled A4 report generator).

Training reports are represented as whole-report classes. The model classifies
the complete report string, including secondary findings and notes, among the
most common report templates.
This learns co-occurrence as a whole and is template-faithful (each class IS a verbatim
training report -> exact KEY/ROUGE match when correct), runs <1ms online.

Decoupled: used ONLY for the final-report step; canonical path/answers (A1/A2/A3)
are unchanged. Predicted report falls back to the calibrator template when the
classifier is low-confidence.

Input: per-case feature (default H-opt1 pooled_mean 1536-d). Output under --out:
report_clf.pt (head + scaler + class report strings), val_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import eval_reg2026_text_calibrator as ec  # noqa: E402

DEF_FEAT = Path("features/hoptimus1_full_256")
DEF_COT = Path("data/train_CoT_v01.json")
DEF_SPLIT = Path("runs/calibrator_assets/v1/split.json")
DEF_OUT = Path("runs/models/report_clf_hopt1")
REPORT_Q = "What is the final pathology report?"
OTHER = "<other_report>"


def report_of(steps):
    for s in steps:
        if str(s.get("question", "")).strip() == REPORT_Q:
            return str(s.get("answer", "") or "")
    return ""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEF_FEAT)
    p.add_argument("--feat-key", type=str, default="pooled_mean")
    p.add_argument("--cot", type=Path, default=DEF_COT)
    p.add_argument("--split", type=Path, default=DEF_SPLIT)
    p.add_argument("--out", type=Path, default=DEF_OUT)
    p.add_argument("--min-count", type=int, default=2, help="min train freq to be its own class")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    import torch
    import torch.nn as nn

    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    split = json.loads(args.split.read_text())
    train_ids, val_ids = set(split["train_case_ids"]), set(split["val_case_ids"])
    rep_by_id = {}
    for c in json.load(open(args.cot)):
        cid = ec.normalize_case_id(c.get("id", ""))
        s = ec.get_steps(c)
        if s and report_of(s):
            rep_by_id[cid] = report_of(s)

    feats = {}
    for npz in sorted(args.features.glob("*.npz")):
        stem = npz.stem
        if stem not in rep_by_id:
            continue
        d = np.load(npz)
        if args.feat_key in d.files:
            feats[stem] = np.asarray(d[args.feat_key], np.float32)
    train_stems = [s for s in feats if s in train_ids]
    val_stems = [s for s in feats if s in val_ids]

    # report classes from train (>= min_count), rest -> OTHER
    rc = Counter(rep_by_id[s] for s in train_stems)
    classes = [r for r, n in rc.most_common() if n >= args.min_count]
    classes.append(OTHER)
    cidx = {r: i for i, r in enumerate(classes)}
    C = len(classes)
    other_i = cidx[OTHER]
    print(f"[repclf] train={len(train_stems)} val={len(val_stems)} report-classes={C} (incl OTHER)", flush=True)

    def lab(stem):
        return cidx.get(rep_by_id[stem], other_i)

    Xtr = np.stack([feats[s] for s in train_stems]).astype(np.float32)
    Ytr = np.array([lab(s) for s in train_stems], np.int64)
    Xva = np.stack([feats[s] for s in val_stems]).astype(np.float32)
    Yva = np.array([lab(s) for s in val_stems], np.int64)
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mean) / std; Xva_n = (Xva - mean) / std

    cnt = np.bincount(Ytr, minlength=C).astype(np.float64)
    w = np.clip(cnt.sum() / (C * np.maximum(cnt, 1)), 0.2, 10.0).astype(np.float32)
    cw = torch.from_numpy(w).to(dev)

    class Head(nn.Module):
        def __init__(s, d, h, c, dp):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dp), nn.Linear(h, c))
        def forward(s, x):
            return s.net(x)
    model = Head(Xtr.shape[1], args.hidden, C, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(weight=cw)
    xtr = torch.from_numpy(Xtr_n).to(dev); ytr = torch.from_numpy(Ytr).to(dev)
    n = len(xtr)

    # GT report text per val (for report-score-proxy = exact-match rate of predicted class report)
    val_gt = [rep_by_id[s] for s in val_stems]
    best_exact, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch_size):
            bi = perm[i:i+args.batch_size]
            loss = ce(model(xtr[bi]), ytr[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 10 == 0 or ep == args.epochs-1:
            model.eval()
            with torch.no_grad():
                pr = model(torch.from_numpy(Xva_n).to(dev)).argmax(1).cpu().numpy()
            # exact-match: predicted class report string == GT report
            exact = float(np.mean([classes[pr[i]] == val_gt[i] for i in range(len(val_stems))]))
            cls_acc = float((pr == Yva).mean())
            if exact > best_exact:
                best_exact = exact
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[repclf] ep{ep+1} cls_acc={cls_acc:.4f} report_exact_match={exact:.4f} best={best_exact:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "classes": classes, "hidden": args.hidden,
                "dropout": args.dropout, "scaler_mean": mean, "scaler_std": std,
                "feat_key": args.feat_key}, args.out / "report_clf.pt")
    (args.out / "val_metrics.json").write_text(json.dumps(
        {"report_classes": C, "best_report_exact_match": best_exact,
         "num_train": len(train_stems), "num_val": len(val_stems)}, indent=2))
    print(f"[repclf] saved -> {args.out} best_report_exact_match={best_exact:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
