#!/usr/bin/env python3
"""Train patch-level MIL heads for high-impact LabelGraph secondary findings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from train_primary_dx_oof_abmil import fit_scaler, infer_patch_key, load_patch_features  # noqa: E402
import eval_reg2026_text_calibrator as ec  # noqa: E402

DEF_FEATURES = Path("features/hoptimus1_full_256")
DEF_COT = Path("data/train_CoT_v01.json")
DEF_SPLIT = Path("runs/calibrator_assets/v1/split.json")
DEF_OUT = Path("runs/models/reg2_secondary_mil_heads")

TASKS = [
    {"name": "breast_dcis_secondary", "organ": "Breast", "finding": "Ductal carcinoma in situ"},
    {"name": "breast_microcalcification", "organ": "Breast", "finding": "Microcalcification"},
    {"name": "bladder_urothelial_cis_secondary", "organ": "Urinary bladder", "finding": "Urothelial carcinoma in situ"},
    {
        "name": "bladder_foreign_body_granulomatous_inflammation",
        "organ": "Urinary bladder",
        "finding": "Chronic granulomatous inflammation with foreign body reaction",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEF_FEATURES)
    p.add_argument("--patch-key", default="")
    p.add_argument("--cot", type=Path, default=DEF_COT)
    p.add_argument("--split", type=Path, default=DEF_SPLIT)
    p.add_argument("--out", type=Path, default=DEF_OUT)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--max-patches", type=int, default=256)
    p.add_argument("--train-max-patches", type=int, default=192)
    p.add_argument("--patch-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--pos-weight-cap", type=float, default=60.0)
    p.add_argument("--scaler-sample-patches", type=int, default=240000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task names to train; default trains all secondary MIL tasks.",
    )
    return p.parse_args()


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_key(value: Any) -> str:
    text = clean(value).lower()
    text = re.sub(r"[\s,.;:()/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def labels_from_cot(cot_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for case in json.load(open(cot_path, encoding="utf-8")):
        cid = ec.normalize_case_id(case.get("id", ""))
        ctx = ec.context_from_steps(ec.get_steps(case), organ_hint=ec.clean(case.get("organ")))
        dxs = ctx.clean_diagnoses()
        rows[cid] = {
            "organ": clean(ctx.organ),
            "primary_dx": dxs[0] if dxs else "",
            "secondary": dxs[1:] if len(dxs) > 1 else [],
        }
    return rows


def select_tasks(names: str) -> list[dict[str, str]]:
    requested = {x.strip() for x in str(names or "").split(",") if x.strip()}
    if not requested:
        return list(TASKS)
    tasks = [task for task in TASKS if task["name"] in requested]
    missing = sorted(requested - {task["name"] for task in tasks})
    if missing:
        raise ValueError(f"unknown --tasks entries: {missing}; valid={[t['name'] for t in TASKS]}")
    return tasks


def make_targets(case_ids: list[str], labels: dict[str, dict[str, Any]], tasks: list[dict[str, str]]) -> np.ndarray:
    y = np.zeros((len(case_ids), len(tasks)), np.float32)
    for i, cid in enumerate(case_ids):
        row = labels[cid]
        secondary = {norm_key(x) for x in row.get("secondary", [])}
        for j, task in enumerate(tasks):
            y[i, j] = float(row.get("organ") == task["organ"] and norm_key(task["finding"]) in secondary)
    return y


def collate_bags(
    mats: list[np.ndarray],
    y: np.ndarray,
    idxs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: Any,
    torch: Any,
    *,
    rng: np.random.Generator | None = None,
    train_max_patches: int = 0,
    patch_dropout: float = 0.0,
) -> tuple[Any, Any, Any]:
    prepared: list[np.ndarray] = []
    for i in idxs:
        raw = np.asarray(mats[int(i)], np.float32)
        if rng is not None:
            if train_max_patches > 0 and raw.shape[0] > train_max_patches:
                rows = rng.choice(raw.shape[0], size=train_max_patches, replace=False)
                raw = raw[rows]
            if patch_dropout > 0.0 and raw.shape[0] > 1:
                keep = rng.random(raw.shape[0]) >= patch_dropout
                if keep.any():
                    raw = raw[keep]
        prepared.append((raw - mean) / std)
    max_len = max(x.shape[0] for x in prepared)
    dim = prepared[0].shape[1]
    xb = np.zeros((len(prepared), max_len, dim), np.float32)
    mb = np.zeros((len(prepared), max_len), np.float32)
    for bi, x in enumerate(prepared):
        xb[bi, : x.shape[0]] = x
        mb[bi, : x.shape[0]] = 1.0
    yb = y[idxs]
    return (
        torch.from_numpy(xb).to(device),
        torch.from_numpy(mb).to(device),
        torch.from_numpy(yb).to(device),
    )


def eval_model(model: Any, mats: list[np.ndarray], y: np.ndarray, idxs: np.ndarray, mean: np.ndarray, std: np.ndarray, args: argparse.Namespace, device: Any, torch: Any) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(idxs), args.batch_size):
            bi = idxs[start : start + args.batch_size]
            xb, mb, _ = collate_bags(mats, y, bi, mean, std, device, torch)
            probs = torch.sigmoid(model(xb, mb)).cpu().numpy()
            outs.append(probs.astype(np.float32))
    return np.concatenate(outs, axis=0)


def threshold_stats(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for thr in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        pred = p >= thr
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[f"{thr:.2f}"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "actions": int(pred.sum()),
        }
    return out


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tasks = select_tasks(args.tasks)

    split = json.loads(args.split.read_text(encoding="utf-8"))
    labels = labels_from_cot(args.cot)
    train_raw = [ec.normalize_case_id(x) for x in split["train_case_ids"] if ec.normalize_case_id(x) in labels]
    val_raw = [ec.normalize_case_id(x) for x in split["val_case_ids"] if ec.normalize_case_id(x) in labels]
    if args.limit:
        train_raw = train_raw[: args.limit]
        val_raw = val_raw[: max(1, args.limit // 5)]
    patch_key = args.patch_key or infer_patch_key(args.features)
    ids, mats = load_patch_features(args.features, patch_key, train_raw + val_raw, args.max_patches)
    id_set = set(ids)
    train_ids = [cid for cid in train_raw if cid in id_set]
    val_ids = [cid for cid in val_raw if cid in id_set]
    pos = {cid: i for i, cid in enumerate(ids)}
    ordered_ids = train_ids + val_ids
    ordered_mats = [mats[pos[cid]] for cid in ordered_ids]
    y_all = make_targets(ordered_ids, labels, tasks)
    train_n = len(train_ids)
    y_train = y_all[:train_n]
    y_val = y_all[train_n:]
    train_idx = np.arange(train_n, dtype=np.int64)
    val_idx = np.arange(train_n, train_n + len(val_ids), dtype=np.int64)
    mean, std = fit_scaler(ordered_mats, train_idx, args.scaler_sample_patches, args.seed)
    dim = ordered_mats[0].shape[1]
    num_tasks = len(tasks)
    pos_count = y_train.sum(axis=0)
    neg_count = len(y_train) - pos_count
    pos_weight = np.clip(neg_count / np.maximum(pos_count, 1.0), 1.0, args.pos_weight_cap).astype(np.float32)
    print(
        f"[secondary-mil] key={patch_key} train={train_n} val={len(val_ids)} "
            f"dim={dim} tasks={num_tasks} task_names={[t['name'] for t in tasks]} "
            f"pos={pos_count.astype(int).tolist()} device={device}",
        flush=True,
    )

    class ABMILMultiLabel(nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
            self.att_v = nn.Linear(hidden, hidden)
            self.att_u = nn.Linear(hidden, hidden)
            self.att = nn.Linear(hidden, 1)
            self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

        def forward(self, x: Any, mask: Any) -> Any:
            h = self.fc(x)
            a = self.att(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
            a = a.masked_fill(mask == 0, float("-inf"))
            w = torch.softmax(a, dim=1)
            pooled = torch.einsum("bn,bnh->bh", w, h)
            return self.out(pooled)

    model = ABMILMultiLabel(dim, args.hidden, num_tasks, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    best_state = None
    best_loss = float("inf")
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(train_idx)
        losses = []
        aug_rng = np.random.default_rng(args.seed + epoch * 1009)
        for start in range(0, len(perm), args.batch_size):
            bi = perm[start : start + args.batch_size]
            xb, mb, yb = collate_bags(
                ordered_mats,
                y_all,
                bi,
                mean,
                std,
                device,
                torch,
                rng=aug_rng,
                train_max_patches=args.train_max_patches,
                patch_dropout=args.patch_dropout,
            )
            loss = crit(model(xb, mb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        p_val = eval_model(model, ordered_mats, y_all, val_idx, mean, std, args, device, torch)
        eps = 1e-7
        val_loss = -np.mean(
            y_val * np.log(np.clip(p_val, eps, 1 - eps)) + (1 - y_val) * np.log(np.clip(1 - p_val, eps, 1 - eps))
        )
        if val_loss < best_loss:
            best_loss = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[secondary-mil] epoch={epoch+1}/{args.epochs} train_loss={np.mean(losses):.4f} val_bce={val_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    p_val = eval_model(model, ordered_mats, y_all, val_idx, mean, std, args, device, torch)
    metrics: dict[str, Any] = {
        "num_train": train_n,
        "num_val": len(val_ids),
        "feature": str(args.features),
        "patch_key": patch_key,
        "tasks": {},
        "best_val_bce": best_loss,
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        for j, task in enumerate(tasks):
            yj = y_val[:, j]
            pj = p_val[:, j]
            auc = float(roc_auc_score(yj, pj)) if len(set(yj.tolist())) > 1 else 0.0
            ap = float(average_precision_score(yj, pj)) if len(set(yj.tolist())) > 1 else 0.0
            metrics["tasks"][task["name"]] = {
                "organ": task["organ"],
                "finding": task["finding"],
                "val_pos": int(yj.sum()),
                "auc": round(auc, 4),
                "ap": round(ap, 4),
                "thresholds": threshold_stats(yj, pj),
            }
            print(f"[secondary-mil] {task['name']} auc={auc:.3f} ap={ap:.3f}", flush=True)
    except Exception as exc:
        metrics["metric_error"] = f"{type(exc).__name__}: {exc}"

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": dim,
            "hidden": args.hidden,
            "num_tasks": num_tasks,
            "dropout": args.dropout,
            "mean": mean,
            "std": std,
            "patch_key": patch_key,
            "tasks": tasks,
        },
        args.out / "secondary_mil_heads.pt",
    )
    (args.out / "secondary_mil_vocab.json").write_text(json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2) + "\n")
    (args.out / "val_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
