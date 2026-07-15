#!/usr/bin/env python3
"""Train primary_dx-only ABMIL with train-split OOF predictions.

This is the ABMIL counterpart to ``train_primary_dx_oof_mlp_blender.py``. It
produces:

  * train OOF probabilities for one patch-feature source
  * fold-averaged val probabilities
  * an optional val prediction JSON for official scoring

Run one source at a time to keep RAM/GPU usage predictable.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_reg2026_text_calibrator as ec  # noqa: E402
from reg2_text_calibrator import LabelContext, TextCalibrator  # noqa: E402
from train_primary_dx_stacker import ASSETS, COT, load_labels, make_target_vocab, special_label  # noqa: E402
from train_reg2026_multitask_heads import macro_f1  # noqa: E402

DEF_FEATURES = Path("features/hoptimus1_full_256")
DEF_OUT = Path("runs/primary_dx_oof_abmil_hopt1_full_256")
BASE_VAL = Path("runs/e_multifm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEF_FEATURES)
    p.add_argument("--source-name", default="hopt1_full_256_abmil_oof")
    p.add_argument("--patch-key", default="", help="Auto-detect when empty.")
    p.add_argument("--assets", type=Path, default=ASSETS)
    p.add_argument("--cot", type=Path, default=COT)
    p.add_argument("--out", type=Path, default=DEF_OUT)
    p.add_argument("--base-val", type=Path, default=BASE_VAL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=1, help="Gated attention heads; >1 is MH-ABMIL/nnMIL-style.")
    p.add_argument("--max-patches", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--class-weight-power", type=float, default=0.35)
    p.add_argument("--effnum-beta", type=float, default=0.0, help="Effective-number class weights; 0 uses inverse-frequency power.")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0, help="Early-stop patience on fold holdout dx acc; 0 disables.")
    p.add_argument(
        "--train-max-patches",
        type=int,
        default=0,
        help="Randomly subsample each training bag to at most this many patches; 0 keeps all loaded patches.",
    )
    p.add_argument(
        "--patch-dropout",
        type=float,
        default=0.0,
        help="Training-time random patch dropout after optional subsampling.",
    )
    p.add_argument(
        "--feature-noise",
        type=float,
        default=0.0,
        help="Stddev of Gaussian noise added to normalized training patch features.",
    )
    p.add_argument("--scaler-sample-patches", type=int, default=240000)
    p.add_argument("--save-fold-assets", action="store_true")
    p.add_argument(
        "--fold-asset-dir",
        type=Path,
        default=Path(""),
        help="Directory for deploy-format fold assets when --save-fold-assets is set.",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def effnum_weights(counts: np.ndarray, beta: float) -> np.ndarray:
    counts = np.maximum(np.asarray(counts, np.float64), 1.0)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_patch_key(features: Path, expected_dim: int | None = None) -> str:
    for path in sorted(features.glob("*.npz"))[:50]:
        data = np.load(path)
        for key in data.files:
            if not key.startswith("patch_features"):
                continue
            arr = data[key]
            if arr.ndim == 2 and (expected_dim is None or arr.shape[1] == expected_dim):
                return key
    raise RuntimeError(f"could not infer patch key in {features}")


def label_indices(case_ids: list[str], labels: dict[str, dict[str, str]], vocab: list[str]) -> np.ndarray:
    idx = {v: i for i, v in enumerate(vocab)}
    other = idx.get("<other>", -1)
    return np.array([idx.get(labels[cid]["primary_dx"], other) for cid in case_ids], np.int64)


def make_folds(y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_id = np.zeros(len(y), np.int64)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, yi in enumerate(y):
        by_class[int(yi)].append(i)
    for idxs in by_class.values():
        arr = np.asarray(idxs, np.int64)
        rng.shuffle(arr)
        for j, idx in enumerate(arr.tolist()):
            fold_id[idx] = j % folds
    return fold_id


def load_patch_features(
    features: Path,
    patch_key: str,
    case_ids: list[str],
    max_patches: int,
) -> tuple[list[str], list[np.ndarray]]:
    ids: list[str] = []
    mats: list[np.ndarray] = []
    for cid in case_ids:
        path = features / f"{cid}.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        if patch_key not in data.files:
            continue
        x = np.asarray(data[patch_key])[:max_patches]
        if x.ndim != 2 or x.shape[0] == 0:
            continue
        ids.append(cid)
        mats.append(x.astype(np.float16, copy=False))
    return ids, mats


def fit_scaler(mats: list[np.ndarray], idxs: np.ndarray, sample_patches: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = int(sample_patches)
    shuffled = np.asarray(idxs, np.int64).copy()
    rng.shuffle(shuffled)
    for i in shuffled:
        x = mats[int(i)]
        if x.shape[0] == 0:
            continue
        take = min(x.shape[0], max(1, remaining // max(1, len(shuffled))))
        if take < x.shape[0]:
            rows = rng.choice(x.shape[0], size=take, replace=False)
            chunks.append(np.asarray(x[rows], np.float32))
        else:
            chunks.append(np.asarray(x, np.float32))
        remaining -= take
        if remaining <= 0:
            break
    if not chunks:
        raise RuntimeError("empty scaler sample")
    allp = np.concatenate(chunks, axis=0)
    mean = allp.mean(0).astype(np.float32)
    std = (allp.std(0) + 1e-6).astype(np.float32)
    return mean, std


def collate(
    mats: list[np.ndarray],
    y: np.ndarray,
    idxs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: Any,
    torch: Any,
    rng: np.random.Generator | None = None,
    train_max_patches: int = 0,
    patch_dropout: float = 0.0,
    feature_noise: float = 0.0,
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
                if not bool(keep.any()):
                    keep[int(rng.integers(0, raw.shape[0]))] = True
                raw = raw[keep]
        prepared.append(raw)

    max_len = max(x.shape[0] for x in prepared)
    dim = mean.shape[0]
    x = np.zeros((len(idxs), max_len, dim), np.float32)
    mask = np.zeros((len(idxs), max_len), np.float32)
    for bi, raw in enumerate(prepared):
        arr = (raw - mean) / std
        if rng is not None and feature_noise > 0.0:
            arr = arr + rng.normal(0.0, feature_noise, size=arr.shape).astype(np.float32)
        x[bi, : arr.shape[0]] = arr
        mask[bi, : arr.shape[0]] = 1.0
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(y[idxs]).to(device),
    )


def eval_probs(probs: np.ndarray, y: np.ndarray, name: str, vocab_size: int) -> dict[str, Any]:
    pred = probs.argmax(1)
    out = {
        "name": name,
        "n": int(len(y)),
        "acc": float((pred == y).mean()),
        "macro_f1": macro_f1(y, pred, vocab_size),
    }
    for k in (3, 5, 10):
        top = np.argsort(-probs, axis=1)[:, : min(k, probs.shape[1])]
        out[f"top{k}"] = float(np.mean([int(y[i]) in top[i] for i in range(len(y))]))
    return out


def write_val_prediction(
    out_dir: Path,
    pred_idx: np.ndarray,
    val_ids: list[str],
    vocab: list[str],
    base_run: Path,
    assets: Path,
) -> dict[str, Any]:
    gt_cases = {ec.normalize_case_id(c.get("id")): c for c in load_json(base_run / "gt_val.json")}
    base_cases = {ec.normalize_case_id(c.get("id")): c for c in load_json(base_run / "pred_ensemble.json")}
    pred_by_id = {cid: int(ix) for cid, ix in zip(val_ids, pred_idx)}
    calib = TextCalibrator.from_assets(assets)
    gt_out: list[dict[str, Any]] = []
    pred_out: list[dict[str, Any]] = []
    base_ok = pred_ok = changed = improved = worsened = 0
    for cid, gt_case in gt_cases.items():
        if cid not in base_cases or cid not in pred_by_id:
            continue
        base_case = base_cases[cid]
        gt_ctx = ec.context_from_steps(ec.get_steps(gt_case))
        base_ctx = ec.context_from_steps(ec.get_steps(base_case))
        dx = vocab[pred_by_id[cid]]
        if special_label(dx):
            dx = ""
        ctx = LabelContext(
            organ=base_ctx.organ,
            procedure=base_ctx.procedure,
            diagnoses=[dx] if dx else [],
            histologic_type=base_ctx.histologic_type,
            grade=base_ctx.grade,
            behavior=base_ctx.behavior,
        )
        bo = base_ctx.primary_dx() == gt_ctx.primary_dx()
        po = ctx.primary_dx() == gt_ctx.primary_dx()
        base_ok += int(bo)
        pred_ok += int(po)
        changed += int(base_ctx.primary_dx() != ctx.primary_dx())
        improved += int(po and not bo)
        worsened += int(bo and not po)
        gt_out.append(gt_case)
        if base_ctx.primary_dx() == ctx.primary_dx():
            pred_out.append(copy.deepcopy(base_case))
        else:
            pred_out.append({"id": cid, "chain-of-thought": calib.render_from_labels(ctx, True)})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_val.json").write_text(json.dumps(gt_out, ensure_ascii=False), encoding="utf-8")
    (out_dir / "pred_abmil_oof.json").write_text(json.dumps(pred_out, ensure_ascii=False), encoding="utf-8")
    n = max(1, len(gt_out))
    return {
        "n": len(gt_out),
        "base_dx_acc": base_ok / n,
        "pred_dx_acc": pred_ok / n,
        "changed": changed,
        "improved": improved,
        "worsened": worsened,
    }


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
    val_raw = [cid for cid in split["val_case_ids"] if cid in labels]
    if args.limit:
        train_raw = train_raw[: args.limit]
        val_raw = val_raw[: max(1, args.limit // 5)]
    target_vocab = make_target_vocab(train_raw, labels)
    patch_key = args.patch_key or infer_patch_key(args.features)
    ids, mats = load_patch_features(args.features, patch_key, train_raw + val_raw, args.max_patches)
    id_set = set(ids)
    train_ids = [cid for cid in train_raw if cid in id_set]
    val_ids = [cid for cid in val_raw if cid in id_set]
    pos = {cid: i for i, cid in enumerate(ids)}
    ordered_ids = train_ids + val_ids
    ordered_mats = [mats[pos[cid]] for cid in ordered_ids]
    y_all = label_indices(ordered_ids, labels, target_vocab)
    train_n = len(train_ids)
    train_y = y_all[:train_n]
    val_y = y_all[train_n:]
    fold_id = make_folds(train_y, args.folds, args.seed)
    print(
        f"[oof-abmil] source={args.source_name} key={patch_key} train={train_n} "
        f"val={len(val_ids)} dim={ordered_mats[0].shape[1]} classes={len(target_vocab)} device={device}",
        flush=True,
    )

    class DxABMIL(nn.Module):
        def __init__(self, in_dim: int, hidden: int, classes: int, dropout: float, num_heads: int):
            super().__init__()
            self.num_heads = int(num_heads)
            self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
            self.att_V = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_heads)])
            self.att_U = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(self.num_heads)])
            self.att_w = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(self.num_heads)])
            self.head = nn.Linear(hidden * self.num_heads, classes)

        def forward(self, x: Any, mask: Any) -> Any:
            h = self.fc(x)
            pooled = []
            for i in range(self.num_heads):
                a = self.att_w[i](torch.tanh(self.att_V[i](h)) * torch.sigmoid(self.att_U[i](h)))
                a = a.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
                a = torch.softmax(a, dim=1)
                pooled.append((a * h).sum(1))
            z = torch.cat(pooled, dim=1) if self.num_heads > 1 else pooled[0]
            return self.head(z)

    C = len(target_vocab)
    oof = np.zeros((train_n, C), np.float32)
    val_fold = np.zeros((args.folds, len(val_ids), C), np.float32)
    fold_metrics: list[dict[str, Any]] = []
    all_train_indices = np.arange(train_n, dtype=np.int64)
    all_val_indices = np.arange(train_n, train_n + len(val_ids), dtype=np.int64)

    for fold in range(args.folds):
        tr = all_train_indices[fold_id != fold]
        ho = all_train_indices[fold_id == fold]
        mean, std = fit_scaler(ordered_mats, tr, args.scaler_sample_patches, args.seed + fold)
        counts = np.bincount(train_y[fold_id != fold], minlength=C).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        if args.effnum_beta > 0.0:
            weights = effnum_weights(counts, args.effnum_beta)
        else:
            weights = (counts.sum() / (C * counts)) ** float(args.class_weight_power)
            weights = weights / max(float(weights.mean()), 1e-6)
        model = DxABMIL(ordered_mats[0].shape[1], args.hidden, C, args.dropout, args.num_heads).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ce = nn.CrossEntropyLoss(
            weight=torch.from_numpy(weights.astype(np.float32)).to(device),
            label_smoothing=float(args.label_smoothing),
        )
        print(f"[oof-abmil] fold={fold} train={len(tr)} hold={len(ho)}", flush=True)
        best_state: dict[str, Any] | None = None
        best_acc = -1.0
        stale = 0
        for epoch in range(args.epochs):
            scale = lr_scale(epoch, args.epochs, args.warmup_epochs)
            for group in opt.param_groups:
                group["lr"] = args.lr * scale
            model.train()
            perm = np.random.permutation(tr)
            aug_rng = np.random.default_rng(args.seed + fold * 100000 + epoch)
            losses = []
            for start in range(0, len(perm), args.batch_size):
                bi = perm[start : start + args.batch_size]
                x, mask, yb = collate(
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
                    feature_noise=args.feature_noise,
                )
                loss = ce(model(x, mask), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            should_eval = args.patience > 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs
            if should_eval:
                model.eval()
                with torch.no_grad():
                    hold_probs = []
                    for start in range(0, len(ho), args.batch_size):
                        bi = ho[start : start + args.batch_size]
                        x, mask, _ = collate(ordered_mats, y_all, bi, mean, std, device, torch)
                        hold_probs.append(torch.softmax(model(x, mask), dim=1).detach().cpu().numpy())
                hold_probs_np = np.concatenate(hold_probs, axis=0).astype(np.float32)
                hold_acc = float((hold_probs_np.argmax(1) == train_y[fold_id == fold]).mean())
                if hold_acc > best_acc:
                    best_acc = hold_acc
                    stale = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                print(
                    f"[oof-abmil] fold={fold} epoch={epoch+1}/{args.epochs} "
                    f"lr={args.lr * scale:.2e} loss={np.mean(losses):.4f} hold={hold_acc:.4f} best={best_acc:.4f}",
                    flush=True,
                )
                if args.patience > 0 and stale >= args.patience:
                    print(f"[oof-abmil] fold={fold} early_stop epoch={epoch+1}", flush=True)
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        def predict(indices: np.ndarray) -> np.ndarray:
            outs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(indices), args.batch_size):
                    bi = indices[start : start + args.batch_size]
                    x, mask, _ = collate(ordered_mats, y_all, bi, mean, std, device, torch)
                    outs.append(torch.softmax(model(x, mask), dim=1).detach().cpu().numpy())
            return np.concatenate(outs, axis=0).astype(np.float32)

        oof[ho] = predict(ho)
        val_fold[fold] = predict(all_val_indices)
        fm = eval_probs(oof[ho], train_y[fold_id == fold], f"{args.source_name}_fold{fold}", C)
        fold_metrics.append(fm)
        print(f"[oof-abmil] fold={fold} oof_acc={fm['acc']:.4f} top5={fm['top5']:.4f}", flush=True)
        if args.save_fold_assets:
            root = args.fold_asset_dir if args.fold_asset_dir else (args.out / "fold_assets")
            fold_dir = root / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            state = {}
            for key, value in model.state_dict().items():
                if key == "head.weight":
                    state["heads.primary_dx.weight"] = value.detach().cpu()
                elif key == "head.bias":
                    state["heads.primary_dx.bias"] = value.detach().cpu()
                else:
                    state[key] = value.detach().cpu()
            torch.save(
                {
                    "model_state": state,
                    "hidden": args.hidden,
                    "num_heads": args.num_heads,
                    "head_sizes": {"primary_dx": C},
                    "scaler_mean": mean,
                    "scaler_std": std,
                    "max_patches": args.max_patches,
                    "patch_key": patch_key,
                    "source_name": args.source_name,
                    "fold": fold,
                },
                fold_dir / "abmil_heads.pt",
            )
            (fold_dir / "label_vocab.json").write_text(
                json.dumps({"vocabs": {"primary_dx": target_vocab}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[oof-abmil] saved fold asset -> {fold_dir}", flush=True)

    val_probs = val_fold.mean(0).astype(np.float32)
    val_probs /= np.maximum(val_probs.sum(1, keepdims=True), 1e-8)
    oof_metrics = eval_probs(oof, train_y, f"{args.source_name}_oof", C)
    val_metrics = eval_probs(val_probs, val_y, f"{args.source_name}_val_foldavg", C)
    print(
        f"[oof-abmil] OOF={oof_metrics['acc']:.4f} VAL={val_metrics['acc']:.4f} "
        f"top5={val_metrics['top5']:.4f}",
        flush=True,
    )
    e2e_val = write_val_prediction(
        args.out / "foldavg_val",
        val_probs.argmax(1),
        val_ids,
        target_vocab,
        args.base_val,
        args.assets,
    )
    arrays = args.out / "abmil_oof_probs.npz"
    np.savez_compressed(
        arrays,
        source_name=np.asarray(args.source_name, object),
        target_vocab=np.asarray(target_vocab, object),
        train_ids=np.asarray(train_ids, object),
        val_ids=np.asarray(val_ids, object),
        train_y=train_y,
        val_y=val_y,
        oof_probs=oof,
        val_probs=val_probs,
        fold_val_probs=val_fold,
    )
    summary = {
        "source_name": args.source_name,
        "features": str(args.features),
        "patch_key": patch_key,
        "train": train_n,
        "val": len(val_ids),
        "folds": args.folds,
        "epochs": args.epochs,
        "num_heads": args.num_heads,
        "train_max_patches": args.train_max_patches,
        "patch_dropout": args.patch_dropout,
        "feature_noise": args.feature_noise,
        "effnum_beta": args.effnum_beta,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "fold_metrics": fold_metrics,
        "oof": oof_metrics,
        "val": val_metrics,
        "e2e_val_dx": e2e_val,
        "arrays": str(arrays),
        "fold_asset_dir": str(args.fold_asset_dir if args.fold_asset_dir else (args.out / "fold_assets"))
        if args.save_fold_assets
        else "",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[oof-abmil] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
