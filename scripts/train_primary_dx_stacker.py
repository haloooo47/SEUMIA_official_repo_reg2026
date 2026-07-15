#!/usr/bin/env python3
"""Train a deployable primary_dx stacker from existing REG2 visual heads.

This script keeps the perception backbone fixed and learns a dynamic selector over the
already extracted H-Optimus-1, Virchow2, TITAN, and fused head probabilities.

Outputs:
  * primary_dx_stacker.pt / metadata.json
  * summary.json with source, ensemble, and stacker dx metrics
  * gt_val.json + pred_stacker_val.json for official workflow scoring
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_reg2026_text_calibrator as ec  # noqa: E402
from reg2_text_calibrator import LabelContext, TextCalibrator  # noqa: E402
from train_reg2026_multitask_heads import (  # noqa: E402
    HEADS,
    NONE,
    OTHER,
    clean,
    labels_from_steps,
    macro_f1,
)

ASSETS = Path("runs/calibrator_assets/v1")
COT = Path("data/train_CoT_v01.json")
OUT = Path("runs/models/primary_dx_stacker_v1")
RUN_OUT = Path("runs/primary_dx_stacker_v1")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    model_dir: Path
    feat_dir: Path


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "fused_mlp",
        "mlp",
        Path("runs/models/multitask_fused_titan_h1_v2"),
        Path("features/report_fused_titan_h1_v2_256"),
    ),
    SourceSpec(
        "titan_full_mlp",
        "mlp",
        Path("runs/models/multitask_titan_full"),
        Path("features/titan_full_256"),
    ),
    SourceSpec(
        "hopt1_full_mlp",
        "mlp",
        Path("runs/models/multitask_hopt1_full"),
        Path("features/hoptimus1_full_256"),
    ),
    SourceSpec(
        "hopt1_full_abmil",
        "abmil",
        Path("runs/models/abmil_hopt1_full"),
        Path("features/hoptimus1_full_256"),
    ),
    SourceSpec(
        "hopt1_1024_abmil",
        "abmil",
        Path("runs/models/abmil_hopt1_1024"),
        Path("features/hoptimus1_full_1024"),
    ),
    SourceSpec(
        "virchow2_full_abmil",
        "abmil",
        Path("runs/models/abmil_virchow2_full"),
        Path("features/virchow2_full_256"),
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ASSETS)
    p.add_argument("--cot", type=Path, default=COT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--run-out", type=Path, default=RUN_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--class-weight-power", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.01)
    p.add_argument("--seeds", default="11,17,23")
    p.add_argument("--context-source", default="fused_mlp")
    p.add_argument("--no-e2e", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x).astype(np.float32)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-8)


def entropy_norm(p: np.ndarray) -> float:
    p = np.asarray(p, np.float32)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(float(p.size)))


def top_probs(p: np.ndarray, k: int = 5) -> list[float]:
    if p.size == 0:
        return [0.0] * k
    vals = np.sort(p)[::-1][:k].astype(np.float32).tolist()
    return vals + [0.0] * (k - len(vals))


def special_label(v: str) -> bool:
    return v in {"", NONE, OTHER, "<other_report>"}


def load_labels(cot: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    labels: dict[str, dict[str, str]] = {}
    steps_by_id: dict[str, list[dict[str, str]]] = {}
    for case in json.loads(cot.read_text(encoding="utf-8")):
        cid = ec.normalize_case_id(case.get("id", ""))
        steps = ec.get_steps(case)
        if not steps:
            continue
        labels[cid] = labels_from_steps(steps, clean(case.get("organ")))
        steps_by_id[cid] = steps
    return labels, steps_by_id


def make_target_vocab(train_ids: list[str], labels: dict[str, dict[str, str]]) -> list[str]:
    counts = Counter(labels[cid]["primary_dx"] for cid in train_ids if labels[cid]["primary_dx"])
    vocab = [dx for dx, _ in counts.most_common()]
    if OTHER not in vocab:
        vocab.append(OTHER)
    return vocab


def make_head_vocabs(train_ids: list[str], labels: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    vocabs: dict[str, list[str]] = {}
    for head in HEADS:
        vals = [labels[cid][head] for cid in train_ids]
        if head == "primary_dx":
            continue
        if head in ("organ", "procedure"):
            vocabs[head] = sorted({v for v in vals if v})
        else:
            vocabs[head] = sorted({v if v else NONE for v in vals})
    return vocabs


def label_to_index(vocab: list[str], value: str, *, allow_other: bool = True) -> int:
    m = {v: i for i, v in enumerate(vocab)}
    if value in m:
        return m[value]
    if allow_other and OTHER in m and value:
        return m[OTHER]
    return -1


class MLPSource:
    def __init__(self, spec: SourceSpec, torch: Any, nn: Any, device: Any):
        self.spec = spec
        self.torch = torch
        self.device = device
        ck = torch.load(spec.model_dir / "multitask_heads.pt", map_location="cpu", weights_only=False)
        self.vocabs: dict[str, list[str]] = json.loads(
            (spec.model_dir / "label_vocab.json").read_text(encoding="utf-8")
        )["vocabs"]
        self.mean = np.asarray(ck["scaler_mean"], np.float32)
        self.std = np.asarray(ck["scaler_std"], np.float32)
        self.input_keys = list(ck.get("input_keys", ["slide_embedding"]))

        class MLPHead(nn.Module):
            def __init__(self, d: int, h: int, hs: dict[str, int], nl: int, dp: float):
                super().__init__()
                layers: list[nn.Module] = []
                for i in range(max(1, int(nl))):
                    layers += [nn.Linear(d if i == 0 else h, h), nn.GELU(), nn.Dropout(float(dp))]
                self.trunk = nn.Sequential(*layers)
                self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in hs.items()})

            def forward(self, x: Any) -> dict[str, Any]:
                z = self.trunk(x)
                return {k: layer(z) for k, layer in self.heads.items()}

        self.model = MLPHead(
            self.mean.shape[0],
            int(ck["hidden"]),
            ck["head_sizes"],
            int(ck.get("num_layers", 1)),
            float(ck.get("dropout", 0.1)),
        ).to(device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

    def _load_vec(self, cid: str) -> np.ndarray | None:
        path = self.spec.feat_dir / f"{cid}.npz"
        if not path.is_file():
            return None
        data = np.load(path)
        try:
            vec = np.concatenate([np.asarray(data[k], dtype=np.float32).ravel() for k in self.input_keys])
        except KeyError:
            return None
        if vec.shape != self.mean.shape:
            return None
        return (vec - self.mean) / self.std

    def predict_many(self, case_ids: list[str], batch_size: int) -> dict[str, dict[str, np.ndarray]]:
        torch = self.torch
        rows: list[tuple[str, np.ndarray]] = []
        for cid in case_ids:
            vec = self._load_vec(cid)
            if vec is not None:
                rows.append((cid, vec.astype(np.float32)))
        out: dict[str, dict[str, np.ndarray]] = {}
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            x = torch.from_numpy(np.stack([r[1] for r in batch])).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
            probs = {h: torch.softmax(logits[h], dim=1).detach().cpu().numpy() for h in HEADS}
            for bi, (cid, _) in enumerate(batch):
                out[cid] = {h: probs[h][bi].astype(np.float32) for h in HEADS}
        return out


class ABMILSource:
    def __init__(self, spec: SourceSpec, torch: Any, nn: Any, device: Any):
        self.spec = spec
        self.torch = torch
        self.device = device
        ck = torch.load(spec.model_dir / "abmil_heads.pt", map_location="cpu", weights_only=False)
        self.vocabs: dict[str, list[str]] = json.loads(
            (spec.model_dir / "label_vocab.json").read_text(encoding="utf-8")
        )["vocabs"]
        self.mean = np.asarray(ck["scaler_mean"], np.float32)
        self.std = np.asarray(ck["scaler_std"], np.float32)
        self.max_patches = int(ck.get("max_patches", 256))
        self.patch_key = self._infer_patch_key()

        class GatedABMIL(nn.Module):
            def __init__(self, d: int, h: int, hs: dict[str, int]):
                super().__init__()
                self.fc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.25))
                self.att_V = nn.Linear(h, h)
                self.att_U = nn.Linear(h, h)
                self.att_w = nn.Linear(h, 1)
                self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in hs.items()})

            def forward(self, x: Any, mask: Any) -> dict[str, Any]:
                h = self.fc(x)
                a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))
                a = a.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
                a = torch.softmax(a, dim=1)
                z = (a * h).sum(1)
                return {k: layer(z) for k, layer in self.heads.items()}

        self.model = GatedABMIL(self.mean.shape[0], int(ck["hidden"]), ck["head_sizes"]).to(device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

    def _infer_patch_key(self) -> str:
        for path in sorted(self.spec.feat_dir.glob("*.npz"))[:20]:
            data = np.load(path)
            for key in data.files:
                if key.startswith("patch_features"):
                    arr = data[key]
                    if arr.ndim == 2 and arr.shape[1] == self.mean.shape[0]:
                        return key
        raise RuntimeError(f"could not infer patch key for {self.spec.name} in {self.spec.feat_dir}")

    def _load_patches(self, cid: str) -> np.ndarray | None:
        path = self.spec.feat_dir / f"{cid}.npz"
        if not path.is_file():
            return None
        data = np.load(path)
        if self.patch_key not in data.files:
            return None
        x = np.asarray(data[self.patch_key], dtype=np.float32)[: self.max_patches]
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.mean.shape[0]:
            return None
        return ((x - self.mean) / self.std).astype(np.float32)

    def predict_many(self, case_ids: list[str], batch_size: int) -> dict[str, dict[str, np.ndarray]]:
        torch = self.torch
        rows: list[tuple[str, np.ndarray]] = []
        for cid in case_ids:
            x = self._load_patches(cid)
            if x is not None:
                rows.append((cid, x))
        out: dict[str, dict[str, np.ndarray]] = {}
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            max_len = max(x.shape[0] for _, x in batch)
            dim = batch[0][1].shape[1]
            xb = np.zeros((len(batch), max_len, dim), np.float32)
            mask = np.zeros((len(batch), max_len), np.float32)
            for bi, (_, x) in enumerate(batch):
                xb[bi, : x.shape[0]] = x
                mask[bi, : x.shape[0]] = 1.0
            with torch.no_grad():
                logits = self.model(
                    torch.from_numpy(xb).to(self.device),
                    torch.from_numpy(mask).to(self.device),
                )
            probs = {h: torch.softmax(logits[h], dim=1).detach().cpu().numpy() for h in HEADS}
            for bi, (cid, _) in enumerate(batch):
                out[cid] = {h: probs[h][bi].astype(np.float32) for h in HEADS}
        return out


def build_source(spec: SourceSpec, torch: Any, nn: Any, device: Any) -> Any:
    if spec.kind == "mlp":
        return MLPSource(spec, torch, nn, device)
    if spec.kind == "abmil":
        return ABMILSource(spec, torch, nn, device)
    raise ValueError(f"unknown source kind: {spec.kind}")


def align_probs(raw: np.ndarray, raw_vocab: list[str], target_vocab: list[str]) -> np.ndarray:
    target_idx = {v: i for i, v in enumerate(target_vocab)}
    out = np.zeros(len(target_vocab), np.float32)
    other_i = target_idx.get(OTHER)
    for j, label in enumerate(raw_vocab):
        if j >= len(raw):
            continue
        if label in target_idx:
            out[target_idx[label]] += float(raw[j])
        elif other_i is not None:
            out[other_i] += float(raw[j])
    s = float(out.sum())
    if s > 0:
        out /= s
    return out


def build_feature_matrix(
    case_ids: list[str],
    source_outputs: dict[str, dict[str, dict[str, np.ndarray]]],
    source_vocabs: dict[str, dict[str, list[str]]],
    target_vocab: list[str],
    head_vocabs: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    feature_names: list[str] = []
    first = True
    for cid in case_ids:
        chunks: list[np.ndarray] = []
        names: list[str] = []
        for src_name in source_outputs:
            src = source_outputs[src_name]
            voc = source_vocabs[src_name]
            present = cid in src
            if present:
                dx = align_probs(src[cid]["primary_dx"], voc["primary_dx"], target_vocab)
            else:
                dx = np.zeros(len(target_vocab), np.float32)
            chunks.append(dx)
            names += [f"{src_name}:dx:{v}" for v in target_vocab]
            tp = top_probs(dx, 5)
            meta = np.array(
                [
                    1.0 if present else 0.0,
                    tp[0],
                    tp[1],
                    tp[2],
                    tp[0] - tp[1],
                    tp[0] - tp[2],
                    entropy_norm(dx) if present else 1.0,
                ],
                np.float32,
            )
            chunks.append(meta)
            names += [
                f"{src_name}:present",
                f"{src_name}:top1",
                f"{src_name}:top2",
                f"{src_name}:top3",
                f"{src_name}:margin12",
                f"{src_name}:margin13",
                f"{src_name}:entropy",
            ]
            for head, hv in head_vocabs.items():
                if present and head in src[cid] and head in voc:
                    hp = align_probs(src[cid][head], voc[head], hv)
                else:
                    hp = np.zeros(len(hv), np.float32)
                chunks.append(hp)
                names += [f"{src_name}:{head}:{v}" for v in hv]
        rows.append(np.concatenate(chunks).astype(np.float32))
        if first:
            feature_names = names
            first = False
    return np.stack(rows).astype(np.float32), feature_names


def y_for_cases(
    case_ids: list[str], labels: dict[str, dict[str, str]], target_vocab: list[str]
) -> tuple[np.ndarray, list[str]]:
    y = np.array([label_to_index(target_vocab, labels[cid]["primary_dx"]) for cid in case_ids], np.int64)
    exact = [labels[cid]["primary_dx"] for cid in case_ids]
    return y, exact


def evaluate_probs(
    probs: np.ndarray,
    y: np.ndarray,
    exact_labels: list[str],
    target_vocab: list[str],
    *,
    name: str,
) -> dict[str, Any]:
    pred = probs.argmax(1)
    valid = y >= 0
    pred_labels = [target_vocab[int(i)] for i in pred]
    exact_ok = [
        (pl == gt and not special_label(pl))
        for pl, gt in zip(pred_labels, exact_labels)
    ]
    metrics = {
        "name": name,
        "n": int(valid.sum()),
        "mapped_acc": float((pred[valid] == y[valid]).mean()) if valid.any() else 0.0,
        "exact_acc": float(np.mean(exact_ok)) if exact_ok else 0.0,
        "macro_f1": macro_f1(y[valid], pred[valid], len(target_vocab)) if valid.any() else 0.0,
    }
    for k in (3, 5, 10):
        kk = min(k, probs.shape[1])
        topk = np.argsort(-probs, axis=1)[:, :kk]
        metrics[f"top{k}_mapped"] = float(np.mean([int(y[i]) in topk[i] for i in range(len(y))]))
        metrics[f"top{k}_exact"] = float(
            np.mean(
                [
                    exact_labels[i] in {target_vocab[int(j)] for j in topk[i]}
                    for i in range(len(exact_labels))
                ]
            )
        )
    return metrics


def train_one_seed(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    exact_va: list[str],
    target_vocab: list[str],
    args: argparse.Namespace,
    seed: int,
    torch: Any,
    nn: Any,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    mean = Xtr.mean(0).astype(np.float32)
    std = (Xtr.std(0) + 1e-6).astype(np.float32)
    Xtr_n = (Xtr - mean) / std
    Xva_n = (Xva - mean) / std

    class Stacker(nn.Module):
        def __init__(self, d: int, h: int, c: int, dp: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dp),
                nn.Linear(h, h // 2),
                nn.GELU(),
                nn.Dropout(dp),
                nn.Linear(h // 2, c),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    model = Stacker(Xtr.shape[1], int(args.hidden), len(target_vocab), float(args.dropout)).to(device)
    counts = np.bincount(ytr[ytr >= 0], minlength=len(target_vocab)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (len(counts) * counts)) ** float(args.class_weight_power)
    weights = weights / max(float(weights.mean()), 1e-6)
    ce = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights.astype(np.float32)).to(device),
        label_smoothing=float(args.label_smoothing),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    xtr = torch.from_numpy(Xtr_n).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    xva = torch.from_numpy(Xva_n).to(device)
    best: dict[str, Any] = {"seed": seed, "epoch": -1, "exact_acc": -1.0, "mapped_acc": -1.0}
    best_state: dict[str, Any] | None = None
    best_probs = np.zeros((Xva.shape[0], len(target_vocab)), np.float32)
    n = xtr.shape[0]
    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            logits = model(xtr[idx])
            loss = ce(logits, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0 or epoch == int(args.epochs) - 1:
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(xva), dim=1).detach().cpu().numpy()
            m = evaluate_probs(probs, yva, exact_va, target_vocab, name=f"stacker_seed{seed}")
            score = (m["exact_acc"], m["mapped_acc"])
            best_score = (best["exact_acc"], best["mapped_acc"])
            if score > best_score:
                best = {"seed": seed, "epoch": epoch + 1, **m}
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_probs = probs.astype(np.float32)
    assert best_state is not None
    ckpt = {
        "model_state": best_state,
        "input_dim": int(Xtr.shape[1]),
        "hidden": int(args.hidden),
        "dropout": float(args.dropout),
        "target_vocab": target_vocab,
        "scaler_mean": mean,
        "scaler_std": std,
        "seed": seed,
        "epoch": int(best["epoch"]),
    }
    return best, ckpt, best_probs


def source_metric_rows(
    case_ids: list[str],
    source_outputs: dict[str, dict[str, dict[str, np.ndarray]]],
    source_vocabs: dict[str, dict[str, list[str]]],
    target_vocab: list[str],
    y: np.ndarray,
    exact: list[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    aligned: dict[str, np.ndarray] = {}
    for name, outputs in source_outputs.items():
        probs = []
        voc = source_vocabs[name]["primary_dx"]
        for cid in case_ids:
            if cid in outputs:
                probs.append(align_probs(outputs[cid]["primary_dx"], voc, target_vocab))
            else:
                probs.append(np.zeros(len(target_vocab), np.float32))
        arr = np.stack(probs).astype(np.float32)
        aligned[name] = arr
        rows.append(evaluate_probs(arr, y, exact, target_vocab, name=name))
    if aligned:
        avg = np.mean(np.stack(list(aligned.values())), axis=0).astype(np.float32)
        avg /= np.maximum(avg.sum(1, keepdims=True), 1e-8)
        rows.append(evaluate_probs(avg, y, exact, target_vocab, name="source_equal_avg"))
        aligned["source_equal_avg"] = avg
        maxprob = []
        for i in range(len(case_ids)):
            best = max((arr[i] for arr in aligned.values()), key=lambda p: float(np.max(p)))
            maxprob.append(best)
        mp = np.stack(maxprob).astype(np.float32)
        rows.append(evaluate_probs(mp, y, exact, target_vocab, name="source_max_conf"))
        aligned["source_max_conf"] = mp
    return rows, aligned


def build_e2e_predictions(
    case_ids: list[str],
    steps_by_id: dict[str, list[dict[str, str]]],
    source_outputs: dict[str, dict[str, dict[str, np.ndarray]]],
    source_vocabs: dict[str, dict[str, list[str]]],
    context_source: str,
    target_vocab: list[str],
    stacker_probs: np.ndarray,
    out_dir: Path,
    assets: Path,
) -> None:
    calib = TextCalibrator.from_assets(assets)
    ctx_out = source_outputs[context_source]
    ctx_voc = source_vocabs[context_source]

    def head_label(cid: str, head: str) -> str:
        if cid not in ctx_out or head not in ctx_out[cid]:
            return ""
        vocab = ctx_voc[head]
        idx = int(np.asarray(ctx_out[cid][head]).argmax())
        val = vocab[idx] if 0 <= idx < len(vocab) else ""
        return "" if special_label(val) else val

    gt_cases: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for i, cid in enumerate(case_ids):
        dx = target_vocab[int(stacker_probs[i].argmax())]
        if special_label(dx):
            dx = ""
        ctx = LabelContext(
            organ=head_label(cid, "organ"),
            procedure=head_label(cid, "procedure"),
            diagnoses=[dx] if dx else [],
            histologic_type=head_label(cid, "histologic_type"),
            grade=head_label(cid, "grade"),
            behavior=head_label(cid, "behavior"),
        )
        gt_cases.append({"id": cid, "chain-of-thought": steps_by_id[cid]})
        pred_steps = calib.render_from_labels(ctx, True)
        preds.append({"id": cid, "chain-of-thought": pred_steps})
        pred_rows.append(
            {
                "id": cid,
                "primary_dx": dx,
                "top5": [
                    [target_vocab[int(j)], float(stacker_probs[i, int(j)])]
                    for j in np.argsort(-stacker_probs[i])[:5]
                ],
                "context": ctx.__dict__,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_val.json").write_text(json.dumps(gt_cases, ensure_ascii=False), encoding="utf-8")
    (out_dir / "pred_stacker_val.json").write_text(
        json.dumps(preds, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "pred_stacker_rows.json").write_text(
        json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn as nn

    args.out.mkdir(parents=True, exist_ok=True)
    args.run_out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    split = json.loads((args.assets / "split.json").read_text(encoding="utf-8"))
    labels, steps_by_id = load_labels(args.cot)
    train_ids = [cid for cid in split["train_case_ids"] if cid in labels]
    val_ids = [cid for cid in split["val_case_ids"] if cid in labels]
    if args.limit:
        train_ids = train_ids[: args.limit]
        val_ids = val_ids[: max(1, args.limit // 5)]
    all_ids = train_ids + val_ids
    target_vocab = make_target_vocab(train_ids, labels)
    head_vocabs = make_head_vocabs(train_ids, labels)
    print(
        f"[stacker] train={len(train_ids)} val={len(val_ids)} "
        f"target_dx={len(target_vocab)} device={device}",
        flush=True,
    )

    sources = []
    for spec in DEFAULT_SOURCES:
        if not spec.model_dir.is_dir() or not spec.feat_dir.is_dir():
            print(f"[stacker] skip missing source {spec.name}", flush=True)
            continue
        print(f"[stacker] loading {spec.name} ({spec.kind})", flush=True)
        sources.append(build_source(spec, torch, nn, device))
    if not sources:
        raise RuntimeError("no usable sources")

    source_outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    source_vocabs: dict[str, dict[str, list[str]]] = {}
    for source in sources:
        print(f"[stacker] predicting {source.spec.name}", flush=True)
        source_outputs[source.spec.name] = source.predict_many(all_ids, args.batch_size)
        source_vocabs[source.spec.name] = source.vocabs
        print(
            f"[stacker] {source.spec.name}: {len(source_outputs[source.spec.name])}/{len(all_ids)} cases",
            flush=True,
        )
    if args.context_source not in source_outputs:
        args.context_source = next(iter(source_outputs))

    X, feature_names = build_feature_matrix(
        all_ids, source_outputs, source_vocabs, target_vocab, head_vocabs
    )
    Xtr = X[: len(train_ids)]
    Xva = X[len(train_ids) :]
    ytr, _ = y_for_cases(train_ids, labels, target_vocab)
    yva, exact_va = y_for_cases(val_ids, labels, target_vocab)
    print(f"[stacker] feature_dim={X.shape[1]}", flush=True)

    source_rows, aligned_val = source_metric_rows(
        val_ids, source_outputs, source_vocabs, target_vocab, yva, exact_va
    )
    for row in source_rows:
        print(
            f"[stacker] {row['name']:22s} exact={row['exact_acc']:.4f} "
            f"mapped={row['mapped_acc']:.4f} top5={row['top5_exact']:.4f}",
            flush=True,
        )

    seed_results = []
    ckpts: list[dict[str, Any]] = []
    probs_by_seed = []
    for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
        best, ckpt, probs = train_one_seed(
            Xtr, ytr, Xva, yva, exact_va, target_vocab, args, seed, torch, nn, device
        )
        seed_results.append(best)
        ckpts.append(ckpt)
        probs_by_seed.append(probs)
        print(
            f"[stacker] seed={seed} epoch={best['epoch']} exact={best['exact_acc']:.4f} "
            f"mapped={best['mapped_acc']:.4f} top5={best['top5_exact']:.4f}",
            flush=True,
        )

    avg_probs = np.mean(np.stack(probs_by_seed), axis=0).astype(np.float32)
    avg_probs /= np.maximum(avg_probs.sum(1, keepdims=True), 1e-8)
    ensemble_metrics = evaluate_probs(avg_probs, yva, exact_va, target_vocab, name="stacker_seed_avg")
    best_idx = max(range(len(seed_results)), key=lambda i: (seed_results[i]["exact_acc"], seed_results[i]["mapped_acc"]))
    best_metrics = seed_results[best_idx]
    best_probs = probs_by_seed[best_idx]
    print(
        f"[stacker] seed_avg exact={ensemble_metrics['exact_acc']:.4f} "
        f"mapped={ensemble_metrics['mapped_acc']:.4f} top5={ensemble_metrics['top5_exact']:.4f}",
        flush=True,
    )

    # Save the seed average as the preferred offline prediction; save the best
    # single seed checkpoint for deployment simplicity.
    torch.save(ckpts[best_idx], args.out / "primary_dx_stacker.pt")
    metadata = {
        "target_vocab": target_vocab,
        "head_vocabs": head_vocabs,
        "feature_names": feature_names,
        "sources": [
            {
                "name": s.spec.name,
                "kind": s.spec.kind,
                "model_dir": str(s.spec.model_dir),
                "feat_dir": str(s.spec.feat_dir),
            }
            for s in sources
        ],
        "context_source": args.context_source,
        "best_single_seed": int(seed_results[best_idx]["seed"]),
        "best_single_epoch": int(seed_results[best_idx]["epoch"]),
        "seed_results": seed_results,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_rows = []
    for i, cid in enumerate(val_ids):
        pred_i = int(avg_probs[i].argmax())
        pred_rows.append(
            {
                "id": cid,
                "gt": exact_va[i],
                "pred": target_vocab[pred_i],
                "ok": bool(target_vocab[pred_i] == exact_va[i]),
                "top5": [
                    [target_vocab[int(j)], float(avg_probs[i, int(j)])]
                    for j in np.argsort(-avg_probs[i])[:5]
                ],
            }
        )
    (args.out / "val_predictions.json").write_text(
        json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "target_vocab_size": len(target_vocab),
        "source_metrics": source_rows,
        "best_single_seed_metrics": best_metrics,
        "stacker_seed_avg_metrics": ensemble_metrics,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_e2e:
        build_e2e_predictions(
            val_ids,
            steps_by_id,
            source_outputs,
            source_vocabs,
            args.context_source,
            target_vocab,
            avg_probs,
            args.run_out,
            args.assets,
        )
        print(f"[stacker] wrote e2e val files -> {args.run_out}", flush=True)
    print(f"[stacker] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
