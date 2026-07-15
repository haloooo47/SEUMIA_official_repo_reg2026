#!/usr/bin/env python3
"""Train small hard-confusion heads for report rescue.

These heads target the report-failure buckets that a generic candidate selector
cannot infer from text features alone: tumor/no-tumor, benign-vs-DCIS-vs-invasive
breast, adenoma-vs-hyperplastic-vs-carcinoma, gastritis/MALT/carcinoma, etc.

The script writes validation predictions and metrics together with the trained
heads used by the report selector.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reg2_text_calibrator import canonicalize_question  # noqa: E402

DEF_FEATURES = Path("features/report_fused_titan_h1_v2_256")
DEF_COT = Path("data/train_CoT_v01.json")
DEF_SPLIT = Path("runs/calibrator_assets/v1/split.json")
DEF_VALREF = Path("runs/e_multifm/gt_val.json")
DEF_OUT = Path("runs/models/report_hard_confusion_heads_v1")
NONE = "<none>"
OTHER = "<other>"


def clean(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def norm_id(raw: Any) -> str:
    cid = clean(raw)
    for suf in (".tiff", ".svs"):
        if cid.lower().endswith(suf):
            return cid[: -len(suf)]
    return cid


def get_steps(case: dict[str, Any]) -> list[dict[str, Any]]:
    raw = case.get("chain-of-thought") or case.get("chain_of_thought") or []
    return raw if isinstance(raw, list) else []


def answer(case: dict[str, Any], question_pat: str) -> str:
    pat = re.compile(question_pat, re.I)
    for step in get_steps(case):
        if pat.search(str(step.get("question", ""))):
            return clean(step.get("answer"))
    return ""


def case_fields(case: dict[str, Any]) -> dict[str, str]:
    organ = answer(case, r"^What is the organ\??$") or clean(case.get("organ"))
    organ_map = {
        "breast": "Breast",
        "prostate": "Prostate",
        "stomach": "Stomach",
        "colon": "Colon",
        "rectum": "Rectum",
        "lung": "Lung",
        "uterine cervix": "Uterine cervix",
        "urinary bladder": "Urinary bladder",
    }
    organ = organ_map.get(organ.lower(), organ)
    return {
        "id": norm_id(case.get("id", "")),
        "organ": organ,
        "procedure": answer(case, r"^What is the procedure\??$"),
        "dx1": answer(case, r"^What is the #1 diagnosis\??$"),
        "num_dx": answer(case, r"number of diagnoses"),
        "grade": answer(case, r"grade of neoplasm"),
        "microcalcification": answer(case, r"microcalcification present"),
    }


def ltxt(x: str) -> str:
    return clean(x).lower()


def prostate_tumor(f: dict[str, str]) -> str:
    if f["organ"] != "Prostate":
        return NONE
    dx = f["dx1"]
    if dx == "No tumor present":
        return "no_tumor"
    if dx == "Acinar adenocarcinoma":
        return "acinar_adenocarcinoma"
    return OTHER


def breast_family(f: dict[str, str]) -> str:
    if f["organ"] != "Breast":
        return NONE
    dx = ltxt(f["dx1"])
    if "ductal carcinoma in situ" in dx:
        return "dcis"
    if dx.startswith("invasive carcinoma of no special type"):
        return "invasive_nst"
    if "invasive lobular carcinoma" in dx:
        return "invasive_lobular"
    if "papillary" in dx:
        return "papillary"
    if "atypical ductal hyperplasia" in dx:
        return "adh"
    if "fibroepithelial" in dx or "fibroadenoma" in dx:
        return "fibroepithelial_fibroadenoma"
    if "no evidence of tumor" in dx:
        return "no_tumor"
    benign_tokens = (
        "fibrocystic",
        "usual ductal hyperplasia",
        "duct ectasia",
        "apocrine",
        "sclerosing adenosis",
        "columnar cell",
        "fibroadenomatoid",
        "adenosis",
    )
    if any(t in dx for t in benign_tokens):
        return "benign_proliferative"
    if "carcinoma" in dx or "sarcoma" in dx or "phyllodes" in dx:
        return "other_malignant"
    return OTHER


def colorectal_family(f: dict[str, str]) -> str:
    if f["organ"] not in {"Colon", "Rectum"}:
        return NONE
    dx = ltxt(f["dx1"])
    if "hyperplastic polyp" in dx:
        return "hyperplastic"
    if any(t in dx for t in ("tubular adenoma", "tubulovillous", "sessile serrated", "traditional serrated", "adenoma")):
        return "adenoma_serrated"
    if any(t in dx for t in ("adenocarcinoma", "signet-ring", "mucinous adenocarcinoma")):
        return "carcinoma"
    if "inflammation" in dx or "colitis" in dx or "inflammatory polyp" in dx:
        return "inflammation"
    if "lymphoma" in dx or "malt" in dx:
        return "lymphoma_malt"
    if any(t in dx for t in ("neuroendocrine", "stromal tumor", "squamous cell", "small cell")):
        return "other_neoplasm"
    return OTHER


def lung_family(f: dict[str, str]) -> str:
    if f["organ"] != "Lung":
        return NONE
    dx = ltxt(f["dx1"])
    if "no evidence of malignancy or granuloma" in dx:
        return "no_malignancy"
    if "granulomatous inflammation" in dx:
        return "granulomatous"
    if "fungal" in dx or "cryptococcus" in dx or "aspergillus" in dx:
        return "fungal"
    if "non-small cell" in dx:
        return "non_small_cell"
    if "small cell" in dx:
        return "small_cell"
    if "squamous cell carcinoma" in dx:
        return "squamous"
    if "adenocarcinoma" in dx:
        return "adenocarcinoma"
    if "neuroendocrine" in dx or "carcinoid" in dx:
        return "neuroendocrine"
    return OTHER


def stomach_family(f: dict[str, str]) -> str:
    if f["organ"] != "Stomach":
        return NONE
    dx = ltxt(f["dx1"])
    if "gastritis" in dx:
        return "gastritis"
    if any(t in dx for t in ("tubular adenoma", "foveolar-type adenoma", "fundic gland polyp", "adenoma")):
        return "adenoma_polyp"
    if "malt" in dx or "lymphoma" in dx:
        return "lymphoma_malt"
    if any(t in dx for t in ("adenocarcinoma", "poorly cohesive", "signet-ring", "mucinous adenocarcinoma")):
        return "carcinoma"
    if any(t in dx for t in ("stromal tumor", "neuroendocrine", "small cell", "squamous cell", "melanoma")):
        return "other_neoplasm"
    return OTHER


def cervix_family(f: dict[str, str]) -> str:
    if f["organ"] != "Uterine cervix":
        return NONE
    dx = ltxt(f["dx1"])
    if "lsil" in dx or "cin 1" in dx:
        return "lsil_cin1"
    if "hsil" in dx and "cin 2" in dx:
        return "hsil_cin2"
    if "hsil" in dx and "cin 3" in dx:
        return "hsil_cin3"
    if "invasive squamous cell carcinoma" in dx:
        return "invasive_scc"
    if any(t in dx for t in ("cervicitis", "polyp", "metaplasia", "microglandular")):
        return "benign_inflammatory"
    if "adenocarcinoma" in dx or "adenocarcinoma in situ" in dx or "ais" in dx:
        return "glandular_neoplasm"
    return OTHER


def bladder_family(f: dict[str, str]) -> str:
    if f["organ"] != "Urinary bladder":
        return NONE
    dx = ltxt(f["dx1"])
    if "no tumor present" in dx:
        return "no_tumor"
    if "urothelial carcinoma in situ" in dx:
        return "cis"
    if "muscle proper" in dx:
        return "muscle_invasive"
    if "subepithelial connective tissue" in dx:
        return "subepithelial_invasive"
    if "non-invasive papillary" in dx and "high grade" in dx:
        return "noninvasive_high_grade"
    if "non-invasive papillary" in dx and "low grade" in dx:
        return "noninvasive_low_grade"
    if "squamous differentiation" in dx or "glandular differentiation" in dx:
        return "variant_differentiation"
    return OTHER


def procedure_colorectal(f: dict[str, str]) -> str:
    if f["organ"] not in {"Colon", "Rectum"}:
        return NONE
    p = ltxt(f["procedure"])
    if "polypectomy" in p:
        return "polypectomy"
    if "mucosal resection" in p:
        return "emr"
    if "submucosal dissection" in p:
        return "esd"
    if "biopsy" in p:
        return "biopsy"
    return OTHER


def organ_colorectal(f: dict[str, str]) -> str:
    """Colon-vs-rectum header selector for dx-correct report-local fixes."""
    if f["organ"] == "Colon":
        return "colon"
    if f["organ"] == "Rectum":
        return "rectum"
    return NONE


def procedure_breast(f: dict[str, str]) -> str:
    if f["organ"] != "Breast":
        return NONE
    p = ltxt(f["procedure"])
    if "core needle" in p:
        return "core_needle"
    if "mammotome" in p:
        return "mammotome"
    if "stereotactic" in p:
        return "stereotactic"
    return OTHER


def procedure_cervix(f: dict[str, str]) -> str:
    if f["organ"] != "Uterine cervix":
        return NONE
    p = ltxt(f["procedure"])
    if "colposcopic" in p:
        return "colposcopic"
    if "punch" in p:
        return "punch"
    if "loop" in p:
        return "leep"
    return OTHER


def breast_microcalcification(f: dict[str, str]) -> str:
    if f["organ"] != "Breast":
        return NONE
    ans = ltxt(f.get("microcalcification", ""))
    if "yes" in ans and "microcalcification" in ans:
        return "present"
    if "no" in ans and "microcalcification" in ans:
        return "absent"
    return OTHER


HEAD_FNS: dict[str, Callable[[dict[str, str]], str]] = {
    "prostate_tumor": prostate_tumor,
    "breast_family": breast_family,
    "colorectal_family": colorectal_family,
    "lung_family": lung_family,
    "stomach_family": stomach_family,
    "cervix_family": cervix_family,
    "bladder_family": bladder_family,
    "organ_colorectal": organ_colorectal,
    "procedure_colorectal": procedure_colorectal,
    "procedure_breast": procedure_breast,
    "procedure_cervix": procedure_cervix,
    "breast_microcalcification": breast_microcalcification,
}


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


def load_valref_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        out = set()
        for x in raw:
            if isinstance(x, dict):
                out.add(norm_id(x.get("id", "")))
            else:
                out.add(norm_id(x))
        return {x for x in out if x}
    return set()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=DEF_FEATURES)
    p.add_argument("--feat-key", default="fused")
    p.add_argument("--extra-features", type=Path, default=None,
                   help="Optional second feature dir to concatenate (e.g. virchow2_full_256 for decorrelation).")
    p.add_argument("--extra-key", default="pooled_mean")
    p.add_argument("--cot", type=Path, default=DEF_COT)
    p.add_argument("--split", type=Path, default=DEF_SPLIT)
    p.add_argument("--valref", type=Path, default=DEF_VALREF)
    p.add_argument("--out", type=Path, default=DEF_OUT)
    p.add_argument("--heads", nargs="+", default=list(HEAD_FNS), choices=list(HEAD_FNS))
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    import torch
    import torch.nn as nn

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids = {norm_id(x) for x in split["train_case_ids"]}
    val_ids = {norm_id(x) for x in split["val_case_ids"]}
    valref_ids = load_valref_ids(args.valref)

    fields: dict[str, dict[str, str]] = {}
    labels: dict[str, dict[str, str]] = {}
    for case in json.loads(args.cot.read_text(encoding="utf-8")):
        f = case_fields(case)
        cid = f["id"]
        if not cid:
            continue
        fields[cid] = f
        labels[cid] = {head: HEAD_FNS[head](f) for head in args.heads}

    feats: dict[str, np.ndarray] = {}
    for npz in sorted(args.features.glob("*.npz")):
        cid = norm_id(npz.stem)
        if cid not in labels:
            continue
        data = np.load(npz)
        if args.feat_key not in data.files:
            continue
        feats[cid] = np.asarray(data[args.feat_key], dtype=np.float32).ravel()

    if args.extra_features is not None:
        extra: dict[str, np.ndarray] = {}
        extra_dim = 0
        for npz in sorted(args.extra_features.glob("*.npz")):
            cid = norm_id(npz.stem)
            if cid not in feats:
                continue
            data = np.load(npz)
            if args.extra_key not in data.files:
                continue
            v = np.asarray(data[args.extra_key], dtype=np.float32).ravel()
            extra[cid] = v
            extra_dim = v.shape[0]
        kept = 0
        for cid in list(feats):
            ev = extra.get(cid)
            if ev is None:
                ev = np.zeros(extra_dim, dtype=np.float32)
            else:
                kept += 1
            feats[cid] = np.concatenate([feats[cid], ev])
        print(f"[hard] extra-features={args.extra_features} key={args.extra_key} dim={extra_dim} "
              f"matched={kept}/{len(feats)}", flush=True)

    train = [cid for cid in feats if cid in train_ids]
    val = [cid for cid in feats if cid in val_ids]
    if not train or not val:
        raise SystemExit("[hard] missing train/val features")

    vocabs: dict[str, list[str]] = {}
    for head in args.heads:
        cnt = Counter(labels[cid][head] for cid in train if labels[cid][head] != NONE)
        vals = [v for v, n in cnt.most_common() if n >= args.min_count and v != OTHER]
        if cnt.get(OTHER, 0) >= args.min_count:
            vals.append(OTHER)
        vocabs[head] = vals or [OTHER]
    idx = {h: {v: i for i, v in enumerate(vs)} for h, vs in vocabs.items()}

    def label_idx(head: str, value: str) -> int:
        if value == NONE:
            return -1
        return idx[head].get(value, idx[head].get(OTHER, -1))

    Xtr = np.stack([feats[cid] for cid in train]).astype(np.float32)
    Xva = np.stack([feats[cid] for cid in val]).astype(np.float32)
    Ytr = {h: np.array([label_idx(h, labels[cid][h]) for cid in train], np.int64) for h in args.heads}
    Yva = {h: np.array([label_idx(h, labels[cid][h]) for cid in val], np.int64) for h in args.heads}

    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    class MultiHeadMLP(nn.Module):
        def __init__(self, d: int, h: int, head_sizes: dict[str, int], nl: int, dp: float):
            super().__init__()
            layers: list[nn.Module] = []
            for i in range(max(1, nl)):
                layers += [nn.Linear(d if i == 0 else h, h), nn.GELU(), nn.Dropout(dp)]
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in head_sizes.items()})

        def forward(self, x):
            z = self.trunk(x)
            return {k: head(z) for k, head in self.heads.items()}

    head_sizes = {h: len(vocabs[h]) for h in args.heads}
    model = MultiHeadMLP(Xtr.shape[1], args.hidden, head_sizes, args.num_layers, args.dropout).to(dev)
    losses: dict[str, nn.Module] = {}
    for head in args.heads:
        y = Ytr[head]
        valid = y >= 0
        weight = None
        if valid.any() and args.class_weight_power > 0:
            counts = np.bincount(y[valid], minlength=head_sizes[head]).astype(np.float32)
            counts = np.maximum(counts, 1.0)
            w = (counts.sum() / (len(counts) * counts)) ** args.class_weight_power
            w = w / w.mean()
            weight = torch.from_numpy(w.astype(np.float32)).to(dev)
        losses[head] = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

    print(
        f"[hard] train={len(train)} val={len(val)} heads={args.heads} "
        f"features={args.features} gpu={args.gpu}",
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xtr = torch.from_numpy(Xtr).to(dev)
    ytr = {h: torch.from_numpy(Ytr[h]).to(dev) for h in args.heads}
    n = len(train)
    for ep in range(args.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        last = 0.0
        for i in range(0, n, args.batch_size):
            bi = perm[i : i + args.batch_size]
            out = model(xtr[bi])
            parts = []
            for head in args.heads:
                if torch.any(ytr[head][bi] >= 0):
                    parts.append(losses[head](out[head], ytr[head][bi]))
            if not parts:
                continue
            loss = sum(parts) / len(parts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss)
        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            print(f"[hard] ep{ep+1}/{args.epochs} loss={last:.4f}", flush=True)

    model.eval()
    xva = torch.from_numpy(Xva).to(dev)
    val_preds: dict[str, dict[str, Any]] = {cid: {} for cid in val}
    valref_mask = np.array([cid in valref_ids for cid in val], dtype=bool) if valref_ids else None
    metrics: dict[str, Any] = {
        "num_train": len(train),
        "num_val": len(val),
        "num_valref": int(valref_mask.sum()) if valref_mask is not None else 0,
        "features": str(args.features),
        "feat_key": args.feat_key,
        "heads": args.heads,
        "gpu": args.gpu,
    }

    with torch.no_grad():
        out = model(xva)
        for head in args.heads:
            probs_t = torch.softmax(out[head], dim=1)
            pred = probs_t.argmax(1).cpu().numpy()
            conf = probs_t.max(1).values.cpu().numpy()
            top2 = torch.topk(probs_t, k=min(2, probs_t.shape[1]), dim=1).values.cpu().numpy()
            margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
            gt = Yva[head]
            mask = gt >= 0
            metrics[f"{head}_classes"] = int(len(vocabs[head]))
            metrics[f"{head}_n_val"] = int(mask.sum())
            metrics[f"{head}_class_counts_train"] = {
                vocabs[head][i]: int(n)
                for i, n in enumerate(np.bincount(Ytr[head][Ytr[head] >= 0], minlength=len(vocabs[head])).tolist())
            }
            if mask.any():
                metrics[f"{head}_acc"] = float((pred[mask] == gt[mask]).mean())
                metrics[f"{head}_macro_f1"] = macro_f1(gt[mask], pred[mask], len(vocabs[head]))
                metrics[f"{head}_majority_acc"] = float(Counter(gt[mask].tolist()).most_common(1)[0][1] / int(mask.sum()))
                metrics[f"{head}_beats_majority"] = bool(metrics[f"{head}_acc"] > metrics[f"{head}_majority_acc"] + 1e-9)
                if valref_mask is not None:
                    mref = mask & valref_mask
                    metrics[f"{head}_n_valref"] = int(mref.sum())
                    if mref.any():
                        metrics[f"{head}_acc_valref"] = float((pred[mref] == gt[mref]).mean())
                        metrics[f"{head}_macro_f1_valref"] = macro_f1(gt[mref], pred[mref], len(vocabs[head]))
                        metrics[f"{head}_majority_acc_valref"] = float(
                            Counter(gt[mref].tolist()).most_common(1)[0][1] / int(mref.sum())
                        )
                        metrics[f"{head}_beats_majority_valref"] = bool(
                            metrics[f"{head}_acc_valref"] > metrics[f"{head}_majority_acc_valref"] + 1e-9
                        )
            for vi, cid in enumerate(val):
                pi = int(pred[vi])
                gi = int(gt[vi])
                val_preds[cid][head] = {
                    "pred": vocabs[head][pi],
                    "prob": float(conf[vi]),
                    "margin": float(margin[vi]),
                    "gt": vocabs[head][gi] if gi >= 0 else "",
                    "organ": fields[cid]["organ"],
                    "dx1": fields[cid]["dx1"],
                    "procedure": fields[cid]["procedure"],
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
            "feat_key": args.feat_key,
            "heads": args.heads,
            "vocabs": vocabs,
        },
        args.out / "hard_confusion_heads.pt",
    )
    (args.out / "head_vocab.json").write_text(
        json.dumps(
            {
                "heads": args.heads,
                "vocabs": vocabs,
                "features": str(args.features),
                "feat_key": args.feat_key,
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
    (args.out / "val_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "val_predictions.json").write_text(json.dumps(val_preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[hard] saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
