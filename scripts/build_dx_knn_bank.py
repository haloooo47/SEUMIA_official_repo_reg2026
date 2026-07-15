"""Build the diagnosis kNN bank asset (dx_knn_bank.npz) for a trained model.

Stores, for every TRAIN-split case that has features, the model input embedding
plus its primary_dx index (in the model's vocab) and organ name. The online
LabelPredictor loads this next to multitask_heads.pt to ensemble MLP+kNN and to
organ-mask the diagnosis. Train-split only -> no leakage into val.
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
import eval_reg2026_text_calibrator as ec  # noqa: E402

DEF_ASSETS = Path("runs/calibrator_assets/v1")
DEF_COT = Path("data/train_CoT_v01.json")
OTHER = "<other>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=DEF_ASSETS)
    p.add_argument("--cot", type=Path, default=DEF_COT)
    return p.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    ckpt = torch.load(args.model / "multitask_heads.pt", map_location="cpu", weights_only=False)
    vocab = json.loads((args.model / "label_vocab.json").read_text())["vocabs"]
    input_keys = ckpt["input_keys"]
    dx_vocab = vocab["primary_dx"]
    dx_idx = {d: i for i, d in enumerate(dx_vocab)}

    split = json.loads((args.assets / "split.json").read_text())
    train_ids = set(split["train_case_ids"])
    cases = json.loads(args.cot.read_text())

    embs, dxs, organs, dx_labels = [], [], [], []
    skipped_unknown = 0
    for c in cases:
        cid = ec.normalize_case_id(c.get("id", ""))
        if cid not in train_ids:
            continue
        steps = ec.get_steps(c)
        if not steps:
            continue
        npz = args.features / f"{cid}.npz"
        if not npz.is_file():
            continue
        ctx = ec.context_from_steps(steps, organ_hint=ec.clean(c.get("organ")))
        dx = ctx.primary_dx()
        organ = ec.clean(ctx.organ)
        if not dx or not organ:
            continue
        d = np.load(npz)
        vec = np.concatenate([np.asarray(d[k], np.float32).ravel() for k in input_keys])
        mapped = dx_idx.get(dx)
        if mapped is None:
            skipped_unknown += 1
            continue
        embs.append(vec)
        dxs.append(mapped)
        organs.append(organ)
        dx_labels.append(dx)

    embs = np.stack(embs).astype(np.float32)
    out = args.model / "dx_knn_bank.npz"
    np.savez_compressed(out, embeddings=embs,
                        dx_idx=np.asarray(dxs, np.int64),
                        organ=np.asarray(organs, dtype=object),
                        dx_label=np.asarray(dx_labels, dtype=object))
    print(
        f"[bank] wrote {len(embs)} train embeddings (dim {embs.shape[1]}) -> {out}; "
        f"skipped_unknown_dx={skipped_unknown}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
