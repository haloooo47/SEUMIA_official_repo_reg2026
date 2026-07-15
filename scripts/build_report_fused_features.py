#!/usr/bin/env python3
"""Build fused slide-level features for whole-report classification.

Each output npz contains:

  fused = concat(TITAN slide_embedding, H-optimus-1 pooled_mean,
                 Virchow2 pooled_mean)

The fused vector is intended for lightweight report heads / rerankers, not for
patch-level ABMIL training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TITAN = Path("features/titan_full_256")
HOPT1 = Path("features/hoptimus1_full_256")
VIRCHOW2 = Path("features/virchow2_full_256")
OUT = Path("features/report_fused_titan_h1_v2_256")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--titan", type=Path, default=TITAN)
    p.add_argument("--hopt1", type=Path, default=HOPT1)
    p.add_argument("--virchow2", type=Path, default=VIRCHOW2)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(
        {p.stem for p in args.titan.glob("*.npz")}
        & {p.stem for p in args.hopt1.glob("*.npz")}
        & {p.stem for p in args.virchow2.glob("*.npz")}
    )
    print(f"[report-fuse] intersection={len(stems)} out={args.out}", flush=True)
    done = skipped = failed = 0
    for i, stem in enumerate(stems, 1):
        out_path = args.out / f"{stem}.npz"
        if out_path.is_file():
            skipped += 1
            continue
        try:
            titan = np.load(args.titan / f"{stem}.npz")
            h1 = np.load(args.hopt1 / f"{stem}.npz")
            v2 = np.load(args.virchow2 / f"{stem}.npz")
            fused = np.concatenate(
                [
                    np.asarray(titan["slide_embedding"], np.float32).ravel(),
                    np.asarray(h1["pooled_mean"], np.float32).ravel(),
                    np.asarray(v2["pooled_mean"], np.float32).ravel(),
                ]
            ).astype(np.float32)
            tmp = out_path.with_name(f".{out_path.name}.tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, fused=fused)
            tmp.replace(out_path)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[report-fuse] FAIL {stem}: {type(exc).__name__}: {exc}", flush=True)
        if i % 1000 == 0:
            print(f"[report-fuse] {i}/{len(stems)} done={done} skip={skipped} fail={failed}", flush=True)
    print(f"[report-fuse] DONE done={done} skip={skipped} fail={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
