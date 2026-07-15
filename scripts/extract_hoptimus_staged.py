#!/usr/bin/env python3
"""RAM-staged H-optimus-0 extraction to use ALL 4 GPUs despite a single HDD.

The WSIs live on one spinning HDD (~260 MB/s sequential, but random tile reads
from 8 concurrent workers thrash the head and stall every GPU). The fix is to
make disk access purely sequential and feed the GPUs from RAM:

    producer (1 proc):  stream whole .tiff files HDD -> /dev/shm  (sequential)
    workers  (N procs): read tiles from the in-RAM copy, encode, write .npz,
                        then delete the staged copy to free RAM.

Coordination is a lock-free filesystem queue in the staging dir:
    <stem>.tiff           staged slide (in /dev/shm)
    <stem>.ready          marker: ready to claim
    <stem>.claim.<pid>    a worker atomically renamed .ready -> .claim (it won)
    _PRODUCER_DONE        sentinel: producer has queued everything

Roles share one file so the heavy extraction code (model + tiling) is reused
from extract_hoptimus_features.py.

Launch (1 producer + 2 workers per card across 4 cards):

    OUT=features/hoptimus0_v2
    PY=python
    $PY scripts/extract_hoptimus_staged.py --role producer --out $OUT &
    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$((i/2)) $PY scripts/extract_hoptimus_staged.py \
        --role worker --worker-id $i --out $OUT &
    done; wait
"""

from __future__ import annotations

import argparse
import errno
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_hoptimus_features as ehf  # noqa: E402  (reuse model + tiling code)

DEFAULT_MANIFEST = ehf.DEFAULT_MANIFEST
DEFAULT_DATASET = ehf.DEFAULT_DATASET
DEFAULT_OUT = ehf.DEFAULT_OUT
DEFAULT_HOPTIMUS = ehf.DEFAULT_HOPTIMUS
DEFAULT_STAGE = Path("/dev/shm/reg2_stage")
DONE_SENTINEL = "_PRODUCER_DONE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["producer", "worker"], required=True)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    p.add_argument("--hoptimus-dir", type=Path, default=DEFAULT_HOPTIMUS)
    p.add_argument("--max-tiles", type=int, default=128)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tissue-thresh", type=float, default=0.10)
    p.add_argument("--scan-target", type=int, default=768)
    p.add_argument("--max-buffered", type=int, default=400,
                   help="Producer pauses when this many staged slides await a worker.")
    p.add_argument("--cap-gb", type=float, default=60.0,
                   help="Producer pauses when the staging dir exceeds this many GB.")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def stem_of(case_id: str) -> str:
    return case_id[:-5] if case_id.lower().endswith(".tiff") else case_id


def stage_size_bytes(stage: Path) -> int:
    total = 0
    for pattern in ("*.tiff", "*.tiff.tmp"):
        for f in stage.glob(pattern):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def stage_buffered_count(stage: Path) -> int:
    return len(list(stage.glob("*.ready"))) + len(list(stage.glob("*.claim.*")))


def wait_for_stage_capacity(stage: Path, max_buffered: int, cap_bytes: int, next_bytes: int) -> None:
    """Block until staging has room for the next full slide copy.

    The visible staged-file size is not enough after workers unlink files that are
    still open, so also consult tmpfs free space. This prevents ENOSPC retry
    storms and the producer finishing with many silently unstaged cases.
    """
    next_bytes = max(0, int(next_bytes))
    safety = min(max(512 << 20, next_bytes // 4), 4 << 30)
    effective_cap = max(cap_bytes, next_bytes + safety)
    last_log = 0.0
    while True:
        buffered = stage_buffered_count(stage)
        visible = stage_size_bytes(stage)
        free = shutil.disk_usage(stage).free
        if buffered < max_buffered and visible + next_bytes <= effective_cap and free > next_bytes + safety:
            return
        now = time.time()
        if now - last_log > 30:
            print(
                "[producer] waiting capacity "
                f"buffered={buffered}/{max_buffered} "
                f"visible_gb={visible / (1 << 30):.1f} "
                f"free_gb={free / (1 << 30):.1f} "
                f"next_gb={next_bytes / (1 << 30):.1f}",
                flush=True,
            )
            last_log = now
        time.sleep(1.0)


def run_producer(args: argparse.Namespace) -> int:
    stage = args.stage_dir
    stage.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    cap_bytes = int(args.cap_gb * (1 << 30))
    missing_log = args.out / "missing_staged.txt"
    failed_log = args.out / "failed_stage_copy.txt"
    staged = missing = skipped = failed = 0
    t0 = time.time()
    for case in cases:
        cid = str(case.get("id", "")).strip()
        stem = stem_of(cid)
        if (args.out / f"{stem}.npz").is_file():
            skipped += 1
            continue
        src = ehf.resolve_slide_path(args.dataset_dir, cid)
        if src is None:
            missing += 1
            with missing_log.open("a", encoding="utf-8") as f:
                f.write(stem + "\n")
            continue
        try:
            src_size = src.stat().st_size
        except OSError:
            src_size = 0
        tmp = stage / f"{stem}.tiff.tmp"
        dst = stage / f"{stem}.tiff"
        attempts = 0
        while True:
            wait_for_stage_capacity(stage, args.max_buffered, cap_bytes, src_size)
            try:
                tmp.unlink(missing_ok=True)
                with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
                    shutil.copyfileobj(fi, fo, length=16 << 20)
                os.replace(tmp, dst)
                (stage / f"{stem}.ready").touch()
                staged += 1
                break
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                if exc.errno == errno.ENOSPC:
                    attempts += 1
                    wait_s = min(60, 5 + attempts * 5)
                    print(
                        f"[producer] ENOSPC copy {stem}; retry={attempts} wait={wait_s}s",
                        flush=True,
                    )
                    time.sleep(wait_s)
                    continue
                failed += 1
                with failed_log.open("a", encoding="utf-8") as f:
                    f.write(f"{stem}\t{type(exc).__name__}: {exc}\n")
                print(f"[producer] FAIL copy {stem}: {type(exc).__name__}: {exc}", flush=True)
                break
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                failed += 1
                with failed_log.open("a", encoding="utf-8") as f:
                    f.write(f"{stem}\t{type(exc).__name__}: {exc}\n")
                print(f"[producer] FAIL copy {stem}: {type(exc).__name__}: {exc}", flush=True)
                break
        if staged % 50 == 0:
            rate = staged / max(1e-6, time.time() - t0)
            print(f"[producer] staged={staged} skip={skipped} miss={missing} "
                  f"fail={failed} buffered={stage_buffered_count(stage)} ({rate:.2f}/s)", flush=True)

    (stage / DONE_SENTINEL).touch()
    print(f"[producer] DONE staged={staged} skipped={skipped} missing={missing} failed={failed}", flush=True)
    return 0


def claim_one(stage: Path, pid: int) -> str | None:
    """Atomically claim a ready slide; return its stem or None."""
    for ready in sorted(stage.glob("*.ready")):
        target = ready.with_suffix(f".claim.{pid}")
        try:
            os.rename(ready, target)  # atomic: only one worker wins
            return ready.name[: -len(".ready")]
        except OSError:
            continue
    return None


def run_worker(args: argparse.Namespace) -> int:
    import torch

    stage = args.stage_dir
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pid = os.getpid()
    print(f"[worker {args.worker_id}] device={device} pid={pid}", flush=True)

    mean = np.array(ehf.HOPTIMUS_MEAN, dtype=np.float32)
    std = np.array(ehf.HOPTIMUS_STD, dtype=np.float32)
    model = ehf.build_model(args.hoptimus_dir, device)
    import tiffslide

    done = failed = 0
    idle_polls = 0
    while True:
        stem = claim_one(stage, pid)
        if stem is None:
            if (stage / DONE_SENTINEL).is_file() and not list(stage.glob("*.ready")):
                break
            idle_polls += 1
            time.sleep(0.3)
            continue
        idle_polls = 0
        tiff = stage / f"{stem}.tiff"
        claim = stage / f"{stem}.claim.{pid}"
        out_path = args.out / f"{stem}.npz"
        try:
            if out_path.is_file():
                done += 1
            else:
                slide = tiffslide.TiffSlide(str(tiff))
                try:
                    chosen = ehf.collect_tissue_tiles(slide, args.max_tiles, args.tissue_thresh, args.scan_target)
                finally:
                    slide.close()
                tiles = [c[3] for c in chosen]
                feats = ehf.encode_tiles(tiles, model, device, mean, std, args.batch_size)
                if feats.shape[0] == 0:
                    failed += 1
                else:
                    k = min(args.topk, feats.shape[0])
                    top = feats[:k]
                    np.savez_compressed(
                        out_path,
                        pooled_mean=feats.mean(axis=0).astype(np.float32),
                        pooled_topk=top.mean(axis=0).astype(np.float32),
                        tile_feats=top.astype(np.float32),
                        tile_coords=np.array([(c[0], c[1]) for c in chosen[:k]], dtype=np.int32),
                        tissue_fraction=np.float32(np.mean([c[2] for c in chosen]) if chosen else 0.0),
                        n_tiles=np.int32(feats.shape[0]),
                    )
                    done += 1
        except Exception as exc:
            failed += 1
            print(f"[worker {args.worker_id}] FAIL {stem}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            tiff.unlink(missing_ok=True)
            claim.unlink(missing_ok=True)
        if (done + failed) % 25 == 0:
            print(f"[worker {args.worker_id}] done={done} failed={failed}", flush=True)

    print(f"[worker {args.worker_id}] FINISHED done={done} failed={failed}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.role == "producer":
        return run_producer(args)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
