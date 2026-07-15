#!/usr/bin/env python3
"""Full-dataset (11220) H-optimus-1 patch extraction with the SAME own-tissue tiling
the ONLINE single-encoder pipeline uses (reg2_hoptimus_online -> reg2_titan_online
collect_patches), so train-serve match holds for a single-encoder H-opt1 model.

Per case: resolve TIFF, strided level-0 tissue scan (max 256 patches, scan_target
768, tissue_thresh 0.10, patch_px from mpp), encode with frozen H-optimus-1 (1536-d),
save patch_features_h1 [N,1536] fp16 + pooled_mean + coords + patch_px + n_patches.

Reads coords in raster (y,x) order for near-sequential HDD access. Resumable.
Shard across 4 GPUs:
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i PY scripts/extract_hopt1_full.py \
     --num-shards 4 --shard-index $i & done; wait
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_hoptimus_features as ehf  # noqa: E402  (build_model, resolver, mean/std, TILE)

DEFAULT_COT = Path("data/train_CoT_v01.json")
DEFAULT_OUT = Path("features/hoptimus1_full_256")
DEFAULT_DATASET = ehf.DEFAULT_DATASET
DEFAULT_H1 = Path("models/H-optimus-1")
TILE = ehf.TILE  # 224
FOV_UM = 256.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cot", type=Path, default=DEFAULT_COT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--hoptimus-dir", type=Path, default=DEFAULT_H1)
    p.add_argument("--max-patches", type=int, default=256)
    p.add_argument("--min-patches", type=int, default=0)
    p.add_argument("--scan-target", type=int, default=768)
    p.add_argument("--tissue-thresh", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def tissue_fraction(tile: np.ndarray) -> float:
    small = tile[::7, ::7].astype(np.int32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-3)
    return float(((mx < 235) & (mx > 25) & (sat > 0.10)).mean())


def patch_px_for_mpp(mpp):
    if mpp is None or not (0.1 <= mpp <= 0.8):
        return 512
    px = int(round(FOV_UM / mpp))
    return 1024 if px > 768 else 512


def slide_mpp(slide):
    props = slide.properties
    for key in ("tiffslide.mpp-x", "openslide.mpp-x", "aperio.MPP"):
        v = props.get(key)
        try:
            parsed = float(v)
        except (TypeError, ValueError):
            continue
        if 0.0 < parsed < 50.0:
            return parsed
    return None


def collect_patches(slide, max_patches, tissue_thresh, patch_px, scan_target, min_patches=0):
    from PIL import Image

    w0, h0 = slide.level_dimensions[0]
    nx, ny = max(1, w0 // patch_px), max(1, h0 // patch_px)
    total = nx * ny
    stride = max(1, int(round((total / max(1, scan_target)) ** 0.5)))
    # Keep the (resized-to-224) tile pixels in a capped heap so each candidate is
    # read exactly ONCE (no second read of the chosen tiles). Heap holds the top
    # max_patches by tissue fraction; tile cached at 224 (~147KB each).
    heap: list[tuple] = []
    seq = 0
    for gy in range(0, ny, stride):
        y = gy * patch_px
        for gx in range(0, nx, stride):
            x = gx * patch_px
            try:
                region = slide.read_region((x, y), 0, (patch_px, patch_px)).convert("RGB")
            except Exception:
                continue
            if region.size != (patch_px, patch_px):
                continue
            frac = tissue_fraction(np.asarray(region, dtype=np.uint8))
            if len(heap) >= max_patches and frac <= heap[0][0]:
                seq += 1
                continue
            if region.size != (TILE, TILE):
                region = region.resize((TILE, TILE), Image.BILINEAR)
            tile = np.asarray(region, dtype=np.uint8)
            if tile.shape[:2] != (TILE, TILE):
                continue
            item = (frac, seq, x, y, tile)
            seq += 1
            if len(heap) < max_patches:
                heapq.heappush(heap, item)
            elif frac > heap[0][0]:
                heapq.heapreplace(heap, item)
    ranked = sorted(heap, key=lambda it: it[0], reverse=True)
    passing = [it for it in ranked if it[0] >= tissue_thresh]
    min_keep = min(max(0, int(min_patches)), max_patches)
    if not passing:
        chosen = ranked
    elif len(passing) < min_keep:
        passing_ids = {it[1] for it in passing}
        backfill = [it for it in ranked if it[1] not in passing_ids]
        chosen = passing + backfill[: max(0, min_keep - len(passing))]
    else:
        chosen = passing
    tiles = [it[4] for it in chosen]
    kept = [(it[2], it[3]) for it in chosen]
    return tiles, np.asarray(kept, dtype=np.int32)


def _tta_views(arr_u8):
    """4 label-preserving views of a uint8 tile batch [B,H,W,3]: id, fliplr, flipud, rot180."""
    return [arr_u8,
            arr_u8[:, :, ::-1, :],
            arr_u8[:, ::-1, :, :],
            arr_u8[:, ::-1, ::-1, :]]


def encode_tiles(tiles, model, device, mean, std, batch_size, tta=False):
    import torch

    # If the model weights are fp16 (memory-saving for many concurrent workers),
    # feed fp16 inputs directly (no autocast); else fp32 + autocast.
    w_dtype = next(model.parameters()).dtype
    half = w_dtype == torch.float16
    feats, buf = [], []

    def _run(arr):
        arr = (arr.astype(np.float32) / 255.0 - mean) / std
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).contiguous().to(device)
        if half:
            with torch.no_grad():
                return model(t.half()).float().cpu().numpy()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(t.float()).float().cpu().numpy()

    def flush():
        if not buf:
            return
        arr = np.stack(buf)
        if tta:
            views = _tta_views(arr)
            out = sum(_run(v) for v in views) / float(len(views))  # TTA: mean over 4 views
        else:
            out = _run(arr)
        feats.append(out)
        buf.clear()

    for tile in tiles:
        buf.append(tile)
        if len(buf) >= batch_size:
            flush()
    flush()
    if not feats:
        return np.zeros((0, model.num_features), np.float32)
    return np.concatenate(feats, axis=0)


def main() -> int:
    args = parse_args()
    import torch
    import tiffslide

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[h1full] device={device} shard={args.shard_index}/{args.num_shards}", flush=True)

    cases = json.loads(args.cot.read_text())
    ids = []
    for c in cases:
        cid = str(c.get("id", "")).strip()
        cid = cid[:-5] if cid.lower().endswith(".tiff") else cid
        if cid:
            ids.append(cid)
    ids = sorted(set(ids))
    shard = [s for i, s in enumerate(ids) if i % args.num_shards == args.shard_index]
    if args.limit:
        shard = shard[: args.limit]
    print(f"[h1full] {len(shard)} cases in shard (of {len(ids)})", flush=True)

    mean = np.array(ehf.HOPTIMUS_MEAN, np.float32)
    std = np.array(ehf.HOPTIMUS_STD, np.float32)
    model = ehf.build_model(args.hoptimus_dir, device)

    done = skipped = missing = empty = failed = 0
    t0 = time.time()
    for idx, stem in enumerate(shard):
        out_path = args.out / f"{stem}.npz"
        if out_path.is_file():
            skipped += 1
            continue
        slide_path = ehf.resolve_slide_path(args.dataset_dir, stem)
        if slide_path is None:
            missing += 1
            continue
        try:
            slide = tiffslide.TiffSlide(str(slide_path))
            mpp = slide_mpp(slide)
            patch_px = patch_px_for_mpp(mpp)
            tiles, kept = collect_patches(
                slide,
                args.max_patches,
                args.tissue_thresh,
                patch_px,
                args.scan_target,
                args.min_patches,
            )
            slide.close()
            if not tiles:
                empty += 1
                continue
            feats = encode_tiles(tiles, model, device, mean, std, args.batch_size)
            if feats.shape[0] == 0:
                empty += 1
                continue
            tmp = out_path.with_name(f".{out_path.name}.tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(
                    fh,
                    patch_features_h1=feats.astype(np.float16),
                    coords=kept.astype(np.int32),
                    pooled_mean=feats.mean(axis=0).astype(np.float32),
                    patch_px=np.int32(patch_px),
                    n_patches=np.int32(feats.shape[0]),
                )
            os.replace(tmp, out_path)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[h1full] FAIL {stem}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        if (idx + 1) % 50 == 0:
            rate = (done + skipped) / max(1e-6, time.time() - t0)
            print(f"[h1full] {idx+1}/{len(shard)} done={done} skip={skipped} miss={missing} "
                  f"empty={empty} fail={failed} ({rate:.2f}/s)", flush=True)

    print(f"[h1full] FINISHED shard {args.shard_index}: done={done} skipped={skipped} "
          f"missing={missing} empty={empty} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
