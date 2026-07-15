#!/usr/bin/env python3
"""Offline H-optimus-0 WSI feature extraction for REG2 v2.0.

Each worker is a SINGLE-GPU (24 GB) job; parallelism is pure data sharding, never
model parallelism. The same per-slide forward pass is what the submission must run
online on one A10G (24 GB). H-optimus-0 weights are ~2.2 GB in fp16, and encoding
runs under torch.autocast(fp16) with a bounded tile batch, so it fits 24 GB.

For each case in a CoT manifest this resolves the local TIFF, samples informative
tissue tiles, encodes them with a frozen H-optimus-0, and writes a compact
per-slide ``.npz`` feature record. Missing WSIs are logged and skipped (not fatal).
Existing outputs are skipped, so runs are resumable.

To use the 4 free 4090s, launch 4 INDEPENDENT single-GPU shards (one per card):

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i \
      python scripts/extract_hoptimus_features.py \
        --num-shards 4 --shard-index $i &
    done; wait

Requires the env from scripts/setup_reg2026_gpu_env.sh (timm + tiffslide).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("data/train_CoT_v01_balanced_3000_by_organ.json")
DEFAULT_DATASET = Path("data/reg2026")
DEFAULT_OUT = Path("features/hoptimus0_v2")
DEFAULT_HOPTIMUS = Path("models/H-optimus-0")

HOPTIMUS_ARCH = "vit_giant_patch14_reg4_dinov2"
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
TILE = 224
# Candidate sub-directories that may hold a slide (later entries override earlier).
SLIDE_DIRS = [
    "train",
    "train_revised_20260519",
    "train_revised_20260527",
    "train_revised_20260604",
    "debug",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--hoptimus-dir", type=Path, default=DEFAULT_HOPTIMUS)
    p.add_argument("--max-tiles", type=int, default=128, help="Max tissue tiles encoded per slide.")
    p.add_argument("--topk", type=int, default=16, help="Top tissue tiles kept as tile_feats.")
    p.add_argument("--batch-size", type=int, default=16, help="Tile batch; 16 fits one 24 GB GPU in fp16.")
    p.add_argument("--tissue-thresh", type=float, default=0.10, help="Min tissue fraction for a tile.")
    p.add_argument("--scan-target", type=int, default=1536,
                   help="Approx number of level-0 tiles to scan per slide for tissue ranking.")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def resolve_slide_path(dataset_dir: Path, case_id: str) -> Path | None:
    stem = case_id[:-5] if case_id.lower().endswith(".tiff") else case_id
    found: Path | None = None
    for sub in SLIDE_DIRS:
        cand = dataset_dir / sub / f"{stem}.tiff"
        if cand.is_file() and not Path(str(cand) + ".aria2").exists():
            found = cand  # later dirs (revised) win
    return found


def build_model(hoptimus_dir: Path, device):
    import timm
    import torch

    # H-optimus-0 is a 224px ViT-g/14 with reg tokens + layerscale (init_values=1e-5).
    # The timm default for this arch is img_size=518 (1369 pos tokens) and NO layerscale,
    # so both must be set explicitly or the checkpoint will not load correctly.
    model = timm.create_model(
        HOPTIMUS_ARCH,
        pretrained=False,
        num_classes=0,
        img_size=TILE,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    state = torch.load(hoptimus_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[extract] WARNING missing keys: {len(missing)} (e.g. {missing[:3]})", flush=True)
    if unexpected:
        print(f"[extract] WARNING unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})", flush=True)
    model.eval().to(device)
    return model


def _tissue_fraction(tile: np.ndarray) -> float:
    """Tissue fraction on a coarsely subsampled tile (not-white/not-black + saturation)."""
    small = tile[::7, ::7].astype(np.int32)  # ~32x32 from a 224 tile
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-3)
    return float(((mx < 235) & (mx > 25) & (sat > 0.10)).mean())


def collect_tissue_tiles(
    slide, max_tiles: int, tissue_thresh: float, scan_target: int
) -> list[tuple[int, int, float, np.ndarray]]:
    """Strided level-0 tile scan returning the top tissue tiles already decoded.

    These slides are single-level tiled JPEGs with NO pyramid, so get_thumbnail()
    would JPEG-decode the whole image (~30 s). Instead we sample tiles on a uniform
    grid (~scan_target candidates), read each once, score tissue from the decoded
    pixels, and reuse those pixels for encoding -- no second read. ~1-2 s/slide.
    """
    w0, h0 = slide.level_dimensions[0]
    nx, ny = max(1, w0 // TILE), max(1, h0 // TILE)
    total = nx * ny
    stride = max(1, int(round((total / max(1, scan_target)) ** 0.5)))

    scanned: list[tuple[int, int, float, np.ndarray]] = []
    for gy in range(0, ny, stride):
        y = gy * TILE
        for gx in range(0, nx, stride):
            x = gx * TILE
            # Many downloaded WSIs are partially truncated: some JPEG tiles are
            # zero-filled and raise on decode. Skip the bad tile and keep the
            # (usually hundreds of) good ones rather than discarding the slide.
            try:
                region = slide.read_region((x, y), 0, (TILE, TILE)).convert("RGB")
            except Exception:
                continue
            tile = np.asarray(region)
            if tile.shape[:2] != (TILE, TILE):
                continue
            scanned.append((x, y, _tissue_fraction(tile), tile))

    scanned.sort(key=lambda c: c[2], reverse=True)
    passing = [c for c in scanned if c[2] >= tissue_thresh]
    chosen = passing if passing else scanned  # fall back to best tiles on near-empty slides
    return chosen[:max_tiles]


def encode_tiles(tiles: list[np.ndarray], model, device, mean, std, batch_size: int) -> np.ndarray:
    import torch

    feats: list[np.ndarray] = []
    batch: list[np.ndarray] = []

    def flush():
        if not batch:
            return
        arr = np.stack(batch).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
        use_cuda = device.type == "cuda"
        with torch.no_grad():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(t)
            else:
                out = model(t)
        feats.append(out.float().cpu().numpy())
        batch.clear()

    for tile in tiles:
        batch.append(tile)
        if len(batch) >= batch_size:
            flush()
    flush()
    if not feats:
        return np.zeros((0, model.num_features), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def main() -> int:
    args = parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device} shard={args.shard_index}/{args.num_shards}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise TypeError("manifest must be a list of cases")
    shard = [c for i, c in enumerate(cases) if i % args.num_shards == args.shard_index]
    if args.limit:
        shard = shard[: args.limit]
    print(f"[extract] {len(shard)} cases in this shard", flush=True)

    mean = np.array(HOPTIMUS_MEAN, dtype=np.float32)
    std = np.array(HOPTIMUS_STD, dtype=np.float32)
    model = build_model(args.hoptimus_dir, device)

    import tiffslide

    missing_log = args.out / f"missing_shard{args.shard_index}.txt"
    done = skipped = missing = failed = 0
    t_start = time.time()
    for idx, case in enumerate(shard):
        cid = str(case.get("id", "")).strip()
        stem = cid[:-5] if cid.lower().endswith(".tiff") else cid
        out_path = args.out / f"{stem}.npz"
        if out_path.is_file():
            skipped += 1
            continue
        slide_path = resolve_slide_path(args.dataset_dir, cid)
        if slide_path is None:
            missing += 1
            with missing_log.open("a", encoding="utf-8") as f:
                f.write(stem + "\n")
            continue
        try:
            slide = tiffslide.TiffSlide(str(slide_path))
            chosen = collect_tissue_tiles(slide, args.max_tiles, args.tissue_thresh, args.scan_target)
            tiles = [c[3] for c in chosen]
            tile_feats = encode_tiles(tiles, model, device, mean, std, args.batch_size)
            if tile_feats.shape[0] == 0:
                failed += 1
                continue
            pooled_mean = tile_feats.mean(axis=0)
            k = min(args.topk, tile_feats.shape[0])
            top_feats = tile_feats[:k]  # tiles already ranked by tissue fraction
            pooled_topk = top_feats.mean(axis=0)
            np.savez_compressed(
                out_path,
                pooled_mean=pooled_mean.astype(np.float32),
                pooled_topk=pooled_topk.astype(np.float32),
                tile_feats=top_feats.astype(np.float32),
                tile_coords=np.array([(c[0], c[1]) for c in chosen[:k]], dtype=np.int32),
                tissue_fraction=np.float32(np.mean([c[2] for c in chosen]) if chosen else 0.0),
                n_tiles=np.int32(tile_feats.shape[0]),
            )
            done += 1
        except Exception as exc:  # keep the shard alive on any single-slide failure
            failed += 1
            print(f"[extract] FAIL {stem}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if (idx + 1) % 20 == 0:
            rate = (done + skipped) / max(1e-6, time.time() - t_start)
            print(f"[extract] {idx + 1}/{len(shard)} done={done} skip={skipped} "
                  f"miss={missing} fail={failed} ({rate:.2f}/s)", flush=True)

    print(f"[extract] FINISHED shard {args.shard_index}: done={done} skipped={skipped} "
          f"missing={missing} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
