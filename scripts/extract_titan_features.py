#!/usr/bin/env python3
"""RAM-staged TITAN slide-embedding extraction (CONCH v1.5 patches -> TITAN).

Pipeline per slide:
  1. tile level-0 into ~512px@20x patches (physical 256 um FOV; patch_px derived
     from slide MPP, default 20x when MPP is missing/bogus),
  2. tissue-filter + skip corrupt JPEG tiles (same robustness as the H-optimus run),
  3. encode each patch with CONCH v1.5 (448px input) -> patch features,
  4. TITAN.encode_slide_from_patch_features(feats, coords_lv0, patch_size_lv0)
     -> a single slide embedding,
  5. save <stem>.npz with slide_embedding (+ meta).

Reuses the HDD->/dev/shm producer + lock-free claim queue from
extract_hoptimus_staged.py so the single spinning HDD only does sequential IO
while N GPU workers consume from RAM (lets all 4 cards run in parallel).

Weights required at --titan-dir (download elsewhere, drop in):
    model.safetensors                (TITAN slide encoder)
    conch_v1_5_pytorch_model.bin     (CONCH v1.5 patch encoder)
  (+ the *.py code files, already present)

Launch (1 producer + 2 workers/card across 4 cards):
    OUT=features/titan_v1
    PY=python
    $PY scripts/extract_titan_features.py --role producer --out $OUT &
    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$((i/2)) $PY scripts/extract_titan_features.py \
        --role worker --worker-id $i --out $OUT &
    done; wait
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_hoptimus_features as ehf  # noqa: E402  (tissue heuristic)
import extract_hoptimus_staged as ehs  # noqa: E402  (producer + claim queue)

DEFAULT_MANIFEST = ehf.DEFAULT_MANIFEST
DEFAULT_DATASET = ehf.DEFAULT_DATASET
DEFAULT_OUT = Path("features/titan_v1")
DEFAULT_TITAN = Path("models/TITAN")
DEFAULT_STAGE = Path("/dev/shm/reg2_stage_titan")
FOV_UM = 256.0  # 512 px * 0.5 mpp = 256 um physical field of view per patch at 20x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["producer", "worker"], required=True)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    p.add_argument("--titan-dir", type=Path, default=DEFAULT_TITAN)
    p.add_argument("--max-patches", type=int, default=1024, help="Max tissue patches encoded per slide.")
    p.add_argument("--min-patches", type=int, default=0, help="Backfill low-tissue ranked patches until this count when sparse tissue passes the threshold.")
    p.add_argument(
        "--scan-target",
        type=int,
        default=768,
        help="Approximate number of level-0 candidate patches to scan for tissue ranking.",
    )
    p.add_argument("--tissue-thresh", type=float, default=0.10)
    p.add_argument("--conch-batch", type=int, default=32, help="Patch batch for CONCH (448px, fp16).")
    p.add_argument("--torch-threads", type=int, default=2, help="CPU threads per worker.")
    p.add_argument("--max-buffered", type=int, default=300)
    p.add_argument("--cap-gb", type=float, default=55.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-patch-features", action="store_true",
                   help="Also store per-patch CONCH features (fp16) + coords for ABMIL training.")
    return p.parse_args()


def patch_px_for_mpp(mpp: float | None) -> int:
    """Level-0 patch size that captures a 256 um FOV; default 512 (20x) if unknown."""
    if mpp is None or not (0.1 <= mpp <= 0.8):
        return 512
    px = int(round(FOV_UM / mpp))
    # snap to the nearest sane WSI patch size
    return 1024 if px > 768 else 512


def slide_mpp(slide) -> float | None:
    pr = slide.properties
    for key in ("tiffslide.mpp-x", "openslide.mpp-x", "aperio.MPP"):
        v = pr.get(key)
        try:
            f = float(v)
            if 0.0 < f < 50.0:
                return f
        except (TypeError, ValueError):
            continue
    return None


def collect_patches(
    slide,
    max_patches: int,
    tissue_thresh: float,
    patch_px: int,
    scan_target: int,
    min_patches: int = 0,
):
    """Grid-scan level-0 patches; return (tiles[list HxWx3 uint8], coords[N,2], scan_count).

    Skips corrupt JPEG tiles individually. Tiles are the raw patch_px crops (PIL->np);
    CONCH's own eval_transform later resizes them to 448.
    """
    w0, h0 = slide.level_dimensions[0]
    nx, ny = max(1, w0 // patch_px), max(1, h0 // patch_px)
    total = nx * ny
    stride = max(1, int(round((total / max(1, scan_target)) ** 0.5)))
    cand: list[tuple[float, int, int, np.ndarray]] = []
    for gy in range(0, ny, stride):
        y = gy * patch_px
        for gx in range(0, nx, stride):
            x = gx * patch_px
            try:
                region = slide.read_region((x, y), 0, (patch_px, patch_px)).convert("RGB")
            except Exception:
                continue
            tile = np.asarray(region)
            if tile.shape[:2] != (patch_px, patch_px):
                continue
            frac = ehf._tissue_fraction(tile)
            cand.append((frac, x, y, tile))
    cand.sort(key=lambda c: c[0], reverse=True)
    passing = [c for c in cand if c[0] >= tissue_thresh]
    min_keep = min(max(0, int(min_patches)), max_patches)
    if not passing:
        chosen = cand[:max_patches]
    elif len(passing) < min_keep:
        passing_ids = {(c[1], c[2]) for c in passing}
        backfill = [c for c in cand if (c[1], c[2]) not in passing_ids]
        chosen = (passing + backfill[: max(0, min_keep - len(passing))])[:max_patches]
    else:
        chosen = passing[:max_patches]
    tiles = [c[3] for c in chosen]
    coords = np.array([(c[1], c[2]) for c in chosen], dtype=np.int64)
    return tiles, coords, len(cand)


def build_titan(titan_dir: Path, device):
    import os
    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast
    import huggingface_hub

    # This box cannot reach huggingface.co / Xet; force fully-offline local loads.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    conch_bin = titan_dir / "conch_v1_5_pytorch_model.bin"
    if not conch_bin.is_file():
        raise FileNotFoundError(f"missing CONCH weights: {conch_bin}")

    def _local_download(repo_id, filename, **kwargs):
        p = titan_dir / filename
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"local weight not found: {p}")

    huggingface_hub.hf_hub_download = _local_download  # type: ignore[assignment]

    _orig_tok = PreTrainedTokenizerFast.from_pretrained

    @classmethod
    def _local_tok(cls, name_or_path, *args, **kwargs):
        if str(name_or_path) in ("MahmoodLab/TITAN", "TITAN"):
            name_or_path = str(titan_dir)
        kwargs.setdefault("local_files_only", True)
        return _orig_tok(name_or_path, *args, **kwargs)

    PreTrainedTokenizerFast.from_pretrained = _local_tok  # type: ignore[method-assign]

    titan = AutoModel.from_pretrained(str(titan_dir), trust_remote_code=True, local_files_only=True)
    titan = titan.to(device).eval()
    conch, eval_transform = titan.return_conch()
    conch = conch.to(device).eval()
    return titan, conch, eval_transform


def conch_encode(conch, eval_transform, tiles, device, batch: int):
    """Encode a list of HxWx3 uint8 patches into (N, D) CONCH features."""
    import torch
    from PIL import Image

    feats = []
    buf = []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            out = conch(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats.append(out.float().cpu())
        buf.clear()

    for t in tiles:
        buf.append(eval_transform(Image.fromarray(t)))
        if len(buf) >= batch:
            flush()
    flush()
    if not feats:
        return None
    import torch as _t
    return _t.cat(feats, dim=0)


def titan_grid_extent(coords: np.ndarray, patch_px: int) -> tuple[int, int]:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return 0, 0
    step = max(1, int(patch_px))
    offset = arr[:, :2].min(axis=0)
    grid = np.floor_divide(arr[:, :2] - offset, step)
    extent = grid.max(axis=0) - grid.min(axis=0) + 1
    return int(extent[0]), int(extent[1])


def pad_sparse_titan_grid(patch_features, coords: np.ndarray, patch_px: int):
    """Pad TITAN-only input when tissue coordinates collapse to a 1D grid.

    TITAN's slide encoder has an alibi/indexing edge case for 1xN / Nx1 grids.
    We add one mean-feature pseudo-patch only for the slide embedding call; the
    saved real patch features and coordinates remain unchanged.
    """
    extent_x, extent_y = titan_grid_extent(coords, patch_px)
    if extent_x >= 2 and extent_y >= 2:
        return patch_features, coords, False
    if patch_features is None or int(patch_features.shape[0]) == 0:
        return patch_features, coords, False
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return patch_features, coords, False

    step = max(1, int(patch_px))
    pad_coord = arr[0, :2].copy()
    if extent_x < 2:
        pad_coord[0] += step
    if extent_y < 2:
        pad_coord[1] += step
    if np.any(np.all(arr[:, :2] == pad_coord[None, :], axis=1)):
        pad_coord = arr[0, :2] + np.asarray([step, step], dtype=np.int64)

    import torch

    pad_feat = patch_features.float().mean(dim=0, keepdim=True).to(patch_features.dtype)
    padded_features = torch.cat([patch_features, pad_feat], dim=0)
    padded_coords = np.concatenate([arr[:, :2], pad_coord[None, :]], axis=0)
    return padded_features, padded_coords, True


def run_worker(args: argparse.Namespace) -> int:
    import torch
    import tiffslide

    stage = args.stage_dir
    args.out.mkdir(parents=True, exist_ok=True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(2, args.torch_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pid = os.getpid()
    print(f"[titan w{args.worker_id}] device={device} pid={pid} loading model...", flush=True)
    titan, conch, eval_transform = build_titan(args.titan_dir, device)
    print(f"[titan w{args.worker_id}] model ready", flush=True)

    done = failed = 0
    while True:
        stem = ehs.claim_one(stage, pid)
        if stem is None:
            if (stage / ehs.DONE_SENTINEL).is_file() and not list(stage.glob("*.ready")):
                break
            time.sleep(0.3)
            continue
        tiff = stage / f"{stem}.tiff"
        claim = stage / f"{stem}.claim.{pid}"
        out_path = args.out / f"{stem}.npz"
        try:
            if out_path.is_file():
                done += 1
            else:
                slide = tiffslide.TiffSlide(str(tiff))
                try:
                    mpp = slide_mpp(slide)
                    patch_px = patch_px_for_mpp(mpp)
                    tiles, coords, n_scan = collect_patches(
                        slide,
                        args.max_patches,
                        args.tissue_thresh,
                        patch_px,
                        args.scan_target,
                        args.min_patches,
                    )
                finally:
                    slide.close()
                if not tiles:
                    failed += 1
                else:
                    pf = conch_encode(conch, eval_transform, tiles, device, args.conch_batch)
                    embed_pf, embed_coords, padded_sparse = pad_sparse_titan_grid(pf, coords, patch_px)
                    if padded_sparse:
                        before = titan_grid_extent(coords, patch_px)
                        after = titan_grid_extent(embed_coords, patch_px)
                        print(
                            f"[titan w{args.worker_id}] padded sparse grid {stem}: "
                            f"{before[0]}x{before[1]} -> {after[0]}x{after[1]}",
                            flush=True,
                        )
                    feats_t = embed_pf.unsqueeze(0).to(device)  # (1, N, C) per TITAN API
                    coords_t = torch.from_numpy(embed_coords).unsqueeze(0).to(device)  # (1, N, 2)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        emb = titan.encode_slide_from_patch_features(feats_t, coords_t, int(patch_px))
                    emb = emb.squeeze().float().cpu().numpy().astype(np.float32)
                    save_kw = dict(
                        slide_embedding=emb,
                        n_patches=np.int32(len(tiles)),
                        patch_px=np.int32(patch_px),
                        mpp=np.float32(mpp if mpp else 0.0),
                        n_scan=np.int32(n_scan),
                    )
                    if args.save_patch_features:
                        save_kw["patch_features"] = pf.cpu().numpy().astype(np.float16)
                        save_kw["coords"] = coords.astype(np.int32)
                    tmp_out = out_path.with_name(f".{out_path.name}.tmp.{pid}")
                    with tmp_out.open("wb") as f:
                        np.savez_compressed(f, **save_kw)
                    os.replace(tmp_out, out_path)
                    done += 1
        except Exception as exc:
            failed += 1
            print(f"[titan w{args.worker_id}] FAIL {stem}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
        finally:
            for tmp in args.out.glob(f".{stem}.npz.tmp.*"):
                tmp.unlink(missing_ok=True)
            tiff.unlink(missing_ok=True)
            claim.unlink(missing_ok=True)
        if (done + failed) % 25 == 0:
            print(f"[titan w{args.worker_id}] done={done} failed={failed}", flush=True)

    print(f"[titan w{args.worker_id}] FINISHED done={done} failed={failed}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.role == "producer":
        # Reuse the identical HDD->shm sequential producer.
        return ehs.run_producer(args)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
