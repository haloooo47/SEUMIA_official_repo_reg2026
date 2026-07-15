#!/usr/bin/env python3
"""Full-dataset Virchow2 patch extraction with the SAME own-tissue tiling as
extract_hopt1_full.py (train-serve match). Multi-FM ensemble lever (PathBench:
Virchow2 + H-optimus-1 are top-2; organ-complementary).

Per case: own-tiling level-0 scan (max 256 patches, scan_target 768, tissue 0.10),
encode with frozen Virchow2 (timm ViT-H/14, ImageNet mean/std), feature =
concat[CLS(1280), mean(patch tokens)(1280)] = 2560-d.

Saves patch_features_v2 [N,2560] fp16 + pooled_mean + coords + patch_px + n_patches.
Reuses h1f.collect_patches/slide_mpp/patch_px_for_mpp. Resumable.
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
import extract_hoptimus_features as ehf  # noqa: E402  (resolver)
import extract_hopt1_full as h1f  # noqa: E402  (own tiling)

DEFAULT_COT = h1f.DEFAULT_COT
DEFAULT_OUT = Path("features/virchow2_full_256")
DEFAULT_DATASET = h1f.DEFAULT_DATASET
DEFAULT_V2 = Path("models/Virchow2")
TILE = 224
# Virchow2 uses ImageNet normalization (per its config.json), NOT the H-optimus stats.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cot", type=Path, default=DEFAULT_COT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--virchow-dir", type=Path, default=DEFAULT_V2)
    p.add_argument("--max-patches", type=int, default=256)
    p.add_argument("--min-patches", type=int, default=0)
    p.add_argument("--scan-target", type=int, default=768)
    p.add_argument("--tissue-thresh", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def build_virchow2(v2_dir: Path, device):
    import timm
    import torch
    from timm.layers import SwiGLUPacked
    from safetensors.torch import load_file

    model = timm.create_model(
        "vit_huge_patch14_224", pretrained=False, num_classes=0, img_size=TILE,
        init_values=1e-5, reg_tokens=4, mlp_ratio=5.3375, global_pool="",
        mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, dynamic_img_size=False,
    )
    sd = load_file(str(v2_dir / "model.safetensors"))
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss:
        print(f"[v2] WARNING missing {len(miss)} (e.g. {miss[:3]})", flush=True)
    if unexp:
        print(f"[v2] WARNING unexpected {len(unexp)} (e.g. {unexp[:3]})", flush=True)
    return model.eval().to(device)


def encode_tiles(tiles, model, device, batch_size):
    import torch

    feats, buf = [], []
    npref = model.num_prefix_tokens

    def flush():
        if not buf:
            return
        arr = np.stack(buf).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            o = model.forward_features(t)
            cls = o[:, 0]
            patch = o[:, npref:]
            concat = torch.cat([cls, patch.mean(1)], dim=-1)  # 2560
        feats.append(concat.float().cpu().numpy())
        buf.clear()

    for tile in tiles:
        buf.append(tile)
        if len(buf) >= batch_size:
            flush()
    flush()
    if not feats:
        return np.zeros((0, 2560), np.float32)
    return np.concatenate(feats, axis=0)


def main() -> int:
    args = parse_args()
    import torch
    import tiffslide
    import json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[v2full] device={device} shard={args.shard_index}/{args.num_shards}", flush=True)

    ids = []
    for c in json.loads(args.cot.read_text()):
        cid = str(c.get("id", "")).strip()
        cid = cid[:-5] if cid.lower().endswith(".tiff") else cid
        if cid:
            ids.append(cid)
    ids = sorted(set(ids))
    shard = [s for i, s in enumerate(ids) if i % args.num_shards == args.shard_index]
    if args.limit:
        shard = shard[: args.limit]
    print(f"[v2full] {len(shard)} cases in shard (of {len(ids)})", flush=True)

    model = build_virchow2(args.virchow_dir, device)
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
            mpp = h1f.slide_mpp(slide)
            patch_px = h1f.patch_px_for_mpp(mpp)
            tiles, kept = h1f.collect_patches(
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
            feats = encode_tiles(tiles, model, device, args.batch_size)
            if feats.shape[0] == 0:
                empty += 1
                continue
            tmp = out_path.with_name(f".{out_path.name}.tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, patch_features_v2=feats.astype(np.float16),
                                    coords=kept.astype(np.int32),
                                    pooled_mean=feats.mean(axis=0).astype(np.float32),
                                    patch_px=np.int32(patch_px), n_patches=np.int32(feats.shape[0]))
            os.replace(tmp, out_path)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[v2full] FAIL {stem}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        if (idx + 1) % 50 == 0:
            rate = (done + skipped) / max(1e-6, time.time() - t0)
            print(f"[v2full] {idx+1}/{len(shard)} done={done} skip={skipped} miss={missing} "
                  f"empty={empty} fail={failed} ({rate:.2f}/s)", flush=True)
    print(f"[v2full] FINISHED shard {args.shard_index}: done={done} skipped={skipped} "
          f"missing={missing} empty={empty} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
