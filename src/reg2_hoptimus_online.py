"""Online H-optimus patch encoder for dual-encoder diagnosis inference.

Mirrors ``reg2_titan_online.extract_titan_slide_embedding`` but encodes the SAME
scanned tissue tiles with a frozen H-optimus (ViT-g/14, 1536-d) instead of CONCH.
Returns ``patch_features`` [N,1536] + ``pooled_mean`` [1536] + ``coords`` so it can
feed the diagnosis ABMIL. The tile scan is identical to the TITAN path, so both
encoders share decoded tiles through ``extract_from_tiles``.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.reg2_titan_online import (
    _float_env,
    _int_env,
    collect_patches,
    patch_px_for_mpp,
    slide_mpp,
)

MODEL_ENV = "REG2_HOPTIMUS_DIR"
ARCH_ENV = "REG2_HOPTIMUS_ARCH"
MAX_PATCHES_ENV = "REG2_TITAN_MAX_PATCHES"  # share the TITAN cap for train-serve parity
MIN_PATCHES_ENV = "REG2_TITAN_MIN_PATCHES"
MIN_PATCHES_TRIGGER_MAX_ENV = "REG2_TITAN_MIN_PATCHES_TRIGGER_MAX"
SCAN_TARGET_ENV = "REG2_TITAN_SCAN_TARGET"
SCAN_CAP_ENV = "REG2_TITAN_SCAN_CAP"
TISSUE_THRESH_ENV = "REG2_TITAN_TISSUE_THRESH"
SCAN_BUDGET_ENV = "REG2_TITAN_SCAN_BUDGET_S"
BATCH_ENV = "REG2_HOPTIMUS_BATCH"
TORCH_THREADS_ENV = "REG2_TITAN_TORCH_THREADS"

DEFAULT_ARCH = "vit_giant_patch14_reg4_dinov2"
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
TILE = 224


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_dirs() -> list[Path]:
    out: list[Path] = []
    env_dir = os.environ.get(MODEL_ENV)
    if env_dir:
        out.append(Path(env_dir))
    out.extend(
        [
            Path("/opt/ml/model/H-optimus"),
            Path("/opt/ml/model/H-optimus-1"),
            Path("/opt/ml/model/H-optimus-0"),
            _template_dir() / "model" / "H-optimus",
        ]
    )
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        try:
            r = p.resolve()
        except OSError:
            r = p
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


# fp16 weight halves the cold-disk read (4.54GB fp32 -> 2.27GB) and so ~halves the
# ViT-g cold-load; encoding runs under fp16 autocast anyway, so it is lossless here.
_WEIGHT_NAMES = ("pytorch_model_fp16.bin", "pytorch_model.bin")


def _weight_file(hopt_dir: Path) -> Path:
    for name in _WEIGHT_NAMES:
        cand = hopt_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no H-optimus weight file in {hopt_dir} (tried {_WEIGHT_NAMES})")


def find_hoptimus_dir() -> Path:
    for path in _candidate_dirs():
        if any((path / name).is_file() for name in _WEIGHT_NAMES):
            return path
    checked = ", ".join(str(p) for p in _candidate_dirs())
    raise FileNotFoundError(f"H-optimus weights not found; checked: {checked}")


@lru_cache(maxsize=1)
def load_hoptimus():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import timm
    import torch

    hopt_dir = find_hoptimus_dir()
    arch = os.environ.get(ARCH_ENV, DEFAULT_ARCH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    threads = _int_env(TORCH_THREADS_ENV, 2)
    if threads > 0:
        torch.set_num_threads(threads)

    model = timm.create_model(
        arch,
        pretrained=False,
        num_classes=0,
        img_size=TILE,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    weight_file = _weight_file(hopt_dir)
    state = torch.load(weight_file, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[hopt-online] WARNING missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[hopt-online] WARNING unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model = model.eval().to(device)
    print(f"[hopt-online] loaded H-optimus ({arch}) from {weight_file} on {device}")
    return model, device


def _encode_tiles(model, device, tiles: list[np.ndarray], batch: int) -> np.ndarray:
    import torch
    from PIL import Image

    mean = np.asarray(HOPTIMUS_MEAN, np.float32)
    std = np.asarray(HOPTIMUS_STD, np.float32)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    feats: list[np.ndarray] = []
    buf: list[np.ndarray] = []

    def flush() -> None:
        if not buf:
            return
        arr = np.stack(buf).astype(np.float32) / 255.0
        arr = (arr - mean) / std
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device, non_blocking=True)
        with torch.inference_mode(), autocast_ctx:
            out = model(t)
        feats.append(out.float().cpu().numpy())
        buf.clear()

    for tile in tiles:
        if tile.shape[:2] != (TILE, TILE):
            tile = np.asarray(Image.fromarray(tile).resize((TILE, TILE), Image.BILINEAR), dtype=np.uint8)
        buf.append(tile)
        if len(buf) >= batch:
            flush()
    flush()
    if not feats:
        return np.zeros((0, model.num_features), dtype=np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_from_tiles(tiles: list[np.ndarray], coords: np.ndarray, patch_px: int) -> dict[str, np.ndarray]:
    """Encode pre-scanned tiles with H-optimus (shared-tile dual-encoder path)."""
    model, device = load_hoptimus()
    batch = _int_env(BATCH_ENV, 16)
    feats = _encode_tiles(model, device, tiles, batch)
    if feats.shape[0] == 0:
        raise RuntimeError("H-optimus produced no patch features")
    return {
        "patch_features": feats,
        "pooled_mean": feats.mean(axis=0).astype(np.float32),
        "coords": np.asarray(coords, np.int64),
        "n_patches": np.asarray([feats.shape[0]], np.int32),
        "patch_px": np.asarray([patch_px], np.int32),
    }


def extract_hoptimus_features(wsi_path: str | Path) -> dict[str, np.ndarray]:
    """Standalone single-encoder path: scan tiles + encode with H-optimus."""
    import tiffslide

    start = time.time()
    max_patches = _int_env(MAX_PATCHES_ENV, 256)
    min_patches = _int_env(MIN_PATCHES_ENV, 0, minimum=0)
    min_patches_trigger_max_raw = os.environ.get(MIN_PATCHES_TRIGGER_MAX_ENV)
    min_patches_trigger_max = (
        _int_env(MIN_PATCHES_TRIGGER_MAX_ENV, 0, minimum=0)
        if min_patches_trigger_max_raw not in (None, "")
        else None
    )
    scan_target = _int_env(SCAN_TARGET_ENV, 768)
    scan_cap = _int_env(SCAN_CAP_ENV, 0, minimum=0)
    tissue_thresh = _float_env(TISSUE_THRESH_ENV, 0.10)
    scan_budget_s = _float_env(SCAN_BUDGET_ENV, 25.0)

    with tiffslide.TiffSlide(str(wsi_path)) as slide:
        mpp = slide_mpp(slide)
        patch_px = patch_px_for_mpp(mpp)
        tiles, coords, scanned = collect_patches(
            slide,
            max_patches=max_patches,
            min_patches=min_patches,
            min_patches_trigger_max=min_patches_trigger_max,
            tissue_thresh=tissue_thresh,
            patch_px=patch_px,
            scan_target=scan_target,
            scan_cap=scan_cap,
            scan_budget_s=scan_budget_s,
        )
    if not tiles:
        raise RuntimeError("no readable tissue patches for H-optimus")

    out = extract_from_tiles(tiles, coords, patch_px)
    out["scanned"] = np.asarray([scanned], np.int32)
    elapsed = time.time() - start
    print(
        "[hopt-online] "
        f"patches={out['n_patches'][0]} scanned={scanned} patch_px={patch_px} "
        f"mpp={mpp if mpp is not None else 0:.4g} elapsed={elapsed:.1f}s"
    )
    return out
