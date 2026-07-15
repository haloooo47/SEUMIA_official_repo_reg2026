from __future__ import annotations

import heapq
import os
import shutil
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


FOV_UM = 256.0
MODEL_ENV = "REG2_TITAN_DIR"
MAX_PATCHES_ENV = "REG2_TITAN_MAX_PATCHES"
MIN_PATCHES_ENV = "REG2_TITAN_MIN_PATCHES"
MIN_PATCHES_TRIGGER_MAX_ENV = "REG2_TITAN_MIN_PATCHES_TRIGGER_MAX"
SCAN_TARGET_ENV = "REG2_TITAN_SCAN_TARGET"
SCAN_CAP_ENV = "REG2_TITAN_SCAN_CAP"
TISSUE_THRESH_ENV = "REG2_TITAN_TISSUE_THRESH"
CONCH_BATCH_ENV = "REG2_TITAN_CONCH_BATCH"
SCAN_BUDGET_ENV = "REG2_TITAN_SCAN_BUDGET_S"
TOTAL_BUDGET_ENV = "REG2_TITAN_TOTAL_BUDGET_S"
TORCH_THREADS_ENV = "REG2_TITAN_TORCH_THREADS"
HF_MODULES_CACHE_ENV = "HF_MODULES_CACHE"


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_titan_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get(MODEL_ENV)
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/opt/ml/model/TITAN"),
            _template_dir() / "model" / "TITAN",
        ]
    )

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def find_titan_dir() -> Path:
    for path in _candidate_titan_dirs():
        if (path / "model.safetensors").is_file() and (
            path / "conch_v1_5_pytorch_model.bin"
        ).is_file():
            return path
    checked = ", ".join(str(p) for p in _candidate_titan_dirs())
    raise FileNotFoundError(f"TITAN assets not found; checked: {checked}")


def _prepare_transformers_module_cache(titan_dir: Path) -> None:
    """Preload TITAN remote-code files for cold, offline containers.

    ``AutoModel.from_pretrained(..., trust_remote_code=True)`` copies local
    remote-code modules into Hugging Face's dynamic-module cache. In a fresh
    container it may miss nested relative imports from TITAN's text tower, so we
    seed the module directory with all local TITAN Python files before loading.
    """
    cache_root = Path(os.environ.setdefault(HF_MODULES_CACHE_ENV, "/tmp/hf_modules"))
    module_dir = cache_root / "transformers_modules" / titan_dir.name
    module_dir.mkdir(parents=True, exist_ok=True)
    (cache_root / "__init__.py").touch()
    (cache_root / "transformers_modules" / "__init__.py").touch()
    (module_dir / "__init__.py").touch()
    for py_file in titan_dir.glob("*.py"):
        shutil.copy2(py_file, module_dir / py_file.name)


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def tissue_fraction(tile: np.ndarray) -> float:
    small = tile[::7, ::7].astype(np.int32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-3)
    return float(((mx < 235) & (mx > 25) & (sat > 0.10)).mean())


def patch_px_for_mpp(mpp: float | None) -> int:
    if mpp is None or not (0.1 <= mpp <= 0.8):
        return 512
    px = int(round(FOV_UM / mpp))
    return 1024 if px > 768 else 512


def slide_mpp(slide: Any) -> float | None:
    props = slide.properties
    for key in ("tiffslide.mpp-x", "openslide.mpp-x", "aperio.MPP"):
        value = props.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < parsed < 50.0:
            return parsed
    return None


def collect_patches(
    slide: Any,
    *,
    max_patches: int,
    min_patches: int = 0,
    min_patches_trigger_max: int | None = None,
    tissue_thresh: float,
    patch_px: int,
    scan_target: int,
    scan_cap: int = 0,
    scan_budget_s: float = 0.0,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    w0, h0 = slide.level_dimensions[0]
    nx, ny = max(1, w0 // patch_px), max(1, h0 // patch_px)
    total = nx * ny
    stride = max(1, int(round((total / max(1, scan_target)) ** 0.5)))

    # The grid scan is DISK-I/O bound (level-0 tiles, no pyramid): on a large
    # sparse slide with a cold OS cache it can dominate the per-case budget, so
    # we cap its wall-clock and stop early with the best tiles found so far.
    scan_start = time.time()
    budget_hit = False
    cap_hit = False

    grid_x = list(range(0, nx, stride))
    grid_y = list(range(0, ny, stride))
    total_candidates = len(grid_x) * len(grid_y)
    if scan_cap > 0 and total_candidates > scan_cap:
        # A row-major early stop is deterministic but spatially biased toward
        # the slide's upper edge.  Build a smaller 2-D lattice instead so a
        # fixed scan cap still observes the complete tissue extent.
        aspect = len(grid_x) / max(1, len(grid_y))
        keep_x = min(len(grid_x), max(1, int(round((scan_cap * aspect) ** 0.5))))
        keep_y = min(len(grid_y), max(1, scan_cap // keep_x))
        while keep_x * keep_y > scan_cap:
            if keep_x >= keep_y and keep_x > 1:
                keep_x -= 1
            elif keep_y > 1:
                keep_y -= 1
            else:
                break
        x_indices = np.unique(np.linspace(0, len(grid_x) - 1, keep_x, dtype=np.int64))
        y_indices = np.unique(np.linspace(0, len(grid_y) - 1, keep_y, dtype=np.int64))
        scan_positions = [(grid_x[int(ix)], grid_y[int(iy)]) for iy in y_indices for ix in x_indices]
        print(
            "[titan-online] spatial scan cap: "
            f"grid={len(grid_x)}x{len(grid_y)} selected={len(x_indices)}x{len(y_indices)} "
            f"candidates={total_candidates}->{len(scan_positions)} cap={scan_cap}"
        )
    else:
        scan_positions = [(gx, gy) for gy in grid_y for gx in grid_x]

    heap: list[tuple[float, int, int, int, np.ndarray]] = []
    seq = 0
    scanned = 0
    for gx, gy in scan_positions:
        if budget_hit:
            break
        y = gy * patch_px
        x = gx * patch_px
        try:
            region = slide.read_region((x, y), 0, (patch_px, patch_px)).convert("RGB")
        except Exception:
            continue
        tile = np.asarray(region, dtype=np.uint8)
        if tile.shape[:2] != (patch_px, patch_px):
            continue
        frac = tissue_fraction(tile)
        item = (frac, seq, x, y, tile)
        seq += 1
        scanned += 1
        if len(heap) < max_patches:
            heapq.heappush(heap, item)
        elif frac > heap[0][0]:
            heapq.heapreplace(heap, item)
        # Check elapsed every ~16 tiles; only stop once we have >=1 tile so
        # we never return empty (which would force the v10 fallback).
        if (
            scan_budget_s > 0.0
            and heap
            and scanned % 16 == 0
            and (time.time() - scan_start) > scan_budget_s
        ):
            budget_hit = True
            break

    if cap_hit:
        print(
            "[titan-online] scan cap hit: "
            f"scanned={scanned} kept={len(heap)} cap={scan_cap}"
        )
    if budget_hit:
        print(
            "[titan-online] scan budget hit: "
            f"scanned={scanned} kept={len(heap)} "
            f"elapsed={time.time() - scan_start:.1f}s budget={scan_budget_s:.1f}s"
        )

    ranked = sorted(heap, key=lambda item: item[0], reverse=True)
    passing = [item for item in ranked if item[0] >= tissue_thresh]
    min_keep = min(max(0, int(min_patches)), max_patches)
    if not passing:
        chosen = ranked
    elif len(passing) < min_keep and (
        min_patches_trigger_max is None or len(passing) <= min_patches_trigger_max
    ):
        passing_ids = {item[1] for item in passing}
        backfill = [item for item in ranked if item[1] not in passing_ids]
        chosen = passing + backfill[: max(0, min_keep - len(passing))]
        if len(chosen) > len(passing):
            print(
                "[titan-online] tissue-min backfill "
                f"passing={len(passing)} chosen={len(chosen)} "
                f"min={min_keep} trigger_max={min_patches_trigger_max} "
                f"threshold={tissue_thresh:.3f}"
            )
    else:
        chosen = passing
    tiles = [item[4] for item in chosen]
    coords = np.asarray([(item[2], item[3]) for item in chosen], dtype=np.int64)
    return tiles, coords, scanned


@lru_cache(maxsize=1)
def load_titan_stack():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault(HF_MODULES_CACHE_ENV, "/tmp/hf_modules")

    import huggingface_hub
    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast

    titan_dir = find_titan_dir()
    _prepare_transformers_module_cache(titan_dir)
    conch_bin = titan_dir / "conch_v1_5_pytorch_model.bin"

    def _local_download(repo_id, filename, **kwargs):
        local_path = titan_dir / filename
        if local_path.is_file():
            return str(local_path)
        raise FileNotFoundError(f"local TITAN file not found: {local_path}")

    huggingface_hub.hf_hub_download = _local_download  # type: ignore[assignment]

    original_tokenizer_loader = PreTrainedTokenizerFast.from_pretrained.__func__

    @classmethod
    def _local_tokenizer(cls, name_or_path, *args, **kwargs):
        if str(name_or_path) in {"MahmoodLab/TITAN", "TITAN"}:
            name_or_path = str(titan_dir)
        kwargs.setdefault("local_files_only", True)
        return original_tokenizer_loader(cls, name_or_path, *args, **kwargs)

    PreTrainedTokenizerFast.from_pretrained = _local_tokenizer  # type: ignore[method-assign]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    threads = _int_env(TORCH_THREADS_ENV, 2)
    if threads > 0:
        torch.set_num_threads(threads)

    titan = AutoModel.from_pretrained(
        str(titan_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    titan = titan.to(device).eval()
    conch, eval_transform = titan.return_conch()
    conch = conch.to(device).eval()
    # The tokenizer monkey-patch is only needed while TITAN remote code is
    # constructing its local tokenizer.  Leaving it installed globally breaks
    # later Qwen-VL processor loading in the optional report reranker.
    PreTrainedTokenizerFast.from_pretrained = classmethod(original_tokenizer_loader)  # type: ignore[method-assign]
    print(f"[titan-online] loaded TITAN from {titan_dir} on {device}; conch={conch_bin.name}")
    return titan, conch, eval_transform, device


def conch_encode(conch, eval_transform, tiles: list[np.ndarray], device, batch: int):
    import torch
    from PIL import Image

    feats = []
    buf = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    def flush() -> None:
        if not buf:
            return
        x = torch.stack(buf).to(device, non_blocking=True)
        with torch.inference_mode(), autocast_ctx:
            out = conch(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats.append(out.float().cpu())
        buf.clear()

    for tile in tiles:
        buf.append(eval_transform(Image.fromarray(tile)))
        if len(buf) >= batch:
            flush()
    flush()
    if not feats:
        return None
    return torch.cat(feats, dim=0)


def _titan_grid_extent(coords: np.ndarray, patch_px: int) -> tuple[int, int]:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return 0, 0
    step = max(1, int(patch_px))
    offset = arr[:, :2].min(axis=0)
    grid = np.floor_divide(arr[:, :2] - offset, step)
    extent = grid.max(axis=0) - grid.min(axis=0) + 1
    return int(extent[0]), int(extent[1])


def _pad_sparse_titan_grid(patch_features, coords: np.ndarray, patch_px: int):
    """Avoid TITAN's sparse 1D-grid alibi edge case for scant tissue slides.

    Some very small biopsies produce valid CONCH features but all retained
    coordinates collapse into a 1-row or 1-column TITAN grid. TITAN's slide
    encoder can raise an IndexError in that geometry. For the TITAN slide
    embedding only, add one mean-feature pseudo-patch on the missing diagonal.
    The original tiles/coords are still returned to downstream H1/V2/MIL paths.
    """
    extent_x, extent_y = _titan_grid_extent(coords, patch_px)
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


def _extract_titan_core(wsi_path: str | Path):
    import tiffslide
    import torch

    start = time.time()
    titan, conch, eval_transform, device = load_titan_stack()
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
    conch_batch = _int_env(CONCH_BATCH_ENV, 16)
    scan_budget_s = _float_env(SCAN_BUDGET_ENV, 25.0)
    total_budget_s = _float_env(TOTAL_BUDGET_ENV, 50.0)

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
        raise RuntimeError("no readable tissue patches for TITAN")

    patch_features = conch_encode(conch, eval_transform, tiles, device, conch_batch)
    if patch_features is None or patch_features.numel() == 0:
        raise RuntimeError("CONCH produced no patch features")

    embed_features, embed_coords, padded_sparse = _pad_sparse_titan_grid(patch_features, coords, patch_px)
    if padded_sparse:
        before = _titan_grid_extent(coords, patch_px)
        after = _titan_grid_extent(embed_coords, patch_px)
        print(
            "[titan-online] padded sparse grid for TITAN "
            f"patches={len(tiles)} grid={before[0]}x{before[1]} -> {after[0]}x{after[1]}"
        )

    feats_t = embed_features.unsqueeze(0).to(device, non_blocking=True)
    coords_t = torch.from_numpy(embed_coords).unsqueeze(0).to(device, non_blocking=True)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        embedding = titan.encode_slide_from_patch_features(feats_t, coords_t, int(patch_px))
    embedding_np = embedding.squeeze().float().cpu().numpy().astype(np.float32)
    elapsed = time.time() - start
    print(
        "[titan-online] "
        f"patches={len(tiles)} scanned={scanned} patch_px={patch_px} "
        f"mpp={mpp if mpp is not None else 0:.4g} elapsed={elapsed:.1f}s"
    )
    if total_budget_s > 0.0 and elapsed > total_budget_s:
        print(
            f"[titan-online] WARNING: extract elapsed={elapsed:.1f}s "
            f"exceeded total budget={total_budget_s:.1f}s"
        )
    result = {
        "slide_embedding": embedding_np,
        "patch_features": patch_features.float().cpu().numpy().astype(np.float32),
        "coords": coords.astype(np.int64),
        "n_patches": np.asarray([len(tiles)], dtype=np.int32),
        "patch_px": np.asarray([patch_px], dtype=np.int32),
        "scanned": np.asarray([scanned], dtype=np.int32),
    }
    return result, tiles, coords, int(patch_px)


def extract_titan_slide_embedding(wsi_path: str | Path) -> dict[str, np.ndarray]:
    """Online TITAN slide embedding (CONCH patches -> TITAN). v12 entry point."""
    result, _tiles, _coords, _patch_px = _extract_titan_core(wsi_path)
    return result


def extract_titan_with_tiles(
    wsi_path: str | Path,
):
    """Like ``extract_titan_slide_embedding`` but also returns the scanned tiles.

    Returns ``(result_dict, tiles, coords, patch_px)``. The tile *scan* is the
    disk-I/O heavy step and is identical for the TITAN and H-optimus encoders, so
    the dual-encoder v14 pipeline reuses these tiles to avoid a second slide scan
    (see ``reg2_hoptimus_online.extract_from_tiles``).
    """
    return _extract_titan_core(wsi_path)
