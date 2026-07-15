"""Online Virchow2 patch encoder for the v15 fused report-classifier path.

v15 reuses the SAME tissue tiles already scanned for TITAN/H-optimus-1 and
re-encodes them with a frozen Virchow2 (timm ViT-H/14, SwiGLU, reg tokens=4,
ImageNet norm). The per-tile feature is concat[CLS(1280), mean(patch tokens)(1280)]
= 2560-d, matching ``scripts/extract_virchow2_full.py`` (train-serve parity).

Only ``pooled_mean`` (2560-d, mean over tiles) is needed online: it is the third
block of the fused report feature ``concat[TITAN(768), H1(1536), V2(2560)]`` that
feeds the whole-report classifier used by the v15 report selector. Virchow2 is
NOT used for the dx ensemble (that stays the proven v14 dual path).
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_ENV = "REG2_VIRCHOW2_DIR"
BATCH_ENV = "REG2_VIRCHOW2_BATCH"
TILE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def find_virchow2_dir() -> Path:
    env_dir = os.environ.get(MODEL_ENV)
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/opt/ml/model/Virchow2"),
            _template_dir() / "model" / "Virchow2",
        ]
    )
    for cand in candidates:
        if (cand / "model.safetensors").is_file():
            return cand
    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Virchow2 model.safetensors not found; checked: {checked}")


@lru_cache(maxsize=1)
def load_virchow2():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import timm
    import torch
    from timm.layers import SwiGLUPacked
    from safetensors.torch import load_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    v2_dir = find_virchow2_dir()
    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        num_classes=0,
        img_size=TILE,
        init_values=1e-5,
        reg_tokens=4,
        mlp_ratio=5.3375,
        global_pool="",
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        dynamic_img_size=False,
    )
    sd = load_file(str(v2_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[v2-online] WARNING missing {len(missing)} keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[v2-online] WARNING unexpected {len(unexpected)} keys (e.g. {unexpected[:3]})")
    model = model.eval().to(device)
    print(f"[v2-online] loaded Virchow2 from {v2_dir} on {device}")
    return model, device


def _encode_tiles(model, device, tiles: list[np.ndarray], batch: int) -> np.ndarray:
    import torch
    from PIL import Image

    npref = model.num_prefix_tokens
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
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device, non_blocking=True)
        with torch.inference_mode(), autocast_ctx:
            o = model.forward_features(t)
            cls = o[:, 0]
            patch = o[:, npref:]
            concat = torch.cat([cls, patch.mean(1)], dim=-1)  # 2560
        feats.append(concat.float().cpu().numpy())
        buf.clear()

    for tile in tiles:
        if tile.shape[:2] != (TILE, TILE):
            tile = np.asarray(Image.fromarray(tile).resize((TILE, TILE), Image.BILINEAR), dtype=np.uint8)
        buf.append(tile)
        if len(buf) >= batch:
            flush()
    flush()
    if not feats:
        return np.zeros((0, 2560), np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_v2_features_from_tiles(tiles: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Return Virchow2 per-tile features and pooled mean over shared scanned tiles."""
    model, device = load_virchow2()
    batch = _int_env(BATCH_ENV, 16)
    feats = _encode_tiles(model, device, tiles, batch)
    if feats.shape[0] == 0:
        raise RuntimeError("Virchow2 produced no patch features")
    return {
        "patch_features_v2": feats.astype(np.float32, copy=False),
        "pooled_mean": feats.mean(axis=0).astype(np.float32),
    }


def extract_v2_pooled_from_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    """Return the 2560-d Virchow2 pooled-mean over the shared scanned tiles."""
    return extract_v2_features_from_tiles(tiles)["pooled_mean"]
