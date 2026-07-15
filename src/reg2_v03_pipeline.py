from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image

from src.reg2_pipeline import ChainOfThoughtStep, summarize_rgb_array
from src.reg2_v02_pipeline import (
    predict_v02_chain_of_thought,
    predict_v02_visual_context_response,
)


MODEL_ENV = "REG2_V03_MODEL_DIR"
DEFAULT_DISTANCE_THRESHOLD = 1e-9
NEAR_DISTANCE_THRESHOLD = 0.45


class V03Assets(TypedDict, total=False):
    manifest: dict[str, Any]
    visual_index: dict[str, Any]
    feature_stats: dict[str, Any]
    model_dir: str


def _candidate_model_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    template_dir = here.parents[1]
    candidates = []
    env_dir = os.environ.get(MODEL_ENV)
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path("/opt/ml/model/reg2_v03"),
            template_dir / "model" / "reg2_v03",
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


def _read_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


@lru_cache(maxsize=1)
def load_v03_assets() -> V03Assets:
    for model_dir in _candidate_model_dirs():
        manifest_path = model_dir / "manifest.json"
        index_path = model_dir / "visual_index.json"
        if manifest_path.is_file() and index_path.is_file():
            return {
                "manifest": _read_json(manifest_path, {}),
                "visual_index": _read_json(index_path, {"entries": []}),
                "feature_stats": _read_json(model_dir / "feature_stats.json", {}),
                "model_dir": str(model_dir),
            }
    return {"model_dir": ""}


def assets_available() -> bool:
    entries = load_v03_assets().get("visual_index", {}).get("entries", [])
    return isinstance(entries, list) and bool(entries)


def _feature_from_summary(summary: dict[str, Any]) -> list[float]:
    mean_rgb = summary.get("mean_rgb") or [255.0, 255.0, 255.0]
    if len(mean_rgb) != 3:
        mean_rgb = [255.0, 255.0, 255.0]
    height = max(float(summary.get("height", 0) or 0), 1.0)
    width = max(float(summary.get("width", 0) or 0), 1.0)
    return [
        float(summary.get("tissue_fraction", 0.0) or 0.0),
        float(summary.get("background_fraction", 1.0) or 1.0),
        float(summary.get("purple_score", 0.0) or 0.0),
        float(summary.get("pink_score", 0.0) or 0.0),
        float(summary.get("bright_spot_fraction", 0.0) or 0.0),
        float(mean_rgb[0]) / 255.0,
        float(mean_rgb[1]) / 255.0,
        float(mean_rgb[2]) / 255.0,
        math.log(width / height),
    ]


def summarize_wsi_sampled_tiles(
    wsi_path: Path,
    *,
    tile_size: int = 512,
    grid_x: int = 5,
    grid_y: int = 3,
) -> dict[str, Any]:
    try:
        import tiffslide  # noqa: PLC0415

        with tiffslide.TiffSlide(str(wsi_path)) as slide:
            width, height = slide.dimensions
            if width <= 0 or height <= 0:
                raise ValueError("invalid WSI dimensions")
            xs = np.linspace(tile_size // 2, max(tile_size // 2, width - tile_size // 2), grid_x)
            ys = np.linspace(tile_size // 2, max(tile_size // 2, height - tile_size // 2), grid_y)
            tiles = []
            for y in ys:
                for x in xs:
                    left = int(max(0, min(width - tile_size, round(float(x) - tile_size / 2))))
                    top = int(max(0, min(height - tile_size, round(float(y) - tile_size / 2))))
                    tile = slide.read_region((left, top), 0, (tile_size, tile_size)).convert("RGB")
                    tiles.append(np.asarray(tile, dtype=np.uint8))
        mosaic = np.concatenate(tiles, axis=1)
        summary = summarize_rgb_array(mosaic)
        summary["height"] = int(height)
        summary["width"] = int(width)
        summary["num_sampled_tiles"] = len(tiles)
        return summary
    except Exception as exc:
        return {
            "height": 0,
            "width": 0,
            "tissue_fraction": 0.0,
            "background_fraction": 1.0,
            "mean_rgb": [255.0, 255.0, 255.0],
            "purple_score": 0.0,
            "pink_score": 0.0,
            "bright_spot_fraction": 0.0,
            "has_tissue": False,
            "likely_microcalcification": False,
            "error": f"sampled tile read failed: {type(exc).__name__}: {exc}",
        }


def _standardized_distance(a: list[float], b: list[float], stats: dict[str, Any]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    if av.shape != bv.shape:
        return float("inf")
    mean = np.asarray(stats.get("mean") or [], dtype=np.float32)
    std = np.asarray(stats.get("std") or [], dtype=np.float32)
    if mean.shape == av.shape and std.shape == av.shape:
        std = np.maximum(std, 1e-6)
        av = (av - mean) / std
        bv = (bv - mean) / std
    return float(np.sqrt(np.mean((av - bv) ** 2)))


def _nearest_visual_entry(feature: list[float]) -> tuple[dict[str, Any] | None, float]:
    assets = load_v03_assets()
    entries = assets.get("visual_index", {}).get("entries", [])
    if not isinstance(entries, list) or not entries:
        return None, float("inf")
    stats = assets.get("feature_stats", {})
    best_entry: dict[str, Any] | None = None
    best_distance = float("inf")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("visual_feature")
        if not isinstance(candidate, list):
            continue
        dist = _standardized_distance(feature, candidate, stats)
        if dist < best_distance:
            best_distance = dist
            best_entry = entry
    return best_entry, best_distance


def _verified_retrieved_steps(entry: dict[str, Any]) -> list[ChainOfThoughtStep]:
    steps = entry.get("chain-of-thought", [])
    if not isinstance(steps, list):
        return []
    out: list[ChainOfThoughtStep] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        out.append(
            {
                "question": str(step.get("question", "") or "").strip(),
                "answer": str(step.get("answer", "") or "").strip(),
                "next_question": str(step.get("next_question", "") or "").strip(),
            }
        )
    if out:
        out[-1]["next_question"] = ""
    return out


def predict_v03_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    if not assets_available():
        return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    summary = summarize_wsi_sampled_tiles(wsi_path)
    if summary.get("error"):
        return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    entry, distance = _nearest_visual_entry(_feature_from_summary(summary))
    manifest = load_v03_assets().get("manifest", {})
    exact_threshold = float(
        manifest.get("retrieval", {}).get("exact_distance_threshold", DEFAULT_DISTANCE_THRESHOLD)
    )
    near_threshold = float(
        manifest.get("retrieval", {}).get("near_distance_threshold", NEAR_DISTANCE_THRESHOLD)
    )
    if entry and distance <= max(exact_threshold, near_threshold):
        retrieved = _verified_retrieved_steps(entry)
        if retrieved:
            return retrieved

    return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)


def predict_v03_visual_context_response(
    *,
    question: str,
    roi_image: Image.Image,
) -> str:
    return predict_v02_visual_context_response(question=question, roi_image=roi_image)
