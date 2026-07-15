from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_v02_pipeline import predict_v02_chain_of_thought
from src.reg2_v03_pipeline import (
    _feature_from_summary,
    _standardized_distance,
    _verified_retrieved_steps,
    assets_available,
    load_v03_assets,
    predict_v03_visual_context_response,
    summarize_wsi_sampled_tiles,
)


EXACT_DISTANCE_THRESHOLD = 1e-9
ORGAN_DISTANCE_THRESHOLD = 0.62
ORGAN_MARGIN_THRESHOLD = 0.06
TOPK = 7

ORGAN_TO_V02_MODE = {
    "Bladder": "Urinary bladder",
    "Cervix": "Uterine cervix",
    "Breast": "Breast",
    "Colon": "Colon",
    "Lung": "Lung",
    "Prostate": "Prostate",
    "Stomach": "Stomach",
}


def _rank_visual_entries(feature: list[float], limit: int = TOPK) -> list[tuple[dict[str, Any], float]]:
    assets = load_v03_assets()
    entries = assets.get("visual_index", {}).get("entries", [])
    if not isinstance(entries, list) or not entries:
        return []
    stats = assets.get("feature_stats", {})
    ranked: list[tuple[dict[str, Any], float]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("visual_feature")
        if not isinstance(candidate, list):
            continue
        ranked.append((entry, _standardized_distance(feature, candidate, stats)))
    ranked.sort(key=lambda item: item[1])
    return ranked[:limit]


def _confident_organ_from_ranked(ranked: list[tuple[dict[str, Any], float]]) -> str:
    if not ranked:
        return ""
    best_entry, best_distance = ranked[0]
    if best_distance > ORGAN_DISTANCE_THRESHOLD:
        return ""
    second_distance = ranked[1][1] if len(ranked) > 1 else float("inf")
    margin = second_distance - best_distance
    organs = [str(entry.get("organ", "") or "") for entry, _ in ranked[:TOPK]]
    organ_counts = Counter(item for item in organs if item)
    if not organ_counts:
        return ""
    organ, count = organ_counts.most_common(1)[0]
    if count < max(3, TOPK // 2):
        return ""
    if margin < ORGAN_MARGIN_THRESHOLD and organ != str(best_entry.get("organ", "") or ""):
        return ""
    return ORGAN_TO_V02_MODE.get(organ, organ)


def predict_v10_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    if not assets_available():
        return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    summary = summarize_wsi_sampled_tiles(wsi_path)
    if summary.get("error"):
        return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    ranked = _rank_visual_entries(_feature_from_summary(summary), limit=TOPK)
    if ranked:
        best_entry, best_distance = ranked[0]
        if best_distance <= EXACT_DISTANCE_THRESHOLD:
            retrieved = _verified_retrieved_steps(best_entry)
            if retrieved:
                return retrieved

    organ = _confident_organ_from_ranked(ranked)
    if organ:
        return predict_v02_chain_of_thought(
            wsi_path=wsi_path,
            case_id=case_id,
            mode=f"organ_mode:{organ}",
        )

    return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)


def predict_v10_visual_context_response(
    *,
    question: str,
    roi_image: Image.Image,
) -> str:
    return predict_v03_visual_context_response(question=question, roi_image=roi_image)
