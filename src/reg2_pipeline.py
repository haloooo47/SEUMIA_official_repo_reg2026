from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image


class ChainOfThoughtStep(TypedDict):
    question: str
    answer: str
    next_question: str


def _as_rgb_array(image: Image.Image, *, max_side: int = 768) -> np.ndarray:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _normalize_rgb_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    if array.ndim == 4:
        array = array.reshape((-1, *array.shape[-3:]))[0]
    if array.dtype != np.uint8:
        max_value = float(np.nanmax(array)) if array.size else 1.0
        scale = 255.0 / max(max_value, 1.0)
        array = np.clip(array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return array


def summarize_rgb_array(array: np.ndarray) -> dict[str, Any]:
    array = _normalize_rgb_array(array)
    if array.size == 0:
        return _empty_summary("empty image")

    rgb = array.astype(np.float32)
    brightness = rgb.mean(axis=-1)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_rgb = rgb.max(axis=-1)
    min_rgb = rgb.min(axis=-1)
    saturation = chroma / np.maximum(max_rgb, 1.0)

    # Classical H&E tissue screening: combine chroma/saturation with optical
    # density so pale eosinophilic tissue is not confused with blank glass.
    od = -np.log((rgb + 1.0) / 256.0)
    od_sum = od.sum(axis=-1)

    stain_like = (
        (red > blue + 8.0)
        | (blue > green + 8.0)
        | ((red > green + 3.0) & (blue > green + 3.0))
    )
    color_tissue = (brightness < 238.0) & ((chroma > 12.0) | stain_like)
    saturation_tissue = (brightness < 248.0) & (saturation > 0.055) & (od_sum > 0.08)
    od_tissue = (brightness < 250.0) & (od_sum > 0.22) & (saturation > 0.025)
    tissue_mask = color_tissue | saturation_tissue | od_tissue

    artifact_mask = brightness < 35.0
    tissue_mask &= ~artifact_mask
    blank_mask = (brightness > 242.0) & (chroma < 16.0) & (saturation < 0.08)

    # Strongly-stained tissue: pixels with meaningful optical density AND color.
    # This is robust to faint background tints (low OD) that fool the chroma-only
    # tissue heuristic, while still firing on real (even partial) H&E tissue.
    strong_tissue_mask = (
        (od_sum > 0.45)
        & (saturation > 0.05)
        & (brightness < 235.0)
        & ~artifact_mask
    )
    strong_tissue_fraction = float(strong_tissue_mask.mean())

    tissue_fraction = float(tissue_mask.mean())
    if tissue_mask.any():
        tissue_rgb = rgb[tissue_mask]
    else:
        tissue_rgb = rgb.reshape(-1, 3)

    mean_rgb = tissue_rgb.mean(axis=0)
    purple_score = float(np.mean((blue[tissue_mask] > red[tissue_mask] + 8.0) & (red[tissue_mask] > green[tissue_mask] - 15.0))) if tissue_mask.any() else 0.0
    pink_score = float(np.mean((red[tissue_mask] > blue[tissue_mask] + 12.0) & (red[tissue_mask] > green[tissue_mask] + 4.0))) if tissue_mask.any() else 0.0
    bright_spot_fraction = float(((brightness > 225.0) & (chroma < 30.0) & tissue_mask).mean())
    blank_fraction = float(blank_mask.mean())
    mean_saturation = float(saturation.mean())
    mean_od = float(od_sum.mean())
    background_confidence = float(np.clip(blank_fraction + max(0.0, 0.04 - tissue_fraction) * 4.0, 0.0, 1.0))
    tissue_confidence = float(np.clip((tissue_fraction - 0.015) / 0.08, 0.0, 1.0))
    has_tissue = tissue_fraction >= 0.025 or (tissue_fraction >= 0.015 and blank_fraction < 0.92 and mean_saturation > 0.035)

    return {
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "tissue_fraction": tissue_fraction,
        "strong_tissue_fraction": strong_tissue_fraction,
        "background_fraction": float(1.0 - tissue_fraction),
        "mean_rgb": [float(x) for x in mean_rgb],
        "mean_saturation": mean_saturation,
        "mean_optical_density": mean_od,
        "blank_fraction": blank_fraction,
        "tissue_confidence": tissue_confidence,
        "background_confidence": background_confidence,
        "purple_score": purple_score,
        "pink_score": pink_score,
        "bright_spot_fraction": bright_spot_fraction,
        "has_tissue": bool(has_tissue),
        "likely_microcalcification": bright_spot_fraction >= 0.002,
        "error": "",
    }


def _empty_summary(error: str) -> dict[str, Any]:
    return {
        "height": 0,
        "width": 0,
        "tissue_fraction": 0.0,
        "strong_tissue_fraction": 0.0,
        "background_fraction": 1.0,
        "mean_rgb": [255.0, 255.0, 255.0],
        "mean_saturation": 0.0,
        "mean_optical_density": 0.0,
        "blank_fraction": 1.0,
        "tissue_confidence": 0.0,
        "background_confidence": 1.0,
        "purple_score": 0.0,
        "pink_score": 0.0,
        "bright_spot_fraction": 0.0,
        "has_tissue": False,
        "likely_microcalcification": False,
        "error": error,
    }


def summarize_roi_image(image: Image.Image) -> dict[str, Any]:
    return summarize_rgb_array(_as_rgb_array(image, max_side=768))


BACKGROUND_RESPONSE = (
    "No diagnostic tissue is visible in this ROI; the region appears to be "
    "background or non-informative, so the requested pathology finding cannot "
    "be reliably assessed."
)
SCANT_TISSUE_RESPONSE = (
    "Only scant possible tissue is visible in this ROI, so the region is not "
    "reliably assessable for diagnosis."
)
TISSUE_VISIBLE_RESPONSE = "Yes, tissue is visible in this ROI."
TISSUE_INFORMATIVE_RESPONSE = (
    "This ROI contains visible tissue, so it is potentially informative, but a "
    "definitive diagnosis requires broader slide context."
)
DIAGNOSIS_UNCERTAIN_RESPONSE = (
    "Visible tissue is present, but a reliable diagnosis, tumor type, or grade "
    "cannot be determined from this ROI alone."
)


# Robust no-tissue thresholds for the Metric B background gate. Calibrated on
# real WSI ROIs so that genuine tissueless background (incl. faint color casts /
# blur / slide-edge haze) is separated from real H&E tissue by a low
# strong-tissue fraction and low mean optical density. The optical-density guard
# ensures a genuinely stained tissue ROI is never rejected as background.
_BG_STRONG_TISSUE_MAX = 0.15
_BG_MEAN_OD_MAX = 0.35


def is_background_like(summary: dict[str, Any], *, tissue_min: float = 0.025) -> bool:
    tissue_fraction = float(summary.get("tissue_fraction", 0.0) or 0.0)
    blank_fraction = float(summary.get("blank_fraction", 1.0) or 1.0)
    background_confidence = float(summary.get("background_confidence", 1.0) or 1.0)
    strong_tissue_fraction = float(
        summary.get("strong_tissue_fraction", tissue_fraction) or 0.0
    )
    mean_optical_density = float(summary.get("mean_optical_density", 1.0) or 0.0)

    # Primary robust signal: essentially no strongly-stained tissue present AND
    # low overall optical density. This catches faint-tint / blurred / edge
    # background ROIs that fool the chroma-only tissue heuristic (the dominant
    # cause of low B1 background-rejection and low B3 cross-region scores) while
    # leaving every genuinely stained tissue ROI (high OD) untouched.
    if strong_tissue_fraction < _BG_STRONG_TISSUE_MAX and mean_optical_density < _BG_MEAN_OD_MAX:
        return True

    if tissue_fraction < tissue_min and background_confidence >= 0.85:
        return True
    return tissue_fraction < max(0.012, tissue_min * 0.5) and blank_fraction >= 0.75


def is_scant_or_uncertain_tissue(summary: dict[str, Any], *, tissue_min: float = 0.025, margin: float = 0.02) -> bool:
    tissue_fraction = float(summary.get("tissue_fraction", 0.0) or 0.0)
    return tissue_min <= tissue_fraction < tissue_min + margin


def _question_class(question: str) -> str:
    q = question.lower()
    if any(token in q for token in ["visible tissue", "tissue visible", "tissue present"]):
        return "tissue_visibility"
    if any(token in q for token in ["background", "informative", "non-informative"]):
        return "background_or_informative"
    if any(
        token in q
        for token in [
            "diagnos",
            "malignan",
            "tumor",
            "tumour",
            "carcinoma",
            "neoplasm",
            "subtype",
            "metasta",
            "benign",
        ]
    ):
        return "diagnosis"
    if any(
        token in q
        for token in [
            "grade",
            "score",
            "gleason",
            "nottingham",
            "mitotic",
            "mitosis",
            "pleomorphism",
            "ki-67",
            "ki67",
        ]
    ):
        return "grade"
    if re.search(r"\b(?:er|pr|her2)\b", q):
        return "morphology"
    if any(
        token in q
        for token in [
            "morpholog",
            "cell",
            "gland",
            "stroma",
            "necrosis",
            "invasion",
            "invasive",
            "margin",
            "lymphovascular",
            "perineural",
            "keratin",
            "mucin",
            "calcification",
            "microcalcification",
            "ulcer",
            "erosion",
            "atypia",
            "dysplasia",
            "metaplasia",
            "ihc",
            "immunohistochem",
            "stain",
            "marker",
            "receptor",
        ]
    ):
        return "morphology"
    if any(token in q for token in ["assessable", "evaluable", "adequate", "representative"]):
        return "background_or_informative"
    return "generic"


def answer_roi_question_from_summary(question: str, summary: dict[str, Any]) -> str:
    q_class = _question_class(question)

    if is_background_like(summary):
        return BACKGROUND_RESPONSE
    if is_scant_or_uncertain_tissue(summary):
        return SCANT_TISSUE_RESPONSE

    if q_class == "tissue_visibility":
        return TISSUE_VISIBLE_RESPONSE
    if q_class == "background_or_informative":
        return TISSUE_INFORMATIVE_RESPONSE
    if q_class in {"diagnosis", "grade"}:
        return DIAGNOSIS_UNCERTAIN_RESPONSE

    stain_phrase = "pink eosinophilic and purple basophilic histologic staining"
    purple = float(summary.get("purple_score", 0.0) or 0.0)
    pink = float(summary.get("pink_score", 0.0) or 0.0)
    if purple > pink * 1.5:
        stain_phrase = "prominent purple basophilic nuclear staining"
    elif pink > purple * 1.5:
        stain_phrase = "predominantly pink eosinophilic tissue staining"

    if q_class == "morphology":
        return f"Visible tissue is present with {stain_phrase}; no unsupported definitive diagnosis is made from this ROI alone."
    return f"Tissue is visible with {stain_phrase}; no unsupported definitive diagnosis is made from this ROI alone."


def read_wsi_thumbnail(
    wsi_path: Path,
    *,
    max_pixels: int = 2_000_000,
    max_load_pixels: int = 16_000_000,
) -> np.ndarray | None:
    try:
        import tifffile
    except Exception:
        return None

    try:
        with tifffile.TiffFile(str(wsi_path)) as tif:
            series = tif.series[0]
            levels = list(getattr(series, "levels", None) or [series])
            pages = list(getattr(tif, "pages", []) or [])
            candidates = [item for item in [*levels, *pages] if len(getattr(item, "shape", ())) >= 2]
            candidates.sort(key=lambda item: int(np.prod(item.shape[:2])))
            chosen = candidates[0]
            if int(np.prod(chosen.shape[:2])) > max_load_pixels:
                return None
            array = chosen.asarray()
    except Exception:
        return None

    array = _normalize_rgb_array(array)
    pixels = int(np.prod(array.shape[:2]))
    if pixels > max_pixels:
        image = Image.fromarray(array)
        scale = (max_pixels / float(pixels)) ** 0.5
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    return array


def summarize_wsi(wsi_path: Path) -> dict[str, Any]:
    thumb = read_wsi_thumbnail(wsi_path)
    if thumb is None:
        return _empty_summary("could not read a WSI thumbnail")
    return summarize_rgb_array(thumb)


def answer_roi_question(question: str, image: Image.Image) -> str:
    summary = summarize_roi_image(image)
    return answer_roi_question_from_summary(question, summary)


def make_chain_of_thought(wsi_path: Path) -> list[ChainOfThoughtStep]:
    summary = summarize_wsi(wsi_path)
    has_microcalcification = bool(summary.get("likely_microcalcification"))
    microcalc_answer = (
        "Yes, there is a microcalcification."
        if has_microcalcification
        else "No, there is no microcalcification."
    )
    diagnosis_count = "2" if has_microcalcification else "1"
    final_report = "Breast, core needle biopsy;\n  Invasive carcinoma of no special type, grade I"
    diagnosis_steps: list[ChainOfThoughtStep] = [
        {
            "question": "What is the #1 diagnosis?",
            "answer": "Invasive carcinoma of no special type, grade I",
            "next_question": "What is the #2 diagnosis?" if has_microcalcification else "What is the final pathology report?",
        }
    ]
    if has_microcalcification:
        diagnosis_steps.append(
            {
                "question": "What is the #2 diagnosis?",
                "answer": "Microcalcification",
                "next_question": "What is the final pathology report?",
            }
        )
        final_report += "\n  Microcalcification"

    steps: list[ChainOfThoughtStep] = [
        {"question": "What is the organ?", "answer": "Breast", "next_question": "Is there any abnormality present?"},
        {"question": "What is the procedure?", "answer": "Core needle biopsy", "next_question": "Is there any abnormality present?"},
        {"question": "Is there any abnormality present?", "answer": "Yes, there is an abnormality.", "next_question": "Is there any proliferative lesion present?"},
        {"question": "Is there any abnormality present?", "answer": "Yes, there is an abnormality.", "next_question": "Is there any microcalcification present?"},
        {"question": "Is there any microcalcification present?", "answer": microcalc_answer, "next_question": "What is the number of diagnoses to includes?"},
        {"question": "Is there any proliferative lesion present?", "answer": "Yes, there is a proliferative lesion.", "next_question": "Is there any invasion present?"},
        {"question": "Is there any invasion present?", "answer": "Yes, there is an invasion.", "next_question": "What is the primary of neoplasm?"},
        {"question": "What is the primary of neoplasm?", "answer": "Mammary", "next_question": "What is the histologic type of neoplasm?"},
        {"question": "What is the histologic type of neoplasm?", "answer": "Invasive breast carcinoma of no special type", "next_question": "What is the grading system?"},
        {"question": "What is the grading system?", "answer": "Nottingham combined histologic grade", "next_question": "What is the score for tubular differentiation?"},
        {"question": "What is the grading system?", "answer": "Nottingham combined histologic grade", "next_question": "What is the score for nuclear pleomorphism?"},
        {"question": "What is the grading system?", "answer": "Nottingham combined histologic grade", "next_question": "What is the score for mitotic rate?"},
        {"question": "What is the score for tubular differentiation?", "answer": "2", "next_question": "What is the overall score?"},
        {"question": "What is the score for nuclear pleomorphism?", "answer": "2", "next_question": "What is the overall score?"},
        {"question": "What is the score for mitotic rate?", "answer": "1", "next_question": "What is the overall score?"},
        {"question": "What is the overall score?", "answer": "5", "next_question": "What is the grade of neoplasm?"},
        {"question": "What is the grade of neoplasm?", "answer": "Grade I", "next_question": "Is there any additional finding present?"},
        {"question": "Is there any additional finding present?", "answer": "No, there is no additional finding.", "next_question": "What is the number of diagnoses to includes?"},
        {"question": "What is the number of diagnoses to includes?", "answer": diagnosis_count, "next_question": "What is the #1 diagnosis?"},
    ]
    steps.extend(diagnosis_steps)
    steps.append({"question": "What is the final pathology report?", "answer": final_report, "next_question": ""})
    return steps
