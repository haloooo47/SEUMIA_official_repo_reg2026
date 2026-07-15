#!/usr/bin/env python3
"""Build a clean, NON-CIRCULAR labeled ROI tissue/background dataset for the
Metric-B (interface-0) learned gate.

Why non-circular:
  * Labels come from a MULTI-SIGNAL texture/structure referee (Laplacian energy,
    dark-nuclei fraction, strong-stain fraction) that is ORTHOGONAL to the
    color/optical-density gate (`is_background_like`) we are trying to replace.
  * Crucially the referee labels ADIPOSE / pale loose stroma (high texture, LOW
    optical density) as TISSUE -- exactly the population the OD gate false-refuses.
  * Splits are BY SLIDE (no slide appears in two splits) to avoid leakage.

Tiles: 512px level-0 crops read via tiffslide.read_region (fast; NOT get_thumbnail).
Stored as JPEG (quality 92) so the CNN/CONCH trainers never re-read WSIs.

Referee (per 512px tile, computed on the RAW crop):
  tex  = Laplacian variance of grayscale (structure / focus / edges)
  dark = fraction of pixels with gray < 140 (nuclei / dense material)
  stf  = strong_tissue_fraction from summarize_rgb_array (stained tissue pixels)

  label = TISSUE      if  tex>=TEX_HI  OR  dark>=DARK_HI  OR  stf>=STRONG_HI
          BACKGROUND  if  tex<=TEX_LO  AND dark<=DARK_LO  AND stf<=STRONG_LO
          SCANT       otherwise (a small amount of structure -> uncertain)

This keeps adipose (high tex, low OD) firmly in TISSUE while still catching
white glass AND faint-tint / edge-haze / blurred-blank as BACKGROUND.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_DIR / "submission" / "algorithm_submission_template"
sys.path.insert(0, str(TEMPLATE_DIR))
from src.reg2_pipeline import summarize_rgb_array  # noqa: E402

SLIDE_DIRS = [
    "train",
    "train_revised_20260519",
    "train_revised_20260527",
    "train_revised_20260604",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/reg2026"))
    p.add_argument("--out", type=Path, default=Path("runs/roi_tissue_clf_data"))
    p.add_argument("--num-slides", type=int, default=380)
    p.add_argument("--tiles-per-slide", type=int, default=44)
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--store-px", type=int, default=512, help="JPEG tile side stored to disk")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--seed", type=int, default=20260607)
    p.add_argument("--val-frac", type=float, default=0.16)
    p.add_argument("--test-frac", type=float, default=0.16)
    p.add_argument("--workers", type=int, default=24)
    # referee thresholds (tissue OR / background AND)
    p.add_argument("--tex-hi", type=float, default=120.0)
    p.add_argument("--tex-lo", type=float, default=22.0)
    p.add_argument("--dark-hi", type=float, default=0.02)
    p.add_argument("--dark-lo", type=float, default=0.006)
    p.add_argument("--strong-hi", type=float, default=0.05)
    p.add_argument("--strong-lo", type=float, default=0.012)
    return p.parse_args()


def laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = (-4.0 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    lap = lap[1:-1, 1:-1]
    return float(lap.var())


def referee_label(tex, dark, stf, t) -> str:
    if tex >= t["tex_hi"] or dark >= t["dark_hi"] or stf >= t["strong_hi"]:
        return "tissue"
    if tex <= t["tex_lo"] and dark <= t["dark_lo"] and stf <= t["strong_lo"]:
        return "background"
    return "scant"


def process_slide(task) -> list[dict]:
    sp, split, n_tiles, patch, store_px, jpeg_q, seed, thresh, tiles_root = task
    import tiffslide
    rng = random.Random(hash((sp, seed)) & 0xFFFFFFFF)
    out_rows: list[dict] = []
    stem = Path(sp).stem
    try:
        slide = tiffslide.TiffSlide(sp)
        W, H = slide.level_dimensions[0]
    except Exception:
        return out_rows
    if H < patch or W < patch:
        slide.close()
        return out_rows
    split_dir = Path(tiles_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(n_tiles):
        x = rng.randint(0, W - patch)
        y = rng.randint(0, H - patch)
        try:
            crop = np.asarray(slide.read_region((x, y), 0, (patch, patch)).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        if crop.shape[:2] != (patch, patch):
            continue
        s = summarize_rgb_array(crop)
        gray = crop.mean(axis=-1)
        tex = laplacian_var(gray)
        dark = float((gray < 140.0).mean())
        stf = float(s["strong_tissue_fraction"])
        label = referee_label(tex, dark, stf, thresh)
        fname = f"{stem}_{x}_{y}.jpg"
        fpath = split_dir / fname
        img = Image.fromarray(crop)
        if store_px != patch:
            img = img.resize((store_px, store_px), Image.Resampling.BILINEAR)
        try:
            img.save(fpath, "JPEG", quality=jpeg_q)
        except Exception:
            continue
        out_rows.append({
            "path": str(fpath),
            "slide": stem,
            "split": split,
            "x": int(x), "y": int(y),
            "label": label,
            "tex": tex,
            "dark_frac": dark,
            "strong_tissue_fraction": stf,
            "tissue_fraction": float(s["tissue_fraction"]),
            "mean_optical_density": float(s["mean_optical_density"]),
            "blank_fraction": float(s["blank_fraction"]),
            "background_confidence": float(s["background_confidence"]),
            "mean_saturation": float(s["mean_saturation"]),
        })
    slide.close()
    return out_rows


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tiles_root = args.out / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    slides: list[Path] = []
    for sub in SLIDE_DIRS:
        d = args.dataset_dir / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.tiff"):
            if Path(str(p) + ".aria2").exists():
                continue
            slides.append(p)
    rng.shuffle(slides)
    slides = slides[: args.num_slides]
    n = len(slides)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    val_slides = slides[:n_val]
    test_slides = slides[n_val:n_val + n_test]
    train_slides = slides[n_val + n_test:]
    split_of = {}
    for sp in train_slides:
        split_of[sp] = "train"
    for sp in val_slides:
        split_of[sp] = "val"
    for sp in test_slides:
        split_of[sp] = "test"
    print(f"[build] slides total={n} train={len(train_slides)} val={len(val_slides)} test={len(test_slides)}", flush=True)

    thresh = {
        "tex_hi": args.tex_hi, "tex_lo": args.tex_lo,
        "dark_hi": args.dark_hi, "dark_lo": args.dark_lo,
        "strong_hi": args.strong_hi, "strong_lo": args.strong_lo,
    }
    tasks = [
        (str(sp), split_of[sp], args.tiles_per_slide, args.patch, args.store_px,
         args.jpeg_quality, args.seed, thresh, str(tiles_root))
        for sp in slides
    ]

    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_slide, t) for t in tasks]
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 25 == 0:
                print(f"[build] processed {done}/{n} slides, rois={len(rows)}", flush=True)

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def counts(split):
        sub = [r for r in rows if r["split"] == split]
        c = {"tissue": 0, "scant": 0, "background": 0}
        for r in sub:
            c[r["label"]] += 1
        return {"n": len(sub), **c, "slides": len({r["slide"] for r in sub})}

    stats = {
        "seed": args.seed,
        "n_rois": len(rows),
        "thresholds": thresh,
        "tiles_per_slide": args.tiles_per_slide,
        "patch": args.patch,
        "store_px": args.store_px,
        "splits": {s: counts(s) for s in ("train", "val", "test")},
        "slide_overlap_check": {
            "train_val": len({r["slide"] for r in rows if r["split"] == "train"} &
                             {r["slide"] for r in rows if r["split"] == "val"}),
            "train_test": len({r["slide"] for r in rows if r["split"] == "train"} &
                              {r["slide"] for r in rows if r["split"] == "test"}),
            "val_test": len({r["slide"] for r in rows if r["split"] == "val"} &
                            {r["slide"] for r in rows if r["split"] == "test"}),
        },
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    # Save example tiles per class (montage dirs) for human eyeball verification.
    ex_root = args.out / "examples"
    for label in ("tissue", "scant", "background"):
        d = ex_root / label
        d.mkdir(parents=True, exist_ok=True)
        sub = [r for r in rows if r["label"] == label]
        rng.shuffle(sub)
        for r in sub[:24]:
            try:
                im = Image.open(r["path"]).convert("RGB")
                tag = f"tex{int(r['tex'])}_dark{r['dark_frac']:.3f}_od{r['mean_optical_density']:.2f}_stf{r['strong_tissue_fraction']:.3f}"
                im.save(d / f"{r['slide']}_{r['x']}_{r['y']}__{tag}.jpg", "JPEG", quality=90)
            except Exception:
                continue
    # adipose-candidate examples: high tex, low OD, labeled tissue
    adi = [r for r in rows if r["label"] == "tissue" and r["mean_optical_density"] < 0.30 and r["tex"] >= args.tex_hi]
    rng.shuffle(adi)
    adi_dir = ex_root / "adipose_candidates"
    adi_dir.mkdir(parents=True, exist_ok=True)
    for r in adi[:24]:
        try:
            im = Image.open(r["path"]).convert("RGB")
            tag = f"tex{int(r['tex'])}_dark{r['dark_frac']:.3f}_od{r['mean_optical_density']:.2f}"
            im.save(adi_dir / f"{r['slide']}_{r['x']}_{r['y']}__{tag}.jpg", "JPEG", quality=90)
        except Exception:
            continue
    print(f"[build] examples -> {ex_root} ; adipose_candidates n={len(adi)}")
    print(f"[build] manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
