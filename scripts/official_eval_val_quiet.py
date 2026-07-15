#!/usr/bin/env python3
"""Quiet wrapper for official REG2026 validation workflow scoring.

``scripts/official_eval_val.py`` is faithful, but the upstream evaluator prints
per-case final-report diagnostics, which makes large runs noisy and slow to
inspect.  This wrapper redirects evaluator chatter to a log file and prints only
the Metric-A summary.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "evaluation" / "official"
sys.path.insert(0, str(EVAL_DIR))

import evaluate_metrics as em  # noqa: E402


def norm_id(raw: str) -> str:
    cid = str(raw or "").strip()
    for suffix in (".tiff", ".svs"):
        if cid.lower().endswith(suffix):
            return cid[: -len(suffix)]
    return cid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument(
        "--subset-to-pred",
        action="store_true",
        help="Evaluate only GT cases that appear in the prediction JSON.",
    )
    p.add_argument(
        "--embedding-model",
        default="models/pubmedbert-base-embeddings",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_path = args.log or args.out.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.gt.open(encoding="utf-8") as f:
            gt_obj = json.load(f)
        actual_num_cases = len(gt_obj) if isinstance(gt_obj, list) else None
    except Exception:
        gt_obj = None
        actual_num_cases = None

    gt_path = args.gt
    pred_path = args.pred
    if args.subset_to_pred:
        if not isinstance(gt_obj, list):
            raise ValueError("--subset-to-pred requires list-style GT JSON")
        pred_obj = json.loads(args.pred.read_text(encoding="utf-8"))
        if not isinstance(pred_obj, list):
            raise ValueError("--subset-to-pred requires list-style prediction JSON")
        pred_ids = {norm_id(case.get("id", "")) for case in pred_obj if isinstance(case, dict)}
        subset_gt = [case for case in gt_obj if norm_id(case.get("id", "")) in pred_ids]
        subset_dir = args.out.parent / "eval_subset"
        subset_dir.mkdir(parents=True, exist_ok=True)
        gt_path = subset_dir / "gt_val.json"
        pred_path = subset_dir / "pred.json"
        gt_path.write_text(json.dumps(subset_gt, ensure_ascii=False), encoding="utf-8")
        pred_path.write_text(json.dumps(pred_obj, ensure_ascii=False), encoding="utf-8")
        actual_num_cases = len(subset_gt)

    with log_path.open("w", encoding="utf-8") as log_f:
        with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
            res = em.run_workflow_batch(
                ground_truth_paths=[gt_path],
                prediction_paths=[pred_path],
                semantic_backend=em.DEFAULT_WORKFLOW_SEMANTIC_BACKEND,
                embedding_model=str(args.embedding_model),
                judge_llm=None,
                voting=em.DEFAULT_VOTING,
                strict_missing_predictions=em.DEFAULT_STRICT_MISSING_PREDICTIONS,
                merge_predictions=em.DEFAULT_MERGE_PREDICTIONS,
            )

    ga = res.get("global_average", res)
    summary = {
        "num_cases": actual_num_cases,
        "num_prediction_files": ga.get("num_prediction_files"),
        "workflow_final_ranking_score": em.extract_workflow_final_ranking_score(res),
        "average_binary_path_validity": ga.get("average_binary_path_validity"),
        "average_edge_f1": ga.get("average_edge_f1"),
        "average_mess_nonfinal": ga.get("average_mess_nonfinal"),
        "average_final_report_score": ga.get("average_final_report_score"),
        "log": str(log_path),
        "subset_to_pred": bool(args.subset_to_pred),
    }
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
