"""Score val predictions with the OFFICIAL evaluate_metrics (real KeywordEvaluator).

Reproduces evaluate.py's workflow scoring exactly: lexical MESS backend + REG25
final report (rouge/bleu + real keyword + PubMedBERT embedding). The Qwen judge is
NOT used for workflow (only for Metric B), so judge_llm=None matches the official
workflow path.

Run in the prepared official-eval env:
  python scripts/official_eval_val.py \
    --gt <gt_val.json> --pred <pred.json> [--out <summary.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "evaluation" / "official"
sys.path.insert(0, str(EVAL_DIR))

import evaluate_metrics as em  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--embedding-model",
                   default="models/pubmedbert-base-embeddings")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backend = em.DEFAULT_WORKFLOW_SEMANTIC_BACKEND
    print(f"[official] backend={backend} embedding={args.embedding_model}", flush=True)
    res = em.run_workflow_batch(
        ground_truth_paths=[args.gt],
        prediction_paths=[args.pred],
        semantic_backend=backend,
        embedding_model=str(args.embedding_model),
        judge_llm=None,  # workflow backend is lexical -> judge unused (matches evaluate.py)
        voting=em.DEFAULT_VOTING,
        strict_missing_predictions=em.DEFAULT_STRICT_MISSING_PREDICTIONS,
        merge_predictions=em.DEFAULT_MERGE_PREDICTIONS,
    )
    ga = res.get("global_average", res)
    workflow = em.extract_workflow_final_ranking_score(res)
    summary = {
        "num_cases": ga.get("num_prediction_files"),
        "workflow_final_ranking_score": workflow,
        "average_binary_path_validity": ga.get("average_binary_path_validity"),
        "average_edge_f1": ga.get("average_edge_f1"),
        "average_mess_nonfinal": ga.get("average_mess_nonfinal"),
        "average_final_report_score": ga.get("average_final_report_score"),
    }
    print("\n===== OFFICIAL WORKFLOW (Metric A) =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
