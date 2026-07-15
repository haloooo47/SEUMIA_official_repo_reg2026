#!/usr/bin/env python3
"""Text-only ceiling evaluation for the REG2026 answer + report calibrator.

This isolates the text side of Metric A (MESS = 0.25, final report = 0.40) by
assuming the workflow PATH and the clinical LABELS are correct (i.e. a perfect
step planner + perfect visual specialists). It measures how much of that 0.455
block the text calibrator can actually capture on a held-out validation split.

For every validation case we:
  * take the ground-truth path (sequence of canonical (question, next_question)
    edges) as the predicted path -> edge sets match, so BPV/Edge-F1 = 1.0;
  * derive a LabelContext from the case's own ground-truth labels (oracle);
  * render answers + final report with one of several configs;
  * score with the official ``evaluate_metrics`` logic.

Backends (no extra installs): MESS uses the official ``lexical`` backend; the
final report uses the official REG25 formula with the same ``SimpleKeywordEvaluator``
fallback the existing v0.2 fallback eval uses, plus a numpy-cosine PubMedBERT
embedder (avoids the missing sklearn dependency). MESS under the official
embedding/LLM backend would generally score >= lexical, so these are a
conservative lower bound.

Run with an env that has torch + transformers, e.g.:
  python scripts/eval_reg2026_text_calibrator.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_DIR / "local_eval" / "submission_evaluation_code"
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(EVAL_DIR))

from reg2_text_calibrator import (  # noqa: E402
    LabelContext,
    TextCalibrator,
    canonicalize_question,
    edge_key,
)

DEFAULT_ASSETS = Path("runs/calibrator_assets/v1")
DEFAULT_COT = Path("data/train_CoT_v01.json")
DEFAULT_OUT = Path("runs/text_calibrator_v1")
LOCAL_PUBMEDBERT = "models/pubmedbert-base-embeddings"

FINAL_REPORT_Q = "what is the final pathology report"
ORGAN_Q = "what is the organ"
PROCEDURE_Q = "what is the procedure"
HISTOLOGIC_TYPE_Q = "what is the histologic type of neoplasm"
GRADE_Q = "what is the grade of neoplasm"
BEHAVIOR_Q = "what is the behavior of neoplasm"
DIAGNOSIS_Q_RE = re.compile(r"^what is the #(\d+) diagnosis$")

STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "but", "by", "can", "cannot",
    "for", "from", "in", "is", "it", "no", "not", "of", "or", "the",
    "there", "this", "to", "visible", "with", "yes",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--cot", type=Path, default=DEFAULT_COT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--embedding-model", default=LOCAL_PUBMEDBERT)
    p.add_argument("--limit", type=int, default=0, help="Eval on first N val cases (0 = all).")
    p.add_argument("--configs", nargs="+",
                   default=["full_pipeline", "calibrator", "calibrator_no_field", "baseline_global"])
    return p.parse_args()


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_case_id(raw: str) -> str:
    cid = str(raw).strip()
    for suffix in (".svs", ".tiff"):
        if cid.lower().endswith(suffix):
            return cid[: -len(suffix)]
    return cid


def get_steps(case: dict[str, Any]) -> list[dict[str, str]]:
    raw = case.get("chain-of-thought") or case.get("chain_of_thought") or []
    steps = []
    if not isinstance(raw, list):
        return steps
    for step in raw:
        if not isinstance(step, dict):
            continue
        q = clean(step.get("question"))
        if not q:
            continue
        steps.append({"question": q, "answer": clean(step.get("answer")),
                      "next_question": clean(step.get("next_question"))})
    return steps


def context_from_steps(steps: list[dict[str, str]], organ_hint: str = "") -> LabelContext:
    organ = procedure = histologic_type = grade = behavior = ""
    diagnoses: dict[int, str] = {}
    for step in steps:
        cq = canonicalize_question(step["question"])
        a = step["answer"]
        if cq == ORGAN_Q and not organ:
            organ = a
        elif cq == PROCEDURE_Q and not procedure:
            procedure = a
        elif cq == HISTOLOGIC_TYPE_Q and not histologic_type:
            histologic_type = a
        elif cq == GRADE_Q and not grade:
            grade = a
        elif cq == BEHAVIOR_Q and not behavior:
            behavior = a
        else:
            m = DIAGNOSIS_Q_RE.match(cq)
            if m:
                diagnoses.setdefault(int(m.group(1)), a)
    return LabelContext(
        organ=organ or organ_hint,
        procedure=procedure,
        diagnoses=[diagnoses[k] for k in sorted(diagnoses)],
        histologic_type=histologic_type,
        grade=grade,
        behavior=behavior,
    )


def render_baseline_global(calib: TextCalibrator, steps: list[dict[str, str]]) -> list[dict[str, str]]:
    """Naive prior: global-mode answer per edge, global-mode report. No labels."""
    out = []
    for s in steps:
        if canonicalize_question(s["question"]) == FINAL_REPORT_Q:
            ans = calib.global_report
        else:
            modes = calib.answer_modes.get(edge_key(s["question"], s["next_question"]), {})
            ans = modes.get("global", "")
        out.append({"question": s["question"], "answer": ans, "next_question": s["next_question"]})
    return out


def build_prediction_cases(config: str, calib: TextCalibrator,
                           val_cases: list[tuple[str, list[dict[str, str]], LabelContext]]) -> list[dict[str, Any]]:
    preds = []
    for cid, steps, ctx in val_cases:
        path = [(s["question"], s["next_question"]) for s in steps]
        if config == "calibrator":
            chain = calib.render_chain(path, ctx, use_field_rules=True)
        elif config == "calibrator_no_field":
            chain = calib.render_chain(path, ctx, use_field_rules=False)
        elif config == "full_pipeline":
            # predicted path from labels (NOT oracle path) -> realistic Edge-F1/BPV
            chain = calib.render_from_labels(ctx, use_field_rules=True)
        elif config == "baseline_global":
            chain = render_baseline_global(calib, steps)
        else:
            raise ValueError(f"unknown config {config}")
        preds.append({"id": cid, "chain-of-thought": chain})
    return preds


def patch_evaluator(embedding_model: str):
    """Monkeypatch em to run without scispaCy and without sklearn."""
    import evaluate_metrics as em

    class SimpleKeywordEvaluator:
        def __init__(self, model_name: str = "") -> None:
            self.model_name = model_name

        def get_keywords(self, text: str, min_length: int = 3) -> list[str]:
            words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
            return sorted({w for w in words if len(w) >= min_length and w not in STOPWORDS})

        @staticmethod
        def get_jaccard(l1, l2) -> float:
            s1, s2 = set(l1), set(l2)
            return len(s1 & s2) / len(s1 | s2) if (s1 or s2) else 0.0

        def get_score(self, ref_text: str, hyp_text: str, min_length: int = 3) -> float:
            return em.clamp01(self.get_jaccard(self.get_keywords(ref_text), self.get_keywords(hyp_text)))

    class NumpyEmbeddingEvaluator:
        """PubMedBERT mean-pooled cosine, numpy only (drops sklearn dependency)."""

        def __init__(self, model_name: str):
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self._cache: dict[str, np.ndarray] = {}

        def get_embedding(self, text: str) -> np.ndarray:
            if text in self._cache:
                return self._cache[text]
            with self.torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=False,
                                        truncation=True, max_length=512)
                out = self.model(**inputs)
                emb = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            self._cache[text] = emb
            return emb

        def get_score(self, ref_text: str, hyp_text: str, scale: float = 0.5) -> float:
            a = self.get_embedding(ref_text)
            b = self.get_embedding(hyp_text)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            score = float(np.dot(a, b) / denom) if denom else 0.0
            if scale != 0 and score > scale:
                score = (score - scale) / (1 - scale)
            return em.clamp01(score)

    em.KeywordEvaluator = SimpleKeywordEvaluator
    em.EmbeddingEvaluator = NumpyEmbeddingEvaluator
    return em


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    calib = TextCalibrator.from_assets(args.assets)
    split = json.loads((args.assets / "split.json").read_text(encoding="utf-8"))
    val_ids = set(split["val_case_ids"])

    cases = json.loads(args.cot.read_text(encoding="utf-8"))
    val_cases: list[tuple[str, list[dict[str, str]], LabelContext]] = []
    gt_cases: list[dict[str, Any]] = []
    for c in cases:
        cid = normalize_case_id(c.get("id", ""))
        if cid not in val_ids:
            continue
        steps = get_steps(c)
        if not steps:
            continue
        ctx = context_from_steps(steps, organ_hint=clean(c.get("organ")))
        val_cases.append((cid, steps, ctx))
        gt_cases.append({"id": cid, "chain-of-thought": steps})
    if args.limit:
        val_cases = val_cases[: args.limit]
        gt_cases = gt_cases[: args.limit]
    print(f"[eval] {len(val_cases)} validation cases", flush=True)

    gt_path = args.out / "gt_val.json"
    gt_path.write_text(json.dumps(gt_cases, ensure_ascii=False), encoding="utf-8")

    em = patch_evaluator(args.embedding_model)

    results: dict[str, Any] = {}
    log_path = args.out / "eval_internal.log"
    log_buf = io.StringIO()
    for config in args.configs:
        preds = build_prediction_cases(config, calib, val_cases)
        pred_path = args.out / f"pred_{config}.json"
        pred_path.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] scoring config={config} ...", flush=True)
        with contextlib.redirect_stdout(log_buf):
            workflow = em.run_workflow_batch(
                ground_truth_paths=[gt_path],
                prediction_paths=[pred_path],
                semantic_backend="lexical",
                embedding_model=args.embedding_model,
                judge_llm=None,
                voting=1,
                strict_missing_predictions=False,
                merge_predictions=False,
            )
        lb = workflow["leaderboard"][0]
        results[config] = {
            "workflow_final_ranking_score": lb["final_ranking_score"],
            "average_binary_path_validity": lb["average_binary_path_validity"],
            "average_edge_f1": lb["average_edge_f1"],
            "average_mess_nonfinal": lb["average_mess_nonfinal"],
            "average_final_report_score": lb["average_final_report_score"],
            "text_block_mess_report": (
                0.25 * lb["average_mess_nonfinal"] + 0.40 * lb["average_final_report_score"]
            ),
        }
        print(f"[eval] {config}: MESS={lb['average_mess_nonfinal']:.4f} "
              f"report={lb['average_final_report_score']:.4f} "
              f"workflowA={lb['final_ranking_score']:.4f}", flush=True)

    log_path.write_text(log_buf.getvalue(), encoding="utf-8")
    summary = {
        "num_val_cases": len(val_cases),
        "mess_backend": "lexical (conservative lower bound vs official embedding/LLM)",
        "report_backend": "REG25 formula; SimpleKeywordEvaluator + numpy-cosine PubMedBERT",
        "assumption": "oracle path + oracle labels -> BPV/Edge-F1 = 1.0; isolates MESS(0.25)+report(0.40)",
        "weights": {"BPV": 0.05, "EdgeF1": 0.30, "MESS": 0.25, "report": 0.40},
        "results": results,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== TEXT-SIDE CEILING SUMMARY =====")
    print(json.dumps(summary["results"], indent=2, ensure_ascii=False))
    print(f"saved -> {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
