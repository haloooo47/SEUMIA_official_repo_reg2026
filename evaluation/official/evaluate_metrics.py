from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import re
import csv
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

"""
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("⏳ Waiting for debugger to attach...")
debugpy.wait_for_client()
print("✅ Debugger attached!")
debugpy.breakpoint()
"""

# ============================================================
# Default config
# ============================================================

DEFAULT_JUDGE_MODEL_PATH = "models/Qwen3-8B"

DEFAULT_DEVICE = "cuda"

# Workflow semantic backend for non-final steps
DEFAULT_WORKFLOW_SEMANTIC_BACKEND = "embedding"
DEFAULT_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"


DEFAULT_WORKFLOW_SEMANTIC_BACKEND = "lexical" #llm


DEFAULT_MERGE_PREDICTIONS = False

DEFAULT_STRICT_MISSING_PREDICTIONS = False


DEFAULT_W1 = 0.3
DEFAULT_W2 = 0.3
DEFAULT_W3 = 0.4

DEFAULT_VOTING = 1
DEFAULT_JUDGE_MAX_NEW_TOKENS = 32768
DEFAULT_SKIP_MISSING_ROI = False

# Revised weights requested by user
DEFAULT_A_BPV_WEIGHT = 0.05
DEFAULT_A_EDGEF1_WEIGHT = 0.30
DEFAULT_A_MESS_NONFINAL_WEIGHT = 0.25
DEFAULT_A_FINAL_REPORT_WEIGHT = 0.40

DEFAULT_OVERALL_A_WEIGHT = 0.70
DEFAULT_OVERALL_B_WEIGHT = 0.30

# REG25-style final report metric weights
DEFAULT_REG25_TEXT_WEIGHT = 0.15  # applied to rouge and bleu each via (rouge + bleu)
DEFAULT_REG25_KEY_WEIGHT = 0.40
DEFAULT_REG25_EMB_WEIGHT = 0.30

FINAL_REPORT_QUESTION_CANONICAL = "what is the final pathology report"



# ============================================================
# Shared helpers
# ============================================================

def safe_divide(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_trailing_punctuation(text: str) -> str:
    return re.sub(r"[\s\.\,\;\:\!\?]+$", "", text).strip()


KNOWN_ALIASES = {
    "pridominant": "predominant",
    "dianoses": "diagnoses",
    "diagnosises": "diagnoses",
    "includes": "include",
}


def canonicalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = normalize_whitespace(text)
    text = strip_trailing_punctuation(text)

    for src, dst in KNOWN_ALIASES.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    return text


TERMINAL_TOKENS = {
    "",
    "end",
    "stop",
    "finish",
    "finished",
    "none",
    "null",
    "no next question",
    "no further question",
}


def canonicalize_next_question(text: Optional[str]) -> str:
    text = canonicalize_text(text)

    if text in TERMINAL_TOKENS:
        return "__END__"

    return text


def parse_judge_label(raw_text: str, allowed: List[str]) -> str:
    if raw_text is None:
        return allowed[-1]
    allowed_pattern = "|".join(re.escape(x) for x in allowed)
    match = re.search(rf"<answer>\s*({allowed_pattern})\s*</answer>", raw_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return -1

def collect_json_files(paths: List[Path]) -> List[Path]:
    """
    Accept JSON files or directories.

    If a directory is given, all *.json files inside it will be collected.
    This is non-recursive by default.
    """
    json_files: List[Path] = []

    for p in paths:
        p = Path(p)

        if p.is_file():
            json_files.append(p)

        elif p.is_dir():
            json_files.extend(sorted(p.glob("*.json")))

        else:
            raise FileNotFoundError(f"Path does not exist: {p}")

    json_files = sorted(list(dict.fromkeys(json_files)))

    if not json_files:
        raise ValueError(f"No JSON files found from paths: {paths}")

    return json_files


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, output_path: Optional[Path]) -> None:
    if output_path is None:
        return

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")


def load_case_list_from_file(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data and isinstance(data["cases"], list):
        return data["cases"]
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError(f"Invalid JSON format in {path}.")


def load_case_list_from_files(files: List[Path]) -> List[Dict[str, Any]]:
    all_cases: List[Dict[str, Any]] = []
    for f in files:
        all_cases.extend(load_case_list_from_file(f))
    return all_cases


def mean_dict_values(items: List[Dict[str, Any]], field: str) -> float:
    if not items:
        return 0.0

    return float(sum(float(x.get(field, 0.0)) for x in items) / len(items))


# ============================================================
# Part 1: Workflow reasoning metric
# ============================================================

@dataclass(frozen=True)
class Edge:
    src: str
    dst: str

    def key(self) -> Tuple[str, str]:
        return self.src, self.dst


@dataclass
class EdgeRecord:
    edge: Edge
    raw_question: str
    raw_answer: str
    raw_next_question: str
    step_index: int


@dataclass
class CaseScore:
    case_id: str
    binary_path_validity: float
    ordered_path_validity: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    mess: float
    mess_nonfinal: float
    final_report_score: float
    ranking_score: float
    gt_edge_count: int
    pred_edge_count: int
    edge_tp: int
    edge_fp: int
    edge_fn: int


def normalize_workflow_case_id(case_id: str) -> str:
    """Strip slide suffix so GT ids match manifest titles."""
    normalized = str(case_id).strip()
    for suffix in (".svs", ".tiff"):
        if normalized.lower().endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def index_cases(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}

    for case in cases:
        raw_id = case.get("id")

        if not raw_id:
            raise ValueError("Each case must contain an 'id'.")

        case_id = normalize_workflow_case_id(raw_id)

        if case_id in indexed:
            raise ValueError(f"Duplicate case id found: {case_id}")

        indexed[case_id] = case

    return indexed


def get_workflow_steps(case_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_keys = ["chain-of-thought", "chain_of_thought", "workflow", "workflow_steps", "reasoning_steps", "steps"]
    for key in candidate_keys:
        if key in case_obj:
            steps = case_obj[key]
            if not isinstance(steps, list):
                raise ValueError(f"Case {case_obj.get('id')} has invalid '{key}'. Expected list.")
            return steps
    raise ValueError(f"Case {case_obj.get('id')} does not contain workflow steps.")



def build_edge_records(case_obj: Dict[str, Any]) -> List[EdgeRecord]:
    steps = get_workflow_steps(case_obj)
    records: List[EdgeRecord] = []
    for i, step in enumerate(steps):
        question = step.get("question", "") or ""
        answer = step.get("answer", "") or ""
        next_question = step.get("next_question", "") or ""
        src = canonicalize_text(question)
        dst = canonicalize_next_question(next_question)
        records.append(
            EdgeRecord(
                edge=Edge(src=src, dst=dst),
                raw_question=str(question),
                raw_answer=str(answer),
                raw_next_question=str(next_question),
                step_index=i,
            )
        )
    return records


def unique_edge_set(records: List[EdgeRecord]) -> set[Tuple[str, str]]:
    return {rec.edge.key() for rec in records}


def ordered_edge_list(records: List[EdgeRecord]) -> List[Tuple[str, str]]:
    return [rec.edge.key() for rec in records]


def edge_answer_map(records: List[EdgeRecord]) -> Dict[Tuple[str, str], List[str]]:
    mapping: Dict[Tuple[str, str], List[str]] = {}

    for rec in records:
        mapping.setdefault(rec.edge.key(), []).append(rec.raw_answer)

    return mapping


def edge_question_map(records: List[EdgeRecord]) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}

    for rec in records:
        mapping.setdefault(rec.edge.key(), rec.raw_question)

    return mapping

def is_final_report_record(rec: EdgeRecord) -> bool:
    return canonicalize_text(rec.raw_question) == FINAL_REPORT_QUESTION_CANONICAL


def filter_nonfinal_records(records: List[EdgeRecord]) -> List[EdgeRecord]:
    return [rec for rec in records if not is_final_report_record(rec)]


def get_final_report_answer(records: List[EdgeRecord]) -> str:
    for rec in records:
        if is_final_report_record(rec):
            return rec.raw_answer
    return ""

def compute_binary_path_validity(gt_edges: set[Tuple[str, str]], pred_edges: set[Tuple[str, str]]) -> float:
    return 1.0 if gt_edges == pred_edges else 0.0


def compute_ordered_path_validity(gt_ordered_edges: List[Tuple[str, str]], pred_ordered_edges: List[Tuple[str, str]]) -> float:
    return 1.0 if gt_ordered_edges == pred_ordered_edges else 0.0


def compute_edge_metrics(gt_edges: set[Tuple[str, str]], pred_edges: set[Tuple[str, str]]) -> Tuple[float, float, float, int, int, int]:
    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = safe_divide(tp, len(pred_edges))
    recall = safe_divide(tp, len(gt_edges))
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def resolve_judge_device(requested: Optional[str] = None) -> str:
    """
    Pick the judge runtime device.

    Honors ``JUDGE_DEVICE`` when set. ``auto`` (or empty) prefers CUDA when available.
    """
    raw = (requested or os.environ.get("JUDGE_DEVICE", "auto")).strip().lower()
    if raw in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def print_runtime_diagnostics(requested_device: Optional[str] = None) -> None:
    """Print environment and device info to simplify post-mortem debugging."""
    print("[Runtime] Environment diagnostics", flush=True)
    print(f"[Runtime] Python {sys.version.split()[0]}", flush=True)
    print(f"[Runtime] requested_device = {requested_device!r}", flush=True)
    judge_env = os.environ.get("JUDGE_DEVICE")
    if judge_env is not None:
        print(f"[Runtime] JUDGE_DEVICE env = {judge_env!r}", flush=True)
    try:
        print(f"[Runtime] torch version = {torch.__version__}", flush=True)
        cuda_available = torch.cuda.is_available()
        print(f"[Runtime] torch.cuda.is_available() = {cuda_available}", flush=True)
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"[Runtime] torch.cuda.device_count() = {device_count}", flush=True)
            for i in range(device_count):
                print(
                    f"[Runtime]   cuda:{i} = {torch.cuda.get_device_name(i)}",
                    flush=True,
                )
        resolved = resolve_judge_device(requested_device)
        use_cuda = cuda_available and str(resolved).startswith("cuda")
        print(f"[Runtime] resolve_judge_device(...) = {resolved!r}", flush=True)
        print(f"[Runtime] use_cuda (preview) = {use_cuda}", flush=True)
    except Exception as exc:
        print(
            f"[Runtime] Failed to inspect torch/CUDA: {type(exc).__name__}: {exc}",
            flush=True,
        )


def _print_llm_timing(label: str, timings: Dict[str, float]) -> None:
    total = sum(timings.values())
    parts = " | ".join(f"{k}={v:.4f}s" for k, v in timings.items())
    print(f"[LLM timing][{label}] total={total:.4f}s | {parts}", flush=True)


class LocalQwenJudgeLLM:
    """
    Local LLM judge.

    Used by:
      1. workflow MESS when DEFAULT_WORKFLOW_SEMANTIC_BACKEND == "llm"
      2. visual grounding metric
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 16,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[Judge LLM] Initializing from {model_path}", flush=True)
        print(f"[Judge LLM] max_new_tokens={max_new_tokens}", flush=True)
        print(f"[Judge LLM] requested device={device!r}", flush=True)

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        print("[Judge LLM] Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print("[Judge LLM] Tokenizer loaded.", flush=True)

        device = resolve_judge_device(device)
        use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
        print(f"[Judge LLM] torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
        print(f"[Judge LLM] resolved device={device!r}, use_cuda={use_cuda}", flush=True)

        if use_cuda:
            dev = torch.device(device)
            major, minor = torch.cuda.get_device_capability(dev)
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            print(
                f"[Judge LLM] GPU cuda capability={major}.{minor}, dtype={dtype}",
                flush=True,
            )
        else:
            dev = torch.device("cpu")
            dtype = torch.float32
            print(f"[Judge LLM] Using CPU, dtype={dtype}", flush=True)

        device_map = "auto" if use_cuda else None
        print(f"[Judge LLM] Loading model (device_map={device_map!r})...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if not use_cuda:
            print("[Judge LLM] Moving model to CPU...", flush=True)
            self.model = self.model.to(dev)

        self.model.eval()

        self.input_device = next(self.model.parameters()).device
        print(f"[Judge LLM] Loaded successfully on {self.input_device}", flush=True)

    def __call__(
        self,
        prompt: str,
        timing_label: Optional[str] = None,
    ) -> Tuple[str, str]:
        label = timing_label or "judge"
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        timings["prompt_creation"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.input_device)
        timings["tokenize_to_device"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            # use_cache=True
        )
        timings["reasoning"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        timings["think_index"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")
        timings["decoding"] = time.perf_counter() - t4

        _print_llm_timing(label, timings)
        return content, thinking_content


class SemanticScorer:
    def __init__(self, backend: str = "lexical", embedding_model_name: Optional[str] = None, llm_judge: Optional[LocalQwenJudgeLLM] = None, voting: int = 1):
        print(
            f"[SemanticScorer] backend={backend!r}, voting={voting}, "
            f"llm_judge={'yes' if llm_judge is not None else 'no'}",
            flush=True,
        )
        self.backend = backend
        self.embedding_model_name = embedding_model_name
        self.llm_judge = llm_judge
        self.voting = max(1, int(voting))
        self.embedding_model = None
        if self.backend == "embedding":
            if embedding_model_name is None:
                raise ValueError("semantic backend is 'embedding', but embedding_model_name is None.")
            print(f"[SemanticScorer] Loading embedding model {embedding_model_name}...", flush=True)
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print("[SemanticScorer] Embedding model loaded.", flush=True)
        if self.backend == "llm" and self.llm_judge is None:
            raise ValueError("semantic backend is 'llm', but no judge LLM is provided.")

    @staticmethod
    def lexical_similarity(a: str, b: str) -> float:
        a_tokens = canonicalize_text(a).split()
        b_tokens = canonicalize_text(b).split()
        if not a_tokens and not b_tokens:
            return 1.0
        if not a_tokens or not b_tokens:
            return 0.0
        a_count = Counter(a_tokens)
        b_count = Counter(b_tokens)
        common = sum((a_count & b_count).values())
        total = sum((a_count | b_count).values())
        return clamp01(safe_divide(common, total))

    def embedding_similarity(self, a: str, b: str) -> float:
        embeddings = self.embedding_model.encode([a, b], normalize_embeddings=True)
        sim = float((embeddings[0] * embeddings[1]).sum())
        return clamp01(sim)

    def llm_similarity(self, question: str, answer_a: str, answer_b: str) -> float:
        def one_vote() -> float:
            prompt = f"""
You are a pathology expert.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Task:
Determine whether the two answers are semantically equivalent in a clinical sense.

Rules:
- If both answers express the same diagnosis, grade, morphology, or clinical meaning, output SAME.
- If they differ in any clinically meaningful way, output DIFFERENT.
- Be strict for clinically meaningful differences.
- Ignore only minor wording differences.

Output:
Return only one word:
SAME or DIFFERENT
"""
            result, _ = self.llm_judge(prompt)
            label = parse_judge_label(result, allowed=["SAME", "DIFFERENT"])
            return 1.0 if label == "SAME" else 0.0
        votes = [one_vote() for _ in range(self.voting)]
        return max(set(votes), key=votes.count)

    def similarity(self, answer_a: str, answer_b: str, question: str = "") -> float:
        if self.backend == "lexical":
            return self.lexical_similarity(answer_a, answer_b)
        if self.backend == "embedding":
            return self.embedding_similarity(answer_a, answer_b)
        if self.backend == "llm":
            return self.llm_similarity(question, answer_a, answer_b)
        raise ValueError(f"Unknown semantic backend: {self.backend}")


class EmbeddingEvaluator:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def get_embedding(self, text: str):
        import torch
        inputs = self.tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_score(self, ref_text: str, hyp_text: str, scale: float = 0.5) -> float:
        from sklearn.metrics.pairwise import cosine_similarity
        ref_embedding = self.get_embedding(ref_text)
        hyp_embedding = self.get_embedding(hyp_text)
        score = cosine_similarity([ref_embedding], [hyp_embedding])[0][0]
        if (scale != 0) and (score > scale):
            score = (score - scale) / (1 - scale)
        return clamp01(score)


class KeywordEvaluator:
    def __init__(self, model_name: str = 'en_core_sci_lg'):
        import spacy
        self.nlp = spacy.load(model_name)

    def get_keywords(self, text: str, min_length: int = 3) -> List[str]:
        doc = self.nlp(text)
        keywords = []
        for ent in doc.ents:
            if len(ent.text) >= min_length:
                keywords.append(ent.text.lower())
        return list(set(keywords))

    @staticmethod
    def get_jaccard(list1: List[str], list2: List[str]) -> float:
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0.0

    def get_score(self, ref_text: str, hyp_text: str, min_length: int = 3) -> float:
        ref_keywords = self.get_keywords(ref_text, min_length)
        hyp_keywords = self.get_keywords(hyp_text, min_length)
        return clamp01(self.get_jaccard(ref_keywords, hyp_keywords))


class REG25FinalReportEvaluator:
    def __init__(self, embedding_model: str, spacy_model: str = 'en_core_sci_lg'):
        print(
            f"[REG25] Loading evaluators "
            f"(embedding_model={embedding_model!r}, spacy_model={spacy_model!r})...",
            flush=True,
        )
        self.embedding_eval = EmbeddingEvaluator(embedding_model)
        self.key_eval = KeywordEvaluator(spacy_model)
        print("[REG25] Evaluators loaded.", flush=True)

    @staticmethod
    def get_bleu4(ref_text: str, hyp_text: str) -> float:
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        ref_fourgrams = [' '.join(ref_words[i:i+4]) for i in range(len(ref_words)-3)]
        hyp_fourgrams = [' '.join(hyp_words[i:i+4]) for i in range(len(hyp_words)-3)]
        count = 0
        total = 0
        for fourgram in hyp_fourgrams:
            count += min(hyp_fourgrams.count(fourgram), ref_fourgrams.count(fourgram))
            total += 1
        if total == 0:
            return 0.0
        return clamp01(count / total)

    @staticmethod
    def get_rouge(ref_text: str, hyp_text: str) -> float:
        def lcs(X, Y):
            m = len(X)
            n = len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i-1] == Y[j-1]:
                        L[i][j] = L[i-1][j-1] + 1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
            return L[m][n]
        ref_tokens = ref_text.lower().split()
        hyp_tokens = hyp_text.lower().split()
        lcs_length = lcs(ref_tokens, hyp_tokens)
        ref_length = len(ref_tokens)
        hyp_length = len(hyp_tokens)
        precision = lcs_length / hyp_length if hyp_length > 0 else 0.0
        recall = lcs_length / ref_length if ref_length > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return clamp01(f1_score)

    def evaluate_text(self, ref_text: str, hyp_text: str) -> Dict[str, float]:
        emb_score = self.embedding_eval.get_score(ref_text, hyp_text)
        key_score = self.key_eval.get_score(ref_text, hyp_text)
        bleu_score = self.get_bleu4(ref_text, hyp_text)
        rouge_score = self.get_rouge(ref_text, hyp_text)
        ranking_score = (
            DEFAULT_REG25_TEXT_WEIGHT * (rouge_score + bleu_score)
            + DEFAULT_REG25_KEY_WEIGHT * key_score
            + DEFAULT_REG25_EMB_WEIGHT * emb_score
        )
        return {
            'final_report_score': clamp01(ranking_score),
            'emb_score': emb_score,
            'key_score': key_score,
            'bleu_score': bleu_score,
            'rouge_score': rouge_score,
        }


def compute_mess(gt_records: List[EdgeRecord], pred_records: List[EdgeRecord], scorer: SemanticScorer) -> float:
    if not gt_records:
        return 0.0
    gt_answer_map = edge_answer_map(gt_records)
    pred_answer_map = edge_answer_map(pred_records)
    gt_question_map = edge_question_map(gt_records)
    sims: List[float] = []
    for edge_key, gt_answers in gt_answer_map.items():
        gt_answer = gt_answers[0] if gt_answers else ""
        pred_answers = pred_answer_map.get(edge_key)
        if not pred_answers:
            sims.append(0.0)
            continue
        pred_answer = pred_answers[0] if pred_answers else ""
        question = gt_question_map.get(edge_key, "")
        sims.append(clamp01(scorer.similarity(answer_a=gt_answer, answer_b=pred_answer, question=question)))
    return float(sum(sims) / len(sims))



def evaluate_workflow_case(case_id: str, gt_case: Dict[str, Any], pred_case: Optional[Dict[str, Any]], scorer: SemanticScorer, final_report_evaluator: REG25FinalReportEvaluator) -> CaseScore:
    gt_records = build_edge_records(gt_case)
    pred_records = build_edge_records(pred_case) if pred_case is not None else []

    gt_edges = unique_edge_set(gt_records)
    pred_edges = unique_edge_set(pred_records)
    gt_ordered_edges = ordered_edge_list(gt_records)
    pred_ordered_edges = ordered_edge_list(pred_records)

    bpv = compute_binary_path_validity(gt_edges, pred_edges)
    ordered_bpv = compute_ordered_path_validity(gt_ordered_edges, pred_ordered_edges)
    precision, recall, edge_f1, tp, fp, fn = compute_edge_metrics(gt_edges=gt_edges, pred_edges=pred_edges)

    nonfinal_gt_records = filter_nonfinal_records(gt_records)
    nonfinal_pred_records = filter_nonfinal_records(pred_records)
    mess_nonfinal = compute_mess(gt_records=nonfinal_gt_records, pred_records=nonfinal_pred_records, scorer=scorer)

    # keep old field for compatibility; now equals non-final MESS
    mess = mess_nonfinal

    gt_final_report = get_final_report_answer(gt_records)
    pred_final_report = get_final_report_answer(pred_records)
    print("CASE ID: ", case_id)
    print(f"GT_FINAL_REPORT: {gt_final_report}-")
    print(f"PR_FINAL_REPORT: {pred_final_report}-")

    final_report_metrics = final_report_evaluator.evaluate_text(gt_final_report, pred_final_report)
    final_report_score = final_report_metrics['final_report_score']
    print("final_report_metrics:", final_report_metrics)

    ranking_score = (
        DEFAULT_A_BPV_WEIGHT * bpv
        + DEFAULT_A_EDGEF1_WEIGHT * edge_f1
        + DEFAULT_A_MESS_NONFINAL_WEIGHT * mess_nonfinal
        + DEFAULT_A_FINAL_REPORT_WEIGHT * final_report_score
    )

    return CaseScore(
        case_id=case_id,
        binary_path_validity=bpv,
        ordered_path_validity=ordered_bpv,
        edge_precision=precision,
        edge_recall=recall,
        edge_f1=edge_f1,
        mess=mess,
        mess_nonfinal=mess_nonfinal,
        final_report_score=final_report_score,
        ranking_score=ranking_score,
        gt_edge_count=len(gt_edges),
        pred_edge_count=len(pred_edges),
        edge_tp=tp,
        edge_fp=fp,
        edge_fn=fn,
    )


def evaluate_workflow_dataset(gt_cases: List[Dict[str, Any]], pred_cases: List[Dict[str, Any]], scorer: SemanticScorer, final_report_evaluator: REG25FinalReportEvaluator, strict_missing_predictions: bool = False) -> Dict[str, Any]:
    gt_index = index_cases(gt_cases)
    pred_index = index_cases(pred_cases)
    case_scores: List[CaseScore] = []
    missing_case_ids: List[str] = []
    extra_case_ids = sorted(set(pred_index.keys()) - set(gt_index.keys()))
    for case_id, gt_case in gt_index.items():
        pred_case = pred_index.get(case_id)
        if pred_case is None:
            missing_case_ids.append(str(gt_case.get("id", case_id)))
            if strict_missing_predictions:
                raise ValueError(f"Missing prediction for case id: {gt_case.get('id', case_id)}")
        case_scores.append(
            evaluate_workflow_case(
                str(gt_case.get("id", case_id)),
                gt_case,
                pred_case,
                scorer,
                final_report_evaluator,
            )
        )

    def avg(field: str) -> float:
        return float(sum(getattr(cs, field) for cs in case_scores) / len(case_scores)) if case_scores else 0.0

    return {
        'num_ground_truth_cases': len(gt_index),
        'num_prediction_cases': len(pred_index),
        'num_missing_prediction_cases': len(missing_case_ids),
        'num_extra_prediction_cases': len(extra_case_ids),
        'missing_prediction_case_ids': missing_case_ids,
        'extra_prediction_case_ids': extra_case_ids,
        'average_binary_path_validity': avg('binary_path_validity'),
        'average_ordered_path_validity': avg('ordered_path_validity'),
        'average_edge_precision': avg('edge_precision'),
        'average_edge_recall': avg('edge_recall'),
        'average_edge_f1': avg('edge_f1'),
        'average_mess': avg('mess'),
        'average_mess_nonfinal': avg('mess_nonfinal'),
        'average_final_report_score': avg('final_report_score'),
        'final_ranking_score': avg('ranking_score'),
        'per_case': [asdict(cs) for cs in case_scores],
    }


def workflow_global_average_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'workflow_final_score': float(summary.get('final_ranking_score', 0.0)),
        'average_binary_path_validity': float(summary.get('average_binary_path_validity', 0.0)),
        'average_ordered_path_validity': float(summary.get('average_ordered_path_validity', 0.0)),
        'average_edge_precision': float(summary.get('average_edge_precision', 0.0)),
        'average_edge_recall': float(summary.get('average_edge_recall', 0.0)),
        'average_edge_f1': float(summary.get('average_edge_f1', 0.0)),
        'average_mess': float(summary.get('average_mess', 0.0)),
        'average_mess_nonfinal': float(summary.get('average_mess_nonfinal', 0.0)),
        'average_final_report_score': float(summary.get('average_final_report_score', 0.0)),
    }



def run_workflow_batch(ground_truth_paths: List[Path], prediction_paths: List[Path], semantic_backend: str, embedding_model: Optional[str], judge_llm: Optional[LocalQwenJudgeLLM], voting: int, strict_missing_predictions: bool, merge_predictions: bool) -> Dict[str, Any]:
    print("[Workflow] Collecting GT/prediction JSON files...", flush=True)
    gt_files = collect_json_files(ground_truth_paths)
    pred_files = collect_json_files(prediction_paths)
    print(
        f"[Workflow] Found {len(gt_files)} GT file(s), {len(pred_files)} prediction file(s)",
        flush=True,
    )
    gt_cases = load_case_list_from_files(gt_files)
    scorer = SemanticScorer(backend=semantic_backend, embedding_model_name=embedding_model, llm_judge=judge_llm, voting=voting)
    final_report_evaluator = REG25FinalReportEvaluator(embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL)

    if merge_predictions:
        pred_cases = load_case_list_from_files(pred_files)
        print(
            f"[Workflow] Evaluating merged predictions ({len(pred_cases)} case(s))...",
            flush=True,
        )
        summary = evaluate_workflow_dataset(gt_cases, pred_cases, scorer, final_report_evaluator, strict_missing_predictions)
        print(
            f"[Workflow] Done. final_ranking_score={summary['final_ranking_score']:.4f}",
            flush=True,
        )
        global_average = workflow_global_average_from_summary(summary)
        global_average['num_prediction_files'] = len(pred_files)
        global_average['best_prediction_name'] = 'merged_predictions'
        global_average['best_prediction_file'] = None
        global_average['best_workflow_final_score'] = global_average['workflow_final_score']
        return {
            'mode': 'workflow',
            'merge_predictions': True,
            'ground_truth_files': [str(x) for x in gt_files],
            'prediction_files': [str(x) for x in pred_files],
            'global_average': global_average,
            'result': summary,
        }

    evaluations: Dict[str, Any] = {}
    leaderboard: List[Dict[str, Any]] = []
    for pred_file in pred_files:
        print(f"[Workflow] Evaluating {pred_file.name}...", flush=True)
        pred_cases = load_case_list_from_file(pred_file)
        summary = evaluate_workflow_dataset(gt_cases, pred_cases, scorer, final_report_evaluator, strict_missing_predictions)
        print(
            f"[Workflow] {pred_file.name}: final_ranking_score={summary['final_ranking_score']:.4f}",
            flush=True,
        )
        name = pred_file.stem
        evaluations[name] = {'prediction_file': str(pred_file), 'result': summary}
        leaderboard.append({
            'prediction_name': name,
            'prediction_file': str(pred_file),
            'final_ranking_score': summary['final_ranking_score'],
            'average_binary_path_validity': summary['average_binary_path_validity'],
            'average_ordered_path_validity': summary['average_ordered_path_validity'],
            'average_edge_precision': summary['average_edge_precision'],
            'average_edge_recall': summary['average_edge_recall'],
            'average_edge_f1': summary['average_edge_f1'],
            'average_mess_nonfinal': summary['average_mess_nonfinal'],
            'average_final_report_score': summary['average_final_report_score'],
        })
    leaderboard = sorted(leaderboard, key=lambda x: x['final_ranking_score'], reverse=True)
    best_item = leaderboard[0] if leaderboard else None
    print(f"[Workflow] Evaluated {len(pred_files)} prediction file(s).", flush=True)
    global_average = {
        'num_prediction_files': len(pred_files),
        'workflow_final_score': mean_dict_values(leaderboard, 'final_ranking_score'),
        'average_binary_path_validity': mean_dict_values(leaderboard, 'average_binary_path_validity'),
        'average_ordered_path_validity': mean_dict_values(leaderboard, 'average_ordered_path_validity'),
        'average_edge_precision': mean_dict_values(leaderboard, 'average_edge_precision'),
        'average_edge_recall': mean_dict_values(leaderboard, 'average_edge_recall'),
        'average_edge_f1': mean_dict_values(leaderboard, 'average_edge_f1'),
        'average_mess_nonfinal': mean_dict_values(leaderboard, 'average_mess_nonfinal'),
        'average_final_report_score': mean_dict_values(leaderboard, 'average_final_report_score'),
        'best_prediction_name': best_item['prediction_name'] if best_item else None,
        'best_prediction_file': best_item['prediction_file'] if best_item else None,
        'best_workflow_final_score': float(best_item['final_ranking_score']) if best_item else 0.0,
    }
    return {
        'mode': 'workflow',
        'merge_predictions': False,
        'ground_truth_files': [str(x) for x in gt_files],
        'prediction_files': [str(x) for x in pred_files],
        'global_average': global_average,
        'leaderboard': leaderboard,
        'evaluations': evaluations,
    }


# # ============================================================
# # Part 2: Visual grounding metric
# # ============================================================
# question_dict_roi_local = {
#     "tissue_presence": [
#         "Does this ROI contain analyzable histological tissue? Answer briefly.",
#     ],

#     "content": [
#         "What is the dominant content in this ROI? Answer briefly.",
#     ],

#     "quality": [
#         "Is this ROI informative for histological image analysis? Answer briefly.",
#     ],
# }



class VisualGroundingEvaluator:
    def __init__(
        self,
        judge_model: Callable[[str], str],
    ):
        self.judge_model = judge_model
        # self.answer_cache: Dict[Tuple[str, str], str] = {}

    # def _get_question_dict(
    #     self,
    #     question_dict: Optional[Dict[str, List[str]]],
    # ) -> Dict[str, List[str]]:
    #     if question_dict is not None:
    #         return question_dict

    #     if self.default_question_dict is not None:
    #         return self.default_question_dict

    #     raise ValueError("question_dict must be provided")

    # def _model_answer(self, roi: Any, question: str) -> str:
    #     """
    #     Cache answers for file-path ROIs.

    #     This avoids repeatedly asking the VLM the same question on the same ROI
    #     across B1/B2/B3.
    #     """
    #     if isinstance(roi, (str, Path)):
    #         key = (str(roi), question)

    #         if key in self.answer_cache:
    #             return self.answer_cache[key]

    #         ans = self.model(roi, question)
    #         self.answer_cache[key] = ans

    #         return ans

    #     return self.model(roi, question)

    def judge_background(
        self,
        question: str,
        answer: str,
        timing_label: str = "visual/background",
    ) -> float:
        t0 = time.perf_counter()
        prompt = f"""
        You are a pathology expert.

        The following image region is from a whole slide image, but it contains NO tissue, background only.

        Question:
        {question}

        Model Answer:
        {answer}

        Task:
        Determine whether the model answer is appropriate for a background-only image region.

        Judgment rules:
        - If the answer claims any diagnostic, morphological, grading, tumor-related, cell-related, tissue-specific, or pathology-specific information, the judgment is INCORRECT.
        - If the answer states that no reliable information can be determined because the region contains no tissue or only background, the judgment is CORRECT.
        - If the answer says background, no tissue, not assessable, not evaluable, or uncertain due to absence of tissue, the judgment is CORRECT.
        - If the answer gives a diagnosis or describes tissue morphology despite the region being background only, the judgment is INCORRECT.

        You may reason internally if needed.

        Final output requirement:
        At the end, output exactly one final judgment using this format:

        <answer>CORRECT</answer>

        or

        <answer>INCORRECT</answer>

        Do not output anything more after your answer.
        """
        prompt_build_s = time.perf_counter() - t0
        print(
            f"[LLM timing][{timing_label}] prompt_build={prompt_build_s:.4f}s",
            flush=True,
        )
        result, reasoning = self.judge_model(prompt, timing_label=timing_label)
        label = parse_judge_label(result, allowed=["CORRECT", "INCORRECT"])
        # print(f"[Judge Background] question: {question}, answer: {answer}, label: {label}, reasoning: {reasoning}")
        if label == "CORRECT":
            return 1.0
        elif label == "INCORRECT":
            return 0.0
        else:
            return -1

    def judge_similarity(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        timing_label: str = "visual/similarity",
    ) -> float:
        t0 = time.perf_counter()
        prompt = f"""
You are a pathology expert.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Task:
Determine whether the two answers are semantically equivalent in a clinical sense.

Judgment rules:
- If both answers express the same diagnosis, grade, morphology, visual content, or clinical meaning, the judgment is SAME.
- If they differ in any clinically meaningful way, the judgment is DIFFERENT.
- Be strict about clinically meaningful differences, including diagnosis, tumor type, benign/malignant status, grade, cell type, morphology, tissue structure, staining pattern, uncertainty level, and clinical implication.
- Ignore only minor wording differences that do not change the clinical meaning.
- If one answer is more specific than the other and the added specificity changes or narrows the clinical meaning, the judgment is DIFFERENT.
- If one answer makes a confident claim while the other says the finding is uncertain, not assessable, or cannot be determined, the judgment is DIFFERENT.

You may reason internally if needed.

Final output requirement:
At the end, output exactly one final judgment using this format:

<answer>SAME</answer>

or

<answer>DIFFERENT</answer>

Do not repeat the answer.
Do not output anything after the closing </answer> tag.
"""
        prompt_build_s = time.perf_counter() - t0
        print(
            f"[LLM timing][{timing_label}] prompt_build={prompt_build_s:.4f}s",
            flush=True,
        )
        result, reasoning = self.judge_model(prompt, timing_label=timing_label)
        label = parse_judge_label(result, allowed=["SAME", "DIFFERENT"])

        if label == "SAME":
            return 1.0
        elif label == "DIFFERENT":
            return 0.0
        else:
            return -1

    def vote(self, func: Callable[[], float], voting: int) -> float:
        voting = max(1, int(voting))
        results = [func() for _ in range(voting)]

        return max(set(results), key=results.count)


def load_visual_answer_json(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load participant visual answer JSON.

    Expected format:
    [
      {
        "id": "roi_000000",
        "image": "...",
        "question": "...",
        "answer": "..."
      },
      ...
    ]
    """
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(f"Visual answer JSON must be a list: {path}")

    answers_by_id: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Visual answer item {idx} is not a dict.")

        for key in ["id", "question", "answer"]:
            if key not in item:
                raise ValueError(f"Visual answer item {idx} missing key: {key}")

        roi_id = str(item["id"])

        if roi_id in answers_by_id:
            raise ValueError(f"Duplicate visual ROI id found: {roi_id}")

        answers_by_id[roi_id] = item

    return answers_by_id

_ROIS_MAPPING_REQUIRED_COLS = {
    "anonymous_id",
    "image",
    "variant",
    "paired_anonymous_id",
    "b3_paired_anonymous_id",
    "label",
    "question",
}


def load_rois_mapping_txt(path: Path) -> List[Dict[str, str]]:
    """
    Load ROI mapping txt for Metric B.

    Required columns:
    anonymous_id, image, variant, paired_anonymous_id,
    b3_paired_anonymous_id, label, question
    """
    rows: List[Dict[str, str]] = []

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"Mapping txt has no header: {path}")

        missing_cols = _ROIS_MAPPING_REQUIRED_COLS - set(reader.fieldnames)

        if missing_cols:
            raise ValueError(
                f"Mapping txt missing columns: {sorted(missing_cols)}"
            )

        for row in reader:
            clean_row = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            rows.append(clean_row)

    return rows


def load_rois_mapping_txt_as_dict(path: Path) -> Dict[str, Dict[str, str]]:
    """Load ROI mapping txt keyed by anonymous_id."""
    rows: Dict[str, Dict[str, str]] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"Mapping txt has no header: {path}")

        missing_cols = _ROIS_MAPPING_REQUIRED_COLS - set(reader.fieldnames)

        if missing_cols:
            raise ValueError(
                f"Mapping txt missing columns: {sorted(missing_cols)}"
            )

        for row in reader:
            clean_row = {
                k: (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            rows[clean_row["anonymous_id"]] = clean_row

    return rows


def get_answer_text(item: Optional[Dict[str, Any]]) -> str:
    if item is None:
        return ""
    return str(item.get("answer", "") or "").strip()


def get_question_text(
    item: Optional[Dict[str, Any]],
    fallback: str = "",
) -> str:
    if item is None:
        return str(fallback or "").strip()

    return str(item.get("question", fallback) or fallback or "").strip()


def score_or_zero(score: float) -> float:
    """
    The judge returns -1 when parsing fails.
    Treat parsing failure as 0.0 score.
    """
    if score == -1:
        return 0.0
    return float(score)

def score_or_zero_b3(score: float) -> float:
    """
    The judge returns -1 when parsing fails.
    Treat parsing failure as 0.0 score.
    """
    if score == -1:
        return 0.0
    b3_score = 1.0 - float(score)
    return b3_score


def compute_b1_background_from_answers(
    evaluator: VisualGroundingEvaluator,
    answers_by_id: Dict[str, Dict[str, Any]],
    mapping_rows: List[Dict[str, str]],
    voting: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    B1:
    Use background original ROIs only.
    """
    scores = []
    used_ids = []
    missing_ids = []

    b1_total = sum(
        1
        for row in mapping_rows
        if row.get("label") == "background" and row.get("variant") == "original"
    )
    b1_done = 0

    for row in mapping_rows:
        if row.get("label") != "background":
            continue

        if row.get("variant") != "original":
            continue

        b1_done += 1
        id = row["anonymous_id"]
        print(f"[Visual B1] {b1_done}/{b1_total} ROI {id} ...", flush=True)
        id_image = row["image"]
        id_question = row["question"]
        item = answers_by_id.get(id)
        if item:
            image_pred = item["image"]
            question_pred = item["question"]

        if item is None or (id_image != image_pred) or (id_question != question_pred):
            missing_ids.append(id)
            scores.append(0.0)
            print(
                f"[Visual B1] {b1_done}/{b1_total} ROI {id} -> missing (0.00)",
                flush=True,
            )
            continue           

        #question = get_question_text(item, fallback=row.get("question", ""))
        question = str(id_question).strip()
        #answer = get_answer_text(item)
        answer = str(item.get("answer", "") or "").strip()

        score = evaluator.vote(
            lambda question=question, answer=answer:
                evaluator.judge_background(question, answer, timing_label="B1"),
            voting,
        )

        b1_score = score_or_zero(score)
        scores.append(b1_score)
        used_ids.append(id)
        print(
            f"[Visual B1] {b1_done}/{b1_total} ROI {id} -> {b1_score:.2f}",
            flush=True,
        )

    return (
        float(np.mean(scores)) if scores else 0.0,
        {
            "num_used_background_original": len(used_ids),
            "missing_background_original_ids": missing_ids,
        },
    )


def compute_b2_sensitivity_from_answers(
    evaluator: VisualGroundingEvaluator,
    answers_by_id: Dict[str, Dict[str, Any]],
    mapping_rows: List[Dict[str, str]],
    mapping_rows_dict: Dict[str, Dict[str, str]],
    voting: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    B2:
    Use tissue original ROIs only.

    Compare:
        tissue original answer
        vs
        its paired tissue perturbed answer

    Pairing is from:
        paired_anonymous_id
    """
    scores = []
    used_pairs = []
    missing_pairs = []

    b2_total = sum(
        1
        for row in mapping_rows
        if row.get("label") == "tissue"
        and row.get("variant") == "original"
        and row.get("paired_anonymous_id") != "none"
    )
    b2_done = 0

    for row in mapping_rows:
        if row.get("label") != "tissue":
            continue

        if row.get("variant") != "original":
            continue

        if row.get("paired_anonymous_id") == "none":
            continue

        b2_done += 1
        original_id = row["anonymous_id"]
        perturbed_id = row["paired_anonymous_id"]
        print(
            f"[Visual B2] {b2_done}/{b2_total} pair {original_id}/{perturbed_id} ...",
            flush=True,
        )
        original_image = row["image"]
        original_question = row["question"]
        perturbed_image = mapping_rows_dict[perturbed_id]['image']
        perturbed_question = mapping_rows_dict[perturbed_id]['question']

        #original_id = row["anonymous_id"]
        #perturbed_id = row.get("paired_anonymous_id", "none")

        #if perturbed_id == "none":
        #    missing_pairs.append({
        #        "original_id": original_id,
        #        "perturbed_id": perturbed_id,
        #        "reason": "paired_anonymous_id is none",
        #    })
        #    continue

        original_item = answers_by_id.get(original_id)
        perturbed_item = answers_by_id.get(perturbed_id)

        if original_item:
            original_image_pred = original_item["image"]
            original_question_pred = original_item["question"]

        if perturbed_item:
            perturbed_image_pred = perturbed_item["image"]
            perturbed_question_pred = perturbed_item["question"]

        if original_item is None or perturbed_item is None or (original_image != original_image_pred) or (perturbed_image != perturbed_image_pred) or (original_question != original_question_pred) or (perturbed_question != perturbed_question_pred):
            missing_pairs.append({
                "original_id": original_id,
                "perturbed_id": perturbed_id,
                "missing_original_answer": original_item is None,
                "missing_perturbed_answer": perturbed_item is None,
            })
            scores.append(0.0)
            print(
                f"[Visual B2] {b2_done}/{b2_total} pair "
                f"{original_id}/{perturbed_id} -> missing (0.00)",
                flush=True,
            )
            continue

        #question = get_question_text(original_item, fallback=row.get("question", ""))
        #original_answer = get_answer_text(original_item)
        #perturbed_answer = get_answer_text(perturbed_item)
        question = str(original_question).strip()
        original_answer = str(original_item.get("answer", "") or "").strip()
        perturbed_answer = str(perturbed_item.get("answer", "") or "").strip()

        sim = evaluator.vote(
            lambda question=question,
                original_answer=original_answer,
                perturbed_answer=perturbed_answer:
                evaluator.judge_similarity(
                    question,
                    original_answer,
                    perturbed_answer,
                    timing_label="B2",
                ),
            voting,
        )

        b2_score = score_or_zero(sim)
        scores.append(b2_score)
        print(
            f"[Visual B2] {b2_done}/{b2_total} pair "
            f"{original_id}/{perturbed_id} -> {b2_score:.2f}",
            flush=True,
        )

        used_pairs.append({
            "original_id": original_id,
            "perturbed_id": perturbed_id,
            "pair_id": row.get("pair_id", ""),

            # per-pair metric
            "similarity_score": float(b2_score),
            "b2_score": float(b2_score),

            # optional debug fields
            "question": question,
            "original_answer": original_answer,
            "perturbed_answer": perturbed_answer,
        })

    return (
        float(np.mean(scores)) if scores else 0.0,
        {
            "num_used_tissue_original_perturbed_pairs": len(used_pairs),
            "used_pairs": used_pairs,
            "missing_pairs": missing_pairs,
        },
    )


def compute_b3_cross_region_from_answers(
    evaluator: VisualGroundingEvaluator,
    answers_by_id: Dict[str, Dict[str, Any]],
    mapping_rows: List[Dict[str, str]],
    mapping_rows_dict: Dict[str, Dict[str, str]],
    voting: int = 1,
) -> Tuple[float, Dict[str, Any]]:
    """
    B3:
    Use tissue original ROIs only.

    Compare each tissue original answer with exactly one paired background original answer.

    Pairing is from:
        b3_paired_anonymous_id

    This avoids full tissue-background pairwise comparison.
    """
    scores = []
    used_pairs = []
    missing_pairs = []
    question_mismatch_pairs = []

    row_by_id = {
        row["anonymous_id"]: row
        for row in mapping_rows
        if row.get("anonymous_id")
    }

    b3_total = sum(
        1
        for row in mapping_rows
        if row.get("label") == "tissue"
        and row.get("variant") == "original"
        and row.get("b3_paired_anonymous_id") != "none"
    )
    b3_done = 0

    for row in mapping_rows:
        if row.get("label") != "tissue":
            continue

        if row.get("variant") != "original":
            continue

        if row.get("b3_paired_anonymous_id") == "none":
            continue

        b3_done += 1
        tissue_id = row["anonymous_id"]
        tissue_image = row["image"]
        tissue_question = row["question"]
        background_id = row["b3_paired_anonymous_id"]
        print(
            f"[Visual B3] {b3_done}/{b3_total} pair "
            f"{tissue_id}/{background_id} ...",
            flush=True,
        )
        background_image = mapping_rows_dict[background_id]['image']
        background_question = mapping_rows_dict[background_id]['question']

        #tissue_id = row["anonymous_id"]
        #background_id = row.get("b3_paired_anonymous_id", "none")

        #if background_id == "none":
        #    missing_pairs.append({
        #        "tissue_id": tissue_id,
        #        "background_id": background_id,
        #        "reason": "b3_paired_anonymous_id is none",
        #    })
        #    continue

        tissue_item = answers_by_id.get(tissue_id)
        background_item = answers_by_id.get(background_id)
        background_row = row_by_id.get(background_id)

        if tissue_item:
            tissue_image_pred = tissue_item["image"]
            tissue_question_pred = tissue_item["question"]

        if background_item:
            background_image_pred = background_item["image"]
            background_question_pred = background_item["question"]

        if tissue_item is None or background_item is None or background_row is None or (tissue_image != tissue_image_pred) or (background_image != background_image_pred) or (tissue_question != tissue_question_pred) or (background_question != background_question_pred):
            missing_pairs.append({
                "tissue_id": tissue_id,
                "background_id": background_id,
                "missing_tissue_answer": tissue_item is None,
                "missing_background_answer": background_item is None,
                "missing_background_mapping": background_row is None,
            })
            scores.append(0.0)
            print(
                f"[Visual B3] {b3_done}/{b3_total} pair "
                f"{tissue_id}/{background_id} -> missing (0.00)",
                flush=True,
            )
            continue


        #tissue_question = get_question_text(tissue_item, fallback=row.get("question", ""))
        tissue_question = str(tissue_question).strip()
        background_question = str(background_question).strip()

        if canonicalize_text(tissue_question) != canonicalize_text(background_question):
            question_mismatch_pairs.append({
                "tissue_id": tissue_id,
                "background_id": background_id,
                "tissue_question": tissue_question,
                "background_question": background_question,
            })

        #tissue_answer = get_answer_text(tissue_item)
        tissue_answer = str(tissue_item.get("answer", "") or "").strip()
        #background_answer = get_answer_text(background_item)
        background_answer = str(background_item.get("answer", "") or "").strip()

        sim = evaluator.vote(
            lambda tissue_question=tissue_question,
                tissue_answer=tissue_answer,
                background_answer=background_answer:
                evaluator.judge_similarity(
                    tissue_question,
                    tissue_answer,
                    background_answer,
                    timing_label="B3/similarity",
                ),
            voting,
        )

        background_correct = evaluator.vote(
            lambda background_question=background_question,
                background_answer=background_answer:
                evaluator.judge_background(
                    background_question,
                    background_answer,
                    timing_label="B3/background",
                ),
            voting,
        )

        
        similarity_score = score_or_zero_b3(sim)
        background_score = score_or_zero(background_correct)


        b3_score = similarity_score * background_score

        scores.append(b3_score)
        print(
            f"[Visual B3] {b3_done}/{b3_total} pair "
            f"{tissue_id}/{background_id} -> {b3_score:.2f} "
            f"(sim={similarity_score:.2f}, bg={background_score:.2f})",
            flush=True,
        )

        used_pairs.append({
            "tissue_id": tissue_id,
            "background_id": background_id,

            # per-pair metric
            "similarity_score": float(similarity_score),
            "background_score": float(background_score),
            "b3_score": float(b3_score),

            # optional debug fields
            "question": tissue_question,
            "tissue_answer": tissue_answer,
            "background_answer": background_answer,
        })

    return (
        float(np.mean(scores)) if scores else 0.0,
        {
            "num_used_b3_pairs": len(used_pairs),
            "used_pairs": used_pairs,
            "missing_pairs": missing_pairs,
            "question_mismatch_pairs": question_mismatch_pairs,
        },
    )


def run_visual_dataset_from_answer_json(
    visual_answer_json_path: Path,
    mapping_txt_path: Path,
    judge_llm: LocalQwenJudgeLLM,
    w1: float,
    w2: float,
    w3: float,
    voting: int,
) -> Dict[str, Any]:
    """
    New Metric B entry.

    No VLM.
    No image reading.
    No perturb_fn.

    Inputs:
    - participant visual answer JSON
    - internal rois_mapping.txt
    """
    answers_by_id = load_visual_answer_json(visual_answer_json_path)
    mapping_rows = load_rois_mapping_txt(mapping_txt_path)
    mapping_rows_dict = load_rois_mapping_txt_as_dict(mapping_txt_path)

    print(
        f"[Visual] Loaded {len(answers_by_id)} answers, "
        f"{len(mapping_rows)} mapping rows",
        flush=True,
    )

    evaluator = VisualGroundingEvaluator(
        judge_model=judge_llm,
    )

    print("[Visual B1] Starting background scoring...", flush=True)
    b1, b1_detail = compute_b1_background_from_answers(
        evaluator=evaluator,
        answers_by_id=answers_by_id,
        mapping_rows=mapping_rows,
        voting=voting,
    )
    print(
        f"[Visual B1] Done. score={b1:.4f}, "
        f"used={b1_detail['num_used_background_original']}, "
        f"missing={len(b1_detail['missing_background_original_ids'])}",
        flush=True,
    )

    print("[Visual B2] Starting sensitivity scoring...", flush=True)
    b2, b2_detail = compute_b2_sensitivity_from_answers(
        evaluator=evaluator,
        answers_by_id=answers_by_id,
        mapping_rows=mapping_rows,
        mapping_rows_dict=mapping_rows_dict,
        voting=voting,
    )
    print(
        f"[Visual B2] Done. score={b2:.4f}, "
        f"pairs={b2_detail['num_used_tissue_original_perturbed_pairs']}, "
        f"missing={len(b2_detail['missing_pairs'])}",
        flush=True,
    )

    print("[Visual B3] Starting cross-region scoring...", flush=True)
    b3, b3_detail = compute_b3_cross_region_from_answers(
        evaluator=evaluator,
        answers_by_id=answers_by_id,
        mapping_rows=mapping_rows,
        mapping_rows_dict=mapping_rows_dict,
        voting=voting,
    )
    print(
        f"[Visual B3] Done. score={b3:.4f}, "
        f"pairs={b3_detail['num_used_b3_pairs']}, "
        f"missing={len(b3_detail['missing_pairs'])}",
        flush=True,
    )

    final_score = w1 * b1 + w2 * b2 + w3 * b3
    print(f"[Visual] Final score={final_score:.4f}", flush=True)

    global_average = {
        "visual_final_score": float(final_score),
        "average_B1_background": float(b1),
        "average_B2_sensitivity": float(b2),
        "average_B3_cross_region": float(b3),
        "num_visual_answers": len(answers_by_id),
        "num_mapping_rows": len(mapping_rows),
    }

    summary = {
        "mode": "visual_answer_json",
        "visual_answer_json": str(visual_answer_json_path),
        "mapping_txt": str(mapping_txt_path),

        "average_B1_background": float(b1),
        "average_B2_sensitivity": float(b2),
        "average_B3_cross_region": float(b3),
        "final_visual_score": float(final_score),

        "global_average": global_average,

        "details": {
            "B1_background": b1_detail,
            "B2_sensitivity": b2_detail,
            "B3_cross_region": b3_detail,
        },
    }

    return summary

# ============================================================
# ALL-only CLI
# ============================================================
def extract_workflow_final_ranking_score(workflow_results: Dict[str, Any]) -> float:
    """
    Extract the official workflow score.

    For challenge use, each submission should ideally correspond to one prediction JSON.
    If DEFAULT_MERGE_PREDICTIONS=True, use the merged result.
    If DEFAULT_MERGE_PREDICTIONS=False and multiple prediction files are evaluated,
    use the mean over prediction files.
    """

    # Case 1: merged prediction files
    if workflow_results.get("merge_predictions", False):
        return float(
            workflow_results
            .get("result", {})
            .get("final_ranking_score", 0.0)
        )

    # Case 2: non-merged prediction files, use leaderboard mean
    leaderboard = workflow_results.get("leaderboard", [])

    if not leaderboard:
        return 0.0

    return float(
        sum(float(x.get("final_ranking_score", 0.0)) for x in leaderboard)
        / len(leaderboard)
    )


def extract_visual_final_score(visual_results: Dict[str, Any]) -> float:
    """
    Extract the official visual grounding score.

    This is:
        final_score = w1 * B1 + w2 * B2 + w3 * B3

    In run_visual_dataset, this is stored as final_visual_score.
    """
    return float(visual_results.get("final_visual_score", 0.0))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ALL-only evaluator: always run workflow metric and visual grounding metric."
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        nargs="+",
        required=True,
        help="Ground-truth workflow JSON file(s) or directory.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction workflow JSON file(s) or directory.",
    )

    parser.add_argument(
        "--visual-json",
        type=Path,
        required=True,
        help="Participant visual answer JSON. Each item contains id, image, question, answer.",
    )

    parser.add_argument(
        "--visual-mapping-txt",
        type=Path,
        required=True,
        help="Internal ROI mapping txt (rois_mapping.txt).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_scores.json"),
        help="Output JSON path. Default: all_scores.json",
    )


    parser.add_argument(
        "--judge-model-path",
        type=str,
        default=DEFAULT_JUDGE_MODEL_PATH,
        help=f"Path to local judge LLM. Default: {DEFAULT_JUDGE_MODEL_PATH}",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device. Default: {DEFAULT_DEVICE}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("[Evaluator] Running ALL metrics")
    print("=" * 100)

    print_runtime_diagnostics(args.device)

    print(f"[GT]          {args.ground_truth}")
    print(f"[Prediction]  {args.predictions}")
    print(f"[Visual JSON] {args.visual_json}")
    print(f"[Visual Mapping TXT] {args.visual_mapping_txt}")
    print(f"[Output]      {args.output}")
    print(f"[Judge LLM]   {args.judge_model_path}")
    print(f"[Device]      {args.device}")

    print("-" * 100)
    print("[Config]")
    print(f"workflow_semantic_backend = {DEFAULT_WORKFLOW_SEMANTIC_BACKEND}")
    print(f"embedding_model            = {DEFAULT_EMBEDDING_MODEL}")
    print(f"merge_predictions          = {DEFAULT_MERGE_PREDICTIONS}")
    print(f"strict_missing_predictions = {DEFAULT_STRICT_MISSING_PREDICTIONS}")
    print(f"visual_weights             = w1={DEFAULT_W1}, w2={DEFAULT_W2}, w3={DEFAULT_W3}")
    print(f"voting                     = {DEFAULT_VOTING}")
    print(f"skip_missing_roi           = {DEFAULT_SKIP_MISSING_ROI}")
    print("=" * 100)

    # ------------------------------------------------------------
    # Load judge LLM
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[0/2] Loading judge LLM")
    print("=" * 100)

    judge_llm = LocalQwenJudgeLLM(
        model_path=args.judge_model_path,
        device=args.device,
        max_new_tokens=DEFAULT_JUDGE_MAX_NEW_TOKENS,
    )
    print("[0/2] Judge LLM ready.", flush=True)

    # ------------------------------------------------------------
    # Run workflow metric
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[1/2] Running workflow reasoning metric")
    print("=" * 100)

    workflow_results = run_workflow_batch(
        ground_truth_paths=args.ground_truth,
        prediction_paths=args.predictions,
        semantic_backend=DEFAULT_WORKFLOW_SEMANTIC_BACKEND,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        judge_llm=judge_llm if DEFAULT_WORKFLOW_SEMANTIC_BACKEND == "llm" else None,
        voting=DEFAULT_VOTING,
        strict_missing_predictions=DEFAULT_STRICT_MISSING_PREDICTIONS,
        merge_predictions=DEFAULT_MERGE_PREDICTIONS,
    )
    workflow_final_ranking_score = extract_workflow_final_ranking_score(workflow_results)
    print(
        f"[1/2] Workflow metric complete. "
        f"final_ranking_score={workflow_final_ranking_score:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Run visual grounding metric
    # ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[2/2] Running visual grounding metric")
    print("=" * 100)

    visual_results = run_visual_dataset_from_answer_json(
        visual_answer_json_path=args.visual_json,
        mapping_txt_path=args.visual_mapping_txt,
        judge_llm=judge_llm,
        w1=DEFAULT_W1,
        w2=DEFAULT_W2,
        w3=DEFAULT_W3,
        voting=DEFAULT_VOTING,
    )
    visual_final_score = extract_visual_final_score(visual_results)
    print(
        f"[2/2] Visual metric complete. final_visual_score={visual_final_score:.4f}",
        flush=True,
    )

    # submission_id:
    # If one prediction JSON is provided, use its filename stem.
    # If a prediction directory or multiple files are provided, use "submission".
    pred_files = collect_json_files(args.predictions)
    if len(pred_files) == 1:
        submission_id = pred_files[0].stem
    else:
        submission_id = "submission"

    results = {
        "submission_id": submission_id,

        # These are the two official challenge metrics.
        "official_scores": {
            "workflow_final_ranking_score": float(workflow_final_ranking_score),
            "visual_final_score": float(visual_final_score),
        },

        "score_definitions": {
            "workflow_final_ranking_score": (
                "0.35 * Binary Path Validity + "
                "0.35 * Edge-F1 + "
                "0.30 * MESS"
            ),
            "visual_final_score": (
                f"{DEFAULT_W1} * B1_background + "
                f"{DEFAULT_W2} * B2_sensitivity + "
                f"{DEFAULT_W3} * B3_cross_region"
            ),
        },

        "config": {
                "workflow_semantic_backend": DEFAULT_WORKFLOW_SEMANTIC_BACKEND,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "merge_predictions": DEFAULT_MERGE_PREDICTIONS,
                "strict_missing_predictions": DEFAULT_STRICT_MISSING_PREDICTIONS,
                "visual_weights": {
                    "w1": DEFAULT_W1,
                    "w2": DEFAULT_W2,
                    "w3": DEFAULT_W3,
                },
                "voting": DEFAULT_VOTING,
                "judge_model_path": args.judge_model_path,
                "device": args.device,
            },

        # Keep full details for debugging, not for official leaderboard display.
        "details": {
            "workflow": workflow_results,
            "visual": visual_results,
        },
    }

    print("\n" + "=" * 100)
    print("[Done] Official challenge scores")
    print("=" * 100)
    print(json.dumps(results["official_scores"], indent=2, ensure_ascii=False))

    print(f"[Done] Saving results to {args.output}...", flush=True)
    save_json(results, args.output)
    print("[Done] Evaluation finished successfully.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] Evaluation crashed:", flush=True)
        traceback.print_exc()
        sys.exit(1)
