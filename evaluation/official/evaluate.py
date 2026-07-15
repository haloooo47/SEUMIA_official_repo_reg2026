"""
The following is a simple example evaluation method.

It is meant to run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the evaluation and reads from ./test/input and writes to ./test/output

Evaluation entrypoint.

Reads `/input/predictions.json`, adapts submissions to the same APIs used by
`evaluate_metrics.py` (`scripts_final` logic), writes `/output/metrics.json`.

"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import evaluate_metrics
from helpers import setup_logger, tree

logger = logging.getLogger("evaluate")


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

_TMP_WORKFLOW_PRED = Path("/tmp/submission_workflow.json")
_TMP_VISUAL_PRED = Path("/tmp/submission_visual.json")

# When true, metrics.json includes anonymized per-case workflow scores under "results".
_INCLUDE_PER_CASE_RESULTS_ENV = "INCLUDE_PER_CASE_RESULTS"
# When true, metrics.json includes the verbose "details" object (workflow/visual trees, config).
_INCLUDE_EVALUATION_DETAILS_ENV = "INCLUDE_EVALUATION_DETAILS"

# Public aggregate / results keys (leaderboard-facing; no "average_" prefix).
KEY_BINARY_PATH_VALIDITY = "binary_path_validity"
KEY_EDGE_F1 = "edge_f1"
KEY_MESS = "mess"
KEY_FINAL_REPORT_SCORE = "final_report_score"
KEY_BACKGROUND_REJECTION = "background_rejection"
KEY_INPUT_SENSITIVITY = "input_sensitivity"
KEY_CROSS_REGION_CONSISTENCY = "cross_region_consistency"

# Per-WSI rows in metrics.json "results" use the same workflow submetrics as aggregates.
_WORKFLOW_PER_CASE_SCORE_KEYS = (
    KEY_BINARY_PATH_VALIDITY,
    KEY_EDGE_F1,
    KEY_MESS,
    KEY_FINAL_REPORT_SCORE,
    "workflow_final_ranking_score",
)

_METRIC_A_GT = (
    GROUND_TRUTH_DIRECTORY / "metric_A" / "chain-of-thoughts-ground-truth.json"
)
_METRIC_B_MAPPING = GROUND_TRUTH_DIRECTORY / "metric_B" / "rois_mapping.txt"
_MANIFEST = GROUND_TRUTH_DIRECTORY / "manifest.json"


INTERF0_SOCKETS = (
    "histopathology-region-of-interest-thumbnail",
    "visual-context-question",
)
INTERF1_SOCKETS = ("whole-slide-image",)


def main() -> int:
    setup_logger(level=logging.INFO)
    print("=" * 100, flush=True)
    print("[Evaluator] Container entrypoint starting", flush=True)
    print("=" * 100, flush=True)
    print(f"[Paths] INPUT_DIRECTORY={INPUT_DIRECTORY}", flush=True)
    print(f"[Paths] OUTPUT_DIRECTORY={OUTPUT_DIRECTORY}", flush=True)
    print(f"[Paths] GROUND_TRUTH_DIRECTORY={GROUND_TRUTH_DIRECTORY}", flush=True)

    judge_model_path = os.environ.get(
        "JUDGE_MODEL_PATH",
        str(GROUND_TRUTH_DIRECTORY / "Qwen3-14B"),
    )
    judge_device = evaluate_metrics.resolve_judge_device(
        os.environ.get("JUDGE_DEVICE")
    )
    _emb_env = os.environ.get(
        "EMBEDDING_MODEL", evaluate_metrics.DEFAULT_EMBEDDING_MODEL
    )
    embedding_for_reg25 = (
        _emb_env.strip() or evaluate_metrics.DEFAULT_EMBEDDING_MODEL
    )

    print(f"[Config] JUDGE_MODEL_PATH={judge_model_path!r}", flush=True)
    print(f"[Config] JUDGE_DEVICE env={os.environ.get('JUDGE_DEVICE')!r}", flush=True)
    print(f"[Config] resolved judge_device={judge_device!r}", flush=True)
    print(f"[Config] EMBEDDING_MODEL={embedding_for_reg25!r}", flush=True)
    evaluate_metrics.print_runtime_diagnostics(judge_device)

    print("[Step] Loading manifest and predictions...", flush=True)
    manifest = load_upload_manifest(_MANIFEST)
    predictions = read_predictions()
    print(
        f"[Step] Loaded manifest with {len(manifest.get('uploads', []))} upload(s), "
        f"{len(predictions)} prediction job(s)",
        flush=True,
    )

    pk_to_title = build_pk_to_title(manifest, predictions)
    print(f"[Step] Built pk_to_title mapping for {len(pk_to_title)} job(s)", flush=True)

    print(f"[Step] Loading visual ROI mapping from {_METRIC_B_MAPPING}...", flush=True)
    mapping_rows_dict = evaluate_metrics.load_rois_mapping_txt_as_dict(
        _METRIC_B_MAPPING
    )
    print(f"[Step] Loaded {len(mapping_rows_dict)} ROI mapping row(s)", flush=True)

    interf0_jobs = []
    interf1_jobs = []
    for job in predictions:
        ik = get_interface_key(job)
        if ik == tuple(sorted(INTERF0_SOCKETS)):
            interf0_jobs.append(job)
        elif ik == tuple(sorted(INTERF1_SOCKETS)):
            interf1_jobs.append(job)
        else:
            raise RuntimeError(f"Unknown interface {ik!r} for job {job.get('pk')!r}")

    print(
        f"[Step] Split jobs: {len(interf1_jobs)} workflow (interf1), "
        f"{len(interf0_jobs)} visual (interf0)",
        flush=True,
    )

    print(f"[Step] Building workflow predictions -> {_TMP_WORKFLOW_PRED}", flush=True)
    build_workflow_predictions(interf1_jobs, pk_to_title, _TMP_WORKFLOW_PRED)
    print(f"[Step] Building visual predictions -> {_TMP_VISUAL_PRED}", flush=True)
    build_visual_predictions(
        interf0_jobs,
        pk_to_title,
        mapping_rows_dict,
        _TMP_VISUAL_PRED,
    )

    print("=" * 100, flush=True)
    print("[Evaluator] Running ALL metrics", flush=True)
    print("=" * 100, flush=True)
    print(f"[GT Workflow]  {_METRIC_A_GT}", flush=True)
    print(f"[Pred Workflow] {_TMP_WORKFLOW_PRED}", flush=True)
    print(f"[Pred Visual]   {_TMP_VISUAL_PRED}", flush=True)
    print(f"[Visual Map]    {_METRIC_B_MAPPING}", flush=True)
    print(f"[Manifest]      {_MANIFEST}", flush=True)
    print(f"[Judge LLM]     {judge_model_path}", flush=True)
    print(f"[Device]        {judge_device}", flush=True)
    print(f"[REG25 embed]   {embedding_for_reg25}", flush=True)
    print(
        f"[Config] workflow_backend={evaluate_metrics.DEFAULT_WORKFLOW_SEMANTIC_BACKEND}",
        flush=True,
    )
    print("=" * 100, flush=True)

    print("\n[0/2] Loading judge LLM", flush=True)
    judge_llm = evaluate_metrics.LocalQwenJudgeLLM(
        model_path=judge_model_path,
        device=judge_device,
        max_new_tokens=evaluate_metrics.DEFAULT_JUDGE_MAX_NEW_TOKENS,
    )
    print("[0/2] Judge LLM ready.", flush=True)

    print("\n[1/2] Running workflow reasoning metric", flush=True)
    workflow_backend = evaluate_metrics.DEFAULT_WORKFLOW_SEMANTIC_BACKEND
    workflow_results = evaluate_metrics.run_workflow_batch(
        ground_truth_paths=[_METRIC_A_GT],
        prediction_paths=[_TMP_WORKFLOW_PRED],
        semantic_backend=workflow_backend,
        embedding_model=embedding_for_reg25,
        judge_llm=(
            judge_llm if workflow_backend == "llm" else None
        ),
        voting=evaluate_metrics.DEFAULT_VOTING,
        strict_missing_predictions=evaluate_metrics.DEFAULT_STRICT_MISSING_PREDICTIONS,
        merge_predictions=evaluate_metrics.DEFAULT_MERGE_PREDICTIONS,
    )
    workflow_final_ranking_score = evaluate_metrics.extract_workflow_final_ranking_score(
        workflow_results
    )
    print(
        f"[1/2] Workflow metric complete. "
        f"final_ranking_score={workflow_final_ranking_score:.4f}",
        flush=True,
    )

    print("\n[2/2] Running visual grounding metric", flush=True)
    visual_results = evaluate_metrics.run_visual_dataset_from_answer_json(
        visual_answer_json_path=_TMP_VISUAL_PRED,
        mapping_txt_path=_METRIC_B_MAPPING,
        judge_llm=judge_llm,
        w1=evaluate_metrics.DEFAULT_W1,
        w2=evaluate_metrics.DEFAULT_W2,
        w3=evaluate_metrics.DEFAULT_W3,
        voting=evaluate_metrics.DEFAULT_VOTING,
    )
    visual_final_score = evaluate_metrics.extract_visual_final_score(visual_results)
    print(
        f"[2/2] Visual metric complete. final_visual_score={visual_final_score:.4f}",
        flush=True,
    )

    official_scores = {
        "workflow_final_ranking_score": float(workflow_final_ranking_score),
        "visual_final_score": float(visual_final_score),
    }
    aggregates = build_aggregates(
        workflow_results=workflow_results,
        visual_results=visual_results,
        official_scores=official_scores,
    )
    include_per_case = parse_bool_env(
        _INCLUDE_PER_CASE_RESULTS_ENV,
        default=False,
    )
    include_details = parse_bool_env(
        _INCLUDE_EVALUATION_DETAILS_ENV,
        default=False,
    )
    per_case_results = (
        build_per_case_results(workflow_results) if include_per_case else []
    )

    metrics: dict[str, Any] = {
        "aggregates": aggregates,
        "results": per_case_results,
    }
    if include_details:
        metrics["details"] = build_evaluation_details(
            workflow_results=workflow_results,
            visual_results=visual_results,
            official_scores=official_scores,
            aggregates=aggregates,
            workflow_backend=workflow_backend,
            embedding_for_reg25=embedding_for_reg25,
            judge_model_path=judge_model_path,
            judge_device=judge_device,
            include_per_case=include_per_case,
        )

    output_path = OUTPUT_DIRECTORY / "metrics.json"
    print(f"[Done] Writing metrics to {output_path}...", flush=True)
    write_json_file(location=output_path, content=metrics)

    print("\n" + "=" * 100, flush=True)
    print("[Done] Official challenge scores", flush=True)
    print("=" * 100, flush=True)
    print(json.dumps(aggregates, indent=2, ensure_ascii=False), flush=True)
    print("[Done] Evaluation finished successfully.", flush=True)

    return 0


def parse_bool_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def extract_workflow_submetrics(workflow_results: dict[str, Any]) -> dict[str, float]:
    if workflow_results.get("merge_predictions", False):
        summary = workflow_results.get("result", {})
    else:
        summary = workflow_results.get("global_average", {})

    return {
        KEY_BINARY_PATH_VALIDITY: float(
            summary.get("average_binary_path_validity", 0.0)
        ),
        KEY_EDGE_F1: float(summary.get("average_edge_f1", 0.0)),
        KEY_MESS: float(summary.get("average_mess_nonfinal", 0.0)),
        KEY_FINAL_REPORT_SCORE: float(
            summary.get("average_final_report_score", 0.0)
        ),
    }


def extract_visual_submetrics(visual_results: dict[str, Any]) -> dict[str, float]:
    global_avg = visual_results.get("global_average", visual_results)
    return {
        KEY_BACKGROUND_REJECTION: float(
            global_avg.get(
                "average_B1_background",
                visual_results.get("average_B1_background", 0.0),
            )
        ),
        KEY_INPUT_SENSITIVITY: float(
            global_avg.get(
                "average_B2_sensitivity",
                visual_results.get("average_B2_sensitivity", 0.0),
            )
        ),
        KEY_CROSS_REGION_CONSISTENCY: float(
            global_avg.get(
                "average_B3_cross_region",
                visual_results.get("average_B3_cross_region", 0.0),
            )
        ),
    }


def build_aggregates(
    *,
    workflow_results: dict[str, Any],
    visual_results: dict[str, Any],
    official_scores: dict[str, float],
) -> dict[str, float]:
    workflow_sub = extract_workflow_submetrics(workflow_results)
    visual_sub = extract_visual_submetrics(visual_results)
    workflow_final = float(official_scores["workflow_final_ranking_score"])
    visual_final = float(official_scores["visual_final_score"])
    overall_score = (
        evaluate_metrics.DEFAULT_OVERALL_A_WEIGHT * workflow_final
        + evaluate_metrics.DEFAULT_OVERALL_B_WEIGHT * visual_final
    )
    return {
        "overall_score": overall_score,
        "workflow_final_ranking_score": workflow_final,
        "visual_final_score": visual_final,
        **workflow_sub,
        **visual_sub,
    }


def build_evaluation_details(
    *,
    workflow_results: dict[str, Any],
    visual_results: dict[str, Any],
    official_scores: dict[str, float],
    aggregates: dict[str, float],
    workflow_backend: str,
    embedding_for_reg25: str,
    judge_model_path: str,
    judge_device: str,
    include_per_case: bool,
) -> dict[str, Any]:
    """Verbose evaluator output for debug / audit (omitted unless INCLUDE_EVALUATION_DETAILS)."""
    return {
        "submission_id": "submission",
        "official_scores": official_scores,
        "aggregates": aggregates,
        "score_definitions": {
            "overall_score": (
                f"{evaluate_metrics.DEFAULT_OVERALL_A_WEIGHT} * workflow_final_ranking_score + "
                f"{evaluate_metrics.DEFAULT_OVERALL_B_WEIGHT} * visual_final_score"
            ),
            "workflow_final_ranking_score": (
                f"{evaluate_metrics.DEFAULT_A_BPV_WEIGHT} * {KEY_BINARY_PATH_VALIDITY} + "
                f"{evaluate_metrics.DEFAULT_A_EDGEF1_WEIGHT} * {KEY_EDGE_F1} + "
                f"{evaluate_metrics.DEFAULT_A_MESS_NONFINAL_WEIGHT} * {KEY_MESS} + "
                f"{evaluate_metrics.DEFAULT_A_FINAL_REPORT_WEIGHT} * {KEY_FINAL_REPORT_SCORE}"
            ),
            "visual_final_score": (
                f"{evaluate_metrics.DEFAULT_W1} * {KEY_BACKGROUND_REJECTION} + "
                f"{evaluate_metrics.DEFAULT_W2} * {KEY_INPUT_SENSITIVITY} + "
                f"{evaluate_metrics.DEFAULT_W3} * {KEY_CROSS_REGION_CONSISTENCY}"
            ),
        },
        "config": {
            "workflow_semantic_backend": workflow_backend,
            "embedding_model": embedding_for_reg25,
            "merge_predictions": evaluate_metrics.DEFAULT_MERGE_PREDICTIONS,
            "strict_missing_predictions": evaluate_metrics.DEFAULT_STRICT_MISSING_PREDICTIONS,
            "visual_weights": {
                "w1": evaluate_metrics.DEFAULT_W1,
                "w2": evaluate_metrics.DEFAULT_W2,
                "w3": evaluate_metrics.DEFAULT_W3,
            },
            "voting": evaluate_metrics.DEFAULT_VOTING,
            "judge_model_path": judge_model_path,
            "device": judge_device,
            "include_per_case_results": include_per_case,
            "include_evaluation_details": True,
            "overall_weights": {
                "workflow": evaluate_metrics.DEFAULT_OVERALL_A_WEIGHT,
                "visual": evaluate_metrics.DEFAULT_OVERALL_B_WEIGHT,
            },
        },
        "workflow": workflow_results,
        "visual": visual_results,
    }


def _collect_workflow_per_case(workflow_results: dict[str, Any]) -> list[dict[str, Any]]:
    if workflow_results.get("merge_predictions", False):
        return list(workflow_results.get("result", {}).get("per_case", []))

    per_case: list[dict[str, Any]] = []
    for evaluation in workflow_results.get("evaluations", {}).values():
        per_case.extend(evaluation.get("result", {}).get("per_case", []))
    return per_case


def _workflow_per_case_scores(entry: dict[str, Any]) -> dict[str, float]:
    """One WSI row: workflow submetrics only (same names as aggregates)."""
    values = {
        KEY_BINARY_PATH_VALIDITY: float(entry.get(KEY_BINARY_PATH_VALIDITY, 0.0)),
        KEY_EDGE_F1: float(entry.get(KEY_EDGE_F1, 0.0)),
        KEY_MESS: float(entry.get("mess_nonfinal", entry.get(KEY_MESS, 0.0))),
        KEY_FINAL_REPORT_SCORE: float(entry.get(KEY_FINAL_REPORT_SCORE, 0.0)),
        "workflow_final_ranking_score": float(
            entry.get("ranking_score", entry.get("workflow_final_ranking_score", 0.0))
        ),
    }
    return {key: values[key] for key in _WORKFLOW_PER_CASE_SCORE_KEYS}


def build_per_case_results(workflow_results: dict[str, Any]) -> list[dict[str, Any]]:
    """
    One anonymized record per WSI in metrics.json "results".

    Each record contains only workflow scores (no case_id, pk, or counts).
    """
    per_case = sorted(
        _collect_workflow_per_case(workflow_results),
        key=lambda entry: str(entry.get("case_id", "")),
    )

    results: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for entry in per_case:
        case_id = str(entry.get("case_id", ""))
        if not case_id:
            raise ValueError(
                "Workflow per_case entry missing case_id; cannot build results."
            )
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate workflow per_case case_id: {case_id!r}")
        seen_case_ids.add(case_id)
        results.append(_workflow_per_case_scores(entry))

    return results


def read_predictions() -> list:
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def load_upload_manifest(path: Path) -> dict[str, Any]:
    data = load_json_file(location=path)
    uploads = data.get("uploads")
    if not isinstance(uploads, list):
        raise ValueError(f"manifest.json must contain an uploads list: {path}")
    return data



def _input_signature_from_job_inputs(inputs: list) -> frozenset[tuple[Any, str]]:
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Each prediction job must include a non-empty inputs list")

    signature: set[tuple[Any, str]] = set()
    for i, inp in enumerate(inputs):
        if not isinstance(inp, dict):
            raise ValueError(f"Prediction inputs[{i}] must be an object")
        value_pk = inp.get("pk")
        interface = inp.get("interface") or {}
        slug = interface.get("slug")
        if value_pk is None or not slug:
            raise ValueError(
                f"Prediction inputs[{i}] must include pk and interface.slug"
            )
        signature.add((value_pk, str(slug)))
    return frozenset(signature)


def _input_signature_from_manifest_values(values: list) -> frozenset[tuple[Any, str]]:
    if not isinstance(values, list) or not values:
        raise ValueError("Each manifest upload must include display_set.values[]")

    signature: set[tuple[Any, str]] = set()
    for i, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"manifest display_set.values[{i}] must be an object")
        value_pk = value.get("pk")
        interface = value.get("interface") or {}
        slug = interface.get("slug")
        if value_pk is None or not slug:
            raise ValueError(
                f"manifest display_set.values[{i}] must include pk and interface.slug"
            )
        signature.add((value_pk, str(slug)))
    return frozenset(signature)


def _upload_case_title(entry: dict[str, Any]) -> str:
    display_set = entry.get("display_set") or {}
    case = entry.get("case") or {}
    title = display_set.get("title") or case.get("title")
    if not title:
        raise ValueError(
            "Each manifest upload must include display_set.title or case.title"
        )
    return str(title)


def _find_manifest_upload_by_input_signature(
    manifest: dict[str, Any], signature: frozenset[tuple[Any, str]]
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for entry in manifest.get("uploads", []):
        display_set = entry.get("display_set") or {}
        values = display_set.get("values")
        if not isinstance(values, list):
            continue
        if _input_signature_from_manifest_values(values) == signature:
            matches.append(entry)

    if not matches:
        raise ValueError(
            f"No manifest upload matches prediction input signature: {sorted(signature)!r}"
        )
    if len(matches) > 1:
        titles = [_upload_case_title(entry) for entry in matches]
        raise ValueError(
            "Multiple manifest uploads match prediction input signature "
            f"{sorted(signature)!r}: {titles!r}"
        )
    return matches[0]


def build_pk_to_title(
    manifest: dict[str, Any], predictions: list[dict[str, Any]]
) -> dict[str, str]:
    """
    Map anonymized prediction job pk -> canonical case title.

    Each job is matched to a manifest upload by comparing the full set of
    (inputs[].pk, inputs[].interface.slug) pairs against display_set.values[].
    """
    pk_to_title: dict[str, str] = {}
    for job in predictions:
        job_pk = job.get("pk")
        if not job_pk:
            raise ValueError("Each prediction job must include pk")

        signature = _input_signature_from_job_inputs(job.get("inputs") or [])
        upload = _find_manifest_upload_by_input_signature(manifest, signature)
        title = _upload_case_title(upload)
        job_pk_str = str(job_pk)

        if job_pk_str in pk_to_title and pk_to_title[job_pk_str] != title:
            raise ValueError(
                f"Conflicting case titles for job pk {job_pk_str!r}: "
                f"{pk_to_title[job_pk_str]!r} vs {title!r}"
            )
        pk_to_title[job_pk_str] = title

    return pk_to_title


def lookup_case_title(*, pk_to_title: dict[str, str], job_pk: str) -> str:
    title = pk_to_title.get(str(job_pk))
    if title is None:
        raise RuntimeError(
            f"Job pk {job_pk!r} not found in manifest.json. "
            "Ensure ground_truth/manifest.json matches the archive."
        )
    return title


def get_interface_key(job: dict) -> tuple[str, ...]:
    socket_slugs = [sv["socket"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_interface_relative_path(*, values: list, slug: str) -> str:
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]
    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk: str, values: list, slug: str) -> Path:
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location: Path):
    with open(location, encoding="utf-8") as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content) -> None:
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4, ensure_ascii=False)


def extract_visual_answer(raw: Any, *, resp_path: Path) -> str:
    """
    Participant visual-context-response payloads contain only the answer text.

    Accepted forms:
    - JSON string literal
    - plain text file
    - {"answer": "..."} (legacy)
    """
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return ""
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return extract_visual_answer(json.loads(stripped), resp_path=resp_path)
            except json.JSONDecodeError:
                return stripped
        return stripped

    if isinstance(raw, dict):
        if "answer" in raw:
            return str(raw["answer"] or "").strip()
        raise ValueError(
            f"{resp_path}: visual-context-response must contain only an answer, "
            f"got object keys {sorted(raw.keys())}."
        )

    raise ValueError(
        f"{resp_path}: unsupported visual-context-response payload type "
        f"{type(raw).__name__}."
    )


def extract_chain_of_thought_steps(raw: Any, *, cot_path: Path) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "chain-of-thought" in raw:
        steps = raw["chain-of-thought"]
        if isinstance(steps, list):
            return steps
    raise ValueError(
        f"{cot_path}: expected chain-of-thought steps as a JSON array, "
        f"got {type(raw).__name__}."
    )


def build_workflow_predictions(
    jobs: list,
    pk_to_title: dict[str, str],
    out_path: Path,
) -> None:
    cases = []
    for job in sorted(jobs, key=lambda j: str(j["pk"])):
        case_id = lookup_case_title(pk_to_title=pk_to_title, job_pk=str(job["pk"]))
        loc = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="chain-of-thought",
        )
        print(f"[Build Workflow] job pk={job['pk']}, case={case_id!r}, file={loc}", flush=True)
        raw = load_json_file(location=loc)
        steps = extract_chain_of_thought_steps(raw, cot_path=loc)
        cases.append({"id": case_id, "chain-of-thought": steps})
    write_json_file(location=out_path, content=cases)
    print(f"[Build Workflow] Wrote {len(cases)} case(s) to {out_path}", flush=True)


def build_visual_predictions(
    jobs: list,
    pk_to_title: dict[str, str],
    mapping_rows_dict: dict[str, dict[str, str]],
    out_path: Path,
) -> None:
    aggregated: dict[str, dict] = {}
    for job in sorted(jobs, key=lambda j: str(j["pk"])):
        anonymous_id = lookup_case_title(
            pk_to_title=pk_to_title, job_pk=str(job["pk"])
        )
        mapping_row = mapping_rows_dict.get(anonymous_id)
        if mapping_row is None:
            raise RuntimeError(
                f"Case title {anonymous_id!r} (job pk {job['pk']!r}) "
                f"not found in {_METRIC_B_MAPPING.name}."
            )

        resp_path = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="visual-context-response",
        )
        raw = load_json_file(location=resp_path)
        answer = extract_visual_answer(raw, resp_path=resp_path)

        entry = {
            "id": anonymous_id,
            "image": mapping_row["image"],
            "question": mapping_row["question"],
            "answer": answer,
        }

        if anonymous_id in aggregated:
            raise RuntimeError(f"Duplicate visual ROI id: {anonymous_id}")

        print(
            f"[Build Visual] job pk={job['pk']}, roi={anonymous_id!r}, file={resp_path}",
            flush=True,
        )
        aggregated[anonymous_id] = entry

    visual_list = [aggregated[k] for k in sorted(aggregated.keys())]
    write_json_file(location=out_path, content=visual_list)
    print(f"[Build Visual] Wrote {len(visual_list)} ROI answer(s) to {out_path}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\n[FATAL] Evaluation crashed:", flush=True)
        traceback.print_exc()
        sys.exit(1)
