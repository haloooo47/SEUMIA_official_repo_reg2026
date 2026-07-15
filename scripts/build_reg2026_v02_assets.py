#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COT = Path("data/train_CoT_v01.json")
DEFAULT_OUT = REPO_DIR / "models" / "reg2_v02"
TERMINAL_QUESTION = "What is the final pathology report?"
ORGAN_QUESTION = "What is the organ?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact REG2026 v0.2 JSON assets.")
    parser.add_argument("--cot", type=Path, default=DEFAULT_COT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-global-paths", type=int, default=64)
    parser.add_argument("--max-paths-per-organ", type=int, default=16)
    parser.add_argument("--max-answers", type=int, default=20)
    parser.add_argument("--max-reports", type=int, default=20)
    parser.add_argument(
        "--include-case-lookup",
        action="store_true",
        help="Write dev_case_lookup.json for explicit local debugging only; omit for deployable assets.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def get_steps(case: dict[str, Any]) -> list[dict[str, str]]:
    raw_steps = case.get("chain-of-thought") or case.get("chain_of_thought") or []
    steps: list[dict[str, str]] = []
    for step in raw_steps:
        question = str(step.get("question", "") or "").strip()
        answer = str(step.get("answer", "") or "").strip()
        next_question = str(step.get("next_question", "") or "").strip()
        if question:
            steps.append(
                {
                    "question": question,
                    "answer": answer,
                    "next_question": next_question,
                }
            )
    return steps


def get_case_id(case: dict[str, Any]) -> str:
    return str(case.get("id", "") or "").strip()


def get_organ(steps: list[dict[str, str]]) -> str:
    for step in steps:
        if step["question"] == ORGAN_QUESTION:
            return step["answer"]
    return ""


def get_final_report(steps: list[dict[str, str]]) -> str:
    for step in reversed(steps):
        if step["question"] == TERMINAL_QUESTION:
            return step["answer"]
    return ""


def path_signature(steps: list[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple((step["question"], step["next_question"]) for step in steps)


def edge_key(question: str, next_question: str) -> str:
    return f"{question}\u241f{next_question}"


def make_path_id(signature: tuple[tuple[str, str], ...]) -> str:
    payload = json.dumps(signature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"path_{hashlib.sha1(payload).hexdigest()[:12]}"


def top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": int(count)}
        for value, count in counter.most_common(limit)
    ]


def top_answer_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [
        {"answer": answer, "count": int(count)}
        for answer, count in counter.most_common(limit)
    ]


def top_report_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [
        {"report": report, "count": int(count)}
        for report, count in counter.most_common(limit)
    ]


def build_assets(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_json(args.cot)
    if not isinstance(cases, list):
        raise TypeError(f"Expected list in {args.cot}, got {type(cases).__name__}")

    question_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    organ_counts: Counter[str] = Counter()
    path_counts: Counter[tuple[tuple[str, str], ...]] = Counter()
    path_by_organ: dict[str, Counter[tuple[tuple[str, str], ...]]] = defaultdict(Counter)
    global_answers: dict[str, Counter[str]] = defaultdict(Counter)
    answers_by_organ: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    global_reports: Counter[str] = Counter()
    reports_by_organ: dict[str, Counter[str]] = defaultdict(Counter)
    path_infos: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    case_lookup: dict[str, list[dict[str, str]]] = {}

    valid_cases = 0
    step_counts: list[int] = []

    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = get_case_id(case)
        steps = get_steps(case)
        if not steps:
            continue

        valid_cases += 1
        step_counts.append(len(steps))

        organ = get_organ(steps)
        if organ:
            organ_counts[organ] += 1

        report = get_final_report(steps)
        if report:
            global_reports[report] += 1
            if organ:
                reports_by_organ[organ][report] += 1

        signature = path_signature(steps)
        path_counts[signature] += 1
        if organ:
            path_by_organ[organ][signature] += 1

        info = path_infos.setdefault(
            signature,
            {
                "first_case_id": case_id,
                "representative_steps": steps,
                "organ_counts": Counter(),
                "answers_by_edge": defaultdict(Counter),
                "final_reports": Counter(),
                "case_ids_sample": [],
            },
        )
        if len(info["case_ids_sample"]) < 20 and case_id:
            info["case_ids_sample"].append(case_id)
        if organ:
            info["organ_counts"][organ] += 1
        if report:
            info["final_reports"][report] += 1

        for step in steps:
            question = step["question"]
            answer = step["answer"]
            next_question = step["next_question"]
            question_counts[question] += 1
            edge_counts[(question, next_question)] += 1
            if answer:
                global_answers[question][answer] += 1
                info["answers_by_edge"][edge_key(question, next_question)][answer] += 1
                if organ:
                    answers_by_organ[organ][question][answer] += 1

        if case_id and args.include_case_lookup:
            case_lookup[case_id] = steps

    selected_signatures: set[tuple[tuple[str, str], ...]] = set()
    global_top_signatures = [sig for sig, _ in path_counts.most_common(args.max_global_paths)]
    selected_signatures.update(global_top_signatures)
    for organ, counter in path_by_organ.items():
        selected_signatures.update(sig for sig, _ in counter.most_common(args.max_paths_per_organ))

    paths: dict[str, Any] = {}
    signature_to_id: dict[tuple[tuple[str, str], ...], str] = {}
    for signature in selected_signatures:
        path_id = make_path_id(signature)
        signature_to_id[signature] = path_id
        info = path_infos[signature]
        step_templates = []
        for question, next_question in signature:
            answers = info["answers_by_edge"].get(edge_key(question, next_question), Counter())
            answer = answers.most_common(1)[0][0] if answers else ""
            step_templates.append(
                {
                    "question": question,
                    "answer": answer,
                    "next_question": next_question,
                }
            )
        organ_top = info["organ_counts"].most_common(1)
        paths[path_id] = {
            "path_id": path_id,
            "count": int(path_counts[signature]),
            "dominant_organ": organ_top[0][0] if organ_top else "",
            "organ_counts": top_counter(info["organ_counts"], 10),
            "steps": step_templates,
            "final_reports": top_report_counter(info["final_reports"], args.max_reports),
            "first_case_id": info["first_case_id"],
            "case_ids_sample": info["case_ids_sample"],
        }

    def path_ref(signature: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        return {
            "path_id": signature_to_id[signature],
            "count": int(path_counts[signature]),
            "dominant_organ": paths[signature_to_id[signature]]["dominant_organ"],
        }

    path_priors = {
        "default_mode": "global_top",
        "default_path_id": signature_to_id[global_top_signatures[0]] if global_top_signatures else "",
        "global_top_paths": [path_ref(sig) for sig in global_top_signatures if sig in signature_to_id],
        "by_organ": {
            organ: [
                path_ref(sig)
                for sig, _ in counter.most_common(args.max_paths_per_organ)
                if sig in signature_to_id
            ]
            for organ, counter in sorted(path_by_organ.items())
        },
        "paths": paths,
    }

    workflow_graph = {
        "terminal_question": TERMINAL_QUESTION,
        "questions": [
            {"question": question, "count": int(count)}
            for question, count in question_counts.most_common()
        ],
        "edges": [
            {"question": question, "next_question": next_question, "count": int(count)}
            for (question, next_question), count in edge_counts.most_common()
        ],
    }

    answer_priors = {
        "global": {
            question: top_answer_counter(counter, args.max_answers)
            for question, counter in sorted(global_answers.items())
        },
        "by_organ": {
            organ: {
                question: top_answer_counter(counter, args.max_answers)
                for question, counter in sorted(question_counters.items())
            }
            for organ, question_counters in sorted(answers_by_organ.items())
        },
    }

    report_priors = {
        "global": top_report_counter(global_reports, args.max_reports),
        "by_organ": {
            organ: top_report_counter(counter, args.max_reports)
            for organ, counter in sorted(reports_by_organ.items())
        },
        "by_path": {
            path_id: data.get("final_reports", [])[: args.max_reports]
            for path_id, data in paths.items()
        },
    }

    organ_priors = {
        "organs": [
            {"organ": organ, "count": int(count)}
            for organ, count in organ_counts.most_common()
        ]
    }

    roi_thresholds = {
        "tissue_fraction_min": 0.03,
        "artifact_brightness_max": 35.0,
        "background_brightness_min": 235.0,
        "chroma_min": 12.0,
        "uncertain_margin": 0.01,
    }

    manifest = {
        "pipeline_version": "v0.2",
        "cot_source": args.cot.name,
        "num_cases": valid_cases,
        "num_steps_min": min(step_counts) if step_counts else 0,
        "num_steps_mean": (sum(step_counts) / len(step_counts)) if step_counts else 0.0,
        "num_steps_max": max(step_counts) if step_counts else 0,
        "num_questions": len(question_counts),
        "num_edges": len(edge_counts),
        "num_unique_paths": len(path_counts),
        "num_selected_paths": len(paths),
        "case_lookup_included": bool(args.include_case_lookup),
        "default_path_id": path_priors["default_path_id"],
    }

    return {
        "manifest.json": manifest,
        "workflow_graph.json": workflow_graph,
        "path_priors.json": path_priors,
        "answer_priors.json": answer_priors,
        "report_priors.json": report_priors,
        "organ_priors.json": organ_priors,
        "roi_thresholds.json": roi_thresholds,
        "dev_case_lookup.json": case_lookup if args.include_case_lookup else {},
    }


def main() -> int:
    args = parse_args()
    assets = build_assets(args)
    for filename, data in assets.items():
        if filename == "dev_case_lookup.json" and not args.include_case_lookup:
            stale = args.out / filename
            if stale.exists():
                stale.unlink()
            continue
        write_json(args.out / filename, data)

    manifest = assets["manifest.json"]
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote assets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
