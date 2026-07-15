#!/usr/bin/env python3
"""Build text-only answer + report calibrator assets from the 11220 CoT cases.

Outputs (under ``--out``):
- ``answer_calibrator.json``: per-edge mode answer for each context bucket.
- ``report_calibrator.json``: per-label-bucket mode final report + global mode.
- ``split.json``: deterministic stratified train/val case ids.
- ``manifest.json``: build stats and coverage.

The split is built first; all priors are accumulated on TRAIN cases only so the
VAL split is a clean held-out set for measuring MESS / final-report ceilings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from reg2_text_calibrator import (  # noqa: E402
    LabelContext,
    answer_bucket_keys,
    canonicalize_question,
    edge_key,
    path_bucket_keys,
    report_bucket_keys,
)

DEFAULT_COT = Path("data/train_CoT_v01.json")
DEFAULT_OUT = Path("runs/calibrator_assets/v1")

FINAL_REPORT_Q = "what is the final pathology report"
ORGAN_Q = "what is the organ"
PROCEDURE_Q = "what is the procedure"
HISTOLOGIC_TYPE_Q = "what is the histologic type of neoplasm"
GRADE_Q = "what is the grade of neoplasm"
BEHAVIOR_Q = "what is the behavior of neoplasm"
DIAGNOSIS_Q_RE = re.compile(r"^what is the #(\d+) diagnosis$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cot", type=Path, default=DEFAULT_COT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--val-pct", type=int, default=15, help="Percent of cases held out for validation.")
    p.add_argument("--min-report-support", type=int, default=1,
                   help="Minimum train support for a report bucket to be kept.")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


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
    steps: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return steps
    for step in raw:
        if not isinstance(step, dict):
            continue
        q = clean(step.get("question"))
        if not q:
            continue
        steps.append(
            {
                "question": q,
                "answer": clean(step.get("answer")),
                "next_question": clean(step.get("next_question")),
            }
        )
    return steps


def context_from_steps(steps: list[dict[str, str]], organ_hint: str = "") -> LabelContext:
    organ = ""  # prefer the canonical "What is the organ?" answer; hint is fallback
    procedure = histologic_type = grade = behavior = ""
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
    ordered = [diagnoses[k] for k in sorted(diagnoses)]
    return LabelContext(
        organ=organ or organ_hint,
        procedure=procedure,
        diagnoses=ordered,
        histologic_type=histologic_type,
        grade=grade,
        behavior=behavior,
    )


def assign_split(case_id: str, val_pct: int) -> str:
    h = int(hashlib.sha1(case_id.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if h < val_pct else "train"


def main() -> int:
    args = parse_args()
    cases = load_json(args.cot)
    if not isinstance(cases, list):
        raise TypeError(f"Expected list in {args.cot}")

    # edge_answer_counts[edge_key][bucket] -> Counter(answer)
    edge_answer_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    # report_counts[bucket] -> Counter(report)
    report_counts: dict[str, Counter] = defaultdict(Counter)
    global_report_counts: Counter = Counter()
    # path_counts[bucket] -> Counter(edge-path signature)
    path_counts: dict[str, Counter] = defaultdict(Counter)
    global_path_counts: Counter = Counter()
    # signature tuple -> list of [question, next_question] edges (for reconstruction)
    path_edges_by_sig: dict[tuple, list[list[str]]] = {}

    split: dict[str, list[str]] = {"train": [], "val": []}
    organ_split: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    n_train_with_report = 0
    seen_ids: set[str] = set()

    for case in cases:
        if not isinstance(case, dict):
            continue
        cid = normalize_case_id(case.get("id", ""))
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        steps = get_steps(case)
        if not steps:
            continue
        which = assign_split(cid, args.val_pct)
        split[which].append(cid)

        ctx = context_from_steps(steps, organ_hint=clean(case.get("organ")))
        organ_split[which][ctx.organ or "<none>"] += 1

        if which != "train":
            continue  # accumulate priors on train only

        # final report
        final_report = ""
        for step in reversed(steps):
            if canonicalize_question(step["question"]) == FINAL_REPORT_Q:
                final_report = step["answer"]
                break

        # workflow edge-path signature (full ordered (question, next_question) list)
        edges = [[step["question"], step["next_question"]] for step in steps]
        sig = tuple((e[0], e[1]) for e in edges)
        path_edges_by_sig.setdefault(sig, edges)
        global_path_counts[sig] += 1
        for bucket in path_bucket_keys(ctx):
            path_counts[bucket][sig] += 1

        a_buckets = answer_bucket_keys(ctx)
        for step in steps:
            cq = canonicalize_question(step["question"])
            if cq == FINAL_REPORT_Q:
                continue
            if not step["answer"]:
                continue
            ek = edge_key(step["question"], step["next_question"])
            for bucket in a_buckets:
                edge_answer_counts[ek][bucket][step["answer"]] += 1

        if final_report:
            n_train_with_report += 1
            global_report_counts[final_report] += 1
            for bucket in report_bucket_keys(ctx):
                report_counts[bucket][final_report] += 1

    # Reduce counters to mode strings.
    edge_answer_modes: dict[str, dict[str, str]] = {}
    for ek, buckets in edge_answer_counts.items():
        reduced: dict[str, str] = {}
        for bucket, counter in buckets.items():
            ans, _ = counter.most_common(1)[0]
            reduced[bucket] = ans
        edge_answer_modes[ek] = reduced

    report_modes: dict[str, str] = {}
    for bucket, counter in report_counts.items():
        rep, support = counter.most_common(1)[0]
        if support >= args.min_report_support:
            report_modes[bucket] = rep
    global_report = global_report_counts.most_common(1)[0][0] if global_report_counts else ""

    path_modes: dict[str, list[list[str]]] = {}
    for bucket, counter in path_counts.items():
        sig, _ = counter.most_common(1)[0]
        path_modes[bucket] = path_edges_by_sig[sig]
    global_path = path_edges_by_sig[global_path_counts.most_common(1)[0][0]] if global_path_counts else []

    write_json(
        args.out / "path_calibrator.json",
        {
            "num_path_buckets": len(path_modes),
            "path_modes": path_modes,
            "global_path": global_path,
        },
    )
    write_json(
        args.out / "answer_calibrator.json",
        {
            "num_edges": len(edge_answer_modes),
            "edge_answer_modes": edge_answer_modes,
        },
    )
    write_json(
        args.out / "report_calibrator.json",
        {
            "num_report_buckets": len(report_modes),
            "report_modes": report_modes,
            "global_report": global_report,
        },
    )
    write_json(
        args.out / "split.json",
        {
            "val_pct": args.val_pct,
            "num_train": len(split["train"]),
            "num_val": len(split["val"]),
            "train_case_ids": sorted(split["train"]),
            "val_case_ids": sorted(split["val"]),
        },
    )
    manifest = {
        "cot_source": args.cot.name,
        "num_cases": len(seen_ids),
        "num_train": len(split["train"]),
        "num_val": len(split["val"]),
        "num_train_with_report": n_train_with_report,
        "num_edge_keys": len(edge_answer_modes),
        "num_report_buckets": len(report_modes),
        "num_path_buckets": len(path_modes),
        "num_unique_global_reports": len(global_report_counts),
        "train_organ_distribution": dict(organ_split["train"].most_common()),
        "val_organ_distribution": dict(organ_split["val"].most_common()),
        "min_report_support": args.min_report_support,
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote calibrator assets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
