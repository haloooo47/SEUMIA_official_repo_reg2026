#!/usr/bin/env python3
"""Build graph-aware REG2026 text calibrator assets.

It uses the same output schema as ``build_reg2026_calibrator_assets`` so
``TextCalibrator.from_assets`` can load it,
but the bucket keys are generated from ``LabelGraph`` instead of the smaller
``LabelContext``. The graph carries report-critical state that previously lived
in independent regex patches:

* secondary diagnoses / additional findings;
* Gleason and prostate volume slots;
* Nottingham and DCIS slots;
* microcalcification, invasion and additional-finding booleans;
* bladder muscle-proper notes.

The default split is shared with the base calibrator asset builder.
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
    GRAPH_FIELD_QUESTIONS,
    LabelContext,
    LabelGraph,
    answer_bucket_keys,
    canonicalize_question,
    edge_key,
    path_bucket_keys,
    report_bucket_keys,
)

DEFAULT_COT = Path("data/train_CoT_v01.json")
DEFAULT_V1_ASSETS = Path("runs/calibrator_assets/v1")
DEFAULT_OUT = Path("runs/calibrator_assets/v2_labelgraph")

FINAL_REPORT_Q = "what is the final pathology report"
ORGAN_Q = "what is the organ"
PROCEDURE_Q = "what is the procedure"
HISTOLOGIC_TYPE_Q = "what is the histologic type of neoplasm"
GRADE_Q = "what is the grade of neoplasm"
BEHAVIOR_Q = "what is the behavior of neoplasm"
DIAGNOSIS_Q_RE = re.compile(r"^what is the #(\d+) diagnosis$")

SLOT_QUESTION_TO_KEY = {
    "what is the gleason score": "gleason_score",
    "is there any gleason pattern 3 present": "gleason_pattern3_present",
    "is there any gleason pattern 4 present": "gleason_pattern4_present",
    "is there any gleason pattern 5 present": "gleason_pattern5_present",
    "what is the secondary pattern constituting more than 5% of tumor": "secondary_pattern_gt5",
    "what is the grade group": "grade_group",
    "what is the tumor volume": "tumor_volume",
    "what is the percentage of gleason pattern 4": "pattern4_percent",
    "what is the score for tubular differentiation": "breast_tubule_score",
    "what is the score for nuclear pleomorphism": "breast_nuclear_score",
    "what is the score for mitotic rate": "breast_mitotic_score",
    "what is the overall score": "breast_overall_score",
    "what is the nuclear grade of lesion": "dcis_nuclear_grade",
    "is there any necrosis present": "dcis_necrosis_present",
    "what is the type of necrosis": "dcis_necrosis_type",
    "what is the architectural pattern of lesion": "dcis_architectural_pattern",
    "what is the extent of invasion": "invasion_extent",
}
FINDING_QUESTION_TO_KEY = {
    "is there any microcalcification present": "microcalcification_present",
    "is there any invasion present": "invasion_present",
    "is there any additional finding present": "additional_finding_present",
}
NOTE_RE = re.compile(r"^\s*note\s*[\):：]?\s*(.*)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cot", type=Path, default=DEFAULT_COT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--v1-assets", type=Path, default=DEFAULT_V1_ASSETS)
    p.add_argument("--split", type=Path, default=DEFAULT_V1_ASSETS / "split.json")
    p.add_argument("--val-pct", type=int, default=15)
    p.add_argument("--min-report-support", type=int, default=1)
    p.add_argument("--use-all-train", action="store_true",
                   help="Accumulate priors from every CoT case for final packaging.")
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


def normalize_case_id(raw: Any) -> str:
    cid = clean(raw)
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


def assign_split(case_id: str, val_pct: int) -> str:
    h = int(hashlib.sha1(case_id.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if h < val_pct else "train"


def load_split(split_path: Path, val_pct: int) -> dict[str, str]:
    if not split_path.is_file():
        return {}
    raw = load_json(split_path)
    out: dict[str, str] = {}
    for cid in raw.get("train_case_ids", []):
        out[normalize_case_id(cid)] = "train"
    for cid in raw.get("val_case_ids", []):
        out[normalize_case_id(cid)] = "val"
    if not out and "case_splits" in raw:
        for cid, split in raw["case_splits"].items():
            out[normalize_case_id(cid)] = split
    return out


def final_report_from_steps(steps: list[dict[str, str]]) -> str:
    for step in reversed(steps):
        if canonicalize_question(step["question"]) == FINAL_REPORT_Q:
            return step["answer"]
    return ""


def note_from_report(report: str) -> str:
    text = str(report or "").replace("\\n", "\n")
    note_lines: list[str] = []
    in_note = False
    for line in text.splitlines():
        m = NOTE_RE.match(line)
        if m:
            in_note = True
            tail = clean(m.group(1))
            if tail:
                note_lines.append(tail)
            continue
        if in_note and line.strip():
            note_lines.append(line.strip())
    return clean(" ".join(note_lines))


def dx_family_map(v1_assets: Path) -> dict[str, str]:
    path = v1_assets / "structured_report_assets" / "labelgraph_taxonomy_v1.json"
    if not path.is_file():
        return {}
    obj = load_json(path)
    return {
        clean(dx): clean(rec.get("family"))
        for dx, rec in (obj.get("dx_to_family") or {}).items()
        if clean(dx) and isinstance(rec, dict)
    }


def merge_presence_answer(question: str, answers: list[str], has_secondary: bool) -> str:
    lows = [a.lower() for a in answers if a]
    if question == "is there any additional finding present" and has_secondary:
        return "Yes, there is an additional finding."
    if not lows:
        return ""
    if any("yes" in a for a in lows):
        if question == "is there any microcalcification present":
            return "Yes, there is a microcalcification."
        if question == "is there any invasion present":
            return "Yes, there is an invasion."
        if question == "is there any additional finding present":
            return "Yes, there is an additional finding."
    if question == "is there any microcalcification present":
        return "No, there is no microcalcification."
    if question == "is there any invasion present":
        return "No, there is no invasion."
    if question == "is there any additional finding present":
        return "No, there is no additional finding."
    return answers[0]


def context_from_steps(
    steps: list[dict[str, str]],
    *,
    organ_hint: str = "",
    family_by_dx: dict[str, str] | None = None,
) -> LabelGraph:
    organ = ""
    procedure = histologic_type = grade = behavior = ""
    diagnoses: dict[int, str] = {}
    slot_values: dict[str, str] = {}
    field_values: dict[str, set[str]] = defaultdict(set)
    edge_field_answers: dict[str, str] = {}
    finding_answers: dict[str, list[str]] = defaultdict(list)

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

        if cq in GRAPH_FIELD_QUESTIONS and a:
            edge_field_answers[edge_key(step["question"], step["next_question"])] = a
            field_values[cq].add(a)
            fkey = FINDING_QUESTION_TO_KEY.get(cq)
            if fkey:
                finding_answers[cq].append(a)
            skey = SLOT_QUESTION_TO_KEY.get(cq)
            if skey and skey not in slot_values:
                slot_values[skey] = a

    ordered = [diagnoses[k] for k in sorted(diagnoses)]
    secondary = ordered[1:] if len(ordered) > 1 else []
    field_answers = {
        q: next(iter(vals))
        for q, vals in field_values.items()
        if len(vals) == 1
    }
    finding_flags: dict[str, str] = {}
    for q, key in FINDING_QUESTION_TO_KEY.items():
        merged = merge_presence_answer(q, finding_answers.get(q, []), bool(secondary))
        if merged:
            finding_flags[key] = merged

    report = final_report_from_steps(steps)
    note = note_from_report(report)
    notes: dict[str, str] = {}
    if note:
        notes["note"] = note
        low = note.lower()
        if "muscle proper" in low:
            notes["muscle_proper"] = "not_included" if "does not include" in low else "included"

    primary = clean(ordered[0] if ordered else "")
    family = (family_by_dx or {}).get(primary, "")
    return LabelGraph(
        organ=organ or organ_hint,
        procedure=procedure,
        diagnoses=ordered,
        histologic_type=histologic_type,
        grade=grade,
        behavior=behavior,
        dx_family=family,
        slot_values=slot_values,
        finding_flags=finding_flags,
        secondary_findings=secondary,
        notes=notes,
        field_answers=field_answers,
        edge_field_answers=edge_field_answers,
    )


def reduce_counter_modes(counts: dict[str, Counter], min_support: int = 1) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, counter in counts.items():
        if not counter:
            continue
        value, support = counter.most_common(1)[0]
        if support >= min_support:
            out[key] = value
    return out


def main() -> int:
    args = parse_args()
    cases = load_json(args.cot)
    if not isinstance(cases, list):
        raise TypeError(f"Expected list in {args.cot}")

    split_by_id = load_split(args.split, args.val_pct)
    family_by_dx = dx_family_map(args.v1_assets)

    edge_answer_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    report_counts: dict[str, Counter] = defaultdict(Counter)
    global_report_counts: Counter = Counter()
    path_counts: dict[str, Counter] = defaultdict(Counter)
    global_path_counts: Counter = Counter()
    path_edges_by_sig: dict[tuple, list[list[str]]] = {}

    split: dict[str, list[str]] = {"train": [], "val": []}
    organ_split: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    graph_state_counts: Counter = Counter()
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
        which = split_by_id.get(cid) or assign_split(cid, args.val_pct)
        if args.use_all_train:
            which = "train"
        split.setdefault(which, []).append(cid)

        ctx = context_from_steps(
            steps,
            organ_hint=clean(case.get("organ")),
            family_by_dx=family_by_dx,
        )
        organ_split.setdefault(which, Counter())[ctx.organ or "<none>"] += 1
        if ctx.slot_values or ctx.finding_flags or ctx.secondary_findings or ctx.notes:
            graph_state_counts[ctx.organ or "<none>"] += 1

        if which != "train":
            continue

        final_report = final_report_from_steps(steps)
        edges = [[step["question"], step["next_question"]] for step in steps]
        sig = tuple((e[0], e[1]) for e in edges)
        path_edges_by_sig.setdefault(sig, edges)
        global_path_counts[sig] += 1
        for bucket in path_bucket_keys(ctx):
            path_counts[bucket][sig] += 1

        for step in steps:
            cq = canonicalize_question(step["question"])
            if cq == FINAL_REPORT_Q or not step["answer"]:
                continue
            ek = edge_key(step["question"], step["next_question"])
            for bucket in answer_bucket_keys(ctx):
                edge_answer_counts[ek][bucket][step["answer"]] += 1

        if final_report:
            n_train_with_report += 1
            global_report_counts[final_report] += 1
            for bucket in report_bucket_keys(ctx):
                report_counts[bucket][final_report] += 1

    edge_answer_modes: dict[str, dict[str, str]] = {}
    for ek, buckets in edge_answer_counts.items():
        edge_answer_modes[ek] = reduce_counter_modes(buckets)

    report_modes = reduce_counter_modes(report_counts, args.min_report_support)
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
            "num_train": len(split.get("train", [])),
            "num_val": len(split.get("val", [])),
            "train_case_ids": sorted(split.get("train", [])),
            "val_case_ids": sorted(split.get("val", [])),
            "use_all_train": bool(args.use_all_train),
        },
    )
    if (args.v1_assets / "dxset_prior.json").is_file():
        write_json(args.out / "dxset_prior.json", load_json(args.v1_assets / "dxset_prior.json"))

    manifest = {
        "version": "v2_labelgraph",
        "cot_source": args.cot.name,
        "split_source": args.split.name if args.split.is_file() else "hash",
        "use_all_train": bool(args.use_all_train),
        "num_cases": len(seen_ids),
        "num_train": len(split.get("train", [])),
        "num_val": len(split.get("val", [])),
        "num_train_with_report": n_train_with_report,
        "num_edge_keys": len(edge_answer_modes),
        "num_report_buckets": len(report_modes),
        "num_path_buckets": len(path_modes),
        "num_unique_global_reports": len(global_report_counts),
        "train_organ_distribution": dict(organ_split.get("train", Counter()).most_common()),
        "val_organ_distribution": dict(organ_split.get("val", Counter()).most_common()),
        "graph_state_cases_by_organ": dict(graph_state_counts.most_common()),
        "min_report_support": args.min_report_support,
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote labelgraph calibrator assets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
