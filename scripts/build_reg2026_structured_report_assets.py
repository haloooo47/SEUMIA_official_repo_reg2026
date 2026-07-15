#!/usr/bin/env python3
"""Build REG2026 structured-report assets and eligibility maps.

It intentionally uses only train CoT/final-report text.  The outputs are meant
to feed candidate generation and selector features, not to overwrite labels by
oracle information.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reg2_text_calibrator import canonicalize_question  # noqa: E402

DEFAULT_COT = Path("data/train_CoT_v01.json")
DEFAULT_OUT = Path("runs/calibrator_assets/structured_report_assets")

FINAL_REPORT_Q = "what is the final pathology report"
ORGAN_Q = "what is the organ"
PROCEDURE_Q = "what is the procedure"
DX_Q_RE = re.compile(r"^what is the #(\d+) diagnosis$")
NUMBERED_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
NOTE_RE = re.compile(r"^\s*note\s*[\):：]?\s*(.*)$", re.IGNORECASE)
SLOT_RE = re.compile(
    r"\b("
    r"tubule formation|nuclear grade|mitoses|gleason'?s? score|grade group|"
    r"gleason pattern 4|tumou?r volume|type|necrosis|percentage|score"
    r")\s*:?\s*([^,;\n)]+)",
    re.IGNORECASE,
)
LOW_INFO_ADDITIONAL = {
    "no tumor present",
    "no evidence of malignancy",
    "no dysplasia",
    "negative for malignancy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cot", type=Path, default=DEFAULT_COT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--val-pct", type=int, default=15)
    p.add_argument("--min-support", type=int, default=2)
    p.add_argument("--top-n", type=int, default=20)
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


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_key(value: Any) -> str:
    text = norm(value).lower()
    text = text.replace("\\n", "\n")
    text = re.sub(r"[\s,.;:()/-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slug(value: Any) -> str:
    key = norm_key(value)
    out = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return out or "blank"


def normalize_case_id(raw: Any) -> str:
    cid = norm(raw)
    for suffix in (".svs", ".tiff"):
        if cid.lower().endswith(suffix):
            return cid[: -len(suffix)]
    return cid


def get_steps(case: dict[str, Any]) -> list[dict[str, str]]:
    raw = case.get("chain-of-thought") or case.get("chain_of_thought") or []
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for step in raw:
        if not isinstance(step, dict):
            continue
        q = norm(step.get("question"))
        if q:
            out.append(
                {
                    "question": q,
                    "answer": norm(step.get("answer")),
                    "next_question": norm(step.get("next_question")),
                }
            )
    return out


def step_answer(steps: list[dict[str, str]], canonical_q: str) -> str:
    for step in steps:
        if canonicalize_question(step["question"]) == canonical_q:
            return step["answer"]
    return ""


def diagnoses_from_steps(steps: list[dict[str, str]]) -> list[str]:
    dx: dict[int, str] = {}
    for step in steps:
        m = DX_Q_RE.match(canonicalize_question(step["question"]))
        if m and step["answer"]:
            dx.setdefault(int(m.group(1)), step["answer"])
    return [dx[k] for k in sorted(dx)]


def final_report_from_steps(steps: list[dict[str, str]]) -> str:
    for step in reversed(steps):
        if canonicalize_question(step["question"]) == FINAL_REPORT_Q:
            return step["answer"].replace("\\n", "\n")
    return ""


def assign_split(case_id: str, val_pct: int) -> str:
    h = int(hashlib.sha1(case_id.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if h < val_pct else "train"


def split_header(report: str) -> tuple[str, str, str, str]:
    text = report.replace("\\n", "\n").strip()
    if ";" not in text:
        return "", "", "", text
    header, body = text.split(";", 1)
    header = norm(header)
    body = body.strip()
    if "," in header:
        organ, procedure = header.split(",", 1)
        return norm(organ), norm(procedure), header, body
    return "", "", header, body


def split_note(lines: list[str]) -> tuple[list[str], str]:
    body_lines: list[str] = []
    note_lines: list[str] = []
    in_note = False
    for line in lines:
        m = NOTE_RE.match(line)
        if m:
            in_note = True
            tail = norm(m.group(1))
            if tail:
                note_lines.append(tail)
            continue
        if in_note:
            note_lines.append(line.strip())
        else:
            body_lines.append(line)
    return body_lines, norm(" ".join(note_lines))


def parse_findings(body: str) -> tuple[list[str], str]:
    raw_lines = [line.rstrip() for line in body.splitlines() if line.strip()]
    lines, note = split_note(raw_lines)
    findings: list[str] = []
    current: list[str] = []
    saw_number = False
    for line in lines:
        m = NUMBERED_RE.match(line)
        if m:
            saw_number = True
            if current:
                findings.append(norm(" ".join(current)))
            current = [m.group(2)]
            continue
        if saw_number:
            current.append(line.strip())
        elif line.strip():
            current.append(line.strip())
    if current:
        findings.append(norm(" ".join(current)))
    if not findings and body.strip():
        findings.append(norm(body))
    return findings, note


def extract_slots(text: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    for m in SLOT_RE.finditer(text):
        key = norm_key(m.group(1))
        val = norm(m.group(2))
        if key and val and key not in slots:
            slots[key] = val
    return slots


def qualifier_with(finding: str) -> dict[str, str]:
    text = norm(finding)
    low = text.lower()
    if " with features of " in low:
        return {}
    m = re.match(r"^(.+?)\s+with\s+(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return {}
    return {"base": norm(m.group(1)), "with": norm(m.group(2))}


def parse_structured(case: dict[str, Any], val_pct: int) -> dict[str, Any]:
    steps = get_steps(case)
    report = final_report_from_steps(steps)
    header_organ, header_proc, header, body = split_header(report)
    findings, note = parse_findings(body)
    primary = findings[0] if findings else ""
    additional = findings[1:] if len(findings) > 1 else []
    dxs = diagnoses_from_steps(steps)
    if dxs:
        primary = primary or dxs[0]
        if not additional and len(dxs) > 1:
            additional = dxs[1:]
    with_bits = []
    for finding in findings:
        q = qualifier_with(finding)
        if q:
            with_bits.append(q)
    organ = step_answer(steps, ORGAN_Q) or header_organ or norm(case.get("organ"))
    procedure = step_answer(steps, PROCEDURE_Q) or header_proc
    return {
        "id": normalize_case_id(case.get("id")),
        "split": assign_split(normalize_case_id(case.get("id")), val_pct),
        "organ": organ,
        "procedure": procedure,
        "header": header,
        "primary": primary,
        "additional": additional,
        "additional_with": with_bits,
        "note": note,
        "diagnoses": dxs,
        "slots": extract_slots(report),
        "report": report,
    }


def key_variants(row: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    organ = norm_key(row.get("organ"))
    proc = norm_key(row.get("procedure"))
    primary = norm_key(row.get("primary"))
    dx0 = norm_key((row.get("diagnoses") or [""])[0])
    keys: list[tuple[str, tuple[str, ...]]] = []
    if organ and proc and primary:
        keys.append(("organ_proc_primary", (organ, proc, primary)))
    if organ and primary:
        keys.append(("organ_primary", (organ, primary)))
    if organ and proc and dx0:
        keys.append(("organ_proc_dx0", (organ, proc, dx0)))
    if organ and dx0:
        keys.append(("organ_dx0", (organ, dx0)))
    if organ:
        keys.append(("organ", (organ,)))
    return keys


def compact_counter(counter: Counter[str], total: int, min_support: int, top_n: int) -> list[dict[str, Any]]:
    out = []
    for value, count in counter.most_common(top_n):
        if count < min_support:
            continue
        out.append({"value": value, "count": int(count), "prob": round(count / max(1, total), 6)})
    return out


def build_eligibility(rows: list[dict[str, Any]], min_support: int, top_n: int) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    add_counts: dict[str, Counter[str]] = defaultdict(Counter)
    note_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with_counts: dict[str, Counter[str]] = defaultdict(Counter)
    report_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("split") != "train":
            continue
        for level, key_parts in key_variants(row):
            key = "|".join((level, *key_parts))
            totals[key] += 1
            report_counts[key][row.get("report", "")] += 1
            for finding in row.get("additional") or []:
                if norm_key(finding) not in LOW_INFO_ADDITIONAL:
                    add_counts[key][finding] += 1
            if row.get("note"):
                note_counts[key][row["note"]] += 1
            for item in row.get("additional_with") or []:
                if isinstance(item, dict) and item.get("with"):
                    with_counts[key][f"{item.get('base')} WITH {item.get('with')}"] += 1

    mappings: dict[str, Any] = {}
    for key, total in totals.items():
        mappings[key] = {
            "count": int(total),
            "additional": compact_counter(add_counts[key], total, min_support, top_n),
            "additional_with": compact_counter(with_counts[key], total, min_support, top_n),
            "note": compact_counter(note_counts[key], total, min_support, top_n),
            "reports": compact_counter(report_counts[key], total, min_support, min(top_n, 10)),
        }
    return {"min_support": min_support, "top_n": top_n, "mappings": mappings}


def build_inverted(rows: list[dict[str, Any]], min_support: int) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("split") != "train":
            continue
        organ = slug(row.get("organ"))
        for finding in row.get("additional") or []:
            counts[organ]["additional|" + finding] += 1
        for item in row.get("additional_with") or []:
            if isinstance(item, dict) and item.get("with"):
                counts[organ][f"additional_with|{item.get('base')} WITH {item.get('with')}"] += 1
        if row.get("note"):
            counts[organ]["note|" + row["note"]] += 1
    out: dict[str, Any] = {}
    for organ, counter in counts.items():
        organ_map: dict[str, Any] = {}
        for raw, count in counter.items():
            if count < min_support:
                continue
            kind, value = raw.split("|", 1)
            organ_map[f"{organ}_{kind}_{slug(value)}:present"] = {
                "finding_string": value,
                "finding_type": kind,
                "predicted_class": "present",
                "is_absence": False,
                "count": int(count),
            }
        out[organ] = {"organ": organ, "inverted_findings": organ_map, "total_findings": len(organ_map)}
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    organ_counts = Counter(row.get("organ") or "<none>" for row in rows)
    split_counts = Counter(row.get("split") for row in rows)
    numbered = sum(1 for row in rows if len(row.get("additional") or []) > 0)
    note = sum(1 for row in rows if row.get("note"))
    with_q = sum(1 for row in rows if row.get("additional_with"))
    add_counter = Counter()
    note_counter = Counter()
    for row in rows:
        for finding in row.get("additional") or []:
            add_counter[(row.get("organ") or "<none>", finding)] += 1
        if row.get("note"):
            note_counter[(row.get("organ") or "<none>", row["note"])] += 1
    return {
        "num_rows": len(rows),
        "split_counts": dict(split_counts),
        "organ_counts": dict(organ_counts.most_common()),
        "reports_with_additional": numbered,
        "reports_with_note": note,
        "reports_with_with_qualifier": with_q,
        "top_additional": [
            {"organ": k[0], "finding": k[1], "count": int(v)}
            for k, v in add_counter.most_common(30)
        ],
        "top_notes": [
            {"organ": k[0], "note": k[1], "count": int(v)}
            for k, v in note_counter.most_common(30)
        ],
    }


def main() -> int:
    args = parse_args()
    cases = load_json(args.cot)
    if not isinstance(cases, list):
        raise TypeError(f"expected list in {args.cot}")
    rows = [parse_structured(case, args.val_pct) for case in cases if isinstance(case, dict)]
    rows = [row for row in rows if row.get("id") and row.get("report")]
    args.out.mkdir(parents=True, exist_ok=True)

    write_json(args.out / "structured_reports.json", rows)
    by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_organ[slug(row.get("organ"))].append(row)
    for organ, organ_rows in by_organ.items():
        write_json(args.out / f"structured_reports_{organ}.json", organ_rows)

    eligibility = build_eligibility(rows, args.min_support, args.top_n)
    inverted = build_inverted(rows, args.min_support)
    summary = summarize(rows)
    manifest = {
        "cot": args.cot.name,
        "val_pct": args.val_pct,
        "min_support": args.min_support,
        "top_n": args.top_n,
        **summary,
        "num_eligibility_keys": len(eligibility["mappings"]),
    }
    write_json(args.out / "eligibility_map.json", eligibility)
    write_json(args.out / "inverted_findings.json", inverted)
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"[structured-assets] wrote -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
