"""Text-only answer + report calibrator for REG2026 Metric A.

This module is the single source of truth for the calibration logic shared by
the asset builder (``scripts/build_reg2026_calibrator_assets.py``) and online
inference. It targets the two largest Metric A sub-scores that need no WSI:

- MESS (0.25 of Metric A): the non-final-step answer semantic similarity.
- Final report score (0.40 of Metric A): the final pathology-report text.

The calibrator is given a chosen workflow path (the sequence of canonical
``(question, next_question)`` edges) plus a ``LabelContext`` (organ / procedure /
diagnoses / attributes that a visual model predicts). It then:

1. fills each non-final step's answer using deterministic field rules first,
   then a context-conditioned mode answer with backoff;
2. composes the final report by retrieving the most common training report for
   the most specific label bucket that has support, with backoff.

It has no heavy dependencies (pure standard library) so it can run inside the
submission container.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEP = "\u241f"  # visible unit separator, matches the asset builder
FINAL_REPORT_QUESTION_CANONICAL = "what is the final pathology report"

# Questions whose answer string is, by construction of the annotations, exactly
# the corresponding label value. Filling these from predicted labels yields the
# exact ground-truth answer under correct labels (MESS ~= 1.0 for those edges).
# Matches the canonicalized question (trailing "?" already stripped).
_DIAGNOSIS_Q_RE = re.compile(r"^what is the #(\d+) diagnosis$")

_GRAPH_SLOT_QUESTION_TO_KEY = {
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

_GRAPH_FINDING_QUESTION_TO_KEY = {
    "is there any microcalcification present": "microcalcification_present",
    "is there any invasion present": "invasion_present",
    "is there any additional finding present": "additional_finding_present",
}

GRAPH_FIELD_QUESTIONS = frozenset(
    set(_GRAPH_SLOT_QUESTION_TO_KEY) | set(_GRAPH_FINDING_QUESTION_TO_KEY)
)

_GRAPH_SLOT_ORDER = [
    "gleason_score",
    "grade_group",
    "gleason_pattern3_present",
    "gleason_pattern4_present",
    "gleason_pattern5_present",
    "secondary_pattern_gt5",
    "pattern4_percent",
    "tumor_volume",
    "breast_tubule_score",
    "breast_nuclear_score",
    "breast_mitotic_score",
    "breast_overall_score",
    "dcis_architectural_pattern",
    "dcis_nuclear_grade",
    "dcis_necrosis_present",
    "dcis_necrosis_type",
    "invasion_extent",
]
_GRAPH_FINDING_ORDER = [
    "microcalcification_present",
    "invasion_present",
    "additional_finding_present",
]
_GRAPH_REPORT_SLOT_GROUPS = [
    ("gleason", ["gleason_score"]),
    ("grade_group", ["grade_group"]),
    ("gleason_grade", ["gleason_score", "grade_group"]),
    ("pattern4", ["pattern4_percent"]),
    ("tumor_volume", ["tumor_volume"]),
    ("breast_nottingham", ["breast_tubule_score", "breast_nuclear_score", "breast_mitotic_score"]),
    ("breast_overall", ["breast_overall_score"]),
    ("dcis_grade", ["dcis_nuclear_grade"]),
    ("dcis_necrosis", ["dcis_necrosis_present", "dcis_necrosis_type"]),
    ("dcis_arch", ["dcis_architectural_pattern"]),
    ("invasion_extent", ["invasion_extent"]),
]
_EMPTY_GRAPH_VALUES = {"", "<none>", "<other>", "unknown", "not applicable", "n/a"}


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonicalize_question(text: Any) -> str:
    """Lowercased, whitespace-normalized, trailing-punctuation-stripped form.

    Mirrors the evaluator's question canonicalization closely enough to detect
    the final-report step robustly.
    """
    t = normalize_whitespace(text).lower()
    return re.sub(r"[\s.,;:!?]+$", "", t).strip()


def edge_key(question: Any, next_question: Any) -> str:
    return f"{normalize_whitespace(question)}{SEP}{normalize_whitespace(next_question)}"


def max_diagnosis_slot(path_steps: list[tuple[str, str]] | list[list[str]]) -> int:
    """Largest ``#k diagnosis`` slot referenced by a candidate path."""
    max_slot = 0
    for question, next_question in path_steps:
        for text in (question, next_question):
            m = _DIAGNOSIS_Q_RE.match(canonicalize_question(text))
            if m:
                max_slot = max(max_slot, int(m.group(1)))
    return max_slot


@dataclass
class LabelContext:
    """Clinical labels used to condition answers and reports.

    At training time these come from the ground-truth CoT. At inference they
    come from the visual specialist predictions.
    """

    organ: str = ""
    procedure: str = ""
    diagnoses: list[str] = field(default_factory=list)
    histologic_type: str = ""
    grade: str = ""
    behavior: str = ""

    def primary_dx(self) -> str:
        for d in self.diagnoses:
            d = normalize_whitespace(d)
            if d:
                return d
        return ""

    def clean_diagnoses(self) -> list[str]:
        return [normalize_whitespace(d) for d in self.diagnoses if normalize_whitespace(d)]


@dataclass
class LabelGraph(LabelContext):
    """Structured report state used by the leaderboard-oriented renderer.

    ``LabelContext`` is still the small visual-label baseline. ``LabelGraph``
    keeps the same public fields and adds the report-critical nodes that were
    previously patched with scattered regex tools: secondary findings, numeric
    slots, finding booleans, and note/procedure state. Empty graph fields are a
    no-op, so older assets and callers fall back to the original buckets.
    """

    dx_family: str = ""
    dx_candidates: list[Any] = field(default_factory=list)
    family_candidates: list[Any] = field(default_factory=list)
    slot_values: dict[str, str] = field(default_factory=dict)
    slot_confidence: dict[str, float] = field(default_factory=dict)
    finding_flags: dict[str, str] = field(default_factory=dict)
    secondary_findings: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    field_answers: dict[str, str] = field(default_factory=dict)
    edge_field_answers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_context(cls, context: LabelContext, **kwargs: Any) -> "LabelGraph":
        return cls(
            organ=context.organ,
            procedure=context.procedure,
            diagnoses=list(context.diagnoses),
            histologic_type=context.histologic_type,
            grade=context.grade,
            behavior=context.behavior,
            **kwargs,
        )

    def clean_secondary_findings(self) -> list[str]:
        values = self.secondary_findings or self.clean_diagnoses()[1:]
        return [normalize_whitespace(v) for v in values if _graph_value(v)]


def _graph_value(value: Any) -> str:
    text = normalize_whitespace(value).replace(SEP, " ")
    if text.lower() in _EMPTY_GRAPH_VALUES:
        return ""
    return text


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _mapping_signature(mapping: dict[str, Any], preferred_order: list[str]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for key in list(preferred_order) + sorted(str(k) for k in mapping):
        if key in seen:
            continue
        seen.add(key)
        value = _graph_value(mapping.get(key))
        if value:
            parts.append(f"{key}={value}")
    return SEP.join(parts)


def _group_signature(mapping: dict[str, Any], keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        value = _graph_value(mapping.get(key))
        if not value:
            return ""
        parts.append(f"{key}={value}")
    return SEP.join(parts)


def graph_slot_signature(context: LabelContext) -> str:
    if not isinstance(context, LabelGraph):
        return ""
    return _mapping_signature(context.slot_values, _GRAPH_SLOT_ORDER)


def graph_finding_signature(context: LabelContext) -> str:
    if not isinstance(context, LabelGraph):
        return ""
    return _mapping_signature(context.finding_flags, _GRAPH_FINDING_ORDER)


def graph_secondary_signature(context: LabelContext) -> str:
    if not isinstance(context, LabelGraph):
        return ""
    return SEP.join(context.clean_secondary_findings())


def graph_note_signature(context: LabelContext) -> str:
    if not isinstance(context, LabelGraph):
        return ""
    return _mapping_signature(context.notes, ["muscle_proper", "note"])


def graph_state_signature(context: LabelContext) -> str:
    if not isinstance(context, LabelGraph):
        return ""
    parts = []
    for prefix, sig in (
        ("S", graph_slot_signature(context)),
        ("F", graph_finding_signature(context)),
        ("D2", graph_secondary_signature(context)),
        ("N", graph_note_signature(context)),
    ):
        if sig:
            parts.append(f"{prefix}:{sig}")
    return SEP.join(parts)


def graph_report_pool_signature(context: LabelContext) -> str:
    """Coarse graph conditions for final-report retrieval.

    This deliberately avoids encoding exact slot values into the report key.
    Exact values are too sparse and noisy online; they should act as evidence
    that a report family is eligible, while L1-L6 labels keep the deterministic
    report floor stable.
    """
    if not isinstance(context, LabelGraph):
        return ""
    organ = normalize_whitespace(context.organ)
    slots = context.slot_values
    parts: list[str] = []
    if organ == "Prostate":
        if _graph_value(slots.get("gleason_score")):
            parts.append("prostate_gleason")
        if _graph_value(slots.get("tumor_volume")):
            parts.append("prostate_volume")
        if _graph_value(slots.get("pattern4_percent")):
            parts.append("prostate_pattern4")
    elif organ == "Breast":
        if all(_graph_value(slots.get(k)) for k in ("breast_tubule_score", "breast_nuclear_score", "breast_mitotic_score")):
            parts.append("breast_nottingham")
        if _graph_value(slots.get("dcis_nuclear_grade")) or _graph_value(slots.get("dcis_necrosis_present")):
            parts.append("breast_dcis")
    if context.clean_secondary_findings():
        parts.append("secondary")
    if graph_note_signature(context):
        parts.append("note")
    for key in ("microcalcification_present", "invasion_present", "additional_finding_present"):
        value = _graph_value(context.finding_flags.get(key))
        if value and value.lower().startswith("yes"):
            parts.append(key)
    return SEP.join(_unique(parts))


def graph_field_answer(question: Any, context: LabelContext) -> str | None:
    if not isinstance(context, LabelGraph):
        return None
    q = canonicalize_question(question)
    direct = _graph_value(context.field_answers.get(q))
    if direct:
        return direct
    slot_key = _GRAPH_SLOT_QUESTION_TO_KEY.get(q)
    if slot_key:
        value = _graph_value(context.slot_values.get(slot_key))
        if value:
            return value
    finding_key = _GRAPH_FINDING_QUESTION_TO_KEY.get(q)
    if finding_key:
        value = _graph_value(context.finding_flags.get(finding_key))
        if value:
            return value
        if finding_key == "additional_finding_present" and context.clean_secondary_findings():
            return "Yes, there is an additional finding."
    return None


def graph_edge_answer(question: Any, next_question: Any, context: LabelContext) -> str | None:
    if not isinstance(context, LabelGraph):
        return None
    return _graph_value(context.edge_field_answers.get(edge_key(question, next_question))) or None


def answer_bucket_keys(context: LabelContext) -> list[str]:
    """Most-specific-first context buckets for per-edge answer priors."""
    organ = normalize_whitespace(context.organ)
    dx = context.primary_dx()
    keys: list[str] = []
    if organ and dx:
        keys.append(f"organ_dx{SEP}{organ}{SEP}{dx}")
    if organ:
        keys.append(f"organ{SEP}{organ}")
    keys.append("global")
    return keys


def path_bucket_keys(context: LabelContext) -> list[str]:
    """Most-specific-first label buckets for workflow-path (edge sequence) retrieval.

    The set of questions a pathologist walks through is driven mainly by organ,
    procedure and the diagnosis (e.g. "No tumor present" -> short path; a
    malignant diagnosis -> long path with grade/invasion branches).
    """
    organ = normalize_whitespace(context.organ)
    proc = normalize_whitespace(context.procedure)
    dx = context.primary_dx()
    dx_set = SEP.join(context.clean_diagnoses())
    keys = [
        "P1" + SEP + SEP.join([organ, proc, dx_set]),
        "P2" + SEP + SEP.join([organ, proc, dx]),
        "P3" + SEP + SEP.join([organ, proc]),
        "P4" + SEP + SEP.join([organ, dx]),
        "P5" + SEP + organ,
        "global",
    ]
    return _unique(keys)


def report_bucket_keys(context: LabelContext) -> list[str]:
    """Most-specific-first label buckets for final-report retrieval."""
    organ = normalize_whitespace(context.organ)
    proc = normalize_whitespace(context.procedure)
    dx = context.primary_dx()
    dx_set = SEP.join(context.clean_diagnoses())
    histo = normalize_whitespace(context.histologic_type)
    grade = normalize_whitespace(context.grade)
    beh = normalize_whitespace(context.behavior)
    pool_sig = graph_report_pool_signature(context)
    family = _graph_value(context.dx_family) if isinstance(context, LabelGraph) else ""
    pool_keys: list[str] = []
    family_keys: list[str] = []
    if pool_sig:
        pool_keys.extend(
            [
                "GC1" + SEP + SEP.join([organ, proc, dx_set, histo, grade, beh, pool_sig]),
                "GC2" + SEP + SEP.join([organ, proc, dx_set, pool_sig]),
                "GC3" + SEP + SEP.join([organ, proc, dx, pool_sig]),
                "GC4" + SEP + SEP.join([organ, dx, pool_sig]),
            ]
        )
    if family:
        family_keys.append("GF1" + SEP + SEP.join([organ, proc, family, pool_sig]))
    dxset_keys = [
        "L1" + SEP + SEP.join([organ, proc, dx_set, histo, grade, beh]),
        "L2" + SEP + SEP.join([organ, proc, dx_set]),
    ]
    base_tail_keys = [
        "L3" + SEP + SEP.join([organ, proc, dx]),
        "L4" + SEP + SEP.join([organ, dx]),
        "L5" + SEP + SEP.join([organ, proc]),
        "L6" + SEP + organ,
        "global",
    ]
    keys = pool_keys + family_keys + dxset_keys + base_tail_keys
    return _unique(keys)


def field_answer(question: Any, context: LabelContext) -> str | None:
    """Return the deterministic answer for a field question, else ``None``.

    Only returns a value when the corresponding label is non-empty so callers
    can fall back to the conditional-mode prior.
    """
    q = canonicalize_question(question)
    ga = graph_field_answer(question, context)
    if ga:
        return ga
    if q == "what is the organ":
        return normalize_whitespace(context.organ) or None
    if q == "what is the procedure":
        return normalize_whitespace(context.procedure) or None
    if q == "what is the histologic type of neoplasm":
        return normalize_whitespace(context.histologic_type) or None
    if q == "what is the grade of neoplasm":
        return normalize_whitespace(context.grade) or None
    if q == "what is the behavior of neoplasm":
        return normalize_whitespace(context.behavior) or None
    if q == "what is the number of diagnoses to includes":
        dx_count = len(context.clean_diagnoses())
        return str(dx_count) if dx_count else None
    m = _DIAGNOSIS_Q_RE.match(q)
    if m:
        idx = int(m.group(1)) - 1
        dx = context.clean_diagnoses()
        if 0 <= idx < len(dx):
            return dx[idx]
    return None


class TextCalibrator:
    """Loads calibrator assets and renders calibrated answers + reports."""

    def __init__(
        self,
        answer_modes: dict[str, dict[str, str]],
        report_modes: dict[str, str],
        global_report: str = "",
        path_modes: dict[str, list[list[str]]] | None = None,
        global_path: list[list[str]] | None = None,
        dxset_prior: dict[str, list[str]] | None = None,
    ) -> None:
        # answer_modes[edge_key][bucket_key] -> answer string
        self.answer_modes = answer_modes
        # report_modes[bucket_key] -> report string
        self.report_modes = report_modes
        self.global_report = global_report
        # path_modes[bucket_key] -> list of [question, next_question] edges
        self.path_modes = path_modes or {}
        self.global_path = global_path or []
        # dxset_prior["<organ>\u241f<primary_dx>"] -> most common full diagnosis
        # list. Optional and backward compatible (no-op when absent).
        self.dxset_prior = dxset_prior or {}

    @classmethod
    def from_assets(cls, assets_dir: str | Path) -> "TextCalibrator":
        assets_dir = Path(assets_dir)
        answers = json.loads((assets_dir / "answer_calibrator.json").read_text(encoding="utf-8"))
        reports = json.loads((assets_dir / "report_calibrator.json").read_text(encoding="utf-8"))
        path_modes: dict[str, list[list[str]]] = {}
        global_path: list[list[str]] = []
        path_file = assets_dir / "path_calibrator.json"
        if path_file.is_file():
            paths = json.loads(path_file.read_text(encoding="utf-8"))
            path_modes = paths.get("path_modes", {})
            global_path = paths.get("global_path", [])
        dxset_prior: dict[str, list[str]] = {}
        dxset_file = assets_dir / "dxset_prior.json"
        if dxset_file.is_file():
            dxset_prior = json.loads(dxset_file.read_text(encoding="utf-8"))
        return cls(
            answer_modes=answers["edge_answer_modes"],
            report_modes=reports["report_modes"],
            global_report=reports.get("global_report", ""),
            path_modes=path_modes,
            global_path=global_path,
            dxset_prior=dxset_prior,
        )

    def expand_diagnoses(self, context: LabelContext) -> LabelContext:
        """Expand a single predicted primary_dx to the training-mode full list."""
        if not self.dxset_prior:
            return context
        primary = context.primary_dx()
        if not (context.organ and primary):
            return context
        if len(context.clean_diagnoses()) > 1:
            return context
        expanded = self.dxset_prior.get(edge_key(context.organ, primary))
        if expanded and len(expanded) > 1 and normalize_whitespace(expanded[0]) == primary:
            import dataclasses

            return dataclasses.replace(context, diagnoses=list(expanded))
        return context

    def select_path(self, context: LabelContext) -> list[tuple[str, str]]:
        """Return the most likely workflow path (edges) for the predicted labels."""
        dx_count = len(context.clean_diagnoses())
        first_available: list[tuple[str, str]] | None = None
        for bucket in path_bucket_keys(context):
            edges = self.path_modes.get(bucket)
            if edges:
                path = [(e[0], e[1]) for e in edges]
                if first_available is None:
                    first_available = path
                if dx_count <= 1 and max_diagnosis_slot(path) > max(dx_count, 1):
                    continue
                return path
        global_path = [(e[0], e[1]) for e in self.global_path]
        if dx_count <= 1 and global_path and max_diagnosis_slot(global_path) <= max(dx_count, 1):
            return global_path
        return first_available or global_path

    def render_from_labels(
        self,
        context: LabelContext,
        use_field_rules: bool = True,
    ) -> list[dict[str, str]]:
        """Full pipeline: select the path from labels, then fill answers + report."""
        context = self.expand_diagnoses(context)
        return self.render_chain(self.select_path(context), context, use_field_rules)

    def fill_answer(
        self,
        question: str,
        next_question: str,
        context: LabelContext,
        use_field_rules: bool = True,
    ) -> str:
        q = canonicalize_question(question)
        dxs = context.clean_diagnoses()
        m = _DIAGNOSIS_Q_RE.match(q)
        if m and int(m.group(1)) > len(dxs):
            return ""
        ga = graph_edge_answer(question, next_question, context)
        if ga:
            return ga
        if use_field_rules:
            fa = field_answer(question, context)
            if fa:
                return fa
        modes = self.answer_modes.get(edge_key(question, next_question))
        if modes:
            for bucket in answer_bucket_keys(context):
                ans = modes.get(bucket)
                if ans:
                    return ans
        return ""

    def compose_report(self, context: LabelContext) -> str:
        for bucket in report_bucket_keys(context):
            rep = self.report_modes.get(bucket)
            if rep:
                return rep
        return self.global_report

    def render_chain(
        self,
        path_steps: list[tuple[str, str]],
        context: LabelContext,
        use_field_rules: bool = True,
    ) -> list[dict[str, str]]:
        """Render a chain-of-thought from a path of (question, next_question) edges."""
        steps: list[dict[str, str]] = []
        for question, next_question in path_steps:
            if canonicalize_question(question) == FINAL_REPORT_QUESTION_CANONICAL:
                answer = self.compose_report(context)
            else:
                answer = self.fill_answer(question, next_question, context, use_field_rules)
            steps.append(
                {
                    "question": str(question),
                    "answer": answer,
                    "next_question": str(next_question),
                }
            )
        return steps
