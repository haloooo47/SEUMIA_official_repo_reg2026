"""Patch final pathology reports with structured slot predictions.

Targets gate-passed fields only:
  * prostate Gleason score + grade group (derived if missing)
  * breast Nottingham tubule / nuclear / mitotic component scores
  * report-local DCIS/procedure consistency slots
"""

from __future__ import annotations

import re
from typing import Any

_GLEASON_RE = re.compile(r"Gleason'?s?\s*score\s*[^,;\n]+", re.IGNORECASE)
_GRADE_GROUP_RE = re.compile(r"grade group\s*\d+", re.IGNORECASE)
_NOTTINGHAM_RE = re.compile(
    r"\((Tubule formation:\s*)\d+(\s*,\s*Nuclear grade:\s*)\d+(\s*,\s*Mitoses:\s*)\d+\)",
    re.IGNORECASE,
)
_NST_GRADE_RE = re.compile(
    r"(Invasive carcinoma of no special type,\s*grade\s*)"
    r"(I{1,3}|IV|V)"
    r"(\s*\(\s*Tubule formation:\s*)([123])"
    r"(\s*,\s*Nuclear grade:\s*)([123])"
    r"(\s*,\s*Mitoses:\s*)([123])"
    r"(\s*\))",
    re.IGNORECASE,
)


def gleason_to_grade_group_num(gleason: str) -> str:
    g = re.sub(r"\s+", " ", str(gleason or "").strip())
    if g == "6 (3+3)":
        return "1"
    if g == "7 (3+4)":
        return "2"
    if g == "7 (4+3)":
        return "3"
    if g.startswith("8"):
        return "4"
    if g.startswith("9") or g.startswith("10"):
        return "5"
    return ""


def grade_group_num(label: str) -> str:
    m = re.search(r"(\d)", str(label or ""))
    if m:
        return m.group(1)
    return gleason_to_grade_group_num(label)


def extract_prostate_slots(report: str) -> dict[str, str]:
    out = {"gleason_score": "", "grade_group_num": ""}
    m = _GLEASON_RE.search(report or "")
    if m:
        tail = m.group(0)
        sm = re.search(r"(\d+(?:\s*\(\s*\d+\s*\+\s*\d+\s*\)?|\s*\(\s*\d+\s*\+\s*\d+\s*\))|\d+)", tail)
        if sm:
            out["gleason_score"] = sm.group(1).strip()
    gm = _GRADE_GROUP_RE.search(report or "")
    if gm:
        nm = re.search(r"(\d)", gm.group(0))
        if nm:
            out["grade_group_num"] = nm.group(1)
    return out


def extract_breast_nottingham(report: str) -> dict[str, str]:
    out = {"tubule": "", "nuclear": "", "mitoses": ""}
    m = _NOTTINGHAM_RE.search(report or "")
    if not m:
        return out
    nums = re.findall(r":\s*(\d+)", m.group(0))
    if len(nums) >= 3:
        out["tubule"], out["nuclear"], out["mitoses"] = nums[0], nums[1], nums[2]
    return out


def patch_prostate_report(
    report: str,
    *,
    gleason_score: str,
    grade_group: str = "",
) -> str:
    if not report or not gleason_score or gleason_score == "<other>":
        return report
    gg = grade_group_num(grade_group) or gleason_to_grade_group_num(gleason_score)
    text = _GLEASON_RE.sub(f"Gleason's score {gleason_score}", report)
    if gg:
        text = _GRADE_GROUP_RE.sub(f"grade group {gg}", text)
    return text


def patch_breast_nottingham_report(
    report: str,
    *,
    tubule: str | None = None,
    nuclear: str | None = None,
    mitoses: str | None = None,
    baseline: dict[str, str] | None = None,
) -> str:
    if not report:
        return report
    base = baseline or extract_breast_nottingham(report)
    if not base.get("tubule"):
        return report
    t = tubule if tubule is not None else base["tubule"]
    n = nuclear if nuclear is not None else base["nuclear"]
    m = mitoses if mitoses is not None else base["mitoses"]
    for val in (t, n, m):
        if not val or val == "<other>" or val not in {"1", "2", "3"}:
            return report
    if _NOTTINGHAM_RE.search(report):
        return _NOTTINGHAM_RE.sub(
            rf"(Tubule formation: {t}, Nuclear grade: {n}, Mitoses: {m})",
            report,
            count=1,
        )
    return report


def nottingham_grade_from_total(total: int) -> str:
    if total <= 5:
        return "I"
    if total <= 7:
        return "II"
    return "III"


def patch_breast_nst_grade_report(
    report: str,
    *,
    tubule: str,
    nuclear: str,
    mitoses: str,
) -> str:
    if not report:
        return report
    if any(v not in {"1", "2", "3"} for v in (tubule, nuclear, mitoses)):
        return report
    m = _NST_GRADE_RE.search(report)
    if not m:
        return report
    grade = nottingham_grade_from_total(int(tubule) + int(nuclear) + int(mitoses))
    repl = "".join(
        (
            m.group(1),
            grade,
            m.group(3),
            tubule,
            m.group(5),
            nuclear,
            m.group(7),
            mitoses,
            m.group(9),
        )
    )
    return report[: m.start()] + repl + report[m.end() :]


def should_apply_slot(
    pred: str,
    prob: float,
    *,
    min_prob: float,
    current: str = "",
    min_margin: float = 0.0,
    margin: float | None = None,
) -> bool:
    if not pred or pred == "<other>":
        return False
    if prob < float(min_prob):
        return False
    if margin is not None and margin < float(min_margin):
        return False
    if current and pred.strip().lower() == current.strip().lower():
        return False
    return True


def patch_report_for_organ(
    report: str,
    organ: str,
    slots: dict[str, dict[str, Any]],
    *,
    min_prob_gleason: float = 0.45,
    min_prob_breast: float = 0.65,
    min_margin_gleason: float = 0.0,
    organ_filter: str = "all",
    breast_partial: bool = True,
) -> tuple[str, list[str]]:
    """Return patched report and list of applied slot names."""
    applied: list[str] = []
    organ_l = str(organ or "").lower()
    filt = str(organ_filter or "all").lower()
    if filt not in ("all", organ_l):
        return report, applied
    text = report

    if organ_l == "prostate" and filt in ("all", "prostate"):
        cur = extract_prostate_slots(text)
        g = slots.get("gleason_score") or {}
        if should_apply_slot(
            str(g.get("pred", "")),
            float(g.get("prob", 0.0)),
            min_prob=min_prob_gleason,
            min_margin=min_margin_gleason,
            margin=g.get("margin"),
            current=cur.get("gleason_score", ""),
        ):
            gg = slots.get("grade_group") or {}
            gg_pred = str(gg.get("pred", "")) if float(gg.get("prob", 0.0)) >= min_prob_gleason else ""
            text = patch_prostate_report(
                text,
                gleason_score=str(g["pred"]),
                grade_group=gg_pred,
            )
            applied.append("gleason_score")
            if gg_pred:
                applied.append("grade_group")

    elif organ_l == "breast" and filt in ("all", "breast"):
        cur = extract_breast_nottingham(text)
        if not cur.get("tubule"):
            return text, applied
        t = slots.get("breast_tubule_score") or {}
        n = slots.get("breast_nuclear_score") or {}
        m = slots.get("breast_mitotic_score") or {}
        comp = (
            ("breast_tubule_score", t, "tubule"),
            ("breast_nuclear_score", n, "nuclear"),
            ("breast_mitotic_score", m, "mitoses"),
        )
        new = dict(cur)
        for slot_name, slot, key in comp:
            pred = str(slot.get("pred", ""))
            if should_apply_slot(
                pred,
                float(slot.get("prob", 0.0)),
                min_prob=min_prob_breast,
                current=cur.get(key, ""),
            ):
                new[key] = pred
                applied.append(slot_name)
        if breast_partial:
            if new != cur:
                text = patch_breast_nottingham_report(
                    text,
                    tubule=new["tubule"],
                    nuclear=new["nuclear"],
                    mitoses=new["mitoses"],
                    baseline=cur,
                )
        else:
            if len(applied) == 3:
                text = patch_breast_nottingham_report(
                    text,
                    tubule=str(t.get("pred", "")),
                    nuclear=str(n.get("pred", "")),
                    mitoses=str(m.get("pred", "")),
                )
            else:
                applied = []

    return text, applied


_PROSTATE_G6_P4_RE = re.compile(
    r"(Gleason'?s?\s*score\s*6\s*\(\s*3\s*\+\s*3\s*\)\s*,\s*grade group\s*1)\s*"
    r"\(\s*Gleason pattern 4:\s*\d+%\s*\)",
    re.IGNORECASE,
)
_DCIS_NUCLEAR_RE = re.compile(r"(-\s*Nuclear grade:\s*)(Low|Intermediate|High)", re.IGNORECASE)
_DCIS_TYPE_RE = re.compile(
    r"(-\s*Type:\s*)(Solid|Cribriform|Micropapillary|Papillary|Flat|"
    r"Cribriform and solid|Solid and cribriform|Cribriform and papillary|"
    r"Solid and papillary|Solid and micropapillary|Cribriform and micropapillary)",
    re.IGNORECASE,
)
_DCIS_NECROSIS_RE = re.compile(
    r"(-\s*Necrosis:\s*)(Absent|Present(?:\s*\((?:Focal|Comedo-type)\))?)",
    re.IGNORECASE,
)
_COLORECTAL_HEADER_RE = re.compile(
    r"^(?P<organ>Colon|Rectum),\s*colonoscopic\s*"
    r"(?P<proc>biopsy|polypectomy|mucosal resection|submucosal dissection)\s*;",
    re.IGNORECASE,
)
_BREAST_HEADER_RE = re.compile(
    r"^Breast,\s*(?:core needle|mammotome|stereotactic|aspiration)\s*biopsy\s*;",
    re.IGNORECASE,
)
_CERVIX_HEADER_RE = re.compile(
    r"^Uterine cervix,\s*(?:colposcopic|punch|loop electrosurgical excision procedure|polypectomy|frozen|curette)\s*biopsy?\s*;",
    re.IGNORECASE,
)
_COLO_PROC_TEXT = {
    "biopsy": "biopsy",
    "polypectomy": "polypectomy",
    "emr": "mucosal resection",
    "esd": "submucosal dissection",
}
_PROC_HEAD_BY_ORGAN = {
    "Breast": "procedure_breast",
    "Colon": "procedure_colorectal",
    "Rectum": "procedure_colorectal",
    "Uterine cervix": "procedure_cervix",
}
_PROC_TEXT = {
    "Breast": {
        "core_needle": ("Core needle biopsy", "Breast, core needle biopsy"),
        "mammotome": ("Mammotome biopsy", "Breast, mammotome biopsy"),
        "stereotactic": ("Stereotactic biopsy", "Breast, stereotactic biopsy"),
    },
    "Colon": {
        "biopsy": ("Colonoscopic biopsy", "Colon, colonoscopic biopsy"),
        "polypectomy": ("Colonoscopic polypectomy", "Colon, colonoscopic polypectomy"),
        "emr": ("Colonoscopic mucosal resection", "Colon, colonoscopic mucosal resection"),
        "esd": ("Colonoscopic submucosal dissection", "Colon, colonoscopic submucosal dissection"),
    },
    "Rectum": {
        "biopsy": ("Colonoscopic biopsy", "Rectum, colonoscopic biopsy"),
        "polypectomy": ("Colonoscopic polypectomy", "Rectum, colonoscopic polypectomy"),
        "emr": ("Colonoscopic mucosal resection", "Rectum, colonoscopic mucosal resection"),
        "esd": ("Colonoscopic submucosal dissection", "Rectum, colonoscopic submucosal dissection"),
    },
    "Uterine cervix": {
        "colposcopic": ("Colposcopic biopsy", "Uterine cervix, colposcopic biopsy"),
        "punch": ("Punch biopsy", "Uterine cervix, punch biopsy"),
        "leep": (
            "Loop electrosurgical excision procedure",
            "Uterine cervix, loop electrosurgical excision procedure",
        ),
    },
}
_PROC_HEADER_RE = {
    "Breast": _BREAST_HEADER_RE,
    "Colon": _COLORECTAL_HEADER_RE,
    "Rectum": _COLORECTAL_HEADER_RE,
    "Uterine cervix": _CERVIX_HEADER_RE,
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slot_ok(slot: dict[str, Any], *, prob: float, margin: float = 0.0) -> bool:
    pred = _norm(slot.get("pred"))
    if not pred or pred == "<other>":
        return False
    if float(slot.get("prob", 0.0)) < prob:
        return False
    if "margin" in slot and float(slot.get("margin", 0.0)) < margin:
        return False
    return True


def _slots_agree(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    *,
    key: str,
    prob: float,
) -> bool:
    sa = a.get(key) or {}
    sb = b.get(key) or {}
    if not _slot_ok(sa, prob=prob) or not _slot_ok(sb, prob=prob):
        return False
    return _norm(sa.get("pred")) == _norm(sb.get("pred"))


def _dcis_necrosis_text(
    slots: dict[str, dict[str, Any]],
    *,
    present_prob: float,
    type_prob: float,
) -> str:
    present = slots.get("dcis_necrosis_present") or {}
    if not _slot_ok(present, prob=present_prob):
        return ""
    pred = _norm(present.get("pred")).lower()
    if pred.startswith("no"):
        return "Absent"
    if not pred.startswith("yes"):
        return ""
    nec_type = slots.get("dcis_necrosis_type") or {}
    if _slot_ok(nec_type, prob=type_prob):
        tp = _norm(nec_type.get("pred")).lower()
        if "comedo" in tp:
            return "Present (Comedo-type)"
        if "focal" in tp:
            return "Present (Focal)"
    return "Present"


def _current_proc_family(organ: str, report: str, procedure_answer: str = "") -> str:
    header = str(report or "").split(";", 1)[0]
    text = f"{procedure_answer} {header}".lower()
    if organ == "Breast":
        if "stereotactic" in text:
            return "stereotactic"
        if "mammotome" in text:
            return "mammotome"
        if "core needle" in text:
            return "core_needle"
    elif organ in {"Colon", "Rectum"}:
        if "submucosal dissection" in text:
            return "esd"
        if "mucosal resection" in text:
            return "emr"
        if "polypectomy" in text:
            return "polypectomy"
        if "biopsy" in text:
            return "biopsy"
    elif organ == "Uterine cervix":
        if "loop electrosurgical" in text or "leep" in text:
            return "leep"
        if "punch" in text:
            return "punch"
        if "colposcopic" in text:
            return "colposcopic"
    return ""


def _replace_proc_header(report: str, organ: str, target_header: str) -> str:
    pat = _PROC_HEADER_RE.get(organ)
    replacement = f"{target_header};"
    if pat and pat.search(report):
        return pat.sub(replacement, report, count=1)
    if ";" in report:
        return replacement + report.split(";", 1)[1]
    return report


def patch_procedure_header_local(
    report: str,
    organ: str,
    procedure_answer: str,
    hard_heads: dict[str, dict[str, Any]],
    *,
    min_prob: float = 0.95,
    min_margin: float = 0.0,
    cervix_min_prob: float = 0.995,
    cervix_min_margin: float = 0.90,
    colorectal_organ_prob: float = 0.95,
    colorectal_organ_margin: float = 0.80,
    colorectal_organ_proc_prob: float = 0.995,
    colorectal_organ_proc_margin: float = 0.90,
) -> tuple[str, str | None, list[str]]:
    """Patch only the procedure/header field from a high-confidence local head."""
    organ_s = str(organ or "").strip()
    head = _PROC_HEAD_BY_ORGAN.get(organ_s)
    if not report or not head:
        return report, None, []
    rec = hard_heads.get(head) or {}
    pred = _norm(rec.get("pred"))

    proc_prob = max(min_prob, cervix_min_prob) if organ_s == "Uterine cervix" else min_prob
    proc_margin = max(min_margin, cervix_min_margin) if organ_s == "Uterine cervix" else min_margin
    if not _slot_ok(rec, prob=proc_prob, margin=proc_margin):
        return report, None, []

    target_organ = organ_s
    organ_switched = False
    if organ_s in {"Colon", "Rectum"}:
        org_rec = hard_heads.get("organ_colorectal") or {}
        pred_organ = _norm(org_rec.get("pred")).lower()
        can_switch_organ = (
            _slot_ok(org_rec, prob=colorectal_organ_prob, margin=colorectal_organ_margin)
            and _slot_ok(
                rec,
                prob=max(proc_prob, colorectal_organ_proc_prob),
                margin=max(proc_margin, colorectal_organ_proc_margin),
            )
        )
        if can_switch_organ and pred_organ in {"colon", "rectum"}:
            target_organ = "Colon" if pred_organ == "colon" else "Rectum"
            organ_switched = target_organ != organ_s

    if pred not in _PROC_TEXT.get(target_organ, {}):
        return report, None, []
    if _current_proc_family(organ_s, report, procedure_answer) == pred and not organ_switched:
        return report, None, []
    proc_answer, header = _PROC_TEXT[target_organ][pred]
    patched = _replace_proc_header(report, target_organ, header)
    if patched == report:
        return report, None, []
    applied = [f"procedure_{organ_s.lower().replace(' ', '_')}"]
    if organ_switched:
        applied.append("organ_colorectal_header")
    return patched, proc_answer, applied


def patch_report_labelgraph_local(
    report: str,
    organ: str,
    slots: dict[str, dict[str, Any]],
    *,
    aux_slots: dict[str, dict[str, Any]] | None = None,
    hard_heads: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """Apply accepted hidden-safe local report field repairs.

    Keep this deliberately narrow.  The frozen v22 replay accepted final-stage
    DCIS nuclear-grade edits at p>=0.85, while DCIS necrosis edits and broad
    graph-state rewrites were not reliable enough to ship.
    """
    text = report
    applied: list[str] = []
    organ_s = str(organ or "").strip()
    aux = aux_slots or slots
    hard = hard_heads or {}

    if organ_s == "Prostate":
        if "Acinar adenocarcinoma" in text and _PROSTATE_G6_P4_RE.search(text):
            text = _PROSTATE_G6_P4_RE.sub(r"\1", text)
            applied.append("prostate_delete_invalid_pattern4")

    elif organ_s == "Breast":
        cur_nottingham = extract_breast_nottingham(text)
        if cur_nottingham.get("tubule"):
            t = slots.get("breast_tubule_score") or {}
            n = slots.get("breast_nuclear_score") or {}
            m = slots.get("breast_mitotic_score") or {}
            comp = (
                ("breast_tubule_score", t, "tubule"),
                ("breast_nuclear_score", n, "nuclear"),
                ("breast_mitotic_score", m, "mitoses"),
            )
            new_nottingham = dict(cur_nottingham)
            for slot_name, slot, key in comp:
                pred = _norm(slot.get("pred"))
                if (
                    pred in {"1", "2", "3"}
                    and pred != cur_nottingham.get(key)
                    and _slot_ok(slot, prob=0.85)
                ):
                    new_nottingham[key] = pred
                    applied.append(slot_name)
            if new_nottingham != cur_nottingham:
                text = patch_breast_nottingham_report(
                    text,
                    tubule=new_nottingham["tubule"],
                    nuclear=new_nottingham["nuclear"],
                    mitoses=new_nottingham["mitoses"],
                    baseline=cur_nottingham,
                )
            if (
                _slot_ok(t, prob=0.95)
                and _slot_ok(n, prob=0.95)
                and _slot_ok(m, prob=0.95)
                and _norm(t.get("pred")) in {"1", "2", "3"}
                and _norm(n.get("pred")) in {"1", "2", "3"}
                and _norm(m.get("pred")) in {"1", "2", "3"}
            ):
                m_nst = _NST_GRADE_RE.search(text)
                pred_grade = nottingham_grade_from_total(
                    int(_norm(t.get("pred"))) + int(_norm(n.get("pred"))) + int(_norm(m.get("pred")))
                )
                if m_nst and pred_grade != _norm(m_nst.group(2)).upper():
                    before_grade = text
                    text = patch_breast_nst_grade_report(
                        text,
                        tubule=_norm(t.get("pred")),
                        nuclear=_norm(n.get("pred")),
                        mitoses=_norm(m.get("pred")),
                    )
                else:
                    before_grade = text
                if text != before_grade:
                    for name in (
                        "breast_nst_grade",
                        "breast_tubule_score",
                        "breast_nuclear_score",
                        "breast_mitotic_score",
                    ):
                        if name not in applied:
                            applied.append(name)

        if "Ductal carcinoma in situ" not in text:
            return text, applied
        if (
            not _DCIS_TYPE_RE.search(text)
            and not _DCIS_NUCLEAR_RE.search(text)
            and not _DCIS_NECROSIS_RE.search(text)
        ):
            return text, applied

        arch = slots.get("dcis_architectural_pattern") or {}
        m_type = _DCIS_TYPE_RE.search(text)
        if m_type and _slot_ok(arch, prob=0.90):
            pred = _norm(arch.get("pred"))
            current = _norm(m_type.group(2))
            if current == "Solid" and pred == "Cribriform":
                new = _DCIS_TYPE_RE.sub(rf"\1{pred}", text, count=1)
                if new != text:
                    text = new
                    applied.append("dcis_architectural_pattern")

        nuclear = slots.get("dcis_nuclear_grade") or {}
        if _DCIS_NUCLEAR_RE.search(text) and _slot_ok(nuclear, prob=0.85):
            pred = _norm(nuclear.get("pred"))
            new = _DCIS_NUCLEAR_RE.sub(rf"\1{pred}", text, count=1)
            if new != text:
                text = new
                applied.append("dcis_nuclear_grade")

    elif organ_s in {"Colon", "Rectum"}:
        m = _COLORECTAL_HEADER_RE.search(text)
        if not m:
            return text, applied
        slot = hard.get("procedure_colorectal") or {}
        pred = _norm(slot.get("pred"))
        if pred not in _COLO_PROC_TEXT or pred == "biopsy":
            return text, applied
        if float(slot.get("prob", 0.0)) < 0.95 or float(slot.get("margin", 0.0)) < 0.80:
            return text, applied
        target = _COLO_PROC_TEXT[pred]
        if m.group("proc").lower() == target:
            return text, applied
        replacement = f"{m.group('organ')}, colonoscopic {target};"
        text = _COLORECTAL_HEADER_RE.sub(replacement, text, count=1)
        applied.append("procedure_colorectal")

    return text, applied
