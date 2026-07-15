from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "submission" / "algorithm_submission_template"
if str(TEMPLATE) in sys.path:
    sys.path.remove(str(TEMPLATE))
sys.path.insert(0, str(TEMPLATE))
import src  # noqa: E402

template_src = str(TEMPLATE / "src")
if template_src not in src.__path__:
    src.__path__ = [template_src, *list(src.__path__)]

from src.reg2_v20_pipeline import (  # noqa: E402
    REPORT_SNAPSHOT_ROLLBACK_ENV,
    _remember_consistent_report_snapshot,
    _restore_consistent_report_snapshot,
)


def steps(organ: str, procedure: str, dx: str, report: str) -> list[dict[str, str]]:
    return [
        {"question": "What is the organ?", "answer": organ, "next_question": ""},
        {"question": "What is the procedure?", "answer": procedure, "next_question": ""},
        {"question": "What is the #1 diagnosis?", "answer": dx, "next_question": ""},
        {"question": "What is the final pathology report?", "answer": report, "next_question": ""},
    ]


class ReportSnapshotRollbackTest(unittest.TestCase):
    def test_restores_only_matching_state_snapshot(self) -> None:
        good = steps(
            "Prostate",
            "Needle biopsy",
            "Acinar adenocarcinoma",
            "Prostate, Needle biopsy;\\n  Acinar adenocarcinoma",
        )
        broken = steps(
            "Prostate",
            "Needle biopsy",
            "Acinar adenocarcinoma",
            "Prostate, Needle biopsy;\\n  No tumor present",
        )
        snapshots: dict[tuple[str, str, str], str] = {}
        with patch.dict(os.environ, {REPORT_SNAPSHOT_ROLLBACK_ENV: "1"}):
            _remember_consistent_report_snapshot(snapshots, good)
            restored, event = _restore_consistent_report_snapshot(
                snapshots=snapshots,
                steps=broken,
                case_id="case",
            )
        self.assertIsNotNone(event)
        self.assertEqual(restored[-1]["answer"], good[-1]["answer"])

    def test_does_not_reuse_report_after_state_change(self) -> None:
        good = steps(
            "Prostate",
            "Needle biopsy",
            "Acinar adenocarcinoma",
            "Prostate, Needle biopsy;\\n  Acinar adenocarcinoma",
        )
        changed = steps(
            "Prostate",
            "Needle biopsy",
            "No tumor present",
            "Prostate, Needle biopsy;\\n  Acinar adenocarcinoma",
        )
        snapshots: dict[tuple[str, str, str], str] = {}
        with patch.dict(os.environ, {REPORT_SNAPSHOT_ROLLBACK_ENV: "1"}):
            _remember_consistent_report_snapshot(snapshots, good)
            restored, event = _restore_consistent_report_snapshot(
                snapshots=snapshots,
                steps=changed,
                case_id="case",
            )
        self.assertIsNone(event)
        self.assertEqual(restored, changed)


if __name__ == "__main__":
    unittest.main()
