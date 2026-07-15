from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission" / "algorithm_submission_template"
if str(SUB) not in sys.path:
    sys.path.insert(0, str(SUB))
try:
    import src  # type: ignore

    sub_src = str(SUB / "src")
    paths = list(src.__path__)  # type: ignore[attr-defined]
    if sub_src not in paths:
        src.__path__ = [sub_src, *paths]  # type: ignore[attr-defined]
except Exception:
    pass

from src.reg2_text_calibrator import LabelContext  # noqa: E402
from src.reg2_v20_pipeline import (  # noqa: E402
    MIL_DX_ORGAN_GUARD_ENV,
    _apply_mil_dx_rescue_to_steps,
    _dx_allowed_for_context_organ,
    _dx_to_organs_from_calibrator,
)


class V20MilDxRescueGuardTest(unittest.TestCase):
    def test_dx_rescue_organ_guard_blocks_cross_organ_but_allows_colorectal_union(self) -> None:
        _dx_to_organs_from_calibrator.cache_clear()
        with mock.patch(
            "src.reg2_v20_pipeline._dx_to_organs_from_calibrator",
            return_value={
                "Non-small cell carcinoma, favor adenocarcinoma": {"Lung"},
                "Adenocarcinoma, moderately differentiated": {"Colon"},
            },
        ):
            self.assertFalse(
                _dx_allowed_for_context_organ(
                    "Breast",
                    "Non-small cell carcinoma, favor adenocarcinoma",
                )
            )
            self.assertTrue(
                _dx_allowed_for_context_organ(
                    "Rectum",
                    "Adenocarcinoma, moderately differentiated",
                )
            )

    def test_dx_rescue_organ_guard_is_env_gated(self) -> None:
        context = LabelContext(
            organ="Breast",
            procedure="Core needle biopsy",
            diagnoses=["Invasive carcinoma of no special type, grade II"],
        )
        steps = [
            {"question": "What is the #1 diagnosis?", "answer": context.primary_dx(), "next_question": ""},
            {"question": "What is the final pathology report?", "answer": "Report", "next_question": ""},
        ]

        class FakeRescue:
            def predict(self, patch_features):
                return "Non-small cell carcinoma, favor adenocarcinoma", 0.99, 0.98

        with mock.patch("src.reg2_v20_pipeline._mil_dx_rescue", return_value=FakeRescue()):
            with mock.patch(
                "src.reg2_v20_pipeline._dx_to_organs_from_calibrator",
                return_value={"Non-small cell carcinoma, favor adenocarcinoma": {"Lung"}},
            ):
                os.environ.pop(MIL_DX_ORGAN_GUARD_ENV, None)
                out, event = _apply_mil_dx_rescue_to_steps(steps, context, patch_features=[])
                self.assertIsNotNone(event)
                self.assertEqual(out[0]["answer"], "Non-small cell carcinoma, favor adenocarcinoma")

                os.environ[MIL_DX_ORGAN_GUARD_ENV] = "1"
                out_guarded, event_guarded = _apply_mil_dx_rescue_to_steps(steps, context, patch_features=[])
                self.assertIsNone(event_guarded)
                self.assertEqual(out_guarded[0]["answer"], "Invasive carcinoma of no special type, grade II")
                os.environ.pop(MIL_DX_ORGAN_GUARD_ENV, None)


if __name__ == "__main__":
    unittest.main()
