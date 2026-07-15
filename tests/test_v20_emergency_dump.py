from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

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

from src.reg2_v20_pipeline import ONLINE_DUMP_DIR_ENV, _dump_emergency_fallback


class V20EmergencyDumpTest(unittest.TestCase):
    def test_emergency_fallback_writes_audit_record(self) -> None:
        old = os.environ.get(ONLINE_DUMP_DIR_ENV)
        with tempfile.TemporaryDirectory() as td:
            os.environ[ONLINE_DUMP_DIR_ENV] = td
            try:
                _dump_emergency_fallback(
                    "case/1",
                    RuntimeError("cuda out of memory"),
                    fallback="v02",
                    cascade_allowed=False,
                )
            finally:
                if old is None:
                    os.environ.pop(ONLINE_DUMP_DIR_ENV, None)
                else:
                    os.environ[ONLINE_DUMP_DIR_ENV] = old

            files = list(Path(td).glob("*.emergency_fallback.json"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["stage"], "emergency_fallback")
            self.assertEqual(record["exception_type"], "RuntimeError")
            self.assertEqual(record["fallback"], "v02")
            self.assertFalse(record["cascade_allowed"])


if __name__ == "__main__":
    unittest.main()
