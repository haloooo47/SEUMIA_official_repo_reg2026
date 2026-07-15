#!/usr/bin/env python3
"""Check that the submission model root contains assets needed by package replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEF_MODEL = REPO / "models"

REQUIRED = {
    "TITAN": ["model.safetensors", "conch_v1_5_pytorch_model.bin"],
    "Virchow2": ["model.safetensors"],
    "reg2_titan_head": ["multitask_heads.pt", "label_vocab.json", "dx_knn_bank.npz"],
    "reg2_abmil_h1": ["abmil_heads.pt", "label_vocab.json"],
    "abmil_hopt1_1024": ["abmil_heads.pt", "label_vocab.json"],
    "reg2_calibrator": ["answer_calibrator.json", "report_calibrator.json", "path_calibrator.json"],
    "reg2_calibrator_v2_labelgraph": [
        "answer_calibrator.json",
        "report_calibrator.json",
        "path_calibrator.json",
        "structured_report_assets/labelgraph_taxonomy_v1.json",
    ],
    "reg2_report_clf_fused": ["report_clf.pt"],
    "reg2_report_clf_fused_s101": ["report_clf.pt"],
    "reg2_report_slot_heads": ["slot_heads.pt", "slot_vocab.json"],
    "reg2_report_slot_heads_breast": ["slot_heads.pt", "slot_vocab.json"],
    "reg2_report_slot_heads_dcis_arch": ["slot_heads.pt", "slot_vocab.json"],
    "reg2_report_hard_heads": ["hard_confusion_heads.pt", "head_vocab.json"],
    "reg2_h1_mil_rescue": ["abmil_heads.pt", "label_vocab.json"],
    "reg2_secondary_mil_heads_virchow2_pos20_s2049": [
        "secondary_mil_heads.pt",
        "secondary_mil_vocab.json",
    ],
    "reg2_roi_gate": ["cnn.pt", "config.json"],
    "reg2_v02": [
        "answer_priors.json",
        "organ_priors.json",
        "path_priors.json",
        "report_priors.json",
        "workflow_graph.json",
    ],
}
REQUIRED_ANY = {
    "H-optimus": ["pytorch_model_fp16.bin", "pytorch_model.bin"],
}
DEFAULT_PROFILE_NAMES = ("package_default.json", "default.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, default=DEF_MODEL)
    p.add_argument("--profile", default="", help="Optional REG2_ENSEMBLE_PROFILE name/path")
    p.add_argument(
        "--require-package-default",
        action="store_true",
        help="Fail when reg2_ensemble_profile/package_default.json or default.json is absent",
    )
    return p.parse_args()


def check_required(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset, files in REQUIRED.items():
        for filename in files:
            path = root / asset / filename
            rows.append({"ok": path.is_file(), "path": str(path)})
    for asset, filenames in REQUIRED_ANY.items():
        paths = [root / asset / filename for filename in filenames]
        rows.append(
            {
                "ok": any(path.is_file() for path in paths),
                "path": " | ".join(str(path) for path in paths),
            }
        )
    return rows


def resolve_profile_path(root: Path, profile: str) -> Path:
    prof_path = Path(profile).expanduser()
    if not prof_path.is_file():
        name = profile if profile.endswith(".json") else f"{profile}.json"
        prof_path = root / "reg2_ensemble_profile" / name
    return prof_path


def check_profile(root: Path, profile: str) -> dict[str, Any] | None:
    if not profile:
        return None
    prof_path = resolve_profile_path(root, profile)
    if not prof_path.is_file():
        return {"ok": False, "error": f"profile not found: {profile}", "resolved": str(prof_path)}
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "check_reg2_ensemble_profile.py"),
        "--profile",
        str(prof_path),
        "--model-root",
        str(root),
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    return {"ok": proc.returncode == 0, "resolved": str(prof_path), "detail": payload}


def profiles_match(a: Path, b: Path) -> bool:
    try:
        return json.loads(a.read_text(encoding="utf-8")) == json.loads(b.read_text(encoding="utf-8"))
    except Exception:
        return False


def find_package_default_profile(root: Path) -> Path | None:
    for name in DEFAULT_PROFILE_NAMES:
        path = root / "reg2_ensemble_profile" / name
        if path.is_file():
            return path
    return None


def main() -> int:
    args = parse_args()
    root = args.model_root.resolve()
    rows = check_required(root)
    profile = check_profile(root, args.profile)
    package_default_path = find_package_default_profile(root)
    package_default = (
        check_profile(root, str(package_default_path))
        if package_default_path is not None and (not args.profile or args.require_package_default)
        else None
    )
    package_default_missing = bool(args.require_package_default and package_default_path is None)
    package_default_matches_profile = None
    if args.profile and args.require_package_default and package_default_path is not None:
        profile_path = resolve_profile_path(root, args.profile)
        package_default_matches_profile = profiles_match(profile_path, package_default_path)
    ok = (
        all(r["ok"] for r in rows)
        and (profile is None or bool(profile.get("ok")))
        and (package_default is None or bool(package_default.get("ok")))
        and not package_default_missing
        and package_default_matches_profile is not False
    )
    summary = {
        "model_root": str(root),
        "required_missing": [r["path"] for r in rows if not r["ok"]],
        "profile": profile,
        "package_default_profile": package_default,
        "package_default_missing": package_default_missing,
        "package_default_matches_profile": package_default_matches_profile,
        "ok": ok,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
