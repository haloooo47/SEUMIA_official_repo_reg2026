#!/usr/bin/env python3
"""Validate REG2 ensemble profile asset references without running GPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEF_MODEL_ROOT = Path("models")

RUNTIME_ASSET_REQUIREMENTS = {
    "REG2_SECONDARY_MIL_ASSET": "secondary_mil_heads.pt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--model-root", type=Path, default=DEF_MODEL_ROOT)
    return p.parse_args()


def resolve_asset(value: str, model_root: Path, required: str) -> tuple[bool, str]:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [model_root / value, path]
    for cand in candidates:
        if (cand / required).is_file():
            return True, str(cand)
    return False, " | ".join(str(c / required) for c in candidates)


def check_specs(
    specs: list[Any],
    *,
    model_root: Path,
    required: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            rows.append({"ok": False, "error": "spec is not an object", "spec": spec})
            continue
        asset = str(spec.get("asset", "") or "").strip()
        ok, resolved = resolve_asset(asset, model_root, required) if asset else (False, "blank asset")
        rows.append({"ok": ok, "asset": asset, "required": required, "resolved": resolved})
    return rows


def check_runtime_env_assets(profile: dict[str, Any], *, model_root: Path) -> list[dict[str, Any]]:
    runtime_env = profile.get("runtime_env") or {}
    if not isinstance(runtime_env, dict):
        return [{"ok": False, "error": "runtime_env is not an object"}]
    rows: list[dict[str, Any]] = []
    for env_name, required in RUNTIME_ASSET_REQUIREMENTS.items():
        if env_name not in runtime_env:
            continue
        value = str(runtime_env.get(env_name, "") or "").strip()
        ok, resolved = resolve_asset(value, model_root, required) if value else (False, "blank asset")
        rows.append(
            {
                "ok": ok,
                "env": env_name,
                "asset": value,
                "required": required,
                "resolved": resolved,
            }
        )
    return rows


def source_name(kind: str, asset: str) -> str:
    return f"{kind}:{Path(asset).name if asset.startswith('/') else asset}"


def unknown_weight_sources(profile: dict[str, Any], weights: Any) -> list[str]:
    if not isinstance(weights, dict):
        return ["<source_weights_by_organ is not an object>"]
    known = {"mlp", "abmil", "knn"}
    for spec in profile.get("extra_mlp_specs") or []:
        if isinstance(spec, dict) and spec.get("asset"):
            known.add(source_name("mlp", str(spec["asset"])))
    for spec in profile.get("extra_abmil_specs") or []:
        if isinstance(spec, dict) and spec.get("asset"):
            known.add(source_name("abmil", str(spec["asset"])))
    for spec in profile.get("candidate_mlp_specs") or []:
        if isinstance(spec, dict) and spec.get("asset"):
            known.add(source_name("mlp", str(spec["asset"])))
        elif isinstance(spec, str):
            known.add(source_name("mlp", spec))
    for spec in profile.get("candidate_abmil_specs") or []:
        if isinstance(spec, dict) and spec.get("asset"):
            known.add(source_name("abmil", str(spec["asset"])))
    unknown: set[str] = set()
    for organ, organ_weights in weights.items():
        if not isinstance(organ_weights, dict):
            unknown.add(f"{organ}:<weights not object>")
            continue
        for name in organ_weights:
            if str(name) not in known:
                unknown.add(str(name))
    return sorted(unknown)


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    mlp = check_specs(
        profile.get("extra_mlp_specs") or [],
        model_root=args.model_root,
        required="multitask_heads.pt",
    )
    abmil = check_specs(
        profile.get("extra_abmil_specs") or [],
        model_root=args.model_root,
        required="abmil_heads.pt",
    )
    candidate_mlp = check_specs(
        [
            spec if isinstance(spec, dict) else {"asset": spec}
            for spec in (profile.get("candidate_mlp_specs") or [])
        ],
        model_root=args.model_root,
        required="multitask_heads.pt",
    )
    candidate_abmil = check_specs(
        profile.get("candidate_abmil_specs") or [],
        model_root=args.model_root,
        required="abmil_heads.pt",
    )
    weights = profile.get("source_weights_by_organ") or {}
    runtime_assets = check_runtime_env_assets(profile, model_root=args.model_root)
    unknown_sources = unknown_weight_sources(profile, weights)
    summary = {
        "profile": str(args.profile),
        "model_root": str(args.model_root),
        "name": profile.get("name"),
        "runtime_env": profile.get("runtime_env") or {},
        "runtime_assets": runtime_assets,
        "organ_mask_groups": profile.get("organ_mask_groups") or {},
        "mlp": mlp,
        "abmil": abmil,
        "candidate_mlp": candidate_mlp,
        "candidate_abmil": candidate_abmil,
        "source_weight_organs": sorted(weights) if isinstance(weights, dict) else [],
        "unknown_weight_sources": unknown_sources,
        "ok": (
            all(r["ok"] for r in mlp + abmil + candidate_mlp + candidate_abmil + runtime_assets)
            and isinstance(weights, dict)
            and not unknown_sources
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
