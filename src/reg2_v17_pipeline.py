"""Dual-encoder inference and structured report selection pipeline.

The pipeline exposes the top 15 diagnosis scores to the report candidate
lattice, applies the slot-aware selector, and patches structured report slots.
It falls back through the preceding pipeline stages on runtime failure.
"""

from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.reg2_ensemble_predictor import EnsemblePredictor
from src.reg2_hoptimus_online import extract_from_tiles
from src.reg2_pipeline import ChainOfThoughtStep
from src.reg2_report_hard_heads import predict_hard_heads
from src.reg2_report_selector_fused import select_final_report_fused
from src.reg2_report_slot_candidate_selector import select_final_report_slot_candidate
from src.reg2_report_slot_heads import apply_slot_heads_to_steps, predict_slots
from src.reg2_report_slot_patch import patch_procedure_header_local, patch_report_labelgraph_local
from src.reg2_text_calibrator import FINAL_REPORT_QUESTION_CANONICAL, canonicalize_question
from src.reg2_text_calibrator import TextCalibrator
from src.reg2_titan_online import extract_titan_with_tiles
from src.reg2_v11_pipeline import (
    _candidate_model_roots,
    _find_asset_dir,
    _sanitize_steps,
    _text_calibrator_asset_dir,
    _truthy,
)
from src.reg2_v15_pipeline import predict_v15_chain_of_thought

DISABLE_ENV = "REG2_DISABLE_V17"
ALLOW_CASCADE_FALLBACK_ENV = "REG2_ALLOW_CASCADE_FALLBACK"
DX_TOPK_ENV = "REG2_V17_DX_TOPK"
DISABLE_LABELGRAPH_PATCH_ENV = "REG2_DISABLE_LABELGRAPH_PATCH"
DISABLE_PROCEDURE_TOOL_ENV = "REG2_DISABLE_PROCEDURE_TOOL"
ENABLE_V2_DX_VOTER_ENV = "REG2_ENABLE_V2_DX_VOTER"
V2_DX_VOTER_WEIGHT_ENV = "REG2_V2_DX_VOTER_WEIGHT"
EXTRA_ABMIL_SPECS_ENV = "REG2_EXTRA_ABMIL_SPECS"
EXTRA_MLP_SPECS_ENV = "REG2_EXTRA_MLP_SPECS"
CANDIDATE_ABMIL_SPECS_ENV = "REG2_CANDIDATE_ABMIL_SPECS"
CANDIDATE_MLP_SPECS_ENV = "REG2_CANDIDATE_MLP_SPECS"
ENSEMBLE_W_MLP_ENV = "REG2_ENSEMBLE_W_MLP"
ENSEMBLE_W_ABMIL_ENV = "REG2_ENSEMBLE_W_ABMIL"
ENSEMBLE_W_KNN_ENV = "REG2_ENSEMBLE_W_KNN"
ENSEMBLE_USE_ORGAN_MASK_ENV = "REG2_ENSEMBLE_USE_ORGAN_MASK"
SOURCE_WEIGHTS_JSON_ENV = "REG2_SOURCE_WEIGHTS_JSON"
ENSEMBLE_PROFILE_ENV = "REG2_ENSEMBLE_PROFILE"
DISABLE_PACKAGE_PROFILE_ENV = "REG2_DISABLE_PACKAGE_PROFILE"
DEFAULT_PROFILE_NAMES = ("package_default.json", "default.json")


def _profile_path_from_name(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    name = raw if raw.endswith(".json") else f"{raw}.json"
    return _find_asset_dir("reg2_ensemble_profile", name) / name


def _package_default_profile_path() -> Path | None:
    if _truthy(os.environ.get(DISABLE_PACKAGE_PROFILE_ENV)):
        return None
    for root in _candidate_model_roots():
        profile_dir = root / "reg2_ensemble_profile"
        for name in DEFAULT_PROFILE_NAMES:
            path = profile_dir / name
            if path.is_file():
                return path
    return None


@lru_cache(maxsize=1)
def _ensemble_profile() -> dict[str, object]:
    raw = str(os.environ.get(ENSEMBLE_PROFILE_ENV, "") or "").strip()
    try:
        if raw.startswith("{"):
            obj = json.loads(raw)
        else:
            path: Path | None
            if raw:
                path = _profile_path_from_name(raw)
            else:
                path = _package_default_profile_path()
                if path is None:
                    return {}
            obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise TypeError(f"profile root must be a JSON object, got {type(obj).__name__}")
        print(f"[v17-profile] loaded {obj.get('name', raw)!r}")
        return obj
    except Exception as exc:
        print(f"[v17-profile] ignored {ENSEMBLE_PROFILE_ENV}: {type(exc).__name__}: {exc}")
        return {}


def _profile_dict(key: str) -> dict[str, object]:
    val = _ensemble_profile().get(key)
    return val if isinstance(val, dict) else {}


def _profile_list(key: str) -> list[object]:
    val = _ensemble_profile().get(key)
    return val if isinstance(val, list) else []


def _profile_float(key: str, default: float) -> float:
    profile = _profile_dict("ensemble")
    try:
        return float(profile.get(key, default))
    except (TypeError, ValueError):
        return default


def _profile_bool(key: str, default: bool) -> bool:
    profile = _profile_dict("ensemble")
    if key not in profile:
        return default
    val = profile.get(key)
    if isinstance(val, bool):
        return val
    return _truthy(str(val))


def _apply_profile_runtime_env() -> None:
    runtime_env = _profile_dict("runtime_env")
    for key, val in runtime_env.items():
        key_s = str(key).strip()
        if not key_s or key_s in os.environ or val is None:
            continue
        if isinstance(val, bool):
            os.environ[key_s] = "1" if val else "0"
        else:
            os.environ[key_s] = str(val)


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _asset_dir_or_path(value: str, required_file: str) -> Path:
    path = Path(value).expanduser()
    if (path / required_file).is_file():
        return path
    return _find_asset_dir(value, required_file)


@lru_cache(maxsize=1)
def _source_weights_by_organ() -> dict[str, dict[str, float]]:
    raw = str(os.environ.get(SOURCE_WEIGHTS_JSON_ENV, "") or "").strip()
    profile_weights = _profile_dict("source_weights_by_organ")
    if not raw and profile_weights:
        raw_obj: object = profile_weights
    elif not raw:
        return {}
    else:
        raw_obj = raw
    try:
        if isinstance(raw_obj, dict):
            obj = raw_obj
        else:
            path = Path(str(raw_obj)).expanduser()
            if path.is_file():
                obj = json.loads(path.read_text(encoding="utf-8"))
            else:
                obj = json.loads(str(raw_obj))
        out: dict[str, dict[str, float]] = {}
        for organ, weights in dict(obj).items():
            if not isinstance(weights, dict):
                continue
            out[str(organ)] = {str(k): float(v) for k, v in weights.items()}
        return out
    except Exception as exc:
        print(f"[v17-source-weights] ignored {SOURCE_WEIGHTS_JSON_ENV}: {type(exc).__name__}: {exc}")
        return {}


@lru_cache(maxsize=1)
def _extra_abmil_specs() -> tuple[tuple[Path, str, float], ...]:
    specs: list[tuple[Path, str, float]] = []
    for obj in _profile_list("extra_abmil_specs"):
        if not isinstance(obj, dict):
            continue
        asset_name = str(obj.get("asset", "") or "").strip()
        patch_key = str(obj.get("patch_key", "patch_features") or "patch_features").strip()
        try:
            weight = float(obj.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if asset_name and patch_key and weight > 0:
            specs.append((_asset_dir_or_path(asset_name, "abmil_heads.pt"), patch_key, weight))
    raw = str(os.environ.get(EXTRA_ABMIL_SPECS_ENV, "") or "").strip()
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(":")]
            if len(parts) != 3:
                print(
                    f"[v17-extra-voter] ignored malformed spec {item!r}; "
                    "expected asset_name:patch_key:weight"
                )
                continue
            asset_name, patch_key, weight_raw = parts
            try:
                weight = float(weight_raw)
            except ValueError:
                print(f"[v17-extra-voter] ignored spec with invalid weight {item!r}")
                continue
            if weight <= 0:
                continue
            specs.append((_asset_dir_or_path(asset_name, "abmil_heads.pt"), patch_key, weight))
    if _truthy(os.environ.get(ENABLE_V2_DX_VOTER_ENV)):
        specs.append(
            (
                _asset_dir_or_path("reg2_abmil_v2_nnmil4", "abmil_heads.pt"),
                "patch_features_v2",
                _f(V2_DX_VOTER_WEIGHT_ENV, 0.15),
            )
        )
    return tuple(specs)


@lru_cache(maxsize=1)
def _candidate_abmil_specs() -> tuple[tuple[Path, str], ...]:
    specs: list[tuple[Path, str]] = []
    for obj in _profile_list("candidate_abmil_specs"):
        if not isinstance(obj, dict):
            continue
        asset_name = str(obj.get("asset", "") or "").strip()
        patch_key = str(obj.get("patch_key", "patch_features") or "patch_features").strip()
        if asset_name and patch_key:
            specs.append((_asset_dir_or_path(asset_name, "abmil_heads.pt"), patch_key))
    raw = str(os.environ.get(CANDIDATE_ABMIL_SPECS_ENV, "") or "").strip()
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(":")]
            if len(parts) != 2:
                print(
                    f"[v17-candidate-voter] ignored malformed spec {item!r}; "
                    "expected asset_name:patch_key"
                )
                continue
            asset_name, patch_key = parts
            if asset_name and patch_key:
                specs.append((_asset_dir_or_path(asset_name, "abmil_heads.pt"), patch_key))
    return tuple(specs)


@lru_cache(maxsize=1)
def _extra_mlp_specs() -> tuple[tuple[Path, float], ...]:
    specs: list[tuple[Path, float]] = []
    for obj in _profile_list("extra_mlp_specs"):
        if not isinstance(obj, dict):
            continue
        asset_name = str(obj.get("asset", "") or "").strip()
        try:
            weight = float(obj.get("weight", 0.0))
        except (TypeError, ValueError):
            continue
        if asset_name and weight > 0:
            specs.append((_asset_dir_or_path(asset_name, "multitask_heads.pt"), weight))
    raw = str(os.environ.get(EXTRA_MLP_SPECS_ENV, "") or "").strip()
    if not raw:
        return tuple(specs)
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 2:
            print(
                f"[v17-extra-mlp] ignored malformed spec {item!r}; "
                "expected asset_name:weight"
            )
            continue
        asset_name, weight_raw = parts
        try:
            weight = float(weight_raw)
        except ValueError:
            print(f"[v17-extra-mlp] ignored spec with invalid weight {item!r}")
            continue
        if weight <= 0:
            continue
        specs.append((_asset_dir_or_path(asset_name, "multitask_heads.pt"), weight))
    return tuple(specs)


@lru_cache(maxsize=1)
def _candidate_mlp_specs() -> tuple[Path, ...]:
    specs: list[Path] = []
    for obj in _profile_list("candidate_mlp_specs"):
        asset_name = ""
        if isinstance(obj, dict):
            asset_name = str(obj.get("asset", "") or "").strip()
        elif isinstance(obj, str):
            asset_name = obj.strip()
        if asset_name:
            specs.append(_asset_dir_or_path(asset_name, "multitask_heads.pt"))
    raw = str(os.environ.get(CANDIDATE_MLP_SPECS_ENV, "") or "").strip()
    if raw:
        for item in raw.split(","):
            asset_name = item.strip()
            if asset_name:
                specs.append(_asset_dir_or_path(asset_name, "multitask_heads.pt"))
    return tuple(specs)


@lru_cache(maxsize=1)
def _extra_mlp_input_keys() -> tuple[str, ...]:
    keys: list[str] = []
    if not _extra_mlp_specs() and not _candidate_mlp_specs():
        return tuple()
    import torch

    for asset_dir, _weight in _extra_mlp_specs():
        ck = torch.load(asset_dir / "multitask_heads.pt", map_location="cpu", weights_only=False)
        keys.extend(str(k) for k in ck.get("input_keys", ["slide_embedding"]))
    for asset_dir in _candidate_mlp_specs():
        ck = torch.load(asset_dir / "multitask_heads.pt", map_location="cpu", weights_only=False)
        keys.extend(str(k) for k in ck.get("input_keys", ["slide_embedding"]))
    return tuple(dict.fromkeys(keys))


def _dx_voter_needs_v2() -> bool:
    return (
        any(patch_key == "patch_features_v2" for _asset_dir, patch_key, _weight in _extra_abmil_specs())
        or any(patch_key == "patch_features_v2" for _asset_dir, patch_key in _candidate_abmil_specs())
        or "fused" in _extra_mlp_input_keys()
    )


def _dx_voter_needs_fused() -> bool:
    return "fused" in _extra_mlp_input_keys()


@lru_cache(maxsize=1)
def _dual_predictor() -> EnsemblePredictor:
    mlp_dir = _find_asset_dir("reg2_titan_head", "multitask_heads.pt")
    abmil_dir = _find_asset_dir("reg2_abmil_h1", "abmil_heads.pt")
    predictor = EnsemblePredictor(
        mlp_dir,
        abmil_dir,
        extra_abmil_specs=_extra_abmil_specs(),
        extra_mlp_specs=_extra_mlp_specs(),
        candidate_abmil_specs=_candidate_abmil_specs(),
        candidate_mlp_specs=_candidate_mlp_specs(),
        source_weights_by_organ=_source_weights_by_organ(),
        organ_mask_groups=_profile_dict("organ_mask_groups"),
        w_mlp=_f(ENSEMBLE_W_MLP_ENV, _profile_float("w_mlp", 0.35)),
        w_abmil=_f(ENSEMBLE_W_ABMIL_ENV, _profile_float("w_abmil", 0.55)),
        w_knn=_f(ENSEMBLE_W_KNN_ENV, _profile_float("w_knn", 0.10)),
        use_organ_mask=_truthy(
            os.environ.get(
                ENSEMBLE_USE_ORGAN_MASK_ENV,
                "1" if _profile_bool("use_organ_mask", True) else "0",
            )
        ),
    )
    if not predictor.available:
        raise FileNotFoundError(
            f"dual-encoder predictor unavailable (mlp={mlp_dir}, abmil={abmil_dir})"
        )
    return predictor


@lru_cache(maxsize=1)
def _text_calibrator() -> TextCalibrator:
    assets_dir = _text_calibrator_asset_dir()
    return TextCalibrator.from_assets(assets_dir)


def _final_report_index(steps: list[dict[str, object]]) -> int | None:
    for i, step in enumerate(steps):
        if canonicalize_question(str(step.get("question", ""))) == FINAL_REPORT_QUESTION_CANONICAL:
            return i
    return None


def _apply_labelgraph_patch(
    steps: list[ChainOfThoughtStep],
    *,
    organ: str,
    fused_feature: np.ndarray,
    hard_heads: dict[str, dict[str, object]] | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_LABELGRAPH_PATCH_ENV)):
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline = str(steps[idx].get("answer", "") or "").strip()
    if not baseline:
        return steps
    try:
        slots = predict_slots(fused_feature, organ=organ) or {}
        arch_slots = predict_slots(fused_feature, asset_name="reg2_report_slot_heads_dcis_arch") or {}
        slots.update(arch_slots)
        aux_slots = predict_slots(fused_feature, asset_name="reg2_report_slot_heads") or slots
        hard = hard_heads or predict_hard_heads(fused_feature) or {}
        patched, applied = patch_report_labelgraph_local(
            baseline,
            organ,
            slots,
            aux_slots=aux_slots,
            hard_heads=hard,
        )
    except Exception as exc:
        print(f"[labelgraph-patch] failed: {type(exc).__name__}: {exc}")
        return steps
    if patched == baseline or not applied:
        return steps
    out = [dict(s) for s in steps]
    out[idx] = dict(out[idx])
    out[idx]["answer"] = patched
    print(f"[labelgraph-patch] patched organ={organ} slots={applied}")
    return _sanitize_steps(out)


def _procedure_question_index(steps: list[dict[str, object]]) -> int | None:
    for i, step in enumerate(steps):
        if str(step.get("question", "")).strip().lower() == "what is the procedure?":
            return i
    return None


def _apply_procedure_tool_patch(
    steps: list[ChainOfThoughtStep],
    *,
    organ: str,
    fused_feature: np.ndarray,
    hard_heads: dict[str, dict[str, object]] | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_PROCEDURE_TOOL_ENV)):
        return steps
    idx = _final_report_index(steps)
    if idx is None:
        return steps
    baseline = str(steps[idx].get("answer", "") or "").strip()
    if not baseline:
        return steps
    try:
        hard = hard_heads or predict_hard_heads(fused_feature) or {}
        proc_idx = _procedure_question_index(steps)
        proc_answer = ""
        if proc_idx is not None:
            proc_answer = str(steps[proc_idx].get("answer", "") or "").strip()
        patched, new_proc_answer, applied = patch_procedure_header_local(
            baseline,
            organ,
            proc_answer,
            hard,
        )
    except Exception as exc:
        print(f"[procedure-tool] failed: {type(exc).__name__}: {exc}")
        return steps
    if patched == baseline or not applied:
        return steps
    out = [dict(s) for s in steps]
    out[idx] = dict(out[idx])
    out[idx]["answer"] = patched
    if proc_idx is not None and new_proc_answer:
        out[proc_idx] = dict(out[proc_idx])
        out[proc_idx]["answer"] = new_proc_answer
    print(f"[procedure-tool] patched organ={organ} applied={applied}")
    return _sanitize_steps(out)


def predict_v17_chain_of_thought(
    *,
    wsi_path: Path,
    case_id: str | None = None,
) -> list[ChainOfThoughtStep]:
    if _truthy(os.environ.get(DISABLE_ENV)):
        return predict_v15_chain_of_thought(wsi_path=wsi_path, case_id=case_id)

    try:
        _apply_profile_runtime_env()
        titan_feats, tiles, coords, patch_px = extract_titan_with_tiles(wsi_path)
        hopt = extract_from_tiles(tiles, coords, patch_px)
        features = {
            "slide_embedding": titan_feats["slide_embedding"],
            "patch_features": hopt["patch_features"],
            "pooled_mean": hopt["pooled_mean"],
            "coords": hopt["coords"],
            "n_patches": hopt["n_patches"],
            "patch_px": hopt["patch_px"],
        }
        v2_feats = None
        if _dx_voter_needs_v2():
            from src.reg2_virchow2_online import extract_v2_features_from_tiles

            v2_feats = extract_v2_features_from_tiles(tiles)
            features["patch_features_v2"] = v2_feats["patch_features_v2"]
            if _dx_voter_needs_fused():
                features["fused"] = np.concatenate(
                    [
                        np.asarray(titan_feats["slide_embedding"], np.float32).ravel(),
                        np.asarray(hopt["pooled_mean"], np.float32).ravel(),
                        np.asarray(v2_feats["pooled_mean"], np.float32).ravel(),
                    ]
                ).astype(np.float32)

        dx_topk = _i(DX_TOPK_ENV, 15)
        context, scores = _dual_predictor().predict_with_scores(features, dx_topk=dx_topk)
        steps = _sanitize_steps(_text_calibrator().render_from_labels(context, True))
        if not steps:
            raise RuntimeError("calibrator produced an empty chain")

        if v2_feats is None:
            from src.reg2_virchow2_online import extract_v2_pooled_from_tiles

            v2_pooled = extract_v2_pooled_from_tiles(tiles)
        else:
            v2_pooled = v2_feats["pooled_mean"]
        fused = np.concatenate(
            [
                np.asarray(titan_feats["slide_embedding"], np.float32).ravel(),
                np.asarray(hopt["pooled_mean"], np.float32).ravel(),
                np.asarray(v2_pooled, np.float32).ravel(),
            ]
        ).astype(np.float32)
        hard = predict_hard_heads(fused) or {}

        steps = _sanitize_steps(
            select_final_report_slot_candidate(
                steps=steps,
                context=context,
                dx_scores=scores,
                fused_feature=fused,
                calib=_text_calibrator(),
            )
        )
        steps = _sanitize_steps(
            apply_slot_heads_to_steps(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
            )
        )
        steps = _sanitize_steps(
            _apply_labelgraph_patch(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
                hard_heads=hard,
            )
        )
        steps = _sanitize_steps(
            select_final_report_fused(
                steps=steps,
                fused_feature=fused,
            )
        )
        steps = _sanitize_steps(
            _apply_procedure_tool_patch(
                steps,
                organ=str(context.organ or ""),
                fused_feature=fused,
                hard_heads=hard,
            )
        )
        if not steps:
            raise RuntimeError("v17 selector produced an empty chain")

        top_dx = scores.get("primary_dx", [("", 0.0)])[0]
        print(
            "[v17] verified-stack "
            f"organ={context.organ or '?'} dx={context.primary_dx() or '?'} "
            f"dx_topk={len(scores.get('primary_dx') or [])} "
            f"dx_score={top_dx[1]:.3f} fused_dim={fused.shape[0]} steps={len(steps)}"
        )
        return steps
    except Exception as exc:
        if _truthy(os.environ.get(ALLOW_CASCADE_FALLBACK_ENV)):
            print(f"[v17] failed; falling back to v15: {type(exc).__name__}: {exc}")
            return predict_v15_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
        print(f"[v17] failed; using single emergency fallback: {type(exc).__name__}: {exc}")
        from src.reg2_v02_pipeline import predict_v02_chain_of_thought

        return predict_v02_chain_of_thought(wsi_path=wsi_path, case_id=case_id)
