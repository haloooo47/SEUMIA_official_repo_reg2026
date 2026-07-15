"""Online MLP + ABMIL (+kNN +organ-mask) ensemble -> LabelContext.

Uses w_mlp=0.35, w_abmil=0.55, w_knn=0.10, knn_k=20, and organ_mask=on.

* organ / procedure / grade / behavior / histologic_type come from the MLP head
  (best there).
* primary_dx comes from the MLP + ABMIL probability ensemble, optionally fused
  with a cosine-kNN bank and restricted to diagnoses seen with the predicted
  organ in training.

The MLP head + kNN bank live in ``reg2_titan_head/`` and the gated-attention
ABMIL head in ``reg2_abmil/``. Both are tiny (<5 MB) and run on the inference GPU
or CPU with negligible cost; the patch features needed by ABMIL are already
produced by the online CONCH pass, so there is no extra vision compute.

v13 adds an optional train-split diagnosis bias (``dx_bias.json``) learned on
the frozen ensemble probabilities. It is deliberately tiny and only shifts the
primary-diagnosis argmax when the ensemble is already close.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.reg2_text_calibrator import LabelContext

NONE = "<none>"
OTHER = "<other>"

W_MLP = 0.35
W_ABMIL = 0.55
W_KNN = 0.10
KNN_K = 20
USE_ORGAN_MASK = True


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class EnsemblePredictor:
    """MLP(all heads) + ABMIL(dx) ensemble producing a calibrator LabelContext."""

    def __init__(
        self,
        mlp_dir: str | Path,
        abmil_dir: str | Path,
        *,
        extra_abmil_dir: str | Path | None = None,
        extra_patch_key: str = "patch_features_v2",
        w_extra_abmil: float = 0.0,
        extra_abmil_specs: list[tuple[str | Path, str, float]] | None = None,
        extra_mlp_specs: list[tuple[str | Path, float]] | None = None,
        candidate_abmil_specs: list[tuple[str | Path, str]] | None = None,
        candidate_mlp_specs: list[str | Path] | None = None,
        source_weights_by_organ: dict[str, dict[str, float]] | None = None,
        organ_mask_groups: dict[str, list[str]] | None = None,
        w_mlp: float = W_MLP,
        w_abmil: float = W_ABMIL,
        w_knn: float = W_KNN,
        knn_k: int = KNN_K,
        use_organ_mask: bool = USE_ORGAN_MASK,
    ):
        import torch
        import torch.nn as nn

        mlp_dir = Path(mlp_dir)
        abmil_dir = Path(abmil_dir)
        self.available = (mlp_dir / "multitask_heads.pt").is_file() and (
            abmil_dir / "abmil_heads.pt"
        ).is_file()
        if not self.available:
            return

        self.torch = torch
        self.w_mlp = float(w_mlp)
        self.w_abmil = float(w_abmil)
        self.w_knn = float(w_knn)
        self.knn_k = int(knn_k)
        self.use_organ_mask = bool(use_organ_mask)
        self.source_weights_by_organ = source_weights_by_organ or {}
        self.organ_mask_groups = {
            str(k): [str(x) for x in v]
            for k, v in (organ_mask_groups or {}).items()
            if isinstance(v, (list, tuple)) and v
        }
        self.organ_to_mask_group: dict[str, str] = {}
        for group, organs in self.organ_mask_groups.items():
            for organ in organs:
                self.organ_to_mask_group[str(organ)] = group

        # ---- MLP head (all label fields) ------------------------------------
        mck = torch.load(mlp_dir / "multitask_heads.pt", map_location="cpu", weights_only=False)
        self.mv: dict[str, list[str]] = json.loads(
            (mlp_dir / "label_vocab.json").read_text(encoding="utf-8")
        )["vocabs"]
        self.mmean = np.asarray(mck["scaler_mean"], np.float32)
        self.mstd = np.asarray(mck["scaler_std"], np.float32)
        self.mlp_input_keys: list[str] = mck.get("input_keys", ["slide_embedding"])

        class MLPHead(nn.Module):
            def __init__(self, d, h, hs, nl=1, dp=0.1):
                super().__init__()
                layers: list[nn.Module] = []
                for i in range(max(1, nl)):
                    layers += [nn.Linear(d if i == 0 else h, h), nn.GELU(), nn.Dropout(dp)]
                self.trunk = nn.Sequential(*layers)
                self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in hs.items()})

            def forward(self, x):
                z = self.trunk(x)
                return {k: layer(z) for k, layer in self.heads.items()}

        self.mlp = MLPHead(
            self.mmean.shape[0],
            mck["hidden"],
            mck["head_sizes"],
            mck.get("num_layers", 1),
            mck.get("dropout", 0.1),
        )
        self.mlp.load_state_dict(mck["model_state"])
        self.mlp.eval()

        # ---- ABMIL head (primary_dx) ----------------------------------------
        ack = torch.load(abmil_dir / "abmil_heads.pt", map_location="cpu", weights_only=False)
        self.av: dict[str, list[str]] = json.loads(
            (abmil_dir / "label_vocab.json").read_text(encoding="utf-8")
        )["vocabs"]
        self.amean = np.asarray(ack["scaler_mean"], np.float32)
        self.astd = np.asarray(ack["scaler_std"], np.float32)
        self.max_patches = int(ack.get("max_patches", 256))

        # Single-head gated ABMIL (vanilla train_abmil_hopt0 checkpoint).
        class GatedABMIL(nn.Module):
            def __init__(self, d, h, hs):
                super().__init__()
                self.fc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.25))
                self.att_V = nn.Linear(h, h)
                self.att_U = nn.Linear(h, h)
                self.att_w = nn.Linear(h, 1)
                self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in hs.items()})

            def forward(self, x, mask):
                h = self.fc(x)
                a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))
                a = a.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
                a = torch.softmax(a, dim=1)
                return {k: layer((a * h).sum(1)) for k, layer in self.heads.items()}

        # Multi-head gated ABMIL (nnMIL recipe, train_abmil_nnmil checkpoint). With
        # n_heads=1 it is numerically equivalent to GatedABMIL but uses ModuleList
        # state keys (att_V.0.*), so we pick the class by checkpoint schema.
        class MHGatedABMIL(nn.Module):
            def __init__(self, d, h, hs, n_heads):
                super().__init__()
                self.n_heads = n_heads
                self.fc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.25))
                self.att_V = nn.ModuleList([nn.Linear(h, h) for _ in range(n_heads)])
                self.att_U = nn.ModuleList([nn.Linear(h, h) for _ in range(n_heads)])
                self.att_w = nn.ModuleList([nn.Linear(h, 1) for _ in range(n_heads)])
                self.heads = nn.ModuleDict({k: nn.Linear(h * n_heads, n) for k, n in hs.items()})

            def forward(self, x, mask):
                h = self.fc(x)
                zs = []
                for k in range(self.n_heads):
                    a = self.att_w[k](torch.tanh(self.att_V[k](h)) * torch.sigmoid(self.att_U[k](h)))
                    a = a.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
                    a = torch.softmax(a, dim=1)
                    zs.append((a * h).sum(1))
                z = torch.cat(zs, dim=-1) if self.n_heads > 1 else zs[0]
                return {k: layer(z) for k, layer in self.heads.items()}

        def build_abmil(ck):
            state = ck["model_state"]
            is_mh = ("num_heads" in ck) or any(k.startswith("att_V.0.") for k in state)
            if is_mh:
                model = MHGatedABMIL(
                    np.asarray(ck["scaler_mean"], np.float32).shape[0],
                    ck["hidden"],
                    ck["head_sizes"],
                    int(ck.get("num_heads", 1)),
                )
            else:
                model = GatedABMIL(
                    np.asarray(ck["scaler_mean"], np.float32).shape[0],
                    ck["hidden"],
                    ck["head_sizes"],
                )
            model.load_state_dict(state)
            model.eval()
            return model

        self.abmil = build_abmil(ack)

        # ---- dx vocab alignment (ABMIL vocab -> MLP vocab) ------------------
        self.dxv = self.mv["primary_dx"]
        midx = {d: i for i, d in enumerate(self.dxv)}
        self.a2m = [midx.get(n, midx.get(OTHER, 0)) for n in self.av["primary_dx"]]
        self.C = len(self.dxv)
        self.dx_bias = self._load_dx_bias(mlp_dir / "dx_bias.json")

        self.emit_source_scores = bool(candidate_abmil_specs or candidate_mlp_specs)

        self.extra_mlps: list[dict[str, object]] = []
        mlp_specs: list[tuple[str | Path, float, bool]] = [
            (asset, weight, False) for asset, weight in (extra_mlp_specs or [])
        ]
        mlp_specs.extend((asset, 0.0, True) for asset in (candidate_mlp_specs or []))
        for extra_mlp_dir_i, weight_i, candidate_only in mlp_specs:
            weight = float(weight_i)
            if weight <= 0 and not candidate_only:
                continue
            extra_dir = Path(extra_mlp_dir_i)
            if not (
                (extra_dir / "multitask_heads.pt").is_file()
                and (extra_dir / "label_vocab.json").is_file()
            ):
                continue
            eck = torch.load(extra_dir / "multitask_heads.pt", map_location="cpu", weights_only=False)
            ev = json.loads((extra_dir / "label_vocab.json").read_text(encoding="utf-8"))["vocabs"]
            if "primary_dx" not in eck.get("head_sizes", {}) or "primary_dx" not in ev:
                continue
            model = MLPHead(
                np.asarray(eck["scaler_mean"], np.float32).shape[0],
                eck["hidden"],
                eck["head_sizes"],
                eck.get("num_layers", 1) or 1,
                eck.get("dropout", 0.1) if eck.get("dropout", None) is not None else 0.1,
            )
            model.load_state_dict(eck["model_state"])
            model.eval()
            self.extra_mlps.append(
                {
                    "name": f"mlp:{extra_dir.name}",
                    "weight": weight,
                    "model": model,
                    "mean": np.asarray(eck["scaler_mean"], np.float32),
                    "std": np.asarray(eck["scaler_std"], np.float32),
                    "input_keys": eck.get("input_keys", ["slide_embedding"]),
                    "a2m": [midx.get(n, midx.get(OTHER, 0)) for n in ev["primary_dx"]],
                    "candidate_only": bool(candidate_only),
                }
            )

        self.extra_abmils: list[dict[str, object]] = []
        specs: list[tuple[str | Path, str, float, bool]] = []
        if extra_abmil_dir and float(w_extra_abmil) > 0:
            specs.append((extra_abmil_dir, str(extra_patch_key), float(w_extra_abmil), False))
        specs.extend((asset, patch_key, weight, False) for asset, patch_key, weight in (extra_abmil_specs or []))
        specs.extend((asset, patch_key, 0.0, True) for asset, patch_key in (candidate_abmil_specs or []))
        for extra_abmil_dir_i, patch_key_i, weight_i, candidate_only in specs:
            weight = float(weight_i)
            if weight <= 0 and not candidate_only:
                continue
            extra_dir = Path(extra_abmil_dir_i)
            if not (
                (extra_dir / "abmil_heads.pt").is_file()
                and (extra_dir / "label_vocab.json").is_file()
            ):
                continue
            eck = torch.load(extra_dir / "abmil_heads.pt", map_location="cpu", weights_only=False)
            ev = json.loads((extra_dir / "label_vocab.json").read_text(encoding="utf-8"))["vocabs"]
            self.extra_abmils.append(
                {
                    "name": f"abmil:{extra_dir.name}",
                    "patch_key": str(patch_key_i),
                    "weight": weight,
                    "mean": np.asarray(eck["scaler_mean"], np.float32),
                    "std": np.asarray(eck["scaler_std"], np.float32),
                    "max_patches": int(eck.get("max_patches", 256)),
                    "model": build_abmil(eck),
                    "a2m": [midx.get(n, midx.get(OTHER, 0)) for n in ev["primary_dx"]],
                    "candidate_only": bool(candidate_only),
                }
            )

        # ---- optional kNN bank + organ->valid-dx mask ----------------------
        self.bank_emb = None
        self.bank_dx = None
        self.organ_dx_mask: dict[str, np.ndarray] | None = None
        if self.w_knn > 0 or self.use_organ_mask:
            bp = mlp_dir / "dx_knn_bank.npz"
            if bp.is_file():
                b = np.load(bp, allow_pickle=True)
                be = np.asarray(b["embeddings"], np.float32)
                self.bank_emb = be / (np.linalg.norm(be, axis=1, keepdims=True) + 1e-8)
                self.bank_dx = np.asarray(b["dx_idx"], np.int64)
                organs = [str(o) for o in b["organ"]]
                masks: dict[str, np.ndarray] = {}
                for o, di in zip(organs, self.bank_dx):
                    masks.setdefault(o, np.zeros(self.C, bool))
                    if 0 <= di < self.C:
                        masks[o][di] = True
                for group, group_organs in self.organ_mask_groups.items():
                    merged = np.zeros(self.C, bool)
                    for organ in group_organs:
                        organ_mask = masks.get(organ)
                        if organ_mask is not None:
                            merged |= organ_mask
                    if merged.any():
                        masks[group] = merged
                self.organ_dx_mask = masks

    def _load_dx_bias(self, path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            bias = np.asarray(obj.get("bias", []), np.float32)
            vocab = obj.get("dx_vocab")
            if bias.shape == (self.C,) and (not vocab or list(vocab) == self.dxv):
                return bias
        except Exception:
            return None
        return None

    def _mlp_vector(self, features: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(features[k], dtype=np.float32).ravel() for k in self.mlp_input_keys]
        )

    def _head_label(self, head: str, idx: int) -> str:
        val = self.mv[head][idx]
        return "" if val in (NONE, OTHER) else val

    def _top_dx_scores(self, probs: np.ndarray, k: int) -> list[tuple[str, float]]:
        k = max(1, min(int(k), len(self.dxv)))
        order = np.argsort(-probs)[:k]
        return [
            (self.dxv[int(i)], float(probs[int(i)]))
            for i in order
            if self.dxv[int(i)] not in (OTHER, NONE)
        ]

    def _top_head_scores(self, head: str, probs: np.ndarray, k: int) -> list[tuple[str, float]]:
        labels = self.mv.get(head) or []
        if not labels:
            return []
        k = max(1, min(int(k), len(labels)))
        order = np.argsort(-probs)[:k]
        return [
            (str(labels[int(i)]), float(probs[int(i)]))
            for i in order
            if str(labels[int(i)]) not in (OTHER, NONE)
        ]

    def _calibrate_dx_scores(self, scores: np.ndarray) -> np.ndarray:
        total = float(scores.sum()) or 1.0
        probs = np.asarray(scores, np.float32) / total
        if self.dx_bias is not None:
            logits = np.log(np.clip(probs, 1e-8, 1.0)) + self.dx_bias
            logits = logits - float(logits.max())
            exp = np.exp(logits).astype(np.float32)
            if exp.sum() > 0:
                probs = exp / exp.sum()
        return probs

    def predict_with_scores(
        self,
        features: dict[str, np.ndarray],
        *,
        dx_topk: int = 5,
    ) -> tuple[LabelContext, dict[str, list[tuple[str, float]]]]:
        if not self.available:
            return LabelContext(), {}
        torch = self.torch
        emb = self._mlp_vector(features)
        pf = np.asarray(features["patch_features"], np.float32)[: self.max_patches]
        if pf.ndim != 2 or pf.shape[0] == 0:
            raise RuntimeError("ensemble needs non-empty 2D patch_features")

        with torch.no_grad():
            mo = self.mlp(torch.from_numpy(((emb - self.mmean) / self.mstd)[None].astype(np.float32)))
            mp = mo["primary_dx"].softmax(1).numpy()[0]
            mlp_scores = {head: logits.softmax(1).numpy()[0] for head, logits in mo.items()}
            x = torch.from_numpy(((pf - self.amean) / self.astd)[None].astype(np.float32))
            ao = self.abmil(x, torch.ones((1, pf.shape[0])))["primary_dx"].softmax(1).numpy()[0]

        organ_name = self._head_label("organ", int(mlp_scores["organ"].argmax()))

        ens = np.zeros(self.C, np.float32)
        terms: list[tuple[str, float, np.ndarray]] = [("mlp", self.w_mlp, mp.astype(np.float32))]
        abmil_mix = np.zeros(self.C, np.float32)
        for j, mi in enumerate(self.a2m):
            abmil_mix[mi] += ao[j]
        terms.append(("abmil", self.w_abmil, abmil_mix))

        if self.w_knn > 0 and self.bank_emb is not None:
            q = emb / (np.linalg.norm(emb) + 1e-8)
            sims = self.bank_emb @ q
            k = min(self.knn_k, sims.shape[0])
            nn_idx = np.argpartition(-sims, k - 1)[:k]
            knn = np.zeros(self.C, np.float32)
            for jj in nn_idx:
                if sims[jj] > 0:
                    knn[self.bank_dx[jj]] += sims[jj]
            if knn.sum() > 0:
                knn /= knn.sum()
                terms.append(("knn", self.w_knn, knn))

        extra_terms: list[tuple[float, np.ndarray]] = []
        for extra in self.extra_mlps:
            input_keys = [str(k) for k in extra["input_keys"]]
            if any(k not in features for k in input_keys):
                continue
            extra_emb = np.concatenate(
                [np.asarray(features[k], dtype=np.float32).ravel() for k in input_keys]
            )
            mean = np.asarray(extra["mean"], np.float32)
            if extra_emb.shape != mean.shape:
                continue
            with torch.no_grad():
                eo = extra["model"](
                    torch.from_numpy(
                        (
                            (extra_emb - mean)
                            / np.asarray(extra["std"], np.float32)
                        )[None].astype(np.float32)
                    )
                )["primary_dx"].softmax(1).numpy()[0]
            mixed = np.zeros(self.C, np.float32)
            for j, mi in enumerate(extra["a2m"]):
                mixed[mi] += eo[j]
            weight = max(0.0, float(extra["weight"]))
            terms.append((str(extra["name"]), weight, mixed))
            if not bool(extra.get("candidate_only")):
                extra_terms.append((weight, mixed))

        for extra in self.extra_abmils:
            patch_key = str(extra["patch_key"])
            if patch_key not in features:
                continue
            epf = np.asarray(features[patch_key], np.float32)[: int(extra["max_patches"])]
            if epf.ndim == 2 and epf.shape[0] > 0:
                with torch.no_grad():
                    ex = torch.from_numpy(
                        (
                            (
                                epf
                                - np.asarray(extra["mean"], np.float32)
                            )
                            / np.asarray(extra["std"], np.float32)
                        )[None].astype(np.float32)
                    )
                    eo = extra["model"](
                        ex,
                        torch.ones((1, epf.shape[0])),
                    )["primary_dx"].softmax(1).numpy()[0]
                mixed = np.zeros(self.C, np.float32)
                for j, mi in enumerate(extra["a2m"]):
                    mixed[mi] += eo[j]
                weight = max(0.0, float(extra["weight"]))
                terms.append((str(extra["name"]), weight, mixed))
                if not bool(extra.get("candidate_only")):
                    extra_terms.append((weight, mixed))

        organ_weights = self._source_weights_for_organ(organ_name)
        if organ_weights:
            matched = [(float(organ_weights.get(name, 0.0)), probs) for name, _weight, probs in terms]
            total = sum(max(0.0, weight) for weight, _probs in matched)
            if total > 0:
                ens = np.zeros(self.C, np.float32)
                for weight, probs in matched:
                    if weight > 0:
                        ens += float(weight) * probs
            else:
                organ_weights = {}
        if not organ_weights:
            ens = np.zeros(self.C, np.float32)
            ens += self.w_mlp * mp
            ens += self.w_abmil * abmil_mix
            for name, weight, probs in terms:
                if name == "knn":
                    ens += float(weight) * probs
            if extra_terms:
                total_extra = sum(weight for weight, _mixed in extra_terms)
                if total_extra > 1.0:
                    ens = sum((weight / total_extra) * mixed for weight, mixed in extra_terms)
                else:
                    ens = (1.0 - total_extra) * ens
                    for weight, mixed in extra_terms:
                        ens += weight * mixed
        elif extra_terms:
            # organ_weights already selected an explicit mixture over all
            # source names; no replacement-style extra blend is applied.
            pass

        if self.use_organ_mask and self.organ_dx_mask is not None:
            mask_key = self.organ_to_mask_group.get(organ_name, organ_name)
            mask = self.organ_dx_mask.get(mask_key)
            if mask is not None and mask.any():
                masked = ens * mask
                if masked.sum() > 0:
                    ens = masked

        dx_scores = self._calibrate_dx_scores(ens)
        di = int(dx_scores.argmax())
        dx_name = self.dxv[di] if self.dxv[di] not in (OTHER, NONE) else ""
        context = LabelContext(
            organ=organ_name,
            procedure=self._head_label("procedure", int(mo["procedure"].argmax(1).item())),
            diagnoses=[dx_name] if dx_name else [],
            histologic_type=self._head_label(
                "histologic_type", int(mo["histologic_type"].argmax(1).item())
            ),
            grade=self._head_label("grade", int(mo["grade"].argmax(1).item())),
            behavior=self._head_label("behavior", int(mo["behavior"].argmax(1).item())),
        )
        k = max(1, min(int(dx_topk), len(self.dxv)))
        scores = {"primary_dx": self._top_dx_scores(dx_scores, k)}
        for head in ("organ", "procedure", "histologic_type", "grade", "behavior"):
            if head in mlp_scores:
                scores[head] = self._top_head_scores(head, mlp_scores[head], k)
        if _truthy(os.environ.get("REG2_DUMP_SOURCE_PROBS")) or self.emit_source_scores:
            scores["source_active_weights"] = [
                (name, float(weight)) for name, weight, _probs in terms
            ]
            for name, weight, probs in terms:
                scores[f"source:{name}"] = self._top_dx_scores(probs, k)
            if organ_weights:
                scores["source_profile_weights"] = [
                    (name, float(weight)) for name, weight in organ_weights.items()
                ]
        return context, scores

    def predict(self, features: dict[str, np.ndarray]) -> LabelContext:
        return self.predict_with_scores(features)[0]

    def _source_weights_for_organ(self, organ_name: str) -> dict[str, float]:
        if not self.source_weights_by_organ:
            return {}
        for key in (organ_name, "", "__default__", "default"):
            raw = self.source_weights_by_organ.get(key)
            if raw:
                return {str(k): float(v) for k, v in raw.items()}
        return {}
