"""Online label predictor: slide features -> LabelContext.

Loads the compact multi-task heads trained by
``scripts/train_reg2026_multitask_heads.py`` and predicts a ``LabelContext`` that
feeds ``TextCalibrator.render_from_labels``. The model is a tiny MLP (<0.1 GB) and
runs fine on CPU or the single inference GPU; it adds negligible time/VRAM.

Diagnosis (the dominant Metric A lever) gets two cheap, zero-extra-compute
boosts when a kNN bank asset (``dx_knn_bank.npz``) is present next to the model:

* MLP + cosine-kNN ensemble on the embedding (complementary error profiles).
* organ-conditioned masking: restrict the primary_dx choice to diagnoses seen
  with the predicted organ in training.

Both are no-ops when the bank is absent, so the predictor stays backward
compatible and degrades gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.reg2_text_calibrator import LabelContext

NONE = "<none>"
OTHER = "<other>"


class LabelPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        knn_alpha: float = 0.5,
        knn_k: int = 20,
        use_organ_mask: bool = True,
    ):
        import torch
        import torch.nn as nn

        model_dir = Path(model_dir)
        self.available = (model_dir / "multitask_heads.pt").is_file()
        if not self.available:
            return

        ckpt = torch.load(model_dir / "multitask_heads.pt", map_location="cpu", weights_only=False)
        vocab = json.loads((model_dir / "label_vocab.json").read_text(encoding="utf-8"))
        self.input_keys: list[str] = ckpt["input_keys"]
        self.vocabs: dict[str, list[str]] = vocab["vocabs"]
        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)
        head_sizes: dict[str, int] = ckpt["head_sizes"]
        in_dim = self.scaler_mean.shape[0]

        class MultiHead(nn.Module):
            def __init__(self, in_dim, hidden, head_sizes, num_layers=1, dropout=0.1):
                super().__init__()
                layers: list[nn.Module] = []
                for layer_idx in range(max(1, int(num_layers))):
                    layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden, hidden))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                self.trunk = nn.Sequential(*layers)
                self.heads = nn.ModuleDict({h: nn.Linear(hidden, n) for h, n in head_sizes.items()})

            def forward(self, x):
                z = self.trunk(x)
                return {h: layer(z) for h, layer in self.heads.items()}

        self.torch = torch
        self.model = MultiHead(
            in_dim,
            ckpt["hidden"],
            head_sizes,
            ckpt.get("num_layers", 1),
            ckpt.get("dropout", 0.1),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # Optional kNN bank for diagnosis: normalized embeddings + dx/organ labels.
        self.knn_alpha = float(knn_alpha)
        self.knn_k = int(knn_k)
        self.use_organ_mask = bool(use_organ_mask)
        self._bank_emb = None       # (N, D) L2-normalized
        self._bank_dx = None        # (N,) int dx index
        self._organ_dx_mask = None  # {organ_name: bool mask over dx vocab}
        bank_path = model_dir / "dx_knn_bank.npz"
        if bank_path.is_file():
            self._load_bank(bank_path)

    def _load_bank(self, path: Path) -> None:
        b = np.load(path, allow_pickle=True)
        emb = np.asarray(b["embeddings"], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        self._bank_emb = emb
        self._bank_dx = np.asarray(b["dx_idx"], dtype=np.int64)
        organs = [str(o) for o in b["organ"]]
        n_dx = len(self.vocabs["primary_dx"])
        masks: dict[str, np.ndarray] = {}
        for organ, di in zip(organs, self._bank_dx):
            if organ not in masks:
                masks[organ] = np.zeros(n_dx, dtype=bool)
            if 0 <= di < n_dx:
                masks[organ][di] = True
        self._organ_dx_mask = masks

    def _vector(self, features: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(features[k], dtype=np.float32).ravel() for k in self.input_keys]
        )

    def _forward_logits(self, vec: np.ndarray):
        x = (vec - self.scaler_mean) / self.scaler_std
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x[None, :].astype(np.float32)))

    def _dx_probs(self, vec: np.ndarray, dx_logits, organ_name: str) -> np.ndarray:
        """Return primary_dx probabilities after kNN ensemble + organ masking."""
        probs = self.torch.softmax(dx_logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        if self._bank_emb is not None and self.knn_alpha < 1.0:
            q = vec / (np.linalg.norm(vec) + 1e-8)
            sims = self._bank_emb @ q
            k = min(self.knn_k, sims.shape[0])
            nn_idx = np.argpartition(-sims, k - 1)[:k]
            knn = np.zeros_like(probs)
            for j in nn_idx:
                s = sims[j]
                if s > 0:
                    knn[self._bank_dx[j]] += s
            ssum = knn.sum()
            if ssum > 0:
                knn /= ssum
                probs = self.knn_alpha * probs + (1.0 - self.knn_alpha) * knn
        if self.use_organ_mask and self._organ_dx_mask is not None:
            mask = self._organ_dx_mask.get(organ_name)
            if mask is not None and mask.any():
                masked = probs * mask
                if masked.sum() > 0:
                    probs = masked
        return probs

    def _context_from_pick(self, pick: dict[str, int]) -> LabelContext:
        def label(head: str) -> str:
            val = self.vocabs[head][pick[head]] if head in pick else ""
            return "" if val in (NONE, OTHER) else val

        primary = label("primary_dx")
        return LabelContext(
            organ=label("organ"),
            procedure=label("procedure"),
            diagnoses=[primary] if primary else [],
            histologic_type=label("histologic_type"),
            grade=label("grade"),
            behavior=label("behavior"),
        )

    def _pick(self, features: dict[str, np.ndarray]) -> tuple[dict[str, int], dict, np.ndarray]:
        vec = self._vector(features)
        out = self._forward_logits(vec)
        pick = {h: int(out[h].argmax(1).item()) for h in out}
        if "primary_dx" in out:
            organ_name = self.vocabs["organ"][pick["organ"]] if "organ" in pick else ""
            dx_probs = self._dx_probs(vec, out["primary_dx"], organ_name)
            pick["primary_dx"] = int(dx_probs.argmax())
        return pick, out, vec

    def predict(self, features: dict[str, np.ndarray]) -> LabelContext:
        if not self.available:
            return LabelContext()
        pick, _, _ = self._pick(features)
        return self._context_from_pick(pick)

    def predict_with_scores(
        self,
        features: dict[str, np.ndarray],
        topk: int = 5,
    ) -> tuple[LabelContext, dict[str, list[tuple[str, float]]]]:
        if not self.available:
            return LabelContext(), {}
        pick, out, _ = self._pick(features)
        scores: dict[str, list[tuple[str, float]]] = {}
        for head, logits in out.items():
            probs = self.torch.softmax(logits, dim=1).squeeze(0)
            k = min(max(1, int(topk)), int(probs.numel()))
            vals, idxs = self.torch.topk(probs, k)
            scores[head] = [
                (self.vocabs[head][int(idx)], float(prob))
                for prob, idx in zip(vals.cpu(), idxs.cpu())
            ]
        return self._context_from_pick(pick), scores
