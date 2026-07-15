#!/usr/bin/env bash
# Additive GPU env bootstrap for REG2 v2.0 offline feature extraction + training.
#
# Adds the missing, lightweight deps to an existing torch env. By default it
# targets the active Python environment. These installs are additive (no torch
# reinstall) and do not touch challenge data.
#
# Usage:
#   bash scripts/setup_reg2026_gpu_env.sh            # use python3
#   REG2_PY=/path/to/python bash scripts/setup_reg2026_gpu_env.sh
set -euo pipefail

REG2_PY="${REG2_PY:-python3}"

echo "[setup] using python: ${REG2_PY}"
"${REG2_PY}" -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo "[setup] installing additive deps (timm tiffslide h5py einops scikit-learn)"
"${REG2_PY}" -m pip install --no-input \
  "timm>=1.0.0" \
  "tiffslide>=2.3.0" \
  "h5py>=3.10.0" \
  "einops>=0.7.0" \
  "scikit-learn>=1.3.0"

echo "[setup] verifying imports + H-optimus architecture build (CPU)"
"${REG2_PY}" - <<'EOF'
import timm, tiffslide, h5py, einops, sklearn
print("[setup] timm", timm.__version__, "| tiffslide", tiffslide.__version__,
      "| h5py", h5py.__version__, "| einops", einops.__version__, "| sklearn", sklearn.__version__)
m = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, num_classes=0)
print("[setup] built H-optimus arch; num_features =", m.num_features)
EOF

echo "[setup] done. Ready for scripts/extract_hoptimus_features.py"
