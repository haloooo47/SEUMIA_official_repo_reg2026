# SEUMIA REG2026 Submission

## Submission information

| Field | Value |
| --- | --- |
| Team Name | SEUMIA |
| Grand Challenge Username | Xcccc Xue |
| Grand Challenge Profile | https://grand-challenge.org/users/SEUMIA/ |

## Method description

SEUMIA is a production v20 ensemble for the REG2026 challenge. Its production
entry point is `src/interf1/model.py`, which uses the active runtime profile at
`models/reg2_ensemble_profile/package_default.json`.

The method extracts frozen features with TITAN, H-optimus-1, and Virchow2, then
combines multi-task, ABMIL, report-classification, report-slot,
hard-confusion, secondary-finding, and ROI-gate components. The repository
contains the trained compact heads, calibrators, priors, and active ensemble
profile, together with the code for training, inference, evaluation, and
container packaging.

## Repository layout

- `src/`: inference pipeline and both challenge interfaces
- `scripts/`: feature extraction, training, calibration, and asset checks
- `evaluation/official/`: official metric implementation used for local scoring
- `models/`: compact trained heads, calibrators, priors, and active profile
- `tests/`: unit tests for the production pipeline and its verifiers
- `Dockerfile`, `do_build.sh`, `do_test_run.sh`, `do_save.sh`: submission packaging

## Inference instructions

Inference uses the CUDA-enabled PyTorch base image declared in `Dockerfile`.
Build it with:

```bash
./do_build.sh
```

For feature extraction and training outside Docker, create a Python environment
with a CUDA-enabled PyTorch build, then install:

```bash
python -m pip install -r requirements-training.txt
```

The original training environment used PyTorch 2.4.1 with CUDA 11.8. The
submission image uses PyTorch 2.5.1 with CUDA 12.1.

### Model weight download instructions

The public SEUMIA submission package is available as the GitHub Container
package [reg2-paper](https://github.com/users/haloooo47/packages/container/package/reg2-paper).
Pull the published package with:

```bash
docker pull ghcr.io/haloooo47/reg2-paper:paper-v1-no-foundation
```

The published package does not include the upstream foundation weights. The
three foundation models below must be downloaded separately before inference.

Inference requires TITAN, H-optimus-1, and Virchow2. They are not redistributed
here because of file-size and upstream-license restrictions. Each verifier must
request access from the corresponding Hugging Face publisher and then run:

```bash
export HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
python scripts/download_foundation_models.py --model-root models
```

The downloader pins publisher revisions. See `models/README.md` for the expected
directory layout. Never commit an access token or the downloaded foundation
weights.

### Asset verification

After downloading the foundation models:

```bash
python scripts/check_submission_model_assets.py \
  --model-root models \
  --require-package-default

python scripts/check_reg2_ensemble_profile.py \
  --profile models/reg2_ensemble_profile/package_default.json \
  --model-root models

sha256sum --check models/SHA256SUMS
```

### Local inference and container reproduction

The challenge mounts model contents at `/opt/ml/model`. For a local forward
pass, place the official interface fixtures under `test/input/interf0` and
`test/input/interf1`, then run:

```bash
./do_test_run.sh
```

To produce the container archive and separate model archive expected by the
challenge platform:

```bash
REG2_SAVE_DIR=/path/with/sufficient/space ./do_save.sh
```

The model archive contains the contents of `models/`, including the downloaded
foundation models. The foundation weights must remain private and be handled in
accordance with their publisher licenses.

## Training reproduction

All training commands accept explicit dataset, split, feature-store, and output
paths. A typical reproduction sequence is:

1. Build the workflow calibrator assets with
   `scripts/build_reg2026_calibrator_assets.py` and
   `scripts/build_reg2026_structured_report_assets.py`.
2. Extract frozen TITAN, H-optimus-1, and Virchow2 features with the matching
   `scripts/extract_*.py` programs.
3. Train the multi-task, ABMIL, report classifier, report slot, hard-confusion,
   secondary-finding, and ROI-gate heads with the corresponding `scripts/train_*.py`
   programs.
4. Evaluate predictions with `scripts/official_eval_val.py` or
   `scripts/official_eval_val_quiet.py`.
5. Promote the selected runtime configuration to
   `models/reg2_ensemble_profile/package_default.json` and run both asset checks.

Use `python <script> --help` for the exact arguments. Dataset files and extracted
feature stores are intentionally not included because they are challenge data or
generated intermediates.

A concrete end-to-end command template and the mapping from each shipped asset
to its generating script are provided in `docs/REPRODUCTION.md`.

## Tests

```bash
python -m pytest -q tests
python -m compileall -q src scripts evaluation tests core.py inference.py
```
