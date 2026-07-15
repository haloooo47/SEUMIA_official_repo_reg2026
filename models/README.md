# Model assets

This directory contains the compact trained heads, calibrators, priors, and the
active runtime profile used by the submitted method. Their checksums are listed
in `SHA256SUMS`.

Three upstream foundation models are required at inference time but are not
redistributed in this repository because their files are too large for GitHub
and their licenses require each user to obtain access directly from the model
publisher:

- `TITAN/` from `MahmoodLab/TITAN`
- `H-optimus/` from `bioptimus/H-optimus-1`
- `Virchow2/` from `paige-ai/Virchow2`

After accepting the upstream access terms, set `HF_TOKEN` in the environment
and run:

```bash
python scripts/download_foundation_models.py --model-root models
python scripts/check_submission_model_assets.py \
  --model-root models \
  --require-package-default
```

Do not commit the downloaded foundation-model directories. They are excluded by
`.gitignore`.
