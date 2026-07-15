# Training and evaluation reproduction

The commands below make every input and output location explicit. Replace the
four path variables with local locations. All generated data should be written
outside the repository.

```bash
export COT=/path/to/train_CoT_v01.json
export WSI_DIR=/path/to/training_slides
export RUN=/path/to/reproduction_workspace
export PY=python

mkdir -p "$RUN"
```

## 1. Build text and workflow assets

```bash
$PY scripts/build_reg2026_calibrator_assets.py \
  --cot "$COT" \
  --out "$RUN/reg2_calibrator"

$PY scripts/build_reg2026_labelgraph_calibrator_assets.py \
  --cot "$COT" \
  --v1-assets "$RUN/reg2_calibrator" \
  --split "$RUN/reg2_calibrator/split.json" \
  --out "$RUN/reg2_calibrator_v2_labelgraph"

$PY scripts/build_reg2026_structured_report_assets.py \
  --cot "$COT" \
  --out "$RUN/reg2_calibrator_v2_labelgraph/structured_report_assets"

$PY scripts/build_reg2026_v02_assets.py \
  --cot "$COT" \
  --out "$RUN/reg2_v02"
```

Do not enable `--include-case-lookup` when building deployable v02 assets.

## 2. Extract frozen visual features

First download the three publisher-controlled models as described in the root
README. The H-optimus-1 and Virchow2 jobs can be sharded by setting
`--num-shards` and `--shard-index` independently on each GPU.

```bash
$PY scripts/extract_hopt1_full.py \
  --cot "$COT" \
  --dataset-dir "$WSI_DIR" \
  --hoptimus-dir models/H-optimus \
  --out "$RUN/hoptimus1_full_256"

$PY scripts/extract_virchow2_full.py \
  --cot "$COT" \
  --dataset-dir "$WSI_DIR" \
  --virchow-dir models/Virchow2 \
  --out "$RUN/virchow2_full_256"
```

TITAN extraction uses one staging producer and one or more workers:

```bash
$PY scripts/extract_titan_features.py \
  --role producer \
  --manifest "$COT" \
  --dataset-dir "$WSI_DIR" \
  --titan-dir models/TITAN \
  --out "$RUN/titan_full_256" \
  --save-patch-features &

CUDA_VISIBLE_DEVICES=0 $PY scripts/extract_titan_features.py \
  --role worker \
  --worker-id 0 \
  --manifest "$COT" \
  --dataset-dir "$WSI_DIR" \
  --titan-dir models/TITAN \
  --out "$RUN/titan_full_256" \
  --save-patch-features

wait
```

Build the 4864-dimensional report feature used by the report and slot heads:

```bash
$PY scripts/build_report_fused_features.py \
  --titan "$RUN/titan_full_256" \
  --hopt1 "$RUN/hoptimus1_full_256" \
  --virchow2 "$RUN/virchow2_full_256" \
  --out "$RUN/report_fused_titan_h1_v2_256"
```

## 3. Train compact heads

The saved checkpoints retain their architecture and training configuration.
Use their JSON metadata together with each script's `--help` output when
matching a particular packaged checkpoint.

```bash
$PY scripts/train_reg2026_multitask_heads.py \
  --features "$RUN/titan_full_256" \
  --input-keys slide_embedding \
  --cot "$COT" \
  --split "$RUN/reg2_calibrator/split.json" \
  --out "$RUN/reg2_titan_head"

$PY scripts/build_dx_knn_bank.py \
  --features "$RUN/titan_full_256" \
  --model "$RUN/reg2_titan_head" \
  --assets "$RUN/reg2_calibrator" \
  --cot "$COT"

$PY scripts/train_abmil_hopt0.py \
  --features "$RUN/hoptimus1_full_256" \
  --feat-key patch_features_h1 \
  --cot "$COT" \
  --split "$RUN/reg2_calibrator/split.json" \
  --out "$RUN/reg2_abmil_h1"

$PY scripts/train_primary_dx_dxabmil_fulltrain.py \
  --features "$RUN/hoptimus1_full_256" \
  --patch-key patch_features_h1 \
  --assets "$RUN/reg2_calibrator" \
  --cot "$COT" \
  --out "$RUN/reg2_h1_mil_rescue"

$PY scripts/train_report_clf.py \
  --features "$RUN/report_fused_titan_h1_v2_256" \
  --feat-key fused \
  --cot "$COT" \
  --split "$RUN/reg2_calibrator/split.json" \
  --out "$RUN/reg2_report_clf_fused_s101"

$PY scripts/train_report_slot_heads.py \
  --features "$RUN/report_fused_titan_h1_v2_256" \
  --feat-key fused \
  --cot "$COT" \
  --split "$RUN/reg2_calibrator/split.json" \
  --out "$RUN/reg2_report_slot_heads"

$PY scripts/train_report_hard_confusion_heads.py \
  --features "$RUN/report_fused_titan_h1_v2_256" \
  --feat-key fused \
  --cot "$COT" \
  --split "$RUN/reg2_calibrator/split.json" \
  --out "$RUN/reg2_report_hard_heads"

$PY scripts/train_labelgraph_secondary_mil_heads.py \
  --features "$RUN/virchow2_full_256" \
  --patch-key patch_features_v2 \
  --cot "$COT" \
  --split "$RUN/reg2_calibrator/split.json" \
  --out "$RUN/reg2_secondary_mil_heads"
```

The organ-specific slot bundles use the same slot trainer with `--organ` and
`--slots`. The dense 1024-patch ABMIL bundle uses the same ABMIL trainer with
`--max-patches 1024`. The diagnosis-bias vector shipped as `dx_bias.json`
beside the multi-task checkpoint is fitted offline on the validation split.

The ROI gate is trained independently:

```bash
$PY scripts/roi_tissue_build_dataset.py \
  --dataset-dir "$WSI_DIR" \
  --out "$RUN/roi_gate_dataset"

$PY scripts/roi_tissue_train_cnn_v3.py \
  --data "$RUN/roi_gate_dataset" \
  --out "$RUN/reg2_roi_gate"
```

## 4. Evaluate and package

```bash
$PY scripts/official_eval_val_quiet.py \
  --gt /path/to/validation_ground_truth.json \
  --pred /path/to/predictions.json \
  --out "$RUN/official_summary.json" \
  --embedding-model /path/to/pubmedbert-base-embeddings

$PY scripts/check_reg2_ensemble_profile.py \
  --profile models/reg2_ensemble_profile/package_default.json \
  --model-root models

$PY scripts/check_submission_model_assets.py \
  --model-root models \
  --require-package-default
```

Only promote outputs after comparing them with the shipped checksums and the
active profile. The `models/` directory is the exact root packed into
`model.tar.gz` by `do_save.sh`.
