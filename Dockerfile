FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS reg2026_algorithm_amd64
# PyTorch + CUDA base so GPU is available at inference time.

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
# Ensure imports like "from core import ..." work from src/ submodules
ENV PYTHONPATH=/opt/app
# Enable the bounded structured-slot report patcher by default. It only edits
# final-report numeric slots and leaves workflow path / non-final answers intact.
ENV REG2_REPORT_SLOT_HEADS=1
ENV REG2_SLOT_MIN_PROB_GLEASON=0.55
ENV REG2_SLOT_MIN_PROB_BREAST=0.70
ENV REG2_SLOT_CANDIDATE_SELECTOR=1
ENV REG2_SLOT_CAND_MAX_RANK=3
ENV REG2_SLOT_CAND_MIN_PROB=0.05
ENV REG2_SLOT_CAND_MIN_JACCARD=0.30
ENV REG2_SLOT_CAND_MIN_PROB_GLEASON=0.45
ENV REG2_SLOT_CAND_MIN_PROB_BREAST=0.75
ENV REG2_V17_DX_TOPK=15
# High-confidence final-report classifier settings.
ENV REG2_FUSED_SELECTOR_ASSET=reg2_report_clf_fused_s101
ENV REG2_FUSED_SELECTOR_MIN_PROB=0.97
ENV REG2_FUSED_SELECTOR_MIN_RATIO=0.0
ENV REG2_FUSED_SELECTOR_BLOCK_UNSUPPORTED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/tmp/hf_cache
# Metric B uses the deterministic ROI gate and evidence templates.

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
# Use a fast mirror + generous timeout/retries: the build host has flaky access
# to the default PyPI CDN. This only affects build-time package download.
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --timeout 120 \
    --retries 10 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user core.py      /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user src/         /opt/app/src/

ENTRYPOINT ["python", "inference.py"]
