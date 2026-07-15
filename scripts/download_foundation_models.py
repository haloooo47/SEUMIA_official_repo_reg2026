#!/usr/bin/env python3
"""Download the licensed foundation-model files required by inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "TITAN": {
        "repo_id": "MahmoodLab/TITAN",
        "revision": "dac6773d9961cfc75503440676ff157a2c6e8d2e",
        "allow_patterns": [
            "conch_tokenizer.py",
            "conch_v1_5.py",
            "conch_v1_5_pytorch_model.bin",
            "config.json",
            "configuration_titan.py",
            "model.safetensors",
            "modeling_titan.py",
            "special_tokens_map.json",
            "text_transformer.py",
            "tokenizer.json",
            "tokenizer_config.json",
            "vision_transformer.py",
        ],
    },
    "H-optimus": {
        "repo_id": "bioptimus/H-optimus-1",
        "revision": "3592cb220dec7a150c5d7813fb56e68bd57473b9",
        "allow_patterns": ["config.json", "pytorch_model.bin"],
    },
    "Virchow2": {
        "repo_id": "paige-ai/Virchow2",
        "revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
        "allow_patterns": ["config.json", "model.safetensors"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(MODELS),
        default=sorted(MODELS),
        help="Download only selected model directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or None
    args.model_root.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        spec = MODELS[name]
        target = args.model_root / name
        print(f"[download] {name} -> {target}")
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            revision=str(spec["revision"]),
            allow_patterns=list(spec["allow_patterns"]),
            local_dir=target,
            token=token,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
