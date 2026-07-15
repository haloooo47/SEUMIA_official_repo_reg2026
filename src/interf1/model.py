"""
Interface 1 — Workflow Reasoning (template implementation).

This is where you plug in your chain-of-thought / reasoning model.

Steps:
  1. Load your model weights (MODEL_PATH is available from core.py).
  2. Replace the body of predict_chain_of_thought with your inference logic.
  3. If your model lives in a separate repository, add it under src/ and import it here:

       from src.your_repo.your_module import YourReasoningModel

Platform rules for this interface:
  - Use the EXACT canonical question / next_question strings from the training
    annotations. Minor whitespace/capitalisation differences are normalised, but
    paraphrasing is penalised.
  - Output is a bare JSON array of steps — do NOT wrap it in an object with an
    "id" key. The platform adds that wrapper automatically.
  - The last step must have "next_question": "".

Output schema per step:
  {
    "question":      "<exact canonical question string>",
    "answer":        "<your model's free-text answer>",
    "next_question": "<exact canonical next question string, or empty string>"
  }
"""

from __future__ import annotations

from pathlib import Path

from src.reg2_pipeline import ChainOfThoughtStep

try:
    from src.reg2_v20_pipeline import predict_v20_chain_of_thought as _predict_chain_of_thought
    _V20_IMPORT_ERROR: Exception | None = None
except Exception as _v20_import_exc:  # pragma: no cover
    _V20_IMPORT_ERROR = _v20_import_exc

    def _predict_chain_of_thought(
        *,
        wsi_path: Path,
    ) -> list[ChainOfThoughtStep]:
        print(
            "[interf1] v20 import failed; using single emergency fallback: "
            f"{type(_V20_IMPORT_ERROR).__name__}: {_V20_IMPORT_ERROR}"
        )
        from src.reg2_v02_pipeline import predict_v02_chain_of_thought

        return predict_v02_chain_of_thought(wsi_path=wsi_path)

def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """
    Run Workflow Reasoning inference for a single whole-slide image.

    Args:
        wsi_path: Path to /input/images/whole-slide-image/<uid>.tiff
            (<uid> is an opaque UUID hash from inputs.json image.name).

    Returns:
        A list of steps, each with keys: question, answer, next_question.
        Use exact canonical question strings from the training annotations.

    IMPORTANT — do not change the return type (must stay
    ``list[ChainOfThoughtStep]``). inference.py serialises this list directly
    to chain-of-thought.json; wrapping it or changing field names will break
    submission validation.
    """
    return _predict_chain_of_thought(wsi_path=wsi_path)
