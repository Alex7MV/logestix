#!/usr/bin/env python3
"""gemma-4-12B-it (`gemma4_unified`) oracle for golden capture.

The checkpoint is a multimodal wrapper (Gemma4UnifiedForConditionalGeneration:
a `gemma4_unified_text` decoder plus a patch-embedding tower and one audio
projection). The text tower is what logestix ingests; the rest is stripped as
ballast and counted. Unlike the Muse-Glimmer 59.5 GB tree, this one FITS the
32 GB card in bf16 (24 GB of weights), so the oracle loads the wrapper as-is
with transformers' own `AutoModelForImageTextToText` — no weight surgery, no
CPU offload, and therefore no risk of the oracle's numerics drifting from the
checkpoint's because a layer was skipped.

Used by tools/{fsi,code,det}_capture.py via OVERRIDES
loader="gemma4_unified".

Why a dedicated loader at all: the generic branch loads through
`AutoModelForCausalLM`, and a `Gemma4UnifiedForConditionalGeneration`
checkpoint is not in that mapping (its class is the image-text-to-text one).
Framing needs nothing special: the checkpoint's `chat_template.jinja` is
byte-identical to gemma-4-31B-it's (sha256-verified at stage 0), and
`capture_opts()` is empty on purpose — every frame the engine serves for this
family is the plain template render with `add_generation_prompt`.

The torch/transformers imports are lazy so this module imports cleanly on a
host without the checkpoint stack (tools/*_capture.py --selftest).
"""


def load(model, device_map=None):
    """Load the whole unified checkpoint (bf16) and return the HF model."""
    import torch
    from transformers import AutoModelForImageTextToText

    lm = AutoModelForImageTextToText.from_pretrained(
        model,
        dtype=torch.bfloat16,
        device_map=device_map or "auto",
    )
    lm.eval()
    return lm


def capture_opts():
    """Capture overrides contributed by this loader (merged by the capture
    tools after the loader module is imported). Nothing to override: the family
    template and the tokenizer are the ones the engine serves."""
    return {}
