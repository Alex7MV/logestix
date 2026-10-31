#!/usr/bin/env python3
"""CPU-BF16 oracle loader for Granite-Switch checkpoints.

The checkpoint is `GraniteSwitchForCausalLM` (model_type granite_switch); the
class ships in the `granite-switch` package, not in transformers. Install it in
a venv:

    python3 -m venv --system-site-packages .venv-gs
    .venv-gs/bin/pip install --ignore-requires-python \
        'transformers==5.17.0' 'granite-switch==0.1.1'

Python 3.14 needs `--ignore-requires-python` (the package metadata caps at
<3.14; the code itself runs). Load on CPU with BF16: `device_map="auto"`
mixing GPU and CPU produced NaNs in the LoRA path, while full-CPU BF16 is
finite and deterministic (measured ~0.2 s/token at 30B on this host).
"""
import torch


def load(model_path, adapter_name=None):
    from transformers import AutoTokenizer
    from granite_switch.hf.modeling_granite_switch import (
        GraniteSwitchForCausalLM,
    )

    tok = AutoTokenizer.from_pretrained(model_path)
    model = GraniteSwitchForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model._granite_switch_adapter_name = adapter_name
    return tok, model
