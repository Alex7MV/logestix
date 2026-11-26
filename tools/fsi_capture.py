#!/usr/bin/env python3
"""FSISelfCheck reference capture (separate functionality from SelfCheck).

Captures the FSI golden answers from an original safetensors model with HF
transformers. The logestix side only verifies the converted model against the
committed golden (libs/lsxcommon/src/fsi_selfcheck.cpp). Run the self-test with
no model present to validate the prompt/formula set.
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "fsi_prompts.json")

# Per-model capture overrides keyed by model type. Absent -> generic defaults.
# `device`: "cuda" / "cpu" / None (auto: cuda when the full checkpoint fits in
# free VRAM, else cpu). `dtype`: torch dtype name or None (checkpoint default).
OVERRIDES = {
    # Muse-Glimmer-30B: the 59.5 GB BF16 multimodal tree does not fit the 32 GB
    # card; tools/muse_golden_oracle.py loads ONLY the text tower (language_model
    # + lm_head, skipping the 809 vision tensors) onto the 750 GB host CPU.
    # That oracle also supplies this model's capture facts (merged after the
    # loader is imported): the chat-template `reasoning_strength: low` switch,
    # which renders `Reasoning strength: low.` into the system block and stops
    # the high-effort reasoning loop that made the first FSI capture abort
    # (docs/OPEN_TASKS.md R21), and the ATEM answer/reasoning channel headers
    # the answer is read from. ignore_eos keeps the budget the verifier replays.
    "muse-glimmer-30b": {"loader": "muse_text", "ignore_eos": True},

    # nemotron-3.5-lightning-30b-a3b: the engine's frame for this checkpoint
    # opens `<|im_start|>assistant\n<think>\n` (chat_template.jinja defaults
    # enable_thinking=True and NemotronModelBase::RenderNemotronConversation
    # hardcodes it). The tool default (False) renders `<think></think>` - a
    # different prompt - and would capture a frame the served model never sees.
    # Pinned by tests/test_nemotron_moe_model.cpp "FSI-capture frame equals the
    # served frame" against tests/fixtures/nemotron/fsi01_frame.txt.
    "nemotron-3.5-lightning-30b-a3b": {"enable_thinking": True},

    # Hy-Memory-30B-A3B publishes the per-expert layout (mlp.experts.E.gate_proj
    # etc.) plus a vestigial `moe_intermediate_size`; tools/hy_v3_oracle.py
    # assembles the fused tensors the installed HYV3ForCausalLM expects. The
    # 60 GB bf16 tree runs on CPU (the card cannot hold it); no accelerate
    # device map, so no meta-device offload path.
    "hy-memory-30b-a3b": {"loader": "hy_v3"},
    # Hy-SimulMT-30B-A3B is the same hy_v3 text backbone (the audio tower
    # is skipped by the oracle); text-only capture, CPU for the 60 GB tree.
    "hy-simulmt-30b-a3b": {"loader": "hy_v3"},
    # ornith-1.5-35b-a3b is served with a leading system turn (its model
    # class injects one when the caller omits it); without it the safety
    # alignment refuses the FSI-02 financial prompt. Thinking stays off so
    # the answer is the short direct continuation.
    "ornith-1.5-35b-a3b": {"system": "You are a helpful assistant."},
    # qwen3.8-flash-next is an experimental "qwen4_exp" arch: the installed
    # transformers ships Qwen4Exp* but its layer-type allowlist and KV-cache
    # mapping predate "qwen_sparse_attention", and the PLE n-gram embedding is
    # shipped as 128 shards. The custom loader patches both.
    "qwen3.8-flash-next": {"loader": "qwen4_exp"},
    # granite-4.2 is a reasoning model whose non-thinking mode gets FSI-02
    # wrong; low-effort thinking answers all three correctly. Its chat template
    # lives in chat_template.jinja (not tokenizer_config).
    "granite-switch-4.1-30b": {"loader": "granite_switch"},
    # Same composed architecture as its 30B sibling and the same CPU-BF16
    # oracle (transformers has no GraniteSwitchForCausalLM; the class ships
    # in the granite-switch package). Base adapter: the gates compare the
    # checkpoint's own continuation, not an adaptation.
    "granite-switch-4.1-8b": {"loader": "granite_switch"},
    "granite-4.2-30b": {"enable_thinking": True, "low_effort": True},
    # Mistral-Small-4-119B ships the Mistral "consolidated" layout: params.json
    # (no config.json, no model_type) plus consolidated-*.safetensors with
    # NVFP4-packed MoE experts. transformers cannot construct it at all, so the
    # source is marked unrunnable and NO golden is committed (declared debt in
    # test_model_suite_guard). To capture it for real, pass one of: an HF-format
    # bf16 export of the same model (no change needed here - the auto loader
    # handles it), or the vendor stack (mistral-inference + mistral_common),
    # which replaces the template-only capture with a `loader: "mistral"` branch
    # on the qwen4_exp/qwen3_5 pattern and drops this override.
    "mistral-small-4-119b": {
        "unrunnable": "consolidated layout (params.json, no config.json) + "
                      "NVFP4-packed experts",
        # Frame exactly as the engine does (RenderText: BOS + [INST]...[/INST],
        # no vendor system turn). The checkpoint's default system prompt tells
        # the model to deflect every data-shaped request to "use tools", which
        # suppresses all three answer gates regardless of the weights. Same
        # override as tools/{code,det}_capture.py; without it the FSI capture
        # measured 0/3 while the model was answering consistently.
        "chat_template": "{%- for m in messages %}{%- if m.role == 'user' %}{{ bos_token }}[INST] {{ m.content }} [/INST]{%- endif %}{%- endfor %}",
    },
    # qwen3.8-27b is a Qwen3_5 VLM checkpoint; the text language model is
    # loaded directly (loader remaps the model.language_model prefix) and
    # sharded GPU+CPU because 27B bf16 does not fit 32 GiB VRAM.
    "qwen3.8-27b": {"loader": "qwen3_5", "device_map": "auto"},
    # qwen3.5-9b is a Qwen3_5 VLM checkpoint; bf16 fits in 32 GiB so the
    # accelerate device map keeps every layer on the GPU.
    "qwen3.5-9b": {"loader": "qwen3_5", "device_map": "auto"},
    # DeepSeek-V4-Pro-0813 ships no HF chat template; frame with its own
    # encoding/encoding_dsv4.py in chat mode (short direct answers). The source
    # reference is the TP-4 FP8 runner, which cannot execute on one 32 GB card,
    # so the golden is the prompt's Expected Template (template-only mode).
    "deepseek-v4-pro-0813": {
        "framing": "deepseek_v4", "thinking_mode": "chat",
        "unrunnable": "TP-4 FP8 reference (inference/generate.py, MP=4) over an 892.7 GB checkpoint; this 32 GB card cannot execute it",
    },
    # DeepSeek-V4.1-Flash and its 1017 update: no HF chat template in
    # tokenizer_config.json; framed with the release's encoding/encoding.py in
    # the reasoning model's default thinking mode (reproduces the committed
    # v4.1-flash golden exactly). The 552B FP8/FP4 tilelang reference targets
    # 2x80 GB, so the golden is the prompt's Expected Template.
    "deepseek-v4.1-flash": {
        "framing": "deepseek_v41", "thinking_mode": "thinking",
        "unrunnable": "552B FP8/FP4 tilelang tensor-parallel reference (2x80 GB target) over a 500 GB checkpoint; this 32 GB card cannot execute it",
    },
    "deepseek-v4.1-flash-1017": {
        "framing": "deepseek_v41", "thinking_mode": "thinking",
        "unrunnable": "552B FP8/FP4 tilelang tensor-parallel reference (2x80 GB target) over a 500 GB checkpoint; this 32 GB card cannot execute it",
    },
}

# MiniMax-M3 (428B / 796 GB BF16, 60-layer MSA + MoE): the source model does
# not fit this card and no quantized reference is published, so the golden is
# the prompt's Expected Template (template-only mode) framed with the
# checkpoint's own chat template and thinking disabled (the engine's own
# generation prefix). Recorded deviation, not a silent one.
OVERRIDES["minimax-m3"] = {
    "template_kwargs": {"thinking_mode": "disabled"},
    "unrunnable": "428B/796 GB BF16 checkpoint: no 32 GB-card execution path "
                  "(vendor stack is TP-8 over the full BF16 tree)",
}
# MiniMax-M3.1-Flash (428B, pre-quantized MXFP8 dense + NVFP4 experts): the
# source is not runnable on one 32 GB card (no published single-GPU runner and
# the quantization kernel stack is QAT/vendor-side), so the golden is the
# prompt's Expected Template (template-only mode) framed with the checkpoint's
# own chat template. The M3.1 template has NO thinking_mode switch: thinking is
# fixed to adaptive and the generation prompt carries no think-END prefix.
OVERRIDES["minimax-m3.1-flash"] = {
    "template_kwargs": {},
    "unrunnable": "428B pre-quantized (MXFP8 dense + ModelOpt NVFP4 experts) "
                  "checkpoint: no 32 GB-card execution path (vendor stack is "
                  "multi-GPU), so the parity oracle is the prompt template",
}

# kimi-k2.7-code: 581 GB in the compressed-tensors INT4 layout; run on CPU
# (device_map="cpu"), there is no GPU that holds it.
OVERRIDES["kimi-k2.7-code"] = {
    "device_map": "cpu",
    # The checkpoint's config pins flash_attention_2; eager is the same MLA
    # math the layer oracle validates and needs no flash_attn build.
    "attn_implementation": "eager",
}

# GigaChat3.5-432B-A28B-Reasoning (416 GB FP8-block-128 source, ~432B params):
# the reference is the vendor's multi-GPU FP8 stack (`quantization_config.fp8`,
# custom `GigaChat35ForCausalLM`); there is no single-GPU path and the BF16
# body (864 GB) exceeds host RAM, so the golden is the prompt's Expected
# Template (template-only mode) framed with the checkpoint's own
# chat_template.jinja. The template spells `<s>` literally, so the ids are
# encoded with add_special_tokens=False — exactly one BOS, like the engine's
# chatml-v5 framing. Recorded deviation, not a silent one.
OVERRIDES["gigachat3.5-432b-a28b-reasoning"] = {
    "add_special_tokens": False,
    "unrunnable": "432B FP8-block-128 vendor stack (multi-GPU reference); no "
                  "single-GPU path and the BF16 body (864 GB) exceeds host RAM",
}

# GigaChat4-37B-A4B (FP8 block-128 source): the operator chose the
# template-derived golden (2026-11-26) with a separate short source-oracle probe
# recorded in the model report, because executing the FP8 source means a
# host-side dequant to BF16 (74 GB) over the 39 GB checkpoint. The template
# spells `<s>` literally, so the ids are encoded with add_special_tokens=False —
# exactly one BOS, like the engine's chatml-v5 framing.
OVERRIDES["gigachat4-37b-a4b"] = {
    "add_special_tokens": False,
    # Instant mode, which the checkpoint's template selects with `reasoning=False`
    # (it then ends the generation prompt with `<think></think>\n\n`). The
    # reasoning default leaves the think block open, the model narrates a long
    # chain in plain text, and the `<\think>` marker that closes it is a SPECIAL
    # token that `arrow_tokenizer_decode` drops — so the gate's `</think>` cut
    # never fires and the answer extractor latches onto the first capitalised
    # token of the prose (measured 2026-11-26: DET extracted `DET` from the
    # quoted context while the model's own last line was the right answer).
    "template_kwargs": {"reasoning": False},
    "unrunnable": "template-derived by operator decision (2026-11-26): the FP8 "
                  "source needs a host-side dequant to BF16 (74 GB) to execute; "
                  "the short oracle probe is recorded in the model report",
}

def _apply_template_kwargs(template_kwargs, opts):
    """Per-model extra chat-template variables (e.g. MiniMax's thinking_mode)."""
    extra = opts.get("template_kwargs") or {}
    for k, v in extra.items():
        template_kwargs[k] = v
    return template_kwargs

GENERIC = {"trust_remote_code": True, "max_new_tokens": 512,
           "device": None, "dtype": None, "enable_thinking": False,
           "device_map": None, "loader": None, "system": None,
           "add_special_tokens": True,
           # ATEM channel framing (empty -> the raw decode is the answer).
           "answer_channel": None, "reasoning_channel": None,
           "unrunnable": None}

_QWEN4_NGRAM_ROWS = 2500012  # rows per PLE n-gram shard (128 shards)


def _load_qwen4_exp(model_path):
    """Load an experimental qwen4_exp checkpoint with the installed
    transformers (config allowlist + cache mapping patched, PLE shards fused)."""
    import glob
    import re

    import torch
    from safetensors.torch import load_file
    import transformers.configuration_utils as cu
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    from transformers import cache_utils as cu_cache

    cu.ALLOWED_ATTN_LAYER_TYPES = tuple(cu.ALLOWED_ATTN_LAYER_TYPES) + (
        "qwen_sparse_attention",)
    cu_cache.DYNAMIC_LAYER_TYPE_MAPPING.setdefault(
        "qwen_sparse_attention", cu_cache.DynamicIndexedLayer)
    cu_cache.STATIC_LAYER_TYPE_MAPPING.setdefault(
        "qwen_sparse_attention", cu_cache.StaticIndexedLayer)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    raw = json.load(open(os.path.join(model_path, "config.json"),
                        encoding="utf-8"))["text_config"]
    raw["mtp_num_hidden_layers"] = 0
    tc = Qwen4ExpTextConfig(**raw)
    lm = Qwen4ExpForCausalLM(tc)
    lm.eval()
    params = dict(lm.named_parameters())
    bufs = dict(lm.named_buffers())
    ngram = params.get(
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight")
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        for key, tensor in load_file(shard).items():
            if key.startswith("model.visual."):
                continue
            if "ngram_embedding.shard_" in key and ngram is not None:
                idx = int(re.search(r"shard_(\d+)\.weight", key).group(1))
                off = idx * _QWEN4_NGRAM_ROWS
                with torch.no_grad():
                    ngram.data[off:off + tensor.shape[0]].copy_(tensor)
                continue
            name = key.replace("model.language_model.", "model.")
            if name in params:
                with torch.no_grad():
                    params[name].data.copy_(tensor)
            elif name in bufs:
                with torch.no_grad():
                    bufs[name].copy_(tensor)
    torch.set_default_dtype(prev_dtype)
    return lm


def _remap_qwen3_5_key(key):
    """Map a Qwen3.8-27B VLM checkpoint key to a text-only
    Qwen3_5ForCausalLM key. Returns None for vision/MTP tensors the text model
    does not own."""
    if key.startswith("model.visual.") or key.startswith("mtp."):
        return None
    return key.replace("model.language_model.", "model.")


def _load_qwen3_5(model_path):
    """Load the Qwen3.8-27B VLM checkpoint as a text-only causal LM.

    A plain AutoModelForCausalLM load of this checkpoint is wrong: `qwen3_5`
    maps to Qwen3_5ForCausalLM, whose weights are prefixed `model.`, while the
    checkpoint stores them under `model.language_model.`. Without the remap the
    language weights are silently dropped and the model is random.
    """
    import glob

    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    raw = json.load(open(os.path.join(model_path, "config.json"),
                         encoding="utf-8"))["text_config"]
    raw = dict(raw)
    raw["mtp_num_hidden_layers"] = 0
    tc = Qwen3_5TextConfig(**raw)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    lm = Qwen3_5ForCausalLM(tc)
    lm.eval()
    params = dict(lm.named_parameters())
    bufs = dict(lm.named_buffers())
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        for key, tensor in load_file(shard).items():
            name = _remap_qwen3_5_key(key)
            if name is None:
                continue
            if name in params:
                with torch.no_grad():
                    params[name].data.copy_(tensor)
            elif name in bufs:
                with torch.no_grad():
                    bufs[name].copy_(tensor)
    torch.set_default_dtype(prev_dtype)
    return lm


def _dispatch_auto(lm, no_split):
    """Shard a model too large for VRAM across GPU + CPU (accelerate)."""
    from accelerate import dispatch_model, infer_auto_device_map

    dm = infer_auto_device_map(
        lm, max_memory={0: "28GiB", "cpu": "640GiB"},
        no_split_module_classes=no_split)
    return dispatch_model(lm, device_map=dm)


def load_prompts():
    with open(PROMPTS, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["format"] == "logestix-fsi-prompts", doc["format"]
    return doc["prompts"]


def normalize(s):
    out = "".join(ch for ch in s if not ch.isspace())
    return out.rstrip(".,;")


def _unfence(raw):
    s = raw.strip()
    open_at = s.find("```")
    if open_at == -1:
        return s
    begin = open_at + 3
    nl = s.find("\n", begin)
    if nl != -1 and re.fullmatch(r"[A-Za-z0-9_+-]*", s[begin:nl]):
        begin = nl + 1
    close = s.find("```", begin)
    inner = s[begin:] if close == -1 else s[begin:close]
    return inner.strip()


def extract(kind, anchor, raw):
    text = _unfence(raw)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    lines = text.splitlines() or [text]
    line = None
    for cand in lines:
        a = cand.find(anchor)
        if a == -1:
            continue
        i = a + len(anchor)
        while i < len(cand) and cand[i].isspace():
            i += 1
        if i >= len(cand):
            continue
        if kind == "arithmetic" and cand[i] == "=":
            line = cand
            break
        if kind == "comparison" and cand[i] in "<>=!":
            line = cand
            break
    if line is None:
        for cand in lines:
            if anchor in cand:
                line = cand
                break
    if line is None:
        line = text
    if kind == "comparison":
        pat = re.compile(re.escape(anchor) + r"\s*[<>=!]+\s*[A-Za-z_]\w*")
        hit = pat.search(line)
        return normalize(hit.group(0)) if hit else ""
    a = line.find(anchor)
    if a == -1:
        return ""
    eq = line.find("=", a + len(anchor))
    if eq == -1:
        return ""
    i = eq + 1
    depth = 0
    first = True
    prev_op = True
    n = len(line)
    while i < n:
        ch = line[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            depth += 1
            prev_op = True
            first = False
            i += 1
            continue
        if ch == ")":
            if depth == 0:
                break
            depth -= 1
            prev_op = False
            first = False
            i += 1
            continue
        if ch in "+-*/":
            prev_op = True
            first = False
            i += 1
            continue
        if ch.isalnum() or ch in "_.":
            if (not first) and depth == 0 and (not prev_op):
                break
            while i < n and (line[i].isalnum() or line[i] in "_."):
                i += 1
            prev_op = False
            first = False
            continue
        break
    return normalize(line[a:i])


def selftest():
    prompts = load_prompts()
    assert [p["id"] for p in prompts] == ["FSI-TEST-01", "FSI-TEST-02",
                                          "FSI-TEST-03"]
    for p in prompts:
        got = extract(p["kind"], p["anchor"], p["expected"])
        assert got == p["expected"], (p["id"], got)
    # Formula checks: FSI-03 is False at the stated boundary; both arithmetic
    # formulas evaluate to positive finite values with the stated parameters.
    assert (14.999991 > 15.000000) is False
    v1 = eval("CF*((1-exp(-r*n))/(1-exp(-r)))", {"__builtins__": {}},
              {"CF": 2500000, "n": 7, "r": 0.0835, "exp": math.exp})
    v2 = eval("(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C",
              {"__builtins__": {}},
              {"Rate_A": 1.083541, "Slip_A": 0.0005, "Rate_B": 83.152402,
               "Slip_B": 0.0005, "Rate_C": 0.011104, "Slip_C": 0.0005})
    assert math.isfinite(v1) and v1 > 0
    assert math.isfinite(v2) and v2 > 0
    # qwen3.8-27b: the VLM -> text-only key remap (vision/MTP dropped).
    assert _remap_qwen3_5_key("model.language_model.embed_tokens.weight") \
        == "model.embed_tokens.weight", "language prefix remap"
    assert _remap_qwen3_5_key("lm_head.weight") == "lm_head.weight"
    assert _remap_qwen3_5_key("model.visual.blocks.0.weight") is None
    assert _remap_qwen3_5_key("mtp.fc.weight") is None
    # qwen3.5-9b is the same Qwen3_5 VLM checkpoint shape as the 27B.
    assert OVERRIDES.get("qwen3.5-9b", {}).get("loader") == "qwen3_5", \
        "qwen3.5-9b must use the qwen3_5 text-only loader"

    # Muse-Glimmer: the FSI capture frames the source at reasoning_strength=low
    # (the ATEM chat template renders `Reasoning strength: low.` into the system
    # block, which switches the high-effort reasoning loop off) and reads the
    # answer from the FIRST COMPLETE ATEM answer channel — the same rule the C++
    # harness applies (fsi::SelectAtemAnswerChannel).
    sys.path.insert(0, HERE)
    import muse_golden_oracle as mo
    assert mo.TEMPLATE_KWARGS == {"reasoning_strength": "low"}, mo.TEMPLATE_KWARGS
    opts = mo.capture_opts()
    assert opts["template_kwargs"] == mo.TEMPLATE_KWARGS, opts
    assert opts["answer_channel"] == "assistant to=user", opts
    assert opts["reasoning_channel"] == "assistant to=self", opts
    loop_raw = ("assistant to=selfWe need PV=CF*n."
                "assistant to=userPV=CF*n"
                "assistant to=selfDone."
                "assistant to=userPV=CF*(")
    assert mo.select_answer_channel(
        loop_raw, mo.ATEM_ANSWER_CHANNEL,
        mo.ATEM_REASONING_CHANNEL) == "PV=CF*n", "first complete channel"
    lone = "assistant to=selfCompute it.assistant to=userPV=CF*n"
    assert mo.select_answer_channel(
        lone, mo.ATEM_ANSWER_CHANNEL,
        mo.ATEM_REASONING_CHANNEL) == "PV=CF*n", "lone truncated channel"
    assert mo.select_answer_channel(
        "no channels here", mo.ATEM_ANSWER_CHANNEL,
        mo.ATEM_REASONING_CHANNEL) == "no channels here", "unchanged"
    # Incremental save: a later single-prompt run keeps the cases it did not
    # re-capture, in prompt order.
    assert merge_cases([{"id": "FSI-TEST-01", "expected": "a"}],
                       [{"id": "FSI-TEST-03", "expected": "c"}],
                       ["FSI-TEST-01", "FSI-TEST-02", "FSI-TEST-03"]) == [
        {"id": "FSI-TEST-01", "expected": "a"},
        {"id": "FSI-TEST-03", "expected": "c"}], "merge keeps prompt order"
    print("fsi selftest OK: pv=%.6f arb=%.6f" % (v1, v2))


def _frame_deepseek_v4(model_dir, body, thinking_mode):
    """Frame with DeepSeek-V4's release encoder (encoding/encoding_dsv4.py);
    V4-Pro ships no HF chat template."""
    import importlib.util

    enc_path = os.path.join(model_dir, "encoding", "encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("dsv4_encoding", enc_path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    return enc.encode_messages([{"role": "user", "content": body}],
                               thinking_mode=thinking_mode)


def _frame_deepseek_v41(model_dir, body, thinking_mode):
    """Frame with DeepSeek-V4.1's release encoder (encoding/encoding.py).

    The V4.1 trees ship no HF chat template in `tokenizer_config.json`, and the
    1017 update ships a `chat_template.jinja` whose *thinking* mode injects a
    "Reasoning Effort: 75" system line the release encoder does not — so both
    V4.1 checkpoints are framed through the release encoder, which is what
    reproduces the committed deepseek-v4.1-flash golden byte for byte
    (verified 2026-11-22)."""
    import importlib.util

    enc_path = os.path.join(model_dir, "encoding", "encoding.py")
    spec = importlib.util.spec_from_file_location("dsv41_encoding", enc_path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    return enc.encode_messages([{"role": "user", "content": body}],
                               thinking_mode=thinking_mode)


def frame(model, tok, body, opts):
    if opts.get("framing") == "deepseek_v4":
        return _frame_deepseek_v4(model, body,
                                  opts.get("thinking_mode", "chat"))
    if opts.get("framing") == "deepseek_v41":
        return _frame_deepseek_v41(model, body,
                                   opts.get("thinking_mode", "thinking"))
    return None


def _frame_messages(opts, body):
    """Chat messages for a prompt: an optional leading system turn (some
    models are only served with one and refuse otherwise) plus the user turn."""
    messages = []
    if opts.get("system"):
        messages.append({"role": "system", "content": opts["system"]})
    messages.append({"role": "user", "content": body})
    return messages


def _auto_model():
    """Lazy import: --template-only must not need the model classes at all."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def capture_from_template(tok, opts, prompts, model, model_type, out,
                          max_new_tokens, transformers, torch):
    """Template-derived golden: no source model is executed.

    For checkpoints whose SOURCE cannot be run locally -- a proprietary
    architecture, a consolidated layout with no HF config, a vendor-only
    inference stack -- the answer recorded in the golden is the canonical one
    from tools/<set>_prompts.json rather than the source model's own
    generation. This is a deliberate, recorded deviation (see the skill's
    "when the source cannot run locally"), not a silent one: `source` records
    why, and the captured `input_ids` still come from the model's own
    tokenizer and chat template, so the converted path is guarded on the same
    prompt the source would have seen.
    """
    cases = []
    for p in prompts:
        prompt_text = p["body"]
        framed = frame(model, tok, p["body"], opts)
        ids = None
        if framed is not None:
            prompt_text = framed
        elif opts.get("template_tokenize") and getattr(tok, "chat_template", None):
            # Ask the template for IDS. Rendering to text and re-tokenizing is
            # not equivalent for a checkpoint whose frame is carried by SPECIAL
            # tokens: hy4-flash's `<|hy_start:opensource|>` markers are added
            # tokens, and the text route also re-applies `add_special_tokens`,
            # which prepends a second start marker the served frame never has.
            # `tokenize=True` returns the ids the engine is actually fed (pinned
            # equal to Hy4FlashModel::BuildPromptInputIds by
            # tests/test_hy_moe_flash_tool_framing.cpp).
            kwargs = {"tokenize": True, "add_generation_prompt": True,
                      "return_dict": False}
            _apply_template_kwargs(kwargs, opts)
            ids = tok.apply_chat_template(_frame_messages(opts, p["body"]),
                                          **kwargs)
        elif getattr(tok, "chat_template", None):
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if opts.get("enable_thinking") is not None:
                kwargs["enable_thinking"] = opts["enable_thinking"]
            if opts.get("low_effort"):
                kwargs["low_effort"] = True
            _apply_template_kwargs(kwargs, opts)
            try:
                prompt_text = tok.apply_chat_template(
                    _frame_messages(opts, p["body"]), **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                prompt_text = tok.apply_chat_template(
                    _frame_messages(opts, p["body"]), **kwargs)
        if ids is None:
            ids = tok(prompt_text, return_tensors="pt",
                      add_special_tokens=opts.get("add_special_tokens", True)
                      ).input_ids[0].tolist()
        got = extract(p["kind"], p["anchor"], p["expected"])
        if not got:
            raise SystemExit("%s: empty template extraction" % p["id"])
        if got.count("(") != got.count(")"):
            raise SystemExit("%s: unbalanced parentheses %r" % (p["id"], got))
        cases.append({"id": p["id"], "kind": p["kind"], "anchor": p["anchor"],
                      "input_ids": ids, "max_new_tokens": max_new_tokens,
                      "expected": got, "raw": p["expected"]})
    golden = {
        "format": "logestix-fsi-golden", "version": 1,
        "model_type": model_type,
        "source": "prompt-template (source not runnable locally: %s)"
                  % (opts.get("unrunnable") or "custom architecture/layout"),
        "runtime": ("encoding/encoding_dsv4.py + tokenizer.json (transformers %s)"
                    % transformers.__version__)
                   if opts.get("framing") == "deepseek_v4" else
                   ("encoding/encoding.py + tokenizer.json (transformers %s)"
                    % transformers.__version__)
                   if opts.get("framing") == "deepseek_v41" else
                   ("tokenizer.json + chat_template.jinja (transformers %s / torch %s)"
                    % (transformers.__version__, torch.__version__)),
        "prompts": cases}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    print("wrote", out, "(template-derived; source not executed)")


# dory-1.0-4b (NVIDIA-Nemotron-Labs-Dory1.0-4B): `model_type: dory` has no
# transformers config class, so AutoTokenizer cannot even build the tokenizer
# (config validation raises before the tokenizer is touched) and
# AutoModelForCausalLM has no class at all. tools/dory_oracle.py builds a
# PreTrainedTokenizerFast from tokenizer.json and reimplements the nGPT looped
# forward exactly as the vendor reference (vllm-project/vllm#60149) does.
# Thinking OFF for these probes: the model answers the FSI arithmetic and
# CODE JSON/regex probes directly and canonically, while its reasoning trace
# is long and self-repeating (it does not close the think block inside 2048
# tokens on FSI-01/02 and CODE-01/03, which makes the answer extraction fall
# back inside the reasoning text). The DET gate captures in reasoning mode
# instead - its letter-match probe needs it.
# Inkling-2-Mini (looped inkling2_loop, 24 layers x 3 passes over one half):
# the release's own transformers 5.24 implementation runs on the HOST CPU (the
# 53 GB BF16 tree fits the 750 GB box; tools/inkling_loop_oracle.py is the
# loader). Its decoded stream does not reach EOS inside a 32-token window, so
# ignore_eos keeps the answer line inside the captured budget.
OVERRIDES["inkling-2-mini"] = {"loader": "inkling_loop", "ignore_eos": True,
                               "threads": 64}
# TML-d24-varloop-research (the same inkling2_loop tower with is_varloop):
# same loader, same 53 GB BF16 host tree, same fixed budget. The engine runs it
# at the checkpoint's declared 3 passes, which is what the capture uses.
OVERRIDES["tml-d24-varloop-research"] = {"loader": "inkling_loop",
                                         "ignore_eos": True, "threads": 64}
# Hy4-flash: the SOURCE is runnable (tools/hy4_flash_reference.py drives
# transformers PR #50903 and was used to validate the architecture layer by
# layer), but a full FSI generation costs hours: that harness feeds the whole
# prefix per step, so 512 new tokens over a ~700-token prompt is O(n^2) at
# ~6 min/step on this host. The golden is therefore the canonical answer from
# tools/fsi_prompts.json (the same `expected` every other family records) with
# the ids taken from the checkpoint's own template via `tokenize=True`, and
# `reasoning_effort=no_think` so the frame is the one the engine serves by
# default. Recorded as a deviation, not passed off as a source run.
OVERRIDES["hy4-flash"] = {
    "template_tokenize": True,
    "template_kwargs": {"reasoning_effort": "no_think"},
    "unrunnable": "canonical answers recorded from tools/fsi_prompts.json; the "
                  "source runs (tools/hy4_flash_reference.py) but a 512-token "
                  "generation is O(n^2) full-prefix decoding (~6 min/step here)",
}
OVERRIDES["dory-1.0-4b"] = {"loader": "dory", "tokenizer": "fast",
                            "enable_thinking": False}

def _load_dory_tokenizer(model):
    """PreTrainedTokenizerFast from tokenizer.json (no AutoTokenizer: the
    checkpoint's config has no transformers class)."""
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(model, "tokenizer.json"),
        bos_token="<s>", eos_token="<|im_end|>", pad_token="<|im_end|>",
        unk_token="<unk>")
    jinja = os.path.join(model, "chat_template.jinja")
    if os.path.exists(jinja):
        with open(jinja, encoding="utf-8") as fh:
            tok.chat_template = fh.read()
    return tok

def capture(model, model_type, out, max_new_tokens, device_map=None,
            template_only=False, only=None, resume=False):
    import torch
    import transformers
    from transformers import AutoTokenizer

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    # capture_case dumps the raw stream next to the golden for diagnosis.
    opts["_out"] = out
    if device_map:
        opts["device_map"] = device_map
    prompts = load_prompts()

    if opts.get("tokenizer") == "fast":
        tok = _load_dory_tokenizer(model)
    else:
        tok = AutoTokenizer.from_pretrained(
            model, trust_remote_code=opts["trust_remote_code"])
    if not getattr(tok, "chat_template", None):
        jinja = os.path.join(model, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as fh:
                tok.chat_template = fh.read()
    # Per-model framing override (engine-exact where the checkpoint's own
    # template injects something the engine does not, e.g. a vendor system
    # turn). tools/{code,det}_capture.py already apply this; without it here the
    # override was silently ignored and the capture used the vendor template.
    if opts.get("chat_template"):
        tok.chat_template = opts["chat_template"]
    if template_only:
        return capture_from_template(tok, opts, prompts, model, model_type,
                                     out, max_new_tokens, transformers, torch)
    if template_only:
        lm = None
    elif opts.get("loader") == "hy_v3":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hy_v3_oracle
        lm = hy_v3_oracle.load(model, opts.get("device_map"))
        print("device:", next(lm.parameters()).device)
    elif opts.get("loader") == "qwen3_5":
        lm = _load_qwen3_5(model)
        if opts.get("device_map"):
            lm = _dispatch_auto(lm, ["Qwen3_5DecoderLayer"])
        print("device:", next(lm.parameters()).device)
    elif opts.get("loader") == "granite_switch":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import granite_switch_oracle
        tok, lm = granite_switch_oracle.load(model, opts.get("adapter_name"))
        print("device:", next(lm.parameters()).device)
    elif opts.get("loader") == "dory":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dory_oracle
        lm = dory_oracle.load(model, opts.get("device_map"))
        print("device:", next(lm.parameters()).device)
    elif opts.get("loader") == "muse_text":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import muse_golden_oracle
        lm = muse_golden_oracle.load(model, opts.get("device_map"))
        # The loader owns its checkpoint's capture facts: the chat-template
        # `reasoning_strength` switch and the ATEM channel headers the answer
        # must be read from (see the oracle's module docstring).
        opts.update(muse_golden_oracle.capture_opts())
        print("device:", next(lm.parameters()).device,
              "template_kwargs:", opts.get("template_kwargs"),
              file=sys.stderr)
    elif opts.get("loader") == "inkling_loop":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import inkling_loop_oracle
        tok, lm = inkling_loop_oracle.load(model, opts.get("dtype"),
                                           opts.get("threads", 64))
        # The loader owns its checkpoint's capture facts: this stream does not
        # reach EOS inside the answer window, so the budget is fixed.
        opts.update(inkling_loop_oracle.capture_opts())
        print("device:", lm.device, "threads:", opts.get("threads", 64),
              file=sys.stderr)
    elif opts.get("loader") == "qwen4_exp":
        lm = _load_qwen4_exp(model)
        print("device:", next(lm.parameters()).device)
    else:
        load_kwargs = {"trust_remote_code": opts["trust_remote_code"]}
        if opts.get("dtype"):
            load_kwargs["dtype"] = opts["dtype"]
        if opts.get("device_map"):
            load_kwargs["device_map"] = opts["device_map"]
        if opts.get("attn_implementation"):
            # The checkpoint JSON pins _attn_implementation on the wrapper AND
            # on the nested text/vision configs; the kwarg alone is not enough
            # (transformers validates against the config's own value).
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(
                model, trust_remote_code=opts["trust_remote_code"])
            impl = opts["attn_implementation"]
            cfg._attn_implementation = impl
            for sub in ("text_config", "vision_config"):
                sub_cfg = getattr(cfg, sub, None)
                if sub_cfg is not None:
                    sub_cfg._attn_implementation = impl
            load_kwargs["config"] = cfg
            load_kwargs["attn_implementation"] = impl
        lm = _auto_model().from_pretrained(model, **load_kwargs)
        lm.eval()

        device = opts.get("device")
        if device is None and not opts.get("device_map") and torch.cuda.is_available():
            nbytes = sum(p.numel() * p.element_size() for p in lm.parameters())
            free, _total = torch.cuda.mem_get_info()
            if nbytes < int(free * 0.9):
                device = "cuda"
        if device:
            lm = lm.to(device)
        print("device:", next(lm.parameters()).device)

    # ---- one prompt at a time, saved incrementally -------------------------
    # An FSI capture on a CPU oracle can run for tens of minutes per prompt (the
    # Muse-Glimmer text tower at 1024 tokens took >1.5 h for 3 cases and was
    # killed with nothing written). Each completed case is therefore flushed to
    # `--out` immediately, `--only` captures a single prompt, and `--resume`
    # merges into the cases an earlier run already wrote.
    order = [p["id"] for p in prompts]
    cases = load_cases(out) if resume else []
    if cases:
        print("[fsi-capture] resume: %d case(s) already in %s"
              % (len(cases), out), file=sys.stderr, flush=True)
    if only:
        unknown = sorted(set(only) - set(order))
        if unknown:
            raise SystemExit("--only: unknown prompt id(s): %s"
                             % ", ".join(unknown))
    selected = [p for p in prompts if not only or p["id"] in only]
    failures = 0
    for idx, p in enumerate(selected, 1):
        print("[fsi-capture] %s start (%d/%d) max_new_tokens=%d"
              % (p["id"], idx, len(selected), max_new_tokens),
              file=sys.stderr, flush=True)
        case, err = capture_case(tok, lm, opts, p, max_new_tokens, torch)
        if case is None:
            failures += 1
            raw_path = "%s.%s.raw.txt" % (out, p["id"])
            print("[fsi-capture] %s FAILED: %s (raw in %s)"
                  % (p["id"], err, raw_path), file=sys.stderr, flush=True)
            continue
        cases = merge_cases(cases, [case], order)
        write_golden(out, model_type, model, cases, transformers, torch)
        print("[fsi-capture] %s captured: expected=%r (%d case(s) in %s)"
              % (p["id"], case["expected"], len(cases), out),
              file=sys.stderr, flush=True)
    if failures:
        raise SystemExit("[fsi-capture] %d of %d prompt(s) failed; %s holds %d "
                         "case(s) -- keep the partial file, re-run the failed "
                         "ids with --only ... --resume"
                         % (failures, len(selected), out, len(cases)))
    print("wrote", out, "(%d case(s))" % len(cases))


def merge_cases(prev, new, order):
    """Merge captured cases by id and order them by the prompt set.

    The incremental save must never reorder or drop the cases an earlier run
    wrote, so the on-disk order is the prompt-set order, not completion order.
    """
    by_id = {c["id"]: c for c in prev}
    for c in new:
        by_id[c["id"]] = c
    rank = {pid: i for i, pid in enumerate(order)}
    return sorted(by_id.values(),
                  key=lambda c: (rank.get(c["id"], len(rank)), c["id"]))


def load_cases(path):
    """Cases already captured into `path` (for --resume). Missing -> []."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit("--resume: cannot read %s: %s" % (path, exc))
    if doc.get("format") != "logestix-fsi-golden":
        raise SystemExit("--resume: %s is not a logestix-fsi-golden" % path)
    return list(doc.get("prompts") or [])


def write_golden(out, model_type, source, cases, transformers, torch):
    """Write the golden atomically: a killed run leaves the previous file."""
    golden = {"format": "logestix-fsi-golden", "version": 1,
              "model_type": model_type, "source": source,
              "runtime": "transformers %s / torch %s"
                         % (transformers.__version__, torch.__version__),
              "prompts": cases}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, out)


def _frame_prompt(tok, opts, body):
    """The prompt text the source sees: the checkpoint's own chat template,
    with the per-model template kwargs (e.g. Muse's `reasoning_strength`)."""
    prompt_text = body
    if not getattr(tok, "chat_template", None):
        return prompt_text
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if opts.get("enable_thinking") is not None:
        template_kwargs["enable_thinking"] = opts["enable_thinking"]
    if opts.get("low_effort"):
        template_kwargs["low_effort"] = True
    _apply_template_kwargs(template_kwargs, opts)
    messages = _frame_messages(opts, body)
    try:
        prompt_text = tok.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        prompt_text = tok.apply_chat_template(messages, **template_kwargs)
    except Exception as exc:  # noqa: BLE001
        print("chat_template failed, using raw prompt:", exc, file=sys.stderr)
        prompt_text = body
    return prompt_text


def _select_channel(raw, opts):
    """ATEM: read the FIRST COMPLETE answer channel, not the raw stream.

    The rule is the loader's (tools/muse_golden_oracle.py::select_answer_channel,
    mirrored by the C++ harness in
    libs/lsxcommon/include/lsxcommon/atem_channel.h), so the golden's `expected`
    is exactly the answer the verifier extracts from the same text. Models
    without a channel override get the raw decode unchanged.
    """
    if not opts.get("answer_channel"):
        return raw
    sys.path.insert(0, HERE)
    import muse_golden_oracle
    text = muse_golden_oracle.select_answer_channel(
        raw, opts["answer_channel"], opts.get("reasoning_channel") or "")
    # Progress lines must stay readable: the reasoning loop can be megabytes.
    print("[fsi-capture] channel selected: %d of %d chars"
          % (len(text), len(raw)), file=sys.stderr, flush=True)
    return text


def _dump_raw(opts, prompt_id, raw):
    """A failed case writes its full raw next to the golden — for a reasoning
    loop that text (megabytes) is the only evidence of what went wrong."""
    path = "%s.%s.raw.txt" % (opts["_out"], prompt_id)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(raw)
    except OSError as exc:  # diagnostic only; never mask the real failure
        print("could not write %s: %s" % (path, exc), file=sys.stderr)


def capture_case(tok, lm, opts, p, max_new_tokens, torch):
    """Run one prompt and return (case, None) or (None, error).

    A failure dumps the full raw next to the golden (`<out>.<id>.raw.txt`) —
    for a reasoning loop that text is the only evidence of what went wrong.
    """
    prompt_text = _frame_prompt(tok, opts, p["body"])
    ids = tok(prompt_text, return_tensors="pt",
              add_special_tokens=opts.get("add_special_tokens", True)).input_ids
    if lm is not None:
        ids = ids.to(lm.device)
    with torch.no_grad():
        gen_kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}
        if opts.get("ignore_eos"):
            gen_kwargs["eos_token_id"] = None
        out_ids = lm.generate(ids, **gen_kwargs)
    gen = out_ids[0][ids.shape[1]:].tolist()
    raw = tok.decode(gen, skip_special_tokens=True)
    channel = _select_channel(raw, opts)
    got = extract(p["kind"], p["anchor"], channel)
    if not got:
        _dump_raw(opts, p["id"], raw)
        return None, ("empty extraction (gen=%d tokens, channel=%r)"
                      % (len(gen), channel[:300]))
    if got.count("(") != got.count(")"):
        _dump_raw(opts, p["id"], raw)
        return None, ("unbalanced parentheses in extracted answer %r "
                      "(channel=%r)" % (got, channel[:300]))
    if got != p["expected"]:
        print("WARNING %s: model answer differs from the prompt template: "
              "%r != %r" % (p["id"], got, p["expected"]), file=sys.stderr)
    case = {"id": p["id"], "kind": p["kind"], "anchor": p["anchor"],
            "input_ids": ids[0].tolist(),
            "max_new_tokens": max_new_tokens,
            "ignore_eos": bool(opts.get("ignore_eos"))}
    if opts.get("answer_channel"):
        case["answer_channel"] = opts["answer_channel"]
        case["reasoning_channel"] = opts.get("reasoning_channel") or ""
    case["expected"] = got
    case["raw"] = raw
    print("[fsi-capture] %s gen=%d tokens, raw=%d chars"
          % (p["id"], len(gen), len(raw)), file=sys.stderr, flush=True)
    return case, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--type")
    ap.add_argument("--out")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--template-only", action="store_true",
                    help="derive the golden from the prompt template without "
                         "executing the source model (records the deviation in "
                         "`source`); for checkpoints transformers cannot load")
    ap.add_argument("--device-map", default=None,
                    help="transformers device_map (requires accelerate); use "
                         "'auto' for FP8/large models on a big-VRAM host")
    ap.add_argument("--only", default=None,
                    help="capture only these prompt ids (comma-separated), e.g. "
                         "FSI-TEST-01: one prompt per run for a slow oracle")
    ap.add_argument("--resume", action="store_true",
                    help="keep the cases already captured in --out and merge "
                         "this run's cases into them (use with --only)")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model and args.type and args.out):
        ap.error("--model, --type and --out are required unless --selftest")
    only = None
    if args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
    capture(args.model, args.type, args.out, args.max_new_tokens,
            args.device_map, args.template_only, only, args.resume)


if __name__ == "__main__":
    main()
