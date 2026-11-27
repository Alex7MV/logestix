#!/usr/bin/env python3
"""Run the AUTHORITATIVE Hy4-flash reference (transformers PR #50903) on the
checkpoint, with the same cut set as the converted artifact.

This is the reference of record: the checkpoint's README pins PR #50903 as the
implementation ("Support for Hy4 flash will be merged into the transformers main
branch soon. Until then, install from the commit below"), so goldens are captured
here and nowhere else.

The README pins the implementation: "Support for Hy4 flash will be merged into the
transformers main branch soon. Until then, install from the commit below:
  pip install transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/50903/head"

This script loads ONLY the text backbone, and skips exactly the tensors the
artifact cuts (Engram, the ViT tower, the DFly2 drafter), so its logits are the
reference the engine must reproduce. It runs on CPU in bf16 with a streamed
state_dict, because the full tree is 411 GB and no single GPU here can hold it.

    /mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_reference.py \
        --model /mnt/fast_storage/lsx_model/tencent/Hy4-flash --prompt "The capital of France is"
"""
import argparse
import json
import os
import struct
import sys

import torch


# The checkpoint's leaf names are NOT the reference model's module names. The
# mapping is the PR's own `conversion_mapping.py` hy_v4_flash list, copied
# verbatim: a reference run with unmapped names silently leaves 422 parameters at
# their random initialisation (that is what the first attempt did).
RENAMES = [
    (".self_attn.learnable_sink_param", ".self_attn.sinks"),
    (".self_attn.linear_gate.weight", ".self_attn.gate_proj.weight"),
    (".mlp.post_fc2_norm.weight", ".post_mlp_layernorm.weight"),
    (".mlp.routed_expert_down_proj.weight", ".mlp.pre_expert_proj.weight"),
    (".mlp.routed_expert_up_proj.weight", ".mlp.post_expert_proj.weight"),
    (".hc_attn_layer.hc_pre.hc_fn", ".attn_hc.fn"),
    (".hc_attn_layer.hc_pre.hc_base", ".attn_hc.base"),
    (".hc_attn_layer.hc_pre.hc_scale", ".attn_hc.scale"),
    (".hc_mlp_layer.hc_pre.hc_fn", ".ffn_hc.fn"),
    (".hc_mlp_layer.hc_pre.hc_base", ".ffn_hc.base"),
    (".hc_mlp_layer.hc_pre.hc_scale", ".ffn_hc.scale"),
    (".hc_head.hc_head_fn", ".hc_head.hc_fn"),
    (".hc_head.hc_head_base", ".hc_head.hc_base"),
    (".hc_head.hc_head_scale", ".hc_head.hc_scale"),
    ("hc_head.hc_head_fn", "hc_head.hc_fn"),
    ("hc_head.hc_head_base", "hc_head.hc_base"),
    ("hc_head.hc_head_scale", "hc_head.hc_scale"),
]


def rename(name):
    for src, dst in RENAMES:
        name = name.replace(src, dst)
    return name


ENGRAM = "engram_layer."


def iter_tensors(root, skip_engram, skip_vision, skip_dfly, keep=None):
    """Yields (name, tensor) reading one shard at a time.

    `keep` filters by NAME BEFORE the bytes are read, which is what makes
    `--layers 1` cheap: a one-layer probe copies ~1 GB out of the 440 GB tree
    instead of streaming all of it.
    """
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    by_file = {}
    for name, fn in wmap.items():
        if skip_engram and ENGRAM in name:
            continue
        if skip_vision and ("vit." in name):
            continue
        if skip_dfly and ("dfly_layers." in name):
            continue
        by_file.setdefault(fn, []).append(name)
    for fn, names in by_file.items():
        path = os.path.join(root, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
            for name in names:
                if keep is not None and not keep(name):
                    continue
                e = hdr[name]
                s, eo = e["data_offsets"]
                dt = {"BF16": torch.bfloat16, "F16": torch.float16,
                      "F32": torch.float32, "I32": torch.int32}[e["dtype"]]
                fh.seek(base + s)
                raw = fh.read(eo - s)
                t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(e["shape"])
                yield name, t


# NVFP4 codec, as the converter stores it: nibble pairs of E2M1 in `packed`
# (low nibble first), one E4M3 block scale per 16 elements in `scales`, and one
# per-tensor `global_scale`. Decoding it here is what lets the reference answer
# "what does the engine's own artifact compute?" — the only way to tell a
# modelling bug from the artifact's precision floor.
def _e2m1():
    import numpy as np
    return np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                     -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def _e4m3():
    import numpy as np
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b >> 7 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0:
            lut[b] = sign * (m / 8.0) * 2.0 ** -6
        elif e == 15:
            lut[b] = np.nan
        else:
            lut[b] = sign * (1 + m / 8.0) * 2.0 ** (e - 7)
    return lut


def artifact_dequant(row):
    """One DenseNvfp4 record -> the [rows, cols] matrix the engine multiplies by."""
    import numpy as np
    rows, cols = row["shape"]
    packed = np.frombuffer(row["packed"], dtype=np.uint8).reshape(rows, cols // 2)
    v = np.empty((rows, cols), dtype=np.float32)
    fp4 = _e2m1()
    v[:, 0::2] = fp4[packed & 0xF]
    v[:, 1::2] = fp4[packed >> 4]
    s = _e4m3()[np.frombuffer(bytes(row["scales"]), dtype=np.uint8)]
    return (v * s.reshape(rows, cols // 16).repeat(16, axis=1)) * row["global_scale"]


def override_from_artifact(model, root, layers, blocks=0):
    """Replace the loaded bf16 weights with the converted artifact's NVFP4 ones.

    Runs the SAME reference code on the SAME numbers the engine has, so a
    remaining disagreement is a modelling bug and an agreement means the gap was
    the 4-bit weights (a precision decision, not a defect).
    """
    import glob
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc

    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    replaced = 0
    for path in sorted(glob.glob(os.path.join(root, "logestix", "pinned",
                                              "layer_*_nvfp4.arrow"))):
        layer = int(path.rsplit("layer_", 1)[1].split("_")[0])
        if layers and layer >= layers:
            continue
        with pa.memory_map(path, "r") as src:
            table = ipc.open_file(src).read_all()
        records = {}
        for i, ckpt_name in enumerate(table.column("name").to_pylist()):
            records[ckpt_name] = {c: table.column(c)[i].as_py()
                                  for c in table.schema.names}
        # `kv_b_proj` is stored SPLIT into per-head [nope | value] blocks; the
        # reference keeps one [heads*(nope+v), kv_lora] matrix, so re-fuse it
        # and skip its nope-only record below.
        handled = set()
        for ckpt_name, row in list(records.items()):
            if not ckpt_name.endswith(".value"):
                continue
            nope_row = records[ckpt_name[:-len(".value")]]
            nope, value = artifact_dequant(nope_row), artifact_dequant(row)
            heads = nope.shape[0] // 192
            assert nope.shape[0] == heads * 192
            fused = np.concatenate([nope.reshape(heads, 192, -1),
                                    value.reshape(heads, value.shape[0] // heads, -1)],
                                   axis=1).reshape(-1, nope.shape[1])
            handled.add(ckpt_name[:-len(".value")])
            key = rename(ckpt_name[:-len(".value")].removeprefix("model."))
            if key in params:
                with torch.no_grad():
                    params[key].copy_(torch.from_numpy(fused))
                replaced += 1
        for ckpt_name, row in records.items():
            if ckpt_name.endswith(".value") or ckpt_name in handled:
                continue
            key = rename(ckpt_name.removeprefix("model."))
            if key not in params:
                continue
            with torch.no_grad():
                params[key].copy_(torch.from_numpy(artifact_dequant(row)))
            replaced += 1
    print("artifact override: %d tensors replaced from %s" %
          (replaced, os.path.join(root, "logestix")), flush=True)
    for layer, blk in enumerate(model.layers):
        if blocks and layer >= blocks:
            continue
        if getattr(blk.mlp, "experts", None) is not None:
            override_experts_from_artifact(model, root, layer)


def override_experts_from_artifact(model, root, layer):
    """Gather a converted layer's 512 routed experts into the reference's tensors.

    The artifact stores one record per expert (`experts/layer_<l>/exp_<e>.arrow`)
    with NVFP4 w1/w2/w3; the reference holds them packed as `gate_up_proj`
    [E, 2I, L] (rows [gate; up]) and `down_proj` [E, L, I].
    """
    import glob
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc

    paths = sorted(glob.glob(os.path.join(root, "logestix", "experts",
                                          "layer_%d" % layer, "exp_*.arrow")),
                   key=lambda p: int(p.rsplit("exp_", 1)[1].split(".")[0]))
    experts = model.layers[layer].mlp.experts
    gate_up = np.empty(experts.gate_up_proj.shape, dtype=np.float32)
    down = np.empty(experts.down_proj.shape, dtype=np.float32)
    for e, path in enumerate(paths):
        with pa.memory_map(path, "r") as src:
            row = ipc.open_file(src).read_all().to_pylist()[0]

        def mat(w, scale, gs):
            packed = np.frombuffer(w, dtype=np.uint8)
            side = int(round((2 * packed.size) ** 0.5))
            return artifact_dequant({"packed": w, "scales": scale,
                                     "shape": [side, side],
                                     "global_scale": gs})

        gate_up[e, : experts.gate_up_proj.shape[1] // 2] = mat(
            row["w1_weight"], row["w1_scale"], row["w1_global_scale"])
        gate_up[e, experts.gate_up_proj.shape[1] // 2:] = mat(
            row["w3_weight"], row["w3_scale"], row["w3_global_scale"])
        down[e] = mat(row["w2_weight"], row["w2_scale"], row["w2_global_scale"])
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.from_numpy(gate_up))
        experts.down_proj.copy_(torch.from_numpy(down))
    print("artifact override: layer %d, %d experts" % (layer, len(paths)),
          flush=True)


def dump_layers(model, ids, head, out_path):
    """One teacher-forced forward, dumping every boundary the engine also has.

    Hooks rather than `output_hidden_states`: the backbone threads a
    `[batch, seq, hc_mult, hidden]` 4-stream residual through the stack, so the
    per-layer "hidden state" is that tensor, not a `[batch, seq, hidden]` one.
    Everything is written at the LAST position in fp32, which is the token whose
    logits a decode step turns into the answer.

    Sub-layer hooks exist so a disagreement can be attributed to ONE component:
    `l0_attn_in` vs `l0_q_resid` vs `l0_kv_pass` separates the block input from
    the q/kv projections, and `l0_attn_out` vs `l0_mlp_out` separates attention
    from the FFN. Hooking costs nothing next to the forward itself.
    """
    import numpy as np

    grabbed = {}
    handles = []

    def take(name, tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if tensor.dim() >= 3:
            # [batch, seq, ...] -> the last position, which is the token whose
            # logits this forward produces.
            tensor = tensor[0, -1]
        grabbed[name] = tensor.detach().float().cpu().numpy().copy()

    def hook_at(module, name):
        handles.append(module.register_forward_hook(
            lambda _m, _i, out: take(name, out[0] if isinstance(out, (tuple, list)) else out)))

    for i, blk in enumerate(model.layers):
        hook_at(blk, "l%d_out" % i)          # [4, hidden] streams
        hook_at(blk.input_layernorm, "l%d_attn_in" % i)
        hook_at(blk.self_attn.q_a_layernorm, "l%d_q_resid" % i)
        hook_at(blk.self_attn.kv_a_layernorm, "l%d_kv_pass" % i)
        hook_at(blk.self_attn.gate_proj, "l%d_gate" % i)
        hook_at(blk.self_attn.o_proj, "l%d_o_proj" % i)
        # attn -> (post_proj_norm(o_proj(gate * value)), attention weights). The
        # weights are the sharpest boundary: they test the scores, the sink and
        # the mask with neither the value path nor any output projection in the
        # way, and they are per head, which localises a wrong head immediately.
        def _attn(_m, _i, o, name="l%d_attn_out" % i, w="l%d_probs" % i):
            take(name, o[0])
            if o[1] is not None:
                take(w, o[1][0][:, -1, :])
        handles.append(blk.self_attn.register_forward_hook(_attn))
        hook_at(blk.ffn_hc, "l%d_ffn_hc" % i)
        hook_at(blk.post_attention_layernorm, "l%d_ffn_in" % i)
        hook_at(blk.mlp, "l%d_mlp_raw" % i)
        hook_at(blk.post_mlp_layernorm, "l%d_mlp_out" % i)
        if hasattr(blk.mlp, "gate_proj"):               # dense layers only
            hook_at(blk.mlp.gate_proj, "l%d_d_gate" % i)
            hook_at(blk.mlp.up_proj, "l%d_d_up" % i)
        # The HC modules return `(post_gains[4], collapsed_input[hidden])`; both
        # are boundaries the engine has, and the gains make the post half of the
        # residual auditable on its own.
        for mod, tag in ((blk.attn_hc, "attn"), (blk.ffn_hc, "ffn")):
            def _hc(_m, _i, o, name="l%d_%s_hc" % (i, tag),
                    gains="l%d_%s_post" % (i, tag)):
                # Must return None: a forward hook that returns a value REPLACES
                # the module output, and this module's output is a pair.
                take(gains, o[0])
                take(name, o[1])
            handles.append(mod.register_forward_hook(_hc))
    handles.append(model.hc_head.register_forward_hook(
        lambda _m, _i, out: take("hc_head", out)))
    handles.append(model.norm.register_forward_hook(
        lambda _m, _i, out: take("norm", out)))

    with torch.no_grad():
        take("embed", model.embed_tokens(torch.tensor([ids])))
        out = model(input_ids=torch.tensor([ids]), use_cache=False,
                    return_dict=True)
        logits = out.last_hidden_state[0, -1].float() @ head.float().T
        take("logits", logits)
    for h in handles:
        h.remove()
    grabbed["ids"] = np.asarray(ids, dtype=np.int64)
    np.savez(out_path, **grabbed)
    top = torch.topk(logits, 5)
    print("dump: %d layers -> %s | argmax %d top5=%s" % (
        len(model.layers), out_path, int(torch.argmax(logits)),
        [(int(i), round(float(v), 4)) for i, v in zip(top.indices, top.values)]),
        flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/fast_storage/lsx_model/tencent/Hy4-flash")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--ids", default="")
    ap.add_argument("--engram", action="store_true",
                    help="keep the Engram tables (needs ~130 GB more RAM)")
    ap.add_argument("--out", default="/mnt/fast_storage/tmp/hy4-ref-step.json")
    ap.add_argument("--artifact", default="",
                    help="replace the bf16 weights with a converted tree's "
                         "NVFP4 weights (the engine's own numbers)")
    ap.add_argument("--layers", type=int, default=0,
                    help="instantiate only the first N layers (a component "
                         "probe: the stack is truncated, so its LOGITS ARE NOT "
                         "the model's — never capture a golden this way)")
    ap.add_argument("--dump", default="",
                    help="npz path: one teacher-forced forward, every layer "
                         "boundary written at the last position, then exit")
    ap.add_argument("--frame", action="store_true",
                    help="wrap the prompt in the checkpoint's chat template")
    args = ap.parse_args()

    from transformers import AutoTokenizer, HYV4FlashTextConfig
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as R

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(os.path.join(args.model, "config.json")) as f:
        full = json.load(f)
    text_cfg = dict(full["text_config"])
    if not args.engram:
        # Same cut set as the converted artifact.
        text_cfg["engram_layer_ids"] = None
        text_cfg.pop("engram_layer_head_vocab_sizes", None)
    if args.layers:
        # Truncation has to reach the per-layer lists as well: the config indexes
        # them by layer id and a 42-long list beside a 1-layer stack mismatches.
        text_cfg["num_hidden_layers"] = args.layers
        for key in ("layer_types", "mlp_layer_types", "indexer_types",
                    "swa_pattern"):
            if isinstance(text_cfg.get(key), list):
                text_cfg[key] = text_cfg[key][: args.layers]
    cfg = HYV4FlashTextConfig(**text_cfg)
    cfg._attn_implementation = "eager"

    # bf16 params: a float32 instantiation is 310 GB and this box refuses the
    # commit (CommitLimit is 402 GB), which killed the first two attempts.
    torch.set_default_dtype(torch.bfloat16)
    model = R.HYV4FlashTextModel._from_config(cfg)
    model.eval()

    # Copy the backbone in TENSOR BY TENSOR: one 155 GB model plus one source
    # tensor is the only footprint the commit limit allows, so materializing a
    # whole state_dict (even with assign=True) is not an option.
    # BOTH parameters and buffers. The router's `e_score_correction_bias` is an
    # `nn.Buffer` of 512 fp32 values (|max| up to 0.075 on the real checkpoint)
    # that moves which 16 of 512 experts are selected; `named_parameters()`
    # misses it, so a parameters-only load leaves every MoE layer routing with a
    # zero bias while the engine uses the real one. That is what the first
    # version of this harness did, and it made the reference look 98% wrong at
    # layer 1's MoE output against an engine that was actually correct.
    # Live objects (parameters AND persistent buffers) keyed by name. The set of
    # names is `state_dict()`'s, which is exactly "what a checkpoint provides":
    # `_non_persistent_buffers_set` lives on the OWNING submodule, so testing it
    # on the top-level model wrongly admits the indexer's derived rotary tables.
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))
    persisted = set(model.state_dict().keys())
    tensors = {k: v for k, v in named.items() if k in persisted}
    loaded = set()

    def wanted(name):
        key = name[len("model."):] if name.startswith("model.") else name
        return rename(key) in tensors

    for name, t in iter_tensors(args.model, skip_engram=not args.engram,
                                skip_vision=True, skip_dfly=True,
                                keep=wanted):
        key = name[len("model."):] if name.startswith("model.") else name
        key = rename(key)
        p = tensors.get(key)
        if p is None:
            continue
        with torch.no_grad():
            p.copy_(t)
        loaded.add(key)
        del t
    # A missing BUFFER is silent in a way a missing parameter is not (its
    # initialisation can be all zeros), so name both counts separately.
    missing = [n for n in tensors if n not in loaded]
    missing_params = [n for n in missing if n in dict(model.named_parameters())]
    missing_buffers = [n for n in missing if n not in missing_params]
    print("loaded:", len(loaded), "missing:", len(missing),
          "params:", len(missing_params), missing_params[:4],
          "buffers:", len(missing_buffers), missing_buffers[:4], flush=True)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    elif args.frame:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=False,
            reasoning_effort="no_think")
    else:
        ids = tok.encode(args.prompt, add_special_tokens=False)
    print("prompt ids:", ids, flush=True)

    # lm_head is not part of the text model: it is a wrapper-level nn.Linear with
    # the checkpoint's [vocab, hidden] table.
    head = None
    for name, t in iter_tensors(args.model, skip_engram=True, skip_vision=True,
                                skip_dfly=True):
        if name == "lm_head.weight":
            head = t.bfloat16()
            break
    assert head is not None, "lm_head.weight not found"
    print("lm_head dtype", head.dtype, tuple(head.shape), flush=True)

    if args.artifact:
        override_from_artifact(model, args.artifact, args.layers,
                                blocks=args.layers)

    if args.dump:
        dump_layers(model, ids, head, args.dump)
        return

    # Decode by feeding the WHOLE prefix every step (`use_cache=False`): the
    # dynamic cache path threads per-layer-type M-RoPE position planes through
    # `DynamicCache`, and a hand-rolled one would be a second implementation to
    # trust. O(n^2) is affordable at golden-capture lengths.
    nxt = None
    hist = list(ids)
    with torch.no_grad():
        for step in range(len(ids) - 1 + args.tokens):
            if step >= len(ids):
                hist.append(nxt)
            prefix = hist[:step + 1]
            out = model(input_ids=torch.tensor([prefix]), use_cache=False,
                        return_dict=True)
            h = out.last_hidden_state[0, -1]
            logits = h.float() @ head.float().T
            if step + 1 >= len(ids):
                nxt = int(torch.argmax(logits))
                top = torch.topk(logits, 8)
                print("gen %d: seq=%d -> %d %r top8=%s" %
                      (step - len(ids) + 1, len(prefix), nxt,
                       tok.decode([nxt]),
                       [(int(i), round(float(v), 4))
                        for i, v in zip(top.indices, top.values)]), flush=True)
                json.dump({"ids": prefix, "next": nxt,
                           "top8": [(int(i), float(v)) for i, v in
                                    zip(top.indices, top.values)]},
                          open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
