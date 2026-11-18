# Inkling (base) ingestion — design

Date: 2026-11-17. Approved by operator ("ок").

## Intent

Ingest the `Inkling` base checkpoint (`/mnt/fast_storage/huggingface/Inkling`,
architecture `inkling_mm_model`, text tower `inkling`) into logestix end to end:
converter, NVFP4 tree, model class, tokenizer + tool bridge, CUDA kernels, all
lifecycle gates (SelfCheck/FSI/CODE/DET, tool round-trip, perf), report.
Work happens on branch `feat/inkling-base-ingest`, TDD per stage, temp files in
`/mnt/fast_storage/tmp`, CUDA docs from the local mirror
`/home/alexmv2025/projects/cuda/docs`.

## Stage-0 findings (frozen)

Source: config.json + all 108 safetensors headers (audited 2026-11-17, shapes
dumped to `/mnt/fast_storage/tmp/inkling_shapes.json`), reference semantics
`transformers 5.15.1 models/inkling/modeling_inkling.py` +
`conversion_mapping.py::inkling_mm_model`.

- NEW FAMILY member, not a sibling of inkling-2-small: `inkling` ≠ `inkling2`
  (no L2 clip, no sinks in attention, no AttnRes depth mixing, no expert
  softcaps, scale 1/head_dim, static convs without normalizer, plain residual).
- Geometry: hidden 6144, 66 layers, vocab 201024 (unpadded 200058), 64 q heads
  × 128; sliding layers (55): window 512, 16 kv heads; full layers (11, ids
  5,11,…,65): 8 kv heads, rel_extent 1024, log-scaling from pos 128000 α 0.1.
- Positions: learned relative bank `rel_logits_proj.proj [16, extent]` per
  layer, `wr_du [1024,6144]` → per-head d_rel=16 states; bias zero outside
  [0, extent). q/k per-head RMS-norm (weight [128]); attention scale 1/128.
- 4 static depthwise causal convs per layer, kernel 4, `conv(x)+x` (no
  normalizer, no dynamic taps): k_sconv/v_sconv on raw projections (pre-norm),
  attn_sconv/mlp_sconv after attn/MLP output (pre-residual-add). FP32 compute.
- MoE layers 2..65: 256 routed experts top-6 + 2 shared; sigmoid scores,
  selection-only bias [256]; weights = softmax over logsigmoid of the 6 chosen
  routed + 2 shared logits (joint renorm), × route_scale 8 × gate global_scale;
  shared experts get their 2 gammas. Expert SwiGLU silu, no softcaps.
  `mlp.experts.w13_weight [256,6144,6144]` is HALVES [gate;up] (Interleave op
  in mapping = de-interleave on load → gate_up rows interleaved in the engine
  target; converter must handle explicitly). `shared_w13_weight [2,6144,6144]`
  is ROW-INTERLEAVED gate/up (Interleave+Chunk). Dense `w13_dn [49152,6144]`
  also row-interleaved. w2/down are plain.
- Dense MLP layers 0..1: intermediate 24576, output × mlp.global_scale.
- Head: embed_norm(embed(ids)); final norm; hidden / logits_mup_width_multiplier
  24.0 before lm_head; no softcap. embed/unembed untied [201024,6144].
- Ballast: vision hmlp (8 tensors), audio (2), MTP (160 + mtp.safetensors
  outside the index) — strip & count, never apply.
- Tokenizer: tiktoken BPE (`tiktoken/tokenizer.model`), kind 2 (Kimi/Inkling-2
  bridge); ranks/added_tokens DIFFER from Inkling-2-Small; eos 200006
  (<|content_model_end_sampling|>), endoftext alt; bos none. Chat template:
  same marker family (<|message_user|>/<|message_model|>/<|content_text|>/
  <|end_message|>, tool_declare XML, content_invoke_tool_json), minor diffs
  vs Inkling-2 (effort map, tool-name header) — verify against THIS jinja.
- Precision: source is pure BF16 (no hf_quant_config.json). Norms/biases/rel
  banks/sconv/router stay FP16/FP32; dense linear matrices → NVFP4; routed
  experts → NVFP4 (~486 GiB) streamed via neutral ExpertStore mmap views.

## Feasibility (RTX 5090 32 GB / host 750 GB / 2.6 TB free disk)

Dense device ≈ 8.9 GiB NVFP4 + 4.6 GiB FP16 embed/unembed ≈ 13.5 GiB (+KV,
workspace) — fits. FP16-everything-dense (~36 GiB) does NOT fit — rejected.
Expert tree ~486 GiB on disk (source read 1.9 TB), host-bounded mmap; expected
read-bound ~0.5–1.5 t/s. ulimit -l unlimited not needed (no full pin).

## Approaches considered

1. **Standard NVFP4 ingestion, new `inkling_moe_base_*` files** (chosen):
   full stage 0–9 protocol, own frozen passport
   `lsxcommon::inkling::base::frozen`, kType `inkling`, forge `--inkling-base`,
   format `logestix-inkling-nvfp4`. Reuse neutral layers (lsx_nvfp4, lsx_gemv,
   ExpertStore, tokenizer kind-2 bridge, parity-gate drivers) — no borrowing
   of inkling-2 geometry/kernels (independence guard).
2. Sibling reuse of Inkling-2-Small class — rejected: different architecture
   semantics (documented wrong-logits trap).
3. FP16 dense planes — rejected: ~36 GiB > VRAM.

## Goldens / validation

952B source cannot run on this box → FSI/CODE/DET goldens template-derived
(Inkling-2-Small precedent), `source` field says so. Numeric validation:
layer-by-layer vs float64 re-derivation + `InklingTextModel` real dense weights
with on-demand expert dequant; knowledge sanity gate (mean NLL + exact top-1
"Paris") as the interleave-defect oracle. Red borderline gate → residual-plane
procedure (bug checklist first).

## Non-goals

Vision/audio/MTP execution; inkling-2 code changes; serving-side tuning beyond
the seeded perf row; Hopper/Ada fallbacks.

## Builder calls (mine, operator may override)

- Variant token `base` added to SchemeRegex in the same commit as first files.
- SelfCheck probe parameters measured from THIS model, not copied.
- KV layout pos-major default; sconv states live beside KV in session state.
- Manifest `model_type: "inkling"`, vault dir `Inkling`, alias `inkling`.