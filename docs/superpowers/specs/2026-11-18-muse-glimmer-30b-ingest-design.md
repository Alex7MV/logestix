# Muse-Glimmer-30B ingestion — design

Date: 2026-11-18. Approved by operator ("ок").

## Intent

Ingest `Muse-Glimmer-30B` (`/mnt/fast_storage/lsx_model/meta-models/Muse-Glimmer-30B`,
architecture `MuseGlimmerForConditionalGeneration`, text tower
`muse_glimmer_text`) into logestix end to end: converter, NVFP4 tree, model
class, tokenizer + ATEM tool bridge, CUDA kernels, all lifecycle gates
(SelfCheck/FSI/CODE/DET, tool round-trip, perf), report. Branch
`feat/muse-glimmer-30b-ingest`, TDD per stage, temp files in
`/mnt/fast_storage/tmp`, CUDA docs from the local mirror
`/home/alexmv2025/projects/cuda/docs`.

## Stage-0 findings (frozen)

Source: config.json + both safetensors headers (1436 tensors, 59.55 GB BF16),
reference semantics `transformers 5.15.1 models/muse_glimmer/` (modular +
modeling; attention inherits AfmoeAttention, decoder Gemma2DecoderLayer).

- NEW FAMILY `muse` (dense). Not a sibling of anything ingested: gated
  attention + NoPE full layers + centered norms + weightless embed norm is a
  unique combination; own passport, converter, loader, kernels.
- Geometry (text_config): hidden 6656, 52 layers, intermediate 19968 (SwiGLU
  **silu**), vocab 202048, 32 Q heads / 2 KV heads × head_dim 128,
  max_position 131072, tie_word_embeddings **false**.
- Layer pattern: [sliding, sliding, sliding, full] × 13 — full layers are
  3,7,…,51 (13 full, 39 sliding); sliding_window 2048.
- RoPE: theta 500000, FULL head dim, **sliding layers only**; full layers are
  NoPE (`layer_rope_theta[i] == 0` → position_embeddings=None).
- Attention: q/k per-head **weightless** RMSNorm (Gemma4RMSNorm with_scale
  =False, eps 1e-5); Q additionally × **qk_scale_factor 3.87**; score scale
  1/sqrt(128); output × **sigmoid(self_attn.gate_proj(x))** elementwise over
  4096 dims BEFORE o_proj. Projections: q [4096,6656], k/v [256,6656],
  gate [4096,6656], o [6656,4096]. attention_bias false.
- Norms: 4 per layer, ALL **centered** Gemma2-style ×(1+w): input_layernorm
  eps 1e-5, post_attention_layernorm eps **1e-8** (post_norm_eps),
  pre_feedforward_layernorm eps 1e-5, post_feedforward_layernorm eps 1e-8.
  Residual sandwich: x + post_norm(attn(norm(x))); x + post_ffn(mlp(pre(x))).
  Final `model.language_model.norm` is Gemma4RMSNorm **with** scale (plain ×w,
  eps 1e-5).
- Embedding: `MuseGlimmerTextNormedEmbedding` = weightless RMSNorm (eps 1e-5)
  applied to gathered rows; **no sqrt(hidden) scaling**. lm_head untied
  [202048,6656].
- Logits: `20 * tanh(raw * 0.19611613513818404 / 20)` — output_multiplier
  (= 1/sqrt(6656/256)) then final_logit_softcapping 20. No attn softcap.
- Ballast: vision_tower (806 tensors, 50-layer ViT width 1536) +
  vision_adapter.fc1/fc2 + vision_projection (3) = **809 tensors ≈ 3.8 GB** —
  strip & count (`skipped_visual_tensors`), never applied.
- Tokenizer: byte-level BPE, 200000 base + 2048 added (ALL special:true,
  ids 200000..202047). Pre-tokenizer is EXACTLY the o200k/Llama-3 case-run
  pattern already implemented as **kind 3** (Llama3WordStream: digits {1,3},
  greedy contractions, ` ?[^\s\p{L}\p{N}]+[\r\n/]*`, the three whitespace
  alternatives; ByteLevel add_prefix_space=false, no normalizer).
  bos 200000 `<|begin_of_text|>`, eos {200001 `<|end_of_text|>`,
  200008 `<|eot|>`}, pad 200018 `<|finetune_right_pad|>` (text_config pad null,
  generation_config supplies it). Template auto-prepends bos.
- Chat/tool format (**ATEM**, from chat_template.jinja): harmony-style
  channels `<|start|>role[ to=recipient]<|message|>…<|eot|>` (`<|eom|>`
  between same-role messages / mid-turn). Default system turn when absent:
  "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\n
  Reasoning strength: high.\n\n# Valid recipients: …". Tool calls:
  `<|start|>assistant to=<fn><|message|><atem:function_calls>\n<atem:invoke
  name="<fn>">\n<atem:parameter name="<k>">v</atem:parameter>\n…</atem:invoke>
  \n</atem:function_calls><|eot|>`. Tool results: `<|start|>tool <fn>
  <|message|><tool_output name="<fn>">\n…\n</tool_output><|eot|>`.
  Reasoning channel: `assistant to=self` … `<|eom|>`. All markers are
  special:true → inject by vocab id, parse by scanning token ids.
- Precision: pure BF16 source. All dense linear matrices (attn q/k/v/o/gate +
  mlp gate/up/down) → NVFP4; embed + lm_head stay **FP16** (phi-4 precedent,
  untied head must stay exact for greedy); norms FP16, never quantized.

## Feasibility (RTX 5090 32 GB / host 750 GB / 2 TB free disk)

NVFP4 linears ≈ 25.7B params → ~13.5 GiB incl. scales; embed+lm_head FP16
≈ 5.0 GiB; device total ≈ 18.6 GiB. KV FP16 = 52 × 2 × 2 × 128 × 2 B
≈ 53 KiB/token → 0.11 GiB @ 2048, 1.7 GiB @ 32768. Fits with headroom;
a residual plane on attention+MLP (~+13 GiB) still fits at short context if
gates demand it (budget honestly per stage-2 procedure). Source BF16 (59.5 GB)
does NOT fit VRAM → HF goldens captured on CPU (text tower only, 750 GB host).

## Approaches considered

1. **Standard NVFP4 ingestion, new `muse_dense_*` files** (chosen): full
   stage 0–9 protocol, passport `lsxcommon::muse_dense::frozen`, kType
   `muse-glimmer-30b`, forge `--muse`, format `logestix-muse-nvfp4`. Reuse
   neutral layers (lsx_nvfp4 GEMV/GEMM, lsx_gemv FP16 head, KvCacheGeometry,
   tokenizer kind-3 bridge, shared selfcheck/parity drivers).
2. Sibling reuse of gemma_dense — rejected: different geometry AND semantics
   (independence guard forbids; gemma has no gate proj, no NoPE, GeGLU,
   tied NVFP4 head, different norms).
3. FP16 everything — rejected: ~56 GiB > 32 GB VRAM.
4. NVFP4 embed/lm_head (gemma-26b style) — rejected initially: untied head +
   weightless embed norm make FP16 cheap (5 GiB) and exact; revisit only if
   VRAM pressure appears.

## Goldens / validation

Source runs on CPU via transformers 5.15.1 (muse_glimmer is natively
supported): text-tower-only oracle loader (skip vision) for
fsi/code/det capture; greedy, chat_template.jinja framing, per-type
OVERRIDES entry. TDD: every stage lands with its failing-test-first cycle.
Kernel parity vs host references with non-unit global_scale fixtures.
SelfCheck probe parameters (frame_prompt/max_gen/bit_exact) measured from
THIS model's source run, not copied. Red borderline gate → stage-2 residual
procedure (bug checklist first), outcome recorded either way.

## Non-goals

Vision execution; DFlash speculative drafter (not in this checkpoint's
index); inkling/gemma code changes; serving-side tuning beyond the seeded
perf row; pre-Blackwell fallbacks.

## Builder calls (mine, operator may override)

- Family token `muse` added to SchemeRegex + FamilyStems in the same commit
  as the first files; `muse_tool_format` in AllowedCommon.
- Constants header at `libs/lsxcommon/include/lsxcommon/muse_dense_constants.h`
  (gemma_dense precedent).
- KV layout pos-major via KvCacheGeometry; uniform per-layer geometry
  (2 KV × 128), window 2048 on sliding layers, full prefix on NoPE layers.
- Sliding layers may cap KV retention at the window; v1 keeps full-length
  cache for simplicity (53 KiB/token is cheap), window applied in the kernel.
- Manifest `model_type: "muse-glimmer-30b"`, vault dir `Muse-Glimmer-30B`,
  alias `muse-glimmer-30b`; lifecycle_tu = muse_dense_model.cpp +
  muse_dense_inference.cu.
- Activations FP16 / FP32 accumulation; `-fmad=false` OBJECT lib; family
  pins nothing — kOff comes from the harness/benchmark flag.