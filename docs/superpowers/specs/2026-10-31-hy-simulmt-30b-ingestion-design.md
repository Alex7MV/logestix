# Hy-SimulMT-30B-A3B — ingestion design

Branch: `feat/ingest-hy-simulmt-30b`
Date: 2026-10-31
Protocol: Model Ingestion Protocol (`ingesting-a-model`), stages 0–8 + 4b.
Source checkpoint: `/mnt/fast_storage/huggingface/Hy-SimulMT-30B-A3B`
(`tencent/Hy-SimulMT-30B-A3B`, 57 GB, 18 shards + one audio shard).

## Stage 0 — triage

### 0.1 Classification: **sibling variant of the `hy` family** (type `moe`)

The checkpoint's `architectures` is `HYV3SpeechForCausalLM` and `model_type`
is `hy_v3`. Its text side is the architecture already ingested as
`hy-memory-30b-a3b`: 18 673 tensors with the same names and shapes
(`model.*` + `lm_head.weight`), the same decoder geometry and the same
tokenizer file (byte-identical `tokenizer.json`, sha256
`ad96ef9f641b79bb…`, 120 000 BPE entries / 818 added tokens).

The only structural delta is **403 `audio_tower.*` tensors** (660 MB, stored in
`model-audio-tower.safetensors` and listed in the index): a 24-layer speech
encoder + speech adapter. It is ballast for this engine — stored and counted,
never applied (same verdict as granite-switch's ASR tower / MTP heads).

Registry rows to extend in the same commit as the files:

| registry | row |
|---|---|
| `test_code_naming_guard.cpp` | `SchemeRegex()` variant token `simulmt` |
| `test_model_independence_guard.cpp` | `kPassports` (`hy-simulmt-30b-a3b` → `hy_moe_simulmt_constants.h`, ns `lsxcommon::hy_simulmt::frozen`) |
| `test_model_independence_guard.cpp` | `consumers` (`hy_moe_simulmt_model.cpp`) |
| `test_model_independence_guard.cpp` | `kLifecycle` (model TU + the shared `hy_moe_inference_pipeline.cu`) |
| `test_model_suite_guard.cpp` | `Suites()` row (convert→load→model→kernels→selfcheck→fsi→code→det→perf) + tool-verdict row |
| `test_ctest_labels_guard.cpp` | — (label every new test) |

A sibling still owns a passport: the independence guard requires a unique
namespace per model, and the repo precedent for a second architecture-identical
member (`nemotron-3.5-lightning-30b-a3b-nvfp4`, `granite-switch-4.1-30b`) is a
separate constants header plus a separate model class **over shared
kernels/loader**. No kernel or loader logic is duplicated.

### 0.2 Geometry (frozen, identical to Hy-Memory)

| field | value |
|---|---|
| hidden | 2048 |
| layers | 48 (layer 0 dense MLP, layers 1..47 routed MoE) |
| attention | 32 Q / 4 KV heads, head_dim 128, GQA |
| rotary | full 128 dims, rotate-half, theta 11 158 840 |
| QK norm | per-head plain RMSNorm (weight form), 128 |
| norm eps | 1e-5 |
| vocab | 120 832 (embed and head untied, both BF16) |
| experts | 128 routed, top-8, intermediate 768 (`expert_hidden_dim`), one gated shared expert at 768, `router_scaling_factor` 2.826, sigmoid router + `expert_bias` |
| dense MLP | SwiGLU at 6912 |
| dtype | source BF16; norms/router/bias stay high precision |

Config deltas that the parser already handles: `moe_intermediate_size` is 768
(no vestigial 1536), `mlp_layer_types` is absent (`first_k_dense_replace: 1`
drives the dispatch), `rope_theta` is flat (no `rope_parameters` object).

### 0.3 Tokenizer

Byte-level BPE, `tokenizer_kind` 0 (digit-run / CJK / 6-alternative GPT-2
pre-tokenizer), no auto-BOS (the template spells it out), no CR byte and no
UTF-8-illegal lead-byte tokens → `TokenizerSpecialPolicy::drop_unmapped_bytes`
(same policy as Hy-Memory, which the identical `tokenizer.json` makes exact).
Special ids: bos 120000, eos 120025, pad 120002, user 120006, assistant 120007.
Extra ids present but not used by the text path: eos-of-dialogue 120026,
`<|turn_end|>` 120008, audio placeholder 120023.

### 0.4 Ballast

`audio_tower.speech_encoder.*` (conv2d1..3, `conv_out`, 24 transformer layers,
`ln_post`, `proj1/2`), `audio_tower.speech_adapter.*`. Stripped and counted in
`skipped_visual_tensors`. Nothing else is skipped; an unrecognised tensor stays
a hard converter error.

### 0.5 Feasibility

NVFP4 resident footprint is the same class as Hy-Memory: ~16.5 GiB pinned
(FP16 embed + lm_head 0.94 GiB, NVFP4 attention/dense/shared, F32 router, FP16
norms, 15 GiB routed-expert slabs), so the 30B model fits the 32 GB RTX 5090
with no expert cache. Source 57 GB → ~17 GB output. Host RAM 750 GB, disk
3.6 TB free.

### 0.6 Semantics pinned (each a candidate for a silently wrong token)

- attention scale `1/sqrt(128)` at QK only;
- full rotary over all 128 dims; no rope scaling;
- residual stream FP16 in this engine; router + routed/shared combination FP32;
- no final logit softcap, no model-level temperature;
- shared expert is gated; routed coefficients are renormalised sigmoid scores
  times `router_scaling_factor`.

## Design

### Files

| file | role |
|---|---|
| `libs/lsxcommon/src/hy_moe_simulmt_constants.h` | frozen passport, ns `lsxcommon::hy_simulmt::frozen`, kType `hy-simulmt-30b-a3b`, tag `logestix-hy-simulmt-nvfp4` |
| `libs/lsxcommon/include/lsxcommon/hy_moe_simulmt_convert.h` + `src/hy_moe_simulmt_convert.cpp` | `ConvertHySimulMTModel`, wraps the shared hy conversion with the SimulMT identity and audio-ballast handling |
| `libs/lsxcommon/src/hy_moe_simulmt_model.{h,cpp}` | `HySimulMTModel` (reuses `hy::LoadHyDenseWeights`, `hy::RunHyInference`, `hy::RenderHyConversation`, `hy::ParseHyToolCalls`) |
| `apps/lsxforge` | `--hy_simulmt` flag + `Pipeline::kHySimulMT` |
| `tools/hy_v3_oracle.py` | skip `audio_tower.*` (the text tensors are the same layout) |
| `tools/{fsi,code,det}_capture.py` | `loader: "hy_v3"` override for `hy-simulmt-30b-a3b` |
| `tools/hy_chat_framing_capture.py` | unchanged; rerun against the SimulMT template |

### Shared-code changes (no logic fork)

- `hy_moe_convert.cpp`: a `HyConvertSpec` (model_type, model name, format tag,
  audio-ballast toggle) threaded through the existing pipeline; the existing
  `ConvertHyMoeModel` keeps the Hy-Memory identity. The classifier returns a
  ballast kind for `audio_tower.*` only when the spec allows it, and ballast is
  excluded from `VerifyAllTensorReads`, the pinned inventory and the VRAM
  inventory.
- `hy_moe_load.{h,cpp}`: `LoadHyDenseWeights(path, const HyPassportGeometry&,
  pw, stream, error)` where the geometry carries the frozen expectations and
  the format tag. The old three-argument overload forwards Hy-Memory's frozen
  passport, so existing tests and the Hy-Memory model are untouched. The
  hard-coded `frozen::kFirstMoeLayer` slab loop becomes a dispatch over
  `top.mlp_types`.

### Gates

- SelfCheck: Paris, `FastMathMode::kOff`, ≥3 stress cycles, `SKIP_RETURN_CODE 4`.
- FSI / CODE / DET: committed goldens captured from the **source safetensors**
  with `tools/*_capture.py` (`loader: "hy_v3"`, CPU because the 60 GB bf16 tree
  does not fit the card). The generated token ids come from the checkpoint's
  own tokenizer + `chat_template.jinja`.
- Tokenizer parity and chat framing: capture from the checkpoint (tokenizer
  only, fast) and compare id-for-id.
- Perf: measured through `test_perf_baseline` at the canonical 2048-token
  protocol; the JSON row is written by the harness, never by hand.

### Known limits (recorded, not silent)

- The speech encoder is out of scope: audio input / speech output is not
  served. The text decoder is the parity surface and the tool bridge.
- FSI/CODE/DET goldens record what the source answers. A red gate is recorded
  as parity debt; goldens and perf rows are never rewritten to pass.

### Stage plan (TDD, each stage lands with its test)

1. Passport + guard registries (red → header).
2. Converter spec + `--hy_simulmt` + synthetic convert test (audio ballast
   present, counted, and excluded from the output graph).
3. Loader passport parameterisation + synthetic loader test for the SimulMT
   identity/format.
4. Model class, factory registration, model/lifecycle test.
5. Tool bridge pinned by a SimulMT framing golden.
6. SelfCheck + FSI/CODE/DET tests; goldens captured from source.
7. Perf baseline + `test_perf_baseline_logic` pairing.
8. Real conversion in place, full lifecycle sweep, docs (AGENTS.md,
   SELFCHECK_MATRIX, OPEN_TASKS, model report).

## As built (2026-10-31)

Three facts discovered while validating the real checkpoint changed the
implementation from the plan above:

1. **The decoder is direction-bound.** Without the checkpoint's own serving
   directive (`SimulInfer/simulinfer/translation.py INSTRUCTIONS`) it only
   summarises/transcribes its input. The model class prepends that directive and
   auto-selects the direction from the source script (Han → zh-en, else en-zh),
   which is also the single-prompt framing. SelfCheck therefore asks the
   canonical fact in the trained direction (source "法国的首都是巴黎。" →
   English containing "Paris"); this is a recorded deviation from the canonical
   English prompt, not a rewritten gate.
2. **Two stop ids.** `generation_config.json` stops on `<eos>` and
   `<|hy_EOT|>`; the shared topology/index and the pipeline now carry the full
   stop set.
3. **The template has one literal delta** (the tool example closes with
   `</tool_calls>`), carried by `HyRenderPolicy`.

Gates as measured: SelfCheck PASS (3 cycles, bit-exact,
`logits_hash 44b846591fd64cb9`); real-weight bidirectional translation PASS;
chat framing 14/14; FSI/CODE/DET **declared debt** — the source answers those
QA bodies with a summary or immediate EOT, so no source-faithful extractable
golden exists (evidence in `docs/OPEN_TASKS.md` §R11 and the model report).
Perf: 87.57 t/s (perf) / 74.85 t/s (r3) at 2048 tokens, TTFT 510/616 ms,
16.5 GiB resident.
