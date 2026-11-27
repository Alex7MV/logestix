# Gemma-4-26B-A4B-it — Design

Date: 2026-08-22
Status: Approved (pending implementation plan)
Protocol: Model Ingestion Protocol (8 steps), etalons: Qwen3.5-9B (pipeline structure), GLM-5.2 (loader + expert store)

## Goal

Integrate `gemma-4-26B-A4B-it` (vault: `D:\Programs\lsx_model\gemma-4-26B-A4B-it`, ~51.6 GB BF16,
2 safetensors shards, 1013 tensors) as a first-class model in the logestix stack: converter in
lsxforge, GLM-style loader, CUDA inference kernels, factory registration, TDD test suite with
real-model selfcheck.

## Source model facts (verified against config.json / safetensors index)

- Architecture: `Gemma4ForConditionalGeneration` (`model_type: gemma4`), text core `gemma4_text`.
  Vision tower (27 layers) and audio/image tokens are multimodal fat — stripped at conversion.
- Text geometry (`text_config`): hidden_size=2816, num_hidden_layers=30, num_attention_heads=16,
  head_dim=256, num_key_value_heads=8, num_global_key_value_heads=2, global_head_dim=512.
- Layer grid: repeating **5×sliding_attention + 1×full_attention** (layers 5/11/17/23/29 are full),
  sliding_window=1024, max_position_embeddings=262144.
- RoPE: full_attention partial_rotary_factor=0.25, theta=1e6 ("proportional"); sliding_attention
  theta=1e4 ("default").
- MoE block enabled: num_experts=128, top_k_experts=8, moe_intermediate_size=704; each layer ALSO
  carries a dense MLP (intermediate_size=2112) with its own pre/post feedforward norms
  (`pre/post_feedforward_layernorm_1` = dense branch, `_2` = MoE branch — verified by shapes during
  conversion).
- Router extras: `router.proj.weight`, `router.scale` (F32 scalar), `router.per_expert_scale`
  (F32 [128]); per-layer `layer_scalar`.
- Norms: RMSNorm eps=1e-6, Gemma-style `(1+w)` offset scaling. Attention has q_norm/k_norm
  (per-head, high-res).
- Activation: gelu_pytorch_tanh everywhere (dense MLP and expert FFN).
- Vocab 262144, tie_word_embeddings=true (NO lm_head tensor — embed doubles as output head),
  final_logit_softcapping=30.0.
- Tokenizer specials: bos=2, eos=[1,106], pad=0. Chat template: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`.

## Decisions made with the operator

| Question | Decision |
|---|---|
| Integration approach | **A**: new isolated `lsxcommon::gemma` pipeline mirroring the Qwen file layout, reusing generic primitives (integrity gates, EstimateVram, ExpertStore, RunParisSelfCheck, framing registry). Not extending qwen_* code, not wholesale-reusing GLM kernels. |
| SelfCheck prompt framing | **Chat template** via a new entry in the `prompt::FramingDescriptor` registry (currently only glm-5.2). |
| Loader pattern | **GLM-style loader** (unification target): GPU in-place BF16→FP16, arena-backed per-layer uploads, pinned staging ring, routed experts on-demand via `ExpertStore` LRU mmap. |
| Dense weight precision | **As GLM**: attention projections and dense MLP stay raw BF16 → uploaded FP16; NVFP4 quantization applies to routed experts only. |

## Architecture

New files (all inside `libs/lsxcommon` unless noted):

```
include/lsxcommon/gemma4_convert.h          TensorKind enum, GemmaTopology, ConvertGemmaModel
src/gemma4_convert.cpp                      classifier, integrity gates, converter
include/lsxcommon/gemma4_inference_load.h   GemmaLayerWeights/GemmaPinnedWeights, GemmaExpertDb, load APIs
src/gemma4_inference_load.cpp               arena-backed loader, staging ring, expert LRU wiring
src/gemma_base_model.h/.cpp                 GemmaBaseModel : ModelBase (Load/Infer/Benchmark/SelfCheck)
src/gemma_4_26b_a4b_model.h/.cpp            kType "gemma-4-26b-a4b-it", frozen constants + static_asserts
src/gemma_inference.cu                      all CUDA kernels
```

Modified files:

```
src/model_registry.cpp                      Register(Gemma4_26BA4BModel::kType, ...)
include/lsxcommon/prompt_framing.h + impl   new FramingDescriptor for gemma chat template
libs/lsxcommon/CMakeLists.txt               host sources + lsxcommon_gemma_cu OBJECT lib
apps/lsxforge/src/main.cpp                  --gemma4 flag → ConvertGemmaModel
tests/CMakeLists.txt                        test targets below
```

## Stage 1 — Converter (lsxforge)

Tensor classifier (`GemmaTensorKind`, regex over HF names, vision/audio skipped):

- `Embed` — `model.language_model.embed_tokens.weight` (tied; also used as head)
- `FinalNorm` — `norm.weight`
- `Attn{Q,K,V,O}`, `Attn{Q,K}Norm`
- `Mlp{Gate,Up,Down}` — dense branch
- `ExpertsGateUp` `[E, 2*I, H]`, `ExpertsDown` `[E, H, I]` — routed experts
- `RouterProj`, `RouterScale` (F32 scalar), `RouterPerExpertScale` (F32 [128])
- Norms: `InputNorm`, `PostNorm`, `PreFFN1/2`, `PostFFN1/2`, `LayerScalar`
- Skip: everything under `vision_tower`/`embed_vision`, MTP-like extras if present

Precision policy (GLM-style):

- Raw BF16 kept: embed, final norm, all attention tensors, dense MLP, ALL norms, router proj/scales,
  layer_scalar. Uploaded later as FP16 (GPU in-place conversion).
- NVFP4 (E2M1 nibbles + per-block E4M3 scales + global_scale): routed experts only
  (`experts.gate_up_proj`, `experts.down_proj`) — one `PackedExpert`-shaped record per (layer, expert).

Output layout (via `lsx_model_metadata()` path authority, never hardcoded paths):

```
<root>/logestix/pinned/global.arrow        embed (+final norm)
<root>/logestix/pinned/layer_N.arrow       raw BF16/F32 per-layer tensors
<root>/logestix/experts/layer_<l>/exp_<e>.arrow   NVFP4 packed expert (30×128 = 3840 files)
<root>/logestix/model.index.json           "format": "logestix-gemma", topology + VRAM profile
```

Gates (reuse `integrity::` helpers): config.json mandatory & authoritative (text_config nesting);
`VerifyAllTensorReads`; `CheckTensorCoverage` (expected names generated from topology, vision
excluded); `CheckLayerContiguity`; geometry cross-check config↔tensor shapes;
`BuildAndVerifyTokenizer(bos_required=true(2), eos_required=true(1|106))`;
final `VerifyConvertedModel`.

`GemmaTopology`: hidden=2816, layers=30, heads=16, kv_heads=8, global_kv_heads=2, head_dim=256,
global_head_dim=512, sliding_window=1024, layer_types vector, moe{experts=128, top_k=8,
moe_intermediate=704}, dense_intermediate=2112, rms_norm_eps=1e-6, rope params (per layer-type),
vocab=262144, softcap=30.0, activation=gelu-tanh.

lsxforge: `--gemma4` flag dispatches to `lsxcommon::gemma::ConvertGemmaModel(input, output)`.

## Stage 3/4 — Loader (GLM-style)

`gemma4_inference_load.{h,cpp}`:

- `GemmaBf16ToHalfInPlace(d_buf, numel, stream)` — bit-exact nearest-even GPU conversion
  (pattern: `GlmBf16ToHalfInPlace`).
- `GemmaLayerWeights` — half norms/projections, F32 router scales + layer_scalar;
  **arena-backed**: every d_* pointer is a view of one cudaMalloc arena (`d_arena`, `arena_bytes`);
  `Free()` releases the arena.
- `LoadGemmaPinnedWeights(model_path, pw, stream[, layer_filter])` — pinned staging ring, async on
  caller's stream, strict errors on missing files, exact `resident_bytes` (VRAM only).
- Routed experts on demand:
  - `GemmaExpertEntry` — zero-copy views into the mmap'd Arrow IPC buffer (w1=gate_up, w2=down +
    scales + global scales), null host pointers mark missing entries (deterministic skip).
  - `GemmaExpertDb = lsxcommon::gpu::expert_store::ExpertStore<GemmaExpertEntry>` — flat index
    `(layer * 128 + expert)`, lazy mmap, bounded LRU (fd cap).
  - `ensure_gemma_expert_resident(db, layer, expert, stream)` — stream-synchronized before eviction
    (DMA use-after-free barrier, same contract as GLM).
  - Cache slot override hooks mirroring `SetGlmExpertCacheSlotsOverride` (governor/tuner ready).

## Stage 3/5/6 — Model class + CUDA kernels

`GemmaBaseModel : ModelBase`:

- `kType = "gemma-4-26b-a4b-it"`; frozen constants with `static_assert`s:
  hidden=2816, layers=30, heads=16, kv_heads=8, global_kv_heads=2, head_dim=256,
  global_head_dim=512, experts=128, top_k=8 (new-model requirement: compile-time geometry lock;
  runtime `ValidateGeometry()` hard-gates Load, escape hatch env `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY`).
- `Infer()`: cublas handle bound to ONE end-to-end `cudaStream_t`; no `cudaDeviceSynchronize` in the
  generation loop (only final sync); incremental sessions like Qwen (KV continuation when prompt
  extends session tail); final logits softcapped: `logits = tanh(logits/30) * 30`.
- KV caches: per-layer `[slots][8][256]` (sliding layers keep only window-relevant positions via ring
  indexing); one shared global cache `[slots][2][512]` across all 30 layers. Exact global-head
  mixing semantics pinned down from actual k_proj/v_proj shapes during conversion (documented in the
  converter test).
- `SelfCheck()` → `runtime::RunParisSelfCheck` with prompt framed through
  `BuildPromptInputIds("gemma-4-26b-a4b-it", ...)`
  → `<start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n`.

Kernels (`src/gemma_inference.cu`, patterns cited):

| Kernel | Purpose | Pattern source |
|---|---|---|
| `gemma_embed_kernel` | token embedding lookup | qwen embed_kernel |
| `gemma_rmsnorm_kernel` | RMSNorm ×(1+w) | qwen rmsnorm_offset_kernel |
| `gemma_rope_cos_sin_kernel`, `gemma_partial_rope_kernel` | RoPE tables (full: 0.25/1e6; sliding: 1e4), HF rotate_half pairing | qwen rotary kernels |
| `gemma_qk_norm_kernel` | per-head QKNorm (high-res weights) | qwen head_norm |
| `gemma_sliding_attn_kernel` | causal attention, band mask window=1024, ring-indexed KV rows (stride arithmetic invariant) | new, softmax chunking from full_attn_softmax_kernel |
| `gemma_full_attn_kernel` | chunked block softmax; reads per-layer KV + global KV heads | qwen full_attn_softmax_kernel |
| `gemma_gelu_mlp_kernel` | dense MLP, GELU-tanh, cuBLAS GEMMs (FP16) | — |
| `gemma_router_topk_kernel` | logits→top-8 of 128, weights × router_scale × per_expert_scale[e], deterministic ties | GlmMoETopkKernel |
| `gemma_moe_stack_kernel` | stage selected expert payload into device staging buffers, GELU FFN accumulate | GLM routed-expert stack pattern |

Shared-memory RMW isolation: writes only from thread 0 under `__syncthreads()` (protocol step 6).

Per-layer forward order:
`input_norm → attn(sliding|full) [+q/k norm] → post_norm → residual`
then dual FFN branches:
`pre_ffn_1 → dense MLP → post_ffn_1` and `pre_ffn_2 → MoE(top-8) → post_ffn_2`,
each added to the residual; `layer_scalar` applied per branch (exact semantics confirmed from
weight shapes at conversion time). Branch-to-norm mapping (_1=dense, _2=moe) is asserted in the
converter test.

## Stage 7 — Registration & networking passport

- `LsxModelFactory::Register("gemma-4-26b-a4b-it", ...)` in `EnsureModelRegistry()`.
- Passport/token translation bridge follows protocol step 7 (`BuildPassport`/`WritePassport`,
  vault `passport/passport.bin`); IoWorker integration unchanged (existing machinery is
  model-agnostic once the model is registered).

## Stage 8 — Tests (TDD)

Each cycle: failing test → minimal implementation → green. Test executables (Catch2, tests/):
synthetic mini-gemma fixture (hidden 64, 4 layers, 8 experts) analogous to `qwen_test_fixture.h`;

| Executable | Coverage | Runs without real model? |
|---|---|---|
| test_gemma4_convert | tensor classifier, vision/audio skip, 128 expert files/layer, index format, branch-norm mapping assert | yes |
| test_gemma4_loader | BF16→FP16 bit-exactness vs host path, arena Free correctness, expert LRU ensure/resident/evict | yes |
| test_gemma4_norm | RMSNorm-(1+w) vs CPU FP64 reference | yes |
| test_gemma4_rotary | partial RoPE vs CPU FP64 reference | yes |
| test_gemma4_attn | sliding band-mask (position outside window provably ignored), full attn + global KV heads | yes |
| test_gemma4_moe | router top-8 determinism, scale application, GELU expert FFN vs CPU | yes |
| test_gemma4_model | factory registration, Load+Infer on synthetic geometry (`LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`) | yes |
| test_gemma4_selfcheck | real-model stress: framed prompt, ≥3 cycles (`LSX_TEST_GEMMA_SELFCHECK_CYCLES` override), bit-exact `final_logits` (memcmp), `first_gen_text=="Paris"` | SKIP (code 4) without model |

Real-model selfcheck path resolution: env `LSX_TEST_GEMMA_MODEL_PATH`, else
`lsxcommon::lsx_model_models_dir()/"gemma-4-26B-A4B-it"`. CTest `SKIP_RETURN_CODE 4`.

## CMake

- Host sources appended to `lsxcommon` sources list.
- New OBJECT library `lsxcommon_gemma_cu`: `src/gemma_inference.cu`, `CUDA_ARCHITECTURES
  "120-real;121-real"`, `--extended-lambda`, links cudart+cublas; PUBLIC-linked into `lsxcommon`
  (same pattern as `lsxcommon_qwen_cu`). No abseil container headers reachable from `.cu` TUs.
- Test executables link `lsxcommon` + `Catch2::Catch2WithMain`; Win32 POST_BUILD arrow DLL copies
  apply automatically.
- lsxforge: `--gemma4` absl flag + dispatch (mirrors `--qwen`).

## Error handling

- Converter: hard errors (missing config, uncovered tensors, layer holes, geometry mismatch,
  tokenizer failures) — abort with descriptive message, consistent with qwen converter integrity
  gates.
- Loader: strict errors on missing pinned dir/files/corrupt tensors (no silent fallbacks), GLM-style.
- Runtime: capacity validated up front (prompt+gen vs slot capacity); deterministic rejection.

## Out of scope

- Vision/audio tower support.
- Benchmark-specific tuning pass (`--benchmark` uses existing generic pipeline).
- Fused decode kernel set (phase 2 follow-up if needed).
