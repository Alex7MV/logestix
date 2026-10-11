# Mistral-Small-4-119B-2603-NVFP4 — ingestion design

Branch: `feat/mistral-small-4-119b`
Date: 2026-10-11
Protocol: Model Ingestion Protocol (`ingesting-a-model`), stages 0–8 + 4b.
Reference checkpoint: `/mnt/llm_models/models/Mistral-Small-4-119B-2603-NVFP4`
(base `mistralai/Mistral-Small-4-119B-2603`, post-training NVFP4 by llm-compressor).

## Stage 0 — triage

### 0.1 Classification: **new family `mistral` (type `moe`)**

No `kType` in `libs/lsxcommon/src/model_registry.cpp` shares this architecture.
The checkpoint's MLA geometry (q_lora 1024 / kv_lora 256 / qk_nope 64 /
qk_rope 64 / v_head 128), its all-36-layer MoE + Yarn/llama-4 long-context
scaling, and its byte-level Tekken BPE differ from every ingested MLA family
(GigaChat `qk=192`, Xing `qk_nope=128`, Kimi, DeepSeek). It is therefore a new
family with its own frozen constants header, converter, loader, kernels, model
class and tool format.

Registry rows to extend (same commit as the files):

| registry | row |
|---|---|
| `test_code_naming_guard.cpp` | `SchemeRegex()` family token `mistral` (already allows `moe`, `convert`, `load`, `kernels`, `inference`, `model`, `expert_cache`, `constants`) |
| `test_model_independence_guard.cpp` | `kPassports` (`mistral-moe` → `mistral_moe_constants.h`, ns `mistral_moe::frozen`) |
| `test_model_independence_guard.cpp` | `consumers` (`mistral_moe_model.cpp`) |
| `test_model_independence_guard.cpp` | `kLifecycle` (model → the TU(s) forming the generation path) |
| `test_model_suite_guard.cpp` | `Suites()` row (convert→load→model→kernels→selfcheck→fsi→code→det→perf) + tool-verdict row |
| `test_ctest_labels_guard.cpp` | — (label every new test) |

### 0.2 Geometry (frozen, from `params.json`)

| field | value |
|---|---|
| `dim` (hidden) | 4096 |
| `n_layers` | 36 |
| `n_heads` | 32 |
| `n_kv_heads` | 32 (MLA latent — one shared KV latent per layer) |
| `head_dim` | 128 |
| `q_lora_rank` | 1024 |
| `kv_lora_rank` | 256 |
| `qk_nope_head_dim` | 64 |
| `qk_rope_head_dim` | 64 |
| `v_head_dim` | 128 |
| `vocab_size` | 131072 |
| `norm_eps` | 1e-6 |
| `rope_theta` | 10000.0 |
| `yarn` | factor 128, original_max_position_embeddings 8192, beta 32, alpha 1, **apply_scale false** |
| `llama_4_scaling` | beta 0.1, original_max_position_embeddings 8192 |
| `moe.num_experts` | 128 |
| `moe.num_experts_per_tok` | 4 |
| `moe.num_shared_experts` | 1 |
| `moe.expert_hidden_dim` | 2048 |
| `moe.first_k_dense_replace` | 0 → **every one of the 36 layers is MoE** |
| `moe.routed_scale` | 1.0 |
| `moe.num_expert_groups` / `_per_tok` | 1 / 1 (no group-limited routing) |
| `tied_embeddings` | false |

Derived MLA row counts (validate against the safetensors shapes):

| weight (per layer) | source shape | derivation |
|---|---|---|
| `attention.wq_a` | `[1024, 4096]` | `q_lora × H` |
| `attention.q_a_norm` | `[1024]` | RMSNorm over q latent |
| `attention.wq_b` | `[4096, 1024]` | `heads·head_dim × q_lora` |
| `attention.wkv_a_with_mqa` | `[320, 4096]` | `(kv_lora + qk_rope) × H` |
| `attention.kv_a_norm` | `[256]` | RMSNorm over kv latent |
| `attention.wkv_b` | `[6144, 256]` | `heads·(qk_nope + v_head) × kv_lora` |
| `attention.wo` | `[4096, 4096]` | `H × heads·v_head` |
| `experts.E.w1/w3` | `[2048, 2048]` | NVFP4 packed `[I, H/2]` |
| `experts.E.w2` | `[4096, 1024]` | NVFP4 packed `[H, I/2]` |
| `gate` | `[128, 4096]` | router, BF16 |
| `shared_experts.w1/w3` | `[2048, 2048]` | NVFP4 packed |
| `shared_experts.w2` | `[4096, 1024]` | NVFP4 packed |

### 0.3 Tokenizer

- Byte-level BPE, shipped as an HF `tokenizer.json` (131072 vocab, **269443
  merges**, 1000 `added_tokens`, all `special: true`) — consumed directly by the
  neutral `BuildTokenizerArrowFiles`; no `sp-bpe` bridge needed.
- `bos_token = <s>` (id 1), `eos_token = </s>` (id 2), `pad_token = <pad>` (id 3
  in `tekken.json` ordering; `unk = <unk>` id 0). Verified from
  `tokenizer_config.json` + `tekken.json` `special_tokens`.
- `tie_word_embeddings = false` → `output.weight` is a real `[131072, 4096]`
  BF16 tensor (not tied to `tok_embeddings`).
- `chat_template.jinja` present → drives stage 4b.

### 0.4 Ballast to strip — and to count

| tensor family | action |
|---|---|
| `vision_encoder.*` (patch_conv, 24 transformer layers, ln_pre) | strip + count |
| `vision_language_adapter.w_in/w_out` | strip + count |
| `patch_merger.merging_layer.weight`, `pre_mm_projector_norm.weight` | strip + count |
| `processor_config.json`, `tekken.json` `image` block | not weights |

Every remaining tensor must classify; an unrecognized tensor is a hard converter
error. `input_global_scale` companions are classified (see 0.6).

### 0.5 Feasibility against THIS machine (RTX 5090 32 GiB VRAM; 750 GiB RAM)

| quantity | value |
|---|---|
| source on disk | 66 GiB (13 shards) |
| BF16 dense device-resident (attention + norms + gate + embed + output) | ~4.2 GiB |
| routed experts (host mmap store) | ~56 GiB (36×128×12 MiB packed + scales) |
| shared experts (NVFP4, resident) | ~0.5 GiB |
| KV cache @ 8192 ctx, FP16, latent `kv_lora(256)+rope(64)`/layer | ~0.2 GiB |
| converted on-disk (logestix layout) | ~62 GiB |
| free host RAM | 676 GiB |

Verdict: **fits.** Dense graph is ~5 GiB resident; the routed experts stream from
a host-mapped store through a device slot-LRU (`LSX_MISTRAL_EXPERT_CACHE_SLOTS`),
exactly the Alice/Xing/GigaChat pattern.

### 0.6 Semantics to verify before trusting any output

1. **Attention precision.** The checkpoint's own `quantization_config.ignore`
   list excludes `re:.*attention.*`, `re:.*self_attn.*`, `re:.*q_a_proj$`,
   `re:.*kv_a_proj_with_mqa$`, `re:.*gate$`, `lm_head`,
   `model.embed_tokens`. Observed: attention projections, `gate`, embeddings and
   `output` are **BF16**; only routed + shared experts are NVFP4. Decision:
   **keep BF16 as-is** (operator-approved) — converter copies them raw.
2. **Numerical DType of the expert scale codec.** `weight_scale` is
   `F8_E4M3 [out, in/16]` + `weight_global_scale` `F32[1]`. The repo's NVFP4
   layer already stores E4M3 per-16 block scales, so this is a **repack**, not a
   decode→requantize. Verify the repo decoder uses the *full* E4M3 value (not
   pow2-only) — if the repo's codec is the full-E4M3 one delivered by
   `2026-10-06-nvfp4-full-e4m3-scale-codec-design.md`, a byte passthrough is
   exact; otherwise fall back to the dequant→`quantize_tensor_gpu` path.
   `input_global_scale` `F32[1]` is an **activation** scale companion
   (`NVFP4A16`: weights 4-bit, activations 16-bit) — classify and drop (the repo
   runs BF16/FP16 activations), never fold into the weight.
3. **Yarn + `llama_4_scaling` (must be verified against mistral-common/vLLM
   before stage 5).** `yarn.apply_scale = false` means the standard Yarn
   attention-scale correction is **not** applied; the tables are almost
   certainly unscaled (cf. Xing, whose `attention_factor` upstream returns 1.0).
   `llama_4_scaling` (beta 0.1, orig 8192) is the Llama-4 long-context
   temperature applied to the attention logits for positions beyond the original
   window. Both must be read from the vendor reference before coding; a wrong
   guess here silently decodes garbage past 8192 tokens.
4. **MLA split.** Q output is 128/head (= `head_dim`, all of it rotary-capable
   but only the last `qk_rope=64` receive RoPE); KV-B output is 192/head
   (`qk_nope=64` + `v_head=128`); the rope key is the 64-wide slice carried by
   `wkv_a_with_mqa` alongside the 256-wide compressed kv latent. Confirm the
   interleaving order against `modeling`/vLLM before writing the kernel.
5. **Router.** `num_expert_groups = 1` → plain top-4. Confirm softmax vs sigmoid
   and whether the top-k weights are renormalised before the shared expert is
   added (`num_shared_experts = 1`, `routed_scale = 1.0`).
6. **No sink bias, no sliding window, no MTP layer** in the source index (only
   the vision tower is extra). Confirm by classification.

## Stage 1 — topology and structure passport (lsxforge)

New converter `libs/lsxcommon/src/mistral_moe_convert.{h,cpp}`, pipeline flag
`--mistral` (`apps/lsxforge`), reading the **Mistral consolidated layout**:

- geometry from `params.json` (not `config.json`);
- weights from `consolidated-*-of-00013.safetensors` via
  `consolidated.safetensors.index.json`;
- emit the standard logestix `logestix-mistral-nvfp4` layout (a `model.index.json`
  with the frozen geometry, `format` tag, and the routed/shared expert tables).

Tensor name taxonomy to classify (44 stems): `layers.N.attention.{wq_a,wq_b,
wkv_a_with_mqa,wkv_b,wo}`, `layers.N.attention.{q_a_norm,kv_a_norm}`,
`layers.N.{attention_norm,ffn_norm}`, `layers.N.gate`, `layers.N.experts.E.{w1,w2,w3}`,
`layers.N.shared_experts.{w1,w2,w3}`, `tok_embeddings`, `output`, `norm`,
plus the vision stems (stripped). Each expert projection carries
`{weight_packed, weight_scale, weight_global_scale, input_global_scale}`.

Converter test: `test_mistral_moe_convert` (label `convert`, TIMEOUT 120 s) on a
synthetic checkpoint built from `test_synth_checkpoint.h` +
`LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`; covers a MoE layer, BF16 attention, the
vision ballast strip+count, and the index layout. A real `params.json` may be
read only behind `SKIP()`.

## Stage 2 — quantization / repack

The experts are already NVFP4 in the repo's block layout, so stage 2 is a
**repack** (`mistral_moe_repack`, role `repack` — add the token to
`SchemeRegex()` if not present) that:
- merges the three per-expert projections into the repo w1/w2/w3 order;
- copies `weight_packed` bytes and `weight_scale` E4M3 values verbatim (only if
  0.6.2 confirms full-E4M3 decoding in the repo; otherwise dequantise +
  `quantize_tensor_gpu`);
- keeps `weight_global_scale` as the weight global scale and drops
  `input_global_scale`.
Norms, biases and `gate` are never quantized.

## Stage 3 — registration

`MistralSmall4MoeModel` (`kType = "mistral-small-4-119b"`), registered in
`model_registry.cpp`, implementing `Load/Infer/Unload/Inspect/SelfCheck/Tokenizer`.
Frozen geometry from `mistral_moe_constants.h` with `static_assert`, cross-checked
against the loaded index in `Load` (relaxed only under
`SyntheticGeometryAllowed()`). Expert host store +
`mistral_moe_expert_cache.{h,cpp}` + `LSX_MISTRAL_EXPERT_CACHE_SLOTS`.

## Stage 4 — tokenizer bridge

`BuildTokenizerArrowFiles` on the checkpoint's `tokenizer.json`; the index records
`bos=1`, `eos=2`, `pad` from config; `tie_word_embeddings=false`.

## Stage 4b — tool-calling bridge

Mistral native format (from `chat_template.jinja`), new
`mistral_moe_tool_format.{h,cpp}` (add stem to `AllowedCommon()`):

- render: `[INST] … [/INST]`, `[AVAILABLE_TOOLS]…[/AVAILABLE_TOOLS]`,
  `[SYSTEM_PROMPT]…[/SYSTEM_PROMPT]`, `[MODEL_SETTINGS]{"reasoning_effort":…}`
  `[/MODEL_SETTINGS]`, assistant tool calls as `[TOOL_CALLS]` + JSON array,
  results as `[TOOL_RESULTS]…[/TOOL_RESULTS]`, thinking as `[THINK]…[/THINK]`.
- parse: detect `[TOOL_CALLS]` (control id in `added_tokens`) and decode the JSON
  array into `ToolCall`; fall back to `tools::ParseToolCallsText` otherwise.
- Test (`fast`): render structure + parse round-trip on a synthetic tokenizer
  fixture in `tests/data/` with the markers in `added_tokens`.

## Stages 5–7 — kernels, sandbox, passport

New CUDA: `mistral_moe_kernels.{h,cu}` (RMSNorm / router / SwiGLU),
`mistral_moe_inference_attn.{h,cu}` (MLA: latent q/kv + inner norms + **Yarn
RoPE** + `llama_4_scaling` logit temperature + absorb or expand form + KV cache
via `KvCacheGeometry`), `mistral_moe_inference_pipeline.{h,cu}` (36-layer MoE
graph + generation loop wired to the shared lifecycle contract:
`RunUnderResidencyLock`, `GenerationStream`/abort, `TeardownContext`, governor
lease, filled `InferenceResult`). Routed MoE runs through the neutral
`lsx_moe_inference`; NVFP4 through neutral `lsx_nvfp4`; GEMV through `lsx_gemv`.
Isolated OBJECT libraries in CMake. Shared-memory sandbox + thread-0 reductions
per `nvfp4-blackwell-conventions`. Stage 7 is model-neutral infrastructure.

## Stage 8 — TDD and SelfCheck

| stage | test (label) |
|---|---|
| 1 converter | `test_mistral_moe_convert` (`convert`, TIMEOUT 120 s) |
| 2 repack | `test_mistral_moe_repack` (`fast`) |
| 3 load | `test_mistral_moe_loader` (`fast`, synthetic) |
| 3 model | `test_mistral_moe_model` + `test_mistral_moe_model_lifecycle` (`inference`) |
| 4b tool | `test_mistral_moe_tool_framing` (`fast`) |
| 5 kernels | `test_mistral_moe_attention`, `test_mistral_moe_rope_yarn`, `test_mistral_moe_moe`, `test_mistral_moe_inference` (`inference`) |
| 8 selfcheck | `test_mistral_moe_selfcheck` (`inference`, SKIP_RETURN_CODE 4) |
| 8 fsi/code/det | `test_mistral_moe_{fsi,code,det}_selfcheck` (`inference`, SKIP 4) |
| 8 perf | `test_perf_baseline_mistral_small_4` + `perf_baselines.json` entry (`perf`, SKIP 4) |

## Definition of done

See `ingesting-a-model`. All four guard registries updated in the same commit as
the files they police; `ctest -L convert|fast|inference|perf` run per stage.
