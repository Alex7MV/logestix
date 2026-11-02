# Mistral-Large-4-1T-A52B ingestion — design

Date: 2026-11-02
Model card: `Mistral-Large-4-1T-A52B` ("Le Chonk")
Source tree: `/mnt/fast_storage/huggingface/Mistral-Large-4-1T-A52B` (988 GB)
Reference implementation: `transformers/models/mistral_large4/{configuration,modeling}_mistral_large4.py`
(upstream `main`; the checkpoint declares `transformers_version 5.21.0.dev0`; the
installed 5.15.1 only ships `mistral4` = the Small-4 sibling).

## 1. Classification (Stage 0)

- Family `mistral` (existing), **new variant** `large4`: geometry, attention
  topology and tokenizer pre-tokenizer all differ from `Mistral-Small-4-119B`,
  so it gets its own frozen passport header and namespace.
  - variant token: `large4` (file names `mistral_moe_large4_*`)
  - namespace: `lsxcommon::mistral_large4` (+ `::frozen` for the passport)
  - `kType = "mistral-large-4-1t"`, index `"model": "Mistral-Large-4-1T"`
  - format tag: `logestix-mistral-large4-nvfp4`
- Not a sibling reusing the Small-4 loader: no Q-LoRA, fused per-head rope
  `wkv_b`, NoPE layers, DSA indexer, cross-layer KV sharing, LeRoPE,
  `ssmax_scaling`, logit softcap, dense layer 0, different vocab.

## 2. Verified geometry

Text config (`config.json` -> `text_config`):

| field | value |
|---|---|
| hidden_size | 6144 |
| num_hidden_layers | 72 |
| num_attention_heads / kv heads | 64 / 64 (MLA, MHA over the latent) |
| kv_lora_rank | 512 |
| qk_nope_head_dim / qk_rope_head_dim | 64 / 64 (qk_head_dim 128) |
| v_head_dim | 128 |
| q_lora_rank | none (direct `wq`, 64x128 = 8192 rows) |
| intermediate_size (dense MLP) | 24576 |
| moe_intermediate_size | 6144 |
| n_routed_experts / top-k | 128 / 4 |
| n_shared_experts | 1 (gated SwiGLU, width 6144) |
| first_k_dense_replace | 1 (layer 0 dense, 1..71 MoE) |
| n_group / topk_group | 1 / 1 |
| routed_scaling_factor | 1.0 |
| vocab_size | 204800 |
| rms_norm_eps | 1e-5 |
| final_logit_softcapping | 30.0 |
| max_position_embeddings | 1048576 |
| index_topk / index_head_dim / index_n_heads | 2048 / 128 / 32 |
| rope | YaRN theta 1e4, factor 2048, original window 512, beta_fast 32, beta_slow 1, attention_factor 1.0, interleaved pairs + LeRoPE |

Per-layer tables (from `config.json`, cross-checked against the tensor
inventory):

- **RoPE layers** (`no_rope_layers[i] == 1`, i.e. `i % 4 != 3`): 54 layers.
  `wkv_a_with_mqa` is `[576, 6144]` = 512 latent + 64 decoupled `k_pe`;
  `wkv_b` is `[12288, 512]` = 64 heads x (nope 64 + v 128).
- **NoPE layers** (`no_rope_layers[i] == 0`, `i % 4 == 3`): 18 layers
  (3,7,...,71). No rotary anywhere in that layer (attention *and* indexer).
  `wkv_a_with_mqa` is `[512, 6144]`; `wkv_b` is `[16384, 512]` = 64 heads x
  (128 + v 128) — the 64 rope dims become plain key dims.
- **KV sharing** (`kv_source_layers`): layers 39,43,47,51,55,59,63,67,71 have
  no `kv_a`/`kv_a_norm` tensors and reuse layer **35**'s latent cache (all
  sharers are NoPE, as is the source). 63 layers own a cache.
- **DSA indexer** (`indexer_types`): 44 `full` layers run their own indexer;
  28 `shared` layers (17,18,21,22,...,69,70) have no indexer tensors and reuse
  the previous full layer's top-k indices.
- **MLP**: layer 0 dense (`feed_forward.w1/w2/w3`, I=24576); layers 1..71 MoE
  (`gate` BF16, `experts.E.w1/w2/w3`, `shared_experts.w1/w2/w3`).

Tensor dtype on disk: FP8-E4M3 with an F32 per-128x128 tile scale
(`qscale_weight`, a **multiplier** — verified: dequantized `wq` RMS 0.011,
absmax 0.078; the divisor reading gives RMS 7.2e5). Exceptions stay BF16:
`tok_embeddings`, `output`, `norm`, all norms, `wkv_a_with_mqa`, `wkv_b`,
`gate`, `indexer.wk`, `indexer.weights_proj`, `ssmax_scaling`, `lerope.alpha`.

## 3. Strip + count

- `vision_encoder.*` (pixtral, 30 layers), `vision_language_adapter.*`,
  `patch_merger.*`, `pre_mm_projector_norm.weight`.
- `mtp_dspark.*` (3-layer DSpark draft head + autoregressive head).
- Both recorded in the index (`skipped_tensors`, `deferred.mtp`), never a
  silent drop. Unknown text tensors are a hard conversion error.

## 4. Attention semantics (must match the reference exactly)

Per layer, input `x` after `input_layernorm` (plain RMSNorm, weight multiply):

1. `q = wq x` -> 64 heads x 128. RoPE layer: split `[nope 64 | rope 64]`.
   NoPE layer: no split, no rotary.
2. RoPE layer: `kv = wkv_a x` -> `[latent 512 | k_pe 64]`;
   `latent = rmsnorm(kv[:512])`; `k_pe = kv[512:]`.
   NoPE layer: `latent = rmsnorm(wkv_a x)`; no `k_pe`.
   Shared-KV layer: skip this step; reuse source layer 35's cached latent.
3. `kv_b = wkv_b latent` -> per head `[k_nope | v]` (nope 64/128, v 128).
4. RoPE layer: interleaved-pair rotation on `q_rope` and `k_pe`.
   `LeRoPE`: `inv_freq = yarn_inv_freq * exp(lerope.alpha)` (alpha length 32).
   YaRN NTK-by-parts (dim 64, theta 1e4, factor 2048, orig 512, beta 32/1),
   no magnitude scaling (`attention_factor = 1.0`).
5. Key per head = `[k_nope | k_pe]` (128). Value = `v` (128).
6. **ssmax scaling applied to q before QK**: per head
   `q[h] *= s[h] * log(pos + 1) + b[h]` (`attention.ssmax_scaling.s/.b`,
   BF16 math, absolute position). Then `logits *= 1/sqrt(128)`.
7. Attention is causal; beyond `index_topk = 2048` positions only the
   indexer's top-k keys are kept (below/at 2048 every key is kept, so the
   canonical gates exercise dense causal attention).
8. `out = concat_h(attn_h) @ wo`; residual add.

Indexer (full layers only), on the same normalized `x`:

1. `qi = q_input_norm(x) @ wq_b` -> 32 heads x 128.
2. `qi[h] *= s_i[h] * log(pos + 1) + b_i[h]` (indexer ssmax).
3. `k = k_norm(wk x)` -> one 128-d head.
4. RoPE layer: interleaved rotation on the first 64 dims of qi and k.
5. `score[h] = relu(qi[h] . k * 128^-0.5)`;
   `w = weights_proj(x) * 32^-0.5`;
   `index_score = sum_h w[h] * score[h]`; causal mask; top-k (k-th largest,
   index tie-break ascending deterministic).
6. Shared-indexer layers reuse the previous full layer's indices.

## 5. Decode architecture

- **Latent MLA cache** (matching the reference's "cache while compressed"
  rule): per cache-owning layer, `[capacity][512]` BF16 latent plus
  `[capacity][64]` BF16 `k_pe` on RoPE layers. 63 caches => 149 MB @2048.
- **Absorbed attention** (mathematically identical to the reference's
  `expand_kv`, reordered): cache `ctx_h = sum_t p_t * latent_t` (512 per head)
  and apply `W_v,h` (128x512) once per step; scores use
  `q_abs_h = q_nope_h @ W_k,h` (512) plus `q_rope_h . k_pe_t`.
  This is the only way a 1 M-token context and a 32 GB card coexist; the
  KV 8 GB expanded layout is rejected.
- **Indexer key cache**: `[capacity][128]` BF16 per full indexer layer
  (44 caches => 23 MB @2048).
- **DSA selection**: per query, `index_score` over `[0, pos]` in one block;
  when `pos + 1 > 2048` an exact threshold-select produces the top-2048
  index list (compacted ascending), and the attention kernel iterates the list
  (gather) instead of the full prefix. Below that, the dense path runs and the
  indexer is skipped (identical result).
- **KV-shared layers** resolve their source cache through
  `kv_source_layers`; the pipeline must expose per-layer `kv_source` to the
  attention module (a layer with no cache reads layer 35's).

## 6. Numeric policy

- Keep BF16: embeddings, lm_head, all norms, `wkv_a`, `wkv_b`, router,
  `indexer.wk`, `indexer.weights_proj`, `ssmax_scaling`, `lerope.alpha`.
- NVFP4: `wq`, `wo`, `indexer.wq_b`, routed experts, shared expert, dense
  layer-0 MLP. (Mirrors the source's `modules_to_not_convert` list.)
- Activation/residual stream FP32, BF16 projections through the neutral GEMV,
  NVFP4 through `lsx_nvfp4` / the neutral MoE layer.
- Router logits in F32, softmax + group top-2 + renormalize (always) x 1.0.
- Final logits: `tanh(z / 30) * 30`.
- Deterministic reductions; no atomics feeding `final_logits`.

## 7. Footprint (measured from the inventory)

| item | size |
|---|---|
| dense + shared NVFP4 | ~9.4 GB (576 MB scales included) |
| BF16 edges (embed+head 5.0, wkv_a 0.44, wkv_b 0.98, router 0.11, norms) | ~6.6 GB |
| pinned total | **~16 GB** |
| routed experts (71 x 128 x 3 x 6144x6144) NVFP4 | **~547 GB** (mmap store) |
| latent KV + indexer cache @2048 | ~172 MB |
| projectors | 1.6 GB @2048 (at 1 M ctx: 84 GB -> not resident) |

Expert store: neutral `ExpertStore` (mmap views + LRU), `max_open` knob
`LSX_MISTRAL_LARGE4_MAX_OPEN` (default 4096), slot cache sized from free VRAM
minus a 2 GiB reserve (~200 slots of ~86 MB each with scales).

## 8. Files

| file | role |
|---|---|
| `libs/lsxcommon/src/mistral_moe_large4_constants.h` | frozen passport (`lsxcommon::mistral_large4::frozen`), per-layer tables, special ids |
| `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_convert.h` | converter + topology declarations |
| `libs/lsxcommon/src/mistral_moe_large4_convert.cpp` | config/consolidated-index parse, FP8->NVFP4, expert split, index+tokenizer |
| `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_load.h` | pinned-weight struct + loader |
| `libs/lsxcommon/src/mistral_moe_large4_load.cpp` | pinned graph upload (`PinnedSchema`/`DenseNvfp4Schema` views), expert store |
| `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_kernels.h` | kernel-level exports (ssmax, LeRoPE, indexer, router, absorb) |
| `libs/lsxcommon/src/mistral_moe_large4_kernels.cu` | elementwise/rope/router/indexer/absorb kernels |
| `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_inference_attn.h` | MLA attention module + caches |
| `libs/lsxcommon/src/mistral_moe_large4_inference_attn.cu` | latent cache, absorbed decode attention, DSA gather path, KV sharing |
| `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_inference_pipeline.h` | session + decode graph |
| `libs/lsxcommon/src/mistral_moe_large4_inference_pipeline.cu` | 72-layer driver, generation loop, streaming/abort |
| `libs/lsxcommon/src/mistral_moe_large4_model.{h,cpp}` | `IModel`, factory registration, tool bridge, selfcheck |
| `apps/lsxforge` | `--mistral_large4` flag |
| `libs/lsxcommon/src/tokenizer_convert.cpp` (+ `pretokenizer.{h,cpp}`, `converter.h`) | new pre-tokenizer kind 12 (o200k case runs, right-grouped digits, `[\r\n/]*` tail) |
| `libs/lsxcommon/src/mistral_moe_tool_format.cpp` | `[MODEL_SETTINGS]` / `reasoning_effort` render policy |

Tests: `test_mistral_moe_large4_{convert,fixture.h,loader,model,kernels,selfcheck,fsi_selfcheck,code_selfcheck,det_selfcheck,tool_framing,perf}`.

Guard registries in the same commit as each stage: `test_code_naming_guard`
(add `large4`), `test_model_independence_guard` (`kPassports`, `consumers`,
`kLifecycle`), `test_model_suite_guard` (`Suites` row + tool verdict + golden
debt), ctest labels, `test_expert_store_guard` (must stay out of
`kLegacyConsumers`).

## 9. Gates / evidence

- SelfCheck: `"The capital of France is"` -> `Paris`, 3 cycles bit-exact.
- FSI / CODE / DET: goldens are **template-derived** (the 1 T FP8 source is a
  multi-node FP8 runner; it cannot execute on this 32 GB card). Declared as
  `kGoldenDebt` with the reason, then run against the converted tree.
  A template-derived golden is recorded as such, never as source parity.
- Tool round-trip: `get_weather {"city":"Paris"}` through the native
  `[TOOL_CALLS]name[ARGS]{json}` codec.
- Perf: canonical 2048-token protocol through the harness
  (`tests/perf/perf_baselines.json` + `TEST_CASE` + `add_test`).

## 10. Risks / open items

- Conversion of 1 T params and a 2048-token perf row are long operations
  (hours); they run on `/mnt/fast_storage/tmp` for scratch.
- Shared-KV + shared-indexer state is the most intricate new code; unit tests
  pin the state machine (cache ownership, source resolution, mask reuse).
- >2048-token DSA sparsity is implemented; the canonical gates cannot exercise
  it, so it gets kernel-level unit tests (masked gather equivalence to a host
  reference) and a documented long-context check.
- Golden parity discipline: no source oracle exists for this tree; goldens are
  template-derived and declared.
