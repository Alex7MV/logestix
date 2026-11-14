# Inkling-2-Small ingestion — stage 0 triage and plan

**Status:** in progress (branch `feat/inkling-2-small`). This file is the stage-0
record required by `ingesting-a-model` (§ Stage 0) and the working plan; the
closing artifact is `docs/model-reports/2026-11-14-Inkling-2-Small.md`.

Source checkpoint: `/mnt/fast_storage/huggingface/Inkling-2-Small`
(`thinkingmachines/Inkling-2-Small`, modelopt NVFP4 release, 24 shards +
`mtp.safetensors`, 367 GB). Converted output: `<vault>/logestix/` (the
`path_registry` root; never anywhere else).

## 0.1 Classification

**New family.** `model_type: inkling2`, architecture `Inkling2ForConditionalGeneration`.
No existing `models/*.json` row shares the geometry (`ling` is Bailing/KDA+MLA,
`kimi-k3` is KDA+MLA, neither is this). The nearest *structural* neighbour is
Kimi-K3 (AttnRes depth mixer, tiktoken BPE, sized MoE, host expert store) and it
is used as the code skeleton only — no geometry, passport or fixture is borrowed
(`test_model_independence_guard`).

Reference semantics: **`transformers` 5.23.0 `models/inkling2/modeling_inkling2.py`**
(the authoritative port; `transformers` 5.15.1 only ships the earlier `inkling`
family, which is a *different* architecture — see 0.6 #6).

## 0.2 Geometry (read from config.json + safetensors headers)

`text_config` (the whole engine; vision + MTP are ballast):

| field | value | note |
|---|---|---|
| hidden_size | 4608 | |
| num_hidden_layers | 48 | all MoE |
| num_attention_heads / kv | 48 / 8 | GQA group 6 |
| head_dim | 128 | full and sliding identical |
| vocab_size / unpadded | 201024 / 200058 | logits sliced to 200058 |
| moe_intermediate_size | 2304 | arrives as `intermediate_size` (`attribute_map`) |
| n_routed_experts / top-k | 384 / 8 | |
| n_shared_experts | 0 | Inkling2 has no shared experts at all |
| route_scale | 8.0 | `routed_scaling_factor` |
| scoring_func | sigmoid | renormalised over the top-k |
| use_gate_bias | true | `mlp.gate.bias`, selection only, FP32 |
| d_rel / rel_extent | 16 / 1024 | learned relative bias bank |
| sliding_window_size | 1024 | 40 sliding layers |
| qk_l2_clip / rel_l2_clip | 22.627417 (=16√2) / 8.0 | learned per-element scales |
| log_scaling_n_floor / alpha | 128000 / 0.1 | full layers only |
| mlp_gate_softcap / up | 4.0 / 25.0 | softcapped SwiGLU |
| final_logit_softcapping | 30.0 | |
| rms_norm_eps | 1e-6 | |
| sconv_kernel_size | 4 | 4 conv states per layer |
| n_layers_per_attn_res_block | 6 | 8 blocks |
| max_position_embeddings | 1048576 | engine default context is far below this |

**Hybrid pattern.** `local_layer_ids` lists 40 layers ⇒ the remaining 8 —
`{5,11,17,23,29,35,41,47}` — are the full-attention layers. They are exactly the
`(layer+1) % 6 == 0` set, the AttnRes block boundaries are `layer % 6 == 0`
(`{0,6,…,42}`), and the same 8 layers are the MTP targets. The two roles must not
be conflated: boundaries gate the *depth mixer*, the 8 full layers gate
*window/log-scaling/dynamic-conv/3-query mixing*.

**AttnRes tensor census** (this is the item that silently decodes garbage):
- `attn_res.mlp_queries.{0..47}` — 48 vectors `[4608]`, one per layer.
- `attn_res.attn_queries.{0..46}` — 47 entries: `[4608]` on sliding layers and
  `[3,4608]` on the 8 full layers. **Indexed by `layer-1`** (layer 0 has no
  attention mixture; `Inkling2*DecoderLayer.attention_input is None`), so
  `attn_queries.46` is layer 47.
- `attn_res.output_query` — `[4608]`, the `final_input` mixture before `norm`.

## 0.3 Tokenizer

Byte-level BPE in the *tiktoken* dialect: `tiktoken/tokenizer.model` (mergeable
ranks) + `tokenizer.json` with `ignore_merges: true` and the o200k `Split`
pre-tokenizer; `added_tokens` = 60, all `special: true`; base vocab 199998,
special band 199998..200057, padded to 201024. `bos` does not exist
(`bos_token_id: null`), `eos = 200006` (`<|end_message|>`-adjacent id from the
config; the checkpoint's own `eos_token_id`). Kind = `2` (the Kimi tiktoken
class), built with `BuildTokenizerArrowFilesTikToken`.

Chat/tool framing comes from `chat_template.jinja` (stage 4b): roles
`<|message_user|>/<|message_model|>/<|message_system|>/<|message_tool|>`,
content wrappers `<|content_text|>`, `<|content_thinking|>`,
`<|content_invoke_tool_json|>`, terminator `<|end_message|>`, end-of-sampling
`<|content_model_end_sampling|>`, tool declaration
`<|message_system|>tool_declare<|content_xml|>{json}<|end_message|>`, tool call
payload `{"name":…,"args":{…}}`, and a mandatory
`<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>`
line before the first non-system message.

## 0.4 Ballast to strip (stored and counted, never applied)

- `model.visual.*` — 422 tensors / 0.33 B params (ViT + merger).
- `mtp.safetensors` — 5-layer MTP head with a `dflare_norm`/`k_norm_ctx`
  variant of the block (not in `model.safetensors.index.json`).
- `mtp_config` in `config.json`, `mtp` block in `hf_quant_config.json`.
Everything else must be classified; an unknown tensor is a converter error.

## 0.5 Feasibility on this box (RTX 5090 32 GB, 750 GB host, 2.0 TB free)

| plane | params | bytes |
|---|---|---|
| dense (embed 926M + unembed 926M + attention/router/attn_res 6.43B) | 8.28 B | ~4.7 GB at NVFP4 |
| routed experts, layers 0–39 (NVFP4 in the checkpoint) | 490 B | ~262 GB on disk, pass-through |
| routed experts, layers 40–47 (MXFP8 in the checkpoint) | 98 B | ~98 GB on disk |
| KV at 2048 tokens (8 full layers × 8 kv heads × 128 × 2 × FP16) | — | 34 MB |
| conv states (4 per layer) + block-state stack (≤8 × 4608 FP32) | — | < 1 MB |

Active read per decoded token: 8 experts/layer ⇒ ~5.1 GB (NVFP4 layers) +
~2.0 GB (MXFP8 layers) ≈ **7.1 GB/token**, dominated by the host expert store.
That is Kimi-K3 territory (0.47 t/s measured) and is the expected perf band
(0.5–2 t/s), with a TTFT dominated by the first-touch page-in of the routed tree.
`ulimit -l` is already unlimited; the store is `mmap` + LRU (`ExpertStore`), not
a pinned arena (R1/R2 in `ingesting-a-model`).

## 0.6 Semantics verified before trusting output

1. **Expert codec, NVFP4 layers.** `w13 = [E][2I=4608][H/2 bytes]`,
   `w2 = [E][H=4608][I/2 bytes]`; per-16 `F8_E4M3` block scales along the packed
   (input) axis; per-expert `amax[E]` and a scalar
   `outer_scale_denominator = 2688 = 6·448`. **Verified numerically**: for six
   experts of layer 0, `6 · max(block_scale) · (amax_e/2688) == amax_e` to
   1e-9 relative. So `w = q_e2m1 · s_e4m3 · g_e`, `g_e = amax_e/2688` — the same
   E2M1 + per-16 E4M3 + float-global-scale codec `lsx_nvfp4` already implements,
   with the global scale carried **per expert** instead of per tensor.
2. **Expert codec, MXFP8 layers (40–47).** `F8_E4M3` weights, `ue8m0` scale per
   32 values along the input axis (`[E][4608][144]`), no amax. See "open
   decision" below.
3. **Attention scale is `head_dim**-0.5`** in this family (Inkling-1 used
   `1/head_dim`) — the single most likely silent-parity defect.
4. **L2 clips are learned and applied before the scale**: `q·scale` then clip to
   ‖·‖≤22.627 (element-wise learned scale, per head×channel), `k` clipped
   weight-free, `rel` with its own `[heads,16]` scale to 8.0; `v` gets a
   **weight-free** RMSNorm. Order: clip → (full layers) log-scale → cache.
5. **Relative bias** is a gather, not a rotation:
   `bias[h,d] = (r_h · proj[:,d])` for distance `d = qpos-kpos`, zero outside
   `0 ≤ d < rel_extent`; sliding layers use `rel_extent = 1024` (the window),
   full layers `rel_extent = 1024` from the config. No RoPE anywhere in this
   model — Q/K are normalized, not rotated.
6. **Log scaling is full-layer only**: `τ = 1 + 0.1·ln(max((pos+1)/128000, 1))`,
   applied to `q` (after clip) and to the *relative states* (before the bias
   gather), FP32.
7. **Depth mixer** (AttnRes): stack = `[block_states (n_blocks), current]`,
   unweighted RMSNorm per row, then single-head SDPA with a *learned fixed
   query* (scale `H**-0.5`); number of outputs = 1 (sliding/mlp) or 3 (full-layer
   attention → separate q/k/v inputs, each with its own RMSNorm weight).
   Block boundary (`layer%6==0`) appends the *incoming* hidden state to the
   stack and drops the local residual; every other layer keeps
   `residual = block_residual + attn_out`.
8. **Convolutions**: post-attention and post-MLP short convs are
   `(conv(x) + x)/√(k+1) = /√5`; sliding k/v convs use the same `/√5`; the
   full-layer k/v convs are `static[tap,channel] + per-token per-head tabs`
   normalised by `/√(2k+1) = /√9`. `tap 0 = current token` (backwards).
9. **Routing**: `σ(logits) + bias` selects top-8, weights = `σ(logits)[top8]`
   renormalised to 1 then × 8.0; experts accumulate in FP32.
10. **Head**: embed → weighted RMSNorm (`embed_norm`), … , `final_input` mix →
    weighted RMSNorm (`norm`) → unembed FP32 → `tanh(x/30)·30` → slice 200058.
11. **MTP is not applied**; the vision tower is not applied.

**Open decision (resolved 2026-11-14):** the 8 MXFP8 layers are **requantized to
NVFP4 at conversion** (option b). Option (a) — passing the FP8 plane through and
adding a second expert codec to the MoE path — was rejected because (i) the
engine's expert store schema (`ExpertSchema`) is a uniform 9-column NVFP4 record,
(ii) it would double the GEMV dispatch surface for 1/6 of the stack, and (iii)
the residual-plane recipe of stage 2 is available if a parity gate later shows
the 4-bit rounding on those layers flipping an answer. The measurement to
record: relL2 of the requantized layer-40..47 experts against their dequantized
FP8 source (to be taken when the engine can decode them).

## Real-weight status: the engine runs, parity is RED (2026-11-14)

`test_inkling_moe_selfcheck` loads the converted 594B tree on the RTX 5090,
frames `"The capital of France is"` through the checkpoint's own chat template,
decodes and returns. It does **not** answer Paris:

```
cycle 0 text: aterinagjernaterinaskar悆inversheavFactor prison𨯂挦 fem sifat
```

Incoherent multibyte output means the forward pass is wrong somewhere, not that
the weights are unusable. This is the "answers, but not the model's answers"
class the ingestion protocol warns about, and it is recorded as a **red gate**,
not a pass. The gate itself is in the tree and fails when run.

Two real defects were found and fixed on the way to this point (both recorded in
the source where they bite):

1. **Relative-position states aliased onto the convolution scratch.** `w_r` was
   projected into `ws.conv_out`, which the k/v short convolutions then
   overwrite, so the position bias read a convolution result. Fixed: the
   relative states have their own buffer.
2. **The recurrent conv state was never reset between requests**, so a second
   identical request decoded differently. Fixed: every request starts with all
   four convolutions per layer zeroed.

Also fixed while building the layer body: the post-MLP conv state used offset
`3*k*kv_dim`, overlapping the post-attention state whenever `kv_dim < hidden`
(correct base is `2*k*kv_dim + k*hidden`); and a static-conv layer passing its
taps for both the `w` and `dyn_taps` arguments silently ran the lag convention
on every sliding layer (the kernel picks the order from `dyn_taps == nullptr`).

**Next checks, in the order of "cheapest to falsify" — none of them done yet:**

- N1. Teacher-force a 2-token prefix and compare the engine's `final_logits`
  against a `transformers` 5.23 reference run on the same ids. This is the only
  way to localize the defect to a layer instead of guessing at the whole stack.
- N2. Zero all routed experts and compare: if the output becomes *partly*
  coherent the expert path is at fault; if it stays garbage the defect is
  upstream of the MoE.
- N3. Repeat with only layer 0 executed (early exit after one layer, comparing
  hidden states against the reference) — the per-layer plumbing is already
  exercised by the synthetic model test, so a divergence there points at the
  weight plane rather than the code.
- N4. Re-verify the two codec invariants on the real artifact: decode one
  expert record on the host and compare against the source safetensors slice
  (`base relL2` ~0.09 for NVFP4 is expected), and check the per-expert global
  scale is applied exactly once in the batch GEMV.
- N5. The `mlp.gate.bias` tensor: the reference adds it to the sigmoid score for
  SELECTION only. Verify the shipped values are not being added to the weights.

## Registry status: PENDING ingestion (no manifest row yet)

`models/inkling-2-small.json` was written when the row looked complete and was
then **withdrawn**, because `test_model_suite_guard` forbids a suite row whose
nine stages do not all exist and the engine-dependent four
(`selfcheck`/`fsi`/`code`/`det`) cannot exist before the decode engine does.
The repo's sanctioned intermediate state for a multi-stage ingestion is the
opposite order: register the targets, declare them in
`test_model_suite_guard.cpp::kPendingSuiteTargets`, and add the manifest row in
the commit that completes the last stage.

Current state, machine-enforced from both directions:

- `tests/CMakeLists.txt` registers the four targets by hand (labels `fast`,
  `convert`, `inference` x2; bounded `TIMEOUT`s; `RESOURCE_LOCK "gpu"`;
  `SKIP_RETURN_CODE 4` where a vault may be absent).
- `test_model_suite_guard` lists `test_inkling_moe_{convert,loader,kernels}` in
  `kPendingSuiteTargets`; the guard fails if a pending name stops being a
  registered unclaimed target **or** if a suite row claims it — so the entry
  cannot outlive the problem.
- `tools/gen_model_manifest.py::_NON_MODEL_TESTS` claims the four test files so
  the `--strict` orphan scan stays clean, with the same delete-on-row comment.
- Because there is no row, `LsxModelFactory` cannot resolve `inkling-2-small`
  and `docs/MODELS.md` (38 rows) does not list it. That is correct: the model is
  not servable.

## Plan (TDD, one stage per commit)

| # | deliverable | test (label) | state |
|---|---|---|---|
| 1 | `inkling_moe_constants.h` passport, naming-guard family token, `models/inkling-2-small.json` | `test_inkling_moe_constants` (fast) | **done, green** (186 assertions) |
| 2 | `inkling_moe_convert.cpp` + `--inkling` in `apps/lsxforge` | `test_inkling_moe_convert` (convert, synthetic) | **done, green** (349 assertions); real tree converted (312 GB, 18 min) |
| 3 | topology reader + artifact layout (`inkling_moe_load.*`) | `test_inkling_moe_loader` (load, synthetic + real index) | **done, green** (topology half); the weight upload half is stage 5 work |
| 4 | `inkling_moe_kernels.cu` (attn-res mix, L2 clip, rel bias, sink, convs, softcapped SwiGLU, router) | `test_inkling_moe_kernels` (kernels) | **done, green** (1524 assertions, 14 host-reference cases) |
| 5 | weight upload (`inkling_moe_load.cpp`) + `inkling_moe_inference_pipeline.cu` | `test_inkling_moe_model` (model) | **done, green** (439 assertions): the pinned plane, the neutral expert store, the KV session, the decode scratch arena and the full 12-step layer body with the 8-expert staging and 3 batch GEMVs per layer |
| 6 | `inkling_moe_tool_format.cpp` (stage 4b, native `chat_template.jinja`) | `test_inkling_moe_tool_framing` (fast) | **done, green** (49 assertions) |
| 7 | `SelfCheck` (Paris), goldens | `test_inkling_moe_selfcheck`, `*_fsi/code/det_selfcheck` | **RED on real weights** (see below); FSI/CODE/DET not written |
| 8 | perf row | `test_perf_baseline_inkling_2_small` (perf) | **open** (declared debt) |
| 9 | report | `docs/model-reports/2026-11-14-Inkling-2-Small.md` | **open** — the stage-0/1/2 evidence is in this file and in `docs/OPEN_TASKS.md` §R19 |

