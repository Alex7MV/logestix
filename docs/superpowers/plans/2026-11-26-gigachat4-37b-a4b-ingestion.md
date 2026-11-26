# GigaChat4-37B-A4B ingestion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (the operator's `AGENTS.md` forbids subagents, so execution is native/inline). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest `ai-sage/GigaChat4-37B-A4B` into logestix through every Model Ingestion Protocol stage, ending with selfcheck/parity/perf gates measured on real weights and a model report.

**Architecture:** a new manifest row (`gigachat4-37b-a4b`) in the existing `gigachat` family: own frozen passport, converter, loader and `IModel` class; the family's attention/norm/pipeline device TUs are shared and parameterised by an explicit variant selector (the `inkling_moe_varloop` precedent), plus one new TU for the DSA indexer.

**Tech Stack:** C++20 / CUDA 13.5 (SM 12.0), CMake presets, Arrow + NVFP4 codec, Catch2, ctest.

**Spec:** `docs/superpowers/specs/2026-11-26-gigachat4-37b-a4b-ingest-design.md`

## Global Constraints

- Blackwell only (CC 12.0/12.1). Never add pre-Blackwell codegen paths.
- TDD: no production code without a failing test first, and a new test gets its ctest label in the same commit.
- Logging only through spdlog / `lsxcommon::log`; no `std::cout`, no `printf` in `libs/` or `apps/`.
- `absl::StrFormat` is printf-style: `%s`/`%d`, never `{}`.
- Headers reachable from `.cu`/`.cuh` must not include `<absl/container/*>`; use `std::` there.
- Arrow reads are mmap-only (`MemoryMappedFile`), writes atomic; retained payloads come from view readers (`LoadTensorBatchViews`, `ExpertStore::Load`), never `UnpackBatch`.
- Routed experts go through the neutral `ExpertStore` (`GcExpertStore` already exists) — no family-private owning container.
- One GPU workload at a time; `test_ctest_labels_guard` owns the `RESOURCE_LOCK "gpu"` rule; skip-capable targets set `SKIP_RETURN_CODE 4`.
- Frozen geometry: compile-time constants + `static_assert`, and `Load` must fail on a passport mismatch. Runtime `config.json` reads are not allowed in the model path.
- Every model target registration comes from `models/<model_type>.json`; never hand-edit `tests/CMakeLists.txt` or a guard registry for a model target.
- NVIDIA docs are read from `/home/alexmv2025/projects/cuda/docs` (the MCP dump) before touching NVFP4/CUDA kernel semantics.
- Working directory for scratch: `/mnt/fast_storage/tmp`. Vault: `/mnt/fast_storage/huggingface`.

## Review Focus

- A prompt containing `<think>`/`</think>` or `｜GCML｜` as *text* must survive framing and not be re-interpreted as a control marker.
- `pos == 2047 → 2048` is the DSA boundary: the first token whose selection is non-trivial must attend to exactly the same 2048 keys the reference picks, and the count must never be 0 or > kv_len.
- `final_logit_softcapping` must saturate (|logit| < 30) and must be applied *before* sampling, so a captured golden never compares a pre-softcap logit.
- A 4096-token session must not read past the rope/indexer tables (capacity is a runtime init parameter, not the model's 1M `max_position_embeddings`).
- A legacy GigaChat3.5 artifact (no indexer tensors, `1+w` norms) must keep loading after the variant refactor: the new loader paths must be additive, and `gigachat3.5-432b-a28b-reasoning`'s recorded `logits_hash` must not move.

---

### Task 1: Naming vocabulary + frozen passport

**Files:**
- Modify: `tests/test_code_naming_guard.cpp` (`SchemeRegex()` variant + subsystem lists, one new `TEST_CASE`)
- Create: `libs/lsxcommon/src/gigachat_moe_37b_constants.h`
- Modify: `tests/test_model_independence_guard.cpp` (passport-owner map row)

**Interfaces:**
- Consumes: nothing.
- Produces: `namespace lsxcommon::gigachat4::frozen` with `kType = "gigachat4-37b-a4b"`, `kFormat = "logestix-gigachat4-nvfp4"`, `kHidden=2560`, `kLayers=32`, `kVocab=128256`, `kFirstKDense=1`, `kRmsEps=1e-6f`, `kNormGatingWeight=2.0f`, `kGateRank=16`, `kLogitSoftcap=30.0f`, `kFullAttnLayers[8]={3,7,11,15,19,23,27,31}`, `kNumFullAttn=8`, `kMtpLayer=32`, `kHeads=32`, `kKVHeads=32`, `kQkNopeHeadDim=128`, `kQkRopeHeadDim=64`, `kQkHeadDim=192`, `kVHeadDim=128`, `kQLoraRank=1536`, `kKvLoraRank=512`, `kRopeTheta=100000.0f`, `kYarnFactor=32.0`, `kYarnOrig=32768.0`, `kYarnBetaFast=32.0`, `kYarnBetaSlow=1.0`, `kYarnMscale=1.0`, `kYarnMscaleAllDim=1.0`, `kAttnScale=0.13086079996295008f`, `kLinearNumKeyHeads=16`, `kLinearNumValueHeads=32`, `kLinearKvRepeat=2`, `kLinearKeyHeadDim=128`, `kLinearValueHeadDim=128`, `kLinearConvKernel=4`, `kLinearKeyDim=2048`, `kLinearValueDim=4096`, `kLinearConvDim=8192`, `kLinearQkvzDim=12288`, `kLinearBaDim=64`, `kLinearSigmoidGateScale=2.0f`, `kLinearONormEps=1e-6f`, `kNumRouted=192`, `kTopK=8`, `kMoeIntermediate=768`, `kSharedExperts=1`, `kSharedIntermediate=768`, `kDenseIntermediate=8192`, `kRoutedScale=2.5f`, `kSwigluLimit=10.0f`, `kIndexHeads=64`, `kIndexHeadDim=128`, `kIndexTopk=2048`, `kIndexRope=64`, `kIndexWeightScale=0.125f`, `kIndexScoreScale=0.08838834764831845f`, `kBosId=1`, `kEosId=2`, `kPadId=2`, `kRoleSepId=128000`, `kMessageSepId=128001`, `kThinkId=128014`, `kThinkEndId=128015`.

- [ ] **Step 1: Write the failing test** — add `TEST_CASE("naming guard accepts gigachat_moe_37b_* stems", "[naming][guard]")` asserting `std::regex_match` for `gigachat_moe_37b_{constants,convert,load,model,kernels,indexer_inference}`… use the real role spellings the plan will create: `gigachat_moe_37b_constants`, `_convert`, `_load`, `_model`, `_kernels`, `_inference_indexer`, `_norm_kernels`; plus `CHECK_FALSE(m("gigachat_37b_model"))` (type marker is mandatory).
- [ ] **Step 2: Run it red** — `cmake --build --preset linux-debug --target test_code_naming_guard && ctest --test-dir build/linux-debug -R code_naming -j1`; expected FAIL on the new case (`37b` and `indexer` are not in the vocabulary).
- [ ] **Step 3: Implement** — add `37b` to the variant alternation and `indexer` to the `subsys` alternation in `SchemeRegex()`; write the passport header (numbers above, `static_assert` on `kQkHeadDim==192`, `kLinearQkvzDim==12288`, `kLinearKvRepeat*kLinearNumKeyHeads==kLinearNumValueHeads`, `kNumFullAttn==8`, `kSharedIntermediate==768`, `kHidden%128==0`); add the `{"gigachat_moe_37b_constants.h", "namespace frozen {"}` row to the independence guard's owner map.
- [ ] **Step 4: Run green** — naming guard + `test_model_independence_guard` pass.
- [ ] **Step 5: Commit** — `feat: add GigaChat4-37B-A4B passport and naming vocabulary`.

---

### Task 2: Converter + NVFP4 quantization (stages 1–2)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_convert.h`, `libs/lsxcommon/src/gigachat_moe_37b_convert.cpp`
- Create: `tests/test_gigachat_moe_37b_convert.cpp`
- Create: `models/gigachat4-37b-a4b.json` (all nine stage keys; stages not yet built declared in `golden_debt` with a one-line reason)
- Modify: `apps/lsxforge/src/main.cpp` (`--gigachat4`), `apps/lsxforge/src/cli.{h,cpp}`, `apps/lsxforge/tests/test_lsxforge_cli.cpp`

**Interfaces:**
- Produces: `arrow::Status lsxcommon::gigachat4::ConvertGigaChat4Model(const std::string &input_dir, const std::string &output_root)`; index schema `logestix-gigachat4-nvfp4` with `num_layers`, `layer_types`, `full_attention_layers`, `hidden_size`, `vocab_size`, `num_attention_heads`, `num_key_value_heads`, `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `first_k_dense_replace`, `rms_norm_eps`, `router_scaling_factor`, `swiglu_limit`, `logit_softcap`, `attn_scale`, `index_heads`, `index_head_dim`, `index_topk`, `index_rope`, `bos_token_id`, `eos_token_id`, `pad_token_id`, `vram{}`, `tokenizer`, `deferred{mtp,mtp_layer,mtp_tensors,dflash2}`, `total_pinned_tensors`, `total_experts`.
- Consumes: `gpu::quantize_tensor_gpu` (BF16 path) and `DequantFp8Block128ToBf16Gpu` (FP8 path), `ExpertSchema`, `PinnedSchema`, `DenseNvfp4Schema`, `integrity::BuildAndVerifyTokenizer`.

- [ ] **Step 1: Write the failing test** — synthetic E2E following `tests/test_gigachat_moe_convert.cpp`: build a checkpoint with (a) FP8 tensors + `[ceil(R/128), ceil(C/128)]` F32 scale tiles, (b) BF16 linears, (c) gated-norm triples, (d) router gate + BF16 `e_score_correction_bias`, (e) GDN scalars, (f) indexer `wq_b` FP8 + `wk`/`weights_proj`/`k_norm` BF16, (g) one MTP layer, (h) one dense layer. Assertions: produced tree has `logestix/model.index.json` with `format == "logestix-gigachat4-nvfp4"`, `full_attention_layers` from the fixture, `index_topk/node 2048`, `logit_softcap == 30`; routed experts written one file per expert with 3 projections; MTP counted and written under `logestix/mtp/`; an unknown tensor name fails conversion; a `.weight` with a `_scale_inv` sibling but a shape that does not tile by 128 fails.
- [ ] **Step 2: Run red** — `ctest --test-dir build/linux-debug -R gigachat_moe_37b_convert` fails (`ConvertGigaChat4Model` not declared).
- [ ] **Step 3: Implement** — converter mirrors the 3.5 one but classifies by *source dtype presence* (`_scale_inv` → FP8 path, else BF16→NVFP4 for linear kinds, raw otherwise); MTP layer index from `kMtpLayer`; `deferred.dflash2` notes the separate drafter dir; tokenizer built through `BuildAndVerifyTokenizer` with `tok_policy.pretokenizer = 1` (kind 3).
- [ ] **Step 4: Run green** — convert target green (bounded TIMEOUT 120 s, synthetic geometry through `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`).
- [ ] **Step 5: Commit** — `feat: gigachat4 converter (FP8/BF16 → NVFP4) with synthetic E2E`.

---

### Task 3: Loader, factory registration, model class shell (stage 3)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_load.h`, `libs/lsxcommon/src/gigachat_moe_37b_load.cpp`
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_model.h`, `libs/lsxcommon/src/gigachat_moe_37b_model.cpp`
- Create: `tests/test_gigachat_moe_37b_load_fast.cpp`, `tests/test_gigachat_moe_37b_loader.cpp`
- Modify: `models/gigachat4-37b-a4b.json` (load stage), `tests/test_model_independence_guard.cpp` (consumers + lifecycle rows are manifest-driven; add only the `consumers` entry)

**Interfaces:**
- Produces: `struct Gc4Index` (the index fields above + `VramProfile vram`), `bool LoadGigaChat4Index(const std::string&, Gc4Index&, std::string&)`, `bool ValidateGigaChat4Geometry(const Gc4Index&, std::string&)`, `struct Gc4GatedNorm {dim, rank, d_weight, d_gate_up, d_gate_down}` reusing the shape contract from `GcGatedNorm`, `struct Gc4LayerWeights` (norms, MLA weights incl. `attn_gate`, indexer `indexer_wq_b`/`indexer_wk`/`indexer_wp`/`indexer_k_norm`, GDN weights, router, shared/dense), `struct Gc4Weights {h_embed, h_embed_pinned, d_lm_head, final_norm, layers, resident_bytes; void Free();}`, `bool LoadGigaChat4Weights(path, idx, Gc4Weights&, cudaStream_t, std::string&)`, `class GigaChat4MoeModel final : public ModelBase` with `static const std::string kType`.

- [ ] **Step 1: Write the failing tests** — `_load_fast` (label `fast`, host-only): index parse of a synthetic `model.index.json` string, format-tag rejection of `logestix-gigachat-nvfp4`, frozen-geometry mismatch (hidden 2560 → 4096) returns false with `"frozen geometry mismatch"`, and `LsxModelFactory::Create("gigachat4-37b-a4b")` is non-null while `LsxModelFactory::Create("gigachat4")` resolves through the manifest alias only if declared. `_loader` (device, synthetic tree): every layer's weights upload, `resident_bytes` equals the upload accumulator, `Free()` releases all (run under cuda-memcheck-style leak assertion used by the 3.5 loader test).
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement** — loader clones the 3.5 shape (pinned staging ring, `LoadDenseNvfp4BatchViews`, `LoadTensorBatchViews`, `ExpertStore` for experts) and adds the indexer tensors and the `w_offset` norm variant; model class implements `Load`/`Unload`/`Inspect`/`TeardownWeights`, publishes `KvLayout()` = `kPosMajor`, registers in `model_registry.cpp` behind the manifest-generated `.inc` (no hand edit beyond the manifest), and gets its `consumers` row.
- [ ] **Step 4: Run green** — `ctest -R "gigachat" -j1` still green including the 3.5 row.
- [ ] **Step 5: Commit** — `feat: gigachat4 loader + factory registration`.

---

### Task 4: Tokenizer bridge (stage 4)

**Files:**
- Create: `tests/test_gigachat_moe_37b_tokenizer_parity.cpp`
- Modify: `models/gigachat4-37b-a4b.json` (load stage list)

**Interfaces:**
- Consumes: `BuildPromptInputIds`, `BuildConversationInputIds` from the model class (Task 3/Task 8 shells).

- [ ] **Step 1: Write the failing test** — the vault tokenizer (`LSX_TEST_GIGACHAT4_37B_A4B_ROOT`, `SKIP()` when absent) must resolve `<s>`=1, `</s>`=2, `<|role_sep|>\n`=128000, `<|message_sep|>\n\n`=128001, `<think>`=128014, `</think>`=128015, and encode/decode round-trip `"The capital of France is"` to the same ids the source tokenizer produces (ids recorded from `tokenizer.json` in the test as the expected vector).
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement** — `BuildPromptInputIds` frames one chatml-v5 turn with an empty `<think></think>` exactly as the 3.5 sibling does (`selfcheck_frame_prompt=false`, Paris probe on the raw prompt), ids taken from `frozen::k*Id` constants, never hardcoded.
- [ ] **Step 4: Run green.**
- [ ] **Step 5: Commit** — `feat: gigachat4 tokenizer bridge`.

---

### Task 5: Tool-calling bridge (stage 4b)

**Files:**
- Create: `tests/test_gigachat_tool_framing_37b.cpp`
- Modify: `tests/test_model_suite_guard.cpp` (tool-verdict table row)
- Modify: `models/gigachat4-37b-a4b.json` (model stage list, `tool_roundtrip: true`)

**Interfaces:**
- Consumes: `RenderGigaChatConversation`, `AppendGigaChatIds`, `ParseGigaChatTokenCalls` (existing family codec).
- Produces: `GigaChat4MoeModel::BuildConversationInputIds` / `ParseToolCalls` overrides.

- [ ] **Step 1: Write the failing test** — on the synthetic `tests/data/` tokenizer: render a conversation with one tool and assert the exact GCML wire text (`<｜GCML｜>tool_calls>`, `<｜GCML｜>invoke name="get_weather">`, `<｜GCML｜>parameter name="city" string="true">Paris`), parse a hand-built token sequence back into `ToolCall{name="get_weather", args={"city":"Paris"}}`, and assert the framing does **not** depend on `reasoning` (both a `<think></think>`-closed and a reasoning-open generation prompt end with `assistant<|role_sep|>`). Do **not** round-trip `ParseToolCalls` on the rendered prompt (its example call is part of the instructions).
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement** — overrides delegate to the family codec; any divergence from `chatml v5` that the GC4 template adds (text-block joining) is host-side only.
- [ ] **Step 4: Run green** + `test_model_suite_guard` green with the new verdict row.
- [ ] **Step 5: Commit** — `feat: gigachat4 GCML tool bridge (stage 4b)`.

---

### Task 6: Variant-parameterised norm / GDN / router / softcap kernels (stages 5–6)

**Files:**
- Modify: `libs/lsxcommon/src/gigachat_moe_norm_kernels.cu`, `libs/lsxcommon/include/lsxcommon/gigachat_moe_kernels.h`, `libs/lsxcommon/src/gigachat_moe_device_math.h`
- Create: `tests/test_gigachat_moe_37b_kernels.cpp`
- Modify: `gigachat_moe_inference_pipeline.cu` call sites (offset `1.0f` for 3.5)

**Interfaces:**
- Produces: `RunGigaChatGatedNorm(const float *x, int rows, int dim, int rank, const uint8_t *w, const uint8_t *w_up, const uint8_t *w_down, float eps, float w_offset, float *out, cudaStream_t)`; `RunGdnOutputNorm(..., float w_offset, ...)` (existing signature gains the offset); `RunGigaChatSoftcap(float *logits, int n, float cap, cudaStream_t)` implementing `cap*tanh(x/cap)` in F32.
- Consumes: the existing `dev::` element functions, extended with `w_offset`.

- [ ] **Step 1: Write the failing test** — host reference in the test computes `xn*(offset+w)*2*sigmoid(Wd·silu(Wu·xn))` in double; assert `w_offset=0.0f` matches GC4 math and `w_offset=1.0f` reproduces the pre-refactor 3.5 result bit-exactly (the same fixture values the 3.5 kernel test uses); GDN norm with offset 0 and sigmoid ×2; router top-k with a bias that reorders selection but must not enter the weights; softcap saturates `|logit| ≤ 30` and is monotone.
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement** — add the offset parameter end-to-end (`Gr` kernels, launchers, 3.5 pipeline passes `1.0f`).
- [ ] **Step 4: Run green** — plus the 3.5 `test_gigachat_moe_kernels`/`_devkernels`/`_math` targets, proving the refactor is arithmetic-neutral.
- [ ] **Step 5: Commit** — `feat: parameterise gigachat gated-norm/GDN gate offset and add logit softcap`.

---

### Task 7: DSA indexer + masked attention (stage 5)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/gigachat_moe_37b_inference_indexer.h`, `libs/lsxcommon/src/gigachat_moe_37b_inference_indexer.cu`
- Create: `tests/test_gigachat_moe_37b_indexer.cu`
- Modify: `gigachat_moe_inference_attn.cu` / `.h` (optional selection mask + attn-scale/rope from `GcMlaDims`)

**Interfaces:**
- Produces: `struct Gc4IndexerDims {int hidden, q_lora, heads, head_dim, rope, capacity;}`, `struct Gc4IndexerScratch`, `cudaError_t RunGigaChat4IndexerStep(const float *q_latent_scaled, const uint8_t *hidden_bf16, const Gc4IndexerWeights&, const Gc4IndexerDims&, const float *rope_cos, const float *rope_sin, int pos, float *index_k_cache, Gc4IndexerScratch&, int32_t *out_idx, int *out_count, cudaStream_t)`; `RunGigaChatMlaDecode(..., const int32_t *sel, int n_sel, ...)`.

- [ ] **Step 1: Write the failing test** — host reference (double) for a tiny geometry (heads 4, head_dim 8, rope 4, topk 3, capacity 16): q/k projection, k-norm, non-interleaved rope, `weights*heads^-0.5`, `relu`, `*head_dim^-0.5`, top-k. Two decisive cases: `pos+1 <= topk` ⇒ selection is the full causal prefix and the decode output equals the unmasked MLA output **bit-exactly**; `pos+1 > topk` ⇒ the selected set equals the host reference's, and the attention output matches the reference within the family tolerance class. Also: a mask with a count of 0 or > kv_len is rejected, and the same input twice gives the same selection (determinism, no atomics).
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement** — per-DSA-layer index key cache `[capacity, 128]` BF16 (post-rope), scores thread-per-position with the family's ascending-`+=` dot, deterministic top-k (one block, thread-0 selection, lowest-index tie-break), softmax kernels take the optional index list.
- [ ] **Step 4: Run green.**
- [ ] **Step 5: Commit** — `feat: gigachat4 DSA indexer and masked MLA decode`.

---

### Task 8: Pipeline wiring, Infer, streaming/abort (stage 5/6)

**Files:**
- Modify: `libs/lsxcommon/src/gigachat_moe_inference_pipeline.cu` / `.h` (variant knobs in `GcPipelineDims`: `norm_w_offset`, `gdn_w_offset`, `attn_scale`, `rope_theta/factor/orig/betas`, `index_topk`, `logit_softcap`, `use_dsa`), `libs/lsxcommon/src/gigachat_moe_37b_model.cpp` (Infer)
- Create: `tests/test_gigachat_moe_37b_model.cpp`
- Modify: `models/gigachat4-37b-a4b.json` (model + kernels stages)

**Interfaces:**
- Consumes: Tasks 3, 6, 7.
- Produces: `GigaChat4Model::Infer` that carries `RunUnderResidencyLock`, `GenerationStream gs` + `gs.Emit` per token, `gs.Aborted()` polling, `Inspect`/`SelfCheck` and a filled `InferenceResult` (`output_text`, `weight_resident_bytes`).

- [ ] **Step 1: Write the failing test** — synthetic tree (8 layers relaxed to the fixture's, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`): `Infer` returns `ok`, one `Emit` per generated token, a pre-set `abort_flag` stops after the first emitted token, `final_logits.size() == vocab` and `max|logit| <= 30 + 1e-3` (softcap applied), `weight_resident_bytes > 0`. The 3.5 `test_gigachat_moe_model` stays green (same pipeline TU).
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement** — pipeline `Step` gains the indexer step on DSA layers, the norm/GDN offsets and the softcap before the logits leave the device (softcap applied on the F32 logits buffer inside `Step`, so `Infer`'s host readback is post-softcap).
- [ ] **Step 4: Run green** — `ctest -R "gigachat" -j1`.
- [ ] **Step 5: Commit** — `feat: gigachat4 decode pipeline (DSA + softcap) and Infer`.

---

### Task 9: SelfCheck, parity goldens, manifest completion (stage 8)

**Files:**
- Create: `tests/test_gigachat_moe_37b_selfcheck.cpp`
- Create: `tests/fixtures/fsi/gigachat4-37b-a4b.json`, `tests/fixtures/code/gigachat4-37b-a4b.json`, `tests/fixtures/det/gigachat4-37b-a4b.json`
- Modify: `tools/fsi_capture.py` (+ `code_capture.py`, `det_capture.py`), `models/gigachat4-37b-a4b.json` (selfcheck/fsi/code/det/perf stages, empty `golden_debt`), `docs/SELFCHECK_MATRIX.md`, `tests/perf/perf_baselines.json`, `tests/test_perf_baseline.cpp`

**Interfaces:**
- Consumes: the model class (`SelfCheck`, `Infer`).
- Produces: `selfcheck_max_gen`/`selfcheck_bit_exact`/`selfcheck_frame_prompt` declared **from measurement** on this model (not copied from the sibling), and `perf baseline: gigachat4-37b-a4b`.

- [ ] **Step 1: Write the failing test** — `test_gigachat_moe_37b_selfcheck.cpp` runs ≥3 cycles with bit-exact `final_logits` (cycle count from `LSX_TEST_GIGACHAT4_37B_SELFCHECK_CYCLES`), asserts `r.selfcheck_passed` and first token of `"The capital of France is"` decodes to `Paris`.
- [ ] **Step 2: Run red** (no vault → `SKIP_RETURN_CODE 4`, then red once the tree exists).
- [ ] **Step 3: Implement** — capture tool override `OVERRIDES["gigachat4-37b-a4b"]` with `add_special_tokens: False`, `source` recorded as template-derived (the 3.5 pattern); capture the three goldens; seed the perf row **through the harness** (`test_perf_baseline_gigachat4`).
- [ ] **Step 4: Run green** — SelfCheck + 3 parity gates 3/3 + perf target, on the Release/AOCC tree.
- [ ] **Step 5: Commit** — `feat: gigachat4 selfcheck, parity goldens and perf row`.

---

### Task 10: Real conversion, real-weight gates, model report (stage 9)

**Files:**
- Modify: `docs/model-reports/2026-11-26-GigaChat4-37B-A4B.md` (new), `docs/model-reports/README.md`, `docs/OPEN_TASKS.md` (+ `docs/open-tasks/live/models/gigachat4-37b-a4b.md`), `docs/MODELS.md` (generated)

- [ ] **Step 1: Convert the real checkpoint** — `build/linux-release-znver4-aocc/apps/lsxforge/lsxforge --gigachat4` with the vault input; record artifact size, per-section bytes, `total_experts`, MTP tensor count.
- [ ] **Step 2: Run the row's gates on the release tree** — `ctest --test-dir build/linux-release-znver4-aocc -R "gigachat4" -j 1`; every target green or explicitly declared debt; a SKIP is not a pass.
- [ ] **Step 3: Short source oracle probe (operator-approved)** — `tools/gigachat4_oracle.py`: dequantize FP8→BF16 on the host, run `modeling_gigachat4.py` teacher-forced on the control prompt for a handful of positions, compare top-1 and the logit margin against the engine; record the measured deltas (or the reason it could not run) in the report.
- [ ] **Step 4: Write the report** — publish-ready post (English, no internal paths) + evidence section; every number points at a gate file; debt named as debt. Update `docs/OPEN_TASKS.md` only for what is genuinely open.
- [ ] **Step 5: Final verification** — `ctest --preset linux-debug -j 24` (fast guards incl. `test_manifest_sync --strict`, `test_model_suite_guard`, `test_model_independence_guard`, `test_ctest_labels_guard`, `test_code_naming_guard`), `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .`, then commit `feat: ingest GigaChat4-37B-A4B`.

---

## What is deliberately not run

The label sweeps (`linux-debug-inference`, 350 targets across 43 models) are
infrastructure tools, not model verification. This plan runs the fast guards
plus `-R "gigachat"`, and says so in the report.
