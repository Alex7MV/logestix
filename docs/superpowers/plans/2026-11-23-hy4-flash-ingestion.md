# Hy4-flash Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest tencent/Hy4-flash (`/mnt/fast_storage/huggingface/Hy4-flash`) into logestix as a text-backbone NVFP4 model with a registered `hy4-flash` identity, a native Hunyuan tokenizer/tool bridge, family CUDA decode kernels, and every lifecycle gate green.

**Architecture:** One family contour under `hy` (`hy_moe_flash_*`): frozen passport → C++ converter (BF16→NVFP4, routed experts streamed to host Arrow, Engram/ViT/DFly2 cut and counted) → NVFP4 loader over the neutral `ExpertStore` → per-token decode pipeline (iHC streams, gated MLA with learnable sinks, SWA ring + IndexPool sparse selection, LatentMoE) → `IModel` with streaming/abort. Goldens come from a new Python oracle (`tools/hy4_flash_oracle.py`) because no published implementation of the flash architecture exists (transformers ships `hy_v4`, not `hy_v4_flash`).

**Tech Stack:** C++20 / C23, CUDA 13.5 (sm_120a, Blackwell), Apache Arrow 26 (mmap views), NVFP4 (`lsx_nvfp4`), Catch2 v3, CMake presets, Python 3 (oracle + capture tools only), PyTorch CPU for the oracle.

**Spec:** `docs/superpowers/specs/2026-11-23-hy4-flash-ingestion-design.md`

## Global Constraints

- Target Blackwell only (sm_120a). No pre-Blackwell code paths.
- Logging via `lsxcommon::log` only. `absl::StrFormat` is printf-style (`%s`/`%d`).
- Headers reachable from `.cu`/`.cuh` contain no `<absl/container/*>`.
- No CLI speed flags. `--bit-exact` maps to `FastMathMode::kOff`.
- Device targets declare `RESOURCE_LOCK "gpu"` via the manifest stage default; `fast` stages stay unlocked.
- Every skip-capable target sets `SKIP_RETURN_CODE 4` (the manifest does this).
- Convert tests are synthetic end-to-end only, label `convert`, TIMEOUT ≤ 600 s (120 s canonical), synthetic geometry unlocked by the runtime env `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`.
- No hand-written registry: `models/hy4-flash.json` is the only hand-edited table; `model_manifest.h`, `model_targets.cmake`, `model_registry.inc` and `docs/MODELS.md` are generated.
- Engine geometry is frozen at compile time and cross-checked at `Load`; a mismatch fails the load.
- One GPU workload at a time; check `nvidia-smi --query-compute-apps` before any GPU-labelled run.
- Temporary files go to `/mnt/fast_storage/tmp`.

## Review Focus

1. **A tensor the converter silently ignores.** Every name must land in exactly one bucket (global / layer-pinned / dense-NVFP4 / routed expert / cut); an unclassified name must fail the conversion, never pass. Pinned by the convert test enumerating the real checkpoint's 1964 names through the classifier (a host-only test that reads only the index JSON, no weights).
2. **iHC applied with the wrong shape or the wrong magnitude.** `post` is per-stream ([4]) and multiplies the *whole* sublayer output before it is added back; a scalar or a mean silently changes every layer. Pinned by a host-reference kernel test on a 2-layer toy with hand-computed values.
3. **Sandwich norms applied to the residual instead of the sublayer output.** Pinned by the same kernel test: a case where the two differ by construction.
4. **SWA ring and IndexPool selection disagreeing on which positions are visible** (off-by-one at the window edge, and the current group's own tokens always visible). Pinned by an index-set equality test against a host reference at positions 0, 511, 512, 513, 2047, 2048, 4096.
5. **Engram/ViT/DFly2 tensors leaking into the artifact** (making it 200 GB instead of 90 GB, or making the load fail on a missing tensor). Pinned by the convert test's `deferred` counts and the loader's "no engram tensor is ever requested" assertion.

---

### Task 1: Frozen passport + layer tables

**Files:**
- Create: `libs/lsxcommon/src/hy_moe_flash_constants.h`
- Test: `tests/test_hy_moe_flash_constants.cpp`

**Interfaces:**
- Produces: `namespace lsxcommon::hy4::frozen` with `kType` (`"hy4-flash"`), `kModelName` (`"Hy4-flash"`), `kFormatTag` (`"logestix-hy4-nvfp4"`), the integer/float constants listed in the spec, and `constexpr` predicates `IsDsa(int)`, `IsSwa(int)`, `Heads(int)`, `IndexerAt(int)`, `IsEngramCut(int)`, `IsMoeLayer(int)`, `KvBytesPerToken()`.

- [ ] **Step 1: Write the failing test** — `tests/test_hy_moe_flash_constants.cpp`: a `TEST_CASE("hy4 passport is frozen")` that `static_assert`s nothing (that is the header's job) but *runtime*-checks the layer tables against the real `config.json` when it exists, and `SKIP()`s otherwise:
  - `IsDsa(l)` agrees with `text_config.layer_types[l] == "deepseek_sparse_attention"` for all 42 layers;
  - `IsSwa(l)` agrees with `"sliding_attention"`;
  - `Heads(l)` equals `num_attention_heads` for DSA layers and `swa_num_attention_heads` for SWA layers;
  - `IsMoeLayer(l)` agrees with `mlp_layer_types[l] == "sparse"`;
  - `IsEngramCut(l)` agrees with `text_config.engram_layer_ids == {1,11,21,31}`;
  - the constants equal `hidden_size`, `num_hidden_layers`, `vocab_size`, `moe_intermediate_size`, `intermediate_size`, `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `sliding_window`, `hc_mult`, `routed_scaling_factor`, `swiglu_limit`, `rms_norm_eps`, `hc_eps`, `hc_magnitude`, `index_n_heads`, `index_head_dim`, `index_topk`, `compressed_indexer_compress_ratio`, `compressed_indexer_group_topk`;
  - the vault root is `LSX_TEST_HY4_FLASH_ROOT`, defaulting to `/mnt/fast_storage/huggingface/Hy4-flash`.

- [ ] **Step 2: Run it to make sure it fails**

Run: `cmake --preset linux-debug >/dev/null && cmake --build --preset linux-debug --target test_hy_moe_flash_constants` (register the target first via Task 2's manifest — until then, expect "no rule to make target").
Expected: FAIL — header and target do not exist.

- [ ] **Step 3: Write the header** with the constants from the spec plus `static_assert`s that pin the relations (`kHeadsSwa == 2*kHeadsDsa`, `kQkNope + kQkRope == 256`, `kDsaLayerCount == 12`, `kSwaLayerCount == 30`, `kLatent == kMoeIntermediate`, `kVocab % 128 == 0`, `IsDsa(0) && IsDsa(1) && IsDsa(5) && !IsDsa(4)`, `IsEngramCut(1) && IsEngramCut(31) && !IsEngramCut(11+1)`).

- [ ] **Step 4: Run test to verify it passes**

Run: `ctest --test-dir build/linux-debug -R '^test_hy_moe_flash_constants$' -j 1`
Expected: PASS (or SKIP if the vault is absent).

- [ ] **Step 5: Commit** `feat(hy4): freeze the Hy4-flash passport and layer tables`

---

### Task 2: Manifest + registration scaffolding

**Files:**
- Create: `models/hy4-flash.json`
- Modify: `docs/MODELS.md` (generated), `libs/lsxcommon/src/model_registry.cpp` (only if registration is not already generated — it is: `model_registry.inc`)
- Test: the existing generated-table guards (`test_manifest_sync`, `test_model_suite_guard`, `test_code_naming_guard`, `test_ctest_labels_guard`)

**Interfaces:**
- Consumes: the passport from Task 1.
- Produces: the ctest target set for every later task (`test_hy_moe_flash_convert`, `_loader`, `_model`, `_kernels`, `_tool_framing`, `_selfcheck`, `_fsi_selfcheck`, `_code_selfcheck`, `_det_selfcheck`, `test_perf_baseline_hy4_flash`), each resolved from `models/hy4-flash.json` by the generator.

- [ ] **Step 1: Write the failing test** — nothing new: the guard tests are the test. Write the manifest with `stages` naming the targets, then run the guards *before* the target sources exist so the generator reports the missing sources.
- [ ] **Step 2: Run** `python3 tools/gen_model_manifest.py --root .` → expect it to list each missing `tests/<target>.cpp`.
- [ ] **Step 3: Create the stub sources** the generator requires (empty Catch2 `TEST_CASE` placeholders that `SKIP()`), commit them together with the manifest so the tree is never red.
- [ ] **Step 4: Run** `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .` then `cmake --preset linux-debug` and `ctest --test-dir build/linux-debug -R 'manifest_sync|model_suite_guard|code_naming_guard|ctest_labels_guard|model_independence_guard' -j 4`
- [ ] **Step 5: Commit** `feat(hy4): register the hy4-flash manifest and its target set`

---

### Task 3: Converter (stage 1–2)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/hy_moe_flash_convert.h`, `libs/lsxcommon/src/hy_moe_flash_convert.cpp`
- Modify: `apps/lsxforge/src/main.cpp` (`--hy4_flash`)
- Test: `tests/test_hy_moe_flash_convert.cpp`, `tests/test_hy_moe_flash_fixture.h`

**Interfaces:**
- Produces: `lsxcommon::hy4::ConvertHy4FlashModel(const std::string &input_path, const std::string &output_path) -> arrow::Status`, and `lsxcommon::hy4::ClassifyTensor(std::string_view name) -> TensorKind` with `enum class TensorKind { kGlobalRaw, kLayerRaw, kDenseNvfp4, kRouterGate, kExpertGateUp, kExpertDown, kCutVision, kCutDfly, kCutEngram, kUnknown }`.
- Layout written: `logestix/model.index.json`, `logestix/pinned/global.arrow`, `logestix/pinned/layer_XXX.arrow` (raw norms/iHC/sinks/router), `logestix/pinned/layer_XXX_nvfp4.arrow` (dense projections), `logestix/experts/layer_XXX/expert_NNNNN.arrow` (one `PackedExpert` per expert: w1 = `gate_up` first half, w3 = `gate_up` second half, w2 = `down`).

- [ ] **Step 1: Write the failing test** — `tests/test_hy_moe_flash_convert.cpp`:
  - `TEST_CASE("hy4 convert: classifier covers every real tensor name")` — host-only, reads ONLY `/mnt/fast_storage/huggingface/Hy4-flash/model.safetensors.index.json` (167 KB), `SKIP()`s when absent, asserts every one of the 1964 names maps to a kind other than `kUnknown`, and asserts the bucket counts: 1632 routed-expert tensors (41 layers × 512 × 2 packed tensors… computed in the test from the index), the Engram/ViT/DFly2 counts, and that all 4 `engram_layer` layers classify as `kCutEngram`.
  - `TEST_CASE("hy4 convert: synthetic checkpoint end to end")` — builds a synthetic tree with `synthtest::WriteSafetensors` (`config.json` with relaxed geometry + one dense layer + one MoE layer + 4 experts + an engram layer + a vision tower sample + a dfly sample), runs the converter, then asserts: index has `format == "logestix-hy4-nvfp4"`, `model == <dir name>`, `deferred.engram == <n>`, every expected Arrow file exists, every dense projection has a `DenseNvfp4` record with `packed_bytes == rows*cols/2`, every norm/scalar is a raw record with its source dtype, and `experts/layer_001/expert_00003.arrow` decodes to three NVFP4 planes whose `gate_up` split is 2*I rows.
  - `TEST_CASE("hy4 convert: an unknown tensor fails the conversion")` — one synthetic tensor named `model.layers.0.mystery.weight` must make `ConvertHy4FlashModel` return a non-OK status naming it.

- [ ] **Step 2: Run** `ctest --test-dir build/linux-debug -R '^test_hy_moe_flash_convert$' -o --output-on-failure` → FAIL (no implementation).

- [ ] **Step 3: Implement the converter** — `ClassifyTensor` by name pattern (the exact leaf set is the design doc's artifact table); raw bucket for norms/`learnable_sink_param`/`hc_*`/`e_score_correction_bias`/indexer F32 scalars; `DenseNvfp4` for every 2-D linear (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`, `linear_gate`, dense `gate/up/down`, `shared_experts.*`, `routed_expert_down_proj`, `routed_expert_up_proj`, `combine_*` stays raw, indexer `query_projection`/`token_key_projection`/`head_weight_projection`, dense-layer `mlp.*`); experts from the 3-D `[512, 3072, 1536]` `gate_up_proj` and `[512, 1536, 1536]` `down_proj` tensors by per-expert slicing (they are single tensors, not per-expert names — this is the one place the flash layout differs from flint); `QuantizeSlice`/`QuantizeSliceBf16` for BF16 sources; write the index with the per-layer list and the `deferred` counts.

- [ ] **Step 4: Run** the convert target → PASS. Then run it against the REAL checkpoint into `/mnt/fast_storage/huggingface/Hy4-flash` (this writes `logestix/` next to the weights; ~90 GB, run it in the background).
- [ ] **Step 5: Commit** `feat(hy4): the converter cuts Engram/ViT/DFly2 and packs LatentMoE experts`

---

### Task 4: Tokenizer bridge (new pre-tokenizer family)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/pretokenizer.h` (new `HunyuanWordStream`), `libs/lsxcommon/src/pretokenizer.cpp`, `libs/lsxcommon/src/tokenizer_convert.cpp` (`policy.pretokenizer == 11` → `tokenizer_kind = 13`), `libs/lsxcommon/src/tokenizer.cpp` (dispatch kind 13)
- Test: `tests/test_hy_moe_flash_tokenizer_parity.cpp`

**Interfaces:**
- Produces: `lsxcommon::HunyuanWordStream` with the same `next(WordSpan&)` contract; `tokenizer_kind = 13` in `TokenizerVocabMeta`.

- [ ] **Step 1: Write the failing test** — parity against the reference `tokenizers` library: the test reads a committed fixture `tests/fixtures/hy4/tokenizer_parity.json` (list of `{text, ids, words}`), generated once from the real `tokenizer.json` with Python and committed, and asserts `arrow_tokenizer_encode` reproduces `ids` for every entry, plus that the C++ word stream yields exactly `words`. The fixture must include: ASCII prose with contractions, `3-digit`/`4-digit` numbers, mixed CJK+latin, emoji, `\r\n`, trailing spaces, `  multiple   spaces`, and the five chat markers.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the six-alternative pattern `[punct][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ' '?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+` over the two isolated stages (`\p{N}{1,3}`, CJK run).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(hy4): the Hunyuan pre-tokenizer family (kind 13)`

---

### Task 5: Loader (stage 3)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/hy_moe_flash_load.h`, `libs/lsxcommon/src/hy_moe_flash_load.cpp`, `libs/lsxcommon/src/hy_moe_flash_load.cu`
- Test: `tests/test_hy_moe_flash_loader.cpp`

**Interfaces:**
- Produces: `Hy4Topology LoadHy4Topology(const std::string&)`, `bool ValidateHy4Geometry(const Hy4Topology&, std::string*)`, `bool LoadHy4Weights(path, Hy4PinnedWeights&, cudaStream_t, std::string&)`, `void FreeHy4PinnedWeights(Hy4PinnedWeights&)`, `using Hy4ExpertDb = gpu::expert_store::ExpertStore<Hy4ExpertEntry>;` with `load_hy4_expert_index` / `ensure_hy4_expert_resident` / `free_hy4_expert_index`.
- `Hy4LayerWeights` carries the NVFP4 attention/dense/shared/latent projections, raw `__half*` norms and `__half*` sinks, F32 iHC parameters, F32 router gate + bias, and (DSA layers) the indexer bundle.

- [ ] **Step 1: Write the failing test** — on the synthetic converted tree from Task 3: topology parses (format/model/layers/experts), geometry validation rejects a wrong `num_layers` and accepts a synthetic one, weights upload (every projection's `rows/cols` matches the fixture's shapes, every norm pointer is non-null), the expert store returns a resident entry for `(layer, expert)` with three non-null spans, `Free()` drops every allocation (RSS/VRAM delta bounded), and a second `Load` on a failed first load leaves no residue.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**, streaming `DenseNvfp4View`s through `fastload::UploadNvfp4Weight` and experts through the neutral `ExpertStore`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(hy4): the NVFP4 loader and the mmap expert store`

---

### Task 6: Model class, registration, request guards

**Files:**
- Create: `libs/lsxcommon/src/hy_moe_flash_model.h`, `libs/lsxcommon/src/hy_moe_flash_model.cpp`
- Test: `tests/test_hy_moe_flash_model.cpp`

**Interfaces:**
- Consumes: Task 5's loader, Task 7's pipeline (declared in the header, implemented next task).
- Produces: `class Hy4FlashModel final : public ModelBase` with `static const std::string kType;`, registered through the generated `model_registry.inc`.

- [ ] **Step 1: Write the failing test** — factory resolves `"hy4-flash"`; `Load()` fails with a clear error on a wrong format tag, on a geometry mismatch, and succeeds on the synthetic tree; `Infer` before `Load` returns `error_msg == "model not loaded"`; `KvLayout()` reports `kPosMajor`; a pre-set abort flag produces zero tokens and no hang.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `Load` (BeginLoad → topology → format gate → frozen-geometry gate → EnsureContext → LoadHy4Weights → AcquirePermanentWeights → FinishLoad), `Infer` (`LsxFastMathScope` + `ValidateInferRequest` + `RunUnderResidencyLock` + `GenerationStream` + `InferenceResult` fill), `SelfCheck`, `TeardownWeights`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(hy4): the model class and its lifecycle contract`

---

### Task 7: Kernels (stages 5–6)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/hy_moe_flash_device_math.h`, `libs/lsxcommon/include/lsxcommon/hy_moe_flash_kernels.h`, `libs/lsxcommon/src/hy_moe_flash_kernels.cu`, `libs/lsxcommon/src/hy_moe_flash_inference_attn.cu`
- Test: `tests/test_hy_moe_flash_kernels.cu`

**Interfaces:**
- Produces (each `cudaError_t`, all taking `cudaStream_t` last):
  `RunHy4IhcMix`, `RunHy4IhcPre`, `RunHy4IhcPost`, `RunHy4IhcHead`, `RunHy4RmsNorm`, `RunHy4Rope`, `RunHy4RouterTopK`, `RunHy4SwigluClamp`, `RunHy4LatentMoE` (single-expert accumulate), `RunHy4SinkAttn` (window + gathered sparse set), `RunHy4SwaIdx`, `RunHy4IndexPool` (pool → score → top-k groups → index list), `RunHy4GatedScale`.

- [ ] **Step 1: Write the failing tests** — one case per kernel against a host reference computed in the test from the same inputs:
  - iHC: 2 token rows, `hc_mult=4`, hand-computed `pre/post` and the stream update; a case where `post` is non-uniform proves it multiplies the sublayer output *per stream*.
  - sandwich norm: `norm(y)` before the residual add, distinguished from `norm(y+res)`.
  - sink attention: with one sink at `+inf` the output must be the zero vector rounded; with the sink at `0` it must equal softmax over keys plus one; window-edge indexing at pos 0/511/512; gathered-set indexing with a set that excludes a visible key.
  - router: 512 rows, top-16, ties resolved to the lower index, renormalized weights sum to `routed_scaling_factor`; the bias moves the *selection* but not the weight.
  - IndexPool: pooling math (gate-weighted sum → key norm) and the selected index set against a host reference at 0/4/512/2048/4096 tokens, including "the current partial group's decoded tokens are always visible".
  - LatentMoE: latent round trip `down → experts → up` with a single expert vs the host.
- [ ] **Step 2: Run** `ctest -R '^test_hy_moe_flash_kernels$'` → FAIL.
- [ ] **Step 3: Implement** the kernels; thread-0-only RMW behind `__syncthreads()`; no `atomicAdd` into anything feeding `final_logits`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(hy4): the family kernels (iHC, gated MLA with sinks, IndexPool, LatentMoE)`

---

### Task 8: Decode pipeline (stage 5)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/hy_moe_flash_inference_pipeline.h`, `libs/lsxcommon/src/hy_moe_flash_inference_pipeline.cu`
- Test: the model-stage test's streaming/abort cases (`tests/test_hy_moe_flash_model.cpp`, Task 6) plus `tests/test_streaming_assert.h`

**Interfaces:**
- Produces: `Hy4Session` (MLA latent KV `[layers][pos][kv_lora + rope]` + SWA ring + indexer K cache per DSA layer + gen tail), `Hy4Scratch`, and `gpu::InferenceResult RunHy4Inference(...)` matching the flint signature.

- [ ] **Step 1: Write the failing test** — the `test_streaming_assert` contract: one `Emit` per generated token in order, `gen_tokens.size() == budget` for a fixed budget with `ignore_eos`, a pre-set abort flag yields zero tokens, and two identical calls from a reset session produce identical `final_logits` (hash equality).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the token loop: embed → 4-stream expand → per layer {attn iHC mix/pre → RMSNorm → q_a/q_a_norm/q_b → RoPE, kv_a/kv_a_norm → latent + rope, cache write, indexer (DSA) or ring (SWA), sink attention, gate, o_proj, post_proj_norm → iHC post → ffn iHC mix/pre → RMSNorm → dense SwiGLU or {router → stage 16 experts → latent accumulate → up_proj → combine_output_norm + shared expert} → iHC post} → hc_head collapse → final norm → fp32 head → argmax/sample.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(hy4): the one-token decode pipeline`

---

### Task 9: Tool-calling bridge (stage 4b)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/hy_moe_flash_tool_format.h`, `libs/lsxcommon/src/hy_moe_flash_tool_format.cpp`
- Test: `tests/test_hy_moe_flash_tool_framing.cpp` (+ a new stem in `AllowedCommon()` if the naming guard needs it — it does not: `hy_moe_flash_tool_format.cpp` matches the scheme)

**Interfaces:**
- Produces: `bool RenderHy4Conversation(const ArrowTokenizerDb&, const ChatConversation&, std::vector<int32_t>&)` and `bool ParseHy4ToolCalls(const ArrowTokenizerDb&, const std::vector<int32_t>&, std::vector<ToolCall>&)`.

- [ ] **Step 1: Write the failing test** — render structure for system/user/assistant/tool and the `<tools>` block; `reasoning_effort` `high` vs `no_think`; parse a native `<tool_call:opensource>` block with two `<arg_key>/<arg_value>` pairs (string and JSON-valued); tokenizer round-trip of the markers against a synthetic tokenizer fixture (add the markers to its `added_tokens`).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the Hunyuan template (verified against the checkpoint's own `chat_template.jinja`, which is the semantic source of truth).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(hy4): the native Hunyuan tool-calling bridge`

---

### Task 10: SelfCheck + parity goldens (stage 8)

**Files:**
- Create: `tools/hy4_flash_oracle.py`
- Modify: `tools/fsi_capture.py` (add `OVERRIDES["hy4-flash"] = {"loader": "hy4_flash", "device": "cpu", "ignore_eos": ...}`), `tools/code_capture.py`, `tools/det_capture.py` (same override)
- Test: `tests/test_hy_moe_flash_selfcheck.cpp` (via `test_selfcheck_driver.cpp`), `test_hy_moe_flash_{fsi,code,det}_selfcheck` (via `test_parity_gate.cpp`)

**Interfaces:**
- Produces: goldens `tests/fixtures/{fsi,code,det}/hy4-flash.json` captured from the oracle on the real weights.

- [ ] **Step 1: Write the oracle.** A `nn.Module` mirroring Task 8's math, loading only the text backbone with `safetensors` mmap + per-layer streaming (engram/vit/dfly skipped), CPU bf16/fp32, and a `load(model_dir, device_map)` entry point returning an object with `generate(input_ids, ...)`/`forward(input_ids) -> logits` matching what `fsi_capture.capture` calls.
- [ ] **Step 2: Validate the oracle against the engine** on teacher-forced logits of `"The capital of France is"` (`SKIP`/report when either side is absent): top-1 must match, and the report records the top-20 agreement and the max |Δlogit|.
- [ ] **Step 3: Capture** FSI, CODE, DET goldens; run the three gates plus SelfCheck with ≥3 bit-exact cycles on the **release/AOCC** tree.
- [ ] **Step 4: Run** `ctest --test-dir build/linux-release-znver4-aocc -R 'hy4' -j 1` → every stage green (a SKIP is reported, not counted as a pass).
- [ ] **Step 5: Commit** `feat(hy4): the oracle and the parity goldens; SelfCheck Paris green`

---

### Task 11: Perf row

**Files:**
- Modify: `tests/perf/perf_baselines.json`, `tests/test_perf_baseline.cpp` (the `TEST_CASE("perf baseline: hy4-flash")`); the ctest registration comes from the manifest.

- [ ] **Step 1** Run the seeded target and let it write its row through the harness: `ctest --test-dir build/linux-release-znver4-aocc -R '^test_perf_baseline_hy4_flash$' -j 1` (canonical 2048-token protocol, `ignore_eos`).
- [ ] **Step 2** Verify `test_perf_baseline_logic` passes in both directions.
- [ ] **Step 3: Commit** `feat: seed the hy4-flash perf row through the harness`

---

### Task 12: Report + docs close-out (stage 9)

**Files:**
- Create: `docs/model-reports/2026-11-23-Hy4-flash.md`
- Modify: `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md` (+ `docs/open-tasks/live/models/hy4-flash.md`), `docs/MODELS.md` (generated), `docs/model-reports/README.md`

- [ ] **Step 1** Regenerate the catalogue: `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .`
- [ ] **Step 2** Write the report with the two parts the skill requires (publish-ready post, evidence), including: Engram cut (with the oracle's measured effect on the gated prompts), ViT/DFly2 stored-not-applied, the IndexPool reconstruction and its selection-recall measurement, the two-rope and rms-eps traps, the `layer_types` 1:3 hybrid trap, and every gate result verbatim.
- [ ] **Step 3** File the debt rows (Engram, vision, speculative head, any gate that could not be made green) in `docs/OPEN_TASKS.md` + the live file.
- [ ] **Step 4** Final verification: `ctest --preset linux-debug -j 24` and `ctest --test-dir build/linux-debug -R 'hy4-flash' -j 1`; record both, and name anything not run.
- [ ] **Step 5: Commit** `docs: the Hy4-flash stage-9 model report`
