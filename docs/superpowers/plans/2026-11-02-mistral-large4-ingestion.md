# Mistral-Large-4-1T-A52B Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest `Mistral-Large-4-1T-A52B` (72L / 6144H MLA + DSA + 128-expert MoE) into logestix as the `mistral_moe_large4` variant, with converter, loader, kernels, model, tokenizer/tool bridge, parity harness and perf row.

**Architecture:** Frozen passport (`lsxcommon::mistral_large4`), bespoke converter (FP8-E4M3 128x128 tiles -> NVFP4), latent-MLA absorbed decode with the neutral `ExpertStore` + `lsx_nvfp4` layers, exact DSA indexer (top-2048) and cross-layer KV sharing.

**Tech Stack:** C++20 / CUDA 13.4 (Blackwell sm_120), Arrow IPC, Catch2, CMake presets.

## Global Constraints

- CUDA Toolkit 13.4 only; no pre-Blackwell arch.
- No `cudaDeviceSynchronize` in the decode path; one stream.
- Model data in `.cu/.cuh`-reachable headers stays `std::` (no absl containers).
- Every new test gets its label in the SAME commit (`fast`/`convert`/`inference`/`perf`).
- Selfcheck / fsi / code / det targets: `SKIP_RETURN_CODE 4`.
- Frozen geometry: compile-time constants + `static_assert`, cross-checked at Load.
- Guard registries extended in the same commit as the model files.
- Host expert storage: the neutral `ExpertStore` only (no family-private owning map).
- No comments unless asked; match the surrounding style.

---

### Task 1: Passport + naming/guard registration

**Files:**
- Create: `libs/lsxcommon/src/mistral_moe_large4_constants.h`
- Modify: `tests/test_code_naming_guard.cpp:31` (variant tokens), `:44` (family stems unchanged)
- Modify: `tests/test_model_independence_guard.cpp` (`kPassports` row; `consumers`; `kLifecycle` row added in Task 8)
- Test: `tests/test_mistral_moe_large4_convert.cpp` (geometry statics)

**Interfaces:**
- Produces: `lsxcommon::mistral_large4::frozen::{kType,kHidden,kLayers,kVocab,kHeads,kKvLora,kQkNope,kQkRope,kVHead,kNumRouted,kTopK,kMoeIntermediate,kDenseIntermediate,kIndexHeads,kIndexHeadDim,kIndexTopk,kFirstKDense,kRopeLayers,kNopeLayers,kKvSource,kIndexerTypes,kEosId,kPadId}`

- [ ] Step 1: write the failing test asserting every constant and the five per-layer tables (54 RoPE, 18 NoPE, 9 KV-shared, 44 full indexers).
- [ ] Step 2: run `ctest --preset linux-debug -R mistral_moe_large4` -> fails to compile.
- [ ] Step 3: add the header with `static_assert`s, and add `large4` to `SchemeRegex()`'s `var` set.
- [ ] Step 4: build + run; guard tests green.
- [ ] Step 5: commit `feat(mistral-large4): frozen passport and naming registration`.

### Task 2: Converter (synthetic E2E, red first)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_convert.h`, `libs/lsxcommon/src/mistral_moe_large4_convert.cpp`
- Create: `tests/test_mistral_moe_large4_fixture.h`, `tests/test_mistral_moe_large4_convert.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (source list), `tests/CMakeLists.txt` (`convert` target, TIMEOUT 240, `ENVIRONMENT LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`)
- Modify: `apps/lsxforge/*` (`--mistral_large4`)

**Interfaces:**
- Produces: `mistral_large4::mistral_large4_classify(name)`, `mistral_large4_load_topology(path)`, `ConvertMistralLarge4Model(input, output)`, format tag `logestix-mistral-large4-nvfp4`.

- [ ] Step 1: fixture writes `config.json` + synthetic shards with: BF16 globals/norms/router/`wkv_a`/`wkv_b`, FP8 `wq`/`wo`/`indexer.wq_b`/experts/shared/dense-MLP with F32 tile scales, `ssmax_scaling`, `lerope.alpha` (32), one NoPE layer, one shared-KV layer, one shared-indexer layer, vision + `mtp_dspark` ballast.
- [ ] Step 2: test asserts the written index (tag, per-layer tables, `deferred.mtp`, `skipped_tensors`), the pinned/global/layer/`nvfp4` files, the expert tree count, and that BASE+SCALE*FP8 reconstructs (`base relL2 < 0.1`).
- [ ] Step 3: run -> FAIL (converter missing).
- [ ] Step 4: implement the converter: config parse (simdjson), classification, FP8 tile dequant (`gpu::DequantFp8Tile128ToBf16Gpu`, promoted from glm) + `quantize_tensor_gpu("BF16")`, per-layer expert flush, tokenizer bridge with `pretokenizer` 10 (Task 5 adds the WordStream; until then the fixture asserts the stored kind), index.
- [ ] Step 5: run -> PASS; commit `feat(mistral-large4): FP8->NVFP4 converter and synthetic E2E`.

### Task 3: Tokenizer pre-tokenizer kind 12

**Files:**
- Modify: `libs/lsxcommon/src/pretokenizer.{h,cpp}` (`O200kGroupedWordStream`), `libs/lsxcommon/src/tokenizer_convert.cpp` (`policy.pretokenizer == 10` -> `tokenizer_kind = 12`), `libs/lsxcommon/include/lsxcommon/converter.h` (doc)
- Test: `tests/test_tokenizer_mistral_large4.cpp` (`fast`)

**Interfaces:**
- Produces: `tokenizer_kind 12` semantics: o200k case-run alternatives without the contraction suffix, digits right-grouped `1..3`, `[\r\n/]*` after punctuation, `\s*[\r\n]+`, `\s+(?!\S)`, `\s+`.

- [ ] Step 1: failing test on a synthetic vocab: `"1234567"` -> `123|456|7`; `"a1234"` -> `a|1|234`; `"Maße"` case runs; punctuation CRLF tail.
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: implement the stream + wiring.
- [ ] Step 4: PASS; parity check against the real `tokenizer.json` via the host `lsxforge --tokenizer` round-trip later (Task 12).
- [ ] Step 5: commit `feat(mistral-large4): o200k grouped-digit pre-tokenizer`.

### Task 4: Loader

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_load.h`, `libs/lsxcommon/src/mistral_moe_large4_load.cpp`
- Test: `tests/test_mistral_moe_large4_loader.cpp` (`inference`, SKIP 4)

**Interfaces:**
- Produces: `MistralLarge4PinnedWeights`, `LoadMistralLarge4Weights(path, pw, stream, error)`, `FreeMistralLarge4Weights`, `MistralLarge4DimsFromWeights`.

- [ ] Step 1: failing test: synthetic tree -> Load; assert per-layer `wkv_a` present/absent per `kv_source`, `wkv_b` rows (192/256 per head), indexer weights on full layers, `ssmax`/`lerope` uploaded, geometry mismatch fails.
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: implement (views via `LoadTensorBatchViews` / `LoadDenseNvfp4BatchViews` / `LoadSharedExpertBatchViews`; neutral `MoEExpertStore`; `resident_bytes` from allocation sites).
- [ ] Step 4: PASS; commit `feat(mistral-large4): pinned loader`.

### Task 5: Kernels — ssmax, interleaved LeRoPE, router, indexer, absorption

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_kernels.h`, `libs/lsxcommon/src/mistral_moe_large4_kernels.cu`
- Test: `tests/test_mistral_moe_large4_kernels.cu` (`inference`)

**Interfaces:**
- Produces:
  - `BuildLarge4RopeTables(capacity, alpha, cos, sin)` (host, YaRN x exp(alpha))
  - `LaunchSsmaxScale(q, s, b, heads, head_dim, pos, stream)`
  - `LaunchRopeInterleave(vec, heads, nope, rope, cos, sin, pos, stream)`
  - `LaunchRouter(logits_w, x, scores, idx, experts, top_k, stream)`
  - `LaunchIndexerScores(...)` + `LaunchTopkSelect(scores, n, k, idx, count, stream)`
  - `LaunchAbsorbQuery(wk, q_nope, q_abs, ...)`, `LaunchValueProject(wv, ctx, out, ...)`

- [ ] Step 1: failing host-reference tests: YaRN NTK-by-parts at 512/2048 factor with alpha scale; interleaved rotation equals the reference formula on known vectors; ssmax equals `s*log(pos+1)+b`; router equals softmax+group+renorm; indexer scores + exact top-k on a 3000-element vector (tie-break ascending); absorption equals expand-then-attend on random data (tolerance 2e-2 relL2).
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: implement kernels (shared-mem sandbox, deterministic; thread-0 RMW behind barriers).
- [ ] Step 4: PASS; commit `feat(mistral-large4): ssmax/LeRoPE/router/indexer/absorption kernels`.

### Task 6: Attention module (latent cache, KV sharing, DSA gather)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_inference_attn.h`, `libs/lsxcommon/src/mistral_moe_large4_inference_attn.cu`
- Test: `tests/test_mistral_moe_large4_attn.cu` (`inference`)

**Interfaces:**
- Produces: `Large4AttnState` (latent caches, indexer caches, mask cache), `AllocateLarge4AttnState`, `Large4AttentionStep(layer, hidden, pos)`.

- [ ] Step 1: failing test: (a) dense path at pos<=2048 equals the reference `expand_kv` math; (b) shared-KV layer 39 reads layer 35's cache (perturb 35, 39 output changes; perturb 39's absent cache is impossible); (c) shared-indexer layer reuses the previous full mask; (d) >2048 masked path equals a host top-k gather reference; (e) rope layer vs NoPE layer shapes.
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: implement.
- [ ] Step 4: PASS; commit `feat(mistral-large4): latent MLA attention, KV sharing, DSA gather`.

### Task 7: Pipeline + MoE step

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/mistral_moe_large4_inference_pipeline.h`, `libs/lsxcommon/src/mistral_moe_large4_inference_pipeline.cu`
- Test: `tests/test_mistral_moe_large4_pipeline.cpp` (`inference`)

**Interfaces:**
- Produces: `Large4Session`, `AllocateLarge4Session`, `Large4DecodeStep` (token+pos -> logits, includes router + routed/shared MoE + softcap).

- [ ] Step 1: failing test: one full layer graph against a host reference (layer 0 dense MLP and a MoE layer), softcap formula, determinism across two calls.
- [ ] Step 2: FAIL -> implement -> PASS.
- [ ] Step 3: commit `feat(mistral-large4): decode pipeline and MoE step`.

### Task 8: Model class, factory, lifecycle guard

**Files:**
- Create: `libs/lsxcommon/src/mistral_moe_large4_model.{h,cpp}`
- Modify: `libs/lsxcommon/src/model_registry.cpp`, `libs/lsxcommon/CMakeLists.txt`, `tests/test_model_independence_guard.cpp` (`kPassports`, `consumers`, `kLifecycle`), `tests/CMakeLists.txt` (`model` target)
- Test: `tests/test_mistral_moe_large4_model.cpp`

- [ ] Step 1: failing test: `Load` rejects a geometry mismatch; `BuildConversationInputIds` renders the Large-4 frame (`[MODEL_SETTINGS]{"reasoning_effort": "high"}`), `ParseToolCalls` parses `[TOOL_CALLS]weather[ARGS]{"city":"Paris"}`; streaming emits one token per step; pre-set abort returns empty.
- [ ] Step 2: FAIL -> implement -> PASS.
- [ ] Step 3: commit `feat(mistral-large4): model class and factory registration`.

### Task 9: Tool codec render policy

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/mistral_moe_tool_format.h`, `libs/lsxcommon/src/mistral_moe_tool_format.cpp`
- Test: `tests/test_mistral_moe_large4_tool_framing.cpp` (`fast`)

- [ ] Step 1: failing test: Large-4 policy emits `"reasoning_effort": "high"` (default) and honours `low`/`medium`; Small-4 behaviour unchanged.
- [ ] Step 2..5: FAIL -> implement `MistralRenderPolicy` -> PASS -> commit.

### Task 10: SelfCheck + streaming harness

**Files:**
- Create: `tests/test_mistral_moe_large4_selfcheck.cpp`
- Modify: `tests/CMakeLists.txt` (`inference`, `SKIP_RETURN_CODE 4`, env cycles)

- [ ] Step 1: test (3 cycles, bit-exact `final_logits`, Paris top-1) written and labelled.
- [ ] Step 2: run without a converted tree -> SKIP 4.
- [ ] Step 3: commit `test(mistral-large4): selfcheck harness`.

### Task 11: FSI / CODE / DET harnesses

**Files:**
- Create: `tests/test_mistral_moe_large4_{fsi,code,det}_selfcheck.cpp`
- Create: goldens `tests/fixtures/{fsi,code,det}/mistral_large4.json` (template-derived)
- Modify: `tests/test_model_suite_guard.cpp` (`Suites` row, tool verdict, `kGoldenDebt`)

- [ ] Step 1: tests + suite row.
- [ ] Step 2: goldens written from the template (documented provenance in the JSON `_provenance` field).
- [ ] Step 3: commit.

### Task 12: Real-weight conversion

- [ ] Step 1: `cmake --preset linux-debug && cmake --build --preset linux-debug`
- [ ] Step 2: run `lsxforge --mistral_large4=/mnt/fast_storage/huggingface/Mistral-Large-4-1T-A52B` with `TMPDIR=/mnt/fast_storage/tmp`, verify index + counts + spot-decodes.
- [ ] Step 3: commit is NOT made (the tree lives in the vault).

### Task 13: Real-weight gates

- [ ] selfcheck x3, tool round-trip, FSI/CODE/DET, streaming/abort.
- [ ] record results in `docs/SELFCHECK_MATRIX.md` + `docs/OPEN_TASKS.md`.

### Task 14: Perf baseline

- [ ] `test_perf_baseline.cpp` TEST_CASE + `add_test` + JSON entry (measured through the harness).
- [ ] commit `perf(mistral-large4): baseline at 2048 tokens`.

### Task 15: Docs (AGENTS.md structure, OPEN_TASKS, report)

- [ ] Stage-9 report `docs/model-reports/2026-11-02-Mistral-Large-4-1T-A52B.md`.
- [ ] AGENTS.md structure bullet; commit `docs(mistral-large4): ingestion report`.
