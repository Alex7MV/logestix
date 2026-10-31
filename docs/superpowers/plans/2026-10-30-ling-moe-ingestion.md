# Ling-3.1-flash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> (this repo forbids subagents — AGENTS.md "Autonomous Mode"). Steps use
> checkbox (`- [ ]`) syntax.

**Goal:** Ingest `Ling-3.1-flash` (`BailingMoeV3ForCausalLM`) as the new
`ling` family with the full Model Ingestion Protocol, TDD, and every runnable
lifecycle gate green.

**Architecture:** Hybrid 63×KDA + 9×gated-MLA + 68×MoE(512 top-8) decoder.
Dense side resident as NVFP4 (≈10.2 GiB), 298 GB routed-expert tree mmap'd
through the neutral `ExpertStore` with a device slot LRU. New family files
reuse the neutral NVFP4/GEMV layer and the verified GLM-5.3 KDA math.

**Tech Stack:** C++20 / CUDA 13.4 (sm_120), CMake presets, Catch2, Arrow,
simdjson, aba/absl (host only).

Spec: `docs/superpowers/specs/2026-10-30-ling-3.1-flash-ingestion-design.md`.

## Global Constraints

- Frozen geometry only; every field cross-checked at Load.
- One stream, no per-token `cudaDeviceSynchronize`; NVFP4 only through
  `lsx_nvfp4`; no abseil containers in `.cu`/`.cuh`-reachable headers.
- Norms/bias/conv/A_log/dt_bias/router stay high precision; FP8 linears →
  NVFP4; embed/lm_head stay FP16 on device.
- Routed experts: neutral zero-copy `ExpertStore`; never an owning map/arena.
- KV addressing through `KvCacheGeometry`.
- Every new test gets its ctest label in the same commit.
- Guard registries extended in the same commit as the files.
- No comments unless the surrounding file already carries them.

---

### Task 0: Registry groundwork (red first)

**Files:**
- Modify: `tests/test_code_naming_guard.cpp` (`SchemeRegex`, `FamilyStems`,
  `AllowedCommon`)
- Modify: `tests/test_model_independence_guard.cpp` (`kPassports`, `consumers`,
  `kLifecycle`)
- Modify: `tests/test_model_suite_guard.cpp` (`FamilyStems`, `Suites`,
  tool verdict)
- Modify: `tests/CMakeLists.txt`, `libs/lsxcommon/CMakeLists.txt`
- Create: empty placeholder-free headers only when Task 2 starts.

- [ ] **Step 1:** Add `ling` to both `FamilyStems()` sets and to
  `SchemeRegex()`'s family alternation; add `ling_moe_tool_format` to
  `AllowedCommon()`.
- [ ] **Step 2:** Add the `kPassports` row
  `{"ling", "ling_moe_constants.h", "lsxcommon::ling::frozen"}`, the
  `consumers` entry `ling_moe_model.cpp`, and the `kLifecycle` row
  `{"ling", {"ling_moe_inference.cu", "ling_moe_model.cpp"}}`.
- [ ] **Step 3:** Add the `Suites()` row (convert/load/model/kernels/selfcheck/
  fsi/code/det/perf) and the tool verdict row pointing at
  `test_ling_tool_framing`.
- [ ] **Step 4:** Build the guards. Expected: the new guard rows FAIL because
  the files/targets do not exist yet — that is the red state.
- [ ] **Step 5:** Commit `test(ling): register the ling family in the guard
  registries`.

### Task 1: Constants passport

**Files:** Create `libs/lsxcommon/src/ling_moe_constants.h`.

- [ ] **Step 1:** Write `namespace lsxcommon::ling::frozen` with `kHidden=4096`,
  `kLayers=72`, `kMtpLayer=72`, `kVocab=157184`, `kHeads=64`, `kKvHeads=64`,
  `kHeadDim=128`, `kQkHeadDim=192`, `kQkNope=128`, `kQkRope=64`, `kVHead=128`,
  `kQLora=1536`, `kKvLora=512`, `kConvKernel=4`, `kNumExperts=512`, `kTopK=8`,
  `kMoeIntermediate=1280`, `kDenseIntermediate=10240`, `kFirstKDense=4`,
  `kMlaLayerCount=9`, `kKdaLayerCount=63`, `kMoeLayers=68`,
  `kFormatTag="logestix-ling-nvfp4"`, `kType="ling"`,
  `kModelName="Ling-3.1-flash"`, `kRoutedScalingNum=25/Den=10`,
  `kGroupCount=8`, `kTopkGroup=4`, `kRmsEps=1e-6`,
  `kRopeTheta=6000000.0`, `kBosId=-1`, `kEosId=156895`, `kPadId=156892`.
- [ ] **Step 2:** Layer tables `kMlaLayerIds[9] = {7,15,23,31,39,47,55,63,71}`
  and `IsKdaLayer/IsMlaLayer/IsMoeLayer` + `static_assert` coverage
  (`9 + 63 == 72`, top-k ≤ experts, hidden % 128 == 0).
- [ ] **Step 3:** Tensor-kind enum (below) + `ParsedLingTensor`.
- [ ] **Step 4:** Commit with the classifier test in Task 2.

### Task 2: Classifier + config parse (TDD)

**Files:** Create `include/lsxcommon/ling_moe_convert.h`,
`src/ling_moe_convert.cpp`; test `tests/test_ling_moe_convert.cpp`
(label `convert`, TIMEOUT 120).

**Interfaces:**
- `enum class LingTensorKind { Unknown, Embed, LmHead, FinalNorm, InputNorm,
  PostNorm, KdaQ, KdaK, KdaV, KdaO, KdaConv, KdaALog, KdaDtBias, KdaB,
  KdaF, KdaG, KdaONorm, MlaQa, MlaQb, MlaKvA, MlaKvB, MlaDense, MlaG, MlaNorm,
  RouterGate, RouterBias, DenseMlpGate, DenseMlpUp, DenseMlpDown, SharedGate,
  SharedUp, SharedDown, RoutedGate, RoutedDown, RoutedUp, Mtp };`
- `ParsedLingTensor classify_ling_tensor(const std::string&);`
- `bool ling_tensor_is_nvfp4(LingTensorKind);`
- `LingTopology ParseLingConfig(const std::string&);`
  fields: hidden/layers/vocab/heads/kv_heads/head_dim/qk_nope/qk_rope/v_head/
  q_lora/kv_lora/conv_kernel/num_experts/top_k/n_group/topk_group/
  moe_intermediate/dense_intermediate/shared_intermediate/first_k_dense/
  rms_eps/rope_theta/routed_scale/eos/pad; `mla_layers`, `kda_layers`,
  `moe_layers`; `error_msg`.

- [ ] **Step 1: failing test** — classify representative names:
  `model.layers.0.attention.q_proj.weight` → KdaQ;
  `model.layers.0.attention.f_proj.weight` → KdaF;
  `model.layers.7.attention.kv_a_proj_with_mqa.weight` → MlaKvA;
  `model.layers.7.attention.dense.weight` → MlaDense;
  `model.layers.4.mlp.gate.expert_bias` → RouterBias;
  `model.layers.4.mlp.experts.17.down_proj.weight` → RoutedDown;
  `model.layers.72.eh_proj.weight` → Mtp; unknown → Unknown.
  Assert `ling_tensor_is_nvfp4` is true for Kda{Q,K,V,O,F,G}, Mla{Qa,Qb,KvA,KvB,Dense},
  DenseMlp*, Shared*, Routed*, and false for norms/conv/A_log/dt_bias/b_proj/
  o_norm/MLA norms/g_proj/router/embed/head/eh_proj.
- [ ] **Step 2: run** `ctest --preset linux-debug -R test_ling_moe_convert`
  → FAIL (not defined).
- [ ] **Step 3:** implement classifier (accept optional `model.` prefix,
  `attention.*` leaves, `model.word_embeddings.weight`, `model.norm.weight`).
- [ ] **Step 4:** add a `ParseLingConfig` test against the real `config.json`
  guarded by `if (!fs::exists(...)) SKIP(...)`; assert the 9-layer MLA table is
  derived from `layer_group_size` + `num_hidden_layers` and that
  `mtp_use_kda == false`.
- [ ] **Step 5:** commit `feat(ling): topology passport and tensor classifier`.

### Task 3: FP8 dequant (GPU + host twin)

**Files:** `src/ling_moe_dequant.cu`, `src/ling_moe_dequant.h`,
`tests/test_ling_moe_dequant.cu` (label `fast`).

**Interfaces:** `arrow::Status LingDequantFp8Block128ToBf16Gpu(...)` (same
contract as gigachat, partial last tile clamped);
`float LingFp8E4m3Decode(uint8_t);` host twin of the kernel.

- [ ] **Step 1: failing test** — 3×130 tile grid with a known FP8 byte and
  scale per tile; compare GPU output with the host twin; include a partial row
  tile (rows 130, scale rows 2) and assert the clamp.
- [ ] **Step 2:** implement (adapt `gigachat_moe_dequant.cu`; keep the
  `__nv_cvt_fp8_to_halfraw(..., __NV_E4M3)` decode).
- [ ] **Step 3:** commit `feat(ling): FP8 block-128 dequant`.

### Task 4: Converter pipeline (TDD, synthetic E2E)

**Files:** `src/ling_moe_convert.cpp`; `tests/test_ling_moe_fixture.h`,
`tests/test_ling_moe_convert.cpp` (label `convert`, TIMEOUT 120,
`ENVIRONMENT LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`).

**Interfaces:** `arrow::Status ConvertLingModel(const std::string& in,
const std::string& out);`

- [ ] **Step 1: failing test** — synthetic checkpoint (H=256, 4 layers so that
  MLA hits one layer with `layer_group_size 2`, 8 experts, vocab 512): write
  `config.json` + shards with `synthtest::WriteSafetensors`; every FP8 linear
  carries a `weight_scale_inv` `[ceil(R/128),ceil(C/128)]` F32; include MTP
  layer and assert it is counted in `deferred.mtp`, absent from
  `experts/`, `pinned/`.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** implement:
  1. parse + frozen gate (skip under synthetic),
  2. inventory + unknown-name rejection + integrity check,
  3. dense pass: FP8 → `LingDequantFp8Block128ToBf16Gpu` → `quantize_tensor_gpu`;
     BF16/F32 raw; global; router; shared,
  4. expert-major second pass: `[gate=w1, down=w2, up=w3]`, expert-major across
     shards so the 535 GB source is never resident,
  5. tokenizer (`pretokenizer=3`, bos -1, eos/pad from config),
  6. `EstimateVram` with `mem_class = RoutedExpert` for experts,
  7. index JSON with `mla_layers`/`kda_layers`/`moe_layers`/`dense_layers`
     and `deferred.mtp`,
  8. re-open every arrow output.
- [ ] **Step 4:** add LP solver-free parity assertion: decode one written
  NVFP4 record on the host (`nvfp4::DecodeBlock` or the existing helper) and
  assert relL2(base, source) < 0.12 and global_scale ≠ 1 (guards the 7.01×
  hot-logit class).
- [ ] **Step 5:** register `lsxforge --ling` in `apps/lsxforge`.
- [ ] **Step 6:** commit `feat(ling): FP8→NVFP4 converter + synthetic E2E`.

### Task 5: Loader

**Files:** `include/lsxcommon/ling_moe_load.h`, `src/ling_moe_load.cpp`;
`tests/test_ling_moe_loader.cpp` (label `fast`).

**Interfaces:**
- `struct LingTopology { ... std::string format; VramProfile vram; };`
- `LingTopology LoadLingTopology(const std::string& path);`
- `struct LingLayerWeights { bool is_mla; QwenNvfp4Weight kda_q,kda_k,v,o,f,g;
  __half *conv_q,*conv_k,*conv_v,*b,*f_conv?; float *A_log,*dt_bias;
  __half *o_norm; QwenNvfp4Weight mla_q_a,q_b,kv_a,kv_b,dense; __half *q_a_norm,
  *kv_a_norm,*mla_g; uint8_t *router; float *router_bias; QwenNvfp4Weight
  shared_w1,w2,w3; };`
- `struct LingPinnedWeights { LingTopology top; QwenNvfp4Weight embed,head;
  __half *final_norm; std::vector<LingLayerWeights> layers; uint64_t
  resident_bytes; void Free(); };`
- `bool LoadLingDenseWeights(...)`, `bool LoadLingExpertStore(...)`,
  `LingExpertDb` aliasing `moe::MoEExpertDb`.

- [ ] **Step 1: failing test** — load the synthetic converted tree, assert
  layer kinds, NVFP4 shapes, router rows=experts, shared shapes, expert store
  entry decodes non-null views, and `Free()` releases every allocation
  (second load in the same process succeeds).
- [ ] **Step 2:** implement with `LoadDenseNvfp4BatchViews` (view readers) and
  `ExpertStore::Load`; geometry cross-check against the passport.
- [ ] **Step 3:** commit `feat(ling): loader + expert store wiring`.

### Task 6: Kernels (TDD with host twins)

**Files:** `include/lsxcommon/ling_moe_inference.h`,
`src/ling_moe_inference.cu`; `tests/test_ling_moe_kernels.cu` (label `fast`).

**Interfaces (all `cudaError_t`, one stream):**
- `RunLingConv1dF32`, `RunLingKdaStepF32` (test twins, FP32)
- `RunLingKdaDecode` (conv+silu, fused q/k l2norm with eps 1e-6 and q scale
  1/sqrt(128), gating `-5·sigmoid(exp(A_log)·(f+dt_bias))`, delta step, gated
  RMSNorm, o_proj)
- `RunLingRopeInterleaved` (pairwise on the 64 rope dims, θ 6e6)
- `RunLingMlaAttn` (causal over a `KvCacheGeometry` cache, scale 1/sqrt(192))
- `RunLingRouterTopK` (sigmoid + bias, group-limited top-2-per-group scoring,
  top-4 groups, top-8 experts, renorm ×2.5, thread-0 RMW)
- `RunLingSharedSwiGLU`, `RunLingMlaDenseGate`

- [ ] **Step 1: failing tests** — numpy-equivalent C++ host references:
  KDA step on a 2-head 4-dim state, l2norm eps, safe-gate decay, gated norm;
  router on 16 experts / 4 groups with a bias that flips the selection;
  interleaved RoPE pair `(x0,x1)`; MLA attention over 3 positions.
- [ ] **Step 2:** implement kernels (KDA adapted from
  `glm_moe_flash_inference.cu` with direct f/g projections; MLA non-absorbed;
  router new).
- [ ] **Step 3:** bit-exactness: same input twice → same bits; no
  `cudaDeviceSynchronize` inside the layer path.
- [ ] **Step 4:** commit `feat(ling): KDA/MLA/router kernels + host parity`.

### Task 7: Model + lifecycle + streaming

**Files:** `src/ling_moe_model.h`, `src/ling_moe_model.cpp`,
`src/model_registry.cpp`, `tests/test_ling_moe_model.cpp`,
`tests/test_ling_moe_model_lifecycle.cu` (labels `fast`).

- [ ] **Step 1: failing tests** — `kType == "ling"` registered; Load rejects a
  wrong-geometry index; pre-set `abort_flag` returns without decoding; the
  token callback receives one `Emit` per decoded token; second `Infer`
  continues the KV session; `SelfCheck` answers Paris.
- [ ] **Step 2:** implement `LingMoeModel final : ModelBase` with
  `RunUnderResidencyLock`, `GenerationStream`, `TeardownContext`,
  `AcquirePermanentWeights` + `AcquireExpertCacheLease`, filled
  `InferenceResult`, session state under the lock; register `ling`.
- [ ] **Step 3:** `KvLayout()` returns `kPosMajor`.
- [ ] **Step 4:** commit `feat(ling): model class, registration, lifecycle`.

### Task 8: Tokenizer parity + tool bridge

**Files:** `src/ling_moe_tool_format.cpp`,
`include/lsxcommon/ling_moe_tool_format.h`,
`tests/test_ling_tokenizer_parity.cpp` (label `fast`),
`tests/test_ling_tool_framing.cpp` (label `fast`),
`tools/ling_tokenizer_capture.py`.

- [ ] **Step 1:** capture 27 prompts + a mixed-script set with the HF tokenizer
  (local `transformers`), commit the expected ids in the test (or a fixture).
- [ ] **Step 2: failing test** — the converter-built tokenizer must reproduce
  every id (`tokenizer_kind == 3`), including `<role>` as a single control id
  and the `<tool_call>` markers.
- [ ] **Step 3:** implement `RenderLingConversation` (exact template:
  `<role>SYSTEM</role>...<|role_end|>`; thinking on/off line; assistant
  `<role>ASSISTANT</role>\n<think>...</think>`; tool results
  `<role>OBSERVATION</role>\n<tool_response>...`), control markers injected by
  `arrow_tokenizer_lookup`, free text BPE-encoded; `ParseLingToolCalls` scans
  decoded text for `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>
  </tool_call>` and falls back to `tools::ParseToolCallsText`.
- [ ] **Step 4:** test a tool round-trip (render → parse) and single-prompt
  framing.
- [ ] **Step 5:** commit `feat(ling): o200k tokenizer parity + native tool
  codec`.

### Task 9: SelfCheck + FSI/CODE/DET debt rows

**Files:** `tests/test_ling_moe_selfcheck.cpp` (label `inference`, SKIP 4),
`tests/test_ling_moe_fsi_selfcheck.cpp`,
`tests/test_ling_moe_code_selfcheck.cpp`,
`tests/test_ling_moe_det_selfcheck.cpp` (all label `inference`, SKIP 4),
`tests/CMakeLists.txt`, `tests/test_model_suite_guard.cpp` (`kGoldenDebt`).

- [ ] **Step 1:** selfcheck test runs 3 cycles (`LSX_TEST_LING_SELFCHECK_CYCLES`)
  and requires bit-exact `logits_hash` + exit 0; skip when the vault is absent.
- [ ] **Step 2:** debt rows + README note that the source oracle cannot run on
  this card (560 B, FLA GPU-only).
- [ ] **Step 3:** commit `test(ling): selfcheck + declared golden debt`.

### Task 10: Perf baseline

**Files:** `tests/perf/perf_baselines.json`, `tests/test_perf_baseline.cpp`,
`tests/CMakeLists.txt`.

- [ ] **Step 1:** add `TEST_CASE("perf baseline: ling")` + `add_test(NAME
  test_perf_baseline_ling ...)` label `perf`, SKIP 4.
- [ ] **Step 2:** run once on real weights to seed the row through the harness
  (never hand-edit).
- [ ] **Step 3:** commit `perf(ling): seed the baseline through the harness`.

### Task 11: Real conversion + acceptance

- [ ] **Step 1:** `lsxforge --ling --input /mnt/fast_storage/huggingface/Ling-3.1-flash
  --output <vault>/Ling-3.1-flash` (experts ~255 GiB; run on
  `/mnt/fast_storage/tmp` scratch).
- [ ] **Step 2:** `lsxfabric --localmodel` SelfCheck ×3, streaming + cancel,
  tool round-trip.
- [ ] **Step 3:** run the full gate sweep and record the matrix in
  `docs/SELFCHECK_MATRIX.md` / `docs/OPEN_TASKS.md` §R7.
- [ ] **Step 4:** commit `feat(ling): real-weight selfcheck + gate matrix`.

### Task 12: Docs

- [ ] `AGENTS.md` Structure entry, `docs/OPEN_TASKS.md` §R7,
  `docs/SELFCHECK_MATRIX.md`, skill map note.
- [ ] commit `docs(ling): ingestion record`.

## Verification

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug                 # guard + fast
ctest --preset linux-debug-convert -R ling
ctest --preset linux-debug-inference -R ling
ctest --preset linux-debug -R ling
```
