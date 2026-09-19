---
name: ingesting-a-model
description: Use when adding, adapting or ingesting a new model architecture into logestix, or porting an existing family to a new variant — lsxforge converter, NVFP4 packing, IModel + LsxModelFactory registration, tokenizer arrow bridge, CUDA inference kernels, token passport, SelfCheck harness, and the ctest lifecycle stages. Triggers - new model, model adaptation, Model Ingestion Protocol, MODEL INGEST INVARIANT, kType, frozen geometry, selfcheck Paris.
---

# Ingesting a new model into logestix

## Overview

**MODEL INGEST INVARIANT**: adding a model is a fixed 8-stage sequence. Every
agent MUST follow the order; skipping or reordering a stage is a broken build.
Reference implementations (real code anchors, not placeholders): GLM-5.2,
DeepSeek-V4-Flash, GLM-5.3-Flash, Qwen3.5-9B, Kimi-K2.7-Code.

This skill is the single source of truth for the protocol, the file-naming
convention and the guard registries. `AGENTS.md` only points here.

Work in **small verified steps**: each stage lands together with the test that
pins it, in the same commit (see TDD below). Verification commands are at the
end.

**REQUIRED SUB-SKILL:** use `test-driven-development` for every stage.

## Iron rules (apply to all 8 stages)

- **TDD.** No production code without a failing test first. A new test MUST get
  its ctest label in the same commit, or `test_ctest_labels_guard` goes red.
- **Frozen geometry.** hidden_size / layers / heads / experts / top-k are locked
  as compile-time constants with `static_assert` (runtime config reads are
  legacy — new models do not do it).
- **Kernel-level rules** — one stream with no per-token `cudaDeviceSynchronize`,
  the shared-memory sandbox, NVFP4 GEMM only through the model-neutral
  `lsx_inference_nvfp4` layer, CCCL/cub priority, tile programming, alignment
  and the NVIDIA docs mirror: **REQUIRED SUB-SKILL**
  `nvfp4-blackwell-conventions` (do not restate them here).
- **No abseil containers in `.cu`/`.cuh`-reachable headers.** Containers and
  streams in device TUs stay `std::`.
- **Norms and biases are never quantized** — they stay FP16/BF16.
- **Model independence.** Each model opens
  `LsxFastMathScope(EffectiveFastMathMode(cfg_))` inside its own `Infer`.

## The 8 stages

### 1. Topology and structure passport (lsxforge)

- Extend the **C++** pipeline in `apps/lsxforge` (never a Python tool) to parse
  the original `config.json` and the safetensors (pattern:
  `lsxcommon::qwen::ConvertQwenModel`, `libs/lsxcommon/src/qwen_dense_convert.cpp`).
- Emit a structural manifest stripped of multimodal ballast.
- **PATH DIRECTION**: results are saved under the central `model-metadata/`
  directory through the single path authority
  `lsxcommon::lsx_model_metadata(rel)` / `lsx_model_metadata_root()`
  (`libs/lsxcommon/src/path_registry.cpp`). Vault-root invariants and path
  switches live in `path_registry` only — no hardcoded paths in converters.
- Test: see **Converter test rules** below — a `convert`-labelled synthetic E2E.

#### Converter test rules (fixed)

Reference: `tests/test_glm_moe_flash_convert.cpp` (synthetic E2E) +
`tests/test_synth_checkpoint.h` (family-neutral primitives
`synthtest::FakeTensor`/`WriteSafetensors`/`MakeBf16*`/`TensorNamesIn`; family
fixtures may re-export them, e.g. `qwentest::` in `test_qwen_dense_fixture.h`).

1. **Label `convert` in the same commit**, and a mandatory bounded TIMEOUT
   (≤600 s; canonically 120 s) — guard `CheckConvertTimeout`. The TIMEOUT is
   the mechanical defence against a test that actually converts a model.
2. **Converting a real model in a test is forbidden.** Run the converter E2E
   pipeline on a synthetic checkpoint: `config.json` + `*.safetensors` written
   with `WriteSafetensors` into a temporary directory.
3. **Synthetic geometry**: the converter's frozen gates are relaxed with
   `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` — a **runtime env var** set through
   `set_tests_properties(... PROPERTIES ENVIRONMENT ...)`, NOT a compile
   define (a compile define does not export env).
4. **The fixture must cover every pipeline path**: globals (embed / lm_head /
   final norm — without them the converter errors), BF16→NVFP4, FP8→NVFP4 with
   scale tiles, raw tensors, routed experts (all 3 projections), empty layers
   of the frozen range, and the `model.index.json` layout (format tag +
   kda/dsa/moe lists).
5. **FP8 fixture tensors**: sides a multiple of 128 (GPU kernel); scale tile
   `[ceil(R/128), ceil(C/128)]` F32; safetensors dtype strings `BF16`,
   `F8_E4M3`, `F32`.
6. **A real checkpoint is allowed only to read one light file** (e.g.
   `config.json` in `ParseFlashConfig`) with `SKIP()` when the model is absent;
   full weight reads/conversions never. Synthetic fixtures do not replace
   tests that validate the real format's field names.

### 2. Quantization and compression (NVFP4)

- Configure the quantization engine for total 4-bit quantization of dense
  linear matrices: E2M1 nibbles (2 per byte) + per-block E4M3 scales + a float
  `global_scale` (pattern: `QwenNvfp4Weight`).
- Routed/shared MoE experts are packed per expert; keep the repo layout so the
  inference kernels can consume it unchanged (no repack at upload unless
  documented).

### 3. Registration in the factory (LsxModelFactory)

- Register the string `kType` in `libs/lsxcommon/src/model_registry.cpp` via
  `LsxModelFactory::Register(kType, ...)` (pattern: `QwenV5Model::kType =
  "qwen3.5-9b"`).
- Implement a `final` `IModel` subclass under `libs/lsxcommon/src`, with
  `Load` / `Infer` / `Unload` / `Inspect` / `SelfCheck` / `Tokenizer`.
- Lock the geometry with constants + `static_assert` **and** cross-check the
  loaded passport against them (a frozen-geometry mismatch must fail Load).
- Extend `test_model_independence_guard.cpp`'s `consumers` list with the new
  model file in the same commit.

### 4. Per-byte tokenization (Token Bridge)

- Register the BPE vocabulary through `BuildTokenizerArrowFiles`
  (`libs/lsxcommon/src/tokenizer_convert.cpp`); merges may be string `"a b"` or
  array `["a","b"]`.
- **ALIGNMENT INVARIANT**: read `bos_id`/`eos_id`/`pad_id` from the model config
  (do not hardcode in the kernel); generate the service prefixes exactly as the
  original tokenizer does. `bos_id >= 0` → a real begin-of-sentence id, else
  `bos_id = -1`.

### 5. Low-level CUDA kernels (inference.cu)

- Write/extend the inference kernels for the model's attention specifics
  (hybrid example: GatedDeltaNet + full attention in
  `libs/lsxcommon/src/qwen_inference.cu`).
- Use the model-neutral NVFP4 layer (rule above); no family-owned dispatcher.
- **Stream invariant** and **stride invariant**: all Q/K/V address offsets follow
  the physical tensor geometry — no overlap of adjacent time intervals in
  strided conv/recurrence buffers.
- Isolate CUDA `.cu` files in their own OBJECT library (`-fmad=false` when
  bit-exact math is required).

### 6. Memory synchronization (Shared Memory Sandbox)

- Apply the shared-memory sandbox and deterministic-reduction rules from
  `nvfp4-blackwell-conventions` (thread-0-only RMW behind `__syncthreads()`;
  no `atomicAdd` in anything feeding `final_logits`). Reference implementation:
  `full_attn_softmax_kernel`.

### 7. Thread-isolated mapping (LsxTokenTranslator)

- Update the binary token-translation passport
  (`lsx_model_metadata("passport/passport.bin")`, built with
  `lsxnetwork::BuildPassport` / `WritePassport`).
- Integrate the translator as a private `alignas(64)` member of the network
  worker `IoWorker` (`libs/lsxnetwork/include/lsxnetwork/io_worker.h`).
- `Load()` (Win32 `MapViewOfFile(PAGE_READONLY)` / POSIX `mmap(PROT_READ)`) MUST
  run strictly AFTER the hardware core pinning (`CpuAffinity::PinToCores0To3`)
  and BEFORE the readiness `set_value()`.

### 8. TDD and SelfCheck harness

- Implement an incremental `SelfCheck()` (1→2 tokens) (pattern:
  `QwenV5Model::SelfCheck`).
- The control prompt `"The capital of France is"` MUST deterministically answer
  `"Paris"` (top-1 on the first decoded token / short prefix),
  `r.selfcheck_passed == true`.
- Add `tests/test_<model>_selfcheck.cpp` running ≥3 stress cycles with
  bit-exact `final_logits` determinism and `exit 0`; cycle count overridable by
  an env var (e.g. `LSX_TEST_<MODEL>_SELFCHECK_CYCLES`).
- The selfcheck ctest target MUST set `SKIP_RETURN_CODE 4` (real weights are
  optional; absence skips, never fails).
- Add the perf baseline: an entry in `tests/perf/perf_baselines.json` plus a
  `test_perf_baseline_<model>` target (label `perf`), and record the measured
  t/s. Add **both** — a JSON entry without a target is an unguarded baseline.
- The determinism/correctness discipline these tests implement (bit-exact vs
  tolerance class, canonical-length validation, revert-on-red-determinism,
  `SKIP_RETURN_CODE 4`) is `Verifying a numeric change` in
  `perf-tuning-a-model`.

## Model code file-naming convention

```
^<family>_<type>(_<variant>)?(_exp)?_<role>(_<subsystem>)?$
```
in `libs/lsxcommon/{src,include}` and `tests/`. The **regex in
`tests/test_code_naming_guard.cpp` (`SchemeRegex()`) is the contract** — the
token sets below are what it currently accepts.

- `family` ∈ {glm, qwen, gemma, kimi, deepseek}
- `type` ∈ {moe, dense} — always required
- `variant` ∈ {27b, flash, 36, v41} — only to disambiguate two models of one `type`
- `_exp` infix (after type/variant, before role): the Qwen3.8 "Flash-Next"
  (`qwen4_exp`) marker, e.g. `qwen_moe_exp_convert.cpp`
- `subsystem` — accepted ONLY on the `inference` role:
  {attn, csa, head, moe, pipeline, exp}
- `role` ∈ {convert, load, repack, load_internal, model, inference, kernels,
  expert_kernels, expert_cache, device_math, dequant, norm_kernels, fp8, constants}
- Family-prefixed files that intentionally carry NO type marker (allowlist in
  `AllowedCommon()`): `qwen_inference`, `qwen_base_model`,
  `qwen_inference_load`, `qwen_device_math`, `qwen_reduce`, `gemma_base_model`.
- The legacy glued prefix `gemma4` is recognised.
- Model versions are NOT in file names; they live in class names and `kType`.
- Files without a family stem are model-neutral infrastructure and are not
  scanned by the naming guard.
- **A new family / variant / role / subsystem token must be added to
  `SchemeRegex()` in the SAME commit as the files that use it** — otherwise the
  guard rejects the file names.

## Guard registries — extend in the SAME commit as the model

These guards are the machine-enforced part of this protocol. Adding a model
without updating them fails CI.

| Guard (in `tests/`) | What to extend | Rule it enforces |
|---|---|---|
| `test_code_naming_guard.cpp` | `SchemeRegex()` token sets, `AllowedCommon()` allowlist | the naming convention above |
| `test_model_independence_guard.cpp` | `kPassports` row (`model_type`, passport `header`, owning `ns`) + `consumers` (new model `.cpp`) | unique frozen-geometry passport per model, one constants header per model in its own namespace, no passport→passport includes, no cross-family geometry borrowing, `LsxFastMathScope` in `Infer` |
| `test_model_suite_guard.cpp` | `Suites()` row: convert → load → model → kernels → selfcheck | full autonomous lifecycle; every stage target registered in `tests/CMakeLists.txt`; `SKIP_RETURN_CODE 4` on selfcheck; no cross-family test fixtures; no vault loading in loader/model stages; synth primitives only in `test_synth_checkpoint.h` |
| `test_ctest_labels_guard.cpp` | — (just label new tests) | every test carries a label in the same commit |
| `test_absl_format_guard.cpp` | — | `absl::StrFormat` is printf-style (`%s/%d/%u/%f`), never `{}` |

## Definition of done

- [ ] `kType` registered; `Load` forbids a frozen-geometry mismatch.
- [ ] Converter test (label `convert`, bounded TIMEOUT, synthetic checkpoint).
- [ ] Loader/model/kernels tests present and labelled.
- [ ] `test_<model>_selfcheck.cpp` ≥3 cycles, bit-exact, `SKIP_RETURN_CODE 4`.
- [ ] perf entry in `tests/perf/perf_baselines.json` + `test_perf_baseline_<model>`.
- [ ] All four guard registries above updated.
- [ ] `AGENTS.md` Structure section gets the new files; `test_model_suite_guard`
      and `test_code_naming_guard` green.

## Verification

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug                                  # fast only
ctest --preset linux-debug-convert                          # converter sweep
ctest --preset linux-debug-inference                        # model/CUDA sweep
cmake -DLSX_PRESET=linux-debug -DLSX_LIST=ON -P tools/test_router.cmake   # dry-run router
```

The four guard tests above MUST be green before the model is considered
ingested.

## Common mistakes

- Adding kernels or a converter without the failing test first (TDD violation).
- Reading geometry from `config.json` at runtime instead of frozen constants.
- Violating a kernel-level rule that already lives in
  `nvfp4-blackwell-conventions` (a family-named NVFP4 dispatcher, a shared-memory
  race, an `atomicAdd` feeding `final_logits`).
- Forgetting a ctest label, or a selfcheck target without `SKIP_RETURN_CODE 4`
  (a skip then reports as a failure).
- Loading the real model vault inside a loader/model-stage test (forbidden —
  vault integration belongs to selfcheck/perf suites; loader/model stages use
  synthetic converted fixtures).
- Borrowing another family's test fixture instead of
  `tests/test_synth_checkpoint.h`.
