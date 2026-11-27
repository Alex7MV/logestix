# Nemotron-3-Ultra-550B ingestion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` as its own lineage in the `nemotron` manifest family — `kType = "nemotron-ultra-550b-a55b"` — and take it through every lifecycle stage on the real weights, ending in a stage-9 report.

**Architecture:** Separate implementation lineage (approach A, operator decision 2026-11-10): own frozen passport (`libs/lsxcommon/src/nemotron_moe_ultra_constants.h`, namespace `lsxcommon::nemotron_ultra::frozen`), own converter / index / device loader / routed-expert store+LRU / CUDA kernels / per-token pipeline / model leaf / tool codec under the `nemotron_moe_ultra_*` and `nemotron_ultra_*` file stems. Model-neutral layers are still mandatory and shared: `lsx_nvfp4` (only NVFP4 path), `lsx_gemv` (only GEMM path, cuBLAS is not linked), neutral `ExpertStore<TEntry>` / `ExpertDeviceCache`, `ArrowTokenizerDb`, `ModelBase`, `lsx_device_math.h`, `lsx_reduce.cuh`. The 30B Lightning lineage is not edited. Precedent for this shape: `granite-switch` (family `granite`, own namespace, own TUs).

**Tech Stack:** C++20 / C23, CUDA 13.5 toolkit (driver 13.4, minor-version compatibility mode), CMake presets, Apache Arrow 26.0.1, Catch2 v3.16.1, NVFP4 (E2M1 + per-16 E4M3 + global scale), simdjson (config parsing), abseil (host C++).

**Spec:** `docs/superpowers/specs/2026-11-10-nemotron-ultra-550b-ingestion-design.md`

## Global Constraints

- CUDA arch comes from `LSX_CUDA_ARCH` via `lsx_cuda_gencode_flags()`; every device target sets `CUDA_ARCHITECTURES OFF`. Never pass `LSX_CUDA_ARCH` to `CUDA_ARCHITECTURES`.
- Target Blackwell only (CC 12.0/12.1). No Hopper/Ada/Ampere fallbacks. 48 warps/SM max.
- `cuBLAS` is not linked: all GEMMs go through `lsx_gemv` (`gemv::RunLxGemmVec`, `RunLsxNvfp4Gemm`, `RunLsxNvfp4GemmSeq1GemmVExpertBatch`). All NVFP4 math goes through `lsx_nvfp4`.
- Logging: spdlog / `lsxcommon::log` only. No `std::cout`, no `printf` in `libs/` or `apps/`.
- `absl::StrFormat` is printf-style: `%s`/`%d`, never `{}`.
- Headers reachable from `.cu`/`.cuh` must NOT include `<absl/container/*>`. Host-only C++ prefers `absl::`.
- Arrow reads are mmap-only (`arrow::io::MemoryMappedFile::Open`); a retained payload comes from a views reader, never `UnpackBatch`.
- No CLI speed flags. The only numeric run flag is `--bit-exact` (`FastMathMode::kOff`); the family opens `LsxFastMathScope(EffectiveFastMathMode(cfg_))` per `Infer`.
- Every device-labelled target declares `RESOURCE_LOCK "gpu"`; one GPU workload at a time. `fast` targets stay unlocked.
- Tests register through the generated `model_targets.cmake` (manifest row) plus `add_executable` wiring in `tests/CMakeLists.txt`. Test registration uses `lsx_add_test`/`lsx_add_cuda_test` for non-model targets.
- Every skip-capable target sets `SKIP_RETURN_CODE 4` (from the manifest stage defaults).
- `models/<model_type>.json` is the only registry written by hand. `docs/MODELS.md` is generated: `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .`.
- Build presets: `linux-release-znver4-aocc` (perf + real-weight gates), `linux-debug` (fast sweep). Binaries land in `build/<preset>/apps/` and `build/<preset>/tests/`.
- Vault root for every run: `LSX_MODEL_ROOT=/mnt/fast_storage/huggingface`.
- Scratch space: `/mnt/fast_storage/tmp` (never `/tmp` for multi-GB artifacts).
- Commit messages: English, `feat:`/`fix:`/`test:`/`docs:`/`build:`/`refactor:` prefixes. Commit only on explicit request from the operator.

## Review Focus

Inputs and conditions the spec implies but no single task's tests cover by name — each line is pinned by the test named in brackets:

1. **A converted tree from the 30B sibling in the Ultra vault path** (a user points `--localmodel` at the wrong directory). Expected: load fails naming the mismatched geometry, never runs with wrong weights. [`test_nemotron_moe_ultra_loader`: "rejects a 30B-shaped tree"]
2. **`moe_latent_size` absent or 0 in a config that still has latent tensors** (or the reverse: latent declared, tensors missing). Expected: converter errors naming the tensor; loader errors naming the field. [`test_nemotron_moe_ultra_convert`: "latent tensors present/absent by config"; loader: "latent mandatory when moe_latent_size>0"]
3. **`n_routed_experts % n_group != 0`** (a config the reference itself would crash on). Expected: converter rejects at parse time with a diagnostic, not a divide-by-zero in the router. [`test_nemotron_moe_ultra_constants`: "group divisibility"; converter: "bad group geometry"]
4. **A route where two experts tie on `scores + bias` exactly.** Expected: deterministic, lowest-index-wins selection that matches the host reference. [`test_nemotron_moe_ultra_math`: "tie determinism"; kernels: same vector on device]
5. **A partially converted tree** (conversion killed at layer 40 of 108). Expected: `--resume` continues at layer 41; a loader run against the partial tree fails naming the first missing layer file, never decodes. [`test_nemotron_moe_ultra_convert`: "layer window resume"; loader: "partial tree fails"]

---

## File Structure

New library sources (`libs/lsxcommon/`):

| file | responsibility |
|---|---|
| `src/nemotron_moe_ultra_constants.h` | frozen passport, one `frozen` namespace, `static_assert`s |
| `include/lsxcommon/nemotron_moe_ultra_convert.h`, `src/nemotron_moe_ultra_convert.cpp` | config parse, tensor classify, per-layer NVFP4 conversion, index write |
| `include/lsxcommon/nemotron_moe_ultra_load.h`, `src/nemotron_moe_ultra_load.cpp` | index parse + geometry validation, device weight upload |
| `include/lsxcommon/nemotron_moe_ultra_expert_cache.h`, `src/nemotron_moe_ultra_expert_cache.cpp` | routed-expert host store (`ExpertStore<UltraExpertEntry>`) + device slot LRU |
| `src/nemotron_moe_ultra_device_math.h` | host reference math (`ref::` namespace) transcribed from the HF reference |
| `src/nemotron_moe_ultra_kernels.h`, `src/nemotron_moe_ultra_kernels.cu` | device kernel launchers |
| `include/lsxcommon/nemotron_moe_ultra_inference.h`, `src/nemotron_moe_ultra_inference.cpp` | session alloc/free/reset |
| `src/nemotron_moe_ultra_inference_pipeline.cu` | one token through 108 layers |
| `src/nemotron_moe_ultra_model.h`, `src/nemotron_moe_ultra_model.cpp` | `NemotronUltra550BModel` leaf: kType, Load, Infer, tool bridge, SelfCheck |
| `include/lsxcommon/nemotron_ultra_tool_format.h`, `src/nemotron_ultra_tool_format.cpp` | ChatML + `<function=…><parameter=…>` render/parse for Ultra's own template |

New tests (`tests/`):

| file | target | label |
|---|---|---|
| `test_nemotron_moe_ultra_fixture.h` | (header, synthetic Ultra checkpoint) | — |
| `test_nemotron_moe_ultra_constants.cpp` | `test_nemotron_moe_ultra_constants` | fast |
| `test_nemotron_moe_ultra_convert.cpp` | `test_nemotron_moe_ultra_convert` | convert |
| `test_nemotron_moe_ultra_loader.cpp` | `test_nemotron_moe_ultra_loader` | fast |
| `test_nemotron_moe_ultra_math.cpp` | `test_nemotron_moe_ultra_math` | fast |
| `test_nemotron_moe_ultra_kernels.cu` | `test_nemotron_moe_ultra_kernels` | inference |
| `test_nemotron_moe_ultra_model.cpp` | `test_nemotron_moe_ultra_model` | fast |
| `test_nemotron_ultra_tool_framing.cpp` | `test_nemotron_ultra_tool_framing` | fast |
| `test_nemotron_moe_ultra_expert_cache.cpp` | `test_nemotron_moe_ultra_expert_cache` | fast |
| `test_nemotron_ultra_tokenizer_parity.cpp` | `test_nemotron_ultra_tokenizer_parity` | load |
| `test_nemotron_ultra_layers.cpp` | `test_nemotron_ultra_layers` | inference |

New tools: `tools/nemotron_ultra_oracle.py` (slice oracle), `tools/nemotron_ultra_capture.py` (FSI/CODE/DET capture driver).

Modified: `models/nemotron-ultra-550b-a55b.json` (new), `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`, `tests/test_code_naming_guard.cpp`, `tests/test_model_suite_guard.cpp`, `tests/test_perf_baseline.cpp`, `tests/perf/perf_baselines.json`, `apps/lsxforge/src/{cli.h,cli.cpp,main.cpp}`, `apps/lsxforge/tests/test_lsxforge_cli.cpp`, `docs/SELFCHECK_MATRIX.md`, `docs/MODELS.md` (generated), `docs/OPEN_TASKS.md`, report file.

### Guard posture during this branch

`test_manifest_sync` runs `gen_model_manifest.py --check --strict`, and the orphan rule fails any `tests/test_*.cpp|cu` file that no manifest row claims. Every task before **Task 11** adds such files while the manifest row does not yet exist, so that one target is expected red until Task 11 lands the row. This is a recorded, temporary deviation: every other guard target (`test_code_naming_guard`, `test_model_independence_guard`, `test_model_suite_guard`, `test_ctest_labels_guard`, `test_infrastructure_guard`) must stay green at every commit, and Task 11's verification step runs the whole fast sweep and requires `test_manifest_sync` green before the real-weight work starts.

---

### Task 1: Frozen passport + naming vocabulary

**Files:**
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_constants.h`
- Create: `tests/test_nemotron_moe_ultra_constants.cpp`
- Modify: `tests/test_code_naming_guard.cpp` (variant vocabulary + `AllowedCommon`)
- Modify: `libs/lsxcommon/CMakeLists.txt` (add the header dir if not already covered)
- Modify: `tests/CMakeLists.txt` (host-only test executable)

**Interfaces:**
- Produces: `namespace lsxcommon::nemotron_ultra::frozen` with `kType`, `kHidden`, `kLayers`, `kVocab`, `kMambaLayers`, `kMoeLayers`, `kAttnLayers`, `kAttnLayerIdx`, `kRmsEps`, `kMambaHeads`, `kMambaHeadDim`, `kMambaDInner`, `kSsmStateSize`, `kSsmGroups`, `kMambaConvKernel`, `kMambaConvDim`, `kMambaInProj`, `kMambaGroupSize`, `kDtMin`, `kHeads`, `kKVHeads`, `kHeadDim`, `kNumRouted`, `kTopK`, `kMoeIntermediate`, `kMoeLatent`, `kSharedIntermediate`, `kRouterGroups`, `kRouterTopkGroup`, `kRoutedScale`, `kAttnScale`.

- [ ] **Step 1: Write the failing test** `tests/test_nemotron_moe_ultra_constants.cpp`

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include "nemotron_moe_ultra_constants.h"

using namespace lsxcommon::nemotron_ultra;

TEST_CASE("ultra passport: kType is the manifest model_type") {
    REQUIRE(std::strcmp(frozen::kType, "nemotron-ultra-550b-a55b") == 0);
}

TEST_CASE("ultra passport: layer mix decodes to 48/48/12 with the reference indices") {
    REQUIRE(frozen::kMambaLayers == 48);
    REQUIRE(frozen::kMoeLayers == 48);
    REQUIRE(frozen::kAttnLayers == 12);
    REQUIRE(frozen::kLayers == 108);
    const int expect[12] = {7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98};
    for (int i = 0; i < frozen::kAttnLayers; ++i)
        REQUIRE(frozen::kAttnLayerIdx[i] == expect[i]);
}

TEST_CASE("ultra passport: mamba geometry is self-consistent") {
    REQUIRE(frozen::kMambaDInner == frozen::kMambaHeads * frozen::kMambaHeadDim);   // 16384
    REQUIRE(frozen::kMambaDInner == 16384);
    REQUIRE(frozen::kMambaConvDim == frozen::kMambaDInner + 2 * frozen::kSsmGroups * frozen::kSsmStateSize); // 18432
    REQUIRE(frozen::kMambaInProj == frozen::kMambaDInner + frozen::kMambaConvDim + frozen::kMambaHeads);      // 35072
    REQUIRE(frozen::kMambaGroupSize == frozen::kMambaDInner / frozen::kSsmGroups);   // 2048
}

TEST_CASE("ultra passport: attention and MoE geometry") {
    REQUIRE(frozen::kHeads * frozen::kHeadDim == 8192);        // full-width q/o
    REQUIRE(frozen::kKVHeads * frozen::kHeadDim == 256);       // k/v rows
    REQUIRE(frozen::kHeads % frozen::kKVHeads == 0);
    REQUIRE(frozen::kTopK == 22);
    REQUIRE(frozen::kNumRouted == 512);
    REQUIRE(frozen::kNumRouted % frozen::kRouterGroups == 0);  // 512 % 8 == 0
    REQUIRE(frozen::kNumRouted / frozen::kRouterGroups == 64);
    REQUIRE(frozen::kRouterTopkGroup == 1);
    REQUIRE(frozen::kTopK <= (frozen::kNumRouted / frozen::kRouterGroups) * frozen::kRouterGroups);
    REQUIRE(frozen::kMoeLatent == 2048);
    REQUIRE(frozen::kSharedIntermediate == 10240);
    REQUIRE(frozen::kRoutedScale == 5.0f);
}

TEST_CASE("ultra passport: geometry is distinct from the 30B sibling") {
    // A copied 30B passport must not satisfy this model's own facts.
    REQUIRE(frozen::kHidden != 2688);
    REQUIRE(frozen::kLayers != 52);
    REQUIRE(frozen::kNumRouted != 128);
    REQUIRE(frozen::kTopK != 6);
    REQUIRE(frozen::kMoeLatent != 0);
}
```

- [ ] **Step 2: Run it and confirm it fails to compile**

Run: `cmake --preset linux-release-znver4-aocc && cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_constants`
Expected: FAIL — `nemotron_moe_ultra_constants.h: No such file or directory`.

- [ ] **Step 3: Write `libs/lsxcommon/src/nemotron_moe_ultra_constants.h`**

Header comment must record the source of truth (path of the real `config.json`) and the NoPE fact. Values exactly:

```cpp
#pragma once
// Frozen NVIDIA-Nemotron-3-Ultra-550B-A55B geometry (compile-time passport).
// Source: /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
// config.json (model_type nemotron_h) + safetensors inventory (2026-11-10):
// 108 layers (48 mamba + 48 moe + 12 full attention), hidden 8192, mamba2
// d_inner 16384 (256x64), n_groups 8, state 128, conv 4, in_proj 35072,
// MoE 512 experts top-22 at latent 2048 + shared 10240, vocab 131072, eps 1e-5.
// The full-attention layers are NoPE: the reference `NemotronHAttention` has no
// rotary module and ignores position_ids.
#include <cstdint>
namespace lsxcommon::nemotron_ultra {
namespace frozen {
inline constexpr const char *kType = "nemotron-ultra-550b-a55b";
inline constexpr int kHidden = 8192;
inline constexpr int kLayers = 108;
inline constexpr int kVocab = 131072;
inline constexpr int kMambaLayers = 48;
inline constexpr int kMoeLayers = 48;
inline constexpr int kAttnLayers = 12;
inline constexpr int kAttnLayerIdx[kAttnLayers] = {7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98};
inline constexpr double kRmsEps = 1e-5;
inline constexpr int kMambaHeads = 256;
inline constexpr int kMambaHeadDim = 64;
inline constexpr int kMambaDInner = kMambaHeads * kMambaHeadDim;                       // 16384
inline constexpr int kSsmStateSize = 128;
inline constexpr int kSsmGroups = 8;
inline constexpr int kMambaConvKernel = 4;
inline constexpr int kMambaConvDim = kMambaDInner + 2 * kSsmGroups * kSsmStateSize;     // 18432
inline constexpr int kMambaInProj = kMambaDInner + kMambaConvDim + kMambaHeads;         // 35072
inline constexpr int kMambaGroupSize = kMambaDInner / kSsmGroups;                       // 2048
inline constexpr float kDtMin = 0.001f;
inline constexpr int kHeads = 64;
inline constexpr int kKVHeads = 2;
inline constexpr int kHeadDim = 128;
inline constexpr float kAttnScale = 0.08838834764831845f;  // 1/sqrt(128)
inline constexpr int kNumRouted = 512;
inline constexpr int kTopK = 22;
inline constexpr int kMoeIntermediate = 5120;
inline constexpr int kMoeLatent = 2048;
inline constexpr int kSharedIntermediate = 10240;
inline constexpr int kRouterGroups = 8;
inline constexpr int kRouterTopkGroup = 1;
inline constexpr float kRoutedScale = 5.0f;
static_assert(kMambaLayers + kMoeLayers + kAttnLayers == kLayers, "layer mix must sum to kLayers");
static_assert(kMambaHeads * kMambaHeadDim == kMambaDInner, "mamba d_inner geometry");
static_assert(kMambaDInner + kMambaConvDim + kMambaHeads == kMambaInProj, "in_proj geometry");
static_assert(kMambaDInner % kSsmGroups == 0, "gated-norm group geometry");
static_assert(kNumRouted % kRouterGroups == 0, "expert groups must divide the expert count");
static_assert(kTopK <= kNumRouted, "top-k exceeds expert count");
static_assert(kHeadDim * kHeads == 8192 && kHeadDim * kKVHeads == 256, "attention projection geometry");
static_assert(kMoeLatent > 0 && kMoeLatent < kHidden, "latent MoE must compress");
} // namespace frozen
} // namespace lsxcommon::nemotron_ultra
```

- [ ] **Step 4: Naming vocabulary** in `tests/test_code_naming_guard.cpp`

Add `ultra` to the `var` alternation in `SchemeRegex()` and add `nemotron_ultra_tool_format` to `AllowedCommon()` with a one-line comment (family-level codec, no type marker). Do not add a new family stem — the family token stays `nemotron`.

- [ ] **Step 5: CMake wiring**

`tests/CMakeLists.txt`, host-only block (mirror the `test_nemotron_moe_math` block):

```cmake
# test_nemotron_moe_ultra_constants — the Ultra frozen passport (host-only).
add_executable(test_nemotron_moe_ultra_constants test_nemotron_moe_ultra_constants.cpp)
target_link_libraries(test_nemotron_moe_ultra_constants PRIVATE Catch2::Catch2WithMain)
target_include_directories(test_nemotron_moe_ultra_constants PRIVATE
    "${PROJECT_SOURCE_DIR}/libs/lsxcommon/src")
```
`add_test` comes from the manifest row (Task 11) — do not hand-register it.

- [ ] **Step 6: Run the test and confirm it passes**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_constants && ./build/linux-release-znver4-aocc/tests/test_nemotron_moe_ultra_constants`
Expected: PASS, 5 test cases, 0 failures.

- [ ] **Step 7: Run the naming guard**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_code_naming_guard && ctest --test-dir build/linux-release-znver4-aocc -R test_code_naming_guard --output-on-failure`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add libs/lsxcommon/src/nemotron_moe_ultra_constants.h tests/test_nemotron_moe_ultra_constants.cpp tests/test_code_naming_guard.cpp tests/CMakeLists.txt
git commit -m "feat(nemotron-ultra): frozen passport for the 550B-A55B geometry"
```

---

### Task 2: Synthetic Ultra checkpoint fixture

**Files:**
- Create: `tests/test_nemotron_moe_ultra_fixture.h`
- Modify: `tests/CMakeLists.txt` (include dir already covers `tests/`; no target change)

**Interfaces:**
- Consumes: `test_synth_checkpoint.h` primitives (`synthtest::FakeTensor`, `MakeBf16`, `MakeF32Values`, `WriteSafetensors`).
- Produces: `namespace ultratest` with `MakeTempDir(tag)`, `WriteUltraCheckpoint(const fs::path&)`, `WriteUltraIndexAndTensors(...)` (converted-tree writer used by loader/model tests), and the geometry constants below.

- [ ] **Step 1: Write the fixture header** (no test yet — this is the shared input for Tasks 3–7)

Tiny Ultra geometry, chosen so every Ultra-only path is exercised and **no value coincides with the 30B fixture** (`kTH=16, kTExperts=4, kTI=16, kTShared=32` there):

```cpp
namespace ultratest {
using synthtest::FakeTensor; using synthtest::MakeBf16;
using synthtest::MakeF32Values; using synthtest::WriteSafetensors;
inline constexpr int kTH = 16;         // hidden
inline constexpr int kTVocab = 32;
inline constexpr int kTAH = 4, kTAHD = 4;   // attn heads 4 x head_dim 4 == 16 == kTH
inline constexpr int kTKVH = 2;             // kv heads (GQA 2:1)
inline constexpr int kTMH = 2, kTMHD = 8;   // mamba heads 2 x head_dim 8 => d_inner 16
inline constexpr int kTSsm = 4, kTGroups = 2;
inline constexpr int kTConvDim = kTMH * kTMHD + 2 * kTGroups * kTSsm;  // 32
inline constexpr int kTMambaIn = kTMH * kTMHD + kTConvDim + kTMH;      // 50
inline constexpr int kTExperts = 8;
inline constexpr int kTGroup = 4;      // n_group (2 experts per group)
inline constexpr int kTTopkGroup = 1;
inline constexpr int kTTopK = 2;
inline constexpr int kTI = 16;         // moe intermediate (at latent)
inline constexpr int kTLatent = 8;     // moe_latent_size
inline constexpr int kTShared = 32;
inline constexpr int kTTokPerLayer = 24;  // layers: mamba, moe, attention
}
```
`WriteUltraCheckpoint` writes `config.json` with `model_type: nemotron_h`,
`layers_block_type: ["mamba","moe","attention"]`, `moe_latent_size: 8`,
`n_routed_experts: 8`, `num_experts_per_tok: 2`, `n_group: 4`, `topk_group: 1`,
`routed_scaling_factor: 5.0`, `norm_topk_prob: true`, `moe_intermediate_size: 16`,
`moe_shared_expert_intermediate_size: 32`, plus 1 `mtp.*` layer pair so the cut
path is exercised, and one shard of BF16 tensors named exactly as the real
checkpoint (`backbone.layers.N.mixer.*`, `backbone.embeddings.weight`,
`lm_head.weight`, `backbone.norm_f.weight`, `backbone.layers.N.norm.weight`).

- [ ] **Step 2: Compile-check the fixture through the constants test**

Add one case to `tests/test_nemotron_moe_ultra_constants.cpp`:

```cpp
TEST_CASE("ultra synthetic fixture: geometry is internally consistent") {
    REQUIRE(ultratest::kTMambaIn == ultratest::kTMH * ultratest::kTMHD + ultratest::kTConvDim + ultratest::kTMH);
    REQUIRE(ultratest::kTExperts % ultratest::kTGroup == 0);
    REQUIRE(ultratest::kTExperts != 4);   // must not collide with the 30B fixture
}
```
Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_constants && ./build/linux-release-znver4-aocc/tests/test_nemotron_moe_ultra_constants`
Expected: PASS (6 cases).

- [ ] **Step 3: Commit**

```bash
git add tests/test_nemotron_moe_ultra_fixture.h tests/test_nemotron_moe_ultra_constants.cpp
git commit -m "test(nemotron-ultra): synthetic Ultra-geometry checkpoint fixture"
```

---

### Task 3: Converter — config parse, classifier, per-layer streaming, index

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_convert.h`
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_convert.cpp`
- Create: `tests/test_nemotron_moe_ultra_convert.cpp`
- Create: `tools/nemotron_ultra_capture.py` (stub that documents the capture contract; filled in Task 12)
- Modify: `libs/lsxcommon/CMakeLists.txt` (add the `.cpp` to the host source list)
- Modify: `tests/CMakeLists.txt` (test executable + `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` define)

**Interfaces:**
- Consumes: `frozen::` passport (Task 1), `ultratest::` fixture (Task 2), `lsxcommon::safetensors` reader, `DenseNvfp4Record`/`PackedExpert`/`PinnedSchema`/`DenseNvfp4Schema`, `WriteExpertArrow`, `integrity::BuildAndVerifyTokenizer`.
- Produces:
```cpp
namespace lsxcommon::nemotron_ultra {
enum class UltraTensorKind { Unknown, Embed, LmHead, FinalNorm, LayerNorm,
    MambaInProj, MambaOutProj, MambaConv, MambaConvBias, MambaALog, MambaD,
    MambaDtBias, MambaNorm, AttnQ, AttnK, AttnV, AttnO, RouterGate, RouterBias,
    MoeLatentUp, MoeLatentDown, SharedUpProj, SharedDownProj,
    ExpertUpProj, ExpertDownProj, Mtp };
struct ParsedUltraTensor { UltraTensorKind kind = UltraTensorKind::Unknown; int layer = -1; int expert = -1; };
ParsedUltraTensor classify_ultra_tensor(const std::string &name);
bool ultra_tensor_is_nvfp4(UltraTensorKind kind);
struct UltraTopology { int num_layers=0; std::vector<std::string> layer_types; int hidden_size=0, vocab_size=0;
    int mamba_num_heads=0, mamba_head_dim=0, ssm_state_size=0, n_groups=0, conv_kernel=0;
    int num_attention_heads=0, num_key_value_heads=0, head_dim=0; double rms_norm_eps=1e-5;
    int num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
        shared_expert_intermediate_size=0, moe_latent_size=0, n_group=0, topk_group=0;
    float routed_scaling_factor=0.0f; bool norm_topk_prob=false;
    int eos_token_id=-1, bos_token_id=-1, pad_token_id=-1; int total_pinned_tensors=0;
    VramProfile vram; std::string error_msg; };
UltraTopology ParseUltraConfig(const std::string &input_dir);
struct ConversionWindow { int first_layer = 0; int last_layer = -1; };  // last_layer < 0 => to the end
arrow::Status ConvertNemotronUltraModel(const std::string &input_dir,
                                        const std::string &output_root,
                                        const ConversionWindow &window = {});
}
```

- [ ] **Step 1: Write the failing tests** `tests/test_nemotron_moe_ultra_convert.cpp`

Cases (assertions only; the implementer writes the Catch2 scaffolding):

```cpp
TEST_CASE("classifier: Ultra tensor names map to kinds") {
    // in_proj/out_proj/conv1d/A_log/D/dt_bias/norm -> mamba kinds, layer parsed
    REQUIRE(classify_ultra_tensor("backbone.layers.0.mixer.in_proj.weight").kind == UltraTensorKind::MambaInProj);
    REQUIRE(classify_ultra_tensor("backbone.layers.0.mixer.in_proj.weight").layer == 0);
    // latent projections are theirs, not shared-experts
    REQUIRE(classify_ultra_tensor("backbone.layers.1.mixer.fc1_latent_proj.weight").kind == UltraTensorKind::MoeLatentUp);
    REQUIRE(classify_ultra_tensor("backbone.layers.1.mixer.fc2_latent_proj.weight").kind == UltraTensorKind::MoeLatentDown);
    REQUIRE(classify_ultra_tensor("backbone.layers.1.mixer.shared_experts.up_proj.weight").kind == UltraTensorKind::SharedUpProj);
    REQUIRE(classify_ultra_tensor("backbone.layers.1.mixer.experts.7.down_proj.weight").kind == UltraTensorKind::ExpertDownProj);
    REQUIRE(classify_ultra_tensor("backbone.layers.1.mixer.experts.7.down_proj.weight").expert == 7);
    REQUIRE(classify_ultra_tensor("mtp.layers.0.mixer.q_proj.weight").kind == UltraTensorKind::Mtp);
    REQUIRE(classify_ultra_tensor("totally.unknown.tensor").kind == UltraTensorKind::Unknown);
}

TEST_CASE("config parse: Ultra fields are read, defaults rejected") {
    auto dir = ultratest::WriteUltraCheckpoint(...);
    auto top = ParseUltraConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    REQUIRE(top.n_groups == 4);
    REQUIRE(top.topk_group == 1);
    REQUIRE(top.moe_latent_size == 8);
    REQUIRE(top.routed_scaling_factor == Catch::Approx(5.0));
    REQUIRE(top.layer_types == std::vector<std::string>{"mamba","moe","attention"});
}

TEST_CASE("config parse: a group count that does not divide the experts is rejected") {
    // rewrite n_routed_experts to 7 (n_group 4) and expect error_msg mentioning grouping
    auto top = ParseUltraConfig(bad_dir.string());
    REQUIRE(top.error_msg.find("group") != std::string::npos);
}

TEST_CASE("convert: synthetic Ultra checkpoint produces the Ultra index") {
    // ConvertNemotronUltraModel(src, out) is OK; read out/logestix/model.index.json
    REQUIRE(idx["format"] == "logestix-nemotron-ultra-nvfp4");
    REQUIRE(idx["model"] == "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16");
    REQUIRE(idx["moe_latent_size"] == 8);
    REQUIRE(idx["n_group"] == 4);
    REQUIRE(idx["routed_scaling_factor"] == Approx(5.0));
    REQUIRE(idx["mtp_tensors"] == 3);        // counted cut, not ignored
    REQUIRE(os::exists(out / "logestix" / "pinned" / "global.arrow"));
    REQUIRE(os::exists(out / "logestix" / "experts" / "layer_1" / "exp_7.arrow"));
    REQUIRE_FALSE(os::exists(out / "logestix" / "experts" / "layer_0"));  // mamba layer carries none
}

TEST_CASE("convert: latent projections are required when the config declares them") {
    // delete fc1_latent_proj from the source, expect an error naming the tensor
    REQUIRE(status.message().find("fc1_latent_proj") != std::string::npos);
}

TEST_CASE("convert: unknown tensors are rejected, not ignored") {
    // add backbone.layers.0.mixer.mystery.weight -> error naming it
}

TEST_CASE("convert: a layer window converts only that layer and marks the tree partial") {
    ConversionWindow w{1, 1};
    REQUIRE(ConvertNemotronUltraModel(src, out, w).ok());
    REQUIRE(os::exists(out / "logestix" / "experts" / "layer_1" / "exp_0.arrow"));
    REQUIRE_FALSE(os::exists(out / "logestix" / "pinned" / "layer_0.arrow"));
    // index carries "partial": true and "layers_converted": [1]
}

TEST_CASE("convert: rerunning the same window resumes instead of rewriting layer 0") {
    // second run with window {2,2}: layer_1 files keep their mtime, layer_2 appears
}
```

- [ ] **Step 2: Run and confirm failure**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_convert`
Expected: FAIL — `lsxcommon/nemotron_moe_ultra_convert.h: No such file or directory`.

- [ ] **Step 3: Implement the converter**

`ParseUltraConfig`: simdjson on-demand over `config.json`; read every field listed in the interface plus `layers_block_type` (normalize `mamba→mamba`, `attention→attention`, `moe→moe`); validate
`hidden_size>0 && num_layers>0 && num_experts>0 && num_experts_per_tok>0 && moe_intermediate_size>0 && layer_types.size()==num_layers`;
when `moe_latent_size > 0` require `num_experts % n_group == 0` and `topk_group <= n_group`, with an `error_msg` that contains the word `group`.

`classify_ultra_tensor`: parse `backbone.layers.<N>.mixer.<rest>`; the suffix map is the 30B map plus
`fc1_latent_proj.weight → MoeLatentUp`, `fc2_latent_proj.weight → MoeLatentDown`; any name starting `mtp.` is `Mtp`; anything else is `Unknown`.

`ConvertNemotronUltraModel`: one pass to classify and count; reject any `Unknown` with `arrow::Status::Invalid("unclassified tensor: ", name)`; assert `Mtp` is the only cut and record its count. Then **per MoE layer in the window**: collect that layer's `experts.*` records from the shard index, quantize on GPU (`QuantizeSliceBf16`), write `experts/layer_<l>/exp_<e>.arrow` via `WriteExpertArrow` (w3 = up, w2 = down, w1 empty [0,0]), free the records, next layer. Dense per-layer records go to `pinned/layer_<l>.arrow` and `pinned/layer_<l>_nvfp4.arrow`; globals to `pinned/global.arrow`. Index fields: `format`, `model`, `num_layers`, `layer_types`, `attn_layer_idx`, every geometry field, `moe_latent_size`, `n_group`, `topk_group`, `routed_scaling_factor`, `norm_topk_prob`, `mtp_tensors`, `partial`/`layers_converted` for a window, `vram`. Tokenizer: `integrity::BuildAndVerifyTokenizer(tokenizer.json, pinned_dir, {bos_required=false, eos_required=true}, pretokenizer=7)`, and require the file unless `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`.

- [ ] **Step 4: Run and confirm all cases pass**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_convert && LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4-aocc/tests/test_nemotron_moe_ultra_convert`
Expected: PASS, 8 cases. Wall time under 120 s (the target's manifest timeout).

- [ ] **Step 5: Prove the streaming keeps peak host use at one layer**

Add to the convert test a case that converts a fixture with 3 MoE layers × 8 experts and asserts the converter never holds more than one layer's records: expose `struct UltraConvertStats { int64_t peak_expert_records = 0; }` through the index (`"peak_expert_records"`) and assert `peak_expert_records == 8`.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_convert.h libs/lsxcommon/src/nemotron_moe_ultra_convert.cpp libs/lsxcommon/CMakeLists.txt tests/test_nemotron_moe_ultra_convert.cpp tests/CMakeLists.txt
git commit -m "feat(nemotron-ultra): converter with latent projections, grouped-router fields and per-layer expert streaming"
```

---

### Task 4: Index loader + geometry validation + device weights

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_load.h`
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_load.cpp`
- Create: `tests/test_nemotron_moe_ultra_loader.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Consumes: `UltraTopology` output schema (Task 3), `lsx_nvfp4::LsxNvfp4Weight`, `LoadDenseNvfp4BatchViews`, `PinnedSchema` views.
- Produces:
```cpp
namespace lsxcommon::nemotron_ultra {
struct UltraIndex { std::string format, model; int num_layers=0; std::vector<std::string> layer_types;
    std::vector<int> attn_layer_idx; int hidden_size=0, vocab_size=0, mamba_num_heads=0, mamba_head_dim=0,
    ssm_state_size=0, n_groups=0, conv_kernel=0, num_attention_heads=0, num_key_value_heads=0, head_dim=0,
    num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0, shared_expert_intermediate_size=0,
    moe_latent_size=0, n_group=0, topk_group=0; float routed_scaling_factor=0.0f; bool norm_topk_prob=false;
    int bos_token_id=-1, eos_token_id=-1, pad_token_id=-1; std::vector<int> eos_token_ids; int mtp_tensors=0;
    bool partial=false; std::vector<int> layers_converted; int total_pinned_tensors=0; VramProfile vram; };
bool LoadUltraIndex(const std::string &model_path, UltraIndex &out, std::string &error);
bool ValidateUltraGeometry(const UltraIndex &idx, std::string &error);
struct UltraLayerWeights { __half *d_norm=nullptr;
    lsx_nvfp4::LsxNvfp4Weight in_proj, out_proj;
    float *d_conv_w=nullptr, *d_conv_b=nullptr, *d_A_log=nullptr, *d_D=nullptr, *d_dt_bias=nullptr, *d_mamba_norm=nullptr;
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o;
    __half *d_router=nullptr; float *d_router_bias=nullptr;
    lsx_nvfp4::LsxNvfp4Weight latent_up, latent_down, shared_up, shared_down;  // latent_* null for a non-latent tree
    void Free(); };
struct UltraWeights { __half *d_embed=nullptr, *d_head=nullptr, *d_final_norm=nullptr;
    std::vector<UltraLayerWeights> layers; std::uint64_t resident_bytes=0; void Free(); };
bool LoadUltraWeights(const std::string &model_path, const UltraIndex &idx, UltraWeights &w,
                      cudaStream_t stream, std::string &error);
}
```

- [ ] **Step 1: Write the failing tests** `tests/test_nemotron_moe_ultra_loader.cpp`

```cpp
TEST_CASE("index: a converted Ultra tree loads and validates") {
    // build the converted tree with the Task-3 converter on the fixture, then:
    UltraIndex idx; std::string err;
    REQUIRE(LoadUltraIndex(tree, idx, err));
    REQUIRE(ValidateUltraGeometry(idx, err));
    REQUIRE(idx.moe_latent_size == 8);
    REQUIRE(idx.n_group == 4);
}

TEST_CASE("index: a foreign format tag is rejected") {
    // rewrite "format" to "logestix-nemotron-nvfp4"; expect LoadUltraIndex false + err naming the tag
}

TEST_CASE("geometry: a 30B-shaped tree is rejected by name") {
    // run the sibling's converter-shaped fixture geometry through ValidateUltraGeometry via an
    // index with num_layers 52, hidden 2688, 128 experts, moe_latent_size 0
    REQUIRE_FALSE(ValidateUltraGeometry(idx, err));
    REQUIRE(err.find("num_layers") != std::string::npos);
}

TEST_CASE("geometry: latent declared but tensors absent is rejected") {
    // index says moe_latent_size 8; delete pinned layer latent entries; expect err naming fc1_latent_proj
}

TEST_CASE("geometry: a partial tree fails naming the first missing layer") {
    // index "partial": true with layers_converted [1]; expect err naming layer 0
}

TEST_CASE("weights: the synthetic tree uploads and frees") {
    // requires a CUDA device; REQUIRE(cudaFree(0) == cudaSuccess) first
    REQUIRE(LoadUltraWeights(tree, idx, w, stream, err));
    REQUIRE(w.resident_bytes > 0);
    REQUIRE(w.layers[1].latent_up.rows == 8);      // latent up 8x16
    REQUIRE(w.layers[1].latent_down.rows == 16);
    w.Free();
    REQUIRE(w.d_embed == nullptr);
}
```

- [ ] **Step 2: Run and confirm failure**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_loader`
Expected: FAIL — missing header.

- [ ] **Step 3: Implement**

`LoadUltraIndex`: simdjson over `<model_path>/logestix/model.index.json`; reject any `format != "logestix-nemotron-ultra-nvfp4"` with the actual tag in the message; read `eos_token_id` as scalar or array (the checkpoint declares `[2, 11]`), keeping `eos_token_ids.front()` as the scalar.
`ValidateUltraGeometry`: compare every field against the passport in a fixed order (`num_layers`, `hidden_size`, `vocab_size`, mamba block, attention block, `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `moe_latent_size`, `shared_expert_intermediate_size`, `n_group`, `topk_group`, `layer_types`, `attn_layer_idx`), each failure naming the field and both values; honour `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` (skip the comparison, keep format/partial checks); reject `partial == true` unless that env is set, naming the first layer not in `layers_converted`.
`LoadUltraWeights`: mmap views per layer file, upload through `LsxNvfp4Weight` for the quantized entries and `cudaMemcpyAsync` for the raw ones; require `latent_up`/`latent_down` iff `moe_latent_size > 0`; accumulate `resident_bytes`; free everything on any failure (the model's `TeardownWeights` calls `UltraWeights::Free`).

- [ ] **Step 4: Run and confirm pass**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_loader && LGX_ALLOW=1 ./build/linux-release-znver4-aocc/tests/test_nemotron_moe_ultra_loader`
Expected: PASS, 6 cases (CUDA cases SKIP nothing — they run, this target is GPU-labelled `fast` and takes the card briefly).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_load.h libs/lsxcommon/src/nemotron_moe_ultra_load.cpp libs/lsxcommon/CMakeLists.txt tests/test_nemotron_moe_ultra_loader.cpp tests/CMakeLists.txt
git commit -m "feat(nemotron-ultra): index loader, frozen-geometry gate and device weight graph"
```

---

### Task 5: Host reference math (the transcription the numerics are pinned to)

**Files:**
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_device_math.h`
- Create: `tests/test_nemotron_moe_ultra_math.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces (all in `namespace lsxcommon::nemotron_ultra::ref`, host-only, no CUDA headers):
```cpp
inline float Silu(float x);
inline float Softplus(float x);
inline float Relu2(float x);
inline void GatedRmsNorm(const float *x, const float *gate, const float *weight, int n, int group_size, float eps, float *out);
inline void SelectiveStateUpdate(const float *state_in, const float *xs, const float *B, const float *C, const float *D, const float *dt, const float *dt_bias, const float *A_log, int heads, int head_dim, int groups, int state_size, int clamp_dt, float *state_out, float *out);
inline void RouterTopKGrouped(const float *logits, const float *e_score_bias, int E, int k, int n_group, int topk_group, float scale, float *weights, int *indices);
inline void LatentMoe(const float *x, const float *fc1, const float *fc2, const float *shared_up, const float *shared_down, const int *expert_ids, const float *expert_w, int H, int L, int I, int SI, int topk, const float *const *up, const float *const *down, float *out);
inline void ExpertMlpRelu2(const float *x, const float *up, const float *down, int in, int i, float *out);
```

- [ ] **Step 1: Write the failing tests** `tests/test_nemotron_moe_ultra_math.cpp`

```cpp
TEST_CASE("router: group stage keeps the best group and zeroes the rest") {
    // 8 experts, n_group 4 (2 per group), topk_group 1, k 2.
    // Craft logits so group 2 (experts 4,5) is best by (top-2 sum), while the
    // single best expert overall is 0. Selection MUST come from group 2 only.
    float logits[8] = {4.0f, 0.0f,   0.0f, 0.0f,   3.0f, 2.9f,   0.0f, 0.0f};
    float bias[8] = {0};
    float w[2]; int idx[2];
    ref::RouterTopKGrouped(logits, bias, 8, 2, 4, 1, 5.0f, w, idx);
    REQUIRE(((idx[0] == 4 && idx[1] == 5) || (idx[0] == 5 && idx[1] == 4)));
    REQUIRE(w[0] + w[1] == Catch::Approx(5.0));   // normalized then scaled
}

TEST_CASE("router: bias selects but never scales") {
    // bias makes expert 6 win; weight must still be sigmoid(logit) normalized
}

TEST_CASE("router: exact ties break to the lowest index, deterministically") {
    // all-zero logits, k 22 over 512 (or 2 over 8 in the tiny case) -> indices ascending? 
    // assert twice-run equality AND that the selected set equals {0,1} for the tiny case
}

TEST_CASE("latent moe: routed path runs at the latent width and is projected back") {
    // H 16, L 8, I 16, SI 32, topk 2; compare against a hand-computed expectation
    // out = fc2(sum_e w_e * down_e(relu2(up_e(fc1(x))))) + shared_down(relu2(shared_up(x)))
}

TEST_CASE("mamba: gated rmsnorm at group 2048 uses the group mean square") {
    // n = 4096, group_size = 2048, two groups with different scales
}

TEST_CASE("mamba: selective state update clamps dt to kDtMin when asked") {
    // dt_bias -50 => softplus ~ 0 => clamped to 1e-3 in both clamp modes' floor
}

TEST_CASE("softplus: matches log1p(exp(x)) across the stable range") {
    // x in {-30,-5,0,5,30}; no inf/nan at the extremes
}
```

- [ ] **Step 2: Run and confirm failure**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_math`
Expected: FAIL — missing header.

- [ ] **Step 3: Implement the reference math**

Transcribe from `transformers/models/nemotron_h/modeling_nemotron_h.py` (5.15.1) — the exact functions: `NemotronHTopkRouter.forward` (L728–753) for the router, `NemotronHMoE.forward` (L699–712) for the latent path, `NemotronHMamba2Mixer.forward` (L427–571) + `Zamba2RMSNormGated` for the recurrence and gated norm, `NemotronHRMSNorm.forward` (L584–589) for the norm. Sizes for the fixed arrays come from the passport (`frozen::kNumRouted`), never from literals.

- [ ] **Step 4: Run and confirm pass**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_math && ./build/linux-release-znver4-aocc/tests/test_nemotron_moe_ultra_math`
Expected: PASS, 7 cases.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/nemotron_moe_ultra_device_math.h tests/test_nemotron_moe_ultra_math.cpp tests/CMakeLists.txt
git commit -m "feat(nemotron-ultra): host reference math transcribed from the NemotronH reference"
```

---

### Task 6: Device kernels (grouped router, mamba, gated norm, NoPE attention, relu²)

**Files:**
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_kernels.h`
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_kernels.cu`
- Create: `tests/test_nemotron_moe_ultra_kernels.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (new CU OBJECT library `lsxcommon_nemotron_ultra_cu`)
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `ref::` (Task 5), `frozen::` (Task 1).
- Produces:
```cpp
namespace lsxcommon::nemotron_ultra {
void RunUltraConvStep(const float *xBC, const float *w, const float *bias, const float *state_in, float *state_out, float *out, int conv_dim, int K, cudaStream_t);
void RunUltraSelectiveStateStep(const float *state_in, const float *xs, const float *B, const float *C, const float *D, const float *dt, const float *dt_bias, const float *A_log, int heads, int head_dim, int groups, int state_size, int clamp_dt, float *state_out, float *out, cudaStream_t);
void RunUltraGatedRmsNorm(const float *x, const float *gate, const float *weight, int n, int group_size, float eps, float *out, cudaStream_t);
void RunUltraAttnDecode(const float *q, const float *k_cache, const float *v_cache, int heads, int kv_heads, int head_dim, int seq_len, int cache_stride, float scale, float *out, cudaStream_t);
void RunUltraRouter(const float *logits, const float *e_score_bias, int E, int k, int n_group, int topk_group, float scale, float *weights, int *indices, cudaStream_t);
void RunUltraRelu2(const float *x, int n, float *out, cudaStream_t);
void RunUltraRelu2Half(__half *x, int n, cudaStream_t);
void RunUltraMoeWeightedSum(const __half *y, const float *weights, int count, int n, float *acc, bool add, cudaStream_t);
void RunUltraAddHalfToF32(const __half *x, int n, float *acc, cudaStream_t);
}
```

- [ ] **Step 1: Write the failing tests** `tests/test_nemotron_moe_ultra_kernels.cu`

Each device case runs the launcher and the matching `ref::` function on the same random input and asserts `max_abs_diff <= 2e-5` (F32) / exact for the selection integers; the router additionally asserts the **index set** matches `ref::RouterTopKGrouped` on 200 random logits and on the crafted tie vector.

```cpp
TEST_CASE("kernels: grouped router selection matches the reference on random logits") { /* 200 draws */ }
TEST_CASE("kernels: grouped router reproduces the reference on the crafted group/tie vectors") { }
TEST_CASE("kernels: depthwise conv step matches the reference") { }
TEST_CASE("kernels: selective state update matches the reference") { }
TEST_CASE("kernels: gated rmsnorm matches the reference at group_size 2048") { }
TEST_CASE("kernels: NoPE attention decode matches the reference") { }
TEST_CASE("kernels: relu2 half is bit-identical to float relu2 then round-to-half") { }
TEST_CASE("kernels: weighted sum matches the reference") { }
```

- [ ] **Step 2: Run and confirm failure**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_kernels`
Expected: FAIL — missing header.

- [ ] **Step 3: Implement the kernels**

- Router: one block (≥ `E` threads, power-of-two reduce), stage 1 computes `sigmoid(logit)+bias`, per-group top-2 sum via shared-memory reduction, selects the best `topk_group` groups (mask the rest), then `k` iterations of parallel max over the surviving experts with already-taken indices excluded; lowest index wins ties (`>` comparison only, never `>=`); then normalize `w /= sum + 1e-20f` and multiply by `scale` on thread 0.
- Conv/SSU/gated norm/attention: same shapes as the sibling but every width comes from the argument list, not from a passport constant in device code; the attention launcher has **no rotary** and its comment states NoPE.
- Keep all reductions inside one block behind `__syncthreads()`; no atomics; the target is `CUDA_ARCHITECTURES OFF` with `LSX_CUDA_ARCH` gencode flags and `-fmad=false` for the bit-exact share.

- [ ] **Step 4: Run and confirm pass**

Run: `cmake --build --preset linux-release-znver4-aocc --target test_nemotron_moe_ultra_kernels && ./build/linux-release-znver4-aocc/tests/test_nemotron_moe_ultra_kernels`
Expected: PASS, 8 cases.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/nemotron_moe_ultra_kernels.h libs/lsxcommon/src/nemotron_moe_ultra_kernels.cu libs/lsxcommon/CMakeLists.txt tests/test_nemotron_moe_ultra_kernels.cu tests/CMakeLists.txt
git commit -m "feat(nemotron-ultra): device kernels incl. grouped top-k router and latent-width expert path"
```

---

### Task 7: Routed-expert store + device slot LRU

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_expert_cache.h`
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_expert_cache.cpp`
- Create: `tests/test_nemotron_moe_ultra_expert_cache.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: neutral `gpu::expert_store::ExpertStore<TEntry>`, `gpu::ExpertDeviceCache`, `lifecycle::ExpertCacheSlots`, `perf::CurrentKnobs()`.
- Produces:
```cpp
namespace lsxcommon::nemotron_ultra {
struct UltraExpertEntry { /* 15 view fields in ExpertStore::DecodeEntry() order */ };
using UltraExpertStore = gpu::expert_store::ExpertStore<UltraExpertEntry>;
inline constexpr int64_t kUltraExpertSlotAlign = 256;
struct UltraExpertSlotLayout { int64_t up_pack_off, down_pack_off, pack_stride, up_scale_off, down_scale_off, scale_stride; bool ok; };
UltraExpertSlotLayout PlanUltraExpertSlot(int64_t up_pack, int64_t up_scale, int64_t down_pack, int64_t down_scale);
int64_t UltraExpertPerExpertBytes(const UltraExpertSlotLayout &l);
struct UltraExpertView { const uint8_t *up_packed, *up_scales, *down_packed, *down_scales; float up_global_scale, down_global_scale; };
struct UltraExpertCache { UltraExpertStore store; gpu::ExpertDeviceCache device; UltraExpertSlotLayout layout;
    int64_t per_expert_bytes = 0; int total_experts = 0, experts_per_layer = 0, moe_layers = 0;
    std::vector<int> moe_index_of_layer; std::uint64_t cache_bytes = 0; };
int UltraExpertCacheSlots(std::size_t free_bytes, std::size_t per_expert_bytes, int total_experts, int topk, const char *env_value);
void LoadUltraExpertStore(const std::string &model_path, int num_layers, int experts_per_layer, UltraExpertStore &store);
bool AllocateUltraExpertCache(UltraExpertCache &cache, int slots, std::string &error);
void FreeUltraExpertCache(UltraExpertCache &cache);
cudaError_t StageUltraExpert(UltraExpertCache &cache, int layer, int eid, cudaStream_t, std::string &error, int *slot_out = nullptr, bool *missed_out = nullptr);
cudaError_t ResolveUltraExperts(UltraExpertCache &cache, int layer, const int *top_ids, int k, cudaStream_t, UltraExpertView *out, std::string &error);
}
```

- [ ] **Step 1: Write the failing tests**

```cpp
TEST_CASE("slot layout: sections are 256-byte aligned and pack_stride covers both projections") {
    // 11_796_480-byte up pack (5120x2048 NVFP4) and its scales -> exact offsets
}
TEST_CASE("slots: floor is topk=22, ceiling is total_experts, env override wins") { }
TEST_CASE("store: a synthetic tree exposes 24 entries (3 moe layers x 8) as mmap views") { }
TEST_CASE("lru: staging beyond the slot count evicts the least recently used") { }
TEST_CASE("lru: resolving one layer's top-k returns k distinct slots and never duplicates") { }
```

- [ ] **Step 2: Run and confirm failure**, then **Step 3: implement** by delegating to the neutral store/cache exactly as the sibling does (no owning containers, `max_open`-bounded), with the Ultra env override `LSX_NEMOTRON_ULTRA_EXPERT_CACHE_SLOTS`.
- [ ] **Step 4: Run and confirm pass** — `./build/linux-release-znver4-aocc/tests/test_nemotron_moe_ultra_expert_cache`, 5 cases.
- [ ] **Step 5: Commit** — `feat(nemotron-ultra): routed-expert host store and device slot LRU`.

---

### Task 8: Session + per-token pipeline over 108 layers

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_inference.h`
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_inference.cpp`
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_inference_pipeline.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (pipeline `.cu` into `lsxcommon_nemotron_ultra_cu`)

**Interfaces:**
- Produces:
```cpp
namespace lsxcommon::nemotron_ultra {
struct UltraSession { int H, num_layers, num_experts, topk, d_inner, conv_dim, heads, head_dim, ssm_groups,
    ssm_state_size, kv_heads, attn_head_dim, max_seq, moe_latent, moe_intermediate, shared_intermediate;
    std::vector<float*> mamba_state, conv_state; std::vector<int> mamba_slot, attn_slot;
    std::vector<float*> k_cache, v_cache; int attn_layers, mamba_layers;
    /* scratch: h, normed, sub, gate, xbc, dt, ssm_out, gn_out, q, k, v, attn, latent,
       router_logits, topk_w, topk_i, expert_acc, head_logits, proj16, out16 */
    void Free(); };
bool AllocUltraSession(const UltraIndex &idx, int max_seq, UltraSession &s, std::string &error);
void FreeUltraSession(UltraSession &s);
void ResetUltraSession(UltraSession &s, cudaStream_t);
cudaError_t UltraForwardToken(const UltraIndex &idx, const UltraWeights &w, UltraSession &s,
                              UltraExpertCache &experts, int token, int pos, bool prefill, cudaStream_t);
}
```

- [ ] **Step 1: Write the failing test** — `tests/test_nemotron_moe_ultra_model.cpp` (shared with Task 9) gets a session case: alloc at `max_seq=64` on the synthetic index, assert `attn_layers==1`, `mamba_layers==1`, `moe_latent==8`, `mamba_state.size()==1`, `k_cache.size()==1`; `ResetUltraSession` zeroes; `FreeUltraSession` nulls every pointer.
- [ ] **Step 2: Run it and confirm failure**, then **Step 3: implement** the session allocator and the pipeline: for each layer `x = residual + Mixer(RMSNorm(x))` where the mixer is conv→SSU→gated-norm→out_proj (mamba), q/k/v→**NoPE**→cache→decode→o_proj (attention), or router→latent fc1→batched expert GEMVs at latent width→weighted sum→fc2→+shared (moe); final `norm_f` then the head GEMV. The MoE branch reads `idx.moe_latent_size`, `idx.n_group`, `idx.topk_group`, `idx.routed_scaling_factor` — never a passport literal.
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit** — `feat(nemotron-ultra): 108-layer decode pipeline with latent MoE and grouped routing`.

---

### Task 9: Model leaf, tool codec, lifecycle contract

**Files:**
- Create: `libs/lsxcommon/src/nemotron_moe_ultra_model.h`, `src/nemotron_moe_ultra_model.cpp`
- Create: `libs/lsxcommon/include/lsxcommon/nemotron_ultra_tool_format.h`, `src/nemotron_ultra_tool_format.cpp`
- Create: `tests/test_nemotron_ultra_tool_framing.cpp`
- Modify: `tests/test_nemotron_moe_ultra_model.cpp`, `libs/lsxcommon/CMakeLists.txt`, `tests/CMakeLists.txt`
- Modify: `tests/test_model_suite_guard.cpp` (tool-verdict row)

**Interfaces:**
- Consumes: `ModelBase`, `model_runtime`, `prompt_framing`, `LsxFastMathScope`.
- Produces: `class NemotronUltra550BModel final : public ModelBase` with `static const std::string kType` (`frozen::kType`), `Load`, `Unload`, `Infer`, `SelfCheck`, `Type`, `BuildPromptInputIds`, `BuildConversationInputIds`, `ParseToolCalls`, and `nemotron_ultra::RenderUltraConversation(const ChatConversation&, std::string&)`.

- [ ] **Step 1: Write the failing framing test** `tests/test_nemotron_ultra_tool_framing.cpp`

Cases: a user-only conversation renders `"<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n"`; a conversation with tools renders the `# Tools` preamble and each tool as JSON inside `<tools></tools>`; a system turn is emitted first when present; a tool result renders as a user turn carrying `<tool_response>`; `ParseToolCalls` recovers `get_weather {"city":"Paris"}` from the model's `<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>` form; and the `<think>`/`<think></think>` reasoning markers are emitted **off** for tool turns and on for a bare assistant turn, matching `chat_template.jinja` (md5 `fa104c9af7c4febc2f5b8febea7b3904` — note this is NOT the 30B sibling's template `a2445a9f…`).

- [ ] **Step 2: Run it and confirm failure.**
- [ ] **Step 3: Implement the codec and the model leaf.**

The model leaf's generation body must satisfy the seven lifecycle markers the independence guard greps for, all in `nemotron_moe_ultra_model.cpp`: `RunUnderResidencyLock`, a streamed `gs->Emit`, a `gs->Aborted()` poll inside the token loop, `TeardownContext`, `AcquirePermanentWeights`/`AcquireExpertCacheLease`, `DecodeGeneratedText`, and `weight_resident_bytes`/`FillInferenceResult`. `Infer` opens `lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_))` before the pipeline and publishes through `FillInferenceResult`. `Load` mirrors the sibling's order: `BeginLoad` → index + geometry → session alloc (`kMaxSeq = 4096`) → `LoadUltraWeights` → `LoadUltraExpertStore` → `AllocateUltraExpertCache` → tokenizer. `SelfCheck` runs the Paris probe with ≥3 cycles and returns the decoded token and `logits_hash`.

- [ ] **Step 4: Extend the model test** with factory/load/infer cases on the synthetic tree: `LsxModelFactory` resolves `nemotron-ultra-550b-a55b`; `Load` on the synthetic tree succeeds under `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`; a 2-token generation returns non-empty `output_text`; `Unload` releases (`weight_resident_bytes == 0`).
- [ ] **Step 5: Run both targets and confirm pass.**
- [ ] **Step 6: Commit** — `feat(nemotron-ultra): model leaf, ChatML tool codec and the wired lifecycle contract`.

---

### Task 10: `lsxforge --nemotron_ultra`

**Files:**
- Modify: `apps/lsxforge/src/cli.h` (add `bool nemotron_ultra = false;`), `apps/lsxforge/src/cli.cpp` (`{"--nemotron_ultra", {Pipeline::kNemotronUltra, flags.nemotron_ultra}}`), `apps/lsxforge/src/main.cpp` (`ABSL_FLAG`, dispatch to `ConvertNemotronUltraModel`, `--layers a:b` mapping to `ConversionWindow`)
- Modify: `apps/lsxforge/tests/test_lsxforge_cli.cpp` (pipeline map row)

**Interfaces:** consumes `ConvertNemotronUltraModel` (Task 3); produces the CLI surface `--nemotron_ultra`, `--layers`.

- [ ] **Step 1: Write/extend the failing CLI test** — the pipeline-name table case must include `{"nemotron_ultra", Pipeline::kNemotronUltra}`; a new case asserts `--layers 40:41` parses to `{40, 41}` and that `--layers` without `--nemotron_ultra` is rejected.
- [ ] **Step 2: Run it and confirm failure**, **Step 3: implement**, **Step 4: run and confirm pass** (`./build/linux-release-znver4-aocc/apps/lsxforge/tests/test_lsxforge_cli` or the `lsxforge` test target name in that directory).
- [ ] **Step 5: Commit** — `feat(lsxforge): --nemotron_ultra pipeline and --layers conversion window`.

---

### Task 11: Manifest row + all remaining targets + tokenizer parity (guards green from here)

**Files:**
- Create: `models/nemotron-ultra-550b-a55b.json`
- Create: `tests/test_nemotron_ultra_tokenizer_parity.cpp`
- Create: `tests/test_nemotron_ultra_layers.cpp` (skeleton with SKIP 4; filled in Task 13)
- Modify: `tests/CMakeLists.txt`, `docs/SELFCHECK_MATRIX.md`, `docs/MODELS.md` (generated)

**Interfaces:** produces the registry that downstream gates read.

- [ ] **Step 1: Write the manifest row.**

```json
{
  "model_type": "nemotron-ultra-550b-a55b",
  "display_name": "nemotron-ultra-550b-a55b",
  "family": "nemotron",
  "variant": null,
  "model_class": "NemotronUltra550BModel",
  "model_header": "nemotron_moe_ultra_model.h",
  "passport": "libs/lsxcommon/src/nemotron_moe_ultra_constants.h",
  "namespace": "lsxcommon::nemotron_ultra::frozen",
  "format_tag": "logestix-nemotron-ultra-nvfp4",
  "forge_flag": "--nemotron_ultra",
  "tokenizer_kind": 9,
  "vault": { "dir": "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
             "env": "LSX_TEST_NEMOTRON_ULTRA_550B_A55B_ROOT" },
  "lifecycle_tu": ["nemotron_moe_ultra_model.cpp", "nemotron_moe_ultra_inference_pipeline.cu"],
  "stages": {
    "convert": ["test_nemotron_moe_ultra_convert"],
    "load": ["test_nemotron_moe_ultra_loader", "test_nemotron_ultra_tokenizer_parity"],
    "model": ["test_nemotron_moe_ultra_model", "test_nemotron_ultra_tool_framing"],
    "kernels": ["test_nemotron_moe_ultra_kernels", "test_nemotron_moe_ultra_math",
                "test_nemotron_moe_ultra_constants", "test_nemotron_moe_ultra_expert_cache",
                "test_nemotron_ultra_layers"],
    "selfcheck": ["test_nemotron_ultra_selfcheck"],
    "fsi": ["test_nemotron_ultra_fsi_selfcheck"],
    "code": ["test_nemotron_ultra_code_selfcheck"],
    "det": ["test_nemotron_ultra_det_selfcheck"],
    "perf": ["test_perf_baseline_nemotron_ultra"]
  },
  "perf": { "target": "test_perf_baseline_nemotron_ultra",
            "vault_subdir": "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "timeout": 7200 },
  "target_overrides": {
    "test_nemotron_moe_ultra_constants": { "label": "fast", "env": [] },
    "test_nemotron_moe_ultra_loader": { "label": "fast", "env": [] },
    "test_nemotron_moe_ultra_math": { "label": "fast", "env": [] },
    "test_nemotron_moe_ultra_expert_cache": { "label": "fast", "env": [] },
    "test_nemotron_moe_ultra_model": { "label": "fast", "env": [] },
    "test_nemotron_ultra_tool_framing": { "label": "fast", "env": [] },
    "test_nemotron_ultra_tokenizer_parity": { "env": [] },
    "test_nemotron_ultra_layers": { "label": "inference", "resource_lock": "gpu" },
    "test_nemotron_moe_ultra_selfcheck": { "form": "selfcheck" },
    "test_nemotron_ultra_fsi_selfcheck": { "form": "parity", "kind": "fsi" },
    "test_nemotron_ultra_code_selfcheck": { "form": "parity", "kind": "code" },
    "test_nemotron_ultra_det_selfcheck": { "form": "parity", "kind": "det" }
  },
  "tool_roundtrip": true,
  "auto_path_determinism": false,
  "golden_debt": { "code": "declared debt (see docs/OPEN_TASKS.md)",
                   "det": "declared debt (see docs/OPEN_TASKS.md)",
                   "fsi": "declared debt (see docs/OPEN_TASKS.md)" },
  "report": "",
  "notes": "separate lineage from nemotron-3.5-lightning: grouped router (n_group 8), latent MoE 2048, 108 layers",
  "selfcheck_max_gen": 2,
  "selfcheck_bit_exact": false,
  "selfcheck_frame_prompt": false
}
```
`golden_debt` is flipped to empty per gate in Task 14 as each golden lands; leaving it until then keeps `test_model_suite_guard` honest rather than declaring gates that do not exist.

- [ ] **Step 2: Write `test_nemotron_ultra_tokenizer_parity.cpp`** — same corpus and golden as the sibling (`tests/fixtures/nemotron/tokenizer_parity.txt`, 22 cases), reading the Ultra vault; `SKIP()` when `<models_dir>/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/tokenizer.json` is absent.
- [ ] **Step 3: Write `test_nemotron_ultra_layers.cpp`** skeleton — SKIP 4 unless the slice-oracle fixture directory (Task 13) and the converted tree both exist.
- [ ] **Step 4: Regenerate and check**

Run:
```bash
python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .
cmake --preset linux-release-znver4-aocc && cmake --build --preset linux-release-znver4-aocc
ctest --test-dir build/linux-release-znver4-aocc -L fast --output-on-failure
```
Expected: `docs/MODELS.md` gains the `nemotron-ultra-550b-a55b` section; the fast sweep is green including `test_manifest_sync`, `test_model_independence_guard`, `test_model_suite_guard`, `test_code_naming_guard`, `test_ctest_labels_guard`, `test_infrastructure_guard`.

- [ ] **Step 5: Commit** — `feat(nemotron-ultra): register the model and its lifecycle suite`.

---

### Task 12: Real-weight conversion + loader verification

**Files:**
- Modify: `docs/OPEN_TASKS.md` (a dated session record), `models/nemotron-ultra-550b-a55b.json` only if a measured value must be recorded in `notes`.

- [ ] **Step 1: Confirm the GPU and disk are free** — `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` empty; `df -h /mnt/fast_storage` shows ≥400 GB free.
- [ ] **Step 2: Convert in the background, resumably**

```bash
export LSX_MODEL_ROOT=/mnt/fast_storage/huggingface
nohup ./build/linux-release-znver4-aocc/apps/lsxforge/lsxforge \
  --nemotron_ultra \
  --input /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  --output /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
  > /mnt/fast_storage/tmp/nemotron_ultra_convert.log 2>&1 &
```
Expected: `logestix/model.index.json` appears, then `pinned/` (109 files) and `experts/layer_<l>/exp_<e>.arrow` for the 48 MoE layers (24 576 files, ≈290 GB). Re-running resumes at the first missing layer.

- [ ] **Step 3: Verify the tree** — `python3 tools/dump_ctest_manifest.py` is not relevant here; instead run `ctest --test-dir build/linux-release-znver4-aocc -R test_nemotron_moe_ultra_loader --output-on-failure` with the real vault present, and assert the index reports 24 576 experts, 108 layers and `mtp_tensors == 1037`.
- [ ] **Step 4: Measure and record** the conversion wall time, the achieved write bandwidth, the tree size, and the converter's peak RSS (`/usr/bin/time -v` or the log line). These are report inputs.
- [ ] **Step 5: Commit** — `docs(nemotron-ultra): real-weight conversion run and measured tree inventory`.

---

### Task 13: Slice oracle + per-layer numeric gate

**Files:**
- Create: `tools/nemotron_ultra_oracle.py`
- Modify: `tests/test_nemotron_ultra_layers.cpp` (fill the skeleton), `tests/fixtures/nemotron_ultra/layers_prompt.txt` (new)

- [ ] **Step 1: Write `tools/nemotron_ultra_oracle.py`**

`--checkpoint DIR --layers 0,1,7 --prompt "The capital of France is" --out FILE` builds a real `NemotronHForCausalLM` whose `config.layers_block_type` is the selected subset (keeping the real embedding, `norm_f`, `lm_head`), loads only those shards' tensors (BF16), runs a no-grad forward, and writes per-layer hidden-state summaries (first 64 dims + L2 norm + position count) as JSON. CPU only, ~4 GB peak.
Run it once: `python3 tools/nemotron_ultra_oracle.py --checkpoint /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 --layers 0,1,7 --prompt "The capital of France is" --out tests/fixtures/nemotron_ultra/layers_slice.json`
Expected: the file exists and contains ≥3 layer entries at ≥8 positions.

- [ ] **Step 2: Write the failing gate** — `test_nemotron_ultra_layers` loads the converted tree, runs the engine's per-layer dump at the same prompt and positions, and asserts per selected layer `relL2 < 2e-2` and `cos > 0.99`; SKIP 4 when either the fixture or the tree is absent.
- [ ] **Step 3: Run it** — `ctest --test-dir build/linux-release-znver4-aocc -R test_nemotron_ultra_layers --output-on-failure`. If it fails on the attention layer, root-cause by first checking the router selection against the reference (Task 5's math on the real logits) before touching the kernel.
- [ ] **Step 4: Commit** — `test(nemotron-ultra): slice-oracle per-layer numeric gate`.

---

### Task 14: Gates on the real weights — SelfCheck, tool round-trip, FSI/CODE/DET

**Files:**
- Modify: `models/nemotron-ultra-550b-a55b.json` (`golden_debt` per landed gate, `selfcheck_*` values measured), `docs/SELFCHECK_MATRIX.md`, `tests/fixtures/{fsi,code,det}/nemotron-ultra-550b-a55b.json` (new), `docs/OPEN_TASKS.md`

- [ ] **Step 1: SelfCheck** — `ctest --test-dir build/linux-release-znver4-aocc -R test_nemotron_ultra_selfcheck --output-on-failure`. Expected: Paris, ≥3 cycles bit-identical, tool round-trip green (`tool_roundtrip: true`). Record `logits_hash` and the decoded tokens.
- [ ] **Step 2: Behavioural sanity** — run a ~200-token free-form generation through `lsxbenchmark --localmodel <vault dir>` and confirm it is readable prose with no tag soup; record the output head in the session notes.
- [ ] **Step 3: Capture the three goldens** with `tools/nemotron_ultra_capture.py` (template-derived; the source cannot execute here) and run the three parity targets. Expected: FSI 3/3, CODE 3/3, DET 3/3 with three byte-identical runs.
- [ ] **Step 4: Flip `golden_debt`** for whatever landed, and leave a declared debt line for whatever did not, with its measured reason. Re-run `ctest -L fast` and confirm `test_model_suite_guard` + `test_manifest_sync` still pass.
- [ ] **Step 5: Commit** — `test(nemotron-ultra): real-weight SelfCheck, tool round-trip and parity goldens`.

---

### Task 15: Perf baseline at the canonical protocol

**Files:**
- Modify: `tests/test_perf_baseline.cpp` (new `PerfModel` row + `TEST_CASE("perf baseline: nemotron-ultra-550b-a55b", "[perf]")`), `tests/perf/perf_baselines.json`, `models/nemotron-ultra-550b-a55b.json` (only if the timeout must move)

- [ ] **Step 1: Confirm exclusive GPU** — `nvidia-smi --query-compute-apps` empty, nothing else on the card; record `nvidia-smi -q -d POWER` warm-up state.
- [ ] **Step 2: Add the case** with the canonical prompt (`"The capital of France is"`) and `max_gen = 2048`, matching the sibling rows.
- [ ] **Step 3: Run twice and require <1 % agreement**

```bash
ctest --test-dir build/linux-release-znver4-aocc -R test_perf_baseline_nemotron_ultra --output-on-failure
```
Expected: first run seeds the row (INFO "entry absent"), second run passes within −3 %. Record `perf` and `r3` t/s, TTFT, watts and resident bytes.

- [ ] **Step 4: Attribute the wall time** — one `nsys profile -t cuda` run over 64 tokens; report the split between expert-store reads, dense GEMVs and attention, and check whether `Prewarm` should be skipped for a ~590-slot cache (measure TTFT both ways).
- [ ] **Step 5: Commit** — `perf(nemotron-ultra): seed the canonical 2048-token baseline`.

---

### Task 16: Stage-9 report

**Files:**
- Create: `docs/model-reports/2026-11-DD-NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.md`
- Modify: `docs/model-reports/README.md`, `models/nemotron-ultra-550b-a55b.json` (`"report"` path), `docs/OPEN_TASKS.md`, `docs/MODELS.md` (regenerate)

- [ ] **Step 1: Write the report** in the shape of `docs/model-reports/2026-11-02-Mistral-Large-4-1T-A52B.md`: header table (model, converted layout, source, converted tree, hardware, runtime, date, reference), a "verdict at a glance", a publish-ready post, an Evidence section listing every gate target with its measured verdict and `logits_hash`, the golden provenance (template-derived vs source), and an explicit "not measured" list.
- [ ] **Step 2: Point the manifest at it and regenerate**

```bash
python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .
ctest --test-dir build/linux-release-znver4-aocc -R "test_manifest_sync|test_model_suite_guard" --output-on-failure
```
Expected: green; `docs/MODELS.md` links the report.

- [ ] **Step 3: Full branch verification** — build both presets, run `ctest -L fast` to completion, then the GPU labels one at a time (`ctest -L inference`, `-L convert`, `-L perf -R nemotron`). Record the exact counts in the report.
- [ ] **Step 4: Commit** — `docs(report): Nemotron-3-Ultra-550B-A55B integration report`.

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task: frozen passport → T1; converted layout + converter (incl. per-layer streaming, resumability, strictness) → T3; index/loader/geometry gate → T4; kernel/math requirements (grouped router, latent MoE, mamba widths, NoPE, relu²) → T5, T6; expert store → T7; pipeline → T8; model leaf + tool codec + seven lifecycle markers → T9; forge flag → T10; guards/registries + tokenizer parity → T11; evidence tier 1 (slice oracle) → T13; tiers 2–3 (behavioural, coherence) → T14; VRAM plan/levers + perf → T15; report → T16; real-weight conversion → T12. Two spec items are deliberately deferred to measurement rather than code: the second oracle slice (`--layers` accepts 98) and the `Prewarm` decision (T15 step 4).

**2. Step scan.** No step says "handle edge cases" or "add validation"; every test step names its cases and their assertions, every implementation step names the signature and the values the spec pins. The three places a body would be a transcript (reference math, kernels, pipeline) point at the exact reference lines instead.

**3. Type consistency.** `UltraIndex` fields used in T8 (`moe_latent_size`, `n_group`, `topk_group`, `routed_scaling_factor`) are declared in T4; `UltraSession.moe_latent` matches; `RunUltraRouter`'s parameter order matches `ref::RouterTopKGrouped`; `UltraExpertView` field names match `ref`/pipeline usage; `ConversionWindow` is defined in T3 and consumed in T10.

**4. Review Focus.** All five lines have an owning task and a named test case (loader 30B-tree rejection T4; latent present/absent T3+T4; group divisibility T3; tie determinism T5+T6; partial-tree resume T3+T4).

**5. Proportion.** The plan is ~1.3× the spec's length, with code only where the spec fixes exact values (passport, manifest row, test assertions). Bodies for the converter, kernels and pipeline are described by responsibility and reference lines, not transcribed.
