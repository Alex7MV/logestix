# NVFP4 full-E4M3 block-scale codec — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` (inline) to implement this plan task-by-task. **Do NOT use subagents** — the repo's `AGENTS.md` forbids Task subagents; all work is done in the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pow2-only NVFP4 block-scale codec with the hardware-native full unsigned E4M3 (UE4M3) codec (RNE encode, mantissa-aware decode) without changing the decode of any existing converted tree.

**Architecture:** A single canonical host/device codec pair (`FloatToE4m3`/`E4m3ToFloat`, integer-only for host/device bit-identity) in `nvfp4_scale.h`; the device duplicate `E4M3ScaleToFloat` and the public host wrappers delegate to it. Mapping every encode/decode call site onto the new pair flips the whole engine at once. The change is proven by an exhaustive decode/encode host test plus a device-parity test, and by the existing pow2 trees decoding bit-identically.

**Tech Stack:** C++20, CUDA 13.4 (CC 12.x), Catch2 v3.16, CMake presets (`linux-debug`), CMake test labels (`fast`).

## Global Constraints

- Host C++20; logging only via spdlog (no `std::cout`/`printf`); comments are REQUIRED on the codec (invariant pinning, per `nvfp4-blackwell-conventions`).
- CUDA docs are external: `/home/alexmv2025/projects/cuda/docs` (MCP dump). Authoritative facts: `NVIDIA_DOCS_NVFP4_MEMORY.md` §1, §B4.
- Block scale semantics: unsigned E4M3 `value = 2^(e-7)*(1+m/8)`, `e=(b>>3)&0xF`, `m=b&0x7`; `e==0 -> 0` (no subnormals); `0x7F` reserved (NaN) never emitted; max finite `0x7E = 448`; min normal `0x08 = 2^-6`.
- Backward-compatibility is mandatory: a byte with `m==0` decodes to exactly `2^(e-7)` (bit-identical to the old codec).
- Every new test target carries a CTest label (`fast`); run the guard tests after adding files.
- Commit only on explicit operator request (AGENTS.md) — the plan's commit steps are marked, but do NOT run them unless told.
- Build: `cmake --preset linux-debug && cmake --build --preset linux-debug`, run with `ctest --preset linux-debug` (or `ctest -R <name>`).

---

### Task 1: Canonical full-E4M3 codec + exhaustive host test

**Files:**
- Modify: `libs/lsxcommon/src/nvfp4_scale.h`
- Create: `tests/test_nvfp4_scale.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `uint8_t lsxcommon::nvfp4::FloatToE4m3(float s)` — `LSX_NVFP4_SCALE_HD`, RNE encode of `s>=0`.
  - `float lsxcommon::nvfp4::E4m3ToFloat(uint8_t b)` — `LSX_NVFP4_SCALE_HD`, field-zero decode.
  - (old `FloatToE4m3P2`/`E4m3P2ToFloat` remain, unchanged, until Task 5.)

- [ ] **Step 1: Write the failing test** — create `tests/test_nvfp4_scale.cpp`:

```cpp
// test_nvfp4_scale.cpp — canonical full-E4M3 (UE4M3) NVFP4 block-scale codec.
// Contract: value = 2^(e-7)*(1+m/8); e=(b>>3)&0xF, m=b&0x7;
// e==0 -> 0 (no subnormals); 0x7F reserved (NaN) never emitted.
#include "nvfp4_scale.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Independent reference (std::ldexp), not the codec under test.
float RefDecode(uint8_t b)
{
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0) return 0.0f;
    if (e == 15 && m == 7) return std::nanf("");
    return std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

bool SameBits(float a, float b)
{
    uint32_t x, y;
    std::memcpy(&x, &a, sizeof(x));
    std::memcpy(&y, &b, sizeof(y));
    return x == y;
}

} // namespace

TEST_CASE("full-E4M3 decode: exhaustive over all 128 positive bytes",
          "[nvfp4][scale][e4m3]")
{
    for (int b = 0; b < 0x80; ++b) {
        const float got = lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(b));
        const float want = RefDecode(static_cast<uint8_t>(b));
        INFO("byte=0x" << std::hex << b);
        if (std::isnan(want)) {
            CHECK(std::isnan(got));
        } else {
            CHECK(SameBits(got, want));
        }
    }
}

TEST_CASE("full-E4M3 encode: RNE points and clamps",
          "[nvfp4][scale][e4m3]")
{
    using lsxcommon::nvfp4::FloatToE4m3;
    CHECK(FloatToE4m3(0.0f) == 0x00);
    CHECK(FloatToE4m3(-0.0f) == 0x00);
    CHECK(FloatToE4m3(std::nanf("")) == 0x00);
    CHECK(FloatToE4m3(1e-45f) == 0x08);   // float subnormal -> min normal
    CHECK(FloatToE4m3(0.0078125f) == 0x08); // 2^-7 clamps up to 2^-6
    CHECK(FloatToE4m3(0.015625f) == 0x08);  // 2^-6
    CHECK(FloatToE4m3(1.0f) == 0x38);
    CHECK(FloatToE4m3(3.0f) == 0x44);       // was 0x48 under pow2
    CHECK(FloatToE4m3(0.1f) == 0x1D);       // 2^-4*1.625
    CHECK(FloatToE4m3(448.0f) == 0x7E);
    CHECK(FloatToE4m3(449.0f) == 0x7E);
    CHECK(FloatToE4m3(1e30f) == 0x7E);
    CHECK(FloatToE4m3(256.0f) == 0x78);     // exact, was clamped to 0x70
}

TEST_CASE("full-E4M3 encode: ties-to-even", "[nvfp4][scale][e4m3]")
{
    using lsxcommon::nvfp4::FloatToE4m3;
    // 1.0625 = 1 + 1/16 -> midpoint between m=0 and m=1 -> even -> m=0.
    CHECK(FloatToE4m3(1.0625f) == 0x38);
    // 1.1875 = 1 + 3/16 -> midpoint between m=1 and m=2 -> even -> m=2.
    CHECK(FloatToE4m3(1.1875f) == 0x3A);
}

TEST_CASE("full-E4M3: pow2 bytes are bit-identical to the old codec",
          "[nvfp4][scale][e4m3]")
{
    // Every exact power of two keeps its old byte and decode.
    for (int e = 1; e <= 14; ++e) {
        const float v = std::ldexp(1.0f, e - 7);
        const uint8_t b = lsxcommon::nvfp4::FloatToE4m3(v);
        CHECK(b == static_cast<uint8_t>(e << 3));
        CHECK(SameBits(lsxcommon::nvfp4::E4m3ToFloat(b), v));
    }
    // Field-zero contract: 0x00..0x07 and 0x80..0x87 all decode to 0.
    for (int b = 0; b <= 7; ++b)
        CHECK(lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(b)) == 0.0f);
    for (int b = 0x80; b <= 0x87; ++b)
        CHECK(lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(b)) == 0.0f);
    CHECK(FloatToE4m3(3.0f) != 0x00);       // sanity: encoder live
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_nvfp4_scale 2>&1 | tail`
Expected: FAIL to compile — `E4m3ToFloat`/`FloatToE4m3` not declared (this is the RED state for a new API).

- [ ] **Step 3: Implement the codec** — append to `libs/lsxcommon/src/nvfp4_scale.h` before the closing namespaces (after `E4m3P2ToFloat`):

```cpp
// Encode s >= 0 as the nearest unsigned E4M3 (UE4M3) byte, ties-to-even.
// Layout value = 2^(e-7)*(1+m/8), e=(b>>3)&0xF, m=b&0x7. s==0 (or negative/
// NaN) -> 0x00; s < 2^-6 -> 0x08 (min normal; no subnormals); s > 448 ->
// 0x7E (max finite). The reserved NaN slot 0x7F is never emitted. Pure
// integer arithmetic so host and device are bit-identical by construction.
LSX_NVFP4_SCALE_HD inline uint8_t FloatToE4m3(float s)
{
    if (!(s > 0.0f)) return 0x00;          // 0, -0, negative, NaN
    uint32_t bits;
    std::memcpy(&bits, &s, sizeof(bits));
    const uint32_t exp8 = (bits >> 23) & 0xFFu;
    if (exp8 == 0xFFu) return 0x7E;        // Inf -> max finite
    if (exp8 == 0u) return 0x08;           // float subnormal -> min normal
    const int e = static_cast<int>(exp8) - 127;  // floor exponent
    if (e < -6) return 0x08;
    if (e > 8) return 0x7E;
    const uint32_t mant = bits & 0x7FFFFFu;
    const uint32_t q = mant >> 20;         // top 3 bits = E4M3 mantissa
    const uint32_t rem = mant & 0xFFFFFu;
    int me = static_cast<int>(q);
    const uint32_t half = 0x80000u;        // 2^19 = half a mantissa step
    if (rem > half || (rem == half && (me & 1))) ++me;  // RNE
    int E = e + 7;
    if (me == 8) { me = 0; ++E; }          // carry into the exponent
    if (E < 1) return 0x08;
    if (E > 15) return 0x7E;
    if (E == 15 && me > 6) return 0x7E;    // cannot reach the 0x7F NaN slot
    return static_cast<uint8_t>((E << 3) | me);
}

// Decode one block-scale byte: e4 field 0 -> 0.0f (field-zero contract,
// no subnormals); else 2^(e-7)*(1+m/8), exact in fp32. 0x7F (reserved,
// never emitted) -> NaN. mantissa==0 bytes reproduce the old pow2 decode
// bit-for-bit, so every existing converted tree is unchanged.
LSX_NVFP4_SCALE_HD inline float E4m3ToFloat(uint8_t b)
{
    const uint32_t e = (b >> 3) & 0xFu;
    if (e == 0u) return 0.0f;
    const uint32_t m = b & 0x7u;
    if (e == 15u && m == 7u) {
        const uint32_t nan_bits = 0x7FC00000u;
        float nan_v;
        std::memcpy(&nan_v, &nan_bits, sizeof(nan_v));
        return nan_v;
    }
    const uint32_t bits = ((e + 120u) << 23) | (m << 20);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}
```

- [ ] **Step 4: Register the test** in `tests/CMakeLists.txt`, next to the other `fast` host tests (follow the `test_deepseek_moe_loader` block: `add_executable`, `target_link_libraries(... Catch2::Catch2WithMain)`, `add_test`, then `set_tests_properties(test_nvfp4_scale PROPERTIES LABELS fast)`). `nvfp4_scale.h` lives in `libs/lsxcommon/src`; the test target must include that dir (mirror the existing `target_include_directories` used by `test_lsx_kernels`).

- [ ] **Step 5: Build + run**

Run: `cmake --build --preset linux-debug --target test_nvfp4_scale && ctest --test-dir build/linux-debug -R test_nvfp4_scale --output-on-failure`
Expected: PASS (4 test cases).

- [ ] **Step 6: Commit** (only on explicit request): `git add libs/lsxcommon/src/nvfp4_scale.h tests/test_nvfp4_scale.cpp tests/CMakeLists.txt && git commit -m "feat: add full unsigned-E4M3 NVFP4 block-scale codec"`

---

### Task 2: Switch the global decode + offline encode onto the new codec

**Files:**
- Modify: `libs/lsxcommon/src/nvfp4_codec.h:47-55`
- Modify: `libs/lsxcommon/include/lsxcommon/schema.h:240-241`
- Modify: `libs/lsxcommon/src/schema.cpp:33-44`
- Modify: `libs/lsxcommon/src/safetensors_gpu.cu:335,340`
- Modify: `libs/lsxcommon/src/lsx_tc_gemm.cu:39,66,810`
- Modify: `tests/test_lsx_kernels.cu:220-296`
- Modify: `tests/test_deepseek_moe_loader.cpp:104-142`
- Modify: `tests/test_deepseek_moe_inference_kernels.cu:462-540`
- Modify: `tests/test_blackwell_tc_mma.cu:482-539`

**Interfaces:**
- Consumes: `nvfp4::FloatToE4m3`, `nvfp4::E4m3ToFloat` (Task 1).
- Produces: `E4M3ScaleToFloat` (device) and `float_to_e4m3`/`e4m3_to_float` (host public API) now full-E4M3.

- [ ] **Step 1: Point the device decode at the canonical codec.** In `libs/lsxcommon/src/nvfp4_codec.h`, add `#include "nvfp4_scale.h"` after `<cuda_runtime.h>`, and replace the `E4M3ScaleToFloat` body (and its comment) with a delegation. New body:

```cpp
// Full unsigned-E4M3 block scale (mantissa-aware). Delegates to the single
// canonical codec in nvfp4_scale.h; e4==0 -> 0.0, else 2^(e-7)*(1+m/8).
__host__ __device__ __forceinline__ float E4M3ScaleToFloat(uint8_t e4m3)
{
    return ::lsxcommon::nvfp4::E4m3ToFloat(e4m3);
}
```

- [ ] **Step 2: Rename + retarget the public host wrappers.** In `schema.h`, rename `float_to_e4m3_p2` → `float_to_e4m3` and `e4m3_p2_to_float` → `e4m3_to_float`. In `schema.cpp:33-44`, rename the definitions and delegate:

```cpp
uint8_t float_to_e4m3(float s) { return nvfp4::FloatToE4m3(s); }
float e4m3_to_float(uint8_t e4m3) { return nvfp4::E4m3ToFloat(e4m3); }
```

- [ ] **Step 3: Update callers.**
  - `safetensors_gpu.cu:335`: `uint8_t e4 = lsxcommon::float_to_e4m3(s);`; `:340`: `lsxcommon::e4m3_to_float(e4)`.
  - `lsx_tc_gemm.cu:66`: `lsxcommon::nvfp4::FloatToE4m3(...)`; `:810`: `lsxcommon::nvfp4::FloatToE4m3(...)`. Update the comment block at `:38-40` to name `FloatToE4m3` (RNE) instead of `FloatToE4m3P2` (ceil pow2).

- [ ] **Step 4: Update the tests that pinned pow2 encode output.**
  - `test_lsx_kernels.cu:220-296`: replace the `FloatToE4m3P2 edge classes...` case with a full-E4M3 case (rename symbols, change the ceil expectations to RNE: `0.1f -> 0x1D`, `1.5f -> 0x3C`, `6.5f -> 0x4D`, `300.0f -> 0x79`, drop `FloatToE4m3P2(256.0f)==0x70` in favour of `0x78`, and drop the property-sweep assertions that the mantissa field is empty / decoded `< 2*s`). Keep the pow2 identity checks.
  - `test_deepseek_moe_loader.cpp:104-142`: rename the case to `float_to_e4m3 handles subnormals safely`; keep zero/subnormal/pow2 sections (bytes unchanged) and replace `"non-power-of-2 rounds up"` with `"non-power-of-2 rounds to nearest"` asserting `float_to_e4m3(3.0f)==0x44` and `e4m3_to_float(0x44)==3.0f`.
  - `test_deepseek_moe_inference_kernels.cu:462-540`: update the comment (floor vs RNE) and rename `e4m3_p2_to_float` → `e4m3_to_float` at `:521`. The magnitude assertions stay.
  - `test_blackwell_tc_mma.cu:482-539`: rename the case and `float_to_e4m3_p2` → `float_to_e4m3` at `:530,532`; the comment becomes "byte-match the canonical RNE full-E4M3 encoder".

- [ ] **Step 5: Build the affected targets**

Run: `cmake --build --preset linux-debug -j $(nproc) 2>&1 | rg -n "error:" | head`
Expected: no errors.

- [ ] **Step 6: Run the unit tests**

Run: `ctest --test-dir build/linux-debug -R "test_nvfp4_scale|test_lsx_kernels|test_deepseek_moe_loader|test_deepseek_moe_inference_kernels|test_blackwell_tc_mma" --output-on-failure`
Expected: PASS.

- [ ] **Step 7: Commit** (explicit request only).

---

### Task 3: Device parity for the codec (exhaustive kernel vs host)

**Files:**
- Modify: `tests/test_lsx_kernels.cu`
- Interference: `nvfp4_codec.h` is included by the test's CUDA object.

**Interfaces:**
- Consumes: `gpu::nvfp4::E4M3ScaleToFloat` (Task 2), `nvfp4::E4m3ToFloat` (Task 1).

- [ ] **Step 1: Write the failing/parity test** — add to `tests/test_lsx_kernels.cu` a small kernel + case (adapt the file's existing device-buffer helpers):

```cpp
__global__ void decode_e4m3_scales_kernel(const uint8_t* in, float* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(in[i]);
}

TEST_CASE("E4M3ScaleToFloat device parity, all 128 positive bytes",
          "[lsxk][quant][scale]")
{
    std::vector<uint8_t> h_in(128);
    for (int i = 0; i < 128; ++i) h_in[i] = static_cast<uint8_t>(i);
    uint8_t* d_in = nullptr; float* d_out = nullptr;
    REQUIRE(cudaMalloc(&d_in, 128) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, 128 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_in, h_in.data(), 128, cudaMemcpyHostToDevice) == cudaSuccess);
    decode_e4m3_scales_kernel<<<1, 128>>>(d_in, d_out, 128);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> h_out(128);
    REQUIRE(cudaMemcpy(h_out.data(), d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < 128; ++i) {
        const float want = lsxcommon::nvfp4::E4m3ToFloat(static_cast<uint8_t>(i));
        uint32_t a, b; std::memcpy(&a, &h_out[i], 4); std::memcpy(&b, &want, 4);
        INFO("byte=0x" << std::hex << i);
        if (std::isnan(want)) CHECK(std::isnan(h_out[i])); else CHECK(a == b);
    }
    cudaFree(d_in); cudaFree(d_out);
}
```

- [ ] **Step 2: Run**

Run: `ctest --test-dir build/linux-debug -R test_lsx_kernels --output-on-failure`
Expected: PASS (device matches the exhaustive host table).

- [ ] **Step 3: Commit** (explicit request only).

---

### Task 4: Remove the retired pow2 codec + pin the invariant in comments

**Files:**
- Modify: `libs/lsxcommon/src/nvfp4_scale.h` (delete `FloatToE4m3P2`, `E4m3P2ToFloat`)
- Modify: `libs/lsxcommon/src/lsx_kernels_quant.cuh:102-104`

- [ ] **Step 1: Confirm no references remain**

Run: `rg -n "FloatToE4m3P2|E4m3P2ToFloat|float_to_e4m3_p2|e4m3_p2_to_float" libs apps tests`
Expected: no output.

- [ ] **Step 2: Delete the old functions** from `nvfp4_scale.h` and update the file header comment (it currently says "canonical power-of-two E4M3 block-scale codecs") to describe the full UE4M3 contract and the RNE encoder. Update `lsx_kernels_quant.cuh:102-104` to reference `nvfp4::FloatToE4m3`.

- [ ] **Step 3: Build + run the whole `fast` label**

Run: `cmake --build --preset linux-debug -j $(nproc) && ctest --test-dir build/linux-debug -L fast --output-on-failure`
Expected: PASS, zero failures (this is the numeric gate before any perf claim).

- [ ] **Step 4: Run the guard tests** (naming, labels, independence)

Run: `ctest --test-dir build/linux-debug -R "guard|naming|labels" --output-on-failure`
Expected: PASS. If `test_code_naming_guard` rejects `test_nvfp4_scale`, rename it to the reviewed pattern and update `tests/CMakeLists.txt`; if a label is missing, `test_ctest_labels_guard` names the target.

- [ ] **Step 5: Commit** (explicit request only).

---

### Task 5: Update the `nvfp4-blackwell-conventions` skill and the NVIDIA mirror note

**Files:**
- Modify: `.opencode/skills/nvfp4-blackwell-conventions/SKILL.md`
- Modify (external, not committed): `/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md`

- [ ] **Step 1: Rewrite the skill's NVFP4 codec section.** Replace the sentence that says the pow2 E4M3 scale decode lives in `nvfp4_codec.h` and the "pow2-only" wording with:

  > The NVFP4 block scale is **full unsigned E4M3 (UE4M3)** — `value = 2^(e-7)*(1+m/8)`, `e=(b>>3)&0xF`, `m=b&0x7`; `e==0 -> 0` (field-zero contract, no subnormals), `0x7F` reserved. Canonical codecs: `nvfp4_scale.h::FloatToE4m3` (RNE encode, integer-only for host/device bit-identity) and `E4m3ToFloat` (device delegate `nvfp4_codec.h::E4M3ScaleToFloat`). The E8M0→E4M3 repack helpers stay pow2-exact because E8M0 is a power of two. A codec change must update this skill + the regression test in the same commit.

- [ ] **Step 2: Update the external mirror** `NVIDIA_DOCS_NVFP4_MEMORY.md` §1: replace the "logestix deviation (deliberate, documented)" paragraph with the now-native UE4M3 contract and note the RNE encode; keep the scale-byte zero-semantics regression paragraph. (Outside the repo — do not `git add`.)

- [ ] **Step 3: Commit the skill** (explicit request only).

---

### Task 6: Re-conversion + per-model lifecycle verification report

**Precondition (operator):** delete the stale `logestix/` converted subdirs (the encoder output changed). The source HF files stay.

**Files:**
- Create: `docs/superpowers/plans/2026-10-06-nvfp4-e4m3-rollout-report.md`
- Modify: `tests/perf/perf_baselines.json` (hashes only, after re-conversion)

- [ ] **Step 1: Enumerate converted trees and their forge flags**

Run: `for d in /mnt/llm_models/models/*/logestix; do echo "$d"; done`
Expected: the set of converted families. Map each to its `lsxforge` flag (Nemotron-NVFP4: `--nemotron-nvfp4` once that flag lands; others per `apps/lsxforge/src/main.cpp`).

- [ ] **Step 2: Re-convert each family** with its flag (one GPU job at a time — `running-a-model` §8). Record wall time and success per family.

- [ ] **Step 3: Run the lifecycle stages per model** and capture pass/fail/skip + `logits_hash` + t/s for: convert, loader, model, `--selfcheck`, FSI, CODE, DET, perf. Commands use the existing env vars (`LSX_TEST_<FAMILY>_ROOT`, `LSX_PERF_VAULT_SUBDIR`).

Run (example shape): `ctest --preset linux-debug -R "<family>.*(selfcheck|fsi|code|det)" --output-on-failure`

- [ ] **Step 4: Re-record moved hashes** in `tests/perf/perf_baselines.json` for every family whose `kAuto` TC act-quant numerics moved; keep the `r3`/`kOff` entries. Run `ctest -R test_perf_baseline` with the write flag the repo uses.

- [ ] **Step 5: Write the report** to the file above: a table of model → stage → verdict, plus the codec evidence (Task 4's `fast` run) and the bit-identical-decode proof for pre-existing trees from Task 2.

- [ ] **Step 6: Commit** (explicit request only).

---

## Self-Review

**Spec coverage:**
- §1 goal / §3 semantics → Task 1 (codec + tests).
- §3.1 backward compat → Task 1 pow2 test + Task 2 tests + Task 4 `fast`.
- §4.1 codec/helpers → Tasks 1–4; `E8m0ToE4m3Pow2Pair` untouched (Task 4 confirms).
- §4.2 non-changes → enforced by leaving `repack_e8m0_ue4m3_kernel`, E2M1, field-zero test as-is.
- §5 testing → Tasks 1–4.
- §6 rollout + lifecycle report → Task 6.
- §7 skill/docs → Task 5.
- §8 non-goals → no task adds a pow2 flavor or changes E8M0.

**Placeholder scan:** no TBD/TODO; Task 1 code complete; edits elsewhere give exact file:line and exact new strings.

**Type consistency:** `FloatToE4m3`/`E4m3ToFloat` names stable across Tasks 1–4; `float_to_e4m3`/`e4m3_to_float` renamed in Task 2 and used consistently in Task 6.
