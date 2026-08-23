# Blackwell CUDA Audit — Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish measured performance baselines for all three models on Blackwell and eliminate the safe (bit-exact-preserving) synchronization/transfer bottlenecks.

**Architecture:** Catch2 perf-harness tests reuse the existing real-model loading pattern (`LsxModelFactory::Create` + env-var path override + SKIP without weights) and report tok/s/ttft through spdlog. Sync fixes stay inside each model's generation loop and keep the AGENTS.md single-stream invariant; every fix is gated by the existing bit-exact selfcheck suites.

**Tech Stack:** CUDA 13.3 (sm_120), Catch2 v3.7.0, CMake presets (`windows-debug`), spdlog.

**Scope note:** Phases 2 (occupancy) and 3 (tensor-core fast-math path) get their own plans *after* this plan's baseline numbers exist — they depend on measurements this plan produces.

**Build commands (this machine):**
```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --build build --config Debug --target <target>
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --preset windows-debug -R <regex>
```

---

### Task 1: Findings registry document

**Files:**
- Create: `docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md`

- [ ] **Step 1: Create the registry with the six known findings**

```markdown
# Blackwell CUDA Audit — Findings Registry

Companion to `2026-08-23-blackwell-cuda-audit-design.md`. Every finding gets a
severity (P0 blocker / P1 major / P2 minor), file:line evidence, remediation
status, and before/after measurements when available.

| ID | Sev | Dimension | Evidence | Summary | Status |
|---|---|---|---|---|---|
| F-01 | TBD | A | glm_inference.cu:84 | Scalar GEMM kernels on CUDA cores; no tensor-core MMA usage anywhere | open |
| F-02 | TBD | B/C | qwen_inference.cu:1996 | Final-logits publish uses blocking null-stream cudaMemcpy inside timed region | open |
| F-03 | TBD | C | glm_inference.cu:1649-1653 | Per-token H2D roundtrip of sampled id + full cudaStreamSynchronize | open |
| F-04 | TBD | C | deepseek_inference_pipeline.cu:772 | cudaDeviceSynchronize inside pipeline | open |
| F-05 | TBD | A | various launch sites | Degenerate launch configs (<<<1,...>>>, flat 256-thread blocks); no occupancy tuning vs 48 warps/SM | open |
| F-06 | TBD | E | libs/lsxcommon/CMakeLists.txt | -fmad=false applied to whole lsxcommon_glm_cu object library | open |

## Baseline measurements (Phase 0)

Filled by Task 5. Format per model:
`<model> @ <gpu>: <tok/s> decode, ttft <ms>, total <ms> (<gen tokens>)`

## New findings discovered during execution

Append below with next free F-nn.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md
git commit -m "docs: add blackwell audit findings registry"
```

---

### Task 2: GLM perf harness test

**Files:**
- Create: `tests/test_blackwell_perf_glm.cpp`
- Modify: `tests/CMakeLists.txt` (append after the `test_glm_selfcheck` block, ~line 444)

- [ ] **Step 1: Write the harness test**

```cpp
// TDD — Blackwell perf baseline for GLM-5.2 (audit Phase 0).
// Reporting test: measures decode throughput and logs a machine-readable
// summary. SKIPs without the real model (SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

namespace {

std::filesystem::path GlmModelPath()
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "GLM-5.2";
}

int PerfGenTokens()
{
    if (const char *p = std::getenv("LSX_TEST_PERF_GEN_TOKENS")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return 16;
}

} // namespace

TEST_CASE("GLM-5.2 perf baseline: decode throughput", "[glm][perf][real-model]")
{
    const std::filesystem::path model = GlmModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("GLM real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    // Minimal deterministic prompt: BOS-framed short text via the model
    // tokenizer; falls back to raw ids {gMASK, sop} framing like the selfcheck
    // harness if template pieces exist.
    lsxcommon::ArrowTokenizerDb &tok = m->Tokenizer();
    std::vector<int32_t> ids;
    const char *prompt = "The capital of France is";
    ids.resize(512);
    int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), nullptr);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = PerfGenTokens();

    auto r = m->Infer(req);
    REQUIRE(r.ok);
    REQUIRE(r.gen_tokens.size() == static_cast<size_t>(req.max_gen_tokens));

    const double gen = static_cast<double>(r.gen_tokens.size());
    const double tok_s = gen / (r.total_ms / 1000.0);
    SPDLOG_INFO("[PERF][glm-5.2] tok/s={:.1f} ttft_ms={:.2f} total_ms={:.2f} "
                "gen_tokens={} resident_mb={:.0f}",
                tok_s, r.ttft_ms, r.total_ms, r.gen_tokens.size(),
                static_cast<double>(r.resident_bytes) / (1024.0 * 1024.0));

    m->Unload();
}
```

NOTE before compiling: verify exact names against headers —
`r.total_ms` type (`double` vs `float`) in `libs/lsxcommon/include/lsxcommon/inference_gpu.h`,
and whether `arrow_tokenizer_encode`'s 4th arg accepts `nullptr` for capacity
(check `libs/lsxcommon/include/lsxcommon/tokenizer.h`; if capacity pointer is
required, pass a local `size_t cap = ids.size();` like `test_glm_selfcheck.cpp:73-75`
does). Adjust the code accordingly — do NOT invent signatures.

- [ ] **Step 2: Register in tests/CMakeLists.txt**

```cmake
add_executable(test_blackwell_perf_glm test_blackwell_perf_glm.cpp)
target_link_libraries(test_blackwell_perf_glm PRIVATE lsxcommon Catch2::Catch2WithMain)
add_custom_command(TARGET test_blackwell_perf_glm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_blackwell_perf_glm>"
    )
add_test(NAME test_blackwell_perf_glm COMMAND test_blackwell_perf_glm)
set_tests_properties(test_blackwell_perf_glm PROPERTIES SKIP_RETURN_CODE 4)
```

(Mirror the arrow.dll POST_BUILD copy exactly from the `test_glm_selfcheck`
block at `tests/CMakeLists.txt:431-441` — including `arrow_cuda.dll` if that
block copies it.)

- [ ] **Step 3: Build and run**

```powershell
& $cmake --build build --config Debug --target test_blackwell_perf_glm
& build\bin\Debug\test_blackwell_perf_glm.exe   # or via ctest -R test_blackwell_perf_glm
```

Expected without model weights: SKIP (exit code 4). Expected with weights:
PASS plus `[PERF][glm-5.2] tok/s=...` in output.

- [ ] **Step 4: Commit**

```bash
git add tests/test_blackwell_perf_glm.cpp tests/CMakeLists.txt
git commit -m "test: add glm-5.2 blackwell perf baseline harness"
```

---

### Task 3: Qwen + DeepSeek perf harnesses

**Files:**
- Create: `tests/test_blackwell_perf_qwen.cpp`
- Create: `tests/test_blackwell_perf_deepseek.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Copy the loading prelude verbatim**

Open `tests/test_qwen_model.cpp` and `test_deepseek`'s real-model test; copy
each file's model-path resolution + prompt-encoding prelude verbatim into the
new files (replace only the tag names: `[qwen][perf][real-model]`,
`[deepseek][perf][real-model]`).

- [ ] **Step 2: Add the shared measurement body**

Both files get the identical body below (adjust model-type string):

```cpp
    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = PerfGenTokens();

    auto r = m->Infer(req);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());

    const double tok_s =
        static_cast<double>(r.gen_tokens.size()) / (r.total_ms / 1000.0);
    SPDLOG_INFO("[PERF][{}] tok/s={:.1f} ttft_ms={:.2f} total_ms={:.2f}",
                "<model-type>", tok_s, r.ttft_ms, r.total_ms);
    m->Unload();
```

plus the same `PerfGenTokens()` helper as Task 2.

- [ ] **Step 3: Register both targets in tests/CMakeLists.txt**

Same block shape as Task 2 Step 2, targets `test_blackwell_perf_qwen` and
`test_blackwell_perf_deepseek`, `add_test` + `SKIP_RETURN_CODE 4`.

- [ ] **Step 4: Build, run, commit**

```bash
& $cmake --build build --config Debug --target test_blackwell_perf_qwen test_blackwell_perf_deepseek
git add tests/test_blackwell_perf_qwen.cpp tests/test_blackwell_perf_deepseek.cpp tests/CMakeLists.txt
git commit -m "test: add qwen/deepseek blackwell perf baseline harnesses"
```

---

### Task 4: Record baselines (Phase 0 gate)

- [ ] **Step 1: Run all three harnesses with the real models present on the RTX 5060 Ti**, capture `[PERF]` lines.
- [ ] **Step 2: Transcribe numbers into the registry's "Baseline measurements" section** of `2026-08-23-blackwell-cuda-audit-findings.md`.
- [ ] **Step 3: Commit**

```bash
git commit -am "docs: record phase-0 blackwell baselines"
```

Expected: three tok/s numbers committed; these gate all later phases.

---

### Task 5: Fix F-02 — async final-logits publish in Qwen (bit-exact)

**Files:**
- Modify: `libs/lsxcommon/src/qwen_inference.cu:1989-2001`

- [ ] **Step 1: Confirm the red state** — run the existing qwen suite green first:

```powershell
ctest -R qwen -C Debug   # all green BEFORE the change
```

- [ ] **Step 2: Replace the blocking copy**

Current (`qwen_inference.cu:1996`):
```cpp
cudaMemcpy(hlast.data(), d_logits + base_idx, V * sizeof(__half),
           cudaMemcpyDeviceToHost);
```

Replacement (ordered on the inference stream; `t_stop` then covers the copy,
which is ~V*2 bytes — sub-0.1ms):
```cpp
cudaMemcpyAsync(hlast.data(), d_logits + base_idx, V * sizeof(__half),
                cudaMemcpyDeviceToHost, stream);
```
and immediately after, the existing `cudaEventRecord(t_stop, stream)` +
`cudaEventSynchronize(t_stop)` (lines 2004-2005) already guarantee completion
before the conversion loop reads `hlast` — move nothing else.

- [ ] **Step 3: Verify bit-exactness + perf**

```powershell
ctest -R qwen -C Debug          # all green, identical assertions
build\bin\test_blackwell_perf_qwen.exe   # tok/s >= baseline
```

- [ ] **Step 4: Update registry F-02 status -> fixed, commit**

```bash
git commit -am "fix: publish qwen final logits via stream-ordered async copy"
```

---

### Task 6: Fix F-03 — GLM decode loop device-side token feed

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference.cu:1628-1688`

- [ ] **Step 1: Verify GlmArgmaxSlot layout** — `best` must be the first member
(check the struct definition near the top of `glm_inference.cu` or its header).
If not first, stop and adapt offsets.

- [ ] **Step 2: Feed the embed kernel from device memory**

Current (`glm_inference.cu:1665-1672`) does H2D copy of `best` then embeds:
```cpp
e = cudaMemcpyAsync(c.d_ids, &best, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
...
GlmEmbedKernel<<<blocks, kGlmLoopBlock, 0, stream>>>(
    pw.d_embed, c.d_ids, c.d_cur + static_cast<size_t>(pos) * hidden,
    1, hidden);
```

Replacement — point the embed kernel directly at the device argmax slot
(`d_argmax->best` already holds the sampled id on device):
```cpp
{
    const int blocks = (hidden + kGlmLoopBlock - 1) / kGlmLoopBlock;
    GlmEmbedKernel<<<blocks, kGlmLoopBlock, 0, stream>>>(
        pw.d_embed, reinterpret_cast<const int32_t *>(&c.d_argmax->best),
        c.d_cur + static_cast<size_t>(pos) * hidden, 1, hidden);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail(e);
}
```
Delete the `cudaMemcpyAsync(c.d_ids, &best, ...)` call (keep the D2H
`c.h_argmax` copy — the host still needs `best` for the EOS check and
`result.tokens`).

- [ ] **Step 3: Narrow the per-token wait to an event**

Replace (`glm_inference.cu:1651-1652`):
```cpp
e = cudaStreamSynchronize(stream);
```
with an event-scoped wait (declare `cudaEvent_t tok_ready` alongside `t_ttft`
in the function's event setup, `cudaEventCreate`/destroy symmetric):
```cpp
cudaEventRecord(tok_ready, stream);
e = cudaEventSynchronize(tok_ready);
if (e != cudaSuccess) return fail(e);
```

- [ ] **Step 4: Verify**

```powershell
ctest -R glm -C Debug                            # incl. test_glm_selfcheck bit-exact 3-cycle stress: MUST be green
build\bin\test_blackwell_perf_glm.exe            # tok/s >= baseline
```

- [ ] **Step 5: Registry + commit**

```bash
git commit -am "fix: feed glm decode embed from device-side argmax slot"
```

---

### Task 7: Fix F-04 — DeepSeek device-wide sync

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu` (~line 772)

- [ ] **Step 1: Read the surrounding 60 lines.** Confirm which streams have
in-flight work at that point: search the file for `cudaStreamCreate` /
`cudaStream_t` members first (`rg "cudaStream" libs/lsxcommon/src/deepseek_inference_pipeline.cu`).

- [ ] **Step 2: If (and only if) a single stream owns all prior work**, replace
```cpp
INFER_CHECK(cudaDeviceSynchronize());
```
with
```cpp
INFER_CHECK(cudaStreamSynchronize(stream));
```
using that function's local stream handle name. If multiple streams feed the
point, replace instead with `cudaEventRecord` + `cudaEventSynchronize` on each
producer stream — never leave a device-wide barrier.

- [ ] **Step 3: Verify**

```powershell
ctest -R deepseek -C Debug
build\bin\test_blackwell_perf_deepseek.exe
```

- [ ] **Step 4: Registry + commit**

```bash
git commit -am "fix: scope deepseek pipeline teardown sync to the inference stream"
```

---

### Task 5-7 wrap-up

- [ ] Re-run ALL harnesses (Task 4 commands), append after-numbers to the registry next to each F-id, re-run `ctest -R "glm|deepseek|qwen"` fully green, commit registry update.

```bash
git commit -am "docs: record phase-1 post-fix measurements"
```

---

## Self-review notes

- Spec coverage: Phase 0 = Tasks 1-4; Phase 1 sync fixes = Tasks 5-7 (F-02, F-03, F-04). F-01/F-05/F-06 intentionally deferred to Phase 2/3 plans (documented in scope note). Bit-exactness gates = existing suites (Task 6 Step 4). Fast-math flag work = Phase 3 plan (not here).
- Type consistency: `gpu::InferenceResult` fields verified against `inference_gpu.h` (`ttft_ms` double, `gen_tokens`, `final_logits`); `IModel::Infer(const ModelRequest&)` verified against `model.h:55`.
- Known risk flagged inline: `arrow_tokenizer_encode` capacity argument must be checked against `tokenizer.h` before compiling Task 2 (explicit NOTE in step).
