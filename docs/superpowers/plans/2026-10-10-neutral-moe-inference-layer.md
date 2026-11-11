# Neutral routed-MoE inference layer — rename `qwen_moe_inference` to `lsx_moe_inference`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Subagents are FORBIDDEN by AGENTS.md — execute
> inline in the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the routed-MoE kernel layer from the qwen-owned name
`qwen_moe_inference.{h,cu}` to the repo's neutral naming (`lsx_moe_inference.{h,cu}`,
`lsxcommon::moe`), so the neutrality that is already a fact — Gemma MoE, GLM MoE
Flash and Qwen MoE Exp all consume it today — is visible in the name and enforced
by a guard test.

**Architecture:** The file currently mixes two things: ~1370 lines of
geometry-parameterised routed-MoE kernels (`H`, top-k `K` and expert count `E`
are all runtime parameters; there is no hardcoded qwen geometry) and ~860 lines of
qwen-owned generation code welded to `QwenMoePinnedWeights`. Split them first, then
rename only the kernel half. This mirrors the existing `qwen_reduce.cuh` →
`lsx_reduce.cuh` alias pattern already in the repo.

**Tech Stack:** C++20, CUDA 13.4 (sm_120/sm_121 only), CMake 3.25+, Catch2 v3.16.0.

## Global Constraints

- C++20 / CUDA 13.4; target architectures `120-real;121-real` ONLY.
- Logging only via spdlog (`lsxcommon/log.h`). No `printf`, no `std::cout`.
- No new comments unless they carry information; preserve existing Doxygen.
- **No subagents** (AGENTS.md). All work inline.
- Do NOT touch `docs/superpowers/specs/*` or `docs/superpowers/plans/*` other than
  this file — they are dated historical records.
- Neutral layer names carry the `lsx_` prefix: `lsx_inference_nvfp4.{h,cu}`,
  `lsx_gemv.{h,cu}`, `lsx_tc_gemm.{h,cu}`, `lsx_reduce.cuh`, `lsx_kv_cache.h`.
- Commit prefixes: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
  English messages.
- The `static_assert` layout contract between `QwenMoeGvSlotDev` and
  `lsx_tc::LsxTcBatchSlot` (`qwen_moe_inference.cu:37`) MUST survive the rename.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `libs/lsxcommon/include/lsxcommon/lsx_moe_inference.h` | Neutral routed-MoE kernel API + POD device slot types | Create (from `qwen_moe_inference.h`) |
| `libs/lsxcommon/src/lsx_moe_inference.cu` | Neutral routed-MoE kernels | Create (from `qwen_moe_inference.cu` lines 1-1369) |
| `libs/lsxcommon/include/lsxcommon/qwen_moe_generation.h` | qwen-owned generation entry point + scratch API | Create (from the tail of `qwen_moe_inference.h`) |
| `libs/lsxcommon/src/qwen_moe_generation.cu` | qwen generation loop, takes `QwenMoePinnedWeights` | Create (from `qwen_moe_inference.cu` lines 1370-2228) |
| `libs/lsxcommon/include/lsxcommon/qwen_moe_inference.h` | — | Delete |
| `libs/lsxcommon/src/qwen_moe_inference.cu` | — | Delete |
| `libs/lsxcommon/CMakeLists.txt` | targets `lsxcommon_qwen_moe_cu` + link list | Modify |
| `tests/test_model_independence_guard.cpp` | neutrality guard | Modify |

### Symbol mapping (mechanical, applied by Task 3)

| Old (`lsxcommon::qwen_moe`) | New (`lsxcommon::moe`) |
|---|---|
| `QwenMoeRoutedSlot` | `MoERoutedSlot` |
| `QwenMoeGvSlotDev` | `MoEGvSlotDev` |
| `QwenMoeExpertStore` | `MoEExpertStore` |
| `RunQwenMoeRouterTopK` | `RunMoERouterTopK` |
| `RunQwenMoeRoutedFFN` | `RunMoERoutedFFN` |
| `RunQwenMoeAccToFp16` | `RunMoEAccToFp16` |
| `RunQwenMoeAccInitForTest` | `RunMoEAccInitForTest` |
| `RunQwenMoeRoutedGemmVBatch` | `RunMoERoutedGemmVBatch` |
| `RunQwenMoeRoutedGemmVBatchCore` | `RunMoERoutedGemmVBatchCore` |
| `RunQwenMoeGemvGatherTable` | `RunMoEGemvGatherTable` |
| `QwenMoeGemvNohandshake` | `MoEGemvNohandshake` |
| `RunQwenMoeSwigluQuantBatch` | `RunMoESwigluQuantBatch` |
| `RunQwenMoeRoutedTcBatch` | `RunMoERoutedTcBatch` |
| `RunQwenMoeRoutedTcBatchTbl` | `RunMoERoutedTcBatchTbl` |
| `RunQwenMoeSharedCombine` | `RunMoESharedCombine` |
| `TruncateTokensAtEos` | `MoETruncateTokensAtEos` |
| `qwen_moe_*_kernel`, `qwen_*_kernel` (device) | `moe_*_kernel` |

**Stays qwen-owned** (moves to `qwen_moe_generation.h`): `RunQwenMoeGeneration`,
`QwenMoeGenScratch`, `EnsureQwenMoeScratch`, `ReserveQwenMoeArena`,
`FreeQwenMoeGenScratch`, `QwenMoePinnedWeights` fwd-decl.

---

### Task 1: Write the neutrality guard (red before the change)

The guard is the failing test that drives this refactor. Against today's tree it
must fail, because the neutral file does not exist.

**Files:**
- Modify: `tests/test_model_independence_guard.cpp` (append a new `TEST_CASE`)
- Test: `test_model_independence_guard`

**Interfaces:**
- Consumes: `LSX_SOURCE_DIR` compile definition, existing `kSrcDir`, `kIncDir`,
  `ReadFile(const fs::path&)` helpers in that file.
- Produces: guard name `test_model_independence_guard` (existing CTest target).

- [ ] **Step 1: Append the guard test case**

Append at the end of `tests/test_model_independence_guard.cpp`:

```cpp
TEST_CASE("the routed-MoE inference layer is model-neutral in name and deps",
          "[independence]")
{
    // The routed-MoE kernels are geometry-parameterised (H, top-k K and the
    // expert count are runtime parameters) and are already consumed by the
    // Gemma, GLM-flash and Qwen-MoE-exp families. They therefore carry the
    // neutral lsx_ name, and must not name a family or reach into a model
    // loader -- otherwise the name promises neutrality the code does not keep.
    const fs::path hdr = kIncDir / "lsx_moe_inference.h";
    const fs::path cu = kSrcDir / "lsx_moe_inference.cu";
    REQUIRE(fs::exists(hdr));
    REQUIRE(fs::exists(cu));

    static const char *kFamilies[] = {
        "qwen", "gemma", "glm",  "deepseek", "kimi",      "ornith",
        "nemotron", "gigachat", "mimo", "alice", "xing", "granite", "phi"};

    for (const fs::path &p : {hdr, cu}) {
        const std::string text = ReadFile(p);
        REQUIRE_FALSE(text.empty());
        for (const char *fam : kFamilies) {
            INFO(p.filename() << " must not name the family '" << fam << "'");
            REQUIRE(text.find(fam) == std::string::npos);
        }
    }

    // No model loader may be pulled into the neutral translation unit.
    INFO("the neutral layer must not include a model loader");
    REQUIRE(ReadFile(cu).find("qwen_moe_load.h") == std::string::npos);
}
```

- [ ] **Step 2: Build and confirm the guard FAILS for the right reason**

Run:
```bash
cmake --build --preset linux-debug -j 32 --target test_model_independence_guard
ctest --test-dir build/linux-debug -R test_model_independence_guard --output-on-failure
```
Expected: FAIL, with a `REQUIRE(fs::exists(hdr))` violation —
`libs/lsxcommon/include/lsxcommon/lsx_moe_inference.h` does not exist.

- [ ] **Step 3: Commit the red guard**

```bash
git add tests/test_model_independence_guard.cpp
git commit -m "test: add the red neutrality guard for the routed-MoE kernel layer"
```

---

### Task 2: Split the qwen generation loop out of the kernel TU

A pure code move, no renames yet. Doing this before the rename keeps each commit
bisectable: a numeric regression after the rename cannot be confused with a
regression from moving code.

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/qwen_moe_generation.h`
- Create: `libs/lsxcommon/src/qwen_moe_generation.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_moe_inference.h` (drop moved decls)
- Modify: `libs/lsxcommon/src/qwen_moe_inference.cu` (drop moved code)
- Modify: `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Consumes: `QwenMoePinnedWeights` (`qwen_moe_load.h:28`).
- Produces:
  - `qwen_moe_generation.h` declares, in `namespace lsxcommon::qwen_moe`:
    `struct QwenMoeGenScratch`, `cudaError_t EnsureQwenMoeScratch(cudaStream_t, const QwenMoePinnedWeights&, QwenMoeGenScratch&)`,
    `cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch&, size_t)`,
    `void FreeQwenMoeGenScratch(QwenMoeGenScratch&)`,
    `cudaError_t RunQwenMoeGeneration(...)` (full existing parameter list).

- [ ] **Step 1: Create `qwen_moe_generation.h`**

Move these declarations out of `qwen_moe_inference.h` into the new header, keeping
them in `namespace lsxcommon::qwen_moe`. `RunQwenMoeGeneration` references types
from five other headers, so the include list must be complete or the build breaks:

```cpp
#pragma once
// Qwen-MoE generation loop. Model-owned: takes QwenMoePinnedWeights, so it
// stays family-named even though the routed-MoE kernels it calls are neutral
// (see lsx_moe_inference.h after the neutrality rename).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_fp16.h>   // __half, used by QwenMoeGenScratch::d_embed
#include <cuda_runtime.h>

#include "lsxcommon/qwen_inference.h"           // QwenGenerationResult
#include "lsxcommon/qwen_inference_load.h"      // QwenKvCache
#include "lsxcommon/qwen_moe_expert_cache.h"    // QwenMoeExpertDeviceCache
#include "lsxcommon/qwen_moe_load.h"            // QwenMoePinnedWeights
#include "lsxcommon/qwen_moe_inference.h"       // kernel entry points
#include "lsxcommon/sampling.h"                 // ISampler
#include "lsxcommon/token_stream.h"             // GenerationStream

namespace lsxcommon {
namespace qwen_moe {

// QwenMoeGenScratch (qwen_moe_inference.h:208-215), then verbatim
// EnsureQwenMoeScratch / ReserveQwenMoeArena / FreeQwenMoeGenScratch
// (:217-221) and RunQwenMoeGeneration (:226-235).

} // namespace qwen_moe
} // namespace lsxcommon
```

- [ ] **Step 2: Delete the moved declarations from `qwen_moe_inference.h`**

Remove the `QwenMoeGenScratch` struct and the four declarations, and the now-unused
`struct QwenMoePinnedWeights;` forward declaration at line 42. Leave everything
else byte-identical.

- [ ] **Step 3: Create `qwen_moe_generation.cu`**

```bash
cd /mnt/fast_storage/projects/cuda/logestix
git mv libs/lsxcommon/src/qwen_moe_inference.cu \
       libs/lsxcommon/src/qwen_moe_generation.cu
```

Then in the new file keep the includes it still needs and add the new header:
```cpp
#include "lsxcommon/qwen_moe_generation.h"
#include "lsxcommon/qwen_moe_inference.h"
```

- [ ] **Step 4: Create the neutral-kernel TU**

Take `qwen_moe_generation.cu` lines 1-1369 (everything before
`EnsureQwenMoeScratch`) as the new `libs/lsxcommon/src/qwen_moe_inference.cu`:

```bash
sed -n '1,1369p' libs/lsxcommon/src/qwen_moe_generation.cu \
    > libs/lsxcommon/src/qwen_moe_inference.cu
```

Close the namespaces at the end of the truncated file, and delete lines 1370-end
from `qwen_moe_generation.cu` (keeping its namespace close).

- [ ] **Step 5: Register both sources in CMake**

Replace the single-source target with:
```cmake
add_library(lsxcommon_qwen_moe_cu OBJECT
    src/qwen_moe_inference.cu
    src/qwen_moe_generation.cu
)
```
Leave the rest of that target block (properties, compile options, includes,
link libraries) unchanged.

- [ ] **Step 6: Build and verify**

```bash
cmake --build --preset linux-debug -j 32
ctest --preset linux-debug
ctest --preset linux-debug-convert
```
Expected: `91/91` and `23/23`. The Task 1 guard still fails (expected — the
neutral name does not exist until Task 3).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: split the qwen-MoE generation loop out of the kernel TU"
```

---

### Task 3: Rename the kernel layer to the neutral name

Purely mechanical. The guard from Task 1 goes green here.

**Files:**
- Rename: `qwen_moe_inference.{h,cu}` → `lsx_moe_inference.{h,cu}`
- Modify: 51 files referencing `QwenMoe*` / `RunQwenMoe*`
- Modify: `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Consumes: the split from Task 2.
- Produces: `lsxcommon::moe` exporting every symbol in the mapping table above.
  `MoETruncateTokensAtEos` replaces `TruncateTokensAtEos` (the old bare name is too
  generic for a neutral header).

- [ ] **Step 1: Apply the symbol renames repo-wide**

Run from the repo root — order matters, longest/most specific first:
```bash
for s in RunQwenMoeRoutedGemmVBatchCore RunQwenMoeRoutedGemmVBatch \
         RunQwenMoeRouterTopK RunQwenMoeRoutedFFN RunQwenMoeAccToFp16 \
         RunQwenMoeAccInitForTest RunQwenMoeGemvGatherTable \
         RunQwenMoeSwigluQuantBatch RunQwenMoeRoutedTcBatchTbl \
         RunQwenMoeRoutedTcBatch RunQwenMoeSharedCombine \
         QwenMoeGemvNohandshake QwenMoeRoutedSlot QwenMoeGvSlotDev \
         QwenMoeExpertResident QwenMoeExpertStore; do
  grep -rlZ "$s" libs tests apps --include=*.h --include=*.cu \
      --include=*.cuh --include=*.cpp 2>/dev/null \
    | xargs -0 -r sed -i "s/\b$s\b/${s/QwenMoe/MoE}/g"
done
# The bare helper needs a family-neutral name of its own.
grep -rlZ "TruncateTokensAtEos" libs tests --include=*.h --include=*.cu \
    --include=*.cpp | xargs -0 -r sed -i 's/\bTruncateTokensAtEos\b/MoETruncateTokensAtEos/g'
```

- [ ] **Step 2: Rename the device kernels and the namespace**

In `libs/lsxcommon/src/qwen_moe_inference.cu` only:
```bash
sed -i -e 's/\bqwen_moe_\([a-z_0-9]*\)_kernel\b/moe_\1_kernel/g' \
       -e 's/\bqwen_\([a-z_0-9]*\)_kernel\b/moe_\1_kernel/g' \
       -e 's/namespace qwen_moe {/namespace moe {/' \
       -e 's|// namespace qwen_moe|// namespace moe|' \
       libs/lsxcommon/src/qwen_moe_inference.cu
sed -i -e 's/namespace qwen_moe {/namespace moe {/' \
       -e 's|// namespace qwen_moe|// namespace moe|' \
       libs/lsxcommon/include/lsxcommon/qwen_moe_inference.h
```

Rewrite the family-specific header comment in the new header to match the
`qwen_reduce.cuh` precedent:
```cpp
// Routed-MoE inference kernels: router top-k, routed expert FFN, batched GEMV
// and the TC-routed path. MODEL-NEUTRAL layer: geometry (hidden size, top-k and
// the expert count) is entirely runtime, so the Gemma, GLM-flash and
// Qwen-MoE-exp families consume it. A family that needs its own generation
// loop owns that loop separately and calls this -- see qwen_moe_generation.h.
```

- [ ] **Step 3: Rename the files and the CMake target**

```bash
git mv libs/lsxcommon/include/lsxcommon/qwen_moe_inference.h \
       libs/lsxcommon/include/lsxcommon/lsx_moe_inference.h
git mv libs/lsxcommon/src/qwen_moe_inference.cu \
       libs/lsxcommon/src/lsx_moe_inference.cu
git rm -q --cached libs/lsxcommon/src/qwen_moe_inference.cu 2>/dev/null || true
sed -i -e 's|src/qwen_moe_inference.cu|src/lsx_moe_inference.cu|' \
       libs/lsxcommon/CMakeLists.txt
```

Rename the CMake target `lsxcommon_qwen_moe_cu` → `lsxcommon_moe_cu` in
`libs/lsxcommon/CMakeLists.txt` (definition at line 246 and the link list at
line 479) and in `tests/CMakeLists.txt:3986`.

- [ ] **Step 4: Fix the include paths repo-wide**

```bash
grep -rlZ '#include "lsxcommon/qwen_moe_inference.h"' libs tests apps \
    --include=*.h --include=*.cu --include=*.cuh --include=*.cpp 2>/dev/null \
  | xargs -0 -r sed -i 's|lsxcommon/qwen_moe_inference.h|lsxcommon/lsx_moe_inference.h|'
```

Update `qwen_moe_generation.{h,cu}` to include the new neutral header.

- [ ] **Step 5: Verify the layout contract survived**

```bash
grep -n "static_assert(sizeof(lsx_tc::LsxTcBatchSlot)" \
    libs/lsxcommon/src/lsx_moe_inference.cu
```
Expected: one match, now reading `sizeof(MoEGvSlotDev)`. If it is missing, restore
it — the neutral TC layer depends on this ABI.

- [ ] **Step 6: Build; fix any residue until clean**

```bash
cmake --build --preset linux-debug -j 32 2>&1 | grep -E "error:" | head -30
```
Expected: no errors. Any `qwen_moe::` reference left is a missed call site.

- [ ] **Step 7: Run the suites — the guard must now be GREEN**

```bash
ctest --preset linux-debug
ctest --preset linux-debug-convert
```
Expected: `91/91` (including `test_model_independence_guard`) and `23/23`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: rename the routed-MoE kernel layer to the neutral lsx_moe_inference"
```

---

### Task 4: Drop the neutral TU's dependency on the model loader

`lsx_moe_inference.cu` includes `qwen_moe_load.h`, and so does the header today
(line 12 of the header's include list), because `MoEExpertStore` is *defined* in
the loader header. That is the one remaining dependency that makes the neutral
layer reach into a family.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_moe_inference.h` (define the store)
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_moe_load.h` (remove the definition)
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_moe_exp_load.h` (`ExpExpertStore`)
- Modify: `libs/lsxcommon/src/lsx_moe_inference.cu` (drop the include)

**Interfaces:**
- Produces: `lsxcommon::moe::MoEExpertStore` defined in `lsx_moe_inference.h`.
  `qwen_moe_load.h` keeps `EnsureQwenMoeExpertResident` (loader-owned) and
  includes the neutral header.

- [ ] **Step 1: Move the struct definition**

Cut the `struct MoEExpertStore { ... };` body (currently `qwen_moe_load.h:74`)
and paste it into `lsx_moe_inference.h` under `namespace lsxcommon { namespace moe {`,
replacing the forward declaration currently at line 41. Keep it
`alignas`/member-for-member identical — it is a device-visible layout.

- [ ] **Step 2: Update the loader header**

In `qwen_moe_load.h`: remove the struct body, keep
`int EnsureQwenMoeExpertResident(MoEExpertStore &store, int layer, int eid, ...)`
as-is, and add `#include "lsxcommon/lsx_moe_inference.h"`.

- [ ] **Step 3: Update the derived store**

In `qwen_moe_exp_load.h:149`, the derivation becomes:
```cpp
struct ExpExpertStore : public moe::MoEExpertStore {
```

- [ ] **Step 4: Drop the loader include from the neutral layer**

Remove `#include "lsxcommon/qwen_moe_load.h"` from **both** `lsx_moe_inference.cu`
and `lsx_moe_inference.h` (the header includes it today at line 12 of its include
list). The Task 1 guard already covers both files — the header's include line
contains the substring `qwen`, so the family-name scan fails on it too; the
explicit `#include` check at the end of that guard is belt-and-braces.

Re-add any include the kernel TU actually still needs (`qwen_moe_generation.h` if
it calls the generation entry point — otherwise it must not).

- [ ] **Step 5: Build and verify**

```bash
cmake --build --preset linux-debug -j 32 2>&1 | grep -E "error:" | head -30
ctest --preset linux-debug
ctest --preset linux-debug-convert
```
Expected: `91/91`, `23/23`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: own MoEExpertStore in the neutral layer, not the qwen loader"
```

---

### Task 5: Verify real weights for all four affected families, then document

The rename touches the real-weight path of qwen, gemma, glm-flash and
qwen-moe-exp. The fast suite does not cover them; only the SelfCheck gates do.

**Files:**
- Modify: `AGENTS.md` (the qwen/gemma/glm entries that name the file)
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md`

- [ ] **Step 1: Run the real-weight SelfCheck gates**

One GPU workload at a time (AGENTS.md resource policy):
```bash
ctest --test-dir build/linux-debug -L inference -R "selfcheck" 2>&1 | tail -20
```
Expected: every present real-model SelfCheck passes. SKIP (return code 4) is
acceptable only where the converted checkpoint is absent. Record which ran.

- [ ] **Step 2: Confirm the build is warning-free for the renamed files**

```bash
touch libs/lsxcommon/src/lsx_moe_inference.cu
cmake --build --preset linux-debug -j 16 --target lsxcommon 2>&1 \
  | grep -iE "warning:.*lsx_moe_inference" | head
```
Expected: no output.

- [ ] **Step 3: Update `AGENTS.md`**

Replace every `qwen_moe_inference.{h,cu}` reference with `lsx_moe_inference.{h,cu}`
and add to the neutral-layer sentence in the `lsxcommon` bullet:
`lsx_moe_inference.{h,cu}` (routed-MoE kernels, model-neutral: Gemma MoE,
GLM MoE Flash and Qwen MoE Exp consume it; the qwen generation loop stays in
`qwen_moe_generation.{h,cu}`).

- [ ] **Step 4: Update the repo-map skill**

In `.opencode/skills/logestix-repo-map/SKILL.md`, replace the qwen-MoE kernel
entries with the two new files and their responsibilities.

- [ ] **Step 5: Final full verification and commit**

```bash
cmake --build --preset linux-debug -j 32
ctest --preset linux-debug
ctest --preset linux-debug-convert
git add -A
git commit -m "docs: point the routed-MoE layer at its neutral name"
```

---

## Out of scope

De-duplicating `ornith_moe_inference.cu` (a 2228-line, 97%-duplicate copy of the
kernel layer) is the *purpose* of this rename, but is a separate plan. It must be
its own change so a throughput regression there cannot be confused with a
regression from the rename. Only after this plan lands and is verified.

## Self-review

- **Spec coverage:** rename (Task 3), guard (Task 1), the split that makes the
  rename honest (Task 2), the last family dependency (Task 4), real-weight
  verification and docs (Task 5). All covered.
- **Placeholder scan:** no TBD/TODO; every step names files, commands and
  expected output. Task 2 Step 1 says "copied verbatim from ... lines 208-235" —
  that is an explicit line-range instruction, not a placeholder, because the
  declarations move unchanged.
- **Type consistency:** `MoEExpertStore` is introduced in the mapping table
  (Task 3, used by GLM) and *defined* in Task 4; Task 3 keeps it as the existing
  forward declaration, so the build stays green in between. `MoEGvSlotDev`
  appears in Task 3's rename and in Task 3 Step 5's `static_assert` check —
  same name.