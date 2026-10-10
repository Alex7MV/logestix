# Neutral attention kernel layer (`qwen_inference.*`) — Phase 5 of the CUDA unification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Subagents are FORBIDDEN by AGENTS.md — execute
> inline in the main session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the attention / dense-MLP kernel layer the same treatment the
routed-MoE layer just got (`docs/superpowers/plans/2026-10-10-neutral-moe-inference-layer.md`):
move `qwen_inference.{h,cu}` to a neutral name so the four families that already
consume it (Qwen MoE, Qwen MoE Exp, Ornith, Alice) are no longer calling a file
named after one of them.

**Architecture:** The layer is *already* family-neutral in content — every
dimension is a runtime parameter. What is left is naming plus two type leaks:
`QwenLayerWeights` (a generic weight bundle with no qwen semantics) and
`QwenTopology` (the model passport, used for exactly 12 scalar fields). This
plan renames the bundle, replaces the passport parameter with a small neutral
geometry POD, splits the qwen generation driver out, then renames the file and
namespace — with a guard test pinning the result.

**Tech Stack:** C++20, CUDA 13.4 (sm_120/sm_121 only), CMake 3.25+, Catch2 v3.16.0.

## Global Constraints

- C++20 / CUDA 13.4; architectures `120-real;121-real` ONLY.
- Logging only via spdlog. No `printf` / `std::cout`.
- No new comments unless they carry information; preserve existing Doxygen.
- **No subagents** (AGENTS.md). All work inline.
- Do NOT touch `docs/superpowers/specs/*` or other archived `plans/*`.
- Commit prefixes: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`. English.
- `qwen_moe`, `qwen_moe_exp`, `ornith` and `alice` share this layer; every task
  ends with their real-weight SelfCheck green.

---

## Findings that shape the plan

Measured on the current tree, not assumed:

| Fact | Consequence |
|---|---|
| `QwenNvfp4Weight` is already `= lsx_nvfp4::LsxNvfp4Weight` (`qwen_inference_load.h:22`) | It is an alias, exactly like `qwen_reduce.cuh` → `lsx_reduce.cuh`. Drop the alias and use the neutral type. |
| `QwenLayerWeights` (`qwen_inference_load.h:25-76`) holds only `QwenNvfp4Weight` members, raw `__half*`/`float*`, sizes and an arena pointer — **no qwen semantics** | A pure rename to a neutral `LayerWeights` is sufficient; no field changes. |
| The kernels use exactly **12** `top.*` fields: `rms_norm_eps`(5) `linear_value_head_dim`(5) `linear_num_value_heads`(5) `linear_num_key_heads`(5) `linear_key_head_dim`(5) `num_key_value_heads`(4) `head_dim`(4) `partial_rotary_factor`(2) `num_attention_heads`(2) `hidden_size`(2) `vocab_size`(1) `rope_theta`(1) `intermediate_size`(1) | No vectors, strings, `VramProfile` or `layer_types` reach the kernel. A small neutral POD replaces the passport. |
| Exactly **4** entry points take the passport: `RunQwenFullAttention`, `RunQwenLinearAttention`, `RunQwenFusedLinearAttnKernel`, `RunQwenFusedMLP` | The signature change is four edits. |
| ~216 `qwen::` references across ~20 files | Mechanical, but needs a scripted pass and a compile loop. |
| `tests/test_model_independence_guard.cpp` hardcodes the path `qwen_inference.cu` in **two** places (the fast-math guard at :264 and the fast-math-scope scan at :509) | Both must be repointed when the file is renamed; missing the second is an easy miss. |
| The dense families (gemma/granite/phi/qwen_dense) never call these kernels | Blast radius stays at the four MoE-derived families. |

### Deliberately out of scope (Phase 5b)

`QwenKvCache` and `QwenGenerationResult` stay qwen-named in this plan, and the
guard allows those two names explicitly. Rationale: Ornith and Qwen MoE Exp
both hold a `QwenKvCache`, and the independence guard does **not** forbid them
from including `qwen_inference.*` — AGENTS.md designates it the shared layer.
Renaming them would force ornith to depend on a qwen-owned header, i.e. make
things worse, for a purely cosmetic gain. Neutralizing them later means
changing `QwenKvCache::Allocate(const QwenPinnedWeights&)` to take scalars,
which is its own change.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `libs/lsxcommon/include/lsxcommon/lsx_inference_load.h` | neutral `LayerWeights` + `AttnGeometry` | Create |
| `libs/lsxcommon/include/lsxcommon/lsx_attn_inference.h` | neutral attention/MLP kernel API (keeps the shared fast-math aliases) | Create (from `qwen_inference.h`) |
| `libs/lsxcommon/src/lsx_attn_inference.cu` | the kernels | Create (from `qwen_inference.cu`) |
| `libs/lsxcommon/include/lsxcommon/qwen_generation.h` | qwen-owned generation driver + `QwenKvCache` + `QwenGenerationResult` | Create |
| `libs/lsxcommon/src/qwen_generation.cu` | `RunQwenGeneration` + KV-cache methods | Create |
| `libs/lsxcommon/include/lsxcommon/qwen_inference.h` | — | Delete |
| `libs/lsxcommon/src/qwen_inference.cu` | — | Delete |
| `libs/lsxcommon/include/lsxcommon/qwen_inference_load.h` | keeps `QwenPinnedWeights`, loses `QwenLayerWeights` | Modify |
| `tests/test_model_independence_guard.cpp` | new guard + repoint the fast-math path | Modify |

### Symbol mapping (mechanical, Task 5)

| Old (`lsxcommon::qwen`) | New (`lsxcommon::attn`) |
|---|---|
| `QwenLayerWeights` | `LayerWeights` |
| `RunQwenFullAttention` / `RunQwenLinearAttention` / `RunQwenFusedLinearAttnKernel` / `RunQwenFusedMLP` | `RunAttnFullAttention` / `RunAttnLinearAttention` / `RunAttnFusedLinearAttnKernel` / `RunAttnFusedMLP` |
| `RunQwenFullAttnKernel`, `RunQwenFullAttnKernelSplitKV`, `RunQwenFusedLinearAttnKernel` (device) | `RunAttn*` |
| `RunQwenRMSNorm`, `RunQwenRopeCosSin`, `RunQwenPartialRope`, `RunQwenEmbed` | `RunAttn*` (or keep — see Task 5) |
| `RunQwenNvfp4Gemm`, `RunQwenNvfp4GemmSeq1GemmV`, `RunQwenSwigluQuant`, `RunQwenSwigluPreciseQuantInternal` | keep names (they are GEMM/act, not attention) |
| `QwenTcFastMath`, `SetQwenTcFastMath`, `QwenSeq1Gemv`, `SetQwenSeq1Gemv`, `QwenGemmLarge`, `QwenGemmLargeActive` | **keep names and namespace-qualified** — they are thin aliases over `lsx_nvfp4::` and are the family-facing fast-math API used by 27+ call sites. |

**Keep** `QwenKvCache`, `QwenGenerationResult`, `QwenTopology`,
`QwenPinnedWeights`, `RunQwenGeneration`, `DecideSession`, `SessionDecision`,
`TruncateTokensAtEos` — they move to `qwen_generation.h` / stay in the qwen
loader (Phase 5b).

---

### Task 1: Write the neutrality guard (red before the change)

**Files:**
- Modify: `tests/test_model_independence_guard.cpp` (append a `TEST_CASE`)
- Test: `test_model_independence_guard`

**Interfaces:**
- Consumes: existing `kSrcDir`, `kIncDir`, `ReadFile`, `LSX_SOURCE_DIR`.
- Produces: a new guard case inside the existing CTest target.

- [ ] **Step 1: Append the guard**

```cpp
TEST_CASE("the attention kernel layer is model-neutral in name and deps",
          "[independence]")
{
    // Geometry-neutral in content (every dimension is a runtime parameter) and
    // consumed by Qwen MoE, Qwen MoE Exp, Ornith and Alice, so it carries the
    // neutral lsx_ name. Scoped the same way as the routed-MoE guard: no model
    // dependency, no family-named symbol of its own. The shared fast-math
    // aliases and the Qwen KV cache are explicitly allowed -- see the plan's
    // Phase 5b note.
    const fs::path hdr = kIncDir / "lsx_attn_inference.h";
    const fs::path cu = kSrcDir / "lsx_attn_inference.cu";
    REQUIRE(fs::exists(hdr));
    REQUIRE(fs::exists(cu));

    for (const fs::path &p : {hdr, cu}) {
        const std::string text = ReadFile(p);
        REQUIRE_FALSE(text.empty());

        // Only the shared fast-math API and the KV cache may keep a qwen name.
        static const char *kAllowed[] = {
            "QwenTcFastMath",   "SetQwenTcFastMath", "QwenSeq1Gemv",
            "SetQwenSeq1Gemv",  "QwenGemmLarge",     "QwenGemmLargeActive",
            "QwenKvCache",      "QwenGenerationResult"};
        std::string stripped = text;
        for (const char *a : kAllowed) {
            const size_t n = std::strlen(a);
            for (size_t at = stripped.find(a); at != std::string::npos;
                 at = stripped.find(a, at + n))
                stripped.replace(at, n, std::string(n, '_'));
        }
        INFO(p.filename() << " must not define a family-named symbol");
        REQUIRE(stripped.find("Qwen") == std::string::npos);

        // No model-owned header may be pulled in.
        static const char *kModelOwned[] = {
            "qwen_dense_convert.h", "qwen_inference_load.h",
            "qwen_generation.h",    "qwen_moe_load.h"};
        for (const char *h : kModelOwned) {
            INFO(p.filename() << " must not include the model header " << h);
            REQUIRE(text.find(h) == std::string::npos);
        }
    }
}
```

- [ ] **Step 2: Build and confirm it fails for the right reason**

```bash
cmake --build --preset linux-debug -j 32 --target test_model_independence_guard
ctest --test-dir build/linux-debug -R test_model_independence_guard --output-on-failure
```
Expected: FAIL on `REQUIRE(fs::exists(hdr))` — `lsx_attn_inference.h` does not exist.

- [ ] **Step 3: Commit the red guard**

```bash
git add tests/test_model_independence_guard.cpp
git commit -m "test: add the red neutrality guard for the attention kernel layer"
```

---

### Task 2: Split the qwen generation driver out

`RunQwenGeneration`, `QwenKvCache` and `QwenGenerationResult` take
`QwenPinnedWeights`, so they cannot live in a neutral header. Split them first,
exactly as Task 2 of the MoE plan did for `RunQwenMoeGeneration`.

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/qwen_generation.h`
- Create: `libs/lsxcommon/src/qwen_generation.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h`, `.../qwen_inference.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Produces `namespace lsxcommon::qwen`: `QwenGenerationResult`, `QwenKvCache`
  (with `Allocate`/`EnsureCapacity`/`Free`), `RunQwenGeneration`, plus
  `SessionDecision`/`DecideSession` if they are only used there.

- [ ] **Step 1: Find the split line**

```bash
grep -n "^struct QwenKvCache\|^cudaError_t RunQwenGeneration\|^struct QwenGenerationResult\|^struct QwenTopology" \
    libs/lsxcommon/src/qwen_inference.cu
```
Expect `QwenKvCache` and `RunQwenGeneration` to be the trailing definitions.

- [ ] **Step 2: Create `qwen_generation.h`** with the moved declarations and the
includes they need (`qwen_inference_load.h` for `QwenPinnedWeights`,
`qwen_inference.h` for `QwenKvCache`'s device-side helper types,
`sampling.h`, `token_stream.h`).

- [ ] **Step 3: Split the `.cu`** — `git mv qwen_inference.cu qwen_generation.cu`,
then `sed -n '1,<kvline-1>p'` back into `qwen_inference.cu`, closing the
namespaces in both. Expect the same TU-local-symbol fallout the MoE split hit:
the generation loop may reach into the kernel TU's anonymous namespace
(`qwen_record_token_kernel` / `qwen_feed_token_kernel` are the known
candidates). Move those into the generation TU — their only caller — rather
than exporting them.

- [ ] **Step 4: Register both sources** in the `lsxcommon_qwen_cu` OBJECT target
in `libs/lsxcommon/CMakeLists.txt`, leaving the rest of that block unchanged.

- [ ] **Step 5: Build and verify**

```bash
cmake --build --preset linux-debug -j 32
ctest --preset linux-debug
ctest --preset linux-debug-convert
```
Expected: 91/91 and 23/23 (the Task 1 guard still fails by design).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: split the qwen generation driver out of the kernel TU"
```

---

### Task 3: Neutralize the weight bundle

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/lsx_inference_load.h`
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference_load.h`

**Interfaces:**
- Produces: `lsxcommon::attn::LayerWeights` — a byte-for-byte copy of
  `QwenLayerWeights`' members, with `QwenNvfp4Weight` replaced by its neutral
  alias `lsx_nvfp4::LsxNvfp4Weight`.

- [ ] **Step 1: Create the neutral header** with the struct renamed and the
weight alias dropped. Copy the member comments verbatim.

- [ ] **Step 2: Remove it from `qwen_inference_load.h`**, leaving
`QwenPinnedWeights`, `LoadQwenPinnedWeights` and `ReadFileBytes`. Have
`qwen_inference_load.h` include the neutral header.

- [ ] **Step 3: Repoint users repo-wide**

```bash
grep -rlZ "QwenLayerWeights" libs tests apps --include=*.h --include=*.cu --include=*.cpp \
  | xargs -0 -r sed -i 's/\bQwenLayerWeights\b/attn::LayerWeights/g'
```
Add `namespace attn = lsxcommon::attn;` (or the full qualifier) where the file
is not already inside `lsxcommon`.

- [ ] **Step 4: Build; run fast + convert**

Expected: 91/91, 23/23. This task is a pure rename — if anything changes
numerically, stop and investigate.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: give the weight bundle a neutral name (attn::LayerWeights)"
```

---

### Task 4: Replace the passport parameter with a neutral geometry POD

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_inference_load.h` (add `AttnGeometry`)
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h`, `.../qwen_inference.cu`
- Modify: the four callers

**Interfaces:**
- Produces:

```cpp
namespace lsxcommon { namespace attn {
// The twelve geometry scalars the attention/MLP kernels read out of the model
// passport. The passport itself (QwenTopology) carries layer_types, vram and
// error text that no kernel touches, so it is deliberately not passed down.
struct AttnGeometry {
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta = 1e7;
};
}} // namespace lsxcommon::attn
```

- [ ] **Step 1: Add the POD** to `lsx_inference_load.h`.

- [ ] **Step 2: Change the four signatures** in the header, replacing
`const QwenTopology &top` with `const AttnGeometry &geo`.

- [ ] **Step 3: Rewrite the twelve `top.` reads** in the `.cu` as `geo.`
(`top.rms_norm_eps` → `geo.rms_norm_eps`, etc.). Do not restructure anything
else — this task is mechanical and must produce a bit-identical kernel.

- [ ] **Step 4: Update the four call sites** to build a geometry from the
passport at the call boundary, e.g. in the model/pipeline `.cu`:

```cpp
const attn::AttnGeometry geo{
    .hidden_size = top.hidden_size,
    .intermediate_size = top.intermediate_size,
    .vocab_size = top.vocab_size,
    .num_attention_heads = top.num_attention_heads,
    .num_key_value_heads = top.num_key_value_heads,
    .head_dim = top.head_dim,
    .linear_num_key_heads = top.linear_num_key_heads,
    .linear_num_value_heads = top.linear_num_value_heads,
    .linear_key_head_dim = top.linear_key_head_dim,
    .linear_value_head_dim = top.linear_value_head_dim,
    .rms_norm_eps = top.rms_norm_eps,
    .partial_rotary_factor = top.partial_rotary_factor,
    .rope_theta = top.rope_theta};
```

- [ ] **Step 5: Build; run fast + convert, then the four real-weight SelfChecks**

```bash
cmake --build --preset linux-debug -j 32
ctest --preset linux-debug
ctest --preset linux-debug-convert
ctest --test-dir build/linux-debug -R "^(test_ornith_moe_selfcheck|test_qwen_moe_36_selfcheck|test_gemma_moe_selfcheck|test_ornith_moe_det_selfcheck)$"
```
Expected: all green. A numeric change here is a bug — this task must be
bit-identical.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: pass a neutral geometry POD instead of the model passport"
```

---

### Task 5: Rename the layer and repoint the fast-math guard

**Files:**
- Rename: `qwen_inference.{h,cu}` → `lsx_attn_inference.{h,cu}`; namespace `qwen` → `attn`
- Modify: ~20 files with `qwen::` references, both guard tests, `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Produces `lsxcommon::attn` (was `lsxcommon::qwen`) holding the kernel API and
the shared fast-math aliases.

- [ ] **Step 1: Apply the symbol renames** (Task 5 table, longest first), then the
kernel/device symbol renames inside the two files only.

- [ ] **Step 2: Rename the files and namespace**, and fix `#include` paths:

```bash
git mv libs/lsxcommon/include/lsxcommon/qwen_inference.h libs/lsxcommon/include/lsxcommon/lsx_attn_inference.h
git mv libs/lsxcommon/src/qwen_inference.cu libs/lsxcommon/src/lsx_attn_inference.cu
grep -rlZ '"lsxcommon/qwen_inference.h"' libs tests apps --include=*.h --include=*.cu --include=*.cpp \
  | xargs -0 -r sed -i 's|"lsxcommon/qwen_inference.h"|"lsxcommon/lsx_attn_inference.h"|'
```

- [ ] **Step 3: Repoint the fast-math guard**, which hardcodes the old path:

```bash
grep -n 'qwen_inference.cu' tests/test_model_independence_guard.cpp
```
Expected: **two** hits (:264 fast-math guard, :509 fast-math-scope scan).
Replace both with `lsx_attn_inference.cu`, renaming the local variables
(`qwen_inf`) accordingly. Then re-run the grep to prove zero remain:

```bash
grep -rn 'qwen_inference' tests/ | grep -v qwen_moe_inference
```
Also check `test_model_suite_guard.cpp` and `test_code_naming_guard.cpp` for
the old stems.

- [ ] **Step 4: Update CMake** — the `lsxcommon_qwen_cu` OBJECT target's source
list and any target name that reads as family-specific.

- [ ] **Step 5: Build until clean; then run everything**

```bash
cmake --build --preset linux-debug -j 32 2>&1 | grep -E "error:" | head -30
ctest --preset linux-debug
ctest --preset linux-debug-convert
```
Expected: 91/91, 23/23 — including the Task 1 guard, now green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename the attention kernel layer to the neutral lsx_attn_inference"
```

---

### Task 6: Real-weight verification, docs, archive

**Files:**
- Modify: `AGENTS.md`, `.opencode/skills/logestix-repo-map/SKILL.md`
- Create: `docs/open-tasks/<date>-<slug>.md` if any item closes
- Modify: `docs/OPEN_TASKS.md`, `docs/open-tasks/README.md`

- [ ] **Step 1: Real-weight gates for all four consuming families** — one GPU
workload at a time:

```bash
ctest --test-dir build/linux-debug -R "^(test_ornith_moe_selfcheck|test_qwen_moe_36_selfcheck|test_gemma_moe_selfcheck|test_ornith_moe_det_selfcheck|test_qwen_dense_selfcheck|test_qwen_dense_27b_selfcheck)$"
```
Record which ran and which SKIPped (an absent converted checkpoint is a skip,
not a pass — do not report it as coverage).

- [ ] **Step 2: Measure whether perf moved** with `lsxbenchmark --bench --max-gen
2048` on one converted MoE model, before/after. A rename must not change
throughput; report the numbers either way.

- [ ] **Step 3: Update `AGENTS.md`** — add the layer to the model-neutral list
next to `lsx_moe_inference`, and name `qwen_generation.{h,cu}` as the qwen-owned
counterpart.

- [ ] **Step 4: Update the repo-map skill** with the two new file pairs.

- [ ] **Step 5: Update `docs/OPEN_TASKS.md` section S** to record that the kernel
layer is now neutral and that only Phase 5b (`QwenKvCache`,
`QwenGenerationResult`, `QwenKvCache::Allocate` still taking
`QwenPinnedWeights`) remains. If the work session closes items, follow the
maintenance rule: move them to `docs/open-tasks/<date>-<slug>.md`, add an
`Archive` row, update the index.

- [ ] **Step 6: Final verification and commit**

```bash
cmake --build --preset linux-debug -j 32
ctest --preset linux-debug
ctest --preset linux-debug-convert
git add -A
git commit -m "docs: point the attention kernel layer at its neutral name"
```

---

## Self-review

- **Coverage:** guard (1), split (2), bundle rename (3), passport removal (4),
  file/namespace rename (5), verification/docs/archive (6). Each ends
  independently testable.
- **Placeholders:** none. Task 2 Step 1 tells the implementer to *find* the split
  line rather than asserting one, because that line has not been measured yet —
  and Task 2 Step 3 names the known TU-local fallout instead of pretending the
  split is a pure move (it was not, in the MoE case).
- **Type consistency:** `LayerWeights` is introduced in Task 3 and used by the
  `AttnGeometry` signatures in Task 4; `AttnGeometry` is defined in Task 4 and
  used by the four call sites in the same task. The guard's `kAllowed` list
  matches exactly the symbols the rename deliberately keeps.
- **Correction applied after verification:** the guard's hardcoded path
  appears twice, not once; the plan originally said one.
- **Known risk:** Tasks 3 and 4 must be bit-identical. If a real-weight gate
  moves, treat it as a bug in that task, not as an acceptable numeric change —
  the whole point of this refactor is to change nothing but names.