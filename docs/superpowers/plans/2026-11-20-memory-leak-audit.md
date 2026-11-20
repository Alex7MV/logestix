# Memory-Leak Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (subagents are forbidden by AGENTS.md — native execution only). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and fix real memory/resource leaks in logestix's raw-ownership surfaces, every fix landing red-test-first, recorded as stable-ID findings.

**Architecture:** Targeted static audit of allocation sites in priority order (expert caches/loaders → inference error/cancel paths → pinned-host parity → network/apps). Each suspicion is confirmed or dismissed by a test in the repo's existing leak-probe patterns before any fix. Findings live in `docs/open-tasks/` per repo convention.

**Tech Stack:** C++20/CUDA (Blackwell), Catch2 v3, CMake presets (`linux-release-znver4` primary), ctest via `cmake/LsxTestHelpers.cmake`.

**Spec:** `docs/superpowers/specs/2026-11-20-memory-leak-audit-design.md`

## Global Constraints

- Branch `audit/memory-leaks` off `main`; all work on it.
- **Commits only on explicit operator request** (AGENTS.md). Task-end "checkpoint" steps report to the operator instead of committing; if the operator approves commits mid-run, use prefixes `fix:`/`test:`/`docs:`, English messages.
- One GPU workload at a time: before any GPU-labelled ctest run, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` must show no other compute process.
- New tests register via `lsx_add_test`/`lsx_add_cuda_test`; GPU-touching tests get a device label + `RESOURCE_LOCK "gpu"` (enforced by `test_ctest_labels_guard`); host-only probes use `fast`. Skip-capable tests set `SKIP` (return code 4).
- No subagents anywhere (AGENTS.md).
- No numeric-contract, golden, or perf-baseline changes. spdlog-only logging; `absl::StrFormat` is printf-style; no `<absl/container/*>` in headers reachable from `.cu`.
- Skills to load at execution: `executing-plans`, `test-driven-development`, `test-lifecycle`, `building-the-project` (build commands), `systematic-debugging` (any unexpected red).
- A suspicion proven intentional (static-lifetime cache/pool) is **documented, not fixed**.

## Review Focus

Failure modes the spec implies that ordinary green-path tests never exercise (each is pinned in the owning task):

1. **Allocation failure mid-sequence** — the Nth alloc fails after N−1 succeeded; every earlier acquisition must be freed (Task 2, 3).
2. **Cancel/abort mid-generation** — a cancelled request must leave VRAM and host state identical to before it (Task 3).
3. **Repeat load/free in one process** — second load must not OOM or ratchet RSS/VRAM; `Free()` twice must be a no-op (Task 2).
4. **Partial workspace teardown** — `Free*` called on a half-initialized struct must free exactly what was acquired, no double-free (Task 3).
5. **Shutdown with live sockets/threads** — app/network teardown joins threads and closes zmq/httplib handles; no handle survives `Stop()` (Task 5).

---

### Task 1: Branch + green baseline

**Files:**
- Create: branch `audit/memory-leaks`
- Create: `docs/open-tasks/live/engine-defects.md` entry stub (ID `A3`, "memory-leak audit in progress") — only after Task 2 produces the first finding; this task only records the baseline.

**Interfaces:**
- Produces: baseline log path `build/linux-release-znver4/Testing/` ctest output; branch name all later tasks work on.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b audit/memory-leaks
```

- [ ] **Step 2: Build the primary preset**

Use skill `building-the-project` for the exact configure/build command for `linux-release-znver4`. Expected: build succeeds, no new warnings.

- [ ] **Step 3: Verify GPU is idle, run the fast suite as baseline**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ctest --test-dir build/linux-release-znver4 -L fast --output-on-failure
```

Expected: all fast tests pass (or SKIP with code 4). Record any pre-existing failure verbatim in the audit notes — it is not this audit's to fix, and later reds must be distinguishable from it.

- [ ] **Step 4: Checkpoint** — report baseline result to operator; continue to Task 2.

---

### Task 2: Stage 1 — expert caches & loaders

**Files:**
- Audit (read): `libs/lsxcommon/src/{mimo,nemotron,nemotron_ultra,gigachat}_moe_expert_cache.cpp`, `libs/lsxcommon/src/glm_moe_flash_load.cpp`, `libs/lsxcommon/src/kimi_moe_load.cpp`, `libs/lsxcommon/src/qwen_moe_exp_load.cpp`, `libs/lsxcommon/src/{minimax,inkling}_moe_load.cpp`, `libs/lsxcommon/src/deepseek_moe_load.{cu,cuh}`, `libs/lsxcommon/src/deepseek_moe_pro_load.{cu,cuh}`, `libs/lsxcommon/src/lsx_moe_inference.cu` (cache section ~2030–2140)
- Modify: whichever of the above a confirmed finding lives in
- Test: extend existing per-family loader tests (`tests/test_*_moe_loader.cpp`) where they exist; new probe file `tests/test_leak_audit_loaders.cu` only for families with no loader test

**Interfaces:**
- Consumes: existing per-family `Allocate*ExpertCache(cache, slots, error)` / `Free*ExpertCache(cache)` pairs (e.g. `AllocateMimoExpertCache(MimoExpertCache&, int, std::string&)`, `FreeMimoExpertCache(MimoExpertCache&)` — all four caches share this shape).
- Produces: findings list `A3.x` (site, verdict leak/intentional, test name) appended to `docs/open-tasks/live/engine-defects.md`; regression tests named `test_leak_audit_*` or new sections in existing loader tests.

**Known suspicions to verify first (from recon):**

- **S2.1** `glm_moe_flash_load.cpp:868` — per-slot `cudaMallocHost` slab; on failure at slot k the function returns false. Verify whether the caller frees `store.h_slabs[0..k-1]` on the false path (find the store's `Free()` and the caller's error handling). If not freed → leak.
- **S2.2** `kimi_moe_load.cpp:265` — `h_gate_bias` pinned alloc inside a per-layer loop with one `cudaFreeHost` at :279; verify every early-return between alloc and free.
- **S2.3** `qwen_moe_exp_load.cpp:995` — arena `cudaMallocHost` with `cudaFreeHost` only at :856 (error path?); verify the success path hands arena ownership to a teardown that frees it, and every failure return after :995 frees it.
- **S2.4** Four expert caches: six raw `new int32_t[]` after a successful `cudaMalloc`. Plain `new` throws on OOM; if allocation k throws, allocations 1..k−1 and `d_base` leak. Verdict likely "theoretical, host OOM only" — decide test-vs-document per spec; prefer converting to the same RAII/nothrow pattern only if a test can make it red.

- [ ] **Step 1: Static pass** — read each audited file's alloc/free pairs and every exit path between them; write the suspicion list (S2.1…S2.n) with file:line and verdict-candidate into working notes.

- [ ] **Step 2: For each confirmed suspicion, write the failing test**

Pattern per finding kind:
- *Failure-path leak* (S2.1–S2.3): drive the real failure (e.g. request an allocation larger than free VRAM/RAM so alloc k fails after earlier ones succeed), then assert `cudaMemGetInfo` free bytes (device) or repeat-cycle RSS (host) returns to pre-call level after the failed call + `Free()`. Follow the probe style of `tests/test_alice_moe_model.cpp:278-307` (warm-up cycle, then delta assertion; tolerance sized to the leaked object, per G9: assert the leak, not the delta).
- *Repeat-cycle leak*: N=3 load→free cycles in one process, free-bytes delta after warm-up < one leaked-object size.
- *Idempotent Free*: call `Free*` twice; second must not crash/double-free (existing pattern in `tests/test_flint_moe_loader.cpp:132`).

- [ ] **Step 3: Run each new test, verify it fails for the predicted reason**

```bash
ctest --test-dir build/linux-release-znver4 -R <test_name> --output-on-failure
```

Expected: FAIL with the free-bytes/RSS delta ≈ the leaked allocation's size. A test that fails for another reason → skill `systematic-debugging` before touching anything. A suspicion whose test stays green is dismissed and documented as intentional/clean.

- [ ] **Step 4: Minimal fix** — free the leaked acquisition on the failing path (or hand it to the existing teardown). No restructuring beyond the fix.

- [ ] **Step 5: Re-run the test, verify PASS; re-run the family's existing loader/kernels tests, verify still green**

- [ ] **Step 6: Record findings** — append `A3.x` rows (site, leak/intentional, evidence: red delta → green) to `docs/open-tasks/live/engine-defects.md`; register any new test file per `test-lifecycle` (label, timeout, SKIP, gpu lock) and run `ctest -R "test_ctest_labels_guard|test_manifest_sync|test_logging_guard|test_absl" -L fast` — all green.

- [ ] **Step 7: Checkpoint** — report Stage 1 findings/fixes to operator; wait for go before Task 3.

---

### Task 3: Stage 2 — inference error/cancel paths

**Files:**
- Audit (read): `libs/lsxcommon/src/lsx_moe_inference.cu` (manual `cudaMallocAsync`/`cudaFreeAsync` blocks ~460–800), `libs/lsxcommon/src/hy_moe_inference_pipeline.cu`, `libs/lsxcommon/src/gemma_moe_inference.cu` (workspace ensure ~1450–1531 + its Free), `libs/lsxcommon/src/{xing,mistral,kimi,qwen,ornith}_moe_inference*.cu`, `libs/lsxcommon/src/qwen_dense_inference_pipeline.cu`
- Modify: confirmed-finding sites only
- Test: new sections in the family's existing `tests/test_*_moe_inference.cpp`, or `tests/test_leak_audit_inference.cu` for uncovered paths

**Interfaces:**
- Consumes: per-family workspace structs + `Free*`/teardown functions found in Step 1; cancel entry points (e.g. the abort path exercised by `tests/test_kimi_moe_inference.cpp:82`).
- Produces: findings `A3.x` continued; regression tests for Review Focus items 1, 2, 4.

**Known suspicion to verify first:**

- **S3.1** `gemma_moe_inference.cu:1521-1523` — three `cudaMallocHost` in a short-circuit `||` chain: if `h_selw` fails, `h_sel` is already allocated; the error return at :1524 must free partial state. Verify what the caller does with the failed `ws` (does teardown run on `ws_ready == false`?). Same question for the `A(...)` device chain above it.

- [ ] **Step 1: Static pass** — for each manual alloc/free block, list every early return/cancel between first alloc and last free; check each frees exactly the acquired set. Include stream/event create/destroy pairing.

- [ ] **Step 2: Write failing tests for confirmed suspicions**

- *Partial-teardown* (Review Focus 4): construct the workspace struct with only the first k pointers set (as a mid-chain failure leaves it), call the family's `Free*`, assert no crash and free-bytes restored. This tests the teardown contract directly without needing to force a real OOM.
- *Cancel mid-generation* (Review Focus 2): follow `tests/test_kimi_moe_inference.cpp` — start Infer, cancel, then assert free-bytes delta after warm-up < smallest leaked buffer size, and a subsequent normal Infer still succeeds.
- *Repeated generation* (steady state): the alice probe pattern, per family lacking one.

- [ ] **Step 3: Run, verify predicted FAIL** (same ctest command shape as Task 2 Step 3).

- [ ] **Step 4: Minimal fix at the confirmed site.**

- [ ] **Step 5: Re-run new tests → PASS; family inference tests → green.**

- [ ] **Step 6: Record `A3.x` findings + guard tests green** (same command as Task 2 Step 6).

- [ ] **Step 7: Checkpoint** — report Stage 2 to operator; wait for go.

---

### Task 4: Stage 3 — pinned-host parity classification

**Files:**
- Audit (read): the 17 files where `cudaMallocHost|cudaHostAlloc` count ≠ `cudaFreeHost` count (recon list: `glm_moe_flash_load.cpp`, `qwen_moe_inference_pipeline.cu`, `qwen_moe_exp_load.cpp`, `ornith_moe_inference_pipeline.cu`, `glm_moe_load.cu`, `deepseek_moe_load.cuh`, `xing_moe_inference.cu`, `gemma_moe_inference.cu`, `deepseek_moe_pro_load.cu`, `deepseek_moe_load.cu`, `kimi_moe_load.cpp`, `kimi_moe_inference.cu`, `glm_moe_load.cpp`, `deepseek_moe_pro_load.cuh`, `glm_moe_inference.cu`, `mistral_moe_inference.cu`, `qwen_dense_inference_pipeline.cu`)
- Modify/Test: only where classification finds an unowned allocation

**Interfaces:**
- Consumes: Task 2/3 findings (several of these files are already covered — do not re-test what a green probe already pins).
- Produces: a parity table (file, allocs, frees, owner-of-each, verdict) in the audit notes; `A3.x` rows only for real leaks.

- [ ] **Step 1: For each file, pair every pinned alloc with its free site** (possibly in another file — grep the pointer's struct field across `libs/`). Classify: paired / freed-elsewhere / static-lifetime-intentional / leaked.

- [ ] **Step 2: For each "leaked" verdict, red test → fix → green** (Task 2 Step 2–5 cycle, host-RSS probe where the allocation is host-pinned).

- [ ] **Step 3: Record the parity table + findings; intentional static pools get a one-line reason each** (e.g. process-lifetime scratch, cited by file:line).

- [ ] **Step 4: Checkpoint** — report Stage 3 to operator; wait for go.

---

### Task 5: Stage 4 — network & apps

**Files:**
- Audit (read): `libs/lsxnetwork/src/` (frame_queue, zmq socket/ctx owners, httplib server/client owners), `apps/{lsxfabric,taskmosaic,lsxcli,lsxbenchmark,lsxforge}/src/` (shutdown paths, thread joins, raw `new` if any)
- Modify/Test: confirmed sites only; prefer extending `tests/test_integrated_pipeline.cpp` / lsxnetwork's existing tests

**Interfaces:**
- Consumes: nothing from earlier tasks (independent surface).
- Produces: `A3.x` findings; Review Focus 5 test.

- [ ] **Step 1: Static pass** — ownership map: every `zmq_socket`/`zmq_ctx`, httplib server thread, `std::thread`, and raw heap acquisition → its release site on both normal and error shutdown.

- [ ] **Step 2: For confirmed suspicions, red test** — start/stop cycle ×N in one process (host-only, `fast` label where no GPU is touched); assert RSS steady-state across cycles after warm-up and that stop joins (no detach without ownership). For zmq: repeated bind of the same endpoint after stop must succeed (a leaked socket keeps the endpoint busy).

- [ ] **Step 3: Verify FAIL → minimal fix → PASS → existing network tests green.**

- [ ] **Step 4: Record findings; checkpoint** — report Stage 4 to operator.

---

### Task 6: Close-out — findings archive + full verification

**Files:**
- Modify: `docs/open-tasks/live/engine-defects.md` (final `A3.x` state)
- Create (if all stages closed): `docs/open-tasks/2026-11-XX-memory-leak-audit-closed.md` + row in `docs/open-tasks/README.md`; if items remain open, they stay live and no archive is written
- Modify: `docs/OPEN_TASKS.md` area row (engine-defects state line)

**Interfaces:**
- Consumes: all `A3.x` rows and their red→green evidence from Tasks 2–5.
- Produces: the audit's publishable record.

- [ ] **Step 1: Full fast suite + every touched family's GPU suite** (GPU idle check first; sequential, one ctest process):

```bash
ctest --test-dir build/linux-release-znver4 -L fast --output-on-failure
ctest --test-dir build/linux-release-znver4 -R "<touched families>" --output-on-failure
```

Expected: green (SKIP 4 allowed where pre-existing).

- [ ] **Step 2: Write the archive/index per OPEN_TASKS rules** — measured evidence per finding (red delta, green delta, test name), dismissed suspicions with reasons, parity table summary.

- [ ] **Step 3: Guard sweep** — `ctest -L fast -R "guard|manifest_sync"` green.

- [ ] **Step 4: Final checkpoint** — summary to operator: findings fixed / dismissed / still open, tests added, branch state. Ask whether to commit (and how to split commits); commit only on the explicit yes.