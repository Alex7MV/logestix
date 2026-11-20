# Memory-leak audit (planned hygiene) — design

Date: 2026-11-20 · Status: awaiting approval · Branch: `audit/memory-leaks`

## Intent and why

Planned hygiene audit, not symptom-driven: no observed RSS/VRAM growth
triggered it. The repo has a standing audit culture (inference audit
2026-10-22, bandwidth-probe leak 2026-11-04), both closed test-first.
This audit extends that practice to the raw-ownership surfaces that have
accumulated since: expert caches, loaders, inference error/cancel paths,
network resources. Operator decisions: separate git branch, TDD for all
fixes (red regression test first, then fix).

## Goals

- Find and fix real memory/resource leaks (host heap, pinned host, device
  VRAM, CUDA streams/events, zmq/httplib handles) in the prioritized
  surfaces below.
- Every fix lands with a regression test that fails before the fix and
  passes after, registered per `test-lifecycle` rules.
- Findings recorded as a live entry in `docs/open-tasks/` with stable IDs
  (next free in area A: `A3`), per repo convention; closed work archived to
  `docs/open-tasks/<date>-memory-leak-audit-closed.md` + README index row.

## Non-goals / anti-goals

- No full sweep of all ~165k lines of `libs/lsxcommon` (diminishing returns;
  a green suite is a sample, not a proof — `OPEN_ENDEDNESS.md`).
- No ASan/LSan preset or new sanitizer infrastructure (approach B was
  considered and rejected for now; may be a follow-up if A surfaces host
  leaks beyond visible inspection).
- No behavior/perf changes beyond leak fixes; no numeric-contract changes;
  no touching goldens or perf baselines.
- No commits unless explicitly requested; branch work stays local until
  the operator asks.

## Constraints

- One GPU workload at a time; every GPU-labelled test carries
  `RESOURCE_LOCK "gpu"`; check `nvidia-smi` before any GPU run.
- No subagents (AGENTS.md autonomous-mode rule) — all audit work in the
  main session; the builder check below is done by re-reading the doc as
  a fresh builder.
- Test registration via `lsx_add_test`/`lsx_add_cuda_test`, labels and
  `SKIP_RETURN_CODE 4` per `test-lifecycle`.
- spdlog-only logging, absl printf-style formats, no absl containers in
  headers reachable from `.cu` (existing guards apply to all new code).

## Approaches considered

- **A (chosen): targeted audit of raw-ownership sites + repo-native leak
  tests.** Static pass over ~11 `new[]`/`malloc` files, error/cancel paths
  with manual `cudaFreeAsync`, host-pinned alloc/free parity, streams/
  events, zmq/httplib. Each suspicion verified by a red test in the
  existing patterns (VRAM free-bytes probe across N load→infer→free
  cycles; idempotent `Free()`; repeat-load-without-OOM). Zero new
  infrastructure; covers the dominant risk (device/pinned leaks) directly.
- **B (rejected for now): ASan/LSan preset + host fast suite.** Automatic
  host-heap coverage, but new CMake plumbing (flags must not leak into
  NVCC), full dependency rebuild, CUDA-runtime false positives needing
  suppressions, and LSan cannot see device leaks. Follow-up candidate if
  A finds host leaks beyond inspection.
- **C (rejected): exhaustive full-tree audit.** Multiple sessions of
  exclusive GPU for diminishing returns.

Operator picked A on 2026-11-20.

## Scope and priority order (audit stages)

1. **Expert caches & loaders** (~11 files with raw `new[]`/`malloc`:
   `mimo_moe_expert_cache.cpp`, `nemotron_moe_*_expert_cache.cpp`,
   `gigachat_moe_expert_cache.cpp`, `*_moe_load*.{cpp,cu}`,
   `lsx_moe_inference.cu` cache section). Check: allocation/free pairing
   on every exit path, partial-init failure cleanup, idempotent teardown,
   reload-in-same-process.
2. **Inference error/cancel paths** (`lsx_moe_inference.cu` manual
   `cudaMallocAsync`/`cudaFreeAsync` blocks, `hy_moe_inference_pipeline.cu`,
   per-family pipelines). Check: every early-return/cancel frees every
   allocation acquired so far; stream/event lifecycle.
3. **Pinned-host parity** (74 `cudaMallocHost|cudaHostAlloc` vs 73
   `cudaFreeHost` — locate the imbalance; may be a static-lifetime cache,
   which must be classified as intentional or leak).
4. **Network & apps** (`lsxnetwork` frame_queue/zmq/httplib; apps' RAII
   review, socket/thread joins on shutdown).

Each stage: static read → suspicion list → red test → fix → green test →
guard suite still green. A suspicion that turns out intentional (static
cache, process-lifetime pool) is recorded as such with the reason, not
"fixed".

## Test patterns (existing, reused)

- VRAM probe: free bytes before first run vs after N repeated full
  runs, tolerance sized per `instruments-and-guards.md` G9 guidance
  (assert the leak, not the delta; one cached context growth tolerated).
- Host-heap probe where measurable: repeated construct/destruct or
  load/free cycles with RSS or allocator accounting, host-only `fast`
  label where possible (no GPU lock needed).
- Idempotent `Free()` / second-load-no-OOM tests already exist for several
  loaders — extend to uncovered families rather than duplicating.

## Deliverables

- Branch `audit/memory-leaks` off `main`.
- Regression tests (red→green evidence recorded per finding).
- Fixes, minimal and local.
- Live findings entry `docs/open-tasks/live/engine-defects.md` (IDs from
  `A3`), routed per OPEN_TASKS rules; on completion, archive file +
  `open-tasks/README.md` row with measured evidence.

## Left to the implementer

- Exact test tolerances per probe (guided by G9 lessons).
- Whether a suspected site needs a test at all (pure static-lifetime
  allocations: document, skip).
- Order within a stage; stop/continue checkpoints reported to the
  operator after each stage.

## Builder-check calls (settled by the author, marked as calls)

- **Build**: reuse existing configured presets (`linux-release-znver4`
  for GPU probes, `linux-debug` where a debug build helps); no new
  preset or dependency rebuild.
- **Baseline first**: before any fix, run the existing fast suite and
  the relevant model suites once to record a green baseline, so a later
  red is attributable to the new test, not pre-existing breakage.
- **Host-heap measurement**: prefer allocator-level accounting or
  repeated-cycle RSS with warm-up cycles over absolute RSS assertions
  (allocator arenas make absolute values unreliable); exact mechanism
  chosen per test.