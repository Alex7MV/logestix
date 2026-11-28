---
name: perf-tuning-a-model
description: Use when raising or measuring inference throughput (t/s) of a logestix model — profiling a decode token with nsys, finding the serial bottleneck per stream, writing RED latency/bandwidth/miss-count budget tests, tuning gpu_perf knobs, validating at the canonical 2048-token protocol, or recording a perf_baselines.json entry. Triggers - decode tok/s, t/s, lsxbenchmark --bench, perf baseline, bottleneck, budget test.
---

# Perf-tuning a model

## Overview

Throughput work is a measured loop, not intuition: reproduce the true baseline
→ attribute the wall time to a *stream* → fix the dominant serial term →
re-measure at the canonical protocol. The two rounds that worked here (gemma
+14.6%, Kimi-K2.7-Code +65%) both came from measurement; every guess that
skipped step 2 was discarded.

**REQUIRED SUB-SKILL:** use `test-driven-development`.

Launching and operating the engine (apps, vault paths, engine session, env knobs)
is the `running-a-model` skill; this skill owns profiling and baselines.

## Hard rules

- **`lsxbenchmark` is GPU-exclusive.** One GPU workload at a time (no other
  inference, CUDA test, or perf suite). VRAM/SM contention skews t/s by up to
  2x, which makes the measurement worthless.
- **Never quote a short run as the baseline.** 128 tokens is warmup-affected;
  the canonical record is `--max-gen 2048` with the prompt from the existing
  baseline entry.
- One change per commit, with its failing test, plus the E2E re-measurement.
- **A rejected experiment is a result.** Record the numbers and the cause.

## Step 1 — reproduce the baseline

```bash
ninja -C build/linux-release-znver4-aocc lsxbenchmark
./build/linux-release-znver4-aocc/apps/lsxbenchmark/lsxbenchmark \
    --model <vault> --bench --max-gen 2048 --prompt "<recorded prompt>"
```

- Read `tests/perf/perf_baselines.json` first: `model_type`, `max_gen`,
  `prompt`, `gen_tokens`. Same prompt and `max_gen`, **and the same prompt
  framing** (see below — for a streamed-expert family the frame decides the
  continuation, which decides the number).
- **Streamed-expert families: decode t/s is a function of the generated text.**
  Measured on ONE GLM-5.3-Flash binary at 2048 tokens: `hit rate 46.6% / 13.0
  t/s` on the Paris prompt, `50.3% / 13.5` on a coding prompt, `57.4% / 14.6`
  on "count to 2000". The routed path streams 2.4 GB/token and the copies do
  not overlap compute, so `wall/token = K + misses × slot_bytes / H2D_bw`
  (GLM: `22.5 ms + misses × 0.293 ms`) — an expert-cache hit rate is the whole
  story, and *any* change that alters the continuation (a prompt, a chat
  template, a numeric fix that re-rolls near-tie routing) moves it by 10-30%.
  Before calling a delta a regression, check that the reference and the run
  **generated the same kind of text**: re-run the old binary on a second
  prompt. GLM's "14.89 → 12.79 regression" turned out to be entirely a framing
  change — on a comparable prompt the two binaries were 13.2 vs 13.5 t/s.
- `linux-release-znver4-aocc` is the perf reference build; `linux-debug` is for
  fast tests.
- Loading is per-process and can dominate wall time (Kimi re-pins 544 GiB ≈
  300 s and needs `ulimit -l unlimited`). Budget for it, and run it
  synchronously — a backgrounded run dies with the tool session.

## Step 2 — attribute the wall time (the decisive step)

`ncu` counters are unavailable on this box. Use `nsys`:

```bash
nsys profile -t cuda --stats=false -o /tmp/p ./.../lsxbenchmark ... --max-gen 64
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format table /tmp/p.nsys-rep
```

- `--delay`/`--duration` capture windows silently produce **no report** here:
  capture the whole run and analyse.
- The `sqlite3` CLI is absent — use `python3`'s `sqlite3` module. Kernel names
  are interned (join `StringIds ON s.id = k.demangledName`) and there is **no
  `duration` column** (use `end - start`).
- **Split memcpy time BY STREAM** (`group by streamId, copyKind`). Copies on
  the compute stream serialize with the kernels; copies on the dedicated copy
  stream can overlap. This is how Kimi's hidden cost was found: 84 ms/token of
  *pageable* shared-expert H2D riding the compute stream, invisible in the
  totals.
- Sanity model: `wall ≈ Σ kernels + Σ H2D − overlap`. If they simply add up,
  there is **zero overlap** and that is the first thing to fix.
- Measure H2D efficiency as bytes/sec per size bucket: ~57 GB/s is the Gen5
  ceiling (no headroom left), ~19 GB/s on the compute stream means a *pageable*
  source (fix by pinning or device residency), and a copy whose time is
  independent of size hides a per-copy latency.

## Step 3 — write the failing test first

Budget test, in the model's existing test TU:

- **Never hard-code a bare number.** Latency budgets live in
  `tests/perf/kernel_budgets.json`, keyed by `(name, gpu_key)`, and go through
  `kernel_budget::Require` (`tests/test_kernel_budget.h`). Semantics: a record
  for this `gpu_key` → assert against it; no record → SKIP (do not judge);
  `LSX_KERNEL_BUDGET_WRITE=1` → record the measurement instead. A bare
  constant cannot tell "regressed" from "measured on another box" — that
  ambiguity was measured on this host (identical code, 2.1x apart), see
  `docs/open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md` § B.
- Assert a latency budget (µs), an aggregate bandwidth budget (GB/s), or a
  count budget (misses/token).
- Per-iteration `cudaEventRecord` around **only** the kernel; flush L2 with a
  >L2 `cudaMemsetAsync` (128 MB) between launches. A "flush+kernel minus
  flush-alone" subtraction is unreliable here.
- Guard the assertion with `#if defined(NDEBUG)`.
- Take the **minimum of several timed series** (best-of-5 is the house rule):
  one series still catches short-lived spikes.
- **Watch it fail.** A test that passes immediately proves nothing.
- Gotcha: gate-dependent kernels need the gate ON in the test (e.g. the seq1
  GEMV warp path is behind `SetQwenTcFastMath(true)`), or you measure and tune
  the wrong kernel; restore the toggle afterwards.
- For scheduling/overlap changes there is no host-observable timing state: pin
  the correctness-critical dependency rule instead (e.g. `wait_miss_pos`) and
  let the E2E bench be the perf gate.
- Recalibrate deliberately:
  `cmake -DLSX_KERNEL_BUDGET_WRITE=1 --build . --target <test>`, then review
  the JSON diff. Never recalibrate to make a red budget green without
  attributing the wall time (Step 2).

## Step 4 — micro-probes for kernel variants

Standalone `.cu` probes under `/tmp/opencode/<model>perf/` comparing candidate
kernels beat guessing.

- **Cross-check a probe against the real path before trusting it** — one probe
  grid bug inflated a variant's result by ~30%.
- Probe the real shapes, and re-probe after the change.

## Step 5 — implement the minimal change

- NVFP4 GEMM goes through the model-neutral layer; no family-named dispatcher
  copies.
- Preserve determinism: same bytes, kernels and accumulation order for a given
  token stream; a scheduling change must not alter `final_logits`.

## Step 6 — verify

- `test_<model>_selfcheck` — Paris ×3 bit-exact. This is the correctness gate
  for any change touching the path.
- Unit tests + `ctest --preset linux-debug -L fast`.
- Re-measure E2E at `--max-gen 2048`, **twice**: agreement within ~1% is the
  reproducibility evidence.
- Gates and traps in full: see **Verifying a numeric change** below.

## Step 7 — record

Both halves are machine-bound, in **both** directions:
`test_perf_baseline_logic` fails if a `perf_baselines.json` entry has no measuring
`TEST_CASE` in `tests/test_perf_baseline.cpp` (an entry without a target is an
unguarded baseline) **and** if a `TEST_CASE("perf baseline: ...")` has no
`add_test(NAME test_perf_baseline_* COMMAND test_perf_baseline "<that case>")`
anywhere in the ctest corpus (`tests/CMakeLists.txt` plus the generated
`${CMAKE_BINARY_DIR}/generated/model_targets.cmake`). Add the JSON entry, the case
**and** the manifest `perf` stage target in the same commit — the generator emits
the `add_test` with the case string, the label, the timeout and the GPU lock; you
never write that call by hand (and no model-path env is emitted: the test resolves
its artifact from the registry).

The second direction is not redundant. A `TEST_CASE` with no `add_test` looks like
coverage when you read the source and like absence when you read ctest, and nothing
contradicts the other view — `xing`, `mimo` and `alice` sat that way for a long time,
so `ctest -L perf` ran 16 of 19 families while the JSON kept xing's and alice's rows
looking maintained. Those two had been recorded by running the binary directly, which
works because Catch2 executes every `TEST_CASE` in a binary regardless of ctest
registration — so the row exists and no `ctest -L perf` run will ever maintain or
challenge it. `mimo` had never been measured at all.

Note also that a baseline row measured by hand is not the same as a row measured by
the test: only the test path captures `ttft_ms` and the NVML power samples, and a row
edited by hand cannot supply them honestly.

- `tests/perf/perf_baselines.json`: `tokens_per_sec`, `max_gen`, `gen_tokens`,
  `date`, `prompt`, plus the recorded evidence `ttft_ms`, `avg_power_w`,
  `peak_power_w`, `power_samples` (NVML board power sampled over `Infer`;
  zeros = unavailable). The perf-baseline test rewrites a higher result
  automatically (`JudgePerf`) and flags a degradation beyond ~3%; TTFT and
  power are recorded but not yet gating.
- One AGENTS.md line for the model + a findings section in the design doc
  (profile tables, changes, rejected experiments, next levers).

## Running the perf test suite (ctest)

Two target families share the `perf` label and answer different questions:

| family | what it measures | notes |
|---|---|---|
| `test_perf_baseline_*` | real-model t/s over 2048 tokens, one model per test | guards the committed `tests/perf/perf_baselines.json`; loads the full vault model (Kimi re-pins 544 GiB ≈ 5 min) |
| `test_blackwell_perf_*` | kernel-level microbenchmarks | needs the weights they name, not a full model run |

`ctest -L perf` is currently **15 tests** on Linux — a long, GPU-exclusive,
disk-heavy run.

Rules that bite:

- **Baseline numbers require a Release build**:
  `ctest --preset linux-release-znver4-perf` (or the AOCC equivalent).
  `test_perf_baseline` is NDEBUG-gated: a Debug run `SKIP`s (exit 4) before
  loading, because an unoptimized binary would land far below the reference and
  fail with `DEGRADED`. There is no useful Debug baseline run.
- **The baseline test rewrites the source tree.** It writes
  `tests/perf/perf_baselines.json` (through `LSX_SOURCE_DIR`) whenever the
  measured value is *higher* (`kUpdate`), or when the generated length changes;
  on a lower value it fails (`kDegrade`) and does **not** rewrite. A perf run
  therefore mutates a tracked file: review and commit that diff deliberately,
  and never run it while other GPU work is in flight — a contaminated
  measurement would be committed as the new reference.
- **A protocol change re-seeds.** An entry whose `max_gen`/`prompt` no longer
  matches the test is treated as absent and rewritten; keep the recorded prompt
  exactly.
- **A perf target resolves its artifact from the registry**: `ModelDir()` is
  `manifest_query::ConvertedDirFor(model_type)`, i.e.
  `<LSX_ARTIFACT_ROOT>/<org>/<model>/<profile>`, with the row's own
  `LSX_TEST_<MODEL>_ROOT` override still winning. A direct binary run needs
  `LSX_ARTIFACT_ROOT` exported; there is no subdir env any more (the old
  `LSX_PERF_VAULT_SUBDIR` carried the store's directory name and was joined onto
  the SOURCE root without the org, which is why every perf target SKIPped).
- **Ordering**: run perf last, after every other GPU task (host-resource
  policy).
- **The harness auto-cools between targets.** One 35–45 s decode heat-soaks the
  card (SM settles ~2812/3150 MHz under the 600 W cap) and the next target
  reads ~4% low → false DEGRADED. `test_perf_baseline` calls
  `perf::CoolDownGpu` (NVML) before each `RunPerfModel` and waits for the
  temperature to **plateau at its running minimum** (~2–3 min back to the
  ~40 °C floor). `LSX_PERF_COOLDOWN_TIMEOUT_S` caps it (default 300; 0
  disables). It removes the back-to-back component only — a genuinely warm room
  whose plateau is above the reference stays unreachable (`docs/open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`
  §W).
- **Missing vault → SKIP, not fail** (`SKIP_RETURN_CODE 4` per target). A model
  with a JSON entry but no ctest target is unguarded — when you ingest a model,
  add both (the `ingesting-a-model` skill's Definition of done).
- **Rebuild the target first**: a stale binary measures the old code (see
  Common mistakes).

## Verifying a numeric change

Any change to a numeric path — a kernel, the quantizer, a dispatch threshold,
the cache policy, stream scheduling — must clear all of these before it lands:

1. `test_<model>_selfcheck` — Paris ×3 with bit-exact `final_logits` between
   cycles, on the **real** model (not a synthetic fixture).
2. A two-call `Infer` `final_logits` memcmp — run-to-run determinism (the
   round-25b gate).
3. The model's unit tests + `ctest -L fast`.
4. E2E at the canonical length, reproduced twice.

Then the traps:

- **Bit-exact vs tolerance class.** The legacy dequant→bf16 GEMV chain is the
  bit-exact reference. The TC/W4A4 path and split-K compositions are *tolerance
  class*: parity with the legacy chain is not required, but determinism is
  binding. Tests that pin the numeric class set `FastMathMode::kOff`.
- **A scheduling change must be numeric-neutral.** Same bytes, kernels and
  accumulation order ⇒ the same token stream. If the stream changes you have a
  bug, not a faster kernel — this is how the LFU eviction policy was caught
  (early EOS at 1386 of 2048 tokens) while the upload-pipelining change was
  cleared (2048 tokens, twice).
- **Validate at the canonical length, not 128 tokens.** LFU measured +15% at
  128 tokens and 3.7 t/s at 2048.
- **Keep any simulator/reference faithful to the shipped code.** An unguarded
  LFU simulation predicted a hit-rate gain the real guarded policy never
  delivered.
- **A red determinism test reverts the lever** (precedent: device-side decode
  was re-parked after the round-25b divergence).
- **Any test that can skip needs `SKIP_RETURN_CODE 4`**, otherwise CTest
  reports the skip as a failure. `test_ctest_labels_guard` now enforces this for
  every target whose source calls `SKIP(` (it caught `test_kimi_moe_tc_diag`).

## GPU perf knobs (`gpu_perf`)

- `gpu_perf` (`include/lsxcommon/gpu_perf.h`) resolves per-GPU knobs at engine
  init (`InitializeEngine`): TC fast-math default, GEMV size limit (MiB),
  split-K threshold, expert-cache VRAM percentage, dual-stream staging.
- Resolution priority: **env > test cli_mode (kOn/kOff, tests only) > vault
  calibration cache (`perf-cache/<gpu-key>.json`) > passport heuristic >
  committed base (`BaseKnobs` in `gpu_perf.cpp`)**.
- `LSX_PERF_AUTO=0` returns the legacy bit-exact defaults; `LSX_TEST_FAST_MATH`
  forces the TC path.
- Call-once guarantee `perf::EnsurePerfPolicyApplied()` runs from
  `LsxModelFactory::Create` and `InferenceEngine::InitializeEngine`; explicit
  test `SetCurrentKnobs` layers on top without re-resolving.
- Unknown GPU: the first `Load` runs a one-time GEMV calibration on synthetic
  NVFP4 buffers and stores the result in the per-user cache
  (`lsx_cache_path("perf-cache/<gpu>.json")`), never in a model store. There is
  no CLI calibrate — `BaseKnobs` entries are authored by hand from calibration
  snippets.
- Knob consumers: `SetLsxGemmMaxMib`, `SetLsxSplitKGridMin` (`lsx_tc_gemm`),
  `lifecycle::ExpertCacheSlots(..., slots_pct)`, `LsxDualStreamStaging()`.
- Tri-state `FastMathMode { kAuto, kOn, kOff }` (`fast_math_mode.h`): models
  open `LsxFastMathScope(EffectiveFastMathMode(cfg_))` inside `Infer`; tests
  pinning the numeric class must set `kOff` explicitly.

## Lessons that cost GPU time

- **Validate policy changes at 2048 tokens.** An LFU cache policy measured +15%
  at 128 tokens and regressed to 3.7 t/s with early EOS at 2048 → reverted.
- **Keep the reference faithful to the shipped code.** An unguarded LFU
  simulator predicted a hit-rate gain the real same-batch-guarded policy never
  delivered.
- **Attribute before optimising.** Dequant was 71% of kernel time, but the wall
  was dominated by an un-overlapped H2D stream — a dequant-only fix would not
  have moved the number.
- **Cheap wins first.** Pinning/residency conversions and a wrong dispatch
  threshold beat writing new kernels.
- **Pin the traffic split before optimising a copy path.** Measured on
  kimi-k3.1-pro: 61.8 GiB/token goes H2D, of which routed experts are 25.5 GiB
  (41%) and the streamed latent/shared/dense plane is 36.4 GiB (59%). A pinned
  staging ring for the expert half cut the host-side copy cost by 62% (118.0 →
  45.3 s inside `cudaMemcpyAsync` in a 32-token run) and moved the wall by
  nothing, because (a) that 41% was already being served from a 665 GB page
  cache at 23.7 GB/s and (b) it is the minority of the traffic. Optimising the
  smaller half of a copy-bound wall is how a 62% improvement in one term becomes
  a 2% change in the number.
- **A pageable `cudaMemcpyAsync` is not asynchronous.** The driver stages the
  pageable source through an internal pinned buffer during the call, so the host
  blocks for the whole copy and the two halves never overlap. That is why host
  time inside the call can exceed wall time on some streams, and why "it is
  async, so it must overlap" is not a measurement.

## Common mistakes

- Measuring at 128 tokens and quoting it.
- Optimising the kernel with the largest *total* share instead of the largest
  *serial* share.
- Anything else running on the GPU during a measurement.
- A stale test binary: after adding test cases, rebuild the target explicitly
  and confirm with `--list-tests` (this silently produced "passing" runs).
- Reading seed/config files as the bottleneck when the wall is PCIe-bound.
