---
name: perf-tuning-a-model
description: Use when raising or measuring inference throughput (t/s) of a logestix model — profiling a decode token with nsys, finding the serial bottleneck per stream, writing RED latency/bandwidth/miss-count budget tests, tuning gpu_perf knobs, validating at the canonical 2048-token protocol, or recording a perf_baselines.json entry. Triggers - decode tok/s, t/s, lsxbenchmark --bench, perf-базлайн, узкое место, budget test.
---

# Perf-tuning a model

## Overview

Throughput work is a measured loop, not intuition: reproduce the true baseline
→ attribute the wall time to a *stream* → fix the dominant serial term →
re-measure at the canonical protocol. The two rounds that worked here (gemma
+14.6%, Kimi-K2.7-Code +65%) both came from measurement; every guess that
skipped step 2 was discarded.

**REQUIRED SUB-SKILL:** use `test-driven-development`.

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
  `prompt`, `gen_tokens`. Same prompt and `max_gen`, or the number is not
  comparable.
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

- Assert a latency budget (µs), an aggregate bandwidth budget (GB/s), or a
  count budget (misses/token).
- Per-iteration `cudaEventRecord` around **only** the kernel; flush L2 with a
  >L2 `cudaMemsetAsync` (128 MB) between launches. A "flush+kernel minus
  flush-alone" subtraction is unreliable here.
- Guard the assertion with `#if defined(NDEBUG)`.
- **Watch it fail.** A test that passes immediately proves nothing.
- Gotcha: gate-dependent kernels need the gate ON in the test (e.g. the seq1
  GEMV warp path is behind `SetQwenTcFastMath(true)`), or you measure and tune
  the wrong kernel; restore the toggle afterwards.
- For scheduling/overlap changes there is no host-observable timing state: pin
  the correctness-critical dependency rule instead (e.g. `wait_miss_pos`) and
  let the E2E bench be the perf gate.

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

## Step 7 — record

- `tests/perf/perf_baselines.json`: `tokens_per_sec`, `max_gen`, `gen_tokens`,
  `date`, `prompt`. The perf-baseline test rewrites a higher result
  automatically (`JudgePerf`) and flags a degradation beyond ~3%.
- One AGENTS.md line for the model + a findings section in the design doc
  (profile tables, changes, rejected experiments, next levers).

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
  NVFP4 buffers and stores the result in the vault. There is no CLI calibrate —
  `BaseKnobs` entries are authored by hand from calibration snippets.
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

## Common mistakes

- Measuring at 128 tokens and quoting it.
- Optimising the kernel with the largest *total* share instead of the largest
  *serial* share.
- Anything else running on the GPU during a measurement.
- A stale test binary: after adding test cases, rebuild the target explicitly
  and confirm with `--list-tests` (this silently produced "passing" runs).
- Reading seed/config files as the bottleneck when the wall is PCIe-bound.
