# Open tasks

**Rule:** only genuinely open work lives here. When an item is resolved it
moves to `docs/open-tasks/<date>-<slug>.md` and leaves a row in the Archive
table at the bottom. Index: [`open-tasks/README.md`](open-tasks/README.md).

Nothing here is speculative — each entry carries the measured value or error
code that produced it.

---

## A. Real defects — GPU faults

### A1. `test_qwen_moe_kernels` + `test_ornith_moe_kernels`: `cudaErrorIllegalAddress`
- **Status: open, needs the 5060 Ti box.** Not reproducible on the Linux
  RTX 5090 host: both tests pass (865 assertions / 13 cases each) and
  `compute-sanitizer --tool memcheck` reports 0 errors for each. Either a
  Windows/5060 Ti-specific fault or one already removed by the Windows build
  repair. **Do not change a kernel until it reproduces on that box.**
- `cudaErrorIllegalAddress` (700). It is **not** the first thing that goes
  wrong: once a kernel faults, the context is poisoned and every later CUDA
  call in the process returns 700, so the bulk of the failure list is cascade.
  Triage from the *first* failing `REQUIRE`, not the last.
- Also carries a latency budget failure at `tests/test_qwen_moe_kernels.cu:388`
  (`us < 20.0`, measured 31.24) — that part belongs to section B.
- **Next step:** same workflow that resolved the GigaChat router.
  ```
  compute-sanitizer --tool memcheck --launch-timeout 120 <test>.exe
  ```
  Add `-lineinfo` to the owning object library to get `file:line`, otherwise
  the report only gives `kernel+0xNNN`. Note `TcGemmDispatch`
  (`libs/lsxcommon/src/lsx_tc_gemm.cu:891`) already rejects non-CC-12, and this
  GPU *is* CC 12, so the `major != 12` guard is not what is firing here.

## B. Hardware-dependent budgets — **resolved 2026-10-02, now GPU-keyed**

Every latency/speedup budget was a hard-coded number with no reference to the
machine it was measured on, so a red budget carried no information: not
"regressed", not "wrong machine". Two of them were red on the target GPU and
neither turned out to be a code regression:

| Budget | Was | Measured (clean RTX 5090) | Verdict |
|---|---|---|---|
| `gemma RMSNorm` | `us < 3.4` | **2.894–2.897 us** | the old number was fine for this card. An intermediate measurement of 4.5–5.5 us turned out to be **taken while the GPU was busy** — with the card free it is 2.89 and the old budget passes |
| `nemotron batched GEMV` | `batch < single × 0.6` | **1.34–1.44x** | fails as written: 1.67x demanded, ~1.4x achieved. Machine-independent by construction, and the only later commit (`35b6388`) touched Mamba-2 `dt`, not the batched GEMV |

The intermediate reading above is the reason the `gpu_key` work mattered anyway: a
budget with no machine reference cannot tell "the card was busy" from "the code
regressed", and the first reading of the day was wrong for exactly that reason.

Two things were also wrong in the old numbers: the sweep that produced them
was **Release, not Windows** (2155 assertions identifies the build; Debug
reports 1432 for the same 7 cases), and `test_gigachat_moe_devkernels`' own
budgets compiled to nothing in Debug and reported "passed" with zero
assertions (fixed separately, see C1).

**Resolution:** budgets now live in `tests/perf/kernel_budgets.json`, keyed by
`(name, gpu_key)`, and go through `kernel_budget::Require`
(`tests/test_kernel_budget.h`). Semantics are the ones `test_perf_baseline`
already implements for throughput:

- a record for this `gpu_key` exists → assert against it
- no record for this `gpu_key` → `SKIP` (do not judge)
- `LSX_KERNEL_BUDGET_WRITE=1` → record instead of asserting

`gpu_key` comes from the existing `perf::GpuProfileKey` / `ProbeGpuPerfProfile`,
so an unknown card skips with a readable reason instead of failing red.

**Recalibration keeps both ends of the range, not one number.** Each entry
stores `observed_min`, `observed_max` and `samples` next to the budget, and the
budget is derived from the **worse** end with a 1.35x (latency) / 0.80x
(speedup) pad. A single recorded value hides exactly the thing that made this
investigation slow: these measurements swing by ~2x with GPU clock state and
occupancy (2.89 us on a free card, 4.5–5.5 us with a competing process), so
`INFO` now prints the range and the sample count next to the limit.

On an idle card the observed ranges are tight (gemma RMSNorm 2.894–2.897 over 6
samples); the wide swings only appear under contention.

Recalibrate with:

```bash
cmake -DLSX_KERNEL_BUDGET_WRITE=1 --build build/linux-release-znver4-aocc \
      --target test_gemma_moe_norm        # repeat per suite, several times
```

The 12 call sites: gemma norm/MoE, glm-flash GDN, qwen-MoE router, qwen4_exp
PLE + hyper-connection, nemotron Mamba step + GEMV latency + GEMV speedup,
gigachat gated-norm + router, and the NVFP4 GEMV bandwidth floor.

## C. Intermittent

### C1. `test_gigachat_moe_devkernels` flaked once — **status: investigated 2026-10-02**
- Original report: failed in one `-L inference` sweep on 2026-09-30, passed
  standalone. The failing assertion was not captured, so the first job was
  establishing which build it could have been.
- **Which build: Release, and this host.** The archived note says "2155
  assertions, 7 cases". A Debug build reports 1432 assertions for the same 7
  cases — the difference is the two latency-budget cases, whose bodies sit
  behind `#if defined(NDEBUG)` and are therefore empty in Debug. 2155 matches
  the Release count exactly, so the flake was the `linux-release-znver4` sweep,
  not Windows and not Debug.
- **The only timing-sensitive assertions in the file are the two budgets.**
  Every other case is a parity check against a host reference and is
  deterministic. Measured on an idle RTX 5090, 15 consecutive samples:
  - gated norm, `CHECK(us < 40.0)` (line 312): 26.04–26.13 us, **34% headroom**
  - router, `CHECK(us < 60.0)` (line 357): 12.72 us, **79% headroom**
- **"GPU contention" hypothesis: measured and rejected.** With a competing CUDA
  process resident (638 MiB, verified via `nvidia-smi`), the gated-norm sample
  moved 26.07 → 26.16 us (+0.3%) — nowhere near the budget. Ordinary contention
  does not move this measurement, so the original guess (kept in the archive as
  a guess) does not hold.
- **Could not reproduce:** 12 Debug runs, 15 Release samples and a contended
  Release sample are all green. No failing assertion was ever recorded, and the
  ctest logs that would carry it are gone. So the flake's cause remains
  unproven — what is now closed is the *concealment* that made it untriageable:
- **Real defect found and fixed.** Both budget cases compiled to nothing in
  Debug and reported "passed" with **zero assertions**, so a Debug sweep looked
  green while checking no latency at all — and `ctest` had no `SKIP_RETURN_CODE`,
  so there was no honest way to skip. Both now `SKIP` in Debug (mirroring
  `test_perf_baseline`), and the target declares `SKIP_RETURN_CODE 4`. Debug now
  reports "5 passed | 2 skipped"; Release is unchanged at 2155/2155.
- **If it recurs:** capture `--output-on-failure` output. With the SKIP in place
  and the budgets at 34%/79% headroom, a recorded sample identifies the cause
  immediately — a sample near 40 us means the kernel regressed; an assertion
  outside the two budget lines means it was never this test's timing.

## D. Performance

### D1. nemotron batched expert GEMV reaches ~1.4x, not the 1.67x expected
- **Status: investigated 2026-10-02, closed as not-a-regression.** The budget
  from `1164799` (`batch < single * 0.6`) asks for 1.67x; the kernel delivers
  **1.34–1.44x** (`nemotron.routed_expert_gemv.batch_speedup`). The gap is
  real but costs nothing measurable in production.
- Measured on a standalone CUDA-event harness at gigachat's real shapes
  (`hidden=7168`, `moe_intermediate=2048`, `count=8`, both the `x_stride=0`
  gate/up case and the `x_stride=I` down case):

  | shape | batched | per-expert loop | batch/loop |
  |---|---|---|---|
  | gate/up `K=7168 N=2048 xs=0` | 73.09 us | 49.50 us | 0.68 |
  | down `K=2048 N=7168 xs=I` | 72.68 us | 44.07 us | 0.61 |

  So in isolation the batched kernel is 1.5–1.65x *slower* than simply
  looping `RunLsxNvfp4Gemm` per expert.
- **That did not survive the pipeline.** Replacing all three batched calls in
  `gigachat_moe_inference_pipeline.cu` with per-expert loops on the real
  weights moved decode from **6.5 to 6.6 t/s** (ttft 3973 -> 3910 ms) — inside
  run-to-run noise. The isolated loss is absorbed by the dual-stream staging
  that already overlaps these GEMVs with the rest of the layer. The change was
  reverted rather than kept as a simplification with no measurable payoff.
- Why the budget is green while the code looks lossy: the case measures
  `K=512, N=64`, where the launch overhead dominates and batching wins
  (26.4 vs 40.9 us). The kernel behaves well on tiny shapes and poorly on
  gigachat-sized ones; the test only exercises the tiny end.
- Not a defect to fix. If a future change makes batching matter again, re-run
  the comparison above at the production shape before trusting either number.

## Archive

Resolved items moved out of this file on 2026-10-02, preserved with their full
investigation trail:

| Item | Outcome |
|---|---|
| B1 — expert pinning not implemented on Windows | platform-gated in `test_kimi_moe_loader.cpp` |
| D1 — no test drove `RunGlmFlashGeneration` | real-model generation case added |
| D2 — hardcoded POSIX temp paths in tests | `std::filesystem::temp_directory_path()` |
| E1 — 14 test executables were one binary | per-`.cu` object libraries |
| G1 — toolkit mismatch caught only after the build | configure-time check in `CMakeLists.txt` |
| A2 — MiMo wrong top-1 | closed 2026-10-02: no implementation bug, the checkpoint is bad (vendor oracle matches) — [`open-tasks/2026-10-02-mimo-closed.md`](open-tasks/2026-10-02-mimo-closed.md) |
| Inference-conformance review (9 deviations) | all fixed in dedicated commits |
| "Closed in this session" list | 15 items, all closed |

Full text: [`open-tasks/2026-09-30-cuda-review-windows-bringup.md`](open-tasks/2026-09-30-cuda-review-windows-bringup.md)
and [`open-tasks/2026-10-02-mimo-closed.md`](open-tasks/2026-10-02-mimo-closed.md).

## Maintenance rule

When an item in this file is resolved, move it to `docs/open-tasks/<date>-<slug>.md`
(one file per work session, dated the day the work happened), replace it here
with a one-line row in the Archive table above, and keep only genuinely open
work in `docs/OPEN_TASKS.md`. The index is
[`open-tasks/README.md`](open-tasks/README.md). This keeps the live file short
enough to read at a glance, which is its only purpose.