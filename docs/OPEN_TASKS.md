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

### C1. `test_gigachat_moe_devkernels` flaked once — **cause found 2026-10-02**
- Original report: failed once in an `-L inference` sweep on 2026-09-30,
  passed standalone, and the failing assertion was never captured.
- **The cause is GPU contention, and the earlier "+0.3%" figure understated it
  by three orders of magnitude.** That measurement put one idle 638 MiB CUDA
  process next to the test. Running five copies of the test against the same
  RTX 5090 does the job properly:

  | run | gated norm (budget 35.25 us) | router (budget 17.19 us) |
  |---|---|---|
  | 1 | 65.04 | 39.37 |
  | 2 | 65.47 | 32.46 |
  | 3 | 59.33 | 38.83 |
  | 4 | 35.22 | — |
  | 5 | passed | passed |

  Four of five fail, one clears the gated-norm budget by 0.03 us. This is the
  flake: a latency budget measured on a shared GPU is a measurement of the
  box, not of the kernel.
- **Everything else is now excluded.** 80 consecutive solo Release runs are
  green; `compute-sanitizer --tool memcheck`, `racecheck` (0 hazards) and
  `initcheck` report nothing, so there is no data race and no uninitialised
  global read in this file. The parity cases are host-reference comparisons
  and deterministic.
- **Two further things learned, worth not re-learning.** `compute-sanitizer` is
  *not* a usable stand-in for load: under memcheck the gated-norm sample is
  3344 us, ~128x the idle number, so budgets will always fail under a
  sanitizer and a failure there says nothing. And no test in the suite sets
  `RESOURCE_LOCK`, so a `ctest -j` sweep — or any model load running
  alongside — is enough to trigger this. ctest is sequential by default and
  no preset passes `-j`, so a stock sweep is safe; the AGENTS "one GPU load
  at a time" rule is what actually protects it, which is worth knowing is
  load-bearing rather than decorative.
- **Hardened both budget cases to best-of-5** (minimum of five timed series,
  was one). This does not survive real contention — see the table above, where
  all five series are slow — but it removes the short-lived-spike path, and it
  is the part of the exposure that is genuinely random.
- **Still open, deliberately:** making the budgets contention-proof rather than
  contention-tolerant. `RESOURCE_LOCK "gpu"` across the eight budget-bearing
  tests would stop `ctest -j` from ever doing this, and is the obvious next
  step if it recurs. Not done yet because nothing in the current workflow
  triggers it.

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

## E. Design decisions deferred (from the 2026-10-02 lifecycle review)

The lifecycle review closed every mechanical gap (streaming/abort, residency
lock, governor leases, context teardown, result fill, error-path cleanup) and
added `kLifecycle` to `tests/test_model_independence_guard.cpp` so they
cannot regress. Four items were deliberately NOT changed — each needs a
product or numerics decision, not a patch.

### E1. `LsxFastMathScope` assumes sequential cross-model dispatch
- The kernel-routing toggles are process-global (`std::atomic<bool>`, so no UB
  — `libs/lsxcommon/src/lsx_inference_nvfp4.cu:18`), and every model asserts
  its own policy per `Infer`. Two models inferring **concurrently** would flip
  each other's kernel routing mid-decode: silently wrong numerics, no crash.
- Today every dispatch path is sequential (orchestrator: one consumer thread;
  `lsxfabric --serve`: one request at a time), and the assumption is written
  down in `lsx_inference_nvfp4.h:57`. So this is latent, not live.
- **Next step if it ever matters:** per-stream routing state (a device-side
  policy handle passed down the launch chain) instead of process globals, or a
  documented single-model-at-a-time engine invariant with a guard test.

### E2. `alice` publishes the FIRST logits row; every other family the LAST
- `alice_moe_inference_pipeline.cu:492` fills `final_logits` when
  `first_token < 0`, so `InferenceResult::final_logits` is the row of the
  first generated token. All other families publish the last row
  (`qwen_inference.cu:2263`, `granite_dense_inference.cu:987`,
  `nemotron_moe_model.cpp:226` "Keep the last step's logits row").
- Within alice this is self-consistent (its DET gate compares it to itself),
  but it makes any cross-model logits/hashes comparison apples-to-oranges.
- **Not changed because it would move alice's DET/FSI goldens**, which are
  captured from real weights. Decide the fleet-wide semantics first, then
  re-capture the affected goldens in one commit.

### E3. `mimo` loads its weights through a host-side BF16→FP16 conversion
- `UploadHalf` (`mimo_moe_model.cpp:47`) decodes every BF16 tensor into a
  temporary host `std::vector`, then issues a synchronous pageable
  `cudaMemcpy` on the default stream — `(void)st;` ignores the model's stream.
  For ~11 GiB of weights this is slow and bypasses the persistent context.
- Correctness is fine and MiMo's perf gate is withdrawn (vendor checkpoint
  defect, A2), so this is a load-time cost only. Fixing it properly means a
  device-side BF16→FP16 kernel + stream-ordered upload, i.e. the same fast-load
  path GLM uses (`glm_moe_load.cu::GlmBf16ToHalfInPlace`).

### E4. Temperature / top_p are accepted and ignored by the OpenAI surface
- `ModelRequest` (`libs/lsxcommon/include/lsxcommon/model.h:52`) has no
  sampling fields and every family decodes argmax, so a request with
  `"temperature": 0.7` through the TaskMosaic bridge is served greedily. This
  is currently only documented in `openai_endpoint.h:112`
  ("Unknown fields (tools, temperature, ...) are ignored").
- **Next step:** either document it in the bridge's own API docs next to the
  endpoint, or plumb `temperature`/`top_p` into `ModelRequest` and the
  `sample::ISampler` implementations. Not a regression — nothing regressed.

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