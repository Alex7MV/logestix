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

## B. Hardware-dependent budgets — expected red off-target, **not** defects

Measured on RTX 5060 Ti (GDDR7, ~179 GB/s peak). Every one of these is a
wall-clock or bandwidth number calibrated on the HBM target, so they cannot
pass on consumer Blackwell. They are correctly filed under `inference`, not
`fast` — the label discipline is right, and `fast` stays green because of it.

| Test | Assertion | Measured on 5060 Ti |
|---|---|---|
| `test_qwen_nvfp4_gemv` (+13 siblings) | `test_qwen_nvfp4_gemv.cu:296` `gbps > 260.0` | 119.8 / 178.5 / 180.4 GB/s |
| `test_qwen_moe_exp_kernels` | `test_qwen_moe_exp_kernels.cu:648` `median < 0.045` | 0.052–0.082 ms |
| `test_gemma_moe_norm` | `test_gemma_moe_norm.cpp:149` `us < 3.4` | 9.24 |
| `test_qwen_moe_kernels` | `test_qwen_moe_kernels.cu:388` `us < 20.0` | 31.24 |
| `test_gemma_moe_moe` | `us < 700.0` | — |
| `test_nemotron_moe_kernels` | `batch_us < single_us * 0.6f` | — |
| `test_glm_moe_flash_kernels` | `us < 155.0` | — |

**Status: open, needs a decision.** Should these `SKIP` on non-target GPUs so
`inference` can go green on dev hardware, or is a per-arch budget table wanted?
The perf-baseline path (`docs/*_SPEED_AUDIT.md`, `test_perf_baseline_*`) already
assumes measured reference hardware, so a precedent exists. Right now nothing
distinguishes "too slow for the target" from "regressed", which is the part that
actually costs debugging time. On the RTX 5090 all of these pass, so the
question only bites on the consumer dev box.

## C. Intermittent

### C1. `test_gigachat_moe_devkernels` flaked once — **status: watching**
- Failed in one `-L inference` sweep; passes standalone (2155 assertions, 7
  cases) and passes under `ctest -R ^test_gigachat_moe_devkernels$`.
- 10 consecutive standalone runs all pass. Suspect GPU contention rather than a
  logic defect, but that is a guess — it has not recurred, so nothing was
  changed. Keep watching; if it recurs, capture the sweep that produced it
  (note: a parallel `ctest -j` over several large models reproduces GPU
  contention by construction — run the `inference` sweep serially).

---

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