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

### D2. `xing` decode: 14.4 -> 117.2 t/s (nine serial kernels + device dispatch + graph replay)
- **Status: CLOSED for the serial-kernel class (2026-10-05).
  14.40 -> 117.20 t/s perf / 9.10 -> 20.10 t/s r3 at 2048 tokens.** Every
  single-block, thread-0-serial and under-occupied kernel on the decode path is
  fixed or measured-and-declined; what is left is the GEMV occupancy, which is a
  different (numerics) project - see the end. Every fix came from the same
  method: nsys + a `--max-gen` sweep to separate fixed from context-dependent
  cost, then a Release-only latency budget at the PRODUCTION shape.
- The three serial costs, in the order nsys found them:

  | kernel | before | after | what it was |
  |---|---|---|---|
  | `xing_hyper_conn_kernel` | 229 us | 61 us | `<<<1,256>>>`, 24 sequential 256-thread tree reductions (192 dependent barriers) |
  | `xing_router_proj_kernel` + `..._topk_kernel` | 129 us | 6.4 us | `<<<1,256>>>`: one SM read all 458 KiB of router weights, and XingSigmoid (a double-precision glibc replica) ran 8x per warp back to back before the top-k could start |
  | `xing_attn_kernel` | 814 us | 209 us | `<<<32,256>>>` (32 of 170 SMs), softmax serial on thread 0, values serial over t on one thread per dim |
  | `xing_attn_split_kernel` | 209 us | 63 us | the fixed grid was 32 blocks - one per head; split-K over position chunks (heads x 8 = 256 blocks) plus a combine kernel |
  | `xing_hyper_conn_gates_kernel` | 36 us | 6.5 us | 20 Sinkhorn iterations + 16 exps serial on thread 0, and `ExpfGlibc` is a DOUBLE-precision replica - a consumer SM retires FP64 at 1/64 rate; then warp-0 registers with shuffles |
  | `xing_hyper_conn_collapse_kernel` | 8 us | 2.0 us | the 57 KB collapse was the tail of the one-block gates kernel (8 warps of loads in flight); its own wide block, no FP64 in it |
  | `xing_attn_combine_kernel` | 6.8 us | 1.8 us | kAttnSplits rescale factors were recomputed per head DIM thread - 16 serial double-precision exps each; now computed once per chunk |

- The attention ones are what the length curve pointed at:
  `us/token = 14258 + 16518 per 1000 cached positions` meant 54% of a 2048-token
  step was context-dependent, all of it in that kernel. After the two attention
  passes the same sweep gives **7016 + 1680**, i.e. the context term shrunk 9.8x
  and is now 19% of the step; the fixed term is 81% and is the whole frontier
  (it was 46% of the step before this series).
- The depth mixer needed TWO passes and the second one had to be re-derived after
  the first looked done. Splitting `<<<1,256>>>` into a projection (24 blocks,
  one per hc_fn row) and a gates block took 61 -> 40 us - and then nsys said the
  gates block alone was 36 of those 40. The cause was arithmetic, not occupancy:
  `ExpfGlibc`/`XingSigmoid` are bit-exact glibc replicas written in double
  precision, so the 16 exps + 4 sigmoids of the Sinkhorn were 20 serial
  double-precision calls on thread 0. One cell per thread (and one row/column
  reciprocal per thread) is numerically IDENTICAL - exp and max are element-wise
  and the Sinkhorn's per-cell updates were already independent - for 40 -> 17 us.
  **The lesson worth keeping: on consumer Blackwell (1/64 FP64) a "tiny" serial
  scalar tail can dominate a kernel, and nsys per-kernel attribution is what
  finds it - the block count looked fine.**
- Fixed cost, where the next levers are: the per-token `[vocab]` logits round
  trip is GONE (the sampler now reads `session_->d_logits` in place and only the
  8-byte argmax comes back, with the full row captured just for the final token
  - worth +0.7% end to end, 106.9 -> 107.9 t/s at 128 tokens, so it was never
  the E2E bottleneck, but it removes 1 MB of PCIe traffic and three
  `cudaMalloc`/`cudaFree` per token, which is what concurrent sessions on one
  model pay for). On the GPU side the remaining single-block work is the gates block
  (13 us x 40 layers = 0.5 ms/token) and the attention could still use more SMs
  (one block per head = 32 of 170; a position-split plus a combine kernel would
  open 8x more). `ResolveXingRouted`'s 38 D2H syncs per token were
  MEASURED as free at 128 tokens (64.3 vs 64.3 t/s with them skipped), so they
  are not worth doing before the two above.
- Numeric gates re-run on the real weights after every change (all three kernel
  rewrites are tolerance class): SelfCheck Paris x3, FSI 3/3, CODE 3/3, DET 3/3.
  Each kernel carries a Release-only latency budget and a host-parity case at the
  production shape - the original cases used hidden=64 / 16 experts, which is why
  all three defects survived a green suite.
- **What the remaining 7.0 ms/token is NOT: bandwidth.** Per token the decode
  path streams the 4-of-64 routed experts, the shared expert, the attention
  projections, the two dense layers and the LM head: ~2.68 GB, so at 113.6 t/s
  that is 304 GB/s, 17% of the 1792 GB/s peak -
  and a layer's ~53 MB should take 31 us at peak, while a layer measures ~222 us
  (9.1 ms / 41). So decode is latency/occupancy-bound, not bandwidth-bound, and
  the next lever is the GEMV path itself: `RunLxGemmVec` splits output rows
  across blocks conservatively and never fills 170 SMs, so each of the ~15
  dependent kernels per layer pays launch + drain latency at low occupancy.
  Raising decode further means wider GEMV tiles (more rows per block, split-K
  over the K dimension) - correctness-first no longer, i.e. a real numerics
  project with its own gate, not another serial-kernel fix.
- Effective bandwidth went 31.6 -> 304 GB/s (113.6 t/s, 8.8 ms/token). The memory roofline for this model
  is 2.19 GB/token => **694 t/s at 85% of peak**; at 113.6 t/s the model reads
  304 GB/s of it - which is the point above: the roofline is not the binding
  constraint any more, the GEMV occupancy is.
- **What the next project is, and it is plumbing, not arithmetic.** After the
  router fix the profile is dominated by kernels that are far too slow for their
  bytes: `xing_rmsnorm_kernel` at 4.8 us for a 3584-float row, 163 launches per
  token, and 81 each of the three mHC kernels. All are single-block, so 169 of
  170 SMs idle during a 42 KB reduction, and a decode step is ~400 dependent
  launches. Replacing the RMSNorm's 8-barrier tree with warp shuffles bought
  +0.4% (19292/19323 ms vs 19366/19392) which is the point: **the floor of a
  one-block launch here is ~4 us regardless of what the block does.** Neither
  more ILP nor fewer barriers moves it. The lever that does is CUDA graph
  capture of a whole decode step, and that requires the routed-id dispatch to be
  device-side - so the id->slot table whose sync measured free above becomes a
  PREREQUISITE here, not a win. Second project: the prefill row-batching (below).
- **DONE (2026-10-05): the decode step is captured and replayed as a CUDA graph**
  (114.1 -> 117.2 t/s at 2048, 113.9 -> 117.1 on the repeat; `LSX_XING_CUDA_GRAPH=0`
  restores the legacy per-token launch path). Three things had to become true
  first, and each is a separate commit:
  1. the routed-expert dispatch resolves on the device (`8bb474c`), because the
     host path drains the pipeline once per MoE layer and a drain cannot be
     recorded;
  2. `{pos, token_id}` live in device memory rather than in kernel arguments
     (`f7fa0bb`) - a graph bakes its arguments in, so a per-token value has to
     travel in memory. The embedding gather, the KV write and both RoPE kernels
     read them there now, and the RoPE kernels take the BASE YaRN tables and
     index their own row;
  3. the attention launches a position-independent geometry when captured
     (`3e0ae5a`): both forms at their maximum, excluding each other on the
     device position, with empty chunks publishing `(m = -inf, l = 0)` so the
     combine cannot read the previous position's slots. Same bits, verified at
     pos 2047 / 1000 / 255.

  What the graph does NOT cover: the sampler, the per-token logits readback and
  the KV-layout work below. `XingCaptureDecodeGraph` refuses (rather than
  capturing) when the step is not capture-legal - fast-math off, the layer-debug
  staging on - and a capture that fails after it started is reported instead of
  being papered over, because it leaves the context unusable.
- Tool calling is verified on the REAL weights
  (`test_xing_moe_selfcheck` "end-to-end tool round-trip", the same case nine
  other families run): the model emits its native markers as the control ids
  11/12, `arrow_tokenizer_decode` drops special tokens, so `output_text` reads
  as a bare `get_weather<param_key>city</param_key><param_value>Paris</...>` with
  no marker in it - which is why `xing_moe_tool_format::Parse` walks the ids and
  splices the marker bytes instead of decoding, and why the test asserts the ids
  are present rather than trusting the decoded text. An earlier "emission not
  observed" note was that same artifact, not a defect.
- Remaining per-token costs, in the order they were identified:
1. ~~`ResolveXingRouted` syncs once per MoE layer (38/token)~~ DONE, and the
   earlier "measured free" conclusion was HALF RIGHT. Skipping only the
   `cudaStreamSynchronize` really is free (64.3 vs 64.3 t/s, then 21569 vs
   21555 ms - 0.06%): a decode-loop sync costs only what it drains. What the
   experiment did NOT skip was the host resolve around it - six `assign()`
   calls plus nine pointer fields per selected expert, 38 times per token - and
   moving the whole thing to the device (uploaded per-layer expert pointer
   tables, ids read in the GEMV kernel) is worth **+3.4%**. Both halves had to
   go: the sync was free, the resolve was not.
2. ~~`XingMoeModel::Infer` round-trips the whole `[vocab]` logits row~~ DONE
   (9108ff4): the sampler now reads `session_->d_logits` in place and only the
   8-byte argmax comes back, with the row captured once for `final_logits`. The
   three per-step `cudaMalloc`/`cudaFree` are gone with it.
3. Prefill replays the prompt one token per step (no incremental/parallel
   prefill), so TTFT grows linearly with prompt length.
- **Next target, measured (2026-10-05): the mHC depth mixer, 1.32 ms/token, 15%
  of the step.** It is four launches per mHC block, two blocks per layer, so 320
  per token: proj 5.93 us (24 blocks), gates 7.06 us (ONE block), collapse 1.97 us,
  mix 1.55 us. The gates kernel is the interesting one - it is 0.56 ms/token on
  its own and it is not bandwidth: the Sinkhorn already lives in warp 0's
  registers at the frozen geometry (hc_mult=4, 16 of 32 lanes), so the cost is
  20 strictly serial iterations of shuffle-sums and divides, with the other seven
  warps idle. The obvious fix - merge gates into the wide collapse block so the
  57 KiB stream read overlaps the Sinkhorn - is the one that was already tried
  and reverted: the collapse block runs 1024 threads, whose 64-register cap makes
  the FP64 exp replica spill, and that cost more than the overlap won. The way
  through is a producer/consumer split INSIDE one launch: warps 1..7 prefetch the
  streams into shared memory while warp 0 runs the iterations, one barrier, then
  the collapse - FP64 stays off the wide path and the load overlaps the chain.
  Budget: `xing.mhc.decode` in `kernel_budgets.json` guards the current 12.5-13.0 us
  for proj+gates+collapse, so a regression here is visible immediately.
- The baseline is recorded in `tests/perf/perf_baselines.json` (mode `perf`
  and `r3`); it is still an explicit **floor**, not a tuned number, so the perf
  guard treats a regression against it as a real signal.
- Parity is unaffected: SelfCheck Paris and the FSI/CODE/DET gates pass at
  `kOff`, so this is pure throughput work.

## F. Decode throughput ramp over a session (2026-10-05) — no ramp on Xing

An operator's separate observation was that throughput sometimes *grows* during
inference. Measured on the production numeric class (default per-GPU policy,
`tests/test_xing_warmup_ramp.cpp`, host-timestamped token callback):

| horizon | Q1 (first quarter) | median | Q4 (last quarter) | Q4/Q1 |
|---|---|---|---|---|
| max-gen 2048 | 147.1 t/s | - | 107.5 t/s | **0.73** |
| max-gen 4096 | 142.0 t/s | - | 82.1 t/s | **0.58** |

So on Xing-4.0 throughput **FALLS** by 27% (2048) and 42% (4096) across a session,
and falls harder at the longer horizon. There is no warm-up ramp: the shape is the
context effect, and Q1 sitting ABOVE the session average (147 vs 127 t/s at 2048) is
why a single average hides a 27-42% spread.

Two consequences worth keeping:

1. **If a ramp was observed, it was a different family or configuration.** The most
   likely mechanism is the opposite of what "ramp" suggests: Xing replays its decode
   step from a CUDA graph, which removes per-step launch overhead, so Xing has little
   warm-up to have. A family without graph replay would show more. Worth identifying
   which family showed growth before concluding anything general.
2. **An average over a long horizon is dominated by the slow tail.** "3-5% at the 4096
   horizon" measured as one average is mostly a statement about the regime where the
   KV read is past L2. That is the regime the float4 score-phase change already
   improved (+12.5% at 8191), and it is where any further work should be aimed.

Method notes, both learned the hard way: the measurement must run ONE horizon per
process (two resident Xing instances need ~30 GiB and the second expert-slab
allocation OOMs on a 32 GiB card, and destroying the first model does not return the
memory within the process), and it must NOT pin `FastMathMode::kOff` — that measures
the legacy bit-exact path at 20 t/s, a ramp that never happens in production.

## G. Attention score dot products: who already vectorised (2026-10-05)

Audited before porting Xing's float4 score change, because the first candidate turned
out to be the wrong one:

| family | score dot product |
|---|---|
| xing | float4 (done 2026-10-05, +6.5% E2E) |
| **phi, granite** | **already `__half2` + two accumulator chains**, with a comment citing the QK critical path |
| qwen (+ornith), gemma | **scalar**, `dot += half2float(q[d]) * half2float(k[d])` per element |
| deepseek, mimo, gigachat | scalar (no `half2` anywhere in the file) |

So the port list is **6 families, not 8**, and phi/granite would have been wasted
work - their split kernel (the live decode path) is already vectorised, and only the
short-context non-split kernel was scalar.

The scalar families are the same shape as Xing's was: one 2-byte load per iteration
over `head_dim` with a serial accumulation, one K row in flight per thread. phi also
records the structural reason to expect the same win there: its own comment says the
single 128-deep FMA chain was the QK critical path, and its KV read is 26.5% of the
decode wall at ctx 2048.

**RESOLVED for the split kernel (+4.3% / +4.9%, bit-identical).** The alignment
hypothesis is dead — a runtime assert on the row pointer never fired, which is
expected: the port changes how a row is READ and moves no pointer, so alignment is
identical to the passing version and could not have been the cause. The real cause
was item 3: the golden gates exercise the **non-split** kernel, because SelfCheck
prompts are short enough that `splits` collapses to one chunk. Vectorising that kernel
breaks parity; vectorising the split kernel does not. The two are not
interchangeable, and the short-context one is deliberately left scalar.

  Qwen3.5-35B-A3B (AOCC perf tree)   base      split-kernel port
    max-gen 2048                   162.3        169.3 t/s  (+4.3%)
    max-gen 4096                   162.1        170.0 t/s  (+4.9%)

  All four real-weight gates pass (SelfCheck x3 Paris, FSI, CODE, DET). Bit-identical.

**STILL BLOCKED: the non-split kernel, in all six families.**
qwen and gemma accumulate with a plain `+=`, not `__fmaf_rn`, and the repo builds
bit-exact TUs with `-fmad=false`. The first thing tried was a vectorised dot that kept
the plain `+=` per element and the ascending d order - and it still broke parity:
`test_qwen_moe_{fsi,code,det}_selfcheck` all fail on Qwen3.5-35B-A3B with the change
and all pass without it (verified by stashing). So "just keep the `+=` and the order"
is NOT sufficient, which corrects the caution recorded above - it was a hypothesis,
not a known-safe recipe.

Not root-caused. The candidates that remain, in the order I would check them:
1. **Alignment.** The 8-byte path assumes each K row starts 8-byte aligned because a
   row offset is a multiple of `head_dim` halves. That holds only if `head_dim % 8 == 0`
   for the model actually loaded; if it does not, the guard falls back to scalar and
   the change is a no-op - so the fact that parity broke says the vector path WAS
   taken, which makes an unaligned `reinterpret_cast<const __half2*>` worth ruling out
   first by asserting the pointer alignment at runtime.
2. **Contraction under the vector form.** `-fmad=false` forbids fusing `a += b*c`, but
   it does not by itself forbid the compiler from changing how the 8-byte load feeds
   the dependency chain. The scalar chain is 1 FMA deep per element; the vector form
   has four independent products per iteration, and whether that stays bit-identical
   is an empirical question, not a spec question.
3. **Which kernel is actually live for these prompts.** SelfCheck uses short prompts,
   where `splits` collapses to one chunk - so the golden gates may be exercising the
   NON-split kernel, not the split one. Both were changed in the same commit, which
   would confound the result.

So the split kernel is ported and shipped; what is left is the non-split kernel, whose
parity break is a separate and unexplained problem. It is also the low-value half: it
is the short-context path, so it does not carry the decode throughput that motivates
the port. It probably reads the same rows with a different accumulation shape, but
that was not established.

**RESULT so far, and a correction to the audit that produced this list.** Shipped:
qwen (+4.3% / +4.9%) and gemma (+3.0%), both bit-identical. But two of the three
remaining families do NOT have the pattern this port targets, and my audit inferred
"scalar means needs port" from a grep that could only see the absence of `half2`:

| family | score structure | port applies? |
|---|---|---|
| **deepseek** | thread-per-DIMENSION: `for t { for d = tid; d < kHD; d += kThreads }`, warp-coalesced, no per-row serial dot | **no** - there is no serial row walk to vectorise |
| **mimo** | stages q and k into SHARED memory first, dots from smem | **different structure** - the lever would be the global->shared staging, not the dot |
| **gigachat** | thread-per-position serial float dot over `qk` (the pattern) | **yes** |

So the real list was 3 families, not 6: qwen, gemma, gigachat. deepseek needs nothing
of this kind, and mimo needs a different change entirely.

**gigachat: bit-identical, and worth NOTHING.** The model WAS on this host — an earlier
note said otherwise because the vault glob was case-sensitive and `*igachat*` does not
match `GigaChat`. With the model found: SelfCheck Paris x3 passes and, since
FSI/CODE/DET goldens were never captured for gigachat (they SKIP by design, unrelated to
any change), bit-identity was proven by the SelfCheck `logits_hash` — **identical,
`9b46473108d82b99`, base and ported**. So the port is correct.

And it buys nothing:

  GigaChat3.5-432B-A28B, AOCC perf tree, max-gen 2048
    base 11.0  ->  ported 11.0 t/s

Zero. The port is kept because it is bit-identical, strictly fewer load instructions
and zero-risk, but it is **NOT counted as a win** and must not be cited as one. The
reason is the same structural point as deepseek and mimo, one step further: gigachat's
attention reads an MLA latent (kv_lora + rope), so the score dot is short, and at
11 t/s the step is ~91 ms dominated by routed-expert GEMVs. The score phase is simply
not where gigachat's time goes. This is the third family in a row where "the pattern
looks right" did not predict the payoff — what predicted it was the fraction of decode
wall the score phase actually occupies.

Final port ledger: **qwen +4.8%, gemma +3.0%, gigachat 0%**. deepseek and mimo do not
have the pattern; phi and granite already had it.

The audit lesson: `grep -c half2` is not a classification. Two of the three families it
flagged have a different blocking structure, and reading the loop shape - not the
absence of a token - is what distinguishes "needs this port" from "needs something
else".

Not done here: an unmeasured kernel change is not a result, and neither is an
unmeasured failure. This entry is the failure and the narrowing plan.

## H. AliceAI-80B: the qwen port does not reach it (2026-10-06)

Asked to port Alice. It needs no port of its own and cannot take the qwen one:
`RunAliceGatedAttentionCore` calls **`qwen::RunQwenFullAttnKernel`**, which launches
`full_attn_softmax_kernel` — the **non-split** kernel, i.e. exactly the one the qwen
port deliberately left scalar because vectorising it fails
`test_qwen_moe_{fsi,code,det}_selfcheck`. Alice is 12 gated-attention layers of 48
(the other 36 are KDA recurrence, which has no score dot at all), and it takes the
non-split path, so the shipped split-kernel win never reaches it.

Alice baseline for the record: **21.1 t/s** at max-gen 2048 (AOCC).

So the options are (a) leave it — the reachable variant is the one with a proven parity
break, and with 12 of 48 layers carrying attention the ceiling is small; (b) solve the
non-split parity break first, which is still unexplained and would then benefit qwen's
and gemma's short-context paths too, not just Alice. (b) is the better lever if the
non-split path is ever worth attacking, and it is a strictly bigger prize than Alice
alone.

This is the fourth family where the filter paid off: routing the request through "what
fraction of the decode wall does the score phase actually own, and which kernel variant
does this family reach" answered the question before any code was written. Three of the
four earlier lookalikes (deepseek, mimo, gigachat) would have been wasted work.

## I. The non-split parity break: a compiler decision, not an arithmetic one (2026-10-06)

Went after it and got the boundary. It stands, and the record now says why it is hard:

- Row addressing in `full_attn_softmax_kernel` is **identical in form** to the split
  kernel's (`kbase + j*kv_heads*head_dim`, `q + (t*heads + h)*head_dim`), so
  alignment and structure are both ruled out as causes.
- The dot is the same helper, the same ascending order, the same plain `+=`, and it is
  bit-identical in the split kernel. Nothing in the arithmetic differs.
- On a **verified-fresh** binary the vectorised non-split kernel fails
  `test_qwen_moe_{fsi,code,det}_selfcheck`, and passes with the scalar one.

So the difference is what the **compiler** does, not what the code says. The only
mechanism available is FMA contraction, and the plausible story is that the scalar
non-split loop contracts and the vector form does not (or vice versa) - both are
allowed, nothing forbids either, and this library has `-fmad` ON. That would also
explain why the split kernel is unaffected.

The fix, if it is ever worth taking, follows from the FMA policy note: explicit
`__fmaf_rn` in BOTH forms makes them flag-independent and equal to each other. But
that is only worth doing if the result is also compared against the CURRENT scalar
output, because making both forms non-contracting is itself a change unless the scalar
loop turns out not to contract today. That comparison has not been made, so the break
remains unexplained.

**The trap, recorded because it produced a wrong conclusion here.** Partway through,
FSI reported PASS with the non-split kernel vectorised. It was a **stale binary**: the
`.cu` had failed to compile (a cross-namespace forward declaration - the helper lives
in `lsxcommon::qwen` while the non-split kernel is inside an anonymous namespace, so
they are distinct entities and it failed as "more than one instantiation"), the build
stopped, and the previous split-only binary ran instead. Counting `error:` in build
output is not the same as the binary being current; comparing mtime is what finally
separated the two states here.

Reverted; the verified split-only port stands and the tree is green.

## J. gemma FSI/CODE goldens captured; both gates RED on real divergences (2026-10-06)

gemma had **no** FSI or CODE golden, so both stages were silently SKIPping and the family
was effectively held up only by SelfCheck and DET. Now captured from the source
safetensors (they run locally: `Gemma4ForConditionalGeneration`, bf16, transformers
5.15.1), so they are model-defined goldens, not template-derived.

Both gates run, and both fail — for **different** reasons:

- **FSI-TEST-01** — semantic. Source: `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)`, the
  correct annuity-due answer. Converted: `PV=CF*((1-exp(-r*n))/(1-exp(-r)))`, the
  prompt's own template line, missing the trailing `exp(r)`. Clean and correctly
  extracted, so it is a real behavioural difference, not a harness artifact. FSI-02/03 pass.
- **CODE-TEST-02** — syntactic, and the gate working as designed. Source:
  `m=l+((h-l)/2)`; converted: `m=(l+(h-l)/2)`. Algebraically identical; CODE applies no
  canonicalization precisely so that this fails. CODE-01/03 pass.

**Neither is the port.** Reverting `gemma_moe_inference.cu` to its pre-port state at
`7810383` reproduces 5 passed / 1 failed on both gates, byte for byte. Both are
NVFP4-vs-bf16 differences on borderline greedy decisions.

Left red deliberately. Two things are *not* acceptable fixes here: writing the template's
answer into the golden (that deletes the model-defined rule and makes the gate vacuous),
and adding algebraic comparison to CODE (that deletes the reason CODE exists). Both
findings are written up in `docs/FSI_SELFCHECK_FINDINGS.md` and
`docs/CODE_SELFCHECK_FINDINGS.md`.

Worth noting what this costs: gemma was previously *appearing* green on 4 stages while two
of them never executed. A SKIP is not a pass, and a missing golden is indistinguishable
from a passing one in the ctest summary. Every other family's `det` golden exists and
gemma's did not, which is why this gap survived — the absence is now closed.

## R. gemma-4-31b-it (dense) ingested; FSI-01 RED; shared TC partial-M bug (2026-10-09)

The dense 31B member of the Gemma-4 family was ingested (`gemma_dense_*`,
format tag `logestix-gemma-dense`, `lsxforge --gemma4dense`). Real-weight
results: SelfCheck PASS (`Paris.`, bit-exact across stress cycles), CODE 3/3,
DET 3/3, FSI 2/3. Open items:

- **FSI-TEST-01 RED.** Source (golden):
  `PV=2500000*((1-exp(-0.0835*7))/(1-exp(-0.0835)))`; the converted model
  repeats the prompt template `PV=CF*((1-exp(-r*n))/(1-exp(-r)))`. Same
  NVFP4-vs-bf16 borderline class as the 26B (section J); the gate stays red.
- **Shared TC defect: RESOLVED 2026-10-10** - `TcGemmKernel` wrote only
  rows `[0,16)` of its 64-row tile (`wm` was computed and never used); all
  four row computations fixed, TC shape sweep extended to seq 17/18/33/64.
  gemma keeps its scalar prefill **by choice** now, because running it
  through TC moves the first token and breaks the tool round-trip. Full
  trail: [`open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md`](open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md).
- **Perf: 47.3 t/s (perf) / 41.0 t/s (r3)** at 2048 tokens on
  `linux-release-znver4-aocc` (16.5 GiB resident; TTFT 337 ms, scalar prefill).
  Profile-guided decode work took it 40.6 -> 47.5: shuffle RMSNorm
  (11.4 -> 3.5 us/call), tied-head logits through the neutral warp-per-row GEMV
  (308 -> ~670 GB/s), unsplit attention at short context.
- **Rejected experiments (each measured, then reverted):**
  - *CUDA-graph replay of the decode step.* Implemented (capture-safe device
    position + KV-write kernel; `LSX_GEMMA_DENSE_CUDA_GRAPH`), capture
    succeeded, but it is SLOWER: 46.8 vs 53.8 t/s at 1024 and 47.6 -> 46.1 at
    2048. The decode is kernel-bound, not launch-bound, and a captured step
    needs a FIXED split count (the position advances in-graph), which
    over-splits short contexts. Default off. The device-pos/KV-kernel refactor
    it required was reverted with it.
  - *Warp-per-head decode attention.* 43.1 vs 47.6 t/s (only ~1k threads in
    flight). Reverted.
- **Comparison to the other dense model, Qwen3.8-27B** (same release tree,
  same prompt, `--max-gen 2048`): 54.9 t/s, but it is 48/64 *linear-attention*
  (GatedDeltaNet) layers, so its per-token work is far lower, and it reads a
  smaller pinned set; gemma-4-31B is 60 full/sliding-window attention layers
  and ~7% more weight bytes (16.1 vs ~15 GiB). Effective HBM bandwidth is
  comparable (gemma ~818 GB/s vs qwen ~884 GB/s). The ~13% gap is layer-type +
  bytes, not a defect; at `--max-gen 1024` they are at parity (53.8 vs 54.1).

## K. Golden-debt: the declared list, and why it is not a hard failure (2026-10-06)

Vector 2 of the three "will it recur" holes is now closed. `test_model_suite_guard`
carries a new case that pairs every `Suites()` row against `tests/fixtures/{fsi,code,det}`
and is machine-enforced in **both** directions:

- a parity golden that is missing **and not declared** in `kGoldenDebt` fails;
- a `kGoldenDebt` row whose golden now exists fails too, so the list cannot outlive
  the problem and rot;
- a `kGoldenDebt` row matching no live gap fails, so renames do not leave dead entries.

Current state: **25 declared debt entries** across the nine families that have none
(or only some) of the three goldens. That number is the honest coverage figure, and
until now it was only obtainable by diffing two directories by hand.

The design choice worth recording: this is **declared debt, not a hard requirement**.
A hard requirement would paint nine families red on the next fast run and demand
capturing goldens for every one — including models with no local safetensors, which
is hours of work plus vault access we do not have here. Declared debt gets the
property that actually matters (a new family cannot be added quietly without
goldens, and the remaining gap is enumerable from test output) without pretending the
existing debt is smaller than it is. A missing golden stays a SKIP; what changes is
that it can no longer be invisible.

`granite-4.2-30b:code` is the one entry here that is a single stage rather than a
whole family, which is worth noticing: a family can look fully wired across all four
parity stages and still have a hole in one of them.

## O. phi-4 DET-TEST-02 RED after the E4M3FN re-conversion (2026-10-07)

Re-converting phi-4 after the E4M3FN (subnormal) block-scale codec change makes
`test_phi_dense_det_selfcheck` DET-TEST-02 fail: the source golden is
`CATASTROPHIC` (the unique word in `[CATASTROPHIC, CRITICAL, FATAL, TERMINAL]`
that begins and ends with `C`), and the converted model now answers `CRITICAL`
(deterministic 3/3). DET-01/03 and Paris/FSI/CODE still pass.

This is a borderline greedy flip from the offline requantization — the same
class as gemma's CODE-TEST-02 (section J) and alice's CODE-TEST-02 (whose
golden is self-consistency and was re-reconciled). The phi DET golden is
source-parity (`Phi3ForCausalLM` runs locally), so it is **not** reconciled and
the gate stays red deliberately. The same re-conversion recovered granite's
tool round-trip (section N archived) — a codec change moves every model's
borderline decisions; each one has to be judged on whether the new answer is
still the canonical one.

**Measured 2026-10-07 (near-tie confirmed).** Feeding the source phi-4 the
golden DET-02 `input_ids` (HF `Phi3ForCausalLM`, bf16, GPU) gives, at the first
generated token:

| token | decode | logit |
|---|---|---|
| 34 | `C` | 30.75 |
| 9150 | `CR` | 29.625 |
| 29296 | `CAT` | 22.75 |

The source spells `C`+`ATA`+`ST`+`RO`+`PH`+`IC` -> `CATASTROPHIC`; `CR` is the
runner-up at only ~1.1 logits behind (a ~2.3x probability ratio). The converted
model takes the `CR` branch -> `CRITICAL`. So the divergence is a genuine
near-tie on the very first token, flipped by the few-percent NVFP4
requantization — not a systematic degradation. The gate stays red (source
parity); the decision itself is inherent to greedy decoding at this margin.

## P. GLM-5.3-Flash FSI-TEST-02 RED after the E4M3FN re-conversion (2026-10-07)

Re-converting GLM-5.3-Flash after the E4M3FN (subnormal) block-scale codec
change makes `test_glm_moe_flash_fsi_selfcheck` FSI-TEST-02 fail. The prompt
asks for a single-line expression and names the expected template; the
converted model instead deliberates (`Let me analyze this request carefully…`),
echoes the initial state and the template inside its reasoning, and never emits
a clean answer line. The FSI extractor's "first `anchor=…` line" rule then
picks the reasoning's `Balance_Alpha=100000000`. FSI-01/03 and
Paris/CODE/DET stay green.

Not a token budget: the same failure reproduces at 128 (committed golden), 512
(drifts into numerically evaluating the expression) and 2048 (still
deliberating). It is a behavioural divergence from the offline requantization
of a borderline compliance prompt — the same family as §J (gemma) and §O (phi).
The golden is template-derived (the `glm5_next` source cannot run locally), so
there is no source answer to compare against; leaving it red rather than
rewriting the extractor/golden keeps the DSA-path guard honest.

**Next step if revisited:** run the FSI-02 prompt through the pre-E4M3
conversion of the same tree (or the `glm5_next` source if it ever becomes
loadable) to confirm the pre-codec tree answered and the flip is purely the
offline encoder.

## Q. DeepSeek-V4.1-Flash FSI/CODE goldens committed model-defined; both RED (2026-10-07)

The DeepSeek-V4.1-Flash source ships its own FP8/FP4 tilelang tensor-parallel
stack (target 2×80 GB) and cannot run locally, so FSI/CODE had always SKIPped.
Following the GLM-5.3-Flash method the model itself supplies the reference: the
DeepSeek-V4.1-Flash assistant answered the prompts (answers = the
`Expected Templates`) and those answers plus `encoding/encoding.py` thinking
framing are committed as the goldens
(`tests/fixtures/{fsi,code}/deepseek-v4.1-flash.json`). Both tests now RUN
instead of SKIP (a deliberate deviation from "commit only if the converted model
passes").

Both are RED on the E4M3FN NVFP4 conversion (`kOff`):

- **FSI 1/3** — FSI-01: the thinking CoT does not finish within the 512-token
  budget, so the first-anchor-line extractor latches onto the truncated prompt
  echo `PV=CF*(1-e`; FSI-02 yields the initial-state echo
  `Balance_Alpha=100000000`; FSI-03 passes.
- **CODE 2/3** — CODE-01: the model's CoT declares the balanced template
  `{"root":{"nodes":{"leaf":true}}}` "invalid", miscounts the braces, and emits
  an object missing its final `}` (extractor empty); CODE-02/03 pass.

This is the `numeric-contract.md` §6 class (a reasoning/requantization divergence
below the format error), not a discrete kernel defect — the same profile as the
2026-09-27 sweep, unchanged by the E4M3FN codec.

**Budget experiment (2026-10-07):** raising the think-CoT budget from 512 to 2048
does **not** rescue either gate, and was reverted. FSI is budget-independent: the
FSI-01 decoded text is byte-identical at 512 and 2048 (the model echoes the
prompt and then emits nothing useful), so FSI-01/02 are not truncation artifacts.
CODE is *sensitive* to the budget under `ignore_eos`: at 2048 the model
degenerates past its answer and the last-`</think>` extractor flips CODE-03 from
PASS to a malformed `\b[0-9A-F]{4}b` (CODE-01 stays empty). The goldens keep
`max_new_tokens=512`, the value that captures the answer before the degeneration.

**Next step:** obtain a hardware-faithful reference from the release's own
`inference/` stack on 2×80 GB; if any case is a genuine converted-model
regression rather than a budget/extractor artifact, raise the precision of the
responsible tensor and re-convert.

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

### E1a. Cooperative-cancel is pinned end-to-end for 5 of 18 families
- `GenerationStream::Aborted()` has unit coverage (`tests/test_generation_stream.cpp`,
  `tests/test_model_engine.cpp`), and five families now assert that a raised
  flag stops a real decode: alice, mimo (the two that reported the bug), kimi,
  qwen3.8-27b and qwen3.5-35b-a3b. The other 13 route the same
  `GenerationStream::Aborted()` call but no test asserts it — a family could
  check the flag in the wrong place and stay green.
- Measured 2026-10-02 (TDD coverage audit), re-measured 2026-10-03 after the
  qwen_moe work. Streaming-order assertions cover 18/18; cancel coverage 5/18.
- The shared helper the audit asked for now exists in
  `tests/test_streaming_assert.h`: `ExpectPreSetAbortStops` and
  `ExpectCallbackAbortStops` (the latter raises the flag *from the token
  callback*, i.e. the real `--serve` path — the pre-set variant alone cannot
  distinguish "checked between steps" from "checked once at entry"). Each use
  needs a control that runs the same budget with no flag and gets the full
  token count, otherwise EOS truncation on random weights makes it pass for the
  wrong reason.
- **Next step:** call those two helpers from each remaining family's model-stage
  target. Mechanical, ~15 lines per family.

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

## S. Attention layer neutral; only the KV session keeps a qwen name (2026-10-10, updated)

- **What landed (Phase 5, 2026-10-11):** `qwen_inference.{h,cu}` is now
  `lsx_attn_inference.{h,cu}` in `lsxcommon::attn`, and the neutrality guard
  enforces it. `QwenLayerWeights` became the neutral `attn::LayerWeights`
  (`lsx_inference_load.h`); the kernels take a neutral `attn::AttnGeometry`
  instead of the `QwenTopology` passport (they read exactly 12 of its scalars
  and none of its vectors/strings/vram). Two pure-passthrough GEMM shims were
  deleted outright. Bit-identical on real weights; throughput unchanged
  (ornith 167.3 vs 167.06 t/s baseline at 2048).
- **What is still qwen-named on purpose (Phase 5b):** `QwenKvCache`,
  `QwenGenerationResult` and the shared fast-math aliases
  (`QwenTcFastMath`/`SetQwenTcFastMath`/`QwenSeq1Gemv`/`QwenGemmLarge`/
  `QwenFastMathScope`). Three families hold a `QwenKvCache`, so renaming it
  into a qwen-owned header would make Ornith depend on another family --
  worse than today, for a cosmetic gain. `QwenKvCache::Allocate` also still
  takes `QwenPinnedWeights`.
- **Next step (5b):** change `QwenKvCache::Allocate`/`EnsureCapacity` to take
  scalars rather than the pinned-weight bundle, then the cache and result
  structs can go neutral too. Separate change; nothing blocks on it.

## T. Mistral-Small-4-119B-2603-NVFP4 ingestion — later stages (2026-10-11)

- **Branch** `feat/mistral-small-4-119b`; design
  `docs/superpowers/specs/2026-10-11-mistral-small-4-119b-ingestion-design.md`.
- **Landed (verified, green):** stages 0–5 + 4b. Stage 1 converter
  (`mistral_moe_convert.{h,cpp}`, `lsxforge --mistral`, synthetic `convert`
  test); stage 3 loader (`mistral_moe_load.{h,cpp}`, synthetic loader test);
  stage 4b tool codec (`mistral_moe_tool_format.{h,cpp}`, framing test); stage 5
  MLA+MoE decode (`mistral_moe_inference.{h,cu}` + `mistral_moe_model.{h,cpp}`,
  registered `mistral-small-4-119b`), kernels + model tests green. Frozen
  passport `mistral_moe_constants.h`. Four guard registries extended
  (`test_model_independence_guard` `kPassports`/`consumers`/`kLifecycle`,
  `test_model_suite_guard` `Suites()` + tool verdict + golden debt,
  `test_perf_baseline_logic` mid list, `test_code_naming_guard` family token).
  Fast suite 93/93; `test_mistral_*` (convert/loader/tool/kernels/model) green.
- **Real-weight conversion: done.** `lsxforge --mistral` converted the vault
  tree in place (`<model>/logestix/`, 66 GiB, 4608 experts, 222 vision tensors
  stripped, tokenizer arrows written) in ~75 s. The loader then loads it
  (70 s: dense BF16 graph 4.7 GiB + 56 GiB pinned expert store). Two real bugs
  were fixed on the way:
  1. `weight_global_scale` is a **divisor** (`w = nibble*scale/gs`), not the
     repo's multiply — the converter now stores `1/weight_global_scale`.
  2. The neutral `RunMoERouterTopK` seq==1 dots path produced a NaN logit on
     finite real inputs (host f64 of the same data is finite, max 7.97), which
     fed a NaN through the MoE. The family now routes on the **host**
     (`gemv::RunLxGemmVec` BF16 + softmax/top-K in `mistral_moe_inference.cu`).
     This may be a real defect in the neutral seq==1 router path — worth a
     separate look (it is not exercised here by any other family at E=128).
- **Real-weight decode: fixed (2026-10-12).** `test_mistral_moe_selfcheck`
  passes (Paris + tool round-trip + 3 bit-exact stress cycles); `--bench`
  completes 2048 tokens at **14.68 t/s** (Release, `r3` and `perf` both;
  the `linux-debug` preset is 45 t/s but a Debug binary is not a valid perf
  record — `test_perf_baseline` SKIPs outside `NDEBUG`); SelfCheck passes in **both** the auto
  and `--bit-exact` (`FastMathMode::kOff`) modes. The layer-4 "outlier" that
  this entry previously reported was a **measurement artefact** — the expert is
  a genuine massive-activation expert (host reference at layer 4 confirms
  routed max 411.9 vs the 403.8 the GPU produced). Five real defects, each
  isolated against a host NVFP4 reference built from the source safetensors:
  1. **Shared expert through the BF16 GEMV.** `attn::RunAttnMLP` reinterpreted
     the shared expert's E2M1 nibbles as BF16: shared rms 0.0080 vs the host
     reference 0.3495 (**44x** under-contribution). Fixed by running the
     neutral NVFP4 seq==1 GEMV (`MistralSharedExpert`, the Xing pattern —
     `RunMoESharedCombine` is gated and does not apply, this model's shared
     expert is ungated). Regression test
     `test_mistral_moe_kernels` "matches the host NVFP4 reference" (mutation
     checked: reverting the scale makes it fail).
  2. **Autodetect name mismatch.** The converter wrote
     `"model": "Mistral-Small-4-119B-2603"`, which `detect_model_type`
     normalizes to `mistral-small-4-119b-2603` — not the registered kType, so
     `lsxbenchmark`/`lsxfabric` could not resolve the model by path at all.
  3. **Expert-cache VRAM over-commit.** A hand-rolled budget with a 2 GiB
     reserve resolved 1838 slots (~25 GiB) and the 2048-token bench OOMed on
     the KV arena. Now the shared `lifecycle::ExpertCacheSlots` (8 GiB
     calibrated reserve + the `expert_slots_pct` knob) → 1382 slots.
  4. **FP16 SwiGLU overflow (the actual NaN).** This model carries massive
     activations: layer 32 expert 67 reaches `|silu(g)*u| = 2.5e5` against the
     FP16 ceiling of 65504 — exactly 4 values overflow, and the inf propagated
     through the down projection into the residual. The scalar (non-TC) MoE
     path stores that product in FP16. `RunMoERoutedFFN` and
     `MistralSharedExpert` now evaluate the product in FP32 and store it under
     a power-of-two prescale, folded back in the FP32 accumulator
     (`SwigluF32Kernel`/`SwigluAbsMaxKernel`/`SwigluAlphaKernel`/
     `SwigluStoreScaledKernel` in `lsx_kernels_act.cuh`, plus
     `moe_coeff_add_scaled_kernel` / `acc_add_half_scaled_kernel`). It is a
     **provable no-op for every existing model** (alpha == 1 whenever the
     product fits FP16, and the value expression is the same F32 silu). The TC
     path was already immune — it quantizes from FP32 — which is why only
     `--bit-exact` diverged. Regression test: "survives an FP16-overflowing
     SwiGLU" (mutation checked).
  5. **FP16 residual + session leak.** The residual itself reaches ~2.4e5 at
     layer 32 (host reference), past FP16 range, so `d_h` is now FP32 (the
     norms and the `d_post` hand-off are unchanged). And `~MistralSession` now
     frees — without it every `Infer` leaked the dense graph + expert cache and
     the *second* request OOMed. (`FreeMistralSession` resets the pointers in
     place; `s = MistralSession{}` recursed infinitely once the destructor
     owned teardown.)
- **CLOSED — the "neutral router NaN" was a mis-attribution, not a defect
  (2026-10-12).** This entry previously recorded
  `RunMoERouterTopK`'s seq==1 path as returning a NaN logit on finite real
  inputs. It does not. The NaN was the shared-expert FP16 overflow propagating
  into the *next* layer's `d_post`, which is the router's input — i.e. a
  downstream symptom of defect 1, measured before that defect was found. With
  the shared expert fixed, the neutral device router runs this family on real
  weights at its real geometry (E=128, H=4096, top-4) in **both** numeric
  modes; the host-side router workaround has been removed.
  The investigation did surface a **real coverage gap**: the only router
  reference test covered `seq=5`, i.e. the *prefill* kernel. The seq==1 decode
  entry (`moe_router_dots_kernel` + `moe_router_select_kernel`) had no
  reference comparison at all, which is why a wrong answer there would have
  been invisible. Now pinned by "router decode-split (seq==1) matches host
  replica" (`test_qwen_moe_kernels`), sweeping E=128/H=4096, E=128/H=2048,
  E=60 (grid not a multiple of the 128-thread block), E=16, E=8 and H=132
  (H not a multiple of 8, the scalar dot path). That test passes unmodified on
  the current neutral code, which is the evidence that the kernel was never
  the problem.
- **FOUND AND FIXED — the bit-exact MoE path multiplied every layer by alpha
  (2026-10-12).** Chasing the golden gates turned up a real defect in the
  overflow-proof SwiGLU above: it divides the activation by `alpha` once, so the
  down projection returns `out / alpha`, but the FP32 accumulator folded back
  `alpha**2`. Every layer's MoE output was therefore `alpha`=16x too large at
  Mistral's layer 32, compounding down the stack.
  **Why nothing caught it:** the Paris SelfCheck still passed in both modes (a
  memorised completion survives large numeric error), the bench still ran, and
  the new shared-expert regression test asserted `cos > 0.99` — **cosine is
  invariant under a uniform scale, so it returned exactly 1.0 for a 16x-wrong
  answer.** Mutation testing had verified that the *prescale* was present, not
  that its factor was right.
  The tell was end-to-end, not numeric: on "Repeat exactly this text: ABCDEFG"
  bit-exact mode returned `The output output output` while the auto (TC) mode
  returned `HIJKLMNOPQRSTUVWXYZ`. Host-oracle confirmation at layer 32: the GPU
  accumulator was 16.0x the reference with `corr = 0.999997`, i.e. right shape,
  wrong scale. After the one-line fix (`coeff * alpha`, not `coeff * alpha*alpha`)
  both modes agree, and the regression test now pins `|got|max / |ref|max` to
  within 5% so a scale error cannot pass again.
- **FIXED — the tensor-core routed FFN overflowed its SwiGLU activation to
  FP16 inf, costing layer 32 ~52x. Root cause, fix and instrument below.**
  The tensor-core branch of `RunMoERoutedFFN` fused SwiGLU into the NVFP4
  activation quantizer (`RunAttnSwigluPreciseQuantInternal`). That kernel does
  `const __half hv = __float2half(sig * g * u);` — it rounds the product to
  **FP16 before quantizing**, and FP16 tops out at 65504. Mistral-Small-4 layer
  32 expert 67 reaches `|silu(g)*u| = 2.5e5` from entirely ordinary FP16
  projections (max 444 and 568), so the store overflowed to `+inf`, the block
  `absmax` became `inf`, `FloatToE4m3(inf)` saturated the block scale, and every
  nibble in the tile quantized from an infinite value. The scalar path was
  immune because it keeps the product in FP32 under a power-of-two prescale.
  Measured on real weights (Paris, layer 32, routed accumulator vs a host
  oracle over the source safetensors):

  | mode | before | after |
  |---|---|---|
  | `kOff` | ratio 0.9998, corr 1.000000 | unchanged (ratio 0.9998) |
  | `kAuto` (TC) | **ratio 0.0192**, corr 0.975 | **ratio 0.9932, corr 1.000000** |

  Fix: `RunAttnSwigluPreScaledQuantInternal` keeps the product in FP32, reduces a
  global `absmax` with an atomic, prescales by the smallest power of two that
  brings it inside both the FP16 and the E4M3-block-scale span, and publishes
  `alpha` in a second slot so `moe_coeff_add_scaled_kernel` folds one reciprocal
  factor into the FP32 accumulator. Two launches instead of one, no host
  synchronization (`alpha` is device-only while `RunLsxTcGemmPreQuant` takes a
  host float). `alpha == 1` is a bit-exact no-op, so every other family is
  unchanged — pinned by a dedicated bit-identity test, and re-verified on real
  weights for Qwen3.6-35B-A3B, Gemma-4-26B-A4B, Xing4.0-29B-A4B and
  Ornith-1.5-35B-A3B, each in both numeric classes. (GLM-5.3-Flash is present
  only as an unconverted HF checkpoint, so it could not be run here.)
  **Why it survived a green suite:** Paris passes in both modes and the bench
  runs, so the suite was green over a layer whose accumulator was off by 52x.
  Nothing compared a *layer accumulator* against an oracle — only final text.
  That gap is now covered by `test_qwen_nvfp4_gemv`: one case forces a
  Mistral-shaped massive activation and asserts the recovered magnitude, one
  asserts bit-identity in the normal range, one asserts a seq==1 projection is
  invariant to a power-of-two activation prescale. Both of the first two fail
  against the old kernel.

  **Cost of the fix, accepted deliberately:** the quantizer is now two launches
  instead of one, so the TC branch pays one extra launch per slot per layer —
  4 slots x 48 layers = 192 extra launches per token, about 1 ms against a
  ~22 ms token. Measured in Release at the canonical 2048-token protocol:
  `perf` (kAuto) 47.13 -> **45.02 t/s (-4.5%)**, re-recorded in
  `perf_baselines.json` at commit `cb60207`; `r3` (kOff) is unaffected at 14.68
  because `kOff` forces `tc=false` and never enters this branch. The guard
  flagged the drop as DEGRADED and refused to move the reference on its own,
  which is the intended behaviour — the cause was identified (launch overhead,
  not bandwidth: the FP32 round-trip is 8 B/element against multi-MB expert
  tiles) and only then was the number changed by hand. A 4.5% decode cost is a
  fair price for a 52x error, but it is a standing cost, so a cheaper shape is
  available if it ever matters: hoist the reduction out of the per-slot loop
  (2 launches per layer instead of 8) by quantizing all K slots in one pass.
  Not done — it is a restructuring of `RunMoERoutedFFN` for 4.5%, which does not
  justify the regression risk across the five families that share this path.

  Two earlier diagnoses of this entry were wrong and are retracted here, since
  the record is the point: (1) "the E4M3 block scale saturates, attenuating the
  expert 94x" — the scale does saturate, but only *because* the FP16 store
  already went to inf; and (2) "Mistral never takes the `tc_fused` branch, so
  the default path is unaffected" — false, because `qwen::SetQwenTcFastMath` is
  an inline alias for `lsx_nvfp4::SetLsxTcFastMath`, the very global the
  engine-init perf policy sets (`gpu_perf.cpp`), so `kAuto` leaves it `true` and
  `kOff` forces it `false`. Checking whether the *model class* calls the setter
  says nothing about the value in force. `LSX_PERF_AUTO=0` (which pins the
  legacy bit-exact policy, `tc=false`) is the cheap one-flag bisect for "is this
  the fast-math path?", and it puts layer 32 back at ratio 0.9998.

- **Still open — the three FSI/CODE/DET golden gates, with the capture now
  understood.** (The old heading said "the four S* golden gates"; there is no
  S1..S4 scheme anywhere in this file, the body and
  `test_model_suite_guard.cpp` both enumerate exactly these three, and the body
  already said "three permanently-red gates" -- the count in the heading was
  the only stale part.)
  Capture tooling improved first: `tools/fsi_capture.py` gained `--template-only`
  (the source is unrunnable), and `det`/`code` gained a `chat_template` override
  in `OVERRIDES`. Findings:
  1. **The source cannot run.** `AutoConfig.from_pretrained` fails — the
     checkpoint has `params.json`, no `config.json`, no `model_type`, and
     NVFP4-packed experts. So a template-derived golden is the only option, and
     it cannot establish *parity*; it can only assert the converted path's own
     answers.
  2. **The vendor default system prompt suppresses every answer gate.** It
     instructs "when the user's request requires up-to-date or specific data, you
     must use the available tools". With it, all nine prompts answer "It seems
     like you're asking about a specific date, you can use tools to fetch ...".
     The model is *complying*; the engine's own `RenderText` does not inject
     that turn, so capturing with a `chat_template` override (BOS + `[INST]`...,
     no system turn) is what matches what we run.
  3. **Re-framed, the converted model is coherent but off-format, and the two
     numeric modes disagree on these prompts.** DET determinism PASSES
     (`runs=3 deterministic=true`, byte-identical) and answers are on-topic —
     DET-02 correctly reasons "The word that starts with the character 'C' and
     ends with the character 'C'" toward `CATASTROPHIC` — but no prompt emits the
     canonical token/formula, several collapse into repetition loops, and the
     auto and bit-exact modes diverge (auto returns `.`, bit-exact returns a
     full sentence) on DET-01.
  **Decision: the goldens are NOT committed.** Shipping three permanently-red
  gates trains people to ignore red, and with the source unrunnable a
  template-derived golden asserts nothing about parity that the Paris SelfCheck
  does not already cover more cheaply. FSI/CODE/DET remain declared debt in
`test_model_suite_guard`. Closing them needs either a runnable source
   (vendor stack) or a documented answer-format contract from the operator.

## U. granite-4.2-30b tool round-trip RED again at `31998bf` (found 2026-10-13)

`test_granite_dense_selfcheck`'s second test case ("end-to-end tool
round-trip", real weights) fails at HEAD:

```
CHECK( calls[0].name == "get_weather" )   with expansion: "get_weather.x" == "get_weather"
```

The model emits `get_weather.x` — it puts the `.x` into its **own reasoning
text** first ("We have tool: get_weather.x, …"), then into
`<function>get_weather.x</function>` and into the bare
`{"name": "get_weather.x", …}` object. So the parse path is not at fault: the
divergence is in the greedy generation. The first test case is unaffected
(`Paris`, `top1_tok=12366`, `logits_hash=3d1604aa75c99a1f` stable over 3
cycles), and FSI/CODE/DET are green.

**Not an Arrow regression.** Found while bumping Arrow 25.0.1 -> 26.0.0, so it
was attributed with a control: a second tree at the *same* commit (`31998bf`)
built against **Arrow 25.0.1** produces the identical `get_weather.x` **and the
identical `logits_hash=3d1604aa75c99a1f`** — the Arrow bump is bit-identical
on this path.

**Where the suspicion points instead:** the granite tool round-trip was
"recovered" on 2026-10-07
(`open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md`, archive row
L/M/N), and 9 commits have landed in `libs/lsxcommon` since — including three
that touch the shared attention layer granite decodes through (neutral-layer
split `2b4b41c`/`65db1c7`, `00e22ec` TC mxf4nvf4 GEMM tile fix, `f99008b`
`__host__ __device__` KV-geometry helpers). Which of them moved this token is
**not yet established** — the missing instrument is a generation-level oracle
(the token ids of the 256-token round-trip, plus the logit margin at the flip
step), not another pass/fail on the same assertion.

## V. Routed-expert host storage: the arena is retired; two follow-ups remain (2026-10-14)

**The migration itself is closed** — the arena is gone for qwen MoE / Ornith /
Mistral / qwen3.8-flash-next, with measurements, gates and the guard test in
[`open-tasks/2026-10-14-expert-store-arena-retired.md`](open-tasks/2026-10-14-expert-store-arena-retired.md).
Headline: qwen3.6 17.5 → **7.0 GB** at unchanged 141 t/s; qwen3.8 63.6 →
**10.9 GB** with load 73.7 → 9.9 s. What is left is not the storage change but
two consequences it exposed:

### V1. RESOLVED (negative): the arena is required when the working set is the whole expert set

Tried the bounded pinned window §V1 called for, measured it, and it does not
work. qwen3.8-flash-next, real weights, `/usr/bin/time -v`, same prompt:

| configuration | peak RSS | throughput |
|---|---|---|
| pinned arena (original) | 63.6 GB | 32.1 t/s |
| views, cap 4096 | 10.9 GB | 9.2 t/s |
| views + pinned window, cap 2048 | 10.9 GB | **9.2 t/s — no speedup** |
| views + pinned window, cap 24576 | 127.2 GB | 29.0 t/s |

The window pins 5400 MiB in 754 ms and buys nothing, because the decode working
set is LARGER than the cap: nearly every touch is an expert that was evicted
and re-mapped, hence unpinned. Speed only returns when essentially everything is
pinned — and then the mmap views are still mapped beside the pinned copy, so it
costs 2x the arena (127 vs 63.6 GB) for slightly less speed. For a model whose
decode working set is the whole expert set the arena is the right design, and
the migration is a net loss. This is a resource constraint, not a defect.

`cudaHostRegister` over the mmap views is not a way back: refused for every
entry (`registered=0 failed=512`), because a file-backed MAP_SHARED mapping is
not registrable. Three sticky-CUDA-error traps on the way there, each of which
presents as "generation dies right after a successful load" with nothing in the
message pointing at pinning: (1) a non-page-aligned start is rejected; (2) the
six payload spans of one expert are adjacent in the file, so aligning each down
to a page makes neighbours overlap and the second registration returns
`cudaErrorHostMemoryAlreadyRegistered`; (3) unregistering a range that was never
registered also raises a sticky error.

qwen3.8 is therefore back on its arena (63.6 GB, 32.0 t/s re-measured). The
neutral store stays for qwen MoE / Ornith / Mistral, whose working set fits the
cap: 17.5 → 7.0 GB at unchanged throughput.

### V3. `test_qwen_moe_kernels` / `test_ornith_moe_kernels`: the batched TC chain and the legacy chain disagree (found 2026-10-14, after bad0792)

Both cases ("tc batch routed ffn matches legacy chain bitwise") compare
`RunMoERoutedFFN` (sequential, per-route, asks for residency at call time)
against `RunMoERoutedTcBatch` (composes its slot table from the residency
mirror). They disagree qualitatively: 64 of 64 elements differ, and the batched
result sits at -4547.5 where the sequential one is -1.27e6. Not an
accumulation-order reassociation.

Chasing it fixed two real defects on the way (`bad0792`): the residency mirror
was never invalidated on device-slot eviction, so it named slots that had moved
to another expert — a silent wrong-expert read in every streamed model; and both
cases warmed 32 experts into 8 slots while asserting a full-residency
precondition that fewer slots can never satisfy.

Ruled out since: the storage representation (both chains reach the same staged
weights), mirror staleness, the TC fast-math flag (the batched output is
identical with it on and off), slot capacity, and the degenerate geometry
(I=16 fails the same way a realistic I=128 does).

**Localised, then identified: `RunLsxTcGemmBatch13` writes ZEROS where the
other two implementations agree exactly.** Three implementations were run on
the same expert, same NVFP4 activations, same staged weights (K=1, one route,
eid 3, coeff 1):

* the per-slot TC GEMM (`RunMoERoutedFFN` with xq/xqs),
* the fp16 V-batch core (the same entry point with xq/xqs nulled),
* the batched TC GEMM (`RunMoERoutedTcBatch` -> `RunLsxTcGemmBatch13`).

Measured against the fp16 core as reference:

| shape | relerr, per-slot TC | relerr, batched TC |
|---|---|---|
| H=64, I=16 (N=32) | **0** | **1** |
| H=64, I=32 (N=64 = kTileN) | **0** | **1** |

relerr 1.0 is exactly the error of a zero vector: the batched GEMM writes
nothing, while the other two agree bit-for-bit and are non-zero. The N-tile
hypothesis (N must be a multiple of kTileN=64) was tested and is **wrong** — a
full 64-wide tile fails the same way.

Earlier narrowing that still holds: the divergence is in the w13 output, before
swiglu and before the reduce (gate slot 0 differs 16/16, up half identical
zeros in both), so batching, the reduce and coefficient handling are all
excluded. It is also not eviction, slot capacity or the TC fast-math flag.

**The production path IS live, and it changes real generations.** `tc_batch` is
enabled when TC fast-math is on (the default perf policy), the segment scales
exist, and neither LSX_GEMV_EXPERTS nor LSX_TC_BATCH_OFF is set -- so qwen MoE
and ornith decode through `RunMoERoutedTcBatch` and the real-weight gates pass
through it. Measured on Qwen3.6-35B-A3B with `lsxfabric --testprompt`, after a
determinism control (two runs of one configuration produce identical text, so
the harness is not the source of variation):

| configuration | output |
|---|---|
| default (batched TC on), run 1 | "Paris is a major tourist destination, attracting millions of visitors each" |
| default, run 2 | identical |
| `LSX_TC_BATCH_OFF=1` (legacy per-slot), run 1 | "Paris is located in the north-central part of France, along the Seine River..." |
| `LSX_TC_BATCH_OFF=1`, run 2 | identical |

So the two paths disagree **deterministically on real weights**, not only on the
synthetic case. That makes the red test a true alarm rather than a fixture
artifact, and it matches the claim in the pipeline comment ("bit-identical to
the legacy per-slot chain, pinned by test_qwen_moe_kernels") being false in
production.

Which path is faithful is still open, and the parity gates cut both ways:
FSI/CODE/DET are captured from the source safetensors (an external oracle) and
pass WITH the batched path on, so it is adequate but demonstrably not bit-
identical; DET only checks run-to-run stability within one configuration, so a
systematically different path passes it. Deciding needs an oracle finer than the
canonical answer -- e.g. a logits-hash comparison against the fp16 core, which
the per-slot TC path already matches exactly on synthetic shapes.

**The missing instrument**, stated after the above: a comparison finer than the
canonical answer. The goldens prove the batched path is adequate, the red test
and the real-weight A/B prove it is not identical to the legacy path, and only a
logits-level oracle (or the fp16 core as reference) can say which one is
faithful. Until that exists the batched TC MoE path should be treated as
unverified in production, and LSX_TC_BATCH_OFF=1 is the known-good fallback. The real models are unaffected — qwen MoE and ornith selfcheck, FSI, CODE
and DET pass on real weights, and those paths do not run this comparison.

### V2. Not reproduced — `max_open` does bound RSS, and the geometry pass does not fault payloads

This was written from a single measurement (Mistral 65.6 GB at cap 512, 1024 and
4096) and it does **not** hold up. Re-measured with the same prompt, twice each:

| `max_open` | peak RSS |
|---|---|
| 64 | 3.2 GB |
| 512 | 7.0 GB / 7.0 GB |
| 8192 (≥ all 4608 experts) | 31.1 GB / 31.2 GB |

So RSS is monotone in the cap and saturates at the whole tree, exactly as
designed; the geometry pass reads each expert once through `DecodeEntry`, which
takes Arrow **views** (`GetView(0)` = pointer + size) and never touches a payload
page. Confirmed from the other side: with generation disabled by a failing run,
the same load peaked at 1.8 GB — the 10.9 GB at cap 4096 is paid during decode,
not during load. The earlier 65.6 GB figure was an artifact; there is no
page-cache defect here and no metadata-in-`model.index.json` work is needed.

## Archive
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
| 2026-10-14 | Routed-expert host storage: the arena retired for qwen MoE / Ornith / Mistral | `MoEExpertRecord` reduced to immutable geometry, payload moved to the neutral `ExpertStore`, six-segment staging from mmap views; qwen3.6 17.5 → 7.0 GB at unchanged 141 t/s, qwen3.8 was tried on the neutral store and REVERTED to its arena — the decode working set is the whole expert set, so a bounded pinned window gives no speedup (9.2 t/s) and pinning everything costs 127 GB against the arena's 63.6 GB; `test_expert_store_guard` locks what did migrate - [`open-tasks/2026-10-14-expert-store-arena-retired.md`](open-tasks/2026-10-14-expert-store-arena-retired.md) |
| 2026-10-10 | Neutral routed-MoE layer + ornith de-duplication + the shared TC M-tile defect | `TcGemmKernel` wrote only rows `[0,16)` of its 64-row M tile (`wm` computed, never used) - fixed in all four row computations, TC shape sweep extended to seq 17/18/33/64; `ornith_moe_inference.cu` (2182 lines, 97% copy, already drifted) deleted in favour of the neutral `lsx_moe_inference` layer, expert-store loader cluster de-duplicated across qwen/glm-flash/ornith (-2776 lines); Ornith fused-expert split and gemma `final_logits` now read back by tests - [`open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md`](open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md) |
| Inference-conformance review (9 deviations) | all fixed in dedicated commits |
| "Closed in this session" list | 15 items, all closed |
| D — KV cache layout `[head][pos][dim]` | closed 2026-10-05, **negative**: implemented, measured, rejected. Head-major reads K/V 6-10% faster in isolation and is 2-6% slower end to end, so pos-major ships; the reusable stride layer, the neutral layout knob, a bit-exact parity gate and a streaming-read ceiling probe all shipped. A related `kAttnSplits = 4` finding (+10% at ctx 8191) was rejected on bit-identity. Two pre-existing defects fixed: a test iteration that verified nothing, and a wrong host pitch in the KV readback |
| L/M/N — NVFP4 E4M3FN codec, Nemotron-NVFP4 ingestion, granite tool round-trip | closed 2026-10-07 — [`open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md`](open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md) |
| Ornith-1.5-35B-A3B-MLX — prompt-framing defect (`BuildPromptInputIds` dropped the required leading system turn) | found + fixed 2026-10-07 — [`open-tasks/2026-10-07-ornith-framing-closed.md`](open-tasks/2026-10-07-ornith-framing-closed.md) |

Full text: [`open-tasks/2026-09-30-cuda-review-windows-bringup.md`](open-tasks/2026-09-30-cuda-review-windows-bringup.md)
and [`open-tasks/2026-10-02-mimo-closed.md`](open-tasks/2026-10-02-mimo-closed.md).

Added 2026-10-05:
[`open-tasks/2026-10-05-xing-kv-layout-closed.md`](open-tasks/2026-10-05-xing-kv-layout-closed.md)
— KV cache layout, closed negative.

Added 2026-10-07:
[`open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md`](open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md)
— the NVFP4 block-scale codec is now the complete E4M3FN table (mantissa +
subnormals, bit-identical to the NVIDIA source); the
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 ingestion is done and its real-weight
SelfCheck passes (router-load dtype fix); the granite-4.2-30b tool round-trip is
recovered.

Added 2026-10-07:
[`open-tasks/2026-10-07-ornith-framing-closed.md`](open-tasks/2026-10-07-ornith-framing-closed.md)
— converting Ornith-1.5-35B-A3B-MLX surfaced a real prompt-framing defect: the
hand-rolled `BuildPromptInputIds` omitted the leading system turn this reasoning
fine-tune requires, so generation (Paris selfcheck, `--run`, `--testprompt`)
collapsed into repetition. The frame now goes through the native Qwen renderer
(mirrors Nemotron), the conversation path injects a default system turn, and the
real-weight selfcheck asserts the frame shape. FSI/CODE/DET goldens remain
declared debt.

## Maintenance rule

When an item in this file is resolved, move it to `docs/open-tasks/<date>-<slug>.md`
(one file per work session, dated the day the work happened), replace it here
with a one-line row in the Archive table above, and keep only genuinely open
work in `docs/OPEN_TASKS.md`. The index is
[`open-tasks/README.md`](open-tasks/README.md). This keeps the live file short
enough to read at a glance, which is its only purpose.