# Doc hygiene and the non-defect records (2026-10-23)

Moved out of `docs/OPEN_TASKS.md` because none of these is open work any more:
each is either a resolved mechanism, a measurement that answered its question,
or a deliberate decision not to change something. They are kept because their
method notes are what stops the same question being asked again.

| item | verdict |
|---|---|
| B (hardware-dependent budgets) | resolved 2026-10-02: budgets are GPU-keyed, and the two red ones were machine, not code |
| D1 (nemotron batched expert GEMV) | closed as not-a-regression: 1.34-1.44x in isolation, no pipeline cost |
| F (decode ramp) | no ramp on Xing; throughput *falls* 27%/42% across a session |
| G (attention score dot vectorisation) | ledger closed: qwen +4.8%, gemma +3.0%, gigachat 0%; the non-split half was closed the same day as a half-row bound bug (see `2026-10-23-non-split-attn-dot-closed.md`) |
| E (lifecycle design decisions) | every subsection closed; only pointers remain below |
| V2 (`max_open` RSS) | not reproduced: the cap bounds RSS as designed, no page-cache defect |

`§V`'s migration (the arena retirement) was already archived in
[`2026-10-14-expert-store-arena-retired.md`](2026-10-14-expert-store-arena-retired.md)
and is not repeated here; what was live in `OPEN_TASKS.md` was only the V2
claim that a follow-up remained, which is refuted below.

---

## B. Hardware-dependent budgets (resolved 2026-10-02)

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

---

## D1. nemotron batched expert GEMV (closed 2026-10-02, not-a-regression)

#### D1. nemotron batched expert GEMV reaches ~1.4x, not the 1.67x expected
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

---

## F. Decode throughput ramp (2026-10-05)

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

---

## G. Attention score dot products (2026-10-05)

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

---

## E. Design decisions deferred (all closed)

The lifecycle review closed every mechanical gap (streaming/abort, residency
lock, governor leases, context teardown, result fill, error-path cleanup) and
added `kLifecycle` to `tests/test_model_independence_guard.cpp` so they
cannot regress. Four items were deliberately NOT changed — each needs a
product or numerics decision, not a patch.

§E1a (cooperative cancel), §E1b (the residency lock) and §E1c (qwen3.8's bulk
drain) were closed on 2026-10-17 and moved to
[`open-tasks/2026-10-17-lifecycle-abort-contract-closed.md`](2026-10-17-lifecycle-abort-contract-closed.md).
§E3, the last purely technical item in this cluster (mimo's host-side BF16→FP16
upload), was fixed the same day and moved to
[`open-tasks/2026-10-17-mimo-host-bf16-upload-closed.md`](2026-10-17-mimo-host-bf16-upload-closed.md).
§E1 (sequential-dispatch guard) and §E2 (alice logits row) were closed on
2026-10-17 and moved to
[`open-tasks/2026-10-17-lifecycle-e1-e2-closed.md`](2026-10-17-lifecycle-e1-e2-closed.md).
§E4 (opt-in sampling) was implemented on 2026-10-23 and moved to
[`open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`](2026-10-23-sampling-neutral-session-and-environment-oracle.md).
§E1d, found while closing §E1c, was fixed the same day and moved to
[`open-tasks/2026-10-17-qwen38-instance-state-closed.md`](2026-10-17-qwen38-instance-state-closed.md).
§E1e, found while closing §E1a, was fixed the same day and moved to
[`open-tasks/2026-10-17-gemma-moe-swallowed-decode-error-closed.md`](2026-10-17-gemma-moe-swallowed-decode-error-closed.md).

---

## V2. `max_open` does bound RSS (not reproduced, 2026-10-23)

**The migration itself is closed** — the arena is gone for qwen MoE / Ornith /
Mistral, with measurements, gates and the guard test in
[`open-tasks/2026-10-14-expert-store-arena-retired.md`](2026-10-14-expert-store-arena-retired.md).
Headline: qwen3.6 17.5 → **7.0 GB** at unchanged 141 t/s. qwen3.8-flash-next
was **tried and reverted** to its 63.6 GB pinned arena: its decode working set
is the whole expert set, and the views path loses 3.5x (V1, negative) —
[`open-tasks/2026-10-14-qwen38-arena-required-negative.md`](2026-10-14-qwen38-arena-required-negative.md).
One consequence the change exposed stays open:

#### V2. Not reproduced — `max_open` does bound RSS, and the geometry pass does not fault payloads

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

---
