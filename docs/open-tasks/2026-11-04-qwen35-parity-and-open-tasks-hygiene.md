# qwen3.5-35b-a3b — real-weight verdicts, two stale benchmark rows, and the
# open-list hygiene pass (2026-11-04)

Session branch `fix/qwen-template-fidelity`. The request was an audit — *is
anything already implemented still sitting in `docs/OPEN_TASKS.md`?* — followed
by finalization. So every claim below is a measurement made in this session, not
a re-read of an older table. Two benchmark rows turned out to be stale in
opposite directions, one gate row was in the wrong section of the matrix, and the
last closed qwen section was still being carried by the live list.

The QA-gate numbers for this member were not taken from
`docs/SELFCHECK_MATRIX.md`: that file placed `qwen3.5-35b-a3b` in its
"golden present, **no converted tree** — SKIP" section, which is false.

## 1. The converted tree exists, and all four gates are green

`/mnt/fast_storage/lsx_model/Qwen/Qwen3.5-35B-A3B/logestix` — 19 GB, converted
2026-10-01 in the Arrow-audit session (the archive row for 2026-11-01 records the
conversion: "Qwen3.5-35B-A3B 217 s/19 GB"). The three goldens
(`tests/fixtures/{fsi,code,det}/qwen3.5-35b-a3b.json`) were already committed, and
`kGoldenDebt` in `tests/test_model_suite_guard.cpp` has no qwen entry at all — so
every gate was *required* to be green and none had ever been run.

Run serially on the real weights, one GPU workload at a time
(`LSX_TEST_QWEN_MOE_MODEL_PATH=/mnt/fast_storage/lsx_model/Qwen/Qwen3.5-35B-A3B`):

| gate | result |
|---|---|
| SelfCheck | PASS, Paris ×3 bit-exact, `logits_hash f4c91be5f2206d56`, top-1 11751, 42 assertions |
| FSI | 3/3 |
| CODE | 3/3 |
| DET | 3/3, `deterministic=true` on all three |

So the member belongs in matrix §1 (real verdicts), not §3. The same audit found
`deepseek-v4.1-flash` also misfiled in §3 ("no converted tree") although its
294 GB tree is in the vault — that one is *not* all-green, see §6.

## 2. Two benchmark rows, stale in opposite directions

Both rows dated **2026-10-06**, i.e. before the neutral routed-MoE migration, the
overflow-proof batched SwiGLU quantizer (`8e34314`, 2026-10-14) and the strict-mode
streaming-GEMV fix (2026-10-23). Its two 35B-A3B siblings were re-seeded at those
moments; this member was skipped both times.

**`perf` (kAuto): 170.10 → 159.81 t/s.** The harness refused the first attempt
by design:

```
DEGRADED baseline=170.1 t/s, actual=163.537 t/s (96.1415%)
(probe 1528.54 vs 1531.68 GB/s: the environment is not the cause)
```

Three harness runs in this session measured **163.54 / 159.59 / 159.81** t/s
(the first was the coldest card of the session), so the row now carries the
harness's own write from the last run, with its own TTFT/power evidence
(33.43 ms, 255.85/275.61 W, 639 samples). Attribution, in the order that decides
it:

1. `8e34314` is a correctness fix (a massive-activation expert produced
   `silu(gate)*up = +inf`, saturating the per-block E4M3 scale and poisoning
   layer 32 of Mistral-Small-4). Its own commit note records the cost: "one
   extra elementwise launch per layer (Qwen3.6 256-token: 166.7 → 161.3 t/s)"
   — 3.3% for the same mechanism on a model with the same routed-FFN geometry.
2. The three 35B-A3B siblings, all on today's code, now measure the same band:
   **159.81** (this row), **161.97** (qwen3.6, 2026-10-23), **160.25** (ornith,
   2026-10-23). The old 170.1 is the outlier, not the new reading.
3. Ornith's re-seed adjudicated exactly this case on 2026-10-23: 167.06 →
   160.20 (−4.1%) was accepted as "the reference predates the neutral MoE
   migration and the overflow-proof batched SwiGLU … an accepted correctness
   fix, not a regression".

What was **not** done: no bisect of `8e34314~1`. A build of the parent commit
would separate the two changes to the percent (the spread between the three runs
here is already 2.4%, i.e. most of the residual), but the row would only be
seeded at a value inside the siblings' own band. The evidence that the delta is
not a regression is the sibling triple plus the change's own A/B, not a
single-commit bisect; that is stated here so a later reader knows which
instrument was used and which was skipped.

**`r3` (kOff): 13.78 → 77.56 t/s.** This one is not a cost, it is a superseded
defect. The strict mode was made *correct* by disabling `gemv_large` (qwen3.5-9b
r3 fell 111 → 19.3 t/s); the 2026-10-23 fix pinned the large-N and ≤32 MiB
streaming GEMV as the strict mode's canonical route and re-seeded every affected
row — qwen3.5-9b 111.06 → 147.0, qwen3.8-27b 39.03 → 52.27, qwen3.6 13.32 →
73.39, ornith 12.70 → 72.02 — but not this one. The recorded 13.78 was therefore
5.6× below what the same binary does, and any r3 regression up to five-fold would
have passed unseen: the row was below every real value. Re-seeded through the
harness (77.22 → 77.56 on the two runs), consistent with its siblings
(73.39 / 72.02) and with a direct `lsxbenchmark --bench --bit-exact` reading of
77.3 t/s (TTFT 64.16 ms).

## 3. `qwen_moe.router.decode` had no budget on this GPU

`tests/test_qwen_moe_kernels.cu` guards the real-shape decode router with a
latency budget, and in Release the case **SKIPped**: `no kernel budget for
'qwen_moe.router.decode' on this GPU (key 'rtx 5090')`. The instrument existed and
did not judge — the exact failure mode the budget table was added to prevent (the
same defect the phi and ornith rows closed).

Recorded with `LSX_KERNEL_BUDGET_WRITE=1` over three observations:
11.7931 / 11.7947 / 11.7955 µs, i.e. `observed_max` 11.7955, budget 15.924
(the helper's 1.35× latency pad), `samples = 3`. Re-running the case in Release
now **judges** (260 assertions, all passed) instead of skipping. The budget is
worth having because the rejected alternative is within the same case's comment:
the deep-unrolled `uint4` stage-1 walk lands ~30 µs (the 31.24 µs reading that
`OPEN_TASKS.md` §A1 still lists was that variant against a 20 µs budget), so a
regression of exactly that shape is caught here and nowhere else.

## 4. Documents corrected (the code was already green)

- `docs/SELFCHECK_MATRIX.md`: `qwen3.5-35b-a3b` moved §3 → §1 with the measured
  row; `deepseek-v4.1-flash` moved §3 → §1 with its real verdicts; the summary
  counts now state their counting rule and were recomputed over §1 (SelfCheck
  22/22 rows, FSI 58/63 cases, CODE 62/63, DET 62/63); the `Reproduce` block no
  longer lists either model under "absent-vault families (all SKIP)".
- `docs/DET_SELFCHECK_FINDINGS.md` and `docs/CODE_SELFCHECK_FINDINGS.md` both
  claimed the qwen3.5-35b-a3b source was absent from the vault — false since
  2026-10-01. Both now carry the measured result.
- The two benchmark rows and the router budget above.

## 5. The live list itself

Two fully-resolved sections were still being carried by
`docs/OPEN_TASKS.md`, which its own rule forbids ("only genuinely open work lives
here"):

- **§J** (gemma-4-26b-a4b-it FSI/CODE) — resolved 2026-10-24 and already
  archived; the live section was a duplicate of the 2026-10-24 archive row.
- **§R16** (qwen3.8-flash-next) — both residuals closed in this branch. Its text
  moved verbatim into
  [`2026-11-04-qwen38-flash-next-lifecycle.md`](2026-11-04-qwen38-flash-next-lifecycle.md)
  §6.

Also added: **§A2**, a new real defect found while auditing (see §6 below).

## 6. What remains open

- **§A1** is the only qwen item left: `test_qwen_moe_kernels` +
  `test_ornith_moe_kernels` raise `cudaErrorIllegalAddress` on the Windows /
  5060 Ti box. Not reproducible here (both suites pass, 1156 assertions / 16
  cases, `compute-sanitizer` clean), and the rule stands: do not change a kernel
  until it reproduces there. The latency-budget half of that item is now
  separated from the fault for this GPU — the router case judges instead of
  skipping (§3).
- **§A2** (new, not fixed): `deepseek-v4.1-flash`'s cooperative-cancel case fails
  with `compressor state reset failed`, and the cause is a *cross-request*
  stale-error leak, not a compressor bug — a cancelled request never records
  `ev.ttft`, the unchecked `cudaEventElapsedTime` returns
  `cudaErrorInvalidResourceHandle`, and the thread-local last-error is then
  reported by the **next** request's first `cudaGetLastError()`. The case had
  never run because the test's default root still points at the retired vault.
  Root cause and fix sketch are in `docs/OPEN_TASKS.md` §A2.
