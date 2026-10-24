# phi-4: the kAuto perf-mode row, the continuation gap, and two instrument defects (closed 2026-10-24)

Session: full phi-4 lifecycle re-audit at the operator's request, then the open
`perf`-mode item ("the 134.67 t/s row of 2026-10-01 is not reproducible"). The
vault had no converted artifact at the start — the `logestix/` tree used for the
10-07/10-23 measurements was gone — so the model was re-converted in place
(`lsxforge --phi`, 7.7 GiB, 59 s) and every stage re-run.

Result: two model/harness defects fixed, one codec cost recovered by a
bit-identical rewrite, and the perf rows re-seeded by the harness with real
evidence. Remaining: the DET-02 near-tie (live §O, unchanged) and the
harness-vs-app measurement offset (recorded below as a lead, bounded).

---

## 1. Continuation never received the audit-X2 fix (P1, fixed)

`phi_dense_inference.cu` is granite's generation body with phi's fused qkv /
gate_up layouts, but it did not get granite's X2 continuation fix. Its prefill
hard-coded `pos[i] = i` and `kv_base = 0`, then reset `kv_rows = seq`, so a
continuation `Infer` (whose `input_ids` equal the previous call's `gen_tokens`,
the session rule `phi_dense_model.cpp` implements) wrote its K/V over cache row
0 and attended only key 0. The session context was dropped and the answer came
from a one-row attention; the request still returned `ok = true`.

Ported granite's contract: `pos0 = cache->kv_rows`, positions `pos0 + i`, the
attention base `pos0`, capacity `pos0 + P + max_new`, `kv_rows = pos0 + seq`.
Fresh sessions are byte-identical (`pos0 = 0`), so no golden moved.

Pinned by the new `test_phi_dense_model` case "phi continuation matches a fresh
prefill": RED before the fix (`kv_slots_used` 3 for the continuation vs 10 for a
fresh prefill of the same context), GREEN after; the case also compares
`gen_tokens` and `final_logits` against the fresh run.

## 2. A Debug perf run rewrote the tracked baseline oracle (P2, fixed)

The `#ifndef NDEBUG` skip lived only in `MeasureModel()`, i.e. after
`RunPerfModel()` had already parked the GPU and run the 512 MiB DRAM probe — and
`UpdateProbeReference()` writes the `__gpu_probe__` row into
`tests/perf/perf_baselines.json`. Two `ctest --preset linux-debug -R
test_perf_baseline_phi_4` runs rewrote that row (1524.37 -> 1528.54 -> 1526.45)
before SKIPping, so a Debug sweep — which measures a debug-compiled probe that is
never compared against the Release reference — moved the committed environment
oracle. The gate now sits at the top of `RunPerfModel`, before the cooldown and
the probe: a Debug run SKIPs in 0.01 s with the file untouched. Pinned by a
source-scanning case in `test_perf_baseline_logic`.

## 3. The kAuto perf-mode row: attribution and resolution

### The measurements

| path | perf (kAuto) | vs 134.67 | r3 (kOff) | vs 128.68 |
|---|---|---|---|---|
| `lsxbenchmark --bench`, before any change | 132.1 / 131.5 | 98.1% / 97.7% | 130.9 | 101.7% |
| `test_perf_baseline`, before any change | 129.89 / 129.54 | 96.5% / 96.2% (DEGRADED) | — | — |
| `lsxbenchmark --bench`, final (decode rewritten) | 133.7 / 133.2 | 99.3% / 98.9% | — | — |
| `test_perf_baseline`, final | 130.72 | 97.1% (PASS) | 129.68 -> pass | — |

Environment oracle nominal throughout (DRAM probe 1524-1534 GB/s against the
1524 reference). qwen3.5-9b reads 100.4% of its 2026-10-23 re-seed.

### What the deficit was

`pow2` below means the pre-2026-10-06 codec (the decode state the reference was
measured with), reproduced locally by reverting only the decode body:

| build | phi perf, 2048 tokens |
|---|---|
| E4M3FN decode, early-return form (shipped until this session) | 131.5-132.1 |
| E4M3FN decode, single-return selects | 132.4-132.7 |
| E4M3FN decode, single-return + FMA subnormal (final) | **133.2-133.7** |
| pow2 decode (pre-codec) | 133.1-133.6 |

So the whole measurable part of the gap was the **E4M3FN block-scale decode in
the seq==1 streaming-GEMV inner loop**, added 2026-10-06 (`b30eb71`, subnormals
in `c037ed7`) — five days *after* the reference row. The cost is real and not
cosmetic: the current phi artifact contains **3 636 148 subnormal scale bytes
(3.40%)** across the sampled layers and 0x7F is absent, so the subnormal path is
hot and the pre-codec decode would have zeroed those blocks.

The final form keeps the codec semantics byte-identical (exhaustive host table +
exhaustive device parity in `test_lsx_kernels` both pass) and removes the
int-to-float convert:
`bits = ((e + 120) << 23) | (m << 20)`, `v = bitcast(bits)`,
`sub = 2*v - 2^-6 == m * 2^-9` (Sterbenz-exact), `0x7F` (bits == 0x43F00000)
-> NaN, `e == 0` -> sub. That is one return path, no branch on the hot path, and
it lands at pow2 parity.

### Hypotheses ruled out (each measured)

- **Compiler/build tree.** The reference predates the AOCC preset (added
  2026-10-06), so it was recorded on the GCC `linux-release-znver4` tree. HEAD
  rebuilt on that tree reads 133.2 / 132.7 — identical to AOCC (132.4-132.7).
- **First-Infer cost / cold DVFS.** A 32-token untimed warm-up before the
  measured run did not move the harness reading (129.65 vs 129.5-130.4).
- **NVML power-sampler contention.** Disabling `GpuPowerSampler::Start()` did
  not move it (130.0 / 129.8).
- **Cooldown state.** `LSX_PERF_COOLDOWN_TIMEOUT_S=0` on a hot card reads 128.6
  (worse), so the cooldown is doing its job rather than causing the deficit.
- **The artifact.** A fresh conversion has the same byte counts and the same
  resident footprint (7865 MiB); weight bytes do not move kernel time.
- **Environment drift.** phi's own r3 row (same streaming-GEMV route) moved
  128.68 -> 132.12 -> 129.68 across consecutive harness runs today, i.e. the
  harness's own run-to-run spread on this model is ~2%. A three-week-old row
  cannot be held to a 3% band with more confidence than that.

### Resolution

Decode rewrite landed (above); the perf rows were then re-seeded **by the
harness**, with its own TTFT/power evidence. The first pass wrote `perf`
134.67 -> 130.72 (ttft 30.17 ms, 565.5 W avg / 602.2 W peak, 781 samples),
then the next sweep tightened both rows as kUpdate is designed to: **perf
130.91**, **r3 130.31**. `test_perf_baseline_phi_4` passes both modes.

## 4. New instruments

- **`phi.dense_gemm.bandwidth`** budget row (min 1013.71, observed 1267.14
  GB/s). The case existed but had **no row for this GPU**, so it SKIPped — the
  silent-skip failure mode. The shape set also gained the real `qkv` shape
  (5120x7680), which the four-shape set never measured (22 MB/token, 18% of the
  decode weights). Isolated rates 2026-10-24: qkv 831, q_o 722->758, gate_up
  1267, down 1112, head 1395 GB/s. The 2026-09-22 four-shape 1327 reference is
  not comparable to the five-shape aggregate.
- **`tests/test_kernel_budget.h` parser.** The loader was strictly
  one-object-per-line and read only the fields on the first line of an entry, so
  the hand-formatted multi-line rows (`xing.kv_read.decode_gbps`,
  `mem.stream_read_ceiling_gbps`) came back with empty `note/date/samples` —
  and any `LSX_KERNEL_BUDGET_WRITE=1` run then rewrote the file from that
  truncated view, destroying their notes (observed). It now joins continuation
  lines (brace-balanced, string-aware) and throws on an unterminated or
  unnamed entry instead of dropping it into a false "no record for this gpu"
  SKIP. Pinned by "kernel budgets: a hand-formatted multi-line entry is parsed",
  which also re-reads the committed file and checks every entry survives whole.

## 5. Still open / leads

- **DET-02** — the C-vs-CR near-tie stays red as source parity (live §O,
  re-verified today: source top-8 `C` 30.75, `CR` 29.625, `CAT` 22.75).
- **Harness-vs-app offset on fast models.** `test_perf_baseline` reads
  130.3-130.7 where `lsxbenchmark --bench` reads 133.2-133.7 on the same binary,
  artifact, prompt protocol and fast-math mode (~1.5-2%, reproducible). NVML,
  warm-up, cooldown state, compiler and artifact are eliminated; the remaining
  candidates are process/VRAM-layout differences (the harness holds an
  `InferenceEngine::Session` and allocates through a different sequence) or a
  Catch2-process effect. Worth one instrument audit, because it sets the floor
  for every fast model's row (granite, 2x slower per token, shows <1%).
- **Reroute lead (not adopted).** Serving the mid-N shapes on the warp-per-row
  kernel measured qkv +9.6% (831 -> 910) and q_o +6.5% (728 -> 775) in
  isolation, but `down` (deep K) loses 9.6% (1112 -> 1005) and granite's N=4096
  shapes already preferred the sliced route (453 vs 514). A rule that fits all
  six measured points is `N >= 5120 && K <= 8192`, which is a fill-ratio
  heuristic with only this family's data behind it — a neutral-dispatch change
  for a different session, with every affected family re-measured.
- **Launch gaps.** The decode step spends 0.28 ms/token (3.7%) between kernels
  across ~360 launches; the CUDA-graph route Xing uses (`LSX_XING_CUDA_GRAPH`)
  has not been ported to phi.
