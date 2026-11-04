# qwen3.8-flash-next: the serialized expert H2D — chunked staged overlap (2026-11-04)

Branch `fix/qwen-template-fidelity`. Closes the first §R16 item (`Open (perf,
measured)`: 5.58 ms/token of expert H2D, 100% unoverlapped) that the
[Qwen3.8-Flash-Next lifecycle session](2026-11-04-qwen38-flash-next-lifecycle.md)
left behind. The second item (framing) is the
[fidelity entry](2026-11-04-qwen-template-fidelity.md).

## The defect, restated precisely

The decode step staged residency for all K = 10 routed experts of a layer
**before** any FFN compute, and it did so on the compute stream
(`EnsureQwen4ExpExpertResident(..., stream, ...)` with no `h2d_stream`). The
`copy_stream`/`ev_uploads`/`ev_front` arguments the driver passed on to
`moe::RunMoERoutedFFN` were therefore dead: by the time that call ran, every
expert was already resident, so nothing was left to overlap.

## What was implemented

1. **Staging rides the copy engine.** The residency loop now passes
   `copy_stream` + `&missed[k]`, records `ev_uploads[k]` after each upload, and
   fences slot reuse with `ev_front` (recorded on `stream`, waited on
   `copy_stream`). Gated by `LsxDualStreamStaging()` like every other staged
   family.
2. **Chunked staged overlap** — a new neutral entry point
   `moe::RunMoERoutedTcBatchStreamed` (`lsx_moe_inference.{h,cu}`). The K slots
   run in ascending contiguous groups; before group *c* the chain waits **only**
   on that group's uploads, so group *c+1*'s H2D keeps running while group *c*
   computes. The fp32 accumulate order into the same caller-seeded `d_acc` is
   unchanged (ascending slot order), and `LsxTcBatchSplitK` scratch is
   slot-disjoint, so the result is **bit-identical** — a scheduling change, not
   a numeric one.
3. **Depth follows the miss count.** `chunks = (n_miss >= 2) ? 2 : 1`. A chunk's
   compute can only hide another chunk's copy; with fewer than two misses in the
   layer there is nothing to hide and the second batch launch (K = 5 instead of
   10 halves the tiles per launch) is pure cost. Measured below.
4. **`chunks <= 1` still fences.** The first cut of (2) degenerated to
   `RunMoERoutedTcBatchTbl` whenever `chunks <= 1`, which skips the upload waits
   entirely — a race against the copy engine for any caller that handed it a
   `copy_stream`. The test now covers `chunks` 1/2/3, and one chunk means
   "serialize", never "skip the wait".

## The instrument was wrong first

The first attribution pass (in the lifecycle session) reported the copies as
0.8% overlapped. That number came from matching each copy against only the
kernel that *started* most recently before it, which cannot see a kernel that
starts inside a copy. An honest interval intersection gives **17.7%** for the
copy stream (`streamId 14`) and 0.0% for the compute stream (`streamId 13`),
over a 68-step trace: kernels 18.51 ms/step, H2D 8.35 ms/step of which
1.1 ms/step overlaps. Correcting the instrument did not change the conclusion —
82% of the copy time still runs with the compute stream completely idle — but it
is why the chunk split was worth trying at all.

The remaining 82% is **structural, not a bug**: `moe_router_dots` +
`moe_router_select` (1.54 ms/step) are followed by a host readback, and the host
cannot enqueue anything else until it has the expert ids, so at the instant the
copies are issued the compute stream is empty. There is no independent work to
hide them under. The copies themselves are at the PCIe roofline: 44.2 µs per
2.46 MB blob = **55.6 GB/s**, back to back on the copy engine.

## Measurements (RTX 5090, `linux-release-znver4-aocc`, 2048 tokens, 2 runs each)

Same machine, same `slots` / `hit_rate` / `misses_per_token` in every pair, so
the only variable is the code:

| build | `perf` (kAuto, 5350 slots, hit 0.862) | forced 4319 slots (hit 0.745) | `r3` (kOff) |
|---|---|---|---|
| HEAD `2b104cca` (staging on compute stream) | 47 679 ms | — | 58 174 ms |
| copy-stream staging, `chunks = 1` | 47 379 ms (−0.63%) | 53 120 ms | — |
| copy-stream staging, `chunks = 2` | 47 484 ms (+0.22% vs above) | 52 560 ms (−1.05%) | — |
| copy-stream staging, **adaptive** | **46 938 ms (−1.55%)** | **52 341 ms (−1.47%)** | 58 135 ms (−0.07%) |

The `4319` column is the capacity oracle: `hit_rate` drops 0.862 → 0.745
(`misses_per_token` 66 → 123), which is the regime chunking exists for. It is
not the `r3` row — `r3` is a *numeric* mode, and under `kOff`
(`tc_batch_ok = QwenTcFastMath()`) the batched chain is not used at all, which
is exactly why its row does not move.

Rows re-seeded through the harness (`ctest -R test_perf_baseline_qwen3_8_flash_next`,
passing): **`perf` 42.98 → 43.39 t/s**, **`r3` 33.88 → 33.91 t/s**, TTFT
188.9 / 197.8 ms. The manual `lsxbenchmark` numbers above read slightly higher
than the harness rows for the known reason (the harness measures through
`InferenceEngine::Session` in one process, which is the path users run).

## Numeric gates (unchanged, as the argument predicts)

- `logits_hash **2e3bcd090441ad65**`, top-1 11751 `Paris` — unchanged.
- `test_qwen_moe_exp_selfcheck` with `LSX_TEST_QWEN38_SELFCHECK_CYCLES=3`:
  Paris ×3 bit-exact, plus the real-weight native tool round-trip
  `get_weather {"city":"Paris"}` (44 assertions, 2 cases).
- New kernel case `routed TC batch chain: streamed staging chunks match one
  launch` (142 assertions): cold-cache 1/2/3-chunk passes and a warm-cache pass
  against a single-launch reference, on a Blackwell-gated path.
- `ctest --preset linux-debug` (label `fast`): **134/134**.

## What this does *not* close

- **The `r3` row is unmoved**, by construction: `kOff` takes the fused per-slot
  `RunMoERoutedFFN` path, which has no batch form to chunk. It *does* have
  per-expert copy-stream staging of its own (`RunMoERoutedFFN`'s `copy_stream`
  contract), but this driver pre-stages every expert to build the TC slot table,
  so that machinery is unreachable from here. Making it reachable means
  skipping the pre-stage when `tc_batch_ok` is false — an ordering change worth
  ~1% on one row, not taken in this pass.
- **The hyper-connection block is now the largest serial host**, not the copies:
  `hc_epilogue` 2.65 + `hc_down_gemv` 2.16 + `hc_norm` 1.57 + `hc_up_gemv` 0.59
  = 6.97 ms/step of an 18.5 ms/step kernel budget (4 launches x 48 layers).
  That is the next lever, and it is arithmetic, not staging.
