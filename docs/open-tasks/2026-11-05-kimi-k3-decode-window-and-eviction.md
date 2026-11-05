# Kimi-K3.1-Pro / Kimi-K3 — the decode-rate "4x gap", the expert window and the
# eviction stall (2026-11-05)

Session branch `main`. Moved out of `docs/OPEN_TASKS.md` §R17 **P5** and **P5a**
when both closed; the only piece left in the live list is the one that is still
open (overlap the expert staging with compute, `§R17 P6`). The text below is the
live-file entry verbatim, so the investigation trail — including the two
hypotheses that were measured and rejected, and the regression this fix itself
introduced — survives the move.

Why it is worth keeping in full: the reported "4x slower than Kimi-K3" was not a
regression at all but **prose that had been committed before the run it
described**, and the defect that prose was hiding was in the shared
`ExpertStore`, not in the model. A one-line "fixed" would destroy both lessons.

Scope of the closure: the window defect (`max_open` 512 against a 1472-expert
working set, 6.95 → 2.96 s/token on K3 / 3.23–3.35 on Pro) and the eviction stall
(a full `cudaStreamSynchronize` per eviction, 20.3 ms → 0.07 ms). Both landed on
`main` (`107a24ef`, `69d898a9`, `583e6dd9`) and were verified bit-exact on real
weights — K3 `5c1e227265d6a27b`, Pro `98ec1bbb372c9181`, mistral-small-4
`cdfd7309bb141772` plus its native tool round-trip.

## The entry as it stood when it closed

- P5. **RESOLVED (2026-11-05) — the 4x gap was never real, and the real defect it
  hid was the expert window.** Two separate errors were conflated:

  1. *The baseline was prose, not a measurement.* `2026-11-03-Kimi-K3.md`
     recorded "the measured ~2 s/token" at commit `3bc13da7` (04:16:28), i.e.
     **before** K3's first real measurement landed in that same run log at
     04:24:38. K3's own cycles are 434.6 s and 426.3 s per 64 tokens =
     **6.66–6.79 s/token**, and Pro's are 461.3 / 446.6 s = **6.98–7.21**. At
     identical binary (both rebuilt from `main`), identical geometry, identical
     tree layout, the same 1472-expert working set and the same window, the two
     models measured **6.95 vs 6.98 s/token — a ratio of 1.00**. The indexer is
     not a factor: below `index_topk` the pipeline takes the `sel == nullptr`
     branch and runs neither the score pass nor the top-k, and at the protocol
     both numbers were taken at that branch is always the one taken.
     `perf_baselines.json` still has no row for either tree.
  2. *The real defect:* `max_open` was **512** against a **1472**-expert
     working set (92 MoE layers × top-16, each layer visited once per token in
     ascending order). A cyclic scan over 1472 items with 512 slots has a
     **provably zero hit rate**, so every expert was re-opened, re-mmap'd,
     re-parsed and re-evicted — with a full `cudaStreamSynchronize` on each
     eviction, because the Kimi call site passes no `evict_done`. Kimi was the
     only family at 512 (DeepSeek-Pro uses 4096; the rest inherit the store's
     65 536), and the K3 plan's `LSX_KIMI_K3_EXPERT_CACHE_SLOTS` /
     `LSX_KIMI_K3_PIN_DENSE` knobs were never implemented — that plan step is
     still unticked.

  **Fixed** (Kimi-local, bit-exact: `logits_hash 5c1e227265d6a27b` on every run):
  the window is clamped up to the working-set floor (`KimiK3ExpertWindowFloor` /
  `KimiK3ResolveExpertWindow`), default 65 536; and `Load()`'s eager pre-parse is
  switched off for these trees (`warm_on_load=false`) because it walks experts in
  *path* order, not decode order — it cost **156 s** of load at a 65 536 window
  and bought nothing.

  Measured on real weights, 64-token cycles, same binary, GPU-exclusive:

  | window | eager warm | load | warm s/token | vs shipped |
  |---|---|---|---|---|
  | 512 (shipped before this fix) | on | 6 s | 6.95 | 1.00× |
  | 1472 (floor) | on | 18 s | 6.05 | 1.15× |
  | 2048 | on | 10 s | 5.57 | 1.25× |
  | 65 536 | on | **156 s** | 3.01 | 2.31× |
  | **65 536 (shipped now)** | **off** | **5 s** | **2.98** (K3) / **3.23–3.35** (Pro) | **2.33×** / **2.16–2.21×** |

  **Rejected after measurement:** a per-layer page-cache prefetch of the
  router's top-k (the 22× serial-vs-parallel cold-read gap, 1.82 GB/s at one
  request in flight vs 39.8 GB/s at 32, made it look promising). Both forms lost
  or drew: reading each file into a discard buffer measured **7.16 s/token**
  (18% *worse* — the sink read is pure waste once the page is cached, and this
  working set is re-visited 3.5× per expert), and `posix_fadvise(WILLNEED)`
  measured 6.11 vs 6.15 (noise). Removed rather than shipped as an unmeasured
  knob. The mechanism that actually pays is **retention**: with a wide window
  experts are never `munmap`'d, so a re-read becomes a page-cache hit.

  At equal binary and window the variant's own cost is now measurable for the
  first time: **Pro decodes 3.23–3.35 s/token against the parent's 2.96, so the
  DSA indexer costs ~9–13 %**, not the 4× that the prose baseline implied (3.23
  before the store's eviction path changed, 3.35 on the verification run after
  it — the same mechanism, two runs, so the range is the honest figure).

  **Still open, and honestly bounded.** The shipped 3.2–3.35 s/token is
  ~7.6–8 GB/s of effective host-store traffic against 39.8 GB/s available, so a
  deeper fix (overlapping the next layer's expert staging with the current
  layer's GEMMs) could still pay — it needs the pipeline restructured around a
  queue instead of one synchronous `EnsureResident` + six `cudaMemcpyAsync` +
  three GEMVs per expert, and that is the shared `ExpertStore` contract, not a
  Kimi-local edit. The *drain* half of that loop is already gone (P5a): what
  remains is the overlap, not the stall.
  The *scale* is also now a floor, not a property: the debt sweep recomputes
  from 2.98 s/token (~13.6 h, see P3), and both perf rows remain unseeded — a
  harness row is what would make these numbers quotable.

  **P5a — DONE (2026-11-05): eviction no longer drains the stream.** The half of
  the synchronous loop that lived in the *store* is fixed. `EnsureResident` used
  to call `cudaStreamSynchronize` whenever the window was full (the Kimi call
  site passes no `evict_done`), so every evicted expert stalled the host until
  **unrelated** queued work finished — measured in `test_kimi_k3_loader` as a
  20.32 ms eviction against 20.20 ms of queued memset traffic, i.e. a full drain.
  An evicted slot's mapping is now **retired with a `cudaEventDisableTiming`
  event** recorded on the stream that issued the copies and released once
  `cudaEventQuery` reports completion (`RetireSlot`/`ReapRetired`), so nothing
  is unmapped under an in-flight DMA and the host never blocks. Memory stays
  bounded by `max_retired` (64), past which only the *oldest* retiree is waited
  on. Same test after the fix: **0.072 ms** per eviction (282x), and a copy
  issued from an expert's mmap is still byte-exact after that expert was evicted
  out from under it (both halves asserted).

  Real-weight effect is in the eviction-heavy regime and neutral everywhere else,
  which is what the mechanism predicts (K3, same binary, 64-token cycles):

  | window | evictions/token | before | after |
  |---|---|---|---|
  | 512 | 1472 (~100 %) | 6.95 | **6.26** (≈0.47 ms of stall removed per eviction) |
  | 2048 | 0 after warm-up | 5.57 | 5.80 (noise) |
  | 65 536 (shipped) | ~80 (~5 %) | 2.96 | 3.05 (noise) |

  So the shipped configuration gains little — it evicts rarely by design — but
  the families that *stream* through a small window (mistral-small-4, ling,
  beam, step, and any Kimi tree run with a small `MAX_OPEN`) keep this path hot.
  Verified on mistral-small-4 real weights: bit-exact
  (`logits_hash cdfd7309bb141772`, "Paris") plus the native tool round-trip, and
  all 21 store-consuming test targets pass.

  The first version of this fix **regressed mistral-small-4 by 6.6%** (perf row
  45.04 → 42.06 t/s, caught by `test_perf_baseline`, probe confirming the card
  was not the cause) because `ReapRetired` walked the whole retired list and
  issued up to 64 `cudaEventQuery` calls per eviction. The events all sit on one
  stream and therefore complete in FIFO order, so only the front can be
  not-ready: checking just the front makes it amortized O(1) and the row passes
  again. That is the value of running the perf harness rather than reasoning
  about the mechanism — the retention idea was right and the bookkeeping around
  it was expensive.
