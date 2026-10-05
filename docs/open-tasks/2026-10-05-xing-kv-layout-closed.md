# KV cache layout `[head][pos][dim]` - closed 2026-10-05 (negative)

Verbatim snapshot of the `docs/OPEN_TASKS.md` section as it stood when the work
closed. Moved out of the live file because the item is RESOLVED: the layout was
implemented, measured, and rejected. Nothing here is open work - it is kept as the
investigation trail, including the parts that were wrong.

Summary for the archive table: the layout change was implemented on a branch behind
a neutral `ModelInitConfig::kv_cache_layout` knob and is bit-exact both ways, but
head-major is 2-6% SLOWER end to end at every context, so pos-major ships. The
reusable parts (the `KvCacheGeometry` stride layer, the layout knob with
`IModel::KvLayout()`, a bit-exact parity gate, and a model-neutral streaming-read
ceiling probe) all shipped. The separate `kAttnSplits` finding was measured the same
day and rejected on bit-identity. Two pre-existing defects surfaced and were fixed.

---

- **CLOSED, NEGATIVE (2026-10-05): the KV cache layout was measured and is NOT the
  bottleneck.** The `[head][pos][dim]` layout was implemented on a branch
  (`feat/kv-head-major-layout`) behind a neutral `ModelInitConfig::kv_cache_layout`
  knob, and it does make an isolated K/V read faster - reproducibly 6-10% at ctx
  4096/8191 (811 vs 744 GB/s), once the measurement harness stops being biased by
  which layout runs first. It STILL makes the decode step slower end to end at every
  context measured, on the AOCC perf tree:

  | max-gen | pos-major | head-major |
  |---|---|---|
  | 2048 | **119.3 t/s** | 116.8 t/s |
  | 4096 | **96.1 t/s** | 92.0 t/s |
  | 8191 | **68.7 t/s** | 64.9 t/s |

  So the inference this section was built on - "the layout is why" - is wrong. At
  long context the pos-major read already sustains 744-817 GB/s; the gap to the
  1792 GB/s DRAM peak is ACCESS GRANULARITY, not head/position order: a 256-thread
  block has each thread walk a whole 192-float K row, and no permutation of head and
  position fixes that. What would is cooperative tiling - load K/V into shared
  memory with fully coalesced 128-byte-per-warp reads. And isolated read bandwidth
  does not predict the step at all: with 39 other layers' traffic competing for the
  same L2, the ranking inverts. Measure end to end or not at all.

- **The real bottleneck, measured (2026-10-05, same day).** With the layout question
  settled, the cheap experiment that decides whether a tiling rewrite is worth
  anything at all: read the SAME 335 MB with the canonical coalesced pattern
  (float4, consecutive threads at consecutive addresses, grid-stride, full occupancy)
  - `tests/test_stream_read_ceiling.cu`, budget `mem.stream_read_ceiling_gbps`.

  | what | GB/s at ctx 8191 |
  |---|---|
  | plain coalesced read (94% of the 1792 GB/s spec peak) | **1687** |
  | dense decode GEMV (math on the loaded data, `xing.gemv.dense_decode_gbps`) | 1202 |
  | Xing attention KV read, pos-major | 744-817 |

  So attention runs at **46-48% of what a plain read of the same bytes achieves**, and
  a kernel that also does arithmetic on what it loads (the GEMV) reaches 71%. That is
  a real 1.5-2.3x of headroom and it is NOT a hardware limit - it is the access
  pattern, exactly as diagnosed: in the score phase a 256-thread block has each
  thread walk a whole 192-float K row, so the warp issues 32 scattered 768-B streams
  24 KB apart instead of one contiguous sweep. The V phase is already coalesced
  (`group = tid / vh`, so 128 consecutive threads read 128 consecutive floats).

  **The ceiling is reachable without tiling: vectorise the load.** Before writing a
  tiling kernel, the cheaper hypothesis was that the score phase was LATENCY bound
  rather than sector bound - it issued one 4-byte load per iteration over 192
  iterations with a serial FP accumulation, so a thread held a single K row in
  flight, while sectors were already 100% consumed. Loading K as float4 fixed it:

  | | KV read @8191 | attention @2k | E2E 2048 | E2E 8191 |
  |---|---|---|---|---|
  | scalar loads | 745 GB/s | 57.0 us | 119.3 t/s | 68.7 t/s |
  | float4 loads | **871 GB/s** | **42.9 us** | **127.0 t/s** | **77.3 t/s** |

  +17% on the isolated read, +25% on attention latency, +6.5% end to end at the
  canonical 2048 and +12.5% at 8191 - bit-identical, since both forms walk i
  ascending through the same __fadd_rn/__fmul_rn and the four real-weight golden
  gates (SelfCheck/FSI/CODE/DET) pass unchanged. Half the ceiling gap went without
  writing a tiling kernel. `splits=16` was tried and is WORSE (852 vs 871 GB/s):
  smaller chunks cost more in partials and combine than the wave balance buys.
  **Where the remaining time actually is** (throwaway probes on the live split-K
  kernel at pos 2047, whole `LaunchXingAttention` = 42.9 us after float4; the probe
  removes one read at a time and the results are wrong on purpose):

  | removed | time | so the phase costs |
  |---|---|---|
  | nothing | 42.90 us | - |
  | the V read | 42.89 us | **~0.01 us (0%)** |
  | both reads | 15.58 us | K read 27.3 us (64%), overhead 15.6 us (36%) |

  Two things follow, and both correct earlier reasoning in this file:

  1. **At the canonical 2048 context attention is not bandwidth bound at all.** The K
     read moves 50 MB in 27.3 us = **1842 GB/s effective, ABOVE the 1687 GB/s DRAM
     ceiling** - so it is being served from L2, and so is V (33 MB), which is why V
     is free and why vectorising it would have been wasted work. Whole KV at 2048 is
     84 MB and fits. The streaming ceiling is the right yardstick only once the
     context exceeds L2; at 2048 it is not, and quoting it there is misleading.
  2. So V-phase vectorisation is NOT the next candidate - there is nothing in it.
     The next lever at 2048 is the **36% overhead**: the online softmax's two block
     reductions (max, then sum, each shfl-down + `warp_max[]`/`warp_sum[]` +
     `__syncthreads`) plus one `XingExpfGlibc` per element - a glibc-replica exp in
     FP64-emulation arithmetic, which is expensive and is the obvious suspect.
     Note this is NOT a bandwidth problem, so the streaming ceiling says nothing
     about it and it needs its own measurement, not a bandwidth budget.

  **Tiling is a dead end - its premise was already removed by the float4 change.**
  Re-probed at the new operating point (splits=4, pos 2047), the split is now:

  | | time | share |
  |---|---|---|
  | K read | 20.20 us (50 MB -> **2492 GB/s**) | 50% |
  | everything else | 20.43 us | 50% |

  2492 GB/s is ~1.5x the 1687 GB/s DRAM ceiling, so the read is served from L2 and is
  no longer the thing worth fixing. At ctx 8191 the whole KV read runs at 1215 GB/s,
  which is the same rate the dense decode GEMVs achieve on this card (1202) - i.e.
  attention is now at parity with a kernel that also does arithmetic on what it
  loads, and the remaining gap to the read-only ceiling is the arithmetic, not the
  access pattern. Cooperative tiling targets access granularity, so it has no target
  left. Recorded rather than attempted.

  What IS left at the canonical context is the other 50%: the online-softmax tail,
  and it GREW from 13.5 us to 20.4 us when splits went 8 -> 4, because each block now
  covers 512 positions instead of 256 and the reductions scale with it. That is the
  other side of the same trade-off, and it is why splits=4 is only +2% at 2048 while
  being +10% at 8191. Cutting the tail means restructuring the softmax - the max pass
  re-reads `sc[]` from shared that the exp pass reads again, and a register-resident
  variant is possible since each thread holds only 2 elements - but the sum order IS
  the bits, so it is a numerics-visible change like kAttnSplits, not a free one.

- **The 36% tail was wave quantisation, not the exp (2026-10-05).** Probing the live
  split-K kernel at pos 2047 by removing one read at a time (wrong results on
  purpose, edits reverted):

  | removed | time | phase cost |
  |---|---|---|
  | nothing | 42.90 us | - |
  | the V read | 42.89 us | ~0 (0%) |
  | the combine kernel | 40.84 us | 2.1 us (5%) |
  | both reads | 15.58 us | K 27.3 us (64%), tail 15.6 us (36%) |

  Two wrong guesses, both mine and both cheap to kill: `XingExpfGlibc` is NOT the
  cost - swapping in hardware `__expf` saves 0.03 us - and V-phase vectorisation has
  nothing in it, because at 2048 the whole 84 MB KV fits in L2 and V is already
  free. Also: at 2048 attention is not bandwidth bound at all. The K read moves 50 MB
  in 27.3 us = 1842 GB/s effective, ABOVE the 1687 GB/s DRAM ceiling, because it is
  served from cache. The streaming ceiling is the right yardstick only past L2.

  What the tail actually was: **wave quantisation.** `kAttnSplits` was 8, i.e.
  32 heads x 8 = 256 blocks on 170 SMs, so the second wave runs at half occupancy.
  The comment justifying 8 claimed the opposite ("fills the SMs twice over, more
  splits only add combine work") and was simply wrong. Measured curve, KV read at
  ctx 8191 / attention at pos 2047:

  | splits | blocks | GB/s | us |
  |---|---|---|---|
  | 2 | 64 | 688 | 52.7 |
  | 3 | 96 | 1129 | 44.9 |
  | **4** | **128** | **1215** | **40.7** |
  | 5 | 160 | 1133 | 40.7 |
  | 6 | 192 | 1020 | 40.5 |
  | 8 | 256 | 867 | 43.1 |

  Two effects cross at 4: parallelism wants >= one block per SM, while per-block
  sequential MLP and the partials->combine traffic (131 KB at splits=8, 65 KB at 4)
  both want FEWER, fatter blocks. "Fill the SM array" is the wrong rule - 128 blocks
  beats 160 - so do NOT make this `SMs / heads`; it would give 5 and lose 7% of the
  read.

  End to end at splits=4 was 129.3 t/s at 2048 and 85.1 at 8191; the shipped,
  bit-identical configuration is 127.1 and 77.3 (see the rejection below).

  **REJECTED ON BIT-IDENTITY (project decision, 2026-10-05).** `kAttnSplits` sets the
  chunk boundaries, hence the reduction order, hence the bits, and the project takes
  bit-identity over throughput. **8 ships; 4 was reverted.** All four real-weight
  golden gates passed at 4, but that is four prompts, not proof of source parity, and
  the decision does not need proof to be the right one: "bit-identical" is a property
  you can verify, "+10% at long context" is not.

  What that costs, stated plainly: **129.3 -> 127.1 t/s at 2048 and 85.1 -> 77.3 t/s
  at 8191.** The float4 change is untouched by this and IS bit-identical, so the
  shipped session result stands at:

  | | main | shipped (bit-identical) | with splits=4 (rejected) |
  |---|---|---|---|
  | E2E 2048 | 119.3 | **127.1 t/s** (+6.5%) | 129.3 |
  | E2E 8191 | 68.7 | **77.3 t/s** (+12.5%) | 85.1 |
  | attention @2k | 57.0 us | 42.8 us | 40.6 us |
  | KV read @8191 | 745 GB/s | 875 GB/s | 1211 GB/s |

  Revisiting costs one constant; the full curve is in the constant's comment and
  above. Nothing else has to change - the layout knob and the bit-exact parity gate
  make the addressing independent of the split count.

  A related note for whoever reopens this: the softmax-tail work that would have come
  next (register-resident `sc[]`, since each thread holds only 1-2 elements) is
  ALSO numerics-visible - the sum order is the bits - so under this decision it is
  out too. At pos 2047 the split is score phase 20.2 us / combine 1.8 us / rest 18.4
  us, and the "rest" was never attributed between the V phase and the reductions.

  **So yes, cooperative tiling of K/V into shared memory is worth attempting - it
  targets the measured deficit rather than a guess. Two honest caveats: the ceiling
  probe is read-only and compute-trivial, so 1687 is an upper bound attention can
  never reach and 1202 (the GEMV) is the more realistic target; and tiling adds
  shared-memory traffic, barriers and reuse pressure across 1280 blocks (40 layers x
  32 heads), so it can cost occupancy. Measure end to end on the AOCC tree.
  Note also that tiling and layout are COUPLED, and the layout verdict above does not
  transfer: under a tile load a head's consecutive positions are contiguous in
  head-major and strided in pos-major, so re-measure both layouts once tiling exists
  (`cfg.kv_cache_layout` makes that a one-line experiment).

  Kept from the attempt, because they are reusable and correct: the neutral
  `KvCacheGeometry` layer (three strides, not a tagged layout, so both layouts
  coexist in one binary and stay bit-exact - `tests/test_kv_cache_layout.cpp`),
  the `KvCacheLayout` knob with `IModel::KvLayout()` so "ignored" is observable, and
  a bit-exact parity gate that runs a decode step under both layouts
  (`tests/test_xing_kv_layout_parity.cpp`). The shipped default stays pos-major
  because it measured faster; flipping it per family is now a cheap, safe,
  reversible experiment rather than a rewrite.
  Two pre-existing defects surfaced on the way and are fixed: the 2k attention
  test's `splits=1` iteration compared the previous iteration's output and verified
  nothing (`xing_attn_kernel` owns short contexts only - it returns immediately when
  `n_pos > nthreads`), and `XingCopyLayerKV`'s host pitch was the whole row instead
  of one head.
