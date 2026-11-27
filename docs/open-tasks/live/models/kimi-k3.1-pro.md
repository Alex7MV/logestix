# Kimi-K3.1-Pro — open items

### R17. Kimi-K3.1-Pro (2.8T / ~104B active) — Gated DSA variant landed, parity sweep is debt (2026-11-05)

Branch `feat/kimi-k31-pro`. Sibling variant of Kimi-K3 (`kimi-k3.1-pro`, files
`kimi_moe_pro_*`, format tag `logestix-kimi-k31-pro-nvfp4`): the same KimiLinear
tower (93 layers, 69 KDA + 24 gated NoPE MLA, AttnRes 12, Stable LatentMoE, SiTU)
with the 24 full-attention layers replaced by **Gated DSA** — a Lightning
Indexer per full-attention layer selecting `index_topk = 2048` keys per query.
Landed (stages 1–8 of the Model Ingestion Protocol): frozen passport,
converter (`lsxforge --kimi_k31_pro`, wrapper-level variant gate both ways),
loader (five BF16 indexer tensors per DSA layer pinned as FP16, never quantized,
plus the `index_norm_eps` cross-check), the four indexer kernels + selectable MLA
attention, the pipeline's indexer key cache with the dormant-at-`n_pos <= 2048`
DSA branch, `KimiK31ProModel` + registry + the lifecycle contract, and the four
real-weight targets with their template-derived goldens and the perf row.
Stage 9 landed too, in place at `/mnt/fast_storage/lsx_model/moonshotai/Kimi-K3.1-Pro/logestix`
(1.5 TB source → **1.5 TB tree** in 17 min: 1.4 TB routed-expert arrows, 23 GB
pinned graph, 8.9 GB streamed latent/shared plane, `total_experts 82432`, 2120
pinned + 460 streamed tensors, 165 vision + 3 projector tensors stripped and
counted). On the real weights: **SelfCheck Paris ×3 bit-exact**
(`logits_hash 98ec1bbb372c9181`, top-1 40) and the **native XTML tool
round-trip** `get_weather {"city":"Paris"}` — both green in
`test_kimi_pro_selfcheck`. Model report:
`docs/model-reports/2026-11-05-Kimi-K3.1-Pro.md`.

**Open, explicitly.**

- P1. The indexer's arithmetic is **reconstructed**, not verified. The
  checkpoint ships no modeling code (`configuration_kimi_k3.py` only), so the
  semantics come from the tensor shapes, `dsa_config` and the repository's prior
  art (`deepseek_moe_pro` CSA, `mistral_moe_large4` DSA). The three open points
  are enumerated in the design (§3.2): the indexer's input is taken to be the
  post-`input_norm` hidden (the DeepSeek contour; the alternative reading is the
  raw AttnRes mixture and would be a one-line change), `index_skip_topk_offset`/
  `index_topk_freq` are not implemented as knobs, and the top-k does **not**
  force the current position in. None of them is reachable at the canonical
  2048-token protocol, where selection is dormant — which is also why no gate
  covers them: **do not read the variant's green gates as DSA verification.**
- P2. The **kimi-k3 synthetic fixture is input-invariant**: two different
  prompts produce bit-equal logits from the converted 4-layer tree (measured on
  the plain Kimi-K3 tree as well, so this predates the variant). Every
  model-stage assertion in the family is therefore a smoke assertion — finite,
  deterministic, lifecycle-correct — and cannot observe a wrong attention key
  set. The DSA branch is instead pinned by exact step/key counts plus a
  score-vector/selection comparison against an independent top-k, and by the
  kernel-level host references. Making the fixture input-sensitive (e.g. a
  scale that keeps the residual from being dominated by a constant block) is the
  instrument this family is missing — `docs/OPEN_ENDEDNESS.md` §"which
  instrument is missing".
- P3. **PERF HALF CLOSED (2026-11-06); the FSI / CODE / DET half is still debt.**
  All three kimi families now have harness rows in `tests/perf/perf_baselines.json`,
  recorded through `test_perf_baseline` (never by hand), 2048 tokens, the recorded
  prompt, GPU-exclusive:

  | model | perf (kAuto) | r3 (kOff) | s/token (perf / r3) | TTFT (perf / r3) |
  |---|---|---|---|---|
  | `kimi-k2.7-code` | 5.05 t/s | 5.04 t/s | 0.20 / 0.20 | 1.0 s / 1.1 s |
  | `kimi-k3` | 0.47 t/s | 0.49 t/s | 2.13 / 2.04 | 131.9 s / 102.1 s |
  | `kimi-k3.1-pro` | 0.50 t/s | 0.51 t/s | 2.00 / 1.96 | 165.1 s / 92.3 s |

  Two things this changed. (a) K3 and Pro had **no rows at all** before — every
  number quoted for them until now was log-derived, and the entry said so. (b) The
  prose figures they replace (2.96 s/token for K3, 3.23–3.35 for Pro, P5) are
  ~1.5x apart from the harness rows on the same tree and the same binary; the
  difference is cache state and TTFT accounting, which is why the rows are now
  the quotable form and the prose is not. Also note **Pro measured faster than
  K3** (2.00 vs 2.13 s/token), which removes the last trace of P5's premise: the
  Pro is not the slower of the two.
  The FSI / CODE / DET cells remain `not run` — that half of the sweep is
  unchanged and still owes ~7 h of exclusive GPU at the measured rate. The cost
  is measured, not guessed, and it shrank again when the rows above replaced the
  prose rate: recomputed at the **harness-measured 2.0 s/token** for this tree
  (window 65 536, no eager warm — see P5; the figures below were first written at
  the pre-fix 0.14 t/s, then at 3.2–3.35):

  | target | generations × tokens | wall clock at 2.0 s/token |
  |---|---|---|
  | `test_kimi_pro_fsi_selfcheck` | 3 × 2048 | ~3.4 h (+load) |
  | `test_kimi_pro_code_selfcheck` | 3 × 512 | ~0.9 h (+load) |
  | `test_kimi_pro_det_selfcheck` | 3 × 3 × 512 | ~2.6 h (+load) |

  (`test_perf_baseline_kimi_k31_pro` used to be the fourth row of this table; it is
  **done** — its two modes are in the table at the top of this entry.)

  so that sweep is ~7 h of exclusive GPU on this host at the now-measured
  2.0 s/token (it was ~32 h at the pre-fix rate, ~14.7–15.3 h when these figures
  were first written as prose, and ~11 h at the 3.2–3.35 estimate). FSI was
  actually started (release tree) and stopped 97 min in,
  still inside FSI-TEST-01; the interruption made the harness emit three empty
  generations, and **an in-flight kill is not a verdict** — the cells stay
  `not run`. The ctest budgets still describe the pre-fix worst case (FSI
  54 000 s, CODE 18 000 s, DET 43 200 s, perf 43 200 s), which is now ~4x
  generous rather than tight; leave them, since a budget may only be wrong in
  the safe direction. To close this item, run each target alone (GPU exclusivity)
  with `LSX_FSI_GOLDEN_DIR`/`LSX_CODE_GOLDEN_DIR`/`LSX_DET_GOLDEN_DIR` pointing
  at `tests/fixtures/<stage>`, then let the harness seed
  `tests/perf/perf_baselines.json` — never by hand. Note what a green row would
  and would not mean: the goldens are **template-derived** (P4), and no gate
  observes the indexer (P1/P2).
- P4. **The FSI/CODE/DET goldens are template-derived**, payload-identical to
  `kimi-k3`'s (same XTML frames, same prompt-set Expected Template values,
  `tools/kimi_k3_template_goldens.py --model kimi-k3.1-pro`): the 1.5 TB MXFP4
  source needs `fla-core` and a multi-GPU runner, so no source oracle can be
  captured on this card. A green row pins the converted path against the
  template, never source parity.
- P5. **RESOLVED (2026-11-05) — archived.** The "4x slower than Kimi-K3" was
  never a regression: that baseline was prose committed before the run it
  described (K3's own log says 6.66–6.79 s/token, and at identical binary,
  geometry and window the two models measured 6.95 vs 6.98 — ratio 1.00). The
  defect it hid was the shared `ExpertStore`'s window: `max_open` 512 against a
  1472-expert working set (92 MoE layers × top-16, each visited once per token),
  i.e. a cyclic scan with a provably **zero** hit rate. Fixed Kimi-locally —
  clamp up to the working-set floor plus a default of 65 536, and
  `warm_on_load = false` because the eager pre-parse walks experts in path order
  and cost 156 s for nothing — which is **6.95 → 2.96 (K3) / 3.23–3.35 (Pro)
  s/token** at ~5 s load, bit-exact *as measured then*; the 2026-11-06 harness
  rows are the current rate (**K3 2.13/2.04**, **Pro 2.00/1.96** s/token — see
  P3's table), so treat the figures in this bullet as history. P5a closed with it: an evicted slot is now
  retired with a CUDA event instead of `cudaStreamSynchronize` (**20.3 → 0.07 ms**
  per eviction; 6.95 → 6.26 at a 512-slot window). Full trail — the sweep table,
  the two prefetch designs measured and rejected, and the 6.6% mistral-small-4
  regression this fix itself introduced and how it was found:
  [`open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md`](open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md).
- P6. **Pinned staging ring: built, tested, bit-exact — and NEUTRAL on this host
  (2026-11-05).** The entry used to say the remaining lever was to "stage layer
  N+1 while layer N computes". That is impossible: the expert ids of a layer are a
  function of the previous layer's output, so there is nothing to stage until the
  layer that would have been overlapped has finished. What was built instead is
  the other structural change — a pinned staging ring in the shared
  `ExpertStore` (persistent pread pool + per-slot DMA events, layer-wide
  prefetch), off by default, with `kimi-k3.1-pro` wired behind
  `LSX_EXPERT_STAGE_RING`. Design and full measurement table:
  [`superpowers/specs/2026-11-05-expert-stage-ring-design.md`](superpowers/specs/2026-11-05-expert-stage-ring-design.md).

  Measured, and this is why it is not switched on: the host-side copy cost drops
  62% (118.0 → 45.3 s of `cudaMemcpyAsync` inside a 32-token run) and the wall
  does not move (119.4–121.2 s with the ring against 122.5 s without). Two
  reasons, both measured. (a) This host keeps 665 GB of 750 GB in buff/cache, so
  the pageable path already copies at **23.7 GB/s**, not the 5.2 GB/s of the cold
  probe — pinning cannot beat a copy the cache already serves. (b) Expert staging
  is a minority of the traffic: **61.8 GiB/token goes H2D, routed experts are
  25.5 GiB (41%), and 36.4 GiB/token is the streamed latent/shared/dense plane**,
  which never enters `ExpertStore`. A perfect expert path therefore touches at
  most 41% of the copy time.

  So the lever this entry named was mis-scoped, and the honest remainder is the
  59%, not the 41%. Two follow-ups, both leaving the ring parked:
  - the **canonical 2048-token rows were not run** for either arm (~2.1 h each);
    the cells are `not run`, not a pass. They are what a future reader needs
    before quoting any of this;
  - the cold-cache win (26.9 ms vs 54.8 ms per layer, probe on real files) is
    **unverified end to end**, because this host's cache holds the tree. GPUDirect
    Storage would remove the host copy from the 59% instead of optimising the 41%
    — `nvidia_fs`/`libcufile` are still absent here.
