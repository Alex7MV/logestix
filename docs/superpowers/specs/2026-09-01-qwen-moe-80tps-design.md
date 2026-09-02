# Qwen3.5-35B-A3B — 80 tok/s decode on RTX 5090 — Design

Date: 2026-09-01
Branch: `feat/qwen-moe-80tps`
Status: approved design (approach A)

## Goal

Raise the DEFAULT production decode throughput of Qwen3.5-35B-A3B on the
RTX 5090 (Linux, AOCC release, full residency) from the measured Round-23
baseline of **63.5 tok/s** to **>= 80 tok/s**, measured by the standard
protocol: `test_blackwell_perf_moe "[qwen35moe]"`, FAST_MATH=1,
LSX_TEST_PERF_GEN_TOKENS=128, interleaved same-session, >= 3 alternations,
within-config spread < 1%.

## Correctness contract (STRICT, amended 2026-09-02)

- Every engaged configuration must keep bit-exact run-to-run
  reproducibility: selfcheck x3 " Paris" deterministic AND a two-call
  Infer `final_logits` memcmp gate.
- Numeric class (TC W4A4 fast_math) is NOT changed.
- AMENDMENT (operator decision, measured session): run-to-run
  determinism + selfcheck gate are BINDING; bitwise equality with the
  legacy per-slot chain is NOT required for the batched TC path on the
  real geometry (its split-K=8 slice composition diverges from legacy
  in a not-yet-root-caused way; synthetic no-split parity stays
  bitwise-pinned). Accepted at 101.1 tok/s.
- No env flags in the final default path; legacy chain remains as the
  fallback for partial residency.

## Baseline profile (Round 23 registry, Linux/5090)

- Wall: 15.7 ms/token (63.5 tok/s). GPU-step: 10.94 ms
  (ffn 64.8% | lin-attn 21.4% | full-attn 10.1% | head 3.6%).
  Host/submission gaps: ~4.7 ms/token.
- Per layer (40x/token): router top-k -> D2H slot readback + busy-poll
  (`qwen_moe_inference.cu:1442-1451`) -> per-slot TC chain
  {w13 GEMM -> swiglu-quant -> w2 GEMM -> coeff_add} x 8 slots.
- Per step: blocking `cudaEventSynchronize` on argmax readback
  (`qwen_moe_inference.cu:1584`).

## Approach (A — device-driven decode pipeline)

### Lever 1 — batched TC routed-expert chain with device slot table

New launcher `RunQwenMoeRoutedTcBatch` in `qwen_moe_inference.cu`:

- `grid.z = K` (8 slots) in the TC GEMM: ONE w13-batch launch
  (M=1, N=2I, K=H, per-slot weight pointers read from a device table,
  per-segment global scales {gs1|gs3} split at N/2 in the epilogue) ->
  batched elementwise swiglu-quant (K*I elements) -> ONE w2-batch launch
  (M=1, N=H, K=I) into per-slot partials `[K][H]` -> ascending-order
  fp32 coeff reduction into the accumulator.
- Bit-exactness by construction: same TC kernels, same per-slot math;
  the reduction replicates the legacy `coeff_add` order k=0..7
  ascending (the round-19b pattern proven for GEMV, ported to TC).
- Device-composed slot table `{dev_slot, gs1|gs3, gs2, coeff}` built
  from `d_routes` plus a new per-flat-expert TC-meta in
  `QwenMoeExpertStore` (mirrors `d_gemv_meta`, round-19c; TC meta
  staged at insert time in `EnsureQwenMoeExpertResident`). In
  steady-state full residency the host never reads the router back ->
  the per-layer D2H fence disappears.
- Engagement gate: full residency (all 40x256 experts device-resident —
  already the default since round 18) AND TC fast_math AND the fused
  slot layout AND `I % 16 == 0`. Any cache miss at any step -> the
  driver falls back permanently (for that generation) to the legacy
  per-slot chain, which stays the bit-identical default path.
  No env flags.

### Lever 2 — device-side token feed

Reference: kimi round 25 (`kimi_feed_token_kernel`). Argmax stays on
device; `qwen_feed_token_kernel` feeds `*d_argmax` into the next
embed-gather; `result.tokens` drained by ONE bulk D2H after the loop;
TTFT via events; EOS handled by host-truncate over the drained tokens
using `Tokenizer().eos_id` (kimi round-25 pattern; no hardcoded ids).
The per-step blocking `cudaEventSynchronize` (line 1584) is removed.

Race gate (kimi round-25 lesson): fully-async decode exposed an
un-root-caused nondeterminism in kimi. Before the default flip, the
two-call Infer `final_logits` memcmp gate MUST pass; if it fails, token
feed reverts to event-scoped readback (small share of wall time) and the
async lever is parked like round 25.

### Reserve (lever 3) — lin-attn decode fusion

21.4% of the GPU step budget (P2 from Round 22). Engaged ONLY if
levers 1+2 leave the default below 80 tok/s by the standard protocol.

## Time budget reasoning

Removing per-layer host handshakes (40/token) and the per-step argmax
sync targets the ~4.7 ms/token host-gap: 15.7 -> ~11 ms/token ≈ 90 tok/s
even without kernel-level gains. The batched launches additionally
compact the M=1 TC GEMM (grid.z=8 fills SMs). Headroom to 80 exists;
the reserve lever covers shortfall risk.

## Error handling

- Legacy fallback chain fully preserved (partial residency, unfused
  geometry, capture failures).
- CUDA errors: fail-fast with the existing named-stage
  `fail(e, "...")` pattern.
- Bit-exactness gates: any red test reverts the corresponding default
  (per kimi round-25 precedent); levers stay opt-in/latent then.

## Tests (TDD, in order)

1. `[tc-batch]` batched chain vs PINNED legacy per-slot chain:
   bitwise-equal `d_moe_acc` (same math, same order).
2. `[tc-batch]` adversarial table: duplicate experts, zero coeffs,
   alternating scale bytes (round-19 scale-mapping lesson).
3. `[tc-batch]` fallback parity: cache-miss path bit-identical to the
   current default.
4. `[decode]` run-to-run determinism: two Infer calls, `final_logits`
   memcmp (the kimi-race gate).
5. `[selfcheck]` real " Paris" x3, bit-stable.
6. Perf: default >= 80 tok/s by the Round-23 protocol; record
   before/after in the findings registry.

## Non-goals

- No CUDA Graph default (Round 23 measured neutral on Linux).
- No GEMV-experts default (Round 23 measured -17%).
- No head-GEMV default (measured -1.6%).
- No numeric-class changes; no W4A8/A8 experiments.
