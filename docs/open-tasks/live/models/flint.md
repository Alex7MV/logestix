# Recursive Flint — open items

### R6. Recursive Flint (39.8B-A2.5B) — ingestion landed, expert precision + goldens open (2026-10-30)

**Landed.** Family `flint` (`logestix-flint-nvfp4`) is registered and runs on the
real 75 GB checkpoint converted in place (`/mnt/fast_storage/huggingface/flint`,
22 GB tree, 12 032 routed experts resident in per-layer device slabs). All
stages of the Model Ingestion Protocol exist with their tests: converter
(synthetic, `convert`), loader (`fast`), kernels (`inference`), model lifecycle,
tokenizer (`tokenizer_kind` 4 — the 3-split digit/CJK + o200k case-run
pre-tokenizer; **parity 27/27 against the checkpoint's own `tokenizer.json`**),
native tool codec (**byte-identical to `chat_template.jinja`** on 5 captured
cases, including nested/parallel calls) and SelfCheck (**Paris ×3, bit-exact
`logits_hash`**, 49 layers / 22 266 MiB resident).

**What the checkpoint forced into the engine (measured, not stylistic).**
Flint carries *massive activations*: the source (BF16, `tools/flint_oracle.py`)
reaches ~1.4e4 in the residual at layer 48 and one layer amplifies a
29-magnitude input ~300x (gate projection 5.3e4, SwiGLU product ~1.5e8). FP16
cannot hold that, so this family keeps
- the residual stream in **FP32** (`RunFlintEmbedF32`, `RunFlintNormF32In`,
  `RunFlintAddHalfToF32`, `RunFlintAddF32`), and
- the FFN intermediates in range by **power-of-two scaling**
  (`kFfnInScale` 64 into the gate/up GEMV, `1/8192` out of the SwiGLU,
  restored on the accumulate/combine), so every projection still runs the FP16
  NVFP4 kernels.

Without this, layer 47 produced NaN on any prompt whose earlier tokens spiked
(the router then read uninitialised expert ids — that path is now also
defensive: a NaN score falls back to the lowest untouched ids instead of an
out-of-bounds shared read).

**Open: routed-expert precision.** A second NVFP4 plane (w − dequant(base),
`acc=true`) was added on the dense side — attention q/k/v/o, the dense-MLP
prefix and the shared expert (`"residual": "attn+mlp+shared"`, +343 records).
Measured against the CPU oracle on a 160-token tool prompt, the first-step
top-1 logit moved from -2.10 to -1.06 (prefix 56: -3.70 → -0.99). The remaining
divergence is the **routed experts at 4 bits**: top-1 agrees at prefix 8/16/32
and flips at ~64 (engine `4067` vs source `3705`). An 8-bit expert
representation needs +21 GB (4-bit set 21.3 GB + its residual plane), and the
card has ~9 GB free after the 22.3 GB resident tree, so the placement that would
clear it does not fit — the same conclusion as gemma-4-31b dense, measured
rather than assumed. Options for a later session: (a) stream the expert residual
plane from a host mmap with a slot LRU (≈282 MB/token at the observed routing
density), (b) a bigger card, (c) accept 4-bit and keep the parity rows declared.

**Golden debt.** FSI/CODE/DET goldens are *not* captured: the only oracle is the
CPU reference (`tools/flint_oracle.py`, 3-5 s/token), and the capture set is 9
prompts × 256-512 tokens plus DET's 3 repeat runs — hours. The oracle itself is
committed and was used to validate the engine (control prompt: 32 greedy tokens
token-for-token identical). The three targets therefore SKIP, with rows in
`test_model_suite_guard.cpp`'s `kGoldenDebt`.

**Perf.** No baseline row yet: `test_perf_baseline` requires a Release tree and
the `linux-release-znver4-aocc` rebuild was out of this session's budget. Debug
observation only: ~0.4-0.5 s/token at 160-token prefill (greedy), i.e. ~2 t/s —
the correctness-first path (token-by-token prefill, per-head serial softmax with
the sink column) has had no perf pass. Do not seed the row by hand.
