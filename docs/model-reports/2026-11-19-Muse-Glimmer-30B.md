# Muse-Glimmer-30B: a 30B agentic dense model, ingested end to end

**Date:** 2026-11-19 · **Hardware:** NVIDIA RTX 5090 (32 GB), CUDA 13.5 ·
**Branch:** `feat/muse-glimmer-30b-ingest`

## Publish-ready post

We converted and ran **Muse-Glimmer-30B** — Meta's 30-billion-parameter dense
agentic model — on a single consumer RTX 5090, end to end: conversion,
inference, tool calling, and a full correctness gate suite. The model now
generates at **~53 tokens/second** at a 2048-token budget, with a 554 ms
time-to-first-token, using 19.5 GB of the card's 32 GB.

### What the checkpoint forced into the engine

Muse-Glimmer is not a vanilla dense transformer. Four semantics are easy to
get wrong, and each is pinned by a test against a float64 reference derived
from the vendor's own `transformers` implementation:

1. **Gated attention.** Every layer carries a fifth attention projection
   (`self_attn.gate_proj`); the attention output is multiplied elementwise by
   `sigmoid(gate)` **before** the output projection. It is present on sliding
   *and* full layers.
2. **NoPE full layers.** The layer pattern is [sliding, sliding, sliding,
   full] × 13. The full-attention layers apply **no rotary embedding at all**
   (`layer_rope_theta == 0`) and attend the whole prefix; the sliding layers
   use θ = 500 000 over the full head dim with a 2048-token window. A uniform
   RoPE port decodes garbage past the first positions.
3. **Centered norms.** The four per-layer norms are Gemma-2-style `×(1+w)`
   (pre-norms ε 1e-5, post-norms ε **1e-8**); the final norm is plain `×w`;
   the embedding carries a **weightless** RMSNorm with no `√hidden` scale.
4. **Softcapped logits with a multiplier.** `logits = 20·tanh(raw·0.196116/20)`.
   Dropping the multiplier ships ~5.1× hot logits that greedy argmax hides —
   the head test uses a non-unit fixture to pin it.

The QK norm is weightless per-head RMSNorm, with Q additionally scaled by
**3.87** on top of the standard `1/√128` score scale.

### Quantization

All eight linear projections per layer are quantized to NVFP4 (E2M1 nibbles +
per-16 E4M3 block scales + a per-tensor float global scale). The embedding and
the **untied** lm_head stay FP16 (exact greedy), and no norm is ever quantized.
The 59.5 GB BF16 source converts to a **19 GB** tree in ~3 minutes. The vision
tower + adapter (809 tensors, ~3.8 GB) is counted and stripped, never applied —
this is a text-only runtime.

### Tool calling: byte-identical to the vendor template

Muse uses the **ATEM** protocol: harmony-style channels
(`<|start|>assistant to=<tool><|message|>…<|eot|>`) wrapping
`<atem:function_calls>/<atem:invoke>/<atem:parameter>` XML. We reproduced the
checkpoint's `chat_template.jinja` render **byte-for-byte** (verified against
HF `apply_chat_template` on both a tools conversation and a full multi-turn
tool-call/result conversation, 1762 and 2099 bytes respectively). The native
parser scans token ids for the answer channel (the markers are `special:true`,
so text decode drops them). A real-weight round-trip produces
`get_weather {"city":"Paris"}`.

### Measured gates

| gate | result |
|---|---|
| SelfCheck ("Paris") | **PASS ×3 bit-exact**, `logits_hash 3374b7bc94231955`, top-1 13796 |
| Tool round-trip (real weights) | **PASS** — `get_weather {"city":"Paris"}` |
| Cooperative cancel | **PASS** |
| CODE (compiler strictness, 3 prompts) | **3/3** vs source-captured golden |
| DET (determinism + near-tie, 3 prompts ×3 runs) | **3/3**, byte-identical runs |
| FSI (financial formulas, 3 prompts) | **debt** — capture attempted twice, see below |
| Throughput @2048 tokens | **52.9 t/s** (perf) / **53.3 t/s** (r3), ttft 554 ms |

### A defect the gates caught (and we fixed)

Muse is a reasoning model: it emits its answer in an `assistant to=user`
channel, then **loops back into `assistant to=self` reasoning**, re-emitting the
answer several times. The last answer channel can be truncated by the token
budget mid-answer, and the reasoning tail echoes the prompt body. The parity
harness's prompt-echo cut then removed everything and the extraction was empty.
We added ATEM answer-channel selection to the CODE/DET harnesses (select the
first *complete* answer channel), with unit tests written failing-first. Both
gates then passed on the converted weights.

### What was not measured, and why

**FSI** is declared debt, not a pass. The source runs (a text-tower CPU oracle
answers the Paris probe and the CODE/DET sets), but the FSI capture did not
complete: at 512 tokens FSI-03 aborts because the source loops inside its
reasoning channel and never emits a clean comparison line; at 1024 tokens the
3-case capture ran >1.5 h on CPU without finishing. The gate is wired and skips
for the golden, not the tree. A longer unattended capture (or a
`reasoning_strength: low` framing) is the next step.

Vision execution and the DFlash speculative drafter are out of scope (the
drafter is not in this checkpoint's index).

## Evidence

- **Design / plan:** `docs/superpowers/specs/2026-11-18-muse-glimmer-30b-ingest-design.md`,
  `docs/superpowers/plans/2026-11-18-muse-glimmer-30b-ingest.md`.
- **Open tasks:** `docs/OPEN_TASKS.md` §R21 (FSI capture open).
- **SelfCheck matrix:** `docs/SELFCHECK_MATRIX.md` §1 row `muse-glimmer-30b`
  (SelfCheck/CODE/DET) and §2 (FSI debt with the measured capture attempts).
- **Perf rows:** `tests/perf/perf_baselines.json` — `muse-glimmer-30b`,
  rtx 5090, linux: perf 52.86 t/s (ttft 554.41 ms, avg 468.55 W), r3 53.34 t/s
  (ttft 553.54 ms, avg 491.22 W), max_gen 2048, seeded 2026-11-19 through
  `test_perf_baseline` (AOCC `linux-release-znver4-aocc`).
- **Conversion:** `lsxforge --muse`, 52 layers, 809 vision tensors counted,
  1459 pinned records, VRAM profile min 18 GiB / optimal 27 GiB (ctx 131072);
  tree 19 GB at `<vault>/Muse-Glimmer-30B/logestix/`.
- **Gate runs (RTX 5090, one GPU workload at a time):**
  - `test_muse_dense_selfcheck`: 3 passed / 1 skipped (auto-path hook not
    declared); Paris ×3 bit-exact, tool round-trip, cancel.
  - `test_muse_dense_code_selfcheck`: 6 assertions, 1 case, PASS.
  - `test_muse_dense_det_selfcheck`: 6 assertions, 1 case, PASS (3×3 runs
    byte-identical).
  - `test_muse_dense_fsi_selfcheck`: SKIP (golden debt).
- **Synthetic suite (TDD evidence):** constants 108, convert 257, loader 207,
  model 24, kernels 12 047, pipeline 27, tool framing 3 725 assertions; all
  green. Guards: manifest_sync 1669, suite_guard 218, independence 1992,
  labels 39, naming 80. Fast sweep 153/153.
- **Goldens:** `tests/fixtures/{code,det}/muse-glimmer-30b.json` captured from
  the executed source via `tools/muse_golden_oracle.py` (627 text tensors
  loaded, 809 vision skipped; CPU, eager attention). HF render goldens:
  `tests/data/muse/hf_render_{tools,full}.txt`.
- **Harness fix:** `libs/lsxcommon/src/{code,det}_selfcheck.cpp`
  `SelectAnswerChannel`; unit tests in `tests/test_{code,det}_selfcheck.cpp`.