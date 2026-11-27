# Muse-Glimmer-30B: a 30B agentic dense model, ingested end to end

**Date:** 2026-11-19 · **Hardware:** NVIDIA RTX 5090 (32 GB), CUDA 13.5 ·
**Branch:** `feat/muse-glimmer-30b-ingest`, merged to `main` (`5cda3613`); the
FSI closure landed on `main` as `b579a362`

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
| FSI (financial formulas, 3 prompts) | **3/3** vs source golden, captured at `reasoning_strength: low` (11 s/case) |
| Throughput @2048 tokens | **52.9 t/s** (perf) / **53.3 t/s** (r3), ttft 554 ms |

### A defect the gates caught (and we fixed)

Muse is a reasoning model: it emits its answer in an `assistant to=user`
channel, then **loops back into `assistant to=self` reasoning**, re-emitting the
answer several times. The last answer channel can be truncated by the token
budget mid-answer, and the reasoning tail echoes the prompt body. The parity
harness's prompt-echo cut then removed everything and the extraction was empty.
We added ATEM answer-channel selection to the parity harnesses (select the first
*complete* answer channel), with unit tests written failing-first. All three
gates then passed on the converted weights.

The rule now lives in exactly one place —
`atem::SelectAnswerChannel`, `libs/lsxcommon/include/lsxcommon/atem_channel.h` —
consumed by the FSI, CODE and DET harnesses (the two copy-pasted versions are
deleted), with a capture-side twin in `tools/muse_golden_oracle.py` so a
golden's `expected` is the answer the verifier extracts from the same text.
Absent the channel fields a golden keeps the historical behaviour, so no
other model's gate moved.

### The FSI gate: two separate defects, and the measured result

FSI was the last gate and was declared debt twice before it was closed on
2026-11-19. Reading the failed run's own raw shows the red was **two defects at
once**, and neither was a quantization error:

1. **The extraction read the reasoning tail, not the answer channel.** The
   retained FSI-03 raw from the `high` 512-token attempt contains
   `assistant to=userDrawdown>Threshold` repeatedly — the model answered
   correctly — yet the run recorded `empty extraction`, because the prompt body
   it echoes (`Variables: Drawdown = 14.999991; Threshold = 15.000000.`) gave
   the extractor a `Drawdown`-plus-`=` line first, and the comparison pattern
   then found no identifier. This is the same defect the CODE/DET gates hit,
   fixed by the ATEM first-complete-channel rule.
2. **The framing decided which formula the model states.** The checkpoint's
   `chat_template.jinja` renders `Reasoning strength: <value>.` into the system
   block and defaults to `high`. At `high` the answer channel is not
   consistently the canonical formula: the 512-token attempt's extraction
   recorded `Balance_Alpha=100000000`, the prompt's own numeric echo, for
   FSI-02 (that raw was not retained, so whether the value came from the answer
   channel or from the echoed prompt cannot be separated — which is itself the
   reason the gate now reads a named channel). At `reasoning_strength: low` —
   the template's own switch — all three answer channels carry the canonical
   formula.

The earlier 1024-token attempt was killed after >1.5 h, and more budget could
not have fixed either defect: the extraction rule ignored the answer channel,
and the framing is set in the system prompt rather than by generation length.
Cost is token-bound anyway (~27 min/case at 512 tokens on the CPU tower, so
roughly double at 1024).

Capture recipe (source text tower, CPU, 512-token budget, `ignore_eos` on —
the reasoning model never emits EOS):

```bash
python3 tools/fsi_capture.py --model /mnt/fast_storage/lsx_model/meta-models/Muse-Glimmer-30B \
    --type muse-glimmer-30b --out tests/fixtures/fsi/muse-glimmer-30b.json \
    --max-new-tokens 512 --only FSI-TEST-01
python3 tools/fsi_capture.py ... --only FSI-TEST-02,FSI-TEST-03 --resume
```

One prompt per run, each completed case written to the golden atomically —
the tool used to write only after the third case, which is why the earlier kill
lost 1.5 h of CPU. Measured cost: ~27 min of wall clock per case (400–470
CPU-minutes across ~21 cores; 627 text tensors loaded, 809 vision tensors
skipped). The loop does **not** disappear at `low`: each raw still carries
12 / 7 / 37 `assistant to=user` channels and 0 / 5 / 0 `assistant to=self`
channels in 512 tokens — the first-complete rule is what makes that harmless,
and it is why the gate needs the channel fields and not a longer budget.

| case | source golden (`expected`) | answer-channel length | converted model (kOff, 512) | verdict |
|---|---|---|---|---|
| FSI-TEST-01 | `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` | 33 chars | `PV=CF*(1-exp(-r*n))/(1-exp(-r))` | pass (canonicalized) |
| FSI-TEST-02 | `Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C` | 71 chars | same (canonicalized to `Balance_Alpha=((100000000*Rate_A-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`) | pass |
| FSI-TEST-03 | `Drawdown>Threshold` | 18 chars | `Drawdown>Threshold` | pass |

`test_muse_dense_fsi_selfcheck` on the converted NVFP4 tree: **3/3, ~11 s per
case** (34.7 s for the target). `golden_debt` is now empty — a stale debt row
fails the suite guard — `docs/SELFCHECK_MATRIX.md` §1 reads 3/3 and its §2 row
is gone, and `docs/MODELS.md` prints `fsi: yes`.

Two capture artifacts are recorded rather than hidden: the captured `input_ids`
begin with a doubled `<|begin_of_text|>` (the template already emits `bos_token`
and the tool also requests special tokens; the CODE/DET goldens share it, and
the verifier replays the exact ids, so parity is unaffected), and
`MuseRenderPolicy::reasoning_strength` exists in the engine but no caller sets
it — **serving runs at `high` while the gate verifies `low`**.
A serve-time knob is the follow-up; the golden must be re-captured at that
strength if the served default changes, because the framing lives in
`input_ids`.

Vision execution and the DFlash speculative drafter are out of scope (the
drafter is not in this checkpoint's index).

## Evidence

- **Design / plan:** `docs/superpowers/specs/2026-11-18-muse-glimmer-30b-ingest-design.md`,
  `docs/superpowers/plans/2026-11-18-muse-glimmer-30b-ingest.md`.
- **Open tasks:** R21 closed 2026-11-19 —
  `docs/open-tasks/2026-11-19-muse-glimmer-30b-fsi-capture.md` (capture recipe,
  measured per-case cost, the two recorded artifacts).
- **SelfCheck matrix:** `docs/SELFCHECK_MATRIX.md` §1 row `muse-glimmer-30b`
  (SelfCheck/FSI/CODE/DET — all four measured, no debt row).
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
  - `test_muse_dense_fsi_selfcheck`: **3/3 passed**, 34.7 s for the target
    (~11 s/case); golden captured 2026-11-19 from the source text tower at
    `reasoning_strength: low`, 512-token budget, `ignore_eos` on, one prompt
    per run via `tools/fsi_capture.py --only ... --resume`.
- **Synthetic suite (TDD evidence):** constants 108, convert 257, loader 207,
  model 24, kernels 12 047, pipeline 27, tool framing 3 725 assertions; all
  green. Guards: manifest_sync 1669, suite_guard 220, independence 1992,
  labels 39, naming 80. Fast sweep 153/153.
- **Goldens:** `tests/fixtures/{fsi,code,det}/muse-glimmer-30b.json` captured from
  the executed source via `tools/muse_golden_oracle.py` (627 text tensors
  loaded, 809 vision skipped; CPU, eager attention; transformers 5.15.1 /
  torch 2.13.0+cu130). FSI is framed at `reasoning_strength: low` (the
  template's own switch, 190/239/156 prompt ids); all three goldens carry
  `answer_channel`/`reasoning_channel`, so the harness reads the first complete
  ATEM channel. HF render goldens:
  `tests/data/muse/hf_render_{tools,full}.txt`.
- **Harness fix:** `atem::SelectAnswerChannel` in
  `libs/lsxcommon/include/lsxcommon/atem_channel.h` (one implementation shared
  by FSI/CODE/DET; `FsiGoldenCase`/`CodeGoldenCase`/`DetGoldenCase` carry
  `answer_channel`/`reasoning_channel`); unit tests in
  `tests/test_{fsi,code,det}_selfcheck.cpp`, capture twin in
  `tools/muse_golden_oracle.py::select_answer_channel` (pinned by
  `tools/fsi_capture.py --selftest`).