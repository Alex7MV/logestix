# SelfCheck matrix — model x parity gate, as measured

Snapshot of **2026-10-24/25**, updated **2026-10-31** (GigaChat3.5-432B-A28B),
on the reference host (RTX 5090, CUDA 13.4, `linux-release-znver4-aocc`),
one test at a time (GPU exclusivity). Commit at capture: `f38fba84`;
GigaChat row measured on `feat/model-GigaChat3.5-432B-A28B-Reasoning`.

This is a **measurement**, not a policy. The machine-checked registry is the
source of truth for which target covers which model and stage:
`tests/test_model_suite_guard.cpp::Suites()` (model passport -> stage ->
ctest target). A row here can go stale; the registry cannot drift silently,
because it fails when a registered target is missing.

## What the four gates mean

| gate | oracle | cases |
|---|---|---|
| **SelfCheck** | built-in prompt ("Paris"), token + text + last-logits determinism across stress cycles | 1 (+ family extras, e.g. `*_eos_selfcheck`) |
| **FSI** | generated answer vs a golden captured from the *source* safetensors (`tools/fsi_capture.py`) | 3 prompts |
| **CODE** | same, compiler-syntactic strictness (`tools/code_capture.py`), exact match, no algebraic canonicalisation | 3 prompts |
| **DET** | same, plus R>=3 byte-identical `gen_tokens` runs (`tools/det_capture.py`) | 3 prompts |

`SKIP` (ctest exit code 4) is **not** a pass: it means the converted tree or the
golden is absent. A golden that was never captured looks exactly like a passing
one in a ctest summary, which is why the debt rows are called out below.

## 1. Converted tree in the vault **and** goldens present — real verdicts

| model (passport) | SelfCheck | FSI | CODE | DET |
|---|---|---|---|---|
| alice-foundation-80b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| glm-5.3-flash | PASS | 3/3 | 3/3 | 3/3 |
| **gemma-4-26b-a4b-it** | PASS | 3/3 | 3/3 | 3/3 |
| **gemma-4-31b-it** | PASS | **2/3** (FSI-01) | 3/3 | 3/3 |
| granite-4.2-30b | PASS | 3/3 | 3/3 | 3/3 |
| ornith-1.5-35b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| phi-4 | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.5-9b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.6-35b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.8-27b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.8-flash-next | PASS (+ eos) | 3/3 | 3/3 | 3/3 |
| xing4.0-29b-a4b | PASS | 3/3 | 3/3 | 3/3 |
| deepseek-v4-pro-0813 | PASS | **2/3** (FSI-02) | 3/3 | 3/3 |
| **minimax-m3** | PASS | **2/3** (FSI-01) | 3/3 | 3/3 |
| **minimax-m3.1-flash** | PASS | 3/3 | 3/3 | 3/3 |
| **hy-memory-30b-a3b** | PASS | 3/3 | 3/3 | 3/3 |
| **granite-switch-4.1-30b** | PASS | 3/3 | 3/3 | **2/3** (DET-02) |
| **gigachat3.5-432b-a28b-reasoning** | PASS | 3/3 | 3/3 | 3/3 |

- SelfCheck **18/18**, FSI **26/29**, CODE **29/29**, DET **28/29**.
- **GigaChat3.5-432B-A28B-Reasoning** landed green on all four gates in its
  2026-10-31 session (241 GB tree, 219 GB of routed experts pinned, 26.2 GiB
  resident). Its goldens are **template-derived** — the 432B FP8-block-128
  vendor reference is multi-GPU and the BF16 body (864 GB) exceeds this host's
  RAM — captured with the checkpoint's own `chat_template.jinja` and a 1536-token
  budget (FSI-02's reasoning trace overruns 512). SelfCheck is Paris ×3
  bit-exact (`logits_hash 2743ee1c26f859d5`, top-1 19904) and the native
  chatml-v5 tool round-trip returns `get_weather {"city":"Paris"}`. The
  tokenizer moved to `tokenizer_kind 3` (o200k case-run) with a 30/30 parity
  golden; the previous kind-0 split mis-split mixed-case identifiers. Perf by
  the harness: **11.32 t/s (perf) / 11.10 (r3)** at 2048 —
  `docs/model-reports/2026-10-31-GigaChat3.5-432B-A28B-Reasoning.md`.
- **Granite-Switch DET-TEST-02 is the third red.** The source CPU-BF16 oracle
  itself answers `CRITICAL` where the prompt's Expected Template is
  `CATASTROPHIC` (capture logged the disagreement); source and converted agree
  3/3 deterministically, so this is a source/prompt mismatch, not a 4-bit flip.
  See `docs/OPEN_TASKS.md` §R10.
- Two reds. The Pro FSI-02 red is an instruction-following deviation, not a
  numeric one: the prompt requires the symbolic form
  `Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`,
  and the model inlines the constants instead
  (`Balance_Alpha=((100000000*1.083541-0.0005)*83.152402-0.0005)*0.011104-0.0005`);
  FSI-01/03 pass and CODE/DET are 3/3. Its goldens are **template-derived**
  (authored Expected Template, Pro's own `encoding_dsv4.py` framing), because
  the TP-4 FP8 source runner cannot execute on this card — they pin the
  converted path only, not source parity (`docs/OPEN_TASKS.md` §R3).
- MiniMax-M3's FSI-01 red is the same class again: the prompt's Expected
  Template is `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` and the model answers the
  annuity-**due** variant `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)`. Its goldens
  are **template-derived** (the 428B BF16 source has no 32 GB-card execution
  path), and the converted path is additionally pinned by a host
  re-implementation over the source safetensors
  (`tools/minimax_reference.py`: same greedy continuation) plus a per-layer
  oracle against the checkpoint's own HF modeling code
  (`tools/minimax_hf_layer_oracle.py`). See `docs/OPEN_TASKS.md` §R4.
- MiniMax-M3.1-Flash (the sibling of MiniMax-M3: theta 1e7, MSA on every
  layer, pre-quantized MXFP8+NVFP4 source) landed green on **all four gates**
  in its 2026-10-31 session, each run serially. Its goldens are
  **template-derived** like M3's (the 428B QAT source has no 32 GB-card
  execution path), captured with the M3.1 tokenizer + `chat_template.jinja`
  (`input_ids` differ from M3's by the new system prompt / model-id suffix).
  SelfCheck is bit-exact across three adaptive-thinking cycles
  (`logits_hash 2daad7a5ba024549`) — the answer lands after the model's own
  `<mm:think>` block, so the probe window is 192 tokens. Perf by the harness:
  6.23/6.18 t/s at 2048 — `docs/open-tasks/2026-10-31-minimax-m31-flash-ingest.md`.
- Hy-SimulMT-30B-A3B (the sibling of Hy-Memory with the same text backbone)
  is PASS on SelfCheck and declared debt on the three QA gates: it is a
  translation model, and the measured source output for those prompts is a
  summary (or immediate EOT), not an answer. The serving contract (the
  checkpoint's own direction directive, auto-selected from the source script)
  is pinned by a real-weight bidirectional test — `docs/OPEN_TASKS.md` §R11.
- Hy-Memory-30B-A3B landed green on all four gates with **source-derived**
  goldens (`tools/hy_v3_oracle.py` assembles the fused expert tensors the stock
  HF class expects; the 60 GB bf16 source runs on CPU). FSI-02 answers the
  numeric form, which is exactly what the source answered, so the golden is
  source parity rather than a template match. See `docs/OPEN_TASKS.md` §R5.
- The other red is the dense Gemma-4 member, and it is understood rather than
  mysterious: FSI-01 is a template near-tie where the dense body needs ~8 bits
  on *every* linear (33.7 GiB resident, over this card). Every partial
  placement that fits was measured and moves *away* from the source —
  `docs/OPEN_TASKS.md` §R and
  `open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md` §5.
- gemma-4-26b went green in this session (second NVFP4 plane on attn + dense
  MLP, routed experts left 4-bit): FSI/CODE/DET were 1/3, 2/3, 3/3 before.

## 2. Converted tree present, **no golden** — declared debt (FSI/CODE/DET SKIP)

| model | SelfCheck | FSI | CODE | DET | why |
|---|---|---|---|---|---|
| mistral-small-4-119b | PASS | SKIP | SKIP | SKIP | source is the Mistral "consolidated" layout (`params.json`, no `config.json`), so Transformers cannot run it and no oracle can be captured — `docs/OPEN_TASKS.md` §T |
| flint | PASS | SKIP | SKIP | SKIP | source runs only on CPU (`tools/flint_oracle.py`, ~3-5 s/token); the capture set is hours long. The converted model is additionally parity-limited: the 4-bit routed-expert plane is the dominant residual error (an 8-bit expert set needs +21 GB, the card has ~9 GB free) — `docs/OPEN_TASKS.md` §R6 |
| beam | PASS | SKIP | SKIP | SKIP | the 934 GB BF16 source (501B) exceeds this host's 750 GB RAM, so the reference cannot be loaded for a capture; kernel parity + SelfCheck are the instruments. Two real-weight defects (the lost `<|call|>` stop id and a reserved-id decode failure) were found and fixed — `docs/OPEN_TASKS.md` §R8 |
| ling | PASS | SKIP | SKIP | SKIP | source is 560B BF16/FP8 with GPU-only FLA KDA kernels; no oracle runnable on one RTX 5090 (a CPU reference was used for single-token probes only). Per-expert global scales + the official SwiGLU clamps were required for a coherent answer — `docs/OPEN_TASKS.md` §R7 |
| step-5-preview | PASS | SKIP | SKIP | SKIP | converted tree present (317 GB, Paris x3 bit-exact, native tool round-trip green); the 606 GB FP8 source oracle (reference `inference/` stack) needs 4x B300 / 8x H200 per the model card, so no source golden can be captured here. The DSA indexer is stored-not-applied (dense-equivalent at <=4096 tokens) — `docs/OPEN_TASKS.md` §R9 |
| hy-simulmt-30b-a3b | PASS | SKIP | SKIP | SKIP | zh<->en simultaneous translator: the source summarises/translates the FSI/CODE/DET bodies instead of answering (two emit EOT immediately), so no source-faithful golden can be extracted; instruments are SelfCheck (translation direction) + a real-weight bidirectional translation test — `docs/OPEN_TASKS.md` §R11 |
| nemotron-3.5-lightning-30b-a3b-nvfp4 | PASS | SKIP | SKIP | SKIP | the source IS runnable now — `tools/nemotron_nvfp4_oracle.py` dequantizes the ModelOpt checkpoint (W4A16_NVFP4 experts + FP8 mixers) to a CPU BF16 `NemotronHForCausalLM` and it answers FSI 3/3 per the Expected Template — but the converted tree does not reproduce them (FSI-01 extracts `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)`, FSI-02/03 emit fenced/repeated bodies), and CODE-TEST-01 / DET-TEST-01/02 make the SOURCE echo the prompt body, so no source-faithful CODE/DET golden can be extracted. Gates stay debt with these measurements; the FSI gap is the 4-bit body precision item in `docs/OPEN_TASKS.md` §R13 |

Both are recorded in the golden-debt list (§K) rather than left to look green.

## 2b. flint — measured engine-vs-source agreement (no golden yet)

`tools/flint_oracle.py` (CPU, BF16, built on the HF Qwen3-MoE primitives plus the
vendor's five deltas) is the external oracle. On the control prompt the
converted engine reproduces the source's greedy continuation token-for-token for
32 steps. On a 160-token tool prompt the top-1 token agrees at prefixes 8/16/32
but diverges at ~64 (engine `4067` vs source `3705`), with first-step logit
deltas of 1-4. The dense-side residual plane (attention + dense MLP + shared
expert) closed part of that gap (prefix 56: top-1 logit -3.70 -> -0.99; prefix
160: -2.10 -> -1.06); the remainder is the routed-expert 4-bit error. This is a
**measurement**, not a pass: the FSI/CODE/DET rows stay SKIP until a capture
fits the budget and the expert precision question is settled.

## 2c. beam — real-weight verification (no golden)

The converted tree (265 GB) exists and SelfCheck answers Paris with the Harmony
framing; the real-weight native tool round-trip returns
`get_weather {"city":"Paris"}` (verified 2026-10-31). The numeric contract is
pinned per kernel against `beam_moe_device_math.h` (plain/per-head RMSNorm,
rotate-half RoPE, NoPE SSMax, sliding-window bounds, group-limited sigmoid
router with correction bias, per-head attention gate, depth-scaled residual,
softcap). FSI/CODE/DET stay SKIP: the BF16 source (934 GB) cannot be loaded in
750 GB of host RAM, so there is no oracle to capture from — a declared debt, not
a pass. Perf is also open (Debug-only observation; rows must come from the
Release harness; now seeded at **1.61 t/s (perf) / 1.60 t/s (r3)**, TTFT ≈27 s,
2048 tokens) — `docs/OPEN_TASKS.md` §R8.

## 3. Golden present, **no converted tree** — SKIP (needs the vault, not a code fix)

`deepseek-v4.1-flash`, `qwen3.5-35b-a3b`. Their FSI/CODE/DET targets run only
when the converted tree exists; SelfCheck skips with them.

## 4. Neither — SKIP

`glm-5.2`, `deepseek-v4-flash`, `nemotron-3.5-lightning-30b-a3b` (bf16 sibling),
`mimo-v2.6-flash-rl`. Registered stages exist; converted trees and goldens do
not. `gigachat3.5-432b-a28b-reasoning` moved out of this section on 2026-10-31
(241 GB tree, all four gates measured). `deepseek-v4-pro-0813` moved out of
this section on 2026-10-26: the 838 GB tree exists and SelfCheck passes on real
weights (Paris x3, bit-exact logits) — its FSI/CODE/DET remain declared debt
because the source oracle is the TP-4 FP8 runner (`docs/OPEN_TASKS.md` §R3).

`kimi-k2.7-code` moved out of this section on 2026-10-26: the tree now exists
(542 GB, 23040 experts) and **SelfCheck passes on real weights** — Paris
(tok 17374), 3 cycles, identical logits hash — with the per-layer reference
oracle green (relL2 0.0034 at layer 0, 0.1226 at layer 60, cos ≥ 0.9925). FSI,
CODE and DET remain **declared golden debt** (`docs/OPEN_TASKS.md` §R2 K1): the
581 GB `compressed-tensors` source has not been run for a capture yet, so their
three targets SKIP (tree present, golden absent) — the debt rows in
`kGoldenDebt` are the honest state, and the capture must come from the source,
never from the engine. Verification record:
`docs/open-tasks/2026-10-26-kimi-audit-verification.md`.

## Reproduce

```bash
# 1. the real verdicts (12 models with a tree + goldens), one at a time
ctest --test-dir build/linux-release-znver4-aocc --timeout 1500 -R \
  "alice.*selfcheck|gemma_(moe|dense).*selfcheck|glm_moe_flash.*selfcheck|\
granite_dense.*selfcheck|nemotron_moe_nvfp4.*selfcheck|ornith_moe.*selfcheck|\
phi_dense.*selfcheck|qwen_dense.*selfcheck|qwen_moe_36.*selfcheck|\
qwen_moe_exp.*selfcheck|xing_moe.*selfcheck|gigachat_moe.*selfcheck"

# 2. the debt rows (SelfCheck only; the three parity stages SKIP by design)
ctest --test-dir build/linux-release-znver4-aocc -R "mistral.*selfcheck"

# 3. the absent-vault families (all SKIP; proves the SKIP is the vault, not code)
ctest --test-dir build/linux-release-znver4-aocc -R \
  "deepseek_moe(_v41)?_selfcheck|glm_moe_selfcheck|kimi_moe_selfcheck|\
mimo_moe_selfcheck|nemotron_moe_selfcheck|qwen_moe_selfcheck"

# 4. the harness' own self-tests (no model): formula extraction/canonicalisation
ctest --test-dir build/linux-release-znver4-aocc -R \
  "test_(fsi|code|det)_selfcheck"
```

The parity stages share one rule: the golden is captured from the source
safetensors and is **never reconciled** to make a conversion pass. When a gate
goes red the question is which instrument is missing, not how to rewrite the
expectation (`docs/OPEN_ENDEDNESS.md`).
