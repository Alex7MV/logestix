# SelfCheck matrix — model x parity gate, as measured

Snapshot of **2026-10-24/25**, updated **2026-10-31** (GigaChat3.5-432B-A28B),
re-verified **2026-11-03** (AliceAI-Foundation-80B-A3B re-ingestion), re-verified
**2026-11-04** (Qwen3.8-Flash-Next full lifecycle; Qwen3.5-35B-A3B and
DeepSeek-V4.1-Flash audited — both had been filed as "no converted tree" while
their trees were in the vault),
re-verified **2026-11-06** (AliceAI-Foundation-80B-A3B
after the inverted workspace predicate was fixed — its row had been unmeasured
since 2026-11-03 19:33, when generation stopped producing tokens at all:
`open-tasks/2026-11-06-alice-workspace-polarity.md`), extended **2026-11-07**
(Granite-Switch 4.1 8B ingested as the sibling of the 30B — see
`docs/model-reports/2026-11-07-Granite-Switch-4.1-8B.md`), extended
**2026-11-10** (NVIDIA-Nemotron-Labs-Dory1.0-4B, looped nGPT — its DET row is
captured in reasoning mode: the letter-match probe answers CRITICAL without a
reasoning trace and CATASTROPHIC with it),
extended **2026-11-15** (Inkling-2-Small: 594B total / 17B active, mixed
NVFP4+MXFP8 experts, all four gates measured on the real 366 GB tree),
extended **2026-11-21** (Inkling-2-Mini: the looped `inkling2_loop`
sibling — 24 layers of which 12 run three passes, so 48 applications per token;
SelfCheck reproduces the release's own greedy ids, and its FSI golden is the
first in the family captured by RUNNING the source rather than from its
template),
closed **2026-11-19** (Muse-Glimmer-30B: the FSI golden that had been declared
debt was captured from the source text tower at `reasoning_strength: low` — the
chat template's own switch, which makes the answer channel the canonical formula
— one prompt per run, and the converted model reproduces all three answers;
CODE/DET came from the same oracle the day before),
on the reference host (RTX 5090, CUDA 13.4, `linux-release-znver4-aocc`),
one test at a time (GPU exclusivity). Commit at capture: `f38fba84`;
GigaChat row measured on `feat/model-GigaChat3.5-432B-A28B-Reasoning`;
`kimi-k3.1-pro` row measured 2026-11-05 on `feat/kimi-k31-pro` (tree converted
that day; SelfCheck green, the parity sweep declared debt — §1 and
`docs/OPEN_TASKS.md` §R17).

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
| granite-switch-4.1-8b | PASS (x3, `930dd592ea5f5834`) | 3/3 | 3/3 | **2/3** (DET-02) |
| glm-5.3-flash | PASS | 3/3 | 3/3 | 3/3 |
| **gemma-4-26b-a4b-it** | PASS | 3/3 | 3/3 | 3/3 |
| **gemma-4-31b-it** | PASS | **2/3** (FSI-01) | 3/3 | 3/3 |
| granite-4.2-30b | PASS | 3/3 | 3/3 | 3/3 |
| ornith-1.5-35b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| phi-4 | PASS | 3/3 | 3/3 | 3/3 |
| **dory-1.0-4b** | PASS (x3, `f100088895adb12d`) | 3/3 | 3/3 | 3/3 (reasoning mode) |
| qwen3.5-9b | PASS | 3/3 | 3/3 | 3/3 |
| **qwen3.5-35b-a3b** | PASS (x3, `f4c91be5f2206d56`) | 3/3 | 3/3 | 3/3 |
| qwen3.6-35b-a3b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.8-27b | PASS | 3/3 | 3/3 | 3/3 |
| qwen3.8-flash-next | PASS (+ eos) | 3/3 | 3/3 | 3/3 |
| xing4.0-29b-a4b | PASS | 3/3 | 3/3 | 3/3 |
| **mistral-large-4-1t** | PASS (x3, `747bf131f1ac557f`) | 3/3 (template) | 3/3 (template) | 3/3 |
| deepseek-v4-pro-0813 | PASS | **2/3** (FSI-02) | 3/3 | 3/3 |
| **deepseek-v4.1-flash** | PASS (x3, `b759bc84b820b4de`) ‡ | **1/3** | **1/3** | **2/3** (DET-03) |
| **minimax-m3** | PASS | **2/3** (FSI-01) | 3/3 | 3/3 |
| **minimax-m3.1-flash** | PASS | 3/3 | 3/3 | 3/3 |
| **hy-memory-30b-a3b** | PASS | 3/3 | 3/3 | 3/3 |
| **granite-switch-4.1-30b** | PASS | 3/3 | 3/3 | **2/3** (DET-02) |
| **gigachat3.5-432b-a28b-reasoning** | PASS | 3/3 | 3/3 | 3/3 |
| **kimi-k3** | PASS (x3, `5c1e227265d6a27b`, framed) | not run (template golden) | not run (template golden) | not run (template golden) |
| **kimi-k3.1-pro** | PASS (x3, `98ec1bbb372c9181`, framed) | not run (debt: ~1.1 h/case) | not run (debt) | not run (debt) |
| **inkling-2-small** | PASS (x3, `323d30463a3d2b72`, framed) | 3/3 (template) | 3/3 (template) | 3/3 |
| **inkling** (base) | PASS (x3, framed, inline-reasoning) | 3/3 (template) | 3/3 (template) | 3/3 |
| **inkling-2-mini** | PASS (x3, `12e684ecd27e40d7`, framed) | 3/3 (source) | 3/3 (source) | 3/3 |
| **muse-glimmer-30b** | PASS (x3, `3374b7bc94231955`, raw) | 3/3 | 3/3 | 3/3 |

- **All 37 SelfCheck targets were re-run serially on 2026-11-07, after the probe
  was restored** (120 min, `-j1`, one GPU). The list is
  `tools/gen_model_manifest.py --print-selfcheck-targets`, derived from the
  manifests — the previous hand-typed filter covered 22 of 37, and an omitted gate
  reads exactly like a passing one.
  **"37 pass" was the wrong reading of that run, and the correction matters more
  than the run:** ctest reported `Passed` for all 37, but only **27** had a
  converted tree and therefore executed the Paris case; the other **10** (the
  models in §4 and the vanished-tree rows) skipped every case that touches weights
  and were green only because an *undeclared-hook* case `SUCCEED`ed in the same
  binary. One non-failing case is enough to keep Catch2 from returning
  `AllTestsSkippedExitCode`, so `SKIP_RETURN_CODE 4` never armed. Those branches
  now `SKIP`, and the honest headline is **27 of 37 measured, 10 skipped with their
  tree named** — `docs/OPEN_TASKS.md` §G6.
  The rows below that carry a hash were checked against the values this run
  produced, not against the previous capture.
- **The SelfCheck probe is per-family, and it is now declared.** Unifying the 35
  per-family selfcheck files into one driver silently flattened four knobs that
  each file had chosen for itself: the prompt framing (17 files fed
  `BuildPromptInputIds`, 18 fed raw BPE), the generation budget (2 for most;
  16/24/32/64/192/256 for the rest), the math mode (14 pinned `FastMathMode::kOff`,
  21 left the kAuto default), and — separately — whether the case also asserted the
  raw greedy continuation (12 files did). All three numeric knobs feed the recorded
  `logits_hash`, so a uniform driver re-baselined every hash in this table and made
  the column unreproducible. They are now manifest fields (`selfcheck_frame_prompt`,
  `selfcheck_max_gen`, `selfcheck_bit_exact`), transcribed from the deleted files
  rather than guessed, and the extra continuation assertion was dropped because the
  canonical contract is `SelfCheck`.
  **Evidence, not intent:** the 2026-11-06 17:08 release binaries predate the
  migration, and replaying four of them against the restored driver reproduces the
  old hashes exactly — phi-4 `cf51b9bbd5c913dd`, ornith `c3fe24e9dbc55b1e`,
  qwen3.5-9b `a2af669548e59c73`, xing `53e18f6dd77a64a2` (top-1 and token count
  identical too). Two of those four were *red* under the uniform driver.

- **A §1 row is a measurement, and three of them are no longer reproducible.**
  Audited 2026-11-07 against the vault: `deepseek-v4-pro-0813`, `gigachat3.5-432b-a28b-reasoning`
  and `mistral-large-4-1t` have **no `logestix/model.index.json`** any more, so
  their three parity gates SKIP today (a SKIP, not a pass) while the row records
  measured cells. The verdicts were real when taken; what is gone is the artifact.
  Two causes, both recorded as open: gigachat was converted into
  `/mnt/fast_storage/tmp/…` and the tmp tree was cleaned (`docs/OPEN_TASKS.md` §G4),
  and the other two are 838 GB / 988 GB trees on a host at 94% disk. Re-running
  those rows needs a re-conversion, not a code change — so the cells stay as
  history and this note is the current state.
- SelfCheck **24/24 rows** (every row above), FSI **61/66**, CODE **64/66**,
  DET **63/66**. Counting rule: one case per prompt, three prompts per measured
  row, so the denominators are 3 × the rows that carry a measured cell (22 —
  `kimi-k3` and `kimi-k3.1-pro` have no parity verdict yet); a `not run` cell is
  excluded rather than counted as a pass. The five FSI reds are gemma-4-31b
  FSI-01, deepseek-v4-pro FSI-02, minimax-m3 FSI-01 and deepseek-v4.1-flash
  FSI-01/02; the two CODE reds are deepseek-v4.1-flash CODE-01 and CODE-03; the
  DET reds are granite-switch-4.1-30b DET-02,
  granite-switch-4.1-8b DET-02 and deepseek-v4.1-flash DET-03. Both
  granite-switch DET-02 cells are the *source's own* answer (`CRITICAL` where the
  prompt template says `CATASTROPHIC`), so they are family behaviour, not engine
  defects.

  **DET-03 for deepseek-v4.1-flash was recorded as 3/3 until 2026-11-07** and is
  actually 2/3: over three identical runs (deterministic, so not a flake) the
  model answers `BIT_ERO` where the prompt's template says `BIT_ZERO` — it drops
  the `Z`, and reasons its way to the drop (`...Final only BIT_ERO.</think>BIT_ERO`).
  The earlier green was therefore either measured against a different converted
  tree or mis-recorded; the value here is the measured one, from the 294 GB tree
  currently in the vault. Two instrument notes came out of the same run: the DET
  reporter used to fold an inference error into `deterministic=false`, and its
  verdict line embedded the model's raw newlines, so a line-oriented reader
  (`grep`) saw only a prefix of the raw and a real `BIT_ERO` answer looked like a
  clean `BIT_ZERO`. Both fixed in `libs/lsxcommon/src/det_selfcheck.cpp`; the raw
  is now escaped onto one line.
- **kimi-k3.1-pro** added 2026-11-05 (branch `feat/kimi-k31-pro`): the 1.5 TB
  source converted in place to a **1.5 TB** tree (1.4 TB routed-expert arrows,
  23 GB pinned graph, 8.9 GB streamed latent/shared plane) and SelfCheck is green
  on the real weights — Paris ×3 bit-exact, plus the native XTML tool round-trip
  `get_weather {"city":"Paris"}` — so it earned its move out of §3. The three
  parity cells are **not run, and that is a measurement, not a formality**. The
  decode rate is now a harness row rather than prose: **0.50 t/s (perf) / 0.51
  (r3) at 2048 tokens = 2.00 / 1.96 s per token**, seeded 2026-11-06 through
  `test_perf_baseline` (the perf half of §R17 P3 is closed; FSI/CODE/DET remain
  debt). That replaces the earlier 3.23–3.35 s/token estimate, which was
  log-derived — and it also removes the last trace of P5's premise: **the Pro
  measures faster than its parent** (`kimi-k3` seeds at 0.47/0.49 t/s = 2.13/2.04
  s per token), so the indexer is not a net cost at this sequence length. The
  history of the rate: 6.95 s/token when it first landed, when the window was 512
  against a 1472-expert working set — a cyclic scan with a zero hit rate
  (`docs/open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md` has the
  sweep and the two prefetch designs that were measured and rejected) — then the
  eviction path in the same loop stopped stalling the host (P5a, archived in the
  same file). At the measured rate the remaining FSI+CODE+DET sweep is ~7 h of
  exclusive GPU, and one 2048-token FSI case is ~1.1 h. An FSI
  run was started and stopped 97 min in — the kill is not a verdict, hence `not run`
  rather than a red. Its goldens are **template-derived** and payload-identical
  to `kimi-k3`'s, so a future green row pins the converted path against the
  template, **not** source parity; and no gate at all observes the DSA indexer,
  whose selection is dormant at the canonical 2048-token protocol. Debt,
  budgets and how to close: `docs/OPEN_TASKS.md` §R17 (P1–P5).
- **Qwen3.5-35B-A3B** was audited on 2026-11-04 (it had been listed under §3 as
  "no converted tree" while its 19 GB tree had been in the vault since
  2026-10-01): SelfCheck Paris ×3 bit-exact (`logits_hash f4c91be5f2206d56`,
  top-1 11751), FSI/CODE/DET 3/3 each with DET `deterministic=true`. Its two
  benchmark rows were stale in opposite directions and were re-seeded — `perf`
  170.10 → 159.81 (the accepted overflow-proof SwiGLU cost its two 35B-A3B
  siblings already carry) and `r3` 13.78 → 77.56 (a superseded strict-mode value,
  5.6× below what the same binary does) — and the missing
  `qwen_moe.router.decode` budget for `rtx 5090` is now recorded —
  [`open-tasks/2026-11-04-qwen35-parity-and-open-tasks-hygiene.md`](open-tasks/2026-11-04-qwen35-parity-and-open-tasks-hygiene.md).
- ‡ **deepseek-v4.1-flash**: the Paris SelfCheck case passes, and since
  2026-11-07 so does the cooperative-cancel case in the same binary. That case
  used to fail (`mid-generation abort: compressor state reset failed`) because a
  cancelled request never recorded its TTFT event and left
  `cudaErrorInvalidResourceHandle` in the thread's last-error state, which the
  **next** request then reported at an unrelated site; root cause and fix:
  `docs/OPEN_TASKS.md` §A2, which also records that the case had stopped running
  at all when the selfcheck driver was unified (§G5).
  Two further corrections, both measured 2026-11-07:
  **CODE is 1/3, not 2/3.** CODE-01 answers with prose instead of the JSON
  envelope (extracted empty) and CODE-03 appends a stray `?` to the regex. The
  same two cases are red on the **pre-refactor** binary built 2026-11-06 17:08
  (`/tmp/pre_refactor/`, string-checked to predate the shared driver), so this is
  a matrix error, not a migration regression.
  **The Paris hash is `b759bc84b820b4de`, not `0ab7077b529eaa3b`,** because of a
  deliberate fix: `total_steps` was `prompt_len + max_gen`, which produced
  `max_gen + 1` tokens (the streaming control run reported 25 for a 24 budget)
  while every other family produces exactly the budget. Reverting just that line
  reproduces `0ab7077b529eaa3b` bit-exactly, which is how the delta was attributed.
  The FSI and DET grids are unchanged by it (both re-measured after the fix).
- **Qwen3.8-Flash-Next** was re-verified end to end on 2026-11-04 (every stage,
  real weights): SelfCheck Paris ×3 bit-exact (`logits_hash 2e3bcd090441ad65`,
  top-1 11751) plus the family's EOS and cross-instance gate, FSI/CODE/DET 3/3
  each, native tool round-trip green. Framing is now template-exact for **every**
  qwen member, not just this one: `render_content|trim` (message content only —
  never tool definitions, parameter names or argument values) and jinja `tojson`
  tool serialisation are unconditional in the shared codec, and only the three
  genuinely per-family fields live in `QwenRenderPolicy` (`enable_thinking`, the
  3.8 `preserve_thinking` default `history_think_block`, and its
  `suppress_empty_system` guard). Pinned against each checkpoint's own
  `chat_template.jinja` in `test_qwen_tool_framing` and, for the model's actual
  policy, in `test_qwen_moe_exp_model`; all six members' real-weight tool
  round-trips are green on that wire — [`open-tasks/2026-11-04-qwen-template-fidelity.md`](open-tasks/2026-11-04-qwen-template-fidelity.md).
  Throughput is expert-cache **capacity** bound
  (43.39 perf / 33.91 r3 t/s at 5350/4319 slots; hit rate 0.86), and the
  section that tracked it left the live list when it closed —
  [`open-tasks/2026-11-04-qwen38-flash-next-lifecycle.md`](open-tasks/2026-11-04-qwen38-flash-next-lifecycle.md)
  §6.
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
- **Granite-Switch DET-TEST-02 is the DET red.** The source CPU-BF16 oracle
  itself answers `CRITICAL` where the prompt's Expected Template is
  `CATASTROPHIC` (capture logged the disagreement); source and converted agree
  3/3 deterministically, so this is a source/prompt mismatch, not a 4-bit flip.
  See `docs/OPEN_TASKS.md` §R10.
- The Pro FSI-02 red is an instruction-following deviation, not a
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
- The dense Gemma-4 member is the last red, and it is understood rather than
  mysterious: FSI-01 is a template near-tie where the dense body needs ~8 bits
  on *every* linear (33.7 GiB resident, over this card). Every partial
  placement that fits was measured and moves *away* from the source —
  `docs/OPEN_TASKS.md` §R and
  `open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md` §5.
- **Inkling-2-Small** is green on all four gates with **template-derived**
  FSI/CODE/DET goldens — the source is 594B with its own NVFP4/MXFP8 stack and
  needs 500 GB of aggregated VRAM, so it cannot be executed here (the same
  deviation GLM-5.3-Flash records). It is also the family whose red gate turned
  out to be a CONVERSION defect that every byte-level check and both of my
  references reproduced: the checkpoint's fused `w13_weight` is row-interleaved
  and the converter split it in half, which cost the model its knowledge
  (perplexity 12 881 -> 30.4 after the fix). See
  `docs/model-reports/2026-11-15-Inkling-2-Small.md`.
- **Inkling (base)** is green on all four gates with **template-derived**
  FSI/CODE/DET goldens (952B pure-BF16 source, not runnable locally). The SAME
  row-interleave defect recurred on the routed `w13_weight` (the release
  `Interleave(dim=1)` de-interleaves it; the converter split it in halves) and
  was caught the same way — the release modelling code driven on real weights
  is the only oracle that sees it. The base release additionally forces an
  **FP32 residual stream**: its residual genuinely reaches ~1e6 by the deep
  layers (the model's own bf16 behaviour), overflowing FP16 at five points
  (residual, residual-path convs, expert/shared down GEMV, wo projection).
  Knowledge sanity perplexity **3.84**; framed-Paris generation answers
  "The capital of France is **Paris**." after an inline-reasoning span. CODE
  uses `answer_start_token_id=200004` (`<|content_text|>`) to cut to the final
  answer. See `docs/model-reports/2026-11-18-Inkling.md`.
- gemma-4-26b went green in this session (second NVFP4 plane on attn + dense
  MLP, routed experts left 4-bit): FSI/CODE/DET were 1/3, 2/3, 3/3 before.

## 2. Converted tree present, **no committed golden** — declared debt (FSI/CODE/DET SKIP or measured red)

| model | SelfCheck | FSI | CODE | DET | why |
|---|---|---|---|---|---|
| mistral-small-4-119b | PASS | 1/3 | 0/3 | 0/3 | the three gates were **run** on 2026-11-06 with prompt-derived goldens and measured, not SKIPped; they stay declared debt because a prompt-derived expected value cannot establish parity. The model is coherent but off-format (DET is deterministic across three runs). A source-faithful golden is blocked by the checkpoint's tensor names/expert layout (50 missing / 3118 unexpected vs `Mistral4ForCausalLM`), not by its config — `docs/OPEN_TASKS.md` §T.1, tool `tools/mistral_small4_oracle.py` |
| flint | PASS | SKIP | SKIP | SKIP | source runs only on CPU (`tools/flint_oracle.py`, ~3-5 s/token); the capture set is hours long. The converted model is additionally parity-limited: the 4-bit routed-expert plane is the dominant residual error (an 8-bit expert set needs +21 GB, the card has ~9 GB free) — `docs/OPEN_TASKS.md` §R6 |
| beam | PASS | SKIP | SKIP | SKIP | the 934 GB BF16 source (501B) exceeds this host's 750 GB RAM, so the reference cannot be loaded for a capture; kernel parity + SelfCheck are the instruments. Two real-weight defects (the lost `<|call|>` stop id and a reserved-id decode failure) were found and fixed — `docs/OPEN_TASKS.md` §R8 |
| nemotron-ultra-550b-a55b | PASS | 2/3 | PASS | PASS | separate lineage from the 30B Lightning sibling (own passport, converter, loader, kernels, pipeline, tool codec). Paris x3 bit-exact, `logits_hash bd82916c6550a2df`, top-1 6993; native tool round-trip `get_weather {"city":"Paris"}`; DET 3/3 deterministic. FSI 2/3 committed: **FSI-01 is excluded in writing** because the model appends `*exp(r)` to the form that is already the annuity-due expression (8.7% high at r=0.0835) -- the same deviation the 30B sibling records in `docs/OPEN_TASKS.md` R13, measured here rather than inferred (the fixture carries the exact output). Goldens are template-derived: the 1.1 TB BF16 source cannot be assembled in 750 GB, so source fidelity rests on the slice oracle (`tools/nemotron_ultra_oracle.py`: real layers 0-7, engine vs reference per layer within the measured 4-bit noise) |
| ling | PASS | SKIP | SKIP | SKIP | source is 560B BF16/FP8 with GPU-only FLA KDA kernels; no oracle runnable on one RTX 5090 (a CPU reference was used for single-token probes only). Per-expert global scales + the official SwiGLU clamps were required for a coherent answer — `docs/OPEN_TASKS.md` §R7 |
| step-5-preview | PASS | SKIP | SKIP | SKIP | converted tree present (317 GB, Paris x3 bit-exact, native tool round-trip green); the 606 GB FP8 source oracle (reference `inference/` stack) needs 4x B300 / 8x H200 per the model card, so no source golden can be captured here. The DSA indexer is stored-not-applied (dense-equivalent at <=4096 tokens) — `docs/OPEN_TASKS.md` §R9 |
| hy-simulmt-30b-a3b | PASS | SKIP | SKIP | SKIP | zh<->en simultaneous translator: the source summarises/translates the FSI/CODE/DET bodies instead of answering (two emit EOT immediately), so no source-faithful golden can be extracted; instruments are SelfCheck (translation direction) + a real-weight bidirectional translation test — `docs/OPEN_TASKS.md` §R11 |
| deepseek-v4-flash | PASS | SKIP | SKIP | SKIP | the tree exists (so this is the §2 shape, not §4's "neither"); its three stages SKIP for a golden, not for a tree, and the goldens are declared debt. It was listed under §4 for a day — moved here 2026-11-07 by the vault audit, which is the third time the §4 heading has disagreed with the disk |
| kimi-k2.7-code | PASS | SKIP | SKIP | SKIP | 542 GB tree, 23040 experts; SelfCheck passes on real weights (Paris, tok 17374, 3 cycles identical) and the per-layer reference oracle is green (relL2 0.0034 layer 0 / 0.1226 layer 60, cos ≥ 0.9925). FSI/CODE/DET are declared golden debt (`docs/OPEN_TASKS.md` §R2 K1): the 581 GB `compressed-tensors` source has not been run for a capture |
| nemotron-3.5-lightning-30b-a3b | PASS | 2/3 | 2/3 | 1/3 (det 3/3) | BF16 distribution, converted 2026-11-13. Tree 18 GB, Paris x3 bit-exact `logits_hash 0dba89277c5d1250` top-1 6993; tokenizer 22/22; tool round-trip `get_weather {"city":"Paris"}`; 127.31 t/s (r3 121.16). The goldens have the strongest provenance in this corpus: the checkpoint loads in `transformers` 5.15.1 directly, so all three were captured from the EXECUTED source (9/9 template-exact, 363-1358 tokens each) rather than reconstructed or template-derived. The rows are red on the CONVERTED tree with a measured mechanism: every case runs to its **token budget** (the raw ends mid-repetition) while the source stops at 363-1358 tokens; it re-emits `</think><answer>`, so FSI-02 / CODE-02 / DET-01+03 extract a truncated fragment or nothing, and the verdict is budget-sensitive (FSI-03 passes at 2048, fails at 3072). Residual plane is the documented lever, not taken - `docs/OPEN_TASKS.md` R20 |
| nemotron-3.5-lightning-30b-a3b-nvfp4 | PASS | SKIP | SKIP | SKIP | the source IS runnable now — `tools/nemotron_nvfp4_oracle.py` dequantizes the ModelOpt checkpoint (W4A16_NVFP4 experts + FP8 mixers) to a CPU BF16 `NemotronHForCausalLM` and it answers FSI 3/3 per the Expected Template — but the converted tree does not reproduce them (FSI-01 extracts `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)`, FSI-02/03 emit fenced/repeated bodies), and CODE-TEST-01 / DET-TEST-01/02 make the SOURCE echo the prompt body, so no source-faithful CODE/DET golden can be extracted. Gates stay debt with these measurements; the FSI gap is the 4-bit body precision item in `docs/OPEN_TASKS.md` §R13 |

Both are recorded in the golden-debt list (§K) rather than left to look green.

`mistral-large-4-1t` (2026-11-02) is the third shape: the parity goldens are
**template-derived** (the 988 GB FP8 source is a multi-node FP8 runner and
cannot execute on this card), `expected` comes from each prompt set's own
Expected Template, and the goldens are committed — so the stages RUN rather than
SKIP. All four gates are green: SelfCheck x3 bit-exact, FSI 3/3, CODE 3/3, DET 3/3
with byte-identical streams. The first runs read CODE/FSI as red because the
model reasons inline and its `[THINK]`/`[/THINK]` markers are special ids that
the decoder drops; two opt-in extraction knobs (default off) cut the chain by id
and both final answers are correct — provenance and the trap are recorded in
`docs/OPEN_TASKS.md` §R14.

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

**2026-11-04:** both former members were audited and moved into §1
(`qwen3.5-35b-a3b`, 19 GB tree in the vault since 2026-10-01, all four gates
green; and `deepseek-v4.1-flash`, 294 GB tree, SelfCheck green, FSI/CODE red as
recorded in `docs/OPEN_TASKS.md` §Q, and its cooperative-cancel case RED — §A2
there).
**2026-11-05:** empty again. `kimi-k3.1-pro` was the last member; its 1.5 TB tree
was converted that day and the row moved into §1 (SelfCheck green, the parity
sweep declared debt). A row in this section SKIPs (ctest exit 4) and must never
be read as a pass.

| model | SelfCheck | FSI | CODE | DET | why |
|---|---|---|---|---|---|


## 4. Neither — SKIP

Re-audited 2026-11-07 against the vault (`<models_dir>/<vault dir>/logestix/model.index.json`
for all 36 manifests: 26 trees, 24 of them in §1). Every entry here is now checked
against the disk rather than inferred:

| model | tree | goldens | note |
|---|---|---|---|
| `glm-5.2` | no | no | never converted on this host |
| `mimo-v2.6-flash-rl` | no | no | never converted on this host |

`gigachat3.5-432b-a28b-reasoning` moved out of this section on 2026-10-31 (241 GB
tree, all four gates measured) and its row in §1 still carries those verdicts —
but the tree it was measured on **no longer exists** (it was converted under
`/mnt/fast_storage/tmp/` and cleaned), so §1's reproducibility note applies.
`deepseek-v4-pro-0813` moved out on 2026-10-26 (838 GB tree, SelfCheck Paris x3
bit-exact) and is in the same position: the verdict is history, the tree is
gone.

`kimi-k2.7-code` and `deepseek-v4-flash` are in §2's table, not here: both have a
converted tree and a passing SelfCheck, so "neither" was never true for them —
their three parity stages SKIP for a **golden**, not for a tree, and those goldens
are declared debt. Audit 2026-11-07 by checking
`<models_dir>/<vault dir>/logestix/model.index.json` for all 36 manifests: 26 have
a tree, 24 of those were in §1, and the six that were neither in §1 nor in §2 are
what this section's rows had been hiding. The capture for kimi-k2.7-code must come
from the source,
never from the engine. Verification record:
`docs/open-tasks/2026-10-26-kimi-audit-verification.md`.

## Reproduce

```bash
# 1. the real verdicts (the §1 rows: a tree AND goldens), one at a time
ctest --test-dir build/linux-release-znver4-aocc --timeout 1500 -R \
  "alice.*selfcheck|gemma_(moe|dense).*selfcheck|glm_moe_flash.*selfcheck|\
granite_dense.*selfcheck|nemotron_moe_nvfp4.*selfcheck|ornith_moe.*selfcheck|\
phi_dense.*selfcheck|qwen_dense.*selfcheck|qwen_moe.*selfcheck|\
xing_moe.*selfcheck|gigachat_moe.*selfcheck"

# 2. the debt rows (SelfCheck only; the three parity stages SKIP by design)
ctest --test-dir build/linux-release-znver4-aocc -R "mistral.*selfcheck"

# 3. the families with neither tree nor golden (§4) — all SKIP, which proves the
#    SKIP is the vault and not the code
ctest --test-dir build/linux-release-znver4-aocc -R \
  "glm_moe_selfcheck|mimo_moe_selfcheck|nemotron_moe_selfcheck"

# 3b. tree present, golden absent (§2 shape): kimi-k2.7-code runs SelfCheck and
#     SKIPs the three parity stages.
ctest --test-dir build/linux-release-znver4-aocc -R \
  "kimi_moe_selfcheck|deepseek_moe_v41_selfcheck"

# 3c. every SelfCheck in the §1 table. The vault path now comes from the manifest
#     (`vault_env` + `lsx_model_models_dir()`), so no per-family override is
#     needed — the old per-family files defaulted to the retired
#     /mnt/llm_models/models path and SKIPped silently, which is how the
#     deepseek-v4.1-flash cancel defect stayed invisible for weeks.
#     `--print-selfcheck-targets` is derived from the manifests, not typed by hand:
#     the 2026-11-07 audit found a hand-written filter that omitted 15 of the 37.
timeout 9000 ctest --test-dir build/linux-debug -j1 -R \
  "^($(python3 tools/gen_model_manifest.py --root . --print-selfcheck-targets \
        | paste -sd'|'))$"

# 4. the harness' own self-tests (no model): formula extraction/canonicalisation
ctest --test-dir build/linux-release-znver4-aocc -R \
  "test_(fsi|code|det)_selfcheck"
```

The parity stages share one rule: the golden is captured from the source
safetensors and is **never reconciled** to make a conversion pass. When a gate
goes red the question is which instrument is missing, not how to rewrite the
expectation (`docs/OPEN_ENDEDNESS.md`).
