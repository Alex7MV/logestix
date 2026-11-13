# NVIDIA-Nemotron-3.5-Lightning-30B-A3B (BF16) — integration report

| | |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, `model_type = "nemotron_h"`, `architectures = ["NemotronHForCausalLM"]` |
| Converted layout | `logestix-nemotron-nvfp4` (NVFP4 dense + streamed NVFP4 experts; BF16 embeddings / head / norms / conv / SSM scalars; F32 router) |
| Source | 14 shards, 65.84 GB BF16, 31.58 B params, 6513 tensors (6243 backbone + 270 MTP) |
| Converted tree | 18 GB on disk (2.2 GB pinned / 107 Arrow files, 16 GB experts / 23 layer dirs × 128); 2944 routed experts, 355 pinned tensors, 270 MTP tensors cut and counted |
| Hardware | 1 × RTX 5090 (32 GB), CC 12.0, 750 GB host RAM, CUDA 13.5 toolkit / 13.4 driver |
| Runtime | token-by-token prefill/decode over the neutral NVFP4 GEMV, FP32 KV cache + FP32 Mamba-2 state, single stream |
| Date | 2026-11-13 |
| Reference | `transformers` 5.15.1 `NemotronHForCausalLM` running **the checkpoint itself** on CPU (bf16) — the source is its own oracle |

**Verdict at a glance:** the model converts, loads and runs correctly at
**127.31 t/s** (r3 121.16) at the canonical 2048-token protocol, with Paris
answered three times bit-exactly, a green tokenizer parity (22/22) and a green
real-weight tool round-trip. Its three parity gates are **red in part** — FSI
2/3, CODE 2/3, DET 1/3 with determinism 3/3 — and the mechanism is measured
rather than inferred: on answer-completion prompts the converted model reaches
the correct answer and then **re-emits `</think><answer>` until the token
budget instead of stopping**, so the answer extractor either lands on a
budget-truncated copy or rejects the repeated line. The BF16 source, given the
same prompts, answers all nine cases and stops (363–1358 tokens). That is a
4-bit-body behaviour, the documented lever is a residual plane, and it was
deliberately not taken in this pass.

---

## Publish-ready post

### Nemotron-3.5-Lightning-30B-A3B (BF16) on one RTX 5090

Nemotron-3.5-Lightning is a 30B-parameter hybrid: 23 Mamba-2 layers, 23
mixture-of-experts layers (128 experts, top-6, plus one shared expert at twice
the expert width) and six full-attention layers at 5 / 12 / 19 / 26 / 33 / 42.
About 3 B parameters are active per token, which is why a 30B-class model fits a
consumer card once the routed experts stream instead of pinning: the converted
tree is 18 GB on disk and 17.5 GiB device-resident, at 127 tok/s.

**What this checkpoint forced into the engine.** Four facts about the
architecture are easy to get wrong and each of them produces fluent garbage
rather than a crash.

- The six attention layers are **NoPE**. The reference class has no rotary
  module and ignores `position_ids`; `rope_theta` and `partial_rotary_factor`
  in `config.json` are vestigial. Position in this model comes entirely from the
  Mamba-2 recurrence. Applying RoPE here is invisible at position 0 and destroys
  any longer context.
- The MoE is **gate-less**: every expert has only `up_proj` and `down_proj`
  (a fused SwiGLU, not the usual three matrices), routed top-6 with
  `norm_topk_prob` renormalisation and a 2.5 routing scale, with the shared
  expert added *outside* the routed sum and the activation being `relu2`.
- **MTP is ballast.** `num_nextn_predict_layers = 1` ships 270 extra tensors
  (an attention mixer, a MoE mixer and four norms under `mtp.*`). The reference
  ignores them on load; we cut and count them so the index says so.
- The tokenizer's frame opens `<|im_start|>assistant\n<think>\n`. Thinking is
  **on** by default, and an assistant turn with empty content renders
  `<think></think>`. Serving a prompt framed any other way still produces
  reasonable short answers, which is exactly why it is worth pinning: the
  parity prompts are ~100–250 tokens and a wrong frame moves them.

**Artifact-level findings, and what the authors could change.**

- *`generation_config.json` and `config.json` disagree on the stop set*:
  `config.json` carries the scalar `eos_token_id = 2`, `generation_config.json`
  carries `[2, 11]`. Anything that reads only the config stops later than the
  model intends. Suggestion: make the config the superset, or state in the model
  card that the generation config is authoritative.
- *The `mtp.*` tensors are shipped unlabelled.* A consumer reading the
  safetensors index has no way to tell 270 speculative-decoding tensors from
  backbone weights except by their prefix. Suggestion: an index-level
  annotation, or a separate shard.
- *The chat template computes `system_message = ""` and then branches on
  `system_message is defined`,* which is therefore always true — so the frame
  always opens an (empty) system turn. Any harness that renders "no system
  turn" gets a different prompt than the model was served. Suggestion: branch on
  truthiness, or document the always-present leading system turn.
- *`rope_theta` / `partial_rotary_factor` are present but unused.* Two
  config fields that a port must ignore is a trap that costs a week; a comment
  in `config.json` (or removing them) would prevent it.

**Measured numbers.** Conversion 44 s (source in page cache) → 18 GB. Per-tensor
quantization error against the BF16 source: attention projections 9.47 %, Mamba
projections 9.50 %, shared expert 9.43 %, routed experts 9.46 %; `lm_head` and
`embeddings` are passthrough BF16 (exactly 0). An ideal E2M1 grid with an exact
float block scale measures 9.37–9.39 % on the same tensors, so the codec sits on
the 4-bit grid bound and the E4M3 block-scale rounding costs about 1 % relative.
SelfCheck: `Paris` ×3 with byte-identical `final_logits` (`logits_hash
0dba89277c5d1250`, top-1 6993). Tokenizer parity 22/22. Tool round-trip:
`get_weather {"city":"Paris"}`. Perf at 2048 generated tokens: 127.31 t/s
(TTFT 86 ms, 249 W mean, 348 W peak); bit-exact mode 121.16 t/s.

**What was not measured, and why.** The three parity gates are only partly
green and should be read as such: the goldens are genuinely source-captured and
the source answers 9/9, but the converted tree's answers are only extractable on
5 of 9 (FSI 2/3, CODE 2/3, DET 1/3) because it does not stop after the answer and
repeats `</think><answer>` to the budget. Whether the
cause is only 4-bit body noise or something in the decode path is **not
established**; what is established is that the two other generation paths this
build exercises (the tool round-trip and the Paris probe) terminate normally, and
that the parity verdict moves with the budget (FSI-03 passes at 2048 and fails
at 3072). Long-context behaviour beyond 4096 tokens, the MTP head, and any
multi-GPU or quantized-KV-cache configuration were not evaluated.

---

## Evidence

### Session record

- Design: `docs/superpowers/specs/2026-11-13-nemotron-lightning-bf16-lifecycle-design.md`
- Plan: `docs/superpowers/plans/2026-11-13-nemotron-lightning-bf16-lifecycle.md`
- Branch: `feat/nemotron-lightning-bf16-lifecycle`
- `docs/OPEN_TASKS.md` §R20 (this session's record), `docs/SELFCHECK_MATRIX.md` §1 (the row)
- Manifest: `models/nemotron-3.5-lightning-30b-a3b.json`
- Perf rows: `tests/perf/perf_baselines.json`, `date 2026-11-13`, modes `perf` / `r3`

### Stage 0 triage

Sibling variant of the already-ingested `nemotron-3.5-lightning-30b-a3b-nvfp4`:
identical frozen geometry, identical `logestix-nemotron-nvfp4` layout, so the
existing converter / loader / kernels / expert cache / tool codec carry it. A
second passport for the same widths is forbidden by
`test_model_independence_guard`. Verified from `config.json` + the safetensors
index: 6513 tensors / 65 842 365 568 B / 31 577 937 344 params; 23 mamba + 23
moe + 6 attention at `[5,12,19,26,33,42]`; 270 `mtp.*`; tokenizer kind 9 (bos 1,
eos 2 with generation also stopping at 11, pad 0, `tie_word_embeddings = false`).

### Gate transcript

| stage | target | result |
|---|---|---|
| convert | `lsxforge --nemotron` | 44 s, exit 0; "52 layers (23 mamba/23 moe/6 attn), 128 experts, 270 MTP tensors cut, 355 pinned tensors" |
| convert (synthetic) | `test_nemotron_moe_convert` | PASS |
| load | `test_nemotron_moe_loader`, `_expert_cache`, `_model`, `_math` | PASS |
| kernels / session | `test_nemotron_moe_kernels`, `test_nemotron_session` | PASS |
| tokenizer | `test_nemotron_moe_tokenizer_parity` | PASS, 290 assertions, 22/22 corpus lines on the real vault |
| selfcheck | `test_nemotron_moe_selfcheck` | PASS — Paris ×3 bit-exact, `logits_hash 0dba89277c5d1250`, top-1 6993 |
| tool round-trip (4b) | `test_nemotron_moe_selfcheck` tool hook | PASS — `get_weather {"city":"Paris"}` |
| FSI | `test_nemotron_moe_fsi_selfcheck` | **2/3** — 01 PASS, 02 FAIL, 03 PASS |
| CODE | `test_nemotron_moe_code_selfcheck` | **2/3** — 01 PASS, 02 FAIL, 03 PASS |
| DET | `test_nemotron_moe_det_selfcheck` | **1/3** answer parity, **determinism 3/3** |
| perf | `test_perf_baseline_nemotron` | PASS — 127.31 / 121.16 t/s |

### Index and tree

`logestix/model.index.json`: `format logestix-nemotron-nvfp4`,
`model Nemotron-3.5-Lightning-30B-A3B` (the BF16 distribution's own name, which
`Load` now enforces), `total_experts 2944`, `num_layers 52`,
`total_pinned_tensors 355`, `deferred {mtp: cut, mtp_tensors: 270}`,
`vram {min 4.94 GB, optimal 6.68 GB, max 23.20 GB, context 262144}`, no
`fp8_kv_scales_dropped` (an NVFP4-source-only field).

### Defect found and fixed this session

`NemotronHModel::Load` accepted the **NVFP4 sibling's** tree. Both distributions
write the same `logestix-nemotron-nvfp4` format tag; only the index's `model`
field separates them, and the NVFP4 leaf has enforced that since ingestion while
the BF16 leaf did not — so a mixed-up tree would have loaded and then verified
against the other model's goldens. Fixed with `frozen::kModelName` plus a check
in `LoadAndValidateIndex`, and the synthetic fixture's index writer now emits the
converter's string (it wrote `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, which
no converter emits — that divergence is why nothing caught the gap). Test:
`tests/test_nemotron_moe_model.cpp` "nemotron Load rejects a tree built for the
other distribution", verified RED with the check disabled.

### Parity provenance — the part that is better than the sibling's

The NVFP4 sibling's goldens could not be source-captured (its ModelOpt tree needs
an oracle script, and the slice-oracle result is qualitative). This checkpoint is
plain BF16 and loads in `transformers` 5.15.1 directly, so all three goldens in
`tests/fixtures/{fsi,code,det}/nemotron-3.5-lightning-30b-a3b.json` record
`source: <the vault path>` — the source was **executed**, and it answers all nine
cases exactly as the prompt templates specify (FSI after 1358 / 734 / 363 tokens).

Alignment between the capture and the served frame is mechanical, not asserted:
the goldens are captured with `enable_thinking=True` (the engine's frame), and
the recorded `input_ids` decode **byte-exactly** to the served frame text pinned
by `tests/fixtures/nemotron/fsi01_frame.txt` (first id 10 = `<|im_start|>`, no
bos — `tokenizer_config.add_bos_token = false`). The host test
"nemotron FSI-capture frame equals the served frame" fails if the renderer and
the template drift; a mutation to the thinking-off frame (`<think></think>`)
was verified to fail it.

### Measured parity mechanism (not a hypothesis)

- Every parity case ran to its **token budget**: the decoded raw ends
  mid-repetition at the truncation point, and each case consumed the full
  budget in wall time (e.g. FSI-02 at 2048 took 38 s of decode). The source, on
  the same prompts, stops at 363–1358 tokens. (An "eos was never emitted" claim
  cannot be made from a decoded raw here: `arrow_tokenizer_decode` drops special
  tokens, so a clean `<|im_end|>` stop is also invisible in the text — the
  repetition-to-the-cap is the observable, not the missing marker.)
- Where the answer is reached it is reached *correctly* and then repeated:
  FSI-02 logs 15 copies of `</think>Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`;
  CODE-02 logs `</think>m=l+((h-l)/2)` repeatedly.
- The extractor's last-line preference therefore lands on the budget-truncated
  copy (`Balance_Alpha=...Rate_B-S`, `PARAD`) or rejects the repetition (`''`).
- The verdict is budget-sensitive, which is why the row is written as debt and
  not as a pass: at 2048 FSI-03 passes, at 3072 it fails and FSI-02 extracts
  `Balance_Alpha=(((`.

The documented lever is the residual plane on the attention + Mamba projections
(~+1.1 GiB resident, doubled attn+Mamba read, every perf row re-seeded) — the
`gemma-4-26b` recipe. It was **not** taken here: the operator decision was to
measure first, and a tree change re-baselines every row and makes the two
distributions incomparable.

### Perf rows

```
perf (kAuto) 127.31 t/s  TTFT 85.68 ms  avg 248.97 W  peak 347.86 W  802 samples
r3   (kOff)  121.16 t/s  TTFT 87.72 ms  avg 249.49 W  peak 348.43 W  843 samples
```

Both at `max_gen 2048`, replaced 2026-09-16 rows (98.24 / r3 24.07) that predated
the NoPE fix and the expert-store migration. The NVFP4 sibling's rows
(127.55 / 120.39) are the expected agreement: the two distributions convert to
the same layout, so the same kernels and the same read volume.

Run-to-run spread on this card, same tree and same target: 127.00 / 127.06 /
127.31 t/s (r3 120.32 / 120.38 / 121.16) over three harness invocations this
session — about 1 %. A re-seed that moves the row inside that band is noise, not
a regression; judge a change against several runs, not one row update.

### Debt, stated as debt

`fsi`, `code` and `det` remain `golden_debt` in the manifest — not for a missing
golden (all three exist and are source-captured) but because the converted tree
does not answer them stably. `docs/SELFCHECK_MATRIX.md` records the row with
those verdicts, and `docs/OPEN_TASKS.md` §R20 carries the mechanism plus the
residual-plane trigger.
