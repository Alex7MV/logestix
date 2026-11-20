# Hy-SimulMT-30B-A3B — open items

### R11. Hy-SimulMT-30B-A3B (~30B / ~3B active) — ingestion landed (2026-10-31)

Branch `feat/ingest-hy-simulmt-30b`. `Hy-SimulMT` is a **sibling of
Hy-Memory-30B-A3B**: `HYV3SpeechForCausalLM` with the same `hy_v3` text
backbone (18 673 text tensors, identical names/shapes, the same
`expert_hidden_dim=768`, 48L / 2048H / 32Q-4KV / 128 experts top-8) plus a
24-layer speech encoder. Frozen passport `hy_moe_simulmt_constants.h`
(`hy-simulmt-30b-a3b`, `logestix-hy-simulmt-nvfp4`); converter wrapper
`hy_moe_simulmt_convert.{h,cpp}` over the shared spec-driven pipeline;
`HySimulMTModel` reuses the parent's loader, kernels and generation pipeline.

**Stage-0 facts that were load-bearing.**
- The 403 `audio_tower.*` tensors (660 MB, a separate shard) are ballast:
  stripped AND counted (`skipped_visual_tensors = 403`). Speech input / speech
  output is out of scope for this engine; only the text decoder is served.
- `tokenizer.json` is **byte-identical to Hy-Memory's** (sha256
  `ad96ef9f641b79bb…`): same 120 000 BPE / 818 added tokens, same specials; the
  Hy-Memory tokenizer policy and its 17/17 parity capture apply unchanged.
- The SimulMT chat template's text branch matches Hy-Memory's renderer except
  for ONE literal: its tool-instruction example closes with `</tool_calls>`
  where Hy-Memory has `</tool_call>`. `HyRenderPolicy` now carries that delta;
  the SimulMT framing capture reproduces the checkpoint template id-for-id
  (14/14 cases, `tests/fixtures/hy/chat_framing_simulmt.json`).
- `generation_config.json` declares **two stop ids** (`[120025, 120008]`:
  `<eos>` and `<|hy_EOT|>`). The shared converter now writes
  `eos_token_ids` and the pipeline stops on any of them; Hy-Memory's legacy
  index (single eos) is unchanged.

**Conversion (real, in place).** 60.8 GB BF16 source → **17 GB** tree: 1.5 GiB
pinned dense graph + 15 GiB routed experts (6016 ExpertSchema files), 403
ballast tensors skipped, tokenizer arrows built into `logestix/pinned`.
`LSX_HY_SKIP_EXPERTS=1` re-converts the dense side only (used for the eos fix).

**Serving contract (discovered on the source, then implemented).** The
checkpoint is a **direction-bound zh<->en simultaneous translator**. Without the
official directive the decoder only summarises/transcribes its input; with it
(`SimulInfer/simulinfer/translation.py INSTRUCTIONS`) it translates. The engine
prepends the checkpoint's own directive and auto-selects the direction from the
source script (Han → zh-en, otherwise en-zh), so a plain text request is a
translation request in both directions. This is the family's single-prompt
`BuildPromptInputIds` / conversation rendering contract.

**Gates as measured (RTX 5090, 32 GB).**
- SelfCheck **PASS** — the canonical English sentence is en→zh for this model,
  so the family probe asks the same fact in the trained direction: source
  "法国的首都是巴黎。" translates to "The capital of France is Paris."
  (top1 628, `logits_hash 44b846591fd64cb9`), bit-exact over 3 cycles at kOff;
  kAuto determinism is byte-identical across two loads.
- Real-weight **bidirectional serving test**: "今天天气很好。" → "The weather is
  very nice today." (zh→en); "What is the capital of France?" → "什么是法国的
  首都？" (en→zh).
- FSI / CODE / DET are **declared golden debt**
  (`kGoldenDebt: hy-simulmt-30b-a3b:{fsi,code,det}`). The source oracle runs
  (`tools/hy_v3_oracle.py` with the `audio_tower.*` skip; the text backbone
  loads in `HYV3ForCausalLM`), and the measured behaviour is that the model
  summarises/translates these QA bodies instead of answering: FSI-TEST-02 and
  DET-TEST-01 emit EOT immediately, CODE-TEST-01/02 emit a prose summary,
  FSI-TEST-01 loses the required exact expression (spaces + translated label).
  No source-faithful golden can be extracted, so the instrument is SelfCheck +
  the bidirectional test, not a rewritten golden.
- Tokenizer parity is inherited from the byte-identical file; chat framing
  14/14 against the SimulMT template.
- Perf rows seeded through the harness at the canonical 2048-token protocol:
  **perf 87.57 t/s / r3 74.85 t/s**, TTFT 510.1 / 616.1 ms, 16.5 GiB resident
  (all routed experts resident; the same backbone and the same roofline as
  Hy-Memory).

**Open work, explicitly.**
- P1. The speech encoder (mel → 24-layer encoder → adapter → audio
  placeholders, and the streaming S2ST loop) is not implemented; audio input
  and speech output are not served. This is the difference between the
  checkpoint's product behaviour and the text-translator surface shipped here.
- P2. The shared perf debt of the `hy` backbone applies unchanged (weight
  streaming GEMV at ~87 t/s against a ~750 t/s roofline; see §R5 P1).
- P3. Tool calling is framing-complete (the native codec is verified) but
  semantically out of distribution for a translator; no real-weight tool
  round-trip is claimed.
