# Hy-SimulMT-30B-A3B — integration report

| | |
|---|---|
| Model | `tencent/Hy-SimulMT-30B-A3B`, `model_type = "hy_v3"`, `architectures = ["HYV3SpeechForCausalLM"]` |
| Converted layout | `logestix-hy-simulmt-nvfp4` (NVFP4 body, FP16 embed / untied head) |
| Source | 60.8 GB BF16, 17 text shards + one 660 MB audio shard, 19 076 indexed tensors |
| Converted tree | 17 GB on disk; 1.5 GiB pinned + 15 GiB routed experts; 16.5 GiB device-resident |
| Hardware | 1 × RTX 5090 (32 GB), 750 GB host RAM, CUDA 13.4 |
| Runtime | token-by-token prefill/decode, persistent KV cache, single stream |
| Date | 2026-10-31 |
| Reference | `transformers 5.15.1` `HYV3ForCausalLM` (CPU BF16) over the source shards |

**Verdict at a glance:** the text decoder of Hy-SimulMT is a faithful,
fully-local **bidirectional zh↔en simultaneous translator**. It shares
Hy-Memory's backbone exactly, so the conversion is a sibling ingestion: the
60.8 GB BF16 checkpoint becomes a 17 GB NVFP4 tree, the speech encoder is
stripped and counted, and the model decodes at **87.6 tokens/s** on one
consumer card. SelfCheck is bit-exact across three cycles and a real-weight
test translates in both directions. The FSI/CODE/DET QA gates are declared
debt: this is a translator, and the source itself summarises or ignores those
prompts instead of answering them.

---

## Publish-ready post

### Hy-SimulMT 30B, running locally as a text translator in a C++/CUDA engine

We ported Tencent Hunyuan's Hy-SimulMT-30B-A3B — a real-time speech-translation
model — into our own Blackwell inference engine. The release ships a 30B
mixture-of-experts text decoder (3B active) wrapped with a 24-layer speech
encoder. We run the text decoder: it is the same `hy_v3` backbone as
Hy-Memory-30B-A3B, which we had already ingested, so the new work was the
ingestion of the checkpoint's differences, not a new model architecture.

The 60.8 GB checkpoint is converted in place to a 17 GB NVFP4 tree and fits a
32 GB RTX 5090 at 16.5 GiB resident. It decodes at **87.6 tokens/s** at 2 048
generated tokens, and the speech encoder arrives as a separate 660 MB shard
that is stripped and counted rather than loaded.

#### What the checkpoint forced into the engine

1. **It is a translator, not a chat model.** The decoder is direction-bound:
   the official serving stack sends a Chinese directive ("complete the
   simultaneous-interpretation task, translate into English/Chinese") as the
   first text part of the user turn. Without it the model only summarises or
   transcribes its input. Our text interface prepends the checkpoint's own
   directive and selects the direction from the source script — Chinese source
   → English output, anything else → Chinese output — so a plain text request
   is a translation request in both directions.
2. **Two stop tokens.** The release's `generation_config.json` ends a turn on
   either `<eos>` or its own `<|hy_EOT|>` marker. A single-EOS loop would keep
   emitting the turn marker to the token budget; the engine now honours the
   full stop set.
3. **The template has one literal that differs from its parent.** Its
   tool-instruction example closes with `</tool_calls>` where Hy-Memory closes
   with `</tool_call>`. The renderer carries that per-checkpoint delta so the
   framing is byte-identical to the shipped `chat_template.jinja`.
4. **Same vocabulary, same traps.** `tokenizer.json` is byte-identical to
   Hy-Memory's, including the missing carriage-return byte and the 818 added
   tokens, so the already-verified tokenizer policy transfers unchanged.

#### Measured

- SelfCheck: the canonical fact, asked in the trained direction, translates
  correctly — 法国的首都是巴黎。 → “The capital of France is Paris.” — and is
  bit-exact across three cycles.
- Bidirectional serving test on the converted weights: 今天天气很好。 →
  “The weather is very nice today.”; “What is the capital of France?” →
  什么是法国的首都？
- 87.6 t/s (perf) / 74.9 t/s (strict bit-exact) at 2 048 tokens, TTFT
  510 / 616 ms, 16.5 GiB resident.

#### Limits

Speech input and speech output are not implemented: the 24-layer speech
encoder, its mel frontend and the streaming speech-to-speech loop are out of
scope for this release, and the QA-style parity prompts are not this model's
task (it summarises them rather than answering). Both are recorded, not
papered over.

---

## Internal evidence

**Stage 0.** `HYV3SpeechForCausalLM` text side = 18 673 tensors, byte-identical
names/shapes to `hy-memory-30b-a3b` (48L / 2048H / 32Q·4KV·128HD / 128 experts
top-8 at I=768 / dense 6912 / vocab 120832 / theta 11 158 840). Tokenizer
sha256 `ad96ef9f641b79bb…` identical. Ballast: 403 `audio_tower.*` tensors
(660 MB in `model-audio-tower.safetensors`), stripped and counted.

**Registry rows touched (same commit as the files).**
`test_code_naming_guard.cpp` variant token `simulmt`;
`test_model_independence_guard.cpp` `kPassports` + `consumers` + `kLifecycle`;
`test_model_suite_guard.cpp` `Suites()` row + tool verdict + golden debt;
`test_perf_baseline_logic.cpp` mid list; `tests/CMakeLists.txt` targets and the
perf `add_test`.

**Shared-code changes (no kernel fork).**
- `hy_moe_convert.{h,cpp}`: `HyConvertSpec` (identity + geometry + audio
  ballast policy) threaded through the shared pipeline; the Hy-Memory entry
  point is unchanged. The spec is what makes `audio_tower.*` a counted ballast
  tensor instead of an unknown-tensor error.
- `hy_moe_load.{h,cpp}`: `LoadHyDenseWeights(path, HyPassportGeometry,
  format_tag, …)`; the old three-argument overload still applies Hy-Memory's
  passport. The expert-slab loop dispatches on `mlp_layer_types` instead of a
  hard-coded first-MoE layer.
- `hy_moe_inference_pipeline.cu`: stop-token set (`eos_token_ids`) instead of
  a single eos; `ignore_eos` still disables all of them.
- `model_runtime.h`: `ParisSelfCheckOptions::prompt` (default is the canonical
  English sentence, so every other family is unchanged).
- `hy_moe_tool_format.{h,cpp}`: `HyRenderPolicy` for the one template literal;
  the default preserves Hy-Memory's output.

**Gates (measured 2026-10-31, RTX 5090).**
- SelfCheck: PASS, top1 token 628, text “The capital of France is Paris.”,
  `logits_hash 44b846591fd64cb9`, 3 cycles bit-exact at kOff; kAuto
  determinism byte-identical across two loads.
- Real-weight bidirectional translation: zh→en and en→zh as in the post.
- FSI/CODE/DET: **declared debt** (`hy-simulmt-30b-a3b:{fsi,code,det}`).
  Evidence from the CPU BF16 source oracle: with the serving directive,
  FSI-TEST-01 returns “预期模板：PV = CF * ((1 - exp(-r * n)) / (1 - exp(-r))).”
  (loses the required unspaced expression), CODE-TEST-01 degenerates to a
  repeated token, DET-TEST-01 emits EOT immediately. With the raw template the
  model summarises the bodies (FSI-01 → a prose summary, CODE-01/02 → prose,
  FSI-02/DET-01 → immediate EOT). There is therefore no source-faithful
  extractable golden; the failed goldens were **not** rewritten.
- Chat framing: 14/14 cases against the checkpoint's own template
  (`tests/fixtures/hy/chat_framing_simulmt.json`), including the tools and
  tool-round-trip cases.
- Tokenizer: byte-identical vocabulary to Hy-Memory (which is pinned by
  `test_hy_moe_tokenizer_parity`, 17/17).
- Perf (`tests/perf/perf_baselines.json`, canonical 2048-token protocol):
  perf 87.57 t/s / TTFT 510.1 ms; r3 74.85 t/s / TTFT 616.1 ms; 16.5 GiB
  resident; DRAM probe 1531 GB/s.

**Artifact-level questions for the model authors.**
1. The released `chat_template.jinja` closes the tool example with
   `</tool_calls>` while the parent checkpoint closes with `</tool_call>` —
   intentional or a typo? We reproduce the release literally.
2. The model has no text-QA behaviour and no explicit direction marker in the
   template; direction lives only in the serving directive text. A structured
   direction field would make the contract explicit.

**Evidence.** `docs/OPEN_TASKS.md` §R11; `docs/SELFCHECK_MATRIX.md` §2;
`docs/superpowers/specs/2026-10-31-hy-simulmt-30b-ingestion-design.md`;
`tools/hy_v3_oracle.py` (audio skip), `tools/hy_chat_framing_capture.py`;
tests `test_hy_moe_simulmt_{convert,loader,model,selfcheck,tool_framing}` and
the shared `test_hy_moe_kernels`.
