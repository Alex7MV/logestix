# Hy-Memory-30B-A3B — open items

### R5. Hy-Memory-30B-A3B (~30B / ~3B active) — ingestion landed (2026-10-30)

Branch `feat/ingest-hy-memory-30b`. New family `hy` (GQA + plain per-head
QK-RMSNorm + full 128-dim RoPE + sigmoid router with an `expert_bias` + one
gated shared expert; 48L / 2048H / 32Q-4KV / 128HD, 128 experts top-8 at
intermediate 768, dense SwiGLU MLP on layer 0). Frozen passport
`hy_moe_constants.h`; converter `lsxforge --hy`; loader, kernels, pipeline,
model, native tool codec and all eight lifecycle stages landed. Guard
registries extended (naming family token + tool-format allowlist,
`kPassports`/`consumers`/`kLifecycle`, `Suites()` + tool verdict, perf
`add_test` + `TEST_CASE`).

**Stage-0 findings that were load-bearing.** (a) `config.json` carries BOTH
`moe_intermediate_size` (1536, a vestigial default inherited from Hy3) and
`expert_hidden_dim` (768); the tensors are 768 wide, so the converter pins 768
and rejects a 1536-wide tree. (b) The vocabulary has **no CR byte token**
(`"\r"` encodes to no token at all in the reference) and no single-byte token
for the 13 UTF-8-illegal lead bytes → `TokenizerSpecialPolicy::
drop_unmapped_bytes` maps them to -1 and the encoder skips them. (c) Its 818
added tokens include **plain (special:false)** control markers, which the
reference `AddedVocabulary` matches literally; the kind-0 encoder now does too
when the vocab metadata carries `base_vocab_size` (only this family writes it,
so every existing family's tokenization is byte-identical — the full fast
suite stays green).

**Conversion (real, in place).** 60.1 GB BF16 source -> **17 GB** tree in
~6 s of expert pass + dense side: 1.5 GiB pinned (FP16 embed+head 0.94 GiB,
NVFP4 attention/dense/shared, router + bias raw, FP16 norms), 24 MiB router,
120 MiB shared, 15 GiB routed experts (`47 x 128` ExpertSchema files, 6016).
No vision tower and no MTP in the tree (`num_nextn_predict_layers = 0`), so
`skipped_visual_tensors = 0`. Expert pass is expert-major per (layer, expert)
across shards; `LSX_HY_SKIP_EXPERTS=1` re-converts the dense side only.

**Real-weight verdicts (RTX 5090, 32 GB, one process at a time).**
- SelfCheck: **PASS** — "The capital of France is Paris." (token 628), logit
  max 25.2172, `logits_hash 4472c7d335d0bf7d` stable over stress cycles;
  streaming order exact; cancel before step 1 publishes nothing.
- FSI **3/3**, CODE **3/3**, DET **3/3** (3 runs each, byte-identical
  `gen_tokens`), all against goldens captured from the **source safetensors**
  (`tools/{fsi,code,det}_capture.py` with `loader: "hy_v3"`). The source oracle
  is `tools/hy_v3_oracle.py`: the published checkpoint is per-expert
  (`mlp.experts.E.{gate,up,down}_proj`, `mlp.shared_mlp.*`,
  `mlp.router.gate`), which the installed `HYV3ForCausalLM` does not accept, so
  the oracle assembles the fused `gate_up_proj`/`down_proj` 3-D tensors and
  sets `moe_intermediate_size = expert_hidden_dim` before constructing the
  config. FSI-02's expected answer is the **numeric** form the source
  produced, not the prompt's symbolic template — source parity, not a template
  match (the opposite convention from DeepSeek-Pro/MiniMax, whose sources
  cannot run here).
- Tool calling (stage 4b) verified on real weights: the model answers the
  round-trip prompt with a native `<tool_calls>\n<tool_call>get_weather<tool_sep>`
  block and `ParseToolCalls` returns `get_weather {"city":"Paris"}`.
- Tokenizer parity **17/17** cases (`tools/hy_tokenizer_capture.py`) and chat
  framing **14/14** cases (`tools/hy_chat_framing_capture.py`: control prompt,
  FSI/CODE/DET bodies, multi-turn, tools, tool round-trip) reproduce the
  checkpoint tokenizer + `chat_template.jinja` id-for-id, including the
  Python-`tojson` tool block.
- Perf rows seeded through the harness at the canonical 2048-token protocol:
  **perf 86.69 t/s / r3 75.08 t/s**, TTFT 193.8 / 227.3 ms, 16.5 GiB resident
  (all routed experts resident; no expert cache, no host store).

Open work, explicitly:
- P1. Perf. ~2.0 GB of active weights per token (of which 0.5 GB is the FP16
  lm_head) against a 1531 GB/s probe is a **~1.3 ms / ~750 t/s roofline**; we
  run at 87 t/s, i.e. ~12% of it. The router-sync hypothesis was tested and
  REFUTED: the ids-streaming expert GEMV (`LsxExpertPtrRow` device tables, no
  per-layer D2H) landed and moved the row 86.7 -> 87.2 t/s, so the 47
  `cudaEventSynchronize` calls were not the bottleneck. What remains is the
  weight-streaming GEMV itself (48 attention + 47x3 expert-batch + shared
  launches per token, ~2 GB streamed at ~180 GB/s effective) — the next step is
  an nsys decode profile under `perf-tuning-a-model`, not more plumbing.
- P2. TTFT is token-by-token prefill (correctness first). A batched prefill is
  the next step if TTFT matters for the memory-writer workload.
- P3. `LSX_HY_DEBUG=1` dumps prompt ids + top-5 logits of the first decoded
  step (diagnostic only, stderr) for bisecting against the oracle.
