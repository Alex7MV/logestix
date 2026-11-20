# MiniMax-M3 — open items

### R4. MiniMax-M3 (~428B/23B active) — ingestion landed (2026-10-27/28)

Branch `feat/ingest-minimax-m3`. New family `minimax` (GQA + MiniMax Sparse
Attention indexer + sigmoid-router MoE, 60L / 6144H / 64Q-4KV / 128HD,
128 experts top-4 + ungated 3072 shared expert, layers 0-2 dense). Frozen
passport `minimax_moe_constants.h`; converter `lsxforge --minimax`; loader,
kernels, pipeline, model, native tool codec and all eight lifecycle stages
landed. Four guard registries extended (naming family token + tool-format
allowlist, `kPassports`/`consumers`/`kLifecycle`, `Suites()` + tool verdict,
perf `add_test` + `TEST_CASE`).

**Conversion (real, in place).** 796 GB BF16 source -> **229 GB** tree in
1 h 45 min: 10.5 GiB pinned (FP16 embed+head 4.9 GiB, NVFP4 attention/dense/
shared, F32 router + bias, FP16 norms) + 217 GiB routed experts (`57 x 128`
ExpertSchema files). `skipped_visual_tensors = 523` (vision tower + projector,
counted); no `mtp.*` in the tree. Expert pass is expert-major (w1/w2/w3 joined
across shards) so the source is never resident. `LSX_MINIMAX_SKIP_EXPERTS=1`
re-converts only the dense side (byte-identical index) when the expert store is
already on disk.

**Real-weight verdicts (RTX 5090, 32 GB, one process at a time).**
- SelfCheck: **PASS** — "The capital of France is **Paris**." ×3 stress cycles,
  `logits_hash 579a4a12cdedc279` stable (bit-exact), streaming order exact.
- CODE **3/3**, DET **3/3** (3 runs each, byte-identical `gen_tokens`).
- FSI **2/3** — FSI-01 RED: the prompt's Expected Template is
  `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` and the model answers
  `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)` (the annuity-**due** form, an extra
  factor). That is a template/form deviation of the same class as
  gemma-4-31b FSI-01 and deepseek-v4-pro FSI-02, not a near-tie flip: the
  golden is NOT rewritten.
- Goldens are **template-derived** (`--template-only` / `--template-golden`):
  the 428B BF16 source has no 32 GB-card execution path (the vendor stack is
  TP-8 over the full tree), so `source` says so in each golden. Same status as
  DeepSeek-V4.1/Pro.

**Defects found and fixed while landing it** (all pinned by tests):
1. The residual add ran `<<<1, 256>>>` for a 6144-wide row, so only the first
   256 hidden dims accumulated — the model produced token salad. Now
   `<<<(H+255)/256, 256>>>` (`test_minimax_moe_model` determinism covers it).
2. `cudaEventElapsedTime` on the never-recorded ttft event left a sticky
   `cudaErrorInvalidResourceHandle` that poisoned the NEXT request ("embed
   (invalid resource handle)"). Guarded; the abort case pins it.
3. An already-aborted request still emitted one token (the abort was polled
   only after `Emit`). Now polled before sampling: pre-set abort => 0 tokens.
4. Tokenizer bridge, two defects: `base_vocab_size` was computed over
   `key_to_id` (which already contained the added tokens) so `literal_ids` was
   empty and every control marker (`]~b]`, `]<]minimax[>[`) was BPE-encoded as
   raw bytes; and the pre-tokenizer was the legacy GPT-2 3-split, which splits
   CJK runs (the checkpoint uses the Llama-3/o200k pattern). Both fixed:
   `TokenizerSpecialPolicy.pretokenizer = 1` selects new `tokenizer_kind = 3`
   (`Llama3WordStream`, exact Lu/Ll/Lt case runs, no CJK split) and
   `base_vocab_size` is now the model-vocab rank space. `tokenizer_kind = 3`
   reuses the MERGE-RANK table (not tiktoken's id-order rule: HF BPE ids are
   not merge-ordered). Pinned by `test_minimax_moe_tokenizer_parity` — 14
   adversarial cases (CJK, CamelCase, contractions, multi-space, digits,
   markers) captured from the checkpoint's own tokenizer: exact.
5. Precision: `embed_tokens`/`lm_head` MUST stay FP16. At 4-bit the head's
   ~9% row error moves a logit by ~3 while this model's first-token margin is
   0.4; with the head 4-bit the answer was token salad, with the head FP16 the
   engine's top-5 reproduced the reference's ids and order. Recorded rule for
   the family (phi-4 / gemma-4 precedent).

**Numeric evidence (external oracle).** `tools/minimax_reference.py` is a host
re-implementation over the source safetensors (loads only the dense graph + the
routed experts actually selected, ~25 GB RAM) and
`tools/minimax_hf_layer_oracle.py` runs the CHECKPOINT'S OWN modeling code for
one layer on the meta device. Layer 0 agrees with the HF oracle to NVFP4
quantization error, and the per-layer cos/relL2 vs the source is 0.9998/0.034
at layer 0 -> 0.965-0.98/0.21-0.27 at layers 40-59 (the residual stream grows
~50x over 60 layers, which is what amplifies 4-bit body error). The reference's
greedy continuation is "The capital of France is **Paris**." — the same text
the engine produces.

**Perf (honest numbers, stage 4 debt).** Measured on this box: ~0.31 s per
step token-by-token (179-token prefill + 16 generated = ~1 min per selfcheck
cycle), dominated by pageable H2D expert staging (6.9 GB/token: 228 expert
stages; nsys: 78% of CUDA API time in `cudaMemcpyAsync`, 80% of kernel time
was the serial router before it was parallelized to one warp per expert).
Rows seeded through the harness: see `tests/perf/perf_baselines.json`
(`perf` / `r3`).

Open work, explicitly:
- P1. Perf: the decode step is PCIe/host-store bound. Real options, in order:
  overlap expert staging with compute (dual-stream prefetch, the neutral layer
  already has the machinery), raise the slot count with a pinned chunk pool,
  and only then consider quantization-side tricks. The canonical 2048-token
  protocol is recorded as a baseline so future work has a reference.
- P2. MSA beyond the block budget: up to `topk_blocks*block_size` = 2048 tokens
  every key block is selected, so the canonical protocol runs on the neutral
  coalesced attention kernel. Past that the two-pass masked kernel engages; it
  is correct but reads the whole KV (no sparse speedup yet).
- P3. FSI-01 stays RED (see above). Not a golden to rewrite.
