# Mistral-Large-4-1T-A52B — open items

### R14. Mistral-Large-4-1T-A52B (1T-A52B) — ingestion landed, four gates green on template goldens (2026-11-02)

Branch `feat/mistral-large4-ingest`. A new `mistral` variant (`large4`,
`kType = mistral-large-4-1t`, format tag `logestix-mistral-large4-nvfp4`):
72L / 6144H / 64 MLA heads without Q-LoRA (kv_lora 512, qk_nope 64 + qk_rope
64, v 128), 128 routed experts top-4 + one gated shared expert, a dense layer 0,
and five components no other family has: **hybrid RoPE/NoPE** (every 4th layer
is NoPE and its 64 rotary dims become plain key dims), **LeRoPE** (a learned
per-frequency `inv_freq *= exp(alpha)` on top of YaRN θ=1e4 / factor 2048 /
original window 512), **ssmax** (per-head, position-dependent query scaling
`s[h]*log(pos+1)+b[h]`), a **DSA Lightning Indexer** (32 heads × 128, top-2048,
44 full layers, 28 reusing the previous full layer's mask) and **cross-layer
KV sharing** (layers 39,43,…,71 have no `kv_a` and read layer 35's latent).

**Measured on real weights** (988 GB FP8-E4M3 block-128 source → 555 GB
NVFP4 tree, 16.1 GB pinned + 547 GB mmap-streamed experts; RTX 5090, CUDA 13.4):

| gate | verdict |
|---|---|
| SelfCheck | PASS ×3, bit-exact (`logits_hash=747bf131f1ac557f`, top-1 `Paris` 13125) |
| tokenizer parity | 50/50 against the checkpoint's own `tokenizer.json` (new `tokenizer_kind` 12: o200k case runs, RIGHT-grouped digits) |
| tool round-trip | PASS (`get_weather {"city": "Paris"}` through the native `[TOOL_CALLS]name[ARGS]` codec) |
| FSI | 3/3 (template-derived golden) |
| CODE | 3/3 (template-derived golden) |
| DET | 3/3 (byte-identical token streams over 3 runs) |

**Open, explicitly.**

- L1. The first CODE/FSI runs read as red for the *wrong* reason and the
  instrument gap is now closed: this model reasons INLINE (its `[THINK]` /
  `[/THINK]` markers are special tokens, so the decoder drops them), and the
  extractors had no boundary to work with. CODE-TEST-03 looked like a duplicated
  regex and FSI-02 like a truncated equation; both were lines from the reasoning
  chain, and both final answers are correct. The fix is opt-in and default-off
  (`answer_start_token_id` cuts by the special id in all three goldens,
  `prefer_last_answer` in FSI) — no other family's verdict changed. With it,
  CODE is 3/3 and FSI 3/3. Kept here because the trap generalizes: a
  reasoning model whose markers are special tokens needs an id-based cut, not a
  text-based one.
- L2. The parity goldens are **template-derived**: the 988 GB FP8 source is a
  multi-node FP8 runner that cannot execute on this 32 GB card, so no
  model-defined golden can be captured. `expected` comes from the prompt set's
  own Expected Template; the source is stated in the golden JSON.
  `tools/mistral_large4_template_goldens.py` reproduces them. This is the
  sanctioned deviation (GLM-5.3-Flash / GigaChat / DeepSeek-Pro precedent), not
  a pass-by-construction: the model must still produce the template's answer.
- L3. FSI-02 originally read as red for the *wrong* reason: the model reasons
  inline (no `[THINK]` markers, so no boundary survives the decoder's
  special-token drop) and `SelectAnswerLine` picked the reasoning line
  `Balance_Alpha = 100000000`. Two opt-in instrumentation knobs were added
  (`answer_start_token_id` in all three goldens, `prefer_last_answer` in FSI) —
  default-off, so no other family changed. With a 2048-token budget (the chain
  is ~700-1000 tokens) it answers the Expected Template exactly.
- L4. DSA beyond `index_topk` (2048) is implemented (indexer key cache,
  exact device top-k, masked gather attention) and covered by kernel-level host
  references, but the canonical gates run at ≤2048 positions where every key is
  selected, so the masked path has no real-weight verdict yet. The 1M context
  the card cannot hold (latent KV ≈ 84 GB at 1M; 0.3 GB at 2048) is likewise
  unmeasured.
- L5. Perf: measured with the AOCC release harness at the canonical 2048-token protocol: **1.04 t/s (perf) / 1.03 t/s (r3)**, TTFT 5.7 s (`tests/perf/perf_baselines.json`), host-store bound
  (4 experts × 71 layers × 3 projections of 6144² at ~0.53 B/param ≈ 17 GB
  streamed per token) plus one slot-table D2H per MoE layer (the same contract
  as Mistral-Small-4). See §R14 perf line.
- L6. The expert device cache is 120 slots of 9216 on this card (16.1 GB pinned
  leaves ~13 GB); the working set per token is 284 expert touches, so the
  streaming is the throughput ceiling.
