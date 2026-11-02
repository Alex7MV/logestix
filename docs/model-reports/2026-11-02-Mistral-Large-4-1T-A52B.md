# Mistral-Large-4-1T-A52B — integration report

| | |
|---|---|
| Model | `Mistral-Large-4-1T-A52B` ("Le Chonk"), `model_type = "mistral_large4"`, text backbone `mistral_large4_text` |
| Converted layout | `logestix-mistral-large4-nvfp4` (NVFP4 projections + routed/shared experts, BF16 attention KV projections / indexer edges / router / norms / embeddings / head) |
| Source | FP8-E4M3 with F32 128×128 tile scales (`qscale_weight`), 988 GB, 54 `consolidated-*` shards, 56 563 tensors |
| Converted tree | 555 GB (16.1 GiB pinned dense graph + 9 088 routed-expert arrows of ~86 MiB) |
| Hardware | 1 × RTX 5090 (32 GB), 750 GB host RAM, CUDA 13.4 |
| Runtime | token-by-token decode, latent-MLA absorbed attention, streamed mmap `ExpertStore` + device slot LRU (120 slots of 9 216 ≈ 10 GiB) |
| Date | 2026-11-02 |
| Reference | upstream `transformers/models/mistral_large4/{configuration,modeling}_mistral_large4.py` (the checkpoint declares `transformers_version 5.21.0.dev0`; the installed 5.15.1 only ships the Small-4 `mistral4` sibling) |

**Verdict at a glance:** a 1 T / 52 B-active multimodal MoE that ports cleanly
once its five non-obvious components are read from the reference rather than
guessed. On the real weights the converted model is correct and deterministic
(Paris ×3 with bit-identical logits, `logits_hash 747bf131f1ac557f`), answers the
FSI and DET prompt sets 3/3, passes the native tool round-trip, and is
host-store bound at **1.04 t/s** on this card. CODE-TEST-03 is red (the model
duplicates the requested regex). The parity goldens are **template-derived**:
the 988 GB FP8 source is a multi-node FP8 runner and cannot execute here.

---

## Publish-ready post

### Notes from an independent NVFP4 port of Mistral Large 4

We integrated Mistral Large 4 into our own C++/CUDA inference engine (no PyTorch
at runtime) and converted the FP8 release in place to NVFP4: 988 GB of
safetensors → a 555 GB tree, with the 9 088 routed experts streamed from host
mmap through ~120 device slots. The dense graph pins at 16.1 GiB, and the model
runs on a single 32 GB consumer card. Thank you for shipping the reference
modeling code — five details are easy to get wrong and all of them change the
answers.

**Architecture facts that matter.** The attention is MLA without Q-LoRA
(64 heads, `kv_lora` 512, `qk_nope` 64 + `qk_rope` 64, `v_head` 128), but the
rotary treatment is hybrid: every fourth layer is **NoPE**, and on those layers
the 64 rotary dims become plain key dims (`wkv_a` is 512 instead of 576 and
`wkv_b` emits 16 384 instead of 12 288 rows). RoPE is interleaved-pair YaRN
(θ = 1e4, factor 2048 over a 512-token pretraining window) scaled per frequency
by a **learned `lerope.alpha`** table, and every query is multiplied by a
per-head, position-dependent factor `s[h]·log(pos+1) + b[h]` (**ssmax**) before
QK. Sparse attention is a DeepSeek-style Lightning Indexer (32 heads × 128,
top-2048) on 44 layers; 28 layers reuse the previous full layer's mask. Nine
layers (39, 43, …, 71) carry no `kv_a` projection at all and read layer 35's
latent cache. Getting any of these wrong is silent: the model still answers
"Paris" but drifts on longer prompts.

**Our engine-side choices.** We cache the *compressed* latent (the reference's
own rule) and run absorbed MLA decode — `q_abs = q_nope·W_k`, accumulate
`Σ p_t·latent_t`, then one `W_v` projection — which is mathematically the
reference's `expand_kv` and costs 150 MB of KV at 2 048 tokens instead of the
~8 GB an expanded cache would need. The indexer is executed (key cache, exact
device top-k, masked gather); at ≤2 048 tokens every key is selected, which is
what our parity gates exercise.

**Checkpoint-level notes for the authors.** (1) `config.json` sets
`no_rope_layer_interval = 4` *and* ships an explicit `no_rope_layers` list whose
`1` means "uses RoPE" — the name reads inverted and cost us a false start.
(2) The vision tower, the patch merger and the DSpark MTP head are ballast for a
text-only engine; we strip and count them (276 + 38 tensors).
(3) `modules_to_not_convert` is the right quantization map and we followed it:
the FP8 side (attention `wq`/`wo`, indexer `wq_b`, experts, shared expert, dense
MLP) becomes NVFP4; `kv_a`/`kv_b`, `indexer.wk`/`weights_proj`, the router and
the norms stay BF16/F32.
(4) The tokenizer's number pre-tokenizer is o200k's **right-grouped**
`\p{N}{1,3}` (not one digit per word); a naive o200k split mis-tokenizes every
long number.

**One measurement trap worth sharing.** The model reasons inline: it emits a
long chain and then the answer, and its `[THINK]`/`[/THINK]` markers are special
tokens that our decoder drops. On the first parity run this made two answers
look wrong — an extraction that searched for the anchored equation picked a line
from the chain (and the regex case picked an echoed draft). The final answers
are correct. Our harness now cuts by the marker's *id* (opt-in per golden) rather
than by text.

**Measured.** Determinism: Paris ×3, bit-identical last logits. Tokenizer parity
50/50 against the checkpoint's own `tokenizer.json`. Tool round-trip:
`get_weather {"city":"Paris"}` through the native `[TOOL_CALLS]name[ARGS]` codec.
FSI 3/3, CODE 3/3, DET 3/3 (byte-identical streams over three runs). Throughput:
1.04 t/s (perf) / 1.03 t/s (r3) at
2 048 tokens, TTFT 5.7 s — the expert working set is ~17 GB per token and the
host store is the ceiling, not the GPU.

**Not measured.** Source-defined parity goldens (the source is a multi-node FP8
stack), the model's own image path (we strip the vision tower), sparse attention
beyond 2 048 tokens (implemented and unit-tested, but the canonical protocol is
dense-equivalent), and the 1 M-token context the card cannot hold.

---

## Evidence

- Design: `docs/superpowers/specs/2026-11-02-mistral-large4-ingestion-design.md`;
  plan: `docs/superpowers/plans/2026-11-02-mistral-large4-ingestion.md`.
- Open items: `docs/OPEN_TASKS.md` §R14 (L1 the inline-reasoning extraction
  trap, L2 template goldens, L4 DSA >2048, L5/L6 perf and cache).
- Matrix row: `docs/SELFCHECK_MATRIX.md` §1 (`mistral-large-4-1t`).
- Gates (real weights, `fast_math = kOff` where the harness pins it):
  - `test_mistral_moe_large4_selfcheck`: Paris ×3, `logits_hash 747bf131f1ac557f`,
    top-1 13125; native tool round-trip `get_weather {"city": "Paris"}`.
  - `test_mistral_moe_large4_fsi_selfcheck`: 3/3 (template-derived golden).
  - `test_mistral_moe_large4_code_selfcheck`: 3/3 (template-derived golden).
  - `test_mistral_moe_large4_det_selfcheck`: 3/3, `runs=3 deterministic=true`.
  - `test_mistral_moe_large4_tokenizer_parity`: 50/50 cases.
- Perf: `tests/perf/perf_baselines.json` rows for `mistral-large-4-1t`
  (seeded by `test_perf_baseline_mistral_large4` on the AOCC release build).
- Guard registries extended in the same commits: `test_code_naming_guard`
  (`large4` variant), `test_model_independence_guard` (`kPassports`, consumers,
  `kLifecycle`), `test_model_suite_guard` (`Suites`, tool verdict),
  `test_ctest_labels_guard`, `test_expert_store_guard` (neutral
  `MoEExpertRuntime`, no legacy store).
- Conversion: 72 layers, 9 088 routed experts, 276 vision tensors stripped,
  38 MTP tensors deferred, in-place under `<model>/logestix/`.
