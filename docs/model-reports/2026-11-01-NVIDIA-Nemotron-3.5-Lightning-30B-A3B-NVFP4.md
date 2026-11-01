# NVIDIA-Nemotron-3.5-Lightning-30B-A3B (NVFP4) — integration report

| | |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, `model_type = "nemotron_h"`, `architectures = ["NemotronHForCausalLM"]` |
| Converted layout | `logestix-nemotron-nvfp4` (NVFP4 dense + streamed NVFP4 experts; F32 router, BF16 norms / conv / SSM scalars, BF16 embeddings + head) |
| Source | 20.1 GB ModelOpt `MIXED_PRECISION`: W4A16_NVFP4 routed + shared experts and `lm_head` (E2M1 + full-E4M3 per-16 scales + F32 `weight_scale_2`), per-tensor FP8 Mamba `in_proj`/`out_proj`, BF16 attention projections carrying FP8 KV-cache scales, F32 router; 52 shards |
| Converted tree | 18 GB on disk: 2944 routed experts (23 MoE layers × 128), 355 pinned tensors, 270 MTP tensors stored+counted, 12 FP8 KV scales dropped; ~13.4 GiB device-resident + 14.7 GiB expert slot cache |
| Hardware | 1 × RTX 5090 (32 GB), 750 GB host RAM, CUDA 13.4 |
| Runtime | token-by-token prefill/decode over the neutral NVFP4 GEMV, FP32 KV cache + FP32 Mamba-2 state, single stream |
| Date | 2026-11-01 |
| Reference | `transformers` 5.15.1 `NemotronHForCausalLM` (source oracle rebuilt locally by `tools/nemotron_nvfp4_oracle.py`) |

**Verdict at a glance:** the model runs locally, correctly, at **127.6 t/s**
(2 048 generated tokens). Getting there required fixing one real semantic defect
that the previous green suite could not see: the six full-attention layers of
this hybrid are **NoPE** — their reference class has no rotary module — and the
engine had been applying full RoPE to q/k since ingestion. RoPE is the identity
at position 0 and the Paris self-check is a short prompt, so the defect survived
weeks of green gates while corrupting every long prompt (tool calling produced
HTML-tag soup). After removing it, the engine tracks a BF16 reference built from
the same converted tree for **24/24 generated tokens** on tool prompts, and the
native tool round-trip returns `get_weather {"city":"Paris"}`.

---

## Publish-ready post

### Nemotron-3.5-Lightning-30B-A3B on one RTX 5090

Nemotron-3.5-Lightning is a 30B-parameter hybrid: 23 Mamba-2 layers, 23
mixture-of-experts layers (128 experts, top-6, plus one shared expert), and six
full-attention layers at 5/12/19/26/33/42. Only ~3B parameters are active per
token, which is why a 30B-class model fits a consumer card once the routed
experts are streamed instead of pinned.

Two facts about this checkpoint drive the whole port. First, it is **already
quantized**: NVIDIA shipped ModelOpt `MIXED_PRECISION` — 4-bit NVFP4 experts and
head, 8-bit FP8 Mamba projections, and *unquantized* BF16 attention. The
vendor's own recipe therefore says the attention projections are
precision-sensitive; our converter follows the same split after undoing the
ModelOpt encodings and re-quantizing through the repo's neutral codec (measured
per-tensor error vs source: 9.5 % on the attention projections, 5.3 % on the
experts, 0.19 % on the BF16 head). Second, its six attention layers are
**NoPE**: the reference implementation builds no rotary embedding and simply
ignores `position_ids`. Position in this model comes entirely from the Mamba-2
recurrence.

That second fact cost the most. Our engine had been applying full-dimension
RoPE to q and k in those six layers, which is invisible at position 0 (cos 1,
sin 0) and harmless for a seven-token self-check, but poisons any longer
context: measured against a BF16 reference built from the *same* converted
weights, the per-layer relative error grew from 0.09 at position 0 to **1.20 at
position 284**, with a 2× jump at every attention layer. A tool-framed prompt
(~300 tokens) degenerated into HTML-tag soup and never emitted a call. Deleting
the rotation fixed it outright: per-layer error at position 284 is now 0.003
(layer 0), 0.011 (first attention layer) and 0.23 at the output, cosine ≥ 0.973
— pure 4-bit accumulation — and greedy generation matches the reference for
24 of 24 tokens.

Finding it required an instrument the family did not have: a CPU BF16 oracle
(`tools/nemotron_nvfp4_oracle.py`) that dequantizes the ModelOpt checkpoint with
torch and assembles a stock `NemotronHForCausalLM`. It runs 16 tokens in eight
seconds on 160 cores, and because the same script can decode our own Arrow tree
back to BF16, it separates "the conversion is lossy" from "the runtime is
wrong" — the first question to ask when a green suite hides a defect.

Measured, on one RTX 5090: **127.55 t/s** and **120.39 t/s** (strict mode) at
2 048 generated tokens, 84 ms to first token, 242 W average, 13.4 GiB resident
plus a 14.7 GiB expert slot cache. Paris is bit-exact across three cycles,
tokenizer parity is 22/22 against the checkpoint's own `tokenizer.json`, and the
tool round-trip is green. FSI/CODE/DET remain declared debt: the source answers
the FSI templates exactly but the converted 4-bit body shifts them (FSI-01 gains
a spurious `*exp(r)` factor), and the source itself echoes the prompt on
CODE-TEST-01 and DET-TEST-01/02 — the residual-plane work that would lift the
body to ~8 effective bits is written up as the next step.

---

## Internal evidence

### What was ingested

The repo already carried both passports (`nemotron_moe_constants.h` for the BF16
distribution, `nemotron_moe_nvfp4_constants.h` for the pre-quantized one) with
identical geometry: 52 layers (23 mamba / 23 moe / 6 attention), hidden 2688,
Mamba-2 d_inner 4096 (64×64), 8 groups, state 128, conv 4, in_proj 10304; MoE
128 experts top-6 relu2 I=1856 + shared I=3712, route scale 2.5; 32 Q / 2 KV
heads at head_dim 128; vocab 131072; eps 1e-5. This session re-verified the
geometry against `config.json` and the 18 487-tensor inventory, then ran the
lifecycle end to end.

### Lifecycle, stage by stage

| stage | evidence |
|---|---|
| 0 triage | geometry cross-checked against both passports; tokenizer kind 9; ballast = 270 MTP tensors (stored+counted) + 12 FP8 KV-cache scales (dropped, counted); 18 GB converted tree fits the card |
| 1 converter | `lsxforge --nemotron_nvfp4` on the 52-shard source in 1m52s: 52 layers, 128 experts, 270 MTP cut, 12 FP8 KV scales dropped, 355 pinned tensors; synthetic E2E test green |
| 2 NVFP4 | per-tensor decode error vs source: attention 9.5 %, mamba/attention projections ~9.5 %, experts 5.3 %, head 0.19 % (BF16); block scales are the repo's full-E4M3 codec |
| 3 registration | `nemotron-3.5-lightning-30b-a3b-nvfp4` in `LsxModelFactory`; frozen-geometry mismatch refused; `lsxfabric --validate` loads and unloads |
| 4 tokenizer | `test_nemotron_moe_tokenizer_parity` (new): 22/22 exact ids, kind 9, no auto-BOS, against `tokenizer.json`; capture tool `tools/nemotron_tokenizer_capture.py`, fixture `tests/fixtures/nemotron/` |
| 4b tools | real-weight round-trip (new case): `get_weather {"city":"Paris"}`; prompt ids byte-identical to `apply_chat_template` |
| 5/6 kernels | conv/SSM/gated-RMSNorm/router/relu2 + GQA tests green; NoPE correction documented next to the launch |
| 8 selfcheck | Paris ×3 bit-exact, `logits_hash fe3b05c75a2033ba`; deterministic over cycles |
| perf | harness-seeded: perf 127.55 t/s, r3 120.39 t/s, TTFT 84.3 / 88.6 ms, 242 / 241 W |
| guards | `fast` 125/125 (naming, independence, suite, labels, absl, expert-store, arrow) |

### The defect, and how it was localised

1. **Observed**: the new real-weight tool round-trip failed (the model answered
   with prose instead of `<tool_call>`), while Paris passed and the prompt ids
   were verified byte-identical to `apply_chat_template`.
2. **Isolated the runtime from the weights**: the oracle was rebuilt from the
   source safetensors (BF16 dequant) and from our converted Arrow tree
   (decoded back to BF16). Both called the tool with identical ids; the engine
   did not. Weights → ruled out; runtime → suspect.
3. **Localised**: per-layer hidden states dumped from the engine against the
   tree reference showed the error flat at ~0.09 for a one-token prompt but
   growing with context (0.28 at pos 3, 0.60 at pos 43, 1.20 at pos 284), with
   a 2× jump at each of the six full-attention layers.
4. **Root cause**: reading the reference class completely (`modeling_nemotron_h.py`)
   showed `NemotronHAttention` has no rotary module — `apply_rotary_pos_emb` is
   defined in the file but never called, and `position_ids` is threaded through
   and dropped. The engine's `RunNemotronRope` was therefore a pure defect.
5. **Fix + verification**: remove the two rope calls, delete the dead kernel and
   its unit test, record the NoPE fact in both passports. Re-measured: 24/24
   token parity on the tool prompts, per-layer error 0.003/0.011/0.23 at pos
   284, tool round-trip green, Paris ×3 still bit-exact (new hash — the fix
   legitimately moves the logits).

### Remaining debt, measured

- **FSI** is red on the converted tree although the source answers 3/3:
  FSI-01 extracts `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)`, FSI-02/03 emit
  fenced / repeatedly-closed bodies. Mechanism: 4-bit body error on the
  attention projections (9.5 % relL2), which the vendor kept BF16. The
  documented lever is a residual plane on attention + Mamba projections
  (~8 effective bits, routed experts left 4-bit) at ~+1.1 GiB resident and a
  doubled attn+Mamba read — the gemma-4-26b recipe, not attempted here.
- **CODE/DET** cannot be captured from the source: it echoes the prompt body on
  CODE-TEST-01 and DET-TEST-01/02 (measured raw kept in `OPEN_TASKS.md` §R13).
- The expert store probes every non-MoE layer's expert files at load (3712
  failed `mmap` opens, debug-logged, ~30 ms); harmless — `OPEN_TASKS.md` §R13
  N2.
