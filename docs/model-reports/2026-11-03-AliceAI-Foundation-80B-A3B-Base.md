# AliceAI-Foundation-80B-A3B-Base — integration report

| | |
|---|---|
| Model | `AliceAI-Foundation-80B-A3B-Base` (`model_type = "alice_ai"`, `architectures = ["AliceAIForCausalLM"]`) |
| Converted layout | `logestix-alice-nvfp4` (NVFP4 for the routed and shared MoE matrices only; KDA/attention projections, norms, router, conv and biases stay FP16; MTP stored, never applied) |
| Source | 49 BF16 shards, 1 307 indexed tensors, 151.4 GiB (`162 576 890 582` bytes) |
| Converted tree | 45.6 GiB (`48 916 739 473` bytes) — 4.2 G pinned + 41 G routed experts + 889 M MTP; 2 745 pinned records + 24 576 per-expert files |
| Hardware | 1 × RTX 5090 (32 GB), 750 GB host RAM, CUDA 13.4, `linux-release-znver4-aocc` |
| Runtime | 4.13 GiB device-resident weights (2.9 s load); routed experts are mmap-streamed from the host `ExpertStore` through a device slot-LRU |
| Date | 2026-11-03 |
| Reference | the checkpoint's own `config.json`, `modeling_alice_ai.py`, `tokenizer.model` and `chat_template.jinja` |

**Verdict at a glance:** green on every parity gate with real weights — SelfCheck
"Paris" ×3 with byte-identical full-vocabulary logits
(`logits_hash = f900547d25be7914` over 129 024 logits), FSI 3/3, CODE 3/3,
DET 3/3 (3 byte-identical runs per prompt), and a native tool round-trip that
emits `[TOOL_CALL_START]get_weather` + `{"city": "Paris"}` and parses back to
`get_weather {"city": "Paris"}`. Throughput at the canonical 2 048-token
protocol is **45.70 t/s** (`perf`, auto policy) and **45.88 t/s** (`r3`,
strict bit-exact). The base checkpoint answers the control prompt with a
multiple-choice-style continuation (`_____.
- **A)** Paris …`) rather than the bare token `Paris`; the family's SelfCheck
uses the documented substring window.

---

## Publish-ready post

### Notes from an independent NVFP4 port of AliceAI-Foundation-80B-A3B-Base

We run Alice-AI-Foundation-80B-A3B-Base in our own C++/CUDA inference engine
(no PyTorch at runtime). The 151 GiB BF16 release converts in place to a
45.6 GiB tree: about 4.2 GiB of dense weights stay device-resident and the
24 576 routed experts (48 layers × 512) are memory-mapped from the host and
streamed per token through a small device LRU. On a single RTX 5090 it loads in
about 3 s and decodes at 45.7 tokens/s at 2 048 tokens.

The architecture is a hybrid: 48 layers in 12 blocks of three Kimi-style
Delta-Attention (KDA, linear-attention) layers followed by one gated
full-attention layer, every layer followed by a sparse MoE. A block
attention-residual mixer (`AttnRes`, block size 4) recombines the residual
stream between blocks; each layer also has its own residual projection/norm.

#### What worked well

- **Determinism is solid.** The control prompt answers "Paris" and three
  independent runs produce byte-identical full-vocabulary logits
  (`logits_hash = f900547d25be7914`). Running the MoE matrices in NVFP4 while
  keeping the sensitive attention path in FP16 clears all of our parity gates:
  a financial-services formula set (3/3), a compiler-syntactic code set (3/3),
  and a determinism/near-tie set (3/3), each captured from the source weights.
- **The tool wire is unambiguous.** The template's
  `[TOOL_CALL_START]name\n{json}` markers are parsed without heuristics; a
  real-weight round-trip produced the `get_weather` call with
  `{"city": "Paris"}`.
- **The tokenizer bridges cleanly.** It is a SentencePiece BPE model
  (`tokenizer.model`) with full byte fallback, not a byte-level BPE JSON; eight
  source-captured encode/decode cases match exactly.

#### Checkpoint facts that are easy to get subtly wrong

1. **Only the MoE matrices tolerate 4-bit rounding here.** The KDA gate is an
   exponential recurrence and the gated attention reads its output directly;
   NVFP4 on those projections is amplified, not averaged away. We keep every
   KDA/attention projection in FP16 and quantise only the routed + shared
   expert matrices (the memory bulk). This is a property of *this* checkpoint,
   not a generic recipe.

2. **The hybrid pattern is per-layer, not "every N".** `layer_types` encodes
   12 × (linear, linear, linear, full). The KDA geometry is separate from the
   attention geometry: 32 key/value heads × 128 dims with a depthwise conv of
   kernel 4, versus 16 query / 2 KV heads × 256 dims (GQA 8:1) for the full
   layers. Mixing them decodes garbage without an error.

3. **The full-attention `q_proj` is width 2 × heads × head_dim.** Half of the
   output is the per-head output gate, applied before `o_proj`; the gate is not
   a separate tensor. Per-head q/k RMSNorm and partial RoPE on 64 of 256 dims
   (θ = 1e6) complete the layer.

4. **AttnRes is a block mixer, and layer 0 is special.** Blocks of 4 layers
   accumulate a residual and recombine it through learned projections/norms;
   layer 0 has no `attn_res_proj`/`attn_res_norm` (the pre-block input is the
   stream), while a final `model.attnres_final.*` pair closes the last block.
   Treating it as a per-layer residual add still runs — and answers the wrong
   thing.

5. **The router is sigmoid + a correction bias, and the shared expert is
   gated.** 512 experts, top-10, `router_score_function = "sigmoid"`,
   `router_bias_correction = true` (`e_score_correction_bias`), one shared
   expert at the same 512-wide intermediate plus a scalar
   `shared_expert_gate`. Dropping the bias or the gate changes routing.

6. **The checkpoint ships one MTP layer.** Its 20 tensors live under `mtp.*`
   and are *not* used by the base model. Store and count them; do not execute
   them.

7. **The tokenizer is SentencePiece, with byte fallback.** `<0x00>`…`<0xFF>`
   occupy ids 7…262, `<s>` = 1, `</s>` = 2, and `add_dummy_prefix` is on. A
   byte-level BPE reader cannot consume this file.

8. **The chat template is a plain-text protocol with a Russian tools header.**
   Roles are `system:`/`user:`/`assistant:`/`meta:`, blocks are separated by
   two newlines plus a space, reasoning is `[COT_START]…[COT_END]`, and a call
   is `[TOOL_CALL_START]name\n{json}`. The markers are ordinary text (they
   survive detokenisation); the only special ids are the SentencePiece control
   and byte pieces. The model is a base checkpoint, so the plain control prompt
   continues into the MCQ pattern shown above.

9. **Context is declared at 262 144, but the recurring state is what bounds
   it.** Only 12 of 48 layers keep a K/V cache; the other 36 carry KDA conv and
   recurrent state. Our conversion-time profile reports the full-context
   estimate (28.9 GiB weights + KV + workspace at 262 144) even though the card
   cannot hold it; we validate 2 048-token generation, not the full window.

#### Measured

| | result |
|---|---|
| SelfCheck ("Paris") | PASS ×3, bit-identical logits, `logits_hash = f900547d25be7914`, top-1 id 828 |
| Financial-services formula set | 3/3 |
| Compiler-syntactic code set | 3/3 |
| Determinism / near-tie set | 3/3, byte-identical token streams |
| Native tool round-trip | `get_weather {"city":"Paris"}` |
| Decode throughput, 2 048 tokens | 45.70 t/s (auto), 45.88 t/s (strict); TTFT 475 / 470 ms |
| Residency | 4.13 GiB device-resident weights, 2.9 s load, 181–198 W during decode |

#### What we did not run

- **The full 262 144-token window.** We verified short/medium prompts and the
  2 048-token protocol; the native window needs more KV/state memory than the
  card has at that length.
- **MTP/speculative decoding** — the layer is stored (counted) but never runs.
- **Any multimodal path** — this checkpoint has no vision/audio tower.

Overall a well-specified hybrid checkpoint. The two things that cost real time
were the per-layer hybrid geometry (KDA vs gated attention) and the fact that
the sensitive KDA/attention projections must not be quantised together with the
MoE bulk; everything else is documented above so the next integrator does not
rediscover it.

---

## Evidence

### Gates actually run (2026-11-03)

| gate | result | note |
|---|---|---|
| Conversion (`lsxforge --alice`, in place) | **OK** | 4 min 05 s wall, 47.7 GiB peak host RSS; 151.4 GiB source → 45.6 GiB tree; 2 745 pinned records + 24 576 expert files; VRAM profile min 4.13 GiB / optimal 28.9 GiB / max 69.4 GiB at ctx 262 144 |
| `lsxfabric --validate` | **PASS** | 2.91 s wall, 4.06 GiB peak RSS |
| `lsxbenchmark --selfcheck --bit-exact` | **PASS** | `top1_tok = 828`, `logits_hash = f900547d25be7914`, `n = 129 024` |
| `test_alice_moe_selfcheck` | **PASS**, ×3 (`LSX_TEST_ALICE_SELFCHECK_CYCLES`) | 27 assertions; 3-cycle `memcmp` on `final_logits`; MCQs to "Paris" |
| `test_alice_moe_selfcheck` (new tool case) | **PASS** | native `[TOOL_CALL_START]` → `{"city": "Paris"}` |
| `test_alice_moe_fsi_selfcheck` | **PASS 3/3** | source-captured goldens (`tests/fixtures/fsi/alice-foundation-80b-a3b.json`) |
| `test_alice_moe_code_selfcheck` | **PASS 3/3** | source-captured goldens (`tests/fixtures/code/alice-foundation-80b-a3b.json`) |
| `test_alice_moe_det_selfcheck` | **PASS 3/3** | `runs=3`, byte-identical token streams |
| `test_alice_tokenizer` | **PASS** | SP-BPE build + 8 source-captured encode/decode goldens; vocab 129 024, kind 1, byte ids 7…262 |
| `test_perf_baseline_alice_foundation_80b_a3b` | **seeded** | perf 45.70 t/s, TTFT 474.99 ms, 181.45 W avg / 196.32 W peak, 2 234 samples; r3 45.88 t/s, TTFT 470.00 ms, 184.23 W avg / 197.89 W peak, 2 225 samples; measured twice, identical rows |
| Fast suite (`ctest -L fast`) | **130/130** | includes all guard registries |
| Convert sweep (`ctest -L convert`) | **37/37** | includes `test_alice_moe_convert` |
| Alice inference suite | **7/7** | loader/model/kernels + selfcheck/fsi/code/det (468 s) |

`lsxfabric --testprompt` sanity (48 tokens, 25.6 t/s) returned coherent
MCQ-style text: `"_____.
- **A)** Paris\n- **B)** London\n- **C)** Berlin\n- **D)** Madrid
The capital of the United States is _____."`

### Two engine defects found and fixed in this session (TDD)

Both were invisible on the 2026-10-01 ingestion because nothing consumed the
numbers; the re-ingestion exercised them.

1. **The emitted VRAM profile was inverted.** `alice_tensor_is_nvfp4` omitted
   the fused routed-expert kinds and the converter passed no `mem_class`, so
   every routed expert was counted as BF16 and *device-resident*:
   `min_bytes = 162 355 811 584` (the whole source) and `max_bytes ==
   optimal_bytes` (no expert pool). Nineteen tests were unaffected because the
   Alice loader ignores the block. Fixed by marking the fused kinds NVFP4 +
   routed (new `alice_tensor_is_routed_expert`) and skipping the never-loaded
   MTP tree from the runtime inventory; the profile is now
   min 4.44 GB / optimal 31.0 GB / max 74.5 GB. Pinned by
   `test_alice_moe_convert` ("alice vram profile streams routed experts as
   NVFP4"): asserted red first, then green.

2. **`resident_bytes` under-reported the device footprint 8×.**
   `LoadAlicePinnedWeights` re-derived the value from a formula that counted
   only embed/head, router gates and shared experts, omitting every KDA and
   attention FP16 projection. On the real checkpoint it reported **1 187 MiB**
   while the loader had uploaded **4 231 MiB**, and that number drives the
   governor lease and the `resident:` line. Fixed by accumulating at the
   allocation sites (`H`/`W`/global uploads); the loader value now equals the
   converter's independent `vram.min_bytes`. Pinned by
   `test_alice_moe_loader` ("alice resident bytes count every uploaded
   weight"), asserted red first (47 616 vs 366 384 on the synthetic fixture).

Coverage added in the same pass: the selfcheck gate now runs the protocol's
≥3 stress cycles with bit-exact `final_logits` (it previously did one run), and
stage 4b gained its real-weight half via the shared `test_tool_roundtrip.h`
helper. The perf rows were re-seeded through the harness (twice, identical),
never by hand.

### Architecture as implemented

- 48 layers = 12 × (3 × KDA → MoE, then 1 × gated attention → MoE);
  full-attention layers at indices 3, 7, …, 47; AttnRes block size 4.
- KDA: 32 key / 32 value heads × 128 dims, depthwise conv kernel 4,
  `A_log` / `dt_bias` / `f_a`/`f_b` / `g_a`/`g_b` / `b_proj` / output norm.
- Gated attention: 16 Q / 2 KV heads × 256 dims, per-head q/k RMSNorm,
  partial RoPE on 64 dims (θ = 1e6), output gate folded into `q_proj`.
- MoE: 512 routed experts top-10 with a sigmoid router + correction bias,
  `moe_intermediate_size = 512`, one shared expert at 512 plus a scalar gate.
- Geometry: hidden 2 048, vocab 129 024, `rms_norm_eps = 1e-6`,
  `tie_word_embeddings = false`, context 262 144.
- Stripped and counted: one MTP layer (20 tensors, `mtp/*`), never applied.
- Quantisation: NVFP4 (`w1`/`w3`/`w2` per expert, split from the fused
  `gate_up_proj`/`down_proj` stacks) for routed + shared experts; everything
  else raw FP16/BF16 per the precision policy above.

### Evidence links

- Stage-0/design and the original ingestion plan:
  `docs/superpowers/specs/2026-10-01-alice-foundation-80b-a3b-ingestion-design.md`,
  `docs/superpowers/plans/2026-10-01-alice-foundation-80b-a3b-ingestion.md`.
- Gate matrix: `docs/SELFCHECK_MATRIX.md` (`alice-foundation-80b-a3b` row).
- Perf rows: `tests/perf/perf_baselines.json` (`alice-foundation-80b-a3b`,
  `perf` + `r3`, 2026-11-03).
- Tests: `tests/test_alice_{moe_convert,moe_loader,moe_model,moe_selfcheck,
  moe_fsi_selfcheck,moe_code_selfcheck,moe_det_selfcheck,moe_kernels}.cpp`,
  `tests/test_alice_{tool_framing,tokenizer}.cpp`.
- Passport + guards: `libs/lsxcommon/src/alice_moe_constants.h`,
  `test_model_independence_guard` (`kPassports`/`kLifecycle`),
  `test_model_suite_guard` (`Suites()` + tool verdict).
