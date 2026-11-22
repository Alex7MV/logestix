# TML-d24-varloop-research — ingestion report (2026-11-22)

## Publish-ready post

**What was converted.** *TML-d24-varloop-research* is a research release from
Thinking Machines Lab: the **variable-loop** variant of the Inkling 2
architecture. It is the same 24-layer sparse mixture-of-experts decoder as the
Inkling 2 Mini release — 24 physical layers, of which the middle twelve
(`[6, 18)`) are run repeatedly with shared weights — with one capability added:
the number of passes through those twelve layers is **chosen at inference time**
instead of being fixed by the checkpoint. The official weights were trained with
1 to 5 passes (sampled 1:2:6:2:1) and post-trained at 3.

The checkpoint ships as a single 18.19 GiB shard (plus a 0.59 GiB
`mtp.safetensors` its index does not list) and converts to an **18 GiB** artifact
with 645 pinned tensors and 3 072 routed-expert records. The dense plane is
device-resident (3.0 GiB minimum, 4.8 GiB at the planner's preferred layout) and
all 3 072 routed experts sit in one 14.71 GiB device slab — the artifact is
resident as a whole on a 32 GiB card. Conversion takes **17 seconds**.

**What "variable loop" is, and what it is not.** The saved config declares
`loop_config.is_varloop: true`, and the reference implementation **never branches
on that flag**: it reads `loop_num_passes` and appends `passes − 1` deep copies of
the looped layers' cache state, whatever that number is. So the flag is a
*capability declaration*, not a second code path, and the capability is exactly
what the model card says: one set of weights, any pass count in the trained
range. The engine therefore keeps the checkpoint's own default of 3 passes and
exposes the count as a load-time choice over `1..5`; the KV/conv slab is sized
for the ceiling (72 slots, one block of 12 per extra pass), so changing the count
never re-allocates and never moves a slot. A count outside the trained range is
clamped with a warning rather than run.

**The architecture forces the same six things as the sibling, and each was
verified against the release's own modelling code** rather than inferred:

1. **No RoPE anywhere.** Position enters through a learned bank of 16 relative
   profiles over 1 024 backward distances; q and k are L2-clipped per head to
   22.627 (`q` through a learned per-head scale, `k` weight-free) and `v` is
   RMS-normalized without a learned scale.
2. **Log scaling starts at position 8 192**, not at the 128 000 of the larger
   sibling — copying the wrong floor would leave the whole 8k engine context
   unscaled.
3. **Grouped-query attention is 3:1** (24 query heads over 8 kv heads).
4. **Depth mixing over block prefixes**: one block every 6 layers; a boundary
   layer appends its incoming hidden state to a stack *instead of* keeping a
   local residual; the four full-attention layers ask for three mixtures and
   everything else for one.
5. **Hybrid attention**: 20 sliding-window layers (1 024) and 4 full layers at
   `{5, 11, 17, 23}`; the full layers' k/v short convolutions take dynamic taps
   (a per-token projection summed with a static table) and swap the normalizer
   from `sqrt(k+1)` to `sqrt(2k+1)`.
6. **Mixture of experts**: 128 routed experts, top-8, no shared expert, sigmoid
   scoring with a selection-only FP32 correction bias, weights renormalized then
   scaled by 8.0, and a softcapped SwiGLU expert
   (`tanh(g/4)·4 · sigmoid(g) · tanh(u/25)·25`) accumulated in FP32.

**Nothing is requantized.** The release is mixed precision and says so only in
`hf_quant_config.json`: routed experts of layers 0–19 are **NVFP4** (E2M1
nibbles, per-16 E4M3 block scales, per-expert global scale `amax/2688`), layers
20–23 are **MXFP8** (E4M3 with a `ue8m0` exponent per 32), and everything else is
BF16 except the depth-mixing queries, the merged cross-pass message queries and
the router correction bias, which are FP32. All of it is carried byte-for-byte;
the eight-bit layers decode through the family's dedicated FP8 GEMV.

**The checkpoint's own traps** (each silent, each now pinned by a test):

- *The loop is declared in `text_config.loop_config` only.* A decoder that reads
  `num_hidden_layers` alone runs 24 layer applications instead of 48 and answers
  confidently.
- *The pass count is not a fixed property of the weights, but the trained range
  is not in the config either.* `num_passes: 3` is the post-trained default; that
  1 and 5 are legal comes from the model card, so the artifact records the
  ceiling explicitly and the loader refuses a ceiling below the declared count.
- *The fused expert slab is row-interleaved.* `w13_weight` stores gate and up rows
  alternating; taking the first half as gate and the second as up keeps every
  byte, plane size and per-expert `amax`, and only destroys knowledge.
- *The expert codec changes mid-stack* (NVFP4 → MXFP8 at layer 20) and only
  `hf_quant_config.json` knows.
- *Two cross-pass query tensors must be merged into one* `[12][4][2304]` in
  `[q, k, v, mlp]` order, and `attn_res.attn_queries.i` belongs to layer `i+1`.
- *`mtp.safetensors` is 0.59 GiB the index does not list* — a converter that walks
  the directory instead of the index would ingest it.
- *The vision tower is 422 tensors of the same checkpoint.* Stripped and counted;
  this ingestion is text-only and does not claim image input.

**A note for the model authors.** `is_varloop` is the single most important
property of this release and it appears nowhere in the checkpoint except one
boolean in `text_config.loop_config` — not in `generation_config.json`, not in the
README's configuration block, and not in the modelling code, which reads only
`num_passes`. A `passes` field in `generation_config.json` (the value the release
was post-trained at) plus a documented range would let an inference stack serve
this model without reading the model card. The same applies to the trained pass
range itself: it exists only in prose.

**What was not measured, and why.**

- **Context beyond 2 048 tokens.** The engine bound is 8 192 positions and the
  canonical throughput protocol is 2 048; the log-scaling path that activates
  from position 8 192 is therefore exercised only by kernel-level reference
  tests.
- **Pass counts above 5, and quality at 1 or 5.** The trained range is enforced,
  and the per-count numbers reported here are throughput and determinism, not
  benchmark accuracy — reproducing the model card's evaluation table is out of
  scope for an ingestion.
- **Image input.** The vision tower is stripped by design (text-tower ingestion),
  so the multimodal numbers in the model card do not apply to this build.
- **The MTP head.** Counted, never applied; speculative decoding is not
  implemented for this family.

## Measured numbers

_This section is filled from the gate runs below; every number is copied from the
gate output or from `tests/perf/perf_baselines.json`._

| measurement | result |
|---|---|
| SelfCheck ("The capital of France is" → "Paris") | PASS, 3 bit-identical cycles, `logits_hash=98a2c3c532c9b013`, 32-token window, 3 passes |
| knowledge probe (teacher-forced, one prefill per position) | mean NLL 3.4006 → perplexity **29.98**, top-1 `152717` (" Paris") |
| FSI parity gate | PENDING |
| CODE parity gate | PENDING |
| DET determinism gate | PENDING |
| native tool round-trip | shared with Inkling-2-Mini (byte-identical tokenizer + template): 56 framing assertions |
| scheduling/state suite | 72 assertions at 1/3/5 passes: application counts follow the active count, the last pass owns the last slab block, two identical requests reproduce their logits bit-for-bit |
| artifact / conversion | 18 GiB, 645 pinned tensors, 3 072 expert records, 422 vision tensors stripped, **17 s** |
| decode throughput, 2 048 tokens | PENDING |

## Pass-count behaviour

The row's reason to exist is that the pass count is a runtime choice, so the
engine reports what it actually ran (`ActiveLoopPasses()`,
`CacheSlotCount()`, `LayerApplications()`), and the synthetic pipeline suite pins
the three properties a "compiles and answers" implementation gets wrong:

- **the requested count is what runs** — 1, 3 and 5 passes produce 24, 48 and 72
  layer applications, the trace row count follows the active count, and the final
  residual row at 5 passes differs from the 3-pass one (a count that only adds
  bookkeeping would reproduce it bit for bit);
- **the slab is the ceiling's** — a 5-pass run addresses the last block of the
  same 72-slot allocation, whose keys are neither pass 0's nor untouched;
- **determinism at any count** — two identical requests at 1, 3 and 5 passes
  reproduce the token stream and the logits row bit for bit.

## Evidence

- Frozen passport and its geometry tests:
  `libs/lsxcommon/src/inkling_moe_varloop_constants.h`,
  `tests/test_inkling_moe_varloop_constants.cpp`.
- Converter: `libs/lsxcommon/src/inkling_moe_varloop_convert.cpp` over the shared
  looped pipeline (`inkling_moe_loop_convert.cpp`, parameterised by a per-row
  spec); `tests/test_inkling_moe_varloop_convert.cpp` proves the format tag, the
  card name, `varloop`/`loop_passes_max` and that `is_varloop` is enforced in
  **both** directions.
- Loader/topology: `tests/test_inkling_moe_varloop_loader.cpp` (capability,
  ceiling, slab-vs-application split, format gate).
- Pipeline: `tests/test_inkling_moe_varloop_pipeline.cu`.
- Model: `libs/lsxcommon/src/inkling_moe_varloop_model.cpp`,
  `tests/test_inkling_moe_varloop_model.cpp` (kType, pass resolution and
  clamping, sibling-artifact refusal, tool bridge).
- SelfCheck: `tests/test_inkling_moe_varloop_selfcheck.cpp`.
- Manifest row: `models/tml-d24-varloop-research.json`; catalogue
  `docs/MODELS.md`; gates `docs/SELFCHECK_MATRIX.md`; stage-0 trail and the
  verified semantics: `docs/open-tasks/live/models/tml-d24-varloop-research.md`;
  design `docs/superpowers/specs/2026-11-22-tml-varloop-ingest-design.md`.
- Source reference runner: `tools/inkling_loop_oracle.py` (the release's own
  transformers 5.24 `inkling2_loop` text tower, CPU bf16) reached through
  `tools/{fsi,code,det}_capture.py --type tml-d24-varloop-research`.
