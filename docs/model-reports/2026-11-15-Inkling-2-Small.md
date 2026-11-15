# Inkling-2-Small — ingestion report (2026-11-15)

## Publish-ready post

**What was converted.** *Inkling-2-Small* (594B total / 17B active, architecture
`inkling2`) was adapted to a C++20/CUDA inference engine that targets Blackwell
only. The 367 GB checkpoint (24 safetensors shards plus a 3.4 GB MTP file that
the index does not list) becomes a **366 GB** tree that is served directly from
host memory: the dense planes upload to a single RTX 5090 (12.4 GiB minimum,
14.5 GiB at the planner's preferred layout) and the 18 432 routed-expert records
stay memory-mapped on the host and stream in per token.

**The artifact keeps the release's own precision — nothing is requantized.** The
checkpoint is mixed-precision, and it says so in exactly one place:
`hf_quant_config.json`. Layers 0–39 ship routed experts in **NVFP4** (E2M1
nibbles, per-16 E4M3 block scales, a per-expert global scale of `amax/2688`);
layers 40–47 ship them in **MXFP8** (E4M3 weights with a `ue8m0` exponent per 32
values); everything else is BF16, except the depth-mixing queries and the router
bias, which are FP32. All of it is carried byte-for-byte.

That decision was measured, not assumed. Requantizing the eight MXFP8 layers to
4 bits — the obvious way to shrink the tree — costs **16.0 / 16.5 / 16.9 %
relative L2** on the MoE output of layers 42 / 45 / 47, against 9.5 % on the
dense projections that were already enough to push the model into a repetition
loop. So those layers keep 8 bits, decoded by a dedicated FP8 GEMV, and the tree
is 45 GB larger and twice as expensive to read on eight of the forty-eight
layers. Dense BF16 weights are widened to FP16 on the device, which is not
bit-exact: 0.33–0.40 % of them land in FP16 subnormals and 0.0001–0.0005 %
underflow to zero.

**What the checkpoint forced into the engine.** Inkling is not a transformer with
rotary positions. Six semantics break a generic decoder, and each of them was
verified against the release's own modelling code rather than inferred:

1. **No RoPE anywhere.** Position enters through a learned bank of `d_rel` = 16
   profiles spanning 1024 backward distances: each token's relative state
   projects onto the bank and gathers one bias per distance, zero beyond
   `rel_extent`. q and k are not rotated; they are **L2-clipped** per head to a
   threshold of 22.627 (k weight-free, and the learned per-head `q_l2_clip`
   scale is applied first), and v is RMS-normalized without a learned scale.
2. **Attention is gated and sink-anchored.** Every head carries a learned sink
   logit that competes in the softmax without contributing a value, and the
   attention output is multiplied by `sigmoid(gate_proj(x))` before the output
   projection. There is no QK-norm.
3. **Four short causal convolutions per layer**, each of the form
   `(conv(x) + x) / sqrt(k+1)` with `k` = 4 — one after attention, one after the
   MLP, and one each on the raw k and v projections. On the eight full-attention
   layers the k/v taps are **dynamic**: a per-token projection sums with a static
   per-channel tap table, and the normalizer becomes `sqrt(2k+1)`. Getting the
   order wrong (convolving the clipped k instead of clipping the convolved k) is
   silent.
4. **Depth mixing replaces a plain residual.** Every layer's attention input
   (layer > 0) and every MLP input is a softmax mixture over a stack of block
   prefixes, with block size 6 and a *position-independent* query. A block
   boundary (`layer % 6 == 0`) appends the incoming hidden state to the stack
   **instead of** keeping a local residual, so a boundary layer's residual is the
   attention output alone. The eight full-attention layers ask for **three**
   mixtures (separate q/k/v inputs); sliding layers and MLP inputs ask for one;
   the final norm input uses a dedicated output query.
5. **MoE routing is sigmoid with a selection-only bias.** 384 routed experts,
   top-8, no shared expert. Scores are `sigmoid(gate(x))`; a learned per-expert
   bias is added **only to choose** the top-8 (it is not part of the weight); the
   chosen scores are renormalized to 1 and scaled by 8.0. The expert itself is a
   softcapped SwiGLU — `tanh(g/4)·4 · sigmoid(g) · tanh(u/25)·25` — accumulated in
   FP32, where the two soft caps are the in-repo weights' own range limiters, not
   a numerical guard.
6. **Hybrid layer types.** 40 sliding-window layers (1024) and 8 full-attention
   layers at ids `{5, 11, 17, 23, 29, 35, 41, 47}`; the full ones additionally
   log-scale q and the relative states from position 128 000. A uniform stack
   assumption decodes fluent garbage.

The reference used throughout was the release's own code —
`transformers.models.inkling2` (5.23), not the older `models.inkling`, which is a
**different architecture** (no sinks, no L2 clip, scale `1/head_dim` instead of
`head_dim**-0.5`, no dynamic convolutions, no depth mixing) and produces
plausible, wrong logits if copied. The engine was validated three ways: layer by
layer against an independent float64 re-derivation from the artifact, against the
release's own `Inkling2TextModel` loaded with real dense weights and on-demand
expert dequantization (relL2 **3.7e-4** over 6 layers × 5 positions), and against
a truncated synthetic-weight build of that same class (identical top-5).

**Artifact-level defects found in the checkpoint, and suggestions to the
authors.** The first three are traps that break a consumer silently; the fourth
cost this ingestion a full week.

- *The fused expert slab is row-interleaved, and nothing says so.*
  `mlp.experts.w13_weight` (and `w13_weight_scale`) stores gate and up rows
  alternating — `[gate₀, up₀, gate₁, up₁, …]` — and must be de-interleaved before
  the two projections are split. Observed failure mode: taking the first half as
  gate and the second as up keeps every byte, every plane size and every
  magnitude, and reproduces the recorded per-expert `amax` exactly, so byte-level
  verification, size checks and *any* reference that reads the artifact all agree
  — including the release's own modelling code fed the dequantized planes. What
  it destroys is the model's knowledge: mean perplexity over "The capital of
  France is Paris" becomes **12 881** instead of 30.4 and the answer becomes
  " the" instead of " Paris". Suggested fix: state the packing order in
  `config.json` (an `expert_slab` field naming the interleave) or ship the fused
  tensor already de-interleaved. For reference, the release's own conversion
  table expresses it as a single `Interleave(dim=1)` on both tensors, which is
  easy to miss because the same table renames the attention projections as plain
  one-to-one renames.
- *The expert codec changes mid-stack, and only `hf_quant_config.json` knows.*
  Layers 0–39 are NVFP4, 40–47 are MXFP8. `config.json`'s quantization block
  describes neither. A converter that assumes one codec decodes a sixth of the
  stack as noise. Suggested fix: emit the per-layer codec into `config.json`, or
  split the layers into separately-named module groups.
- *`attn_res.attn_queries.{i}` belongs to layer `i+1`, not layer `i`,* because
  layer 0 has no attention mixture at all. `attn_res.mlp_queries.{i}` is indexed
  by layer, as expected. Two banks, both keyed by layer, both adjacent in the same
  file, with different origins. Suggested fix: rename to
  `attn_queries_of_layer_{i+1}` or pad the list to 48 entries with the unused
  first slot.
- *`mtp.safetensors` is a 3.4 GB file that `model.safetensors.index.json` does not
  list.* It holds 5 multi-token-prediction layers, correctly ignored, but a
  converter that enumerates the directory rather than the index silently ingests
  them, and one that trusts the index silently ignores the file — both without a
  warning. Suggested fix: name it in the index with a role field, or move it out
  of the checkpoint directory.
- *`eos_token_id` is `<|content_model_end_sampling|>` (200006), not
  `<|endoftext|>` (199999).* Both stop the model, so a consumer that picks the
  wrong one still terminates — it just terminates somewhere else. There is no
  `bos_token_id` and no pad id. Suggested fix: document the intended stop token
  in the model card; the two ids are both "end" tokens with different meanings.
- *The chat template appends no assistant opener.* The frame ends at the user's
  `<|end_message|>`, so a 20-id prompt asks the model to continue, and the model
  emits `<|message_model|><|content_thinking|>` itself. This is not a defect so
  much as a fact that decides whether a deployment looks broken — a harness that
  expects an assistant marker in the prompt will conclude the model is looping.
  Suggested fix: document the framing contract (the model writes its own
  opener), or add the opener to the template behind a flag.

**Measured numbers.** All on the real 366 GB tree, one target at a time, RTX
5090 (32 GiB), CUDA 13.5 on a 13.4 driver:

| measurement | result |
|---|---|
| SelfCheck ("The capital of France is" → "Paris") | PASS, 3 bit-exact cycles, `logits_hash=323d30463a3d2b72`; the stream opens `<\|message_model\|><\|content_thinking\|>The user is asking a simple factual question:` |
| FSI parity gate | 3/3 (template-derived goldens — see "not measured") |
| CODE parity gate | 3/3 (template-derived goldens) |
| DET determinism gate | 3/3, three byte-identical runs per probe |
| native tool round-trip | PASS, 49 framing assertions against the checkpoint's own template |
| knowledge probe (6-token sentence, one prefill per position) | mean NLL 3.41 → perplexity **30.4**, top-1 `" Paris"` |
| layer-by-layer parity vs an independent float64 reference | 2–5e-4 relative L2 at 7 depths spanning both codecs |
| agreement with the release's own modelling code | 3.7e-4 over 6 layers × 5 positions |
| decode throughput, 2048 tokens, auto policy | **1.47 tok/s** (TTFT 14.49 s, 95.3 W average, 120.9 W peak) |
| decode throughput, 2048 tokens, strict/bit-exact | **1.46 tok/s** (TTFT 14.47 s, 96.7 W average) |
| resident dense weights | 12.4 GiB minimum / 14.5 GiB preferred of a 32 GiB card |
| artifact | 366 GB on disk, 18 432 expert records + 996 pinned tensors |
| conversion | 10 min wall (367 GB read, GPU-side dequantization of the 8 FP8 layers) |
| stripped and counted, never applied | 422 vision tensors, 5 MTP layers |

Throughput is read-bound and it is the arithmetic, not a tuning failure: one token
reads 8 of 384 experts in each of 48 layers — 384 expert records, at 0.5 B/element
on layers 0–39 and 1.0 B/element on 40–47, i.e. roughly 7 GB per token against a
366 GB artifact on a 32 GiB card, so every token is a first-touch read. The stage-0
estimate written before the engine existed predicted "0.5–2 tok/s" from that
figure. Two corroborations: **95 W average on a 575 W card** means the SMs idle
through each layer waiting for the next expert slab, and the auto and bit-exact
modes differ by 0.7 %, so there is almost no arithmetic left that tuning could
speed up.

**What was not measured, and why.**

- **The three parity goldens are template-derived, not source-derived.** The
  594B source needs ~500 GB of aggregated VRAM across several GPUs to execute and
  ships its own quantization stack, so it cannot be run on a single 32 GiB card.
  The goldens therefore certify that the converted model produces the canonical
  answers, not that it reproduces the source's answers. The gap is narrower than
  usual here — the release's own modelling code was executed on real dense
  weights and agrees with the engine to 3.7e-4 — but it is a gap and it is
  recorded as one.
- **Long-context behaviour beyond 2048 tokens,** and the sliding-window (1024)
  versus full-attention split at long offsets. The perf protocol is fixed at
  2048; the checkpoint's own 1M-token context is not exercisable here. The
  log-scaling path that only activates from position 128 000 is consequently
  **untested on real weights** — it is exercised only by the kernel-level
  reference tests at synthetic positions.
- **The MTP head** (5 layers, 3.4 GB). Stripped and counted; speculative decoding
  is not implemented for this family.
- **The vision tower** (422 tensors, 0.33B). Stored and counted, never applied;
  this ingestion is text-only.
- **A distributional perplexity sweep** against the FP8/BF16 source. The gates are
  exact-answer probes and the knowledge probe is a single sentence; the 30.4
  perplexity figure is a sanity bound, not a quality measurement.
- **Multi-GPU or expert-parallel serving.** The engine is single-device; the
  working set streams from host memory.

## Evidence

### Stage 0 (triage) and the plan

`docs/open-tasks/2026-11-14-inkling-2-small-ingest.md` — classification (new
family, own frozen passport), the geometry table read from `config.json` +
`hf_quant_config.json` + safetensors headers, the tokenizer verdict (tiktoken BPE,
kind 2, `added_tokens` audited), ballast counted, and the feasibility numbers for
this box. That file also carries the full investigation trail, including the
eleven defects found and fixed.

### Registry and generated artifacts

- `models/inkling-2-small.json` — the only hand-written registry row. It
  generates `model_manifest.h` (39 rows), `model_targets.cmake` (all nine stages
  with label / timeout / skip code / GPU lock / env), `model_registry.inc` and the
  `Suites()` row.
- `docs/MODELS.md` — regenerated; the family's section reports `perf: 1.46 t/s
  (r3, 2026-11-15)`.
- Format tag `logestix-inkling-2-small` (renamed from
  `logestix-inkling-2-small-nvfp4` once the tree carried two codecs plus BF16
  planes; the tree was re-converted).
- `docs/OPEN_TASKS.md` — §R19 has moved to the Archive with the full trail.

### Gate results

| gate | target | result |
|---|---|---|
| SelfCheck | `test_inkling_moe_selfcheck` | PASS, `logits_hash=323d30463a3d2b72`, 3 bit-exact cycles |
| FSI | `test_inkling_moe_fsi_selfcheck` | 3/3, 1406 s through ctest |
| CODE | `test_inkling_moe_code_selfcheck` | 3/3 |
| DET | `test_inkling_moe_det_selfcheck` | 3/3, `deterministic=true` on all, 3 runs each |
| tool round-trip | `test_inkling_moe_tool_framing` | 49 assertions (label `fast`) |
| knowledge | `[sanity]` case in the selfcheck TU | mean NLL 3.41, top-1 `" Paris"` |
| kernels | `test_inkling_moe_kernels` | 1588 assertions / 16 cases |
| model / loader / convert | `test_inkling_moe_model`, `..._loader`, `..._convert` | 440 / 111 / 448 assertions |
| constants | `test_inkling_moe_constants` | frozen passport, label `fast` |

Goldens: `tests/fixtures/{fsi,code,det}/inkling-2-small.json`, each recording in
`source` that it is template-derived and why.

Perf rows: `tests/perf/perf_baselines.json` — `inkling-2-small`, modes `perf`
(1.47 tok/s) and `r3` (1.46 tok/s), 2048 max_gen, TTFT 14 485 / 14 469 ms,
power 95.28/120.85 W and 96.71/101.01 W, 69 448 / 69 880 samples. Seeded by
running `test_perf_baseline` on a Release build, not by editing the file.

`docs/SELFCHECK_MATRIX.md` — §1 row: SelfCheck PASS (framed), FSI 3/3, CODE 3/3,
DET 3/3, with the template-derived caveat stated next to the row.

### The defect that mattered, with its measurements

The red gate was **one** defect, and it was in the conversion, not the kernels:
the row-interleaved expert slab. The measurements that broke the symmetry between
engine and references:

| measurement | split-half (before) | de-interleaved (after) |
|---|---|---|
| mean NLL over "The capital of France is Paris" | 9.46 → ppl **12 881** | 3.41 → ppl **30.4** |
| top-1 after "The capital of France is" | `" the"` | **`" Paris"`** |
| ppl of "France" after "The capital of" | 5.9e5 | 109 |
| SelfCheck first ids | `" 0.9.9.0.0.9.0.0…"` | `<\|message_model\|><\|content_thinking\|>` |

It was confirmed before committing to a 366 GB re-conversion by a host-side row
gather injected behind an env flag in the expert-staging path. `test_inkling_moe_convert`
now pins the pairing row by row (gate row `r` == source row `2r`, up row `r` ==
source row `2r+1`) for both codecs, weights and scales.

The same knowledge gate then exposed a second, independent defect:
`InklingSession::EnsureCapacity` copied `kv_rows * LayerStride()` bytes per layer,
but `LayerStride()` includes the capacity, so the copy over-read the old
allocation on every growth. The runtime rejected it with
`cudaErrorInvalidValue`, which was picked up by the *next* request's first
unchecked launch and reported as `[embed] embed row (cuda 1)` — with every pointer
valid and no pending error on entry. Pinned by `inkling KV session growth keeps
every layer's rows intact` (red at `l0 h0 p0 d1` on the old code).

### Tools added for this family

- `tools/inkling_layer_ref.py` — independent float64 per-layer reference from the
  artifact, with `--from-trace`, `--head`/`--head-row`, and the selected expert
  ids printed so "the maths disagrees" and "the selection flipped" are
  distinguishable.
- `tools/inkling_full_forward.py` — whole-sequence float64 forward.
- `tools/inkling_transformers_real.py` — builds the RELEASE's own
  `Inkling2TextModel` with a truncated stack, loads real dense weights and
  dequantizes only the experts the router selects, then compares against the
  engine's trace. This is the one reference that does not read my artifact
  interpretation, so it is what located the expert-interleave defect in the
  conversion rather than the kernels.
- `tools/inkling_transformers_crosscheck.py` — synthetic-weight comparison against
  `transformers`, at both fixture and real-ratio geometry.

### Known traps for the next person (also in the converter's header comment)

1. `w13_weight` / `w13_weight_scale` are row-interleaved — de-interleave before
   splitting gate from up. Splitting in half is invisible to every byte-level
   check.
2. The expert codec is mixed (NVFP4 < 40, MXFP8 ≥ 40); only `hf_quant_config.json`
   says so.
3. `attn_res.attn_queries.i` belongs to layer `i+1`; `mlp_queries.i` is by layer.
4. `mtp.safetensors` is a separate 3.4 GB file the index does not list.
5. EOS is `<|content_model_end_sampling|>` (200006), not `<|endoftext|>`
   (199999); there is no BOS and no pad id.
6. `transformers.models.inkling` is a **different** architecture from
   `transformers.models.inkling2`.
7. The index must carry `"model": "Inkling-2-Small"` — `detect_model_type`
   normalizes it into the factory key, and without it
   `lsxbenchmark --model <vault>` / `lsxfabric --localmodel` report "cannot
   autodetect model type" for a complete tree.
