# Inkling-2-Small ingestion — stage 0 triage, plan and full trail

**Status: COMPLETE (2026-11-15).** Every stage of the Model Ingestion Protocol is
landed and measured; the family is in the registry (39 rows) with all four gates
green. This file is the stage-0 record required by `ingesting-a-model` (§ Stage 0)
plus the investigation trail, because this ingestion's red gate turned out to be a
conversion defect that no byte-level check could see (see "Root cause");
the publishable artifact is
`docs/model-reports/2026-11-15-Inkling-2-Small.md`.

Source checkpoint: `/mnt/fast_storage/lsx_model/thinkingmachines/Inkling-2-Small`
(`thinkingmachines/Inkling-2-Small`, modelopt NVFP4 release, 24 shards +
`mtp.safetensors`, 367 GB). Converted output: `<vault>/logestix/` (the
`path_registry` root; never anywhere else).

## 0.1 Classification

**New family.** `model_type: inkling2`, architecture `Inkling2ForConditionalGeneration`.
No existing `models/<org>/<model>.json` row shares the geometry (`ling` is Bailing/KDA+MLA,
`kimi-k3` is KDA+MLA, neither is this). The nearest *structural* neighbour is
Kimi-K3 (AttnRes depth mixer, tiktoken BPE, sized MoE, host expert store) and it
is used as the code skeleton only — no geometry, passport or fixture is borrowed
(`test_model_independence_guard`).

Reference semantics: **`transformers` 5.23.0 `models/inkling2/modeling_inkling2.py`**
(the authoritative port; `transformers` 5.15.1 only ships the earlier `inkling`
family, which is a *different* architecture — see 0.6 #6).

## 0.2 Geometry (read from config.json + safetensors headers)

`text_config` (the whole engine; vision + MTP are ballast):

| field | value | note |
|---|---|---|
| hidden_size | 4608 | |
| num_hidden_layers | 48 | all MoE |
| num_attention_heads / kv | 48 / 8 | GQA group 6 |
| head_dim | 128 | full and sliding identical |
| vocab_size / unpadded | 201024 / 200058 | logits sliced to 200058 |
| moe_intermediate_size | 2304 | arrives as `intermediate_size` (`attribute_map`) |
| n_routed_experts / top-k | 384 / 8 | |
| n_shared_experts | 0 | Inkling2 has no shared experts at all |
| route_scale | 8.0 | `routed_scaling_factor` |
| scoring_func | sigmoid | renormalised over the top-k |
| use_gate_bias | true | `mlp.gate.bias`, selection only, FP32 |
| d_rel / rel_extent | 16 / 1024 | learned relative bias bank |
| sliding_window_size | 1024 | 40 sliding layers |
| qk_l2_clip / rel_l2_clip | 22.627417 (=16√2) / 8.0 | learned per-element scales |
| log_scaling_n_floor / alpha | 128000 / 0.1 | full layers only |
| mlp_gate_softcap / up | 4.0 / 25.0 | softcapped SwiGLU |
| final_logit_softcapping | 30.0 | |
| rms_norm_eps | 1e-6 | |
| sconv_kernel_size | 4 | 4 conv states per layer |
| n_layers_per_attn_res_block | 6 | 8 blocks |
| max_position_embeddings | 1048576 | engine default context is far below this |

**Hybrid pattern.** `local_layer_ids` lists 40 layers ⇒ the remaining 8 —
`{5,11,17,23,29,35,41,47}` — are the full-attention layers. They are exactly the
`(layer+1) % 6 == 0` set, the AttnRes block boundaries are `layer % 6 == 0`
(`{0,6,…,42}`), and the same 8 layers are the MTP targets. The two roles must not
be conflated: boundaries gate the *depth mixer*, the 8 full layers gate
*window/log-scaling/dynamic-conv/3-query mixing*.

**AttnRes tensor census** (this is the item that silently decodes garbage):
- `attn_res.mlp_queries.{0..47}` — 48 vectors `[4608]`, one per layer.
- `attn_res.attn_queries.{0..46}` — 47 entries: `[4608]` on sliding layers and
  `[3,4608]` on the 8 full layers. **Indexed by `layer-1`** (layer 0 has no
  attention mixture; `Inkling2*DecoderLayer.attention_input is None`), so
  `attn_queries.46` is layer 47.
- `attn_res.output_query` — `[4608]`, the `final_input` mixture before `norm`.

## 0.3 Tokenizer

Byte-level BPE in the *tiktoken* dialect: `tiktoken/tokenizer.model` (mergeable
ranks) + `tokenizer.json` with `ignore_merges: true` and the o200k `Split`
pre-tokenizer; `added_tokens` = 60, all `special: true`; base vocab 199998,
special band 199998..200057, padded to 201024. `bos` does not exist
(`bos_token_id: null`), `eos = 200006` (`<|end_message|>`-adjacent id from the
config; the checkpoint's own `eos_token_id`). Kind = `2` (the Kimi tiktoken
class), built with `BuildTokenizerArrowFilesTikToken`.

Chat/tool framing comes from `chat_template.jinja` (stage 4b): roles
`<|message_user|>/<|message_model|>/<|message_system|>/<|message_tool|>`,
content wrappers `<|content_text|>`, `<|content_thinking|>`,
`<|content_invoke_tool_json|>`, terminator `<|end_message|>`, end-of-sampling
`<|content_model_end_sampling|>`, tool declaration
`<|message_system|>tool_declare<|content_xml|>{json}<|end_message|>`, tool call
payload `{"name":…,"args":{…}}`, and a mandatory
`<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>`
line before the first non-system message.

## 0.4 Ballast to strip (stored and counted, never applied)

- `model.visual.*` — 422 tensors / 0.33 B params (ViT + merger).
- `mtp.safetensors` — 5-layer MTP head with a `dflare_norm`/`k_norm_ctx`
  variant of the block (not in `model.safetensors.index.json`).
- `mtp_config` in `config.json`, `mtp` block in `hf_quant_config.json`.
Everything else must be classified; an unknown tensor is a converter error.

## 0.5 Feasibility on this box (RTX 5090 32 GB, 750 GB host, 2.0 TB free)

| plane | params | bytes |
|---|---|---|
| dense (embed 926M + unembed 926M + attention/router/attn_res 6.43B) | 8.28 B | ~4.7 GB at NVFP4 |
| routed experts, layers 0–39 (NVFP4 in the checkpoint) | 490 B | ~262 GB on disk, pass-through |
| routed experts, layers 40–47 (MXFP8 in the checkpoint) | 98 B | ~98 GB on disk |
| KV at 2048 tokens (8 full layers × 8 kv heads × 128 × 2 × FP16) | — | 34 MB |
| conv states (4 per layer) + block-state stack (≤8 × 4608 FP32) | — | < 1 MB |

Active read per decoded token: 8 experts/layer ⇒ ~5.1 GB (NVFP4 layers) +
~2.0 GB (MXFP8 layers) ≈ **7.1 GB/token**, dominated by the host expert store.
That is Kimi-K3 territory (0.47 t/s measured) and is the expected perf band
(0.5–2 t/s), with a TTFT dominated by the first-touch page-in of the routed tree.
`ulimit -l` is already unlimited; the store is `mmap` + LRU (`ExpertStore`), not
a pinned arena (R1/R2 in `ingesting-a-model`).

## 0.6 Semantics verified before trusting output

1. **Expert codec, NVFP4 layers.** `w13 = [E][2I=4608][H/2 bytes]`,
   `w2 = [E][H=4608][I/2 bytes]`; per-16 `F8_E4M3` block scales along the packed
   (input) axis; per-expert `amax[E]` and a scalar
   `outer_scale_denominator = 2688 = 6·448`. **Verified numerically**: for six
   experts of layer 0, `6 · max(block_scale) · (amax_e/2688) == amax_e` to
   1e-9 relative. So `w = q_e2m1 · s_e4m3 · g_e`, `g_e = amax_e/2688` — the same
   E2M1 + per-16 E4M3 + float-global-scale codec `lsx_nvfp4` already implements,
   with the global scale carried **per expert** instead of per tensor.
2. **Expert codec, MXFP8 layers (40–47).** `F8_E4M3` weights, `ue8m0` scale per
   32 values along the input axis (`[E][4608][144]`), no amax. See "open
   decision" below.
3. **Attention scale is `head_dim**-0.5`** in this family (Inkling-1 used
   `1/head_dim`) — the single most likely silent-parity defect.
4. **L2 clips are learned and applied before the scale**: `q·scale` then clip to
   ‖·‖≤22.627 (element-wise learned scale, per head×channel), `k` clipped
   weight-free, `rel` with its own `[heads,16]` scale to 8.0; `v` gets a
   **weight-free** RMSNorm. Order: clip → (full layers) log-scale → cache.
5. **Relative bias** is a gather, not a rotation:
   `bias[h,d] = (r_h · proj[:,d])` for distance `d = qpos-kpos`, zero outside
   `0 ≤ d < rel_extent`; sliding layers use `rel_extent = 1024` (the window),
   full layers `rel_extent = 1024` from the config. No RoPE anywhere in this
   model — Q/K are normalized, not rotated.
6. **Log scaling is full-layer only**: `τ = 1 + 0.1·ln(max((pos+1)/128000, 1))`,
   applied to `q` (after clip) and to the *relative states* (before the bias
   gather), FP32.
7. **Depth mixer** (AttnRes): stack = `[block_states (n_blocks), current]`,
   unweighted RMSNorm per row, then single-head SDPA with a *learned fixed
   query* (scale `H**-0.5`); number of outputs = 1 (sliding/mlp) or 3 (full-layer
   attention → separate q/k/v inputs, each with its own RMSNorm weight).
   Block boundary (`layer%6==0`) appends the *incoming* hidden state to the
   stack and drops the local residual; every other layer keeps
   `residual = block_residual + attn_out`.
8. **Convolutions**: post-attention and post-MLP short convs are
   `(conv(x) + x)/√(k+1) = /√5`; sliding k/v convs use the same `/√5`; the
   full-layer k/v convs are `static[tap,channel] + per-token per-head tabs`
   normalised by `/√(2k+1) = /√9`. `tap 0 = current token` (backwards).
9. **Routing**: `σ(logits) + bias` selects top-8, weights = `σ(logits)[top8]`
   renormalised to 1 then × 8.0; experts accumulate in FP32.
10. **Head**: embed → weighted RMSNorm (`embed_norm`), … , `final_input` mix →
    weighted RMSNorm (`norm`) → unembed FP32 → `tanh(x/30)·30` → slice 200058.
11. **MTP is not applied**; the vision tower is not applied.

**Open decision (resolved 2026-11-14):** the 8 MXFP8 layers are **requantized to
NVFP4 at conversion** (option b). Option (a) — passing the FP8 plane through and
adding a second expert codec to the MoE path — was rejected because (i) the
engine's expert store schema (`ExpertSchema`) is a uniform 9-column NVFP4 record,
(ii) it would double the GEMV dispatch surface for 1/6 of the stack, and (iii)
the residual-plane recipe of stage 2 is available if a parity gate later shows
the 4-bit rounding on those layers flipping an answer. The measurement to
record: relL2 of the requantized layer-40..47 experts against their dequantized
FP8 source (to be taken when the engine can decode them).

## Real-weight status: parity GREEN (2026-11-15)

`test_inkling_moe_selfcheck` runs the engine on the converted 594B tree, frames
`"The capital of France is"` through the checkpoint's own chat template, decodes
and answers. Three cycles, bit-identical (`logits_hash 323d30463a3d2b72`), and
the stream opens

    <|message_model|><|content_thinking|>The user is asking a simple factual
    question: ...

which also retires the "the template adds no generation prompt" hypothesis: the
model writes its own assistant marker. The red gate that stood here until
2026-11-15 was one defect, described below; the seven engine defects listed
after it were real too, but each only moved the output from garbage towards
English.

## Root cause: the fused expert slab is ROW-INTERLEAVED (2026-11-15)

This is the defect that cost the model its knowledge, and it is worth recording in
full because every check in the tree was blind to it.

`model.llm.layers.L.mlp.experts.w13_weight` stores the gate and up projections in
one slab whose rows alternate: `[gate_0, up_0, gate_1, up_1, ...]`. The release's
own conversion de-interleaves that before splitting the two projections —
`Interleave(dim=1)` on both `w13_weight` and `w13_weight_scale` in
`transformers/conversion_mapping.py`'s `"inkling2_text"` entry, whose docstring is
literally "Deinterleaves a tensor along `dim` by splitting in two and
transposing". The converter copied the first `I` rows as the gate plane and the
second `I` as the up plane, so every expert's gate rows met the wrong up rows and
the wrong block scales. 18 432 expert records, all of them.

**Why nothing caught it.** Every property a byte-level check can test still held:
the bytes are the release's, the planes have exactly the right sizes, the
dequantized magnitudes are right (std 0.01473 against 0.0147 for the dense
weights) and the maximum equals the recorded `amax` to the last digit. Worse, it
is *symmetric*: my float64 reference reads the artifact and reproduced it, and so
did `tools/inkling_transformers_real.py` driving the RELEASE's own
`Inkling2TextModel` — that tool feeds the release's code my dequantized planes,
so a wrong plane split passes straight through it. The synthetic `transformers`
cross-check could not see it either: the fixture's experts were packed by my own
converter.

**What broke the symmetry.** The model's own output, measured on real weights:

| measurement | split-half (before) | de-interleaved (after) |
|---|---|---|
| mean NLL over "The capital of France is Paris" | 9.46 → ppl **12 881** | 3.41 → ppl **30.4** |
| top-1 after "The capital of France is" | `" the"` | **`" Paris"`** |
| "France" after "The capital of" | ppl 5.9e5 | ppl 109 |
| SelfCheck first ids | `" 0.9.9.0.0.9.0.0..."` | `<\|message_model\|><\|content_thinking\|>` |

The intermediate confirmation, taken before 366 GB was re-converted, was a
host-side row gather injected into the expert staging path behind
`LSX_INKLING_W13_DEINTERLEAVE`: with it the model answered " Paris", without it
it answered " the". The converter then got the real fix, test-first
(`test_inkling_moe_convert` now asserts gate row `r` == source row `2r` and up
row `r` == source row `2r+1`, for both codecs, weights and scales), and the tree
was re-converted in 10 min.

**New gate.** `inkling real-weight sanity: the model keeps its knowledge` scores
the model's own continuation: one prefill per position over a six-token sentence,
mean NLL < log(100) and the exact top-1 " Paris". Deterministic, no sampling, and
the bound sits 100x above a healthy value so it only fires when a defect costs
the model its knowledge. That is the class this bug belonged to, and the class no
artifact-reading reference can ever see.

### A second defect found by the same gate: KV growth over-read

`InklingSession::EnsureCapacity` copied `kv_rows * LayerStride()` bytes per layer,
but `LayerStride()` includes the capacity, so the copy ran past the old
allocation whenever the cache grew; the runtime rejected it with
`cudaErrorInvalidValue`, which was then picked up by the NEXT request's first
unchecked launch and reported as `[embed] embed row (cuda 1)` — with every
pointer non-null and no pending error on entry. Fixed test-first
(`inkling KV session growth keeps every layer's rows intact`, red at
`l0 h0 p0 d1` on the old code) and the error is now propagated where it happens.

### What was fixed before it (seven real defects, all found by measurement)

1. **BF16 tensors were read as FP16 bit patterns.** `UploadHalf`/`UploadF32`
   used `__ushort_as_half`, which reinterprets the 16 bits as IEEE half. The two
   formats share a width but not an exponent bias (127 vs 15), so BF16 1.0
   (0x3F80) read as FP16 1.875 and every norm weight, L2 clip scale, conv tap,
   attention sink, relative-position bank, router weight and router bias was
   corrupted. The synthetic fixtures could not catch it (they write their BF16
   payload and read it back through the same path), so it took a real-weight
   trace to see. This turned incoherent multibyte garbage into English.
2. **A scratch-arena overlap corrupted the AttnRes block stack.** `o_taps` and
   `o_proj` reserved room for ONE array while the code derives a second pointer
   at `base + size` (`taps_v`, `proj_v`); only the FULL layers use the second
   one, so it wrote ~16 KiB into the block stack on exactly those layers. The
   symptom named it: full and boundary layers disagreed with the reference while
   sliding non-boundary layers matched.
3. **The relative-position states were projected into the conv scratch buffer**,
   so the position bias read a convolution result.
4. **The recurrent conv state was never reset between requests** — a second
   identical request decoded differently.
5. **The post-MLP conv state overlapped the post-attention one** when
   `kv_dim < hidden`, and a static-conv layer passing its taps for both the `w`
   and `dyn_taps` arguments silently ran the lag convention on every sliding
   layer.
6. **`embed`/`unembed` were requantized to NVFP4.** The release keeps both at
   BF16 (`hf_quant_config.json` puts them in the `bfloat16` group even though the
   model is natively NVFP4 elsewhere), and 4-bit rounding added a measured
   4.6-degree direction error to every token vector at the input
   (cos 0.9968 vs the source). Keeping them BF16 is what moved the output from
   garbage to English.
7. **The dense projections were requantized to NVFP4.** `hf_quant_config.json`
   puts ONLY the routed experts in a low-precision codec; every attention
   projection is BF16. Requantizing them costs a measured **9.5% relative L2**,
   which is the difference between a working model and a repetition loop. The
   artifact now carries the release's own precision: BF16 dense (widened to FP16
   on device, which is lossless for these magnitudes) and the routed experts
   byte-verbatim.

### The engine is now VERIFIED against two independent implementations

- **`tools/inkling_transformers_crosscheck.py`** writes a synthetic checkpoint in
  the released naming convention, converts it with the repo's own converter,
  dequantizes the artifact back to float32, loads those exact tensors into
  **`transformers.models.inkling2.Inkling2ForCausalLM`** (the authoritative
  implementation, 5.23) and compares logits and per-layer hidden states. Because
  both sides consume identical numbers, any disagreement is a semantic defect.
  Result: **identical top-5 and per-layer agreement to 1e-3** — twice, once at
  the tiny fixture geometry and once at the REAL model's ratios (GQA group 6,
  top-8 of 16 experts, two AttnRes blocks, 12 layers, 4 positions).
- **`tools/inkling_layer_ref.py`** is an independent float64 reference over the
  same artifact: per-layer replay from the engine's own state, N-position replay,
  MoE-isolated replay and deliberately-wrong variants for bisection.
- **`InklingMoeModel::TraceForward`** dumps the residual stream after every layer
  for a token sequence, plus the logits row.

Measured with those instruments:

| check | result |
|---|---|
| **all 996 pinned tensors vs the source safetensors** | **exact** (relL2 < 1e-6), including the `attn_queries` re-indexing |
| converted routed experts, NVFP4 layers | **byte-exact** (0.000000) |
| converted routed experts, MXFP8 layers 40-47 | 0.094 (the 4-bit requant; the release keeps them at 8 bits) |
| engine vs `transformers`, tiny fixture, final logits | identical top-5, relL2 1.0e-3 |
| engine vs `transformers`, real ratios, final logits | **identical top-5**, relL2 3.1e-3 |
| per-layer replay (real weights), position 0, layers 0-47 | 3e-4 .. 7e-4 |
| full layer 5 and sliding layer 20 across positions 0-3 | 3-6e-4 |
| MoE seeded with the engine's own `mlp_norm` | identical expert ids, coefficients to 1e-8 |
| prompt ids vs `transformers` `apply_chat_template` | identical, all 20 |

### The multi-position path is verified at depth (2026-11-14)

`TraceForward` now also dumps the post-`embed_norm` row and a configurable
layer's input row **for every position**, so any SLIDING NON-BOUNDARY layer can
be replayed at any depth: its mixture needs only `[block-0 state, current
residual]`, both of which the dump supplies, and its output then depends on the
conv rings and the KV cache carried from earlier positions.

Layer 4 over the real 20-token framed prompt, with 20 cached keys and the
sliding window active:

```
layer 4 at pos 19/19 (kv_len=20): relL2=4.562e-04
```

So the conv rings, the KV append, the sliding bound and the relative-bias gather
are all correct at depth, not just in the first few positions.

### The FP8 expert plane is IN — and it exposed a codec bug worth recording

The eight MXFP8 expert layers (40-47) now carry the release's own 8-bit plane:
E4M3 weight bytes verbatim plus each ue8m0/32 exponent duplicated to the neutral
schema's per-16 count. A dedicated FP8 expert GEMV
(`RunInklingFp8ExpertGemm`, one warp per output row, FP32 accumulate) decodes
`e4m3(w) * 2^(ue8m0-127)` directly, so there is no requantization and no FP16
intermediate. Two host-referenced kernel cases cover it (a K=64 fixture and a
K=512 fixture with a non-trivial `x_stride`, i.e. the down-projection shape).

Cost: the artifact grew 321 -> 366 GB (+45 GB) because the plane is one byte per
weight instead of packed nibbles, and the per-token read for those eight layers
doubles.

**The bug it found.** The first version decoded weights with
`gpu::nvfp4::E4M3ScaleToFloat`, which is the *unsigned* block-scale codec — it
masks bit 7 by design. E4M3 **weights** are signed, so every negative weight lost
its sign and a signed dot product became a sum of magnitudes: the layer-41 gate
row came out 43x too large (1926 vs ~45) and the residual stream went non-finite
from layer 42 onward. The kernel test could not see it because its fixture wrote
`& 0x7E` — positive bytes only. Two lessons, both now enforced in the test:

- weight decode must apply the sign bit explicitly
  (`E4m3WeightToFloat`: magnitude from the canonical codec, sign from bit 7);
- a codec fixture MUST contain negative values, or a dropped sign bit is
  invisible. The FP8 fixtures now use the full byte range (minus the two reserved
  NaNs, 0x7F/0xFF).

After the fix all 49 trace rows are finite and the magnitudes are level (layer 41
at 102.3 vs 121.6 under the 4-bit requant).

### What the FIRST FP8 attempt got wrong (kept as a warning)

The first attempt passed a **host array of `mmap` pointers** into device code,
which is undefined behaviour — it happened to look correct for expert 0 and gave
a 2x error elsewhere, and one uniform-exponent bisection is what exposed it. The
fix is the by-value `InklingFp8ExpertPtrs` struct plus an explicit device staging
step. The second attempt then hit the unsigned-codec bug above. Both are the
reason the FP8 path is now covered by two host-referenced kernel cases instead of
one.

### Stage 8 gates on the real tree (2026-11-15)

Measured on the 366 GB artifact at `FastMathMode::kOff`, one target at a time:

| gate | result | evidence |
|---|---|---|
| SelfCheck | **PASS** | 3 cycles bit-identical, `logits_hash 323d30463a3d2b72` |
| FSI | **3/3** | PV formula, three-leg balance chain, strict comparison |
| CODE | **3/3** | unspaced JSON, midpoint assignment, word-bounded hex regex |
| DET | **3/3** | all three classes bit-identical over 3 runs each |
| tool round-trip | **PASS** | `test_inkling_moe_tool_framing`, 49 framing assertions |
| knowledge sanity | **PASS** | mean NLL 3.41 (ppl 30.4), top-1 " Paris" |

The FSI/CODE/DET goldens are **template-derived** (`--template-only` /
`--template-golden`), which each file's `source` field records: the source is
594B with its own NVFP4/MXFP8 stack and needs 500 GB of aggregated VRAM, so it
cannot be executed here. This is the deviation GLM-5.3-Flash already records, not
a silent relaxation — and it is worth naming what it costs: a template golden
proves the converted model produces the canonical answer, not that it produces
the SOURCE's answer. For this family the difference is unusually well covered by
`tools/inkling_transformers_real.py` (the release's own modelling code on real
weights, 3.7e-4 against the engine's trace).

Two harness gaps were found and closed on the way, both test-first:

1. `fsi::ExtractFsiFormula` applied `prefer_last_answer` to the LINE choice only,
   not to the anchor position inside it. A model that reasons inline emits no
   line break between its chain and its answer, so both live on one line and the
   first `anchor=` on it belongs to the chain. Inkling's answer was correct and
   was the last thing in the stream; the extractor returned a reasoning
   paragraph. Fixed and pinned in both directions (opt-in models move, every
   other model's extraction is byte-identical).
2. `ExplainCases` printed expected/extracted/error but not the RAW text, so "the
   extraction latched onto a reasoning line", "the budget cut the answer
   mid-thought" and "the answer is wrong" were indistinguishable. It now prints
   the raw tail.

### Stage 4: throughput (2026-11-15)

Seeded through the harness at the canonical protocol (2048 generated tokens,
Release build, RTX 5090, one target at a time):

| mode | t/s | TTFT | avg power | peak |
|---|---|---|---|---|
| `perf` (kAuto) | **1.47** | 14 485 ms | 95.3 W | 120.9 W |
| `r3` (kOff) | **1.46** | 14 469 ms | 96.7 W | 101.0 W |

The pre-engine estimate in §0.5 predicted "~0.5-2 t/s is the expected band" from
the active read alone (8 of 384 experts × 48 layers ≈ 7 GB per token against a
366 GB artifact on a 32 GB card), and the measurement lands in it. Two
consequences worth recording: **95 W average on a 575 W card** says the SMs spend
most of each layer waiting for the next expert slab, and the two numeric modes
differ by 0.7 %, i.e. there is almost no arithmetic left to speed up.

Remaining headroom, unmeasured on this family: overlap the next layer's expert
staging with this layer's compute (the §R17 P6 pattern), and raise
`LSX_INKLING_MAX_OPEN` so the residency window holds a sequence's working set
instead of evicting between tokens. Neither is a correctness change.

### What remains

Nothing in the lifecycle. Every stage is landed and measured:

| stage | artifact |
|---|---|
| 1-2 convert | `--inkling` in lsxforge, 366 GB tree, `test_inkling_moe_convert` |
| 3 load | `test_inkling_moe_loader`, generated VRAM profile |
| 4b framing | `test_inkling_moe_tool_framing`, 49 assertions |
| 5 kernels | `test_inkling_moe_kernels`, 1588 assertions |
| 8 gates | SelfCheck x3 `323d30463a3d2b72`, FSI/CODE/DET 3/3, knowledge sanity |
| 4 throughput | 1.47/1.46 t/s @ 2048, seeded |
| 9 report | `docs/model-reports/2026-11-15-Inkling-2-Small.md` |

Follow-ups that are NOT lifecycle gaps, recorded so they are not mistaken for
them: the six traps in the model report (interleaved slab, mixed codec, the
`layer-1` query index, the off-index `mtp.safetensors`, the EOS id, and
`inkling` vs `inkling2`), plus the template-derived goldens, which would become
source-derived only on a host that can execute a 594B checkpoint.

## Registry status: LANDED (2026-11-15)

`models/inkling-2-small.json` is the single hand-written registry row; everything
else is generated at configure time and must not be edited:

| generated | covers |
|---|---|
| `model_manifest.h` | `kModels` (39 rows), the vault dir/env, the passport |
| `model_targets.cmake` | `add_test` + timeout/skip code/`gpu` lock/label/env for all nine stages |
| `model_registry.inc` | the `LsxModelFactory` registration and its aliases |
| `test_model_suite_guard.cpp::Suites()` | the suite row, built from the manifests |

Consequences, machine-enforced:

- The six targets that used to be declared in `kPendingSuiteTargets` are claimed
  by the row, so the declaration was **deleted** (the guard fails if a pending
  name outlives its row, and it fails if a claimed target is not registered).
- `tools/gen_model_manifest.py::_NON_MODEL_TESTS` keeps only
  `test_inkling_moe_constants`: that one is the frozen-passport guard, not a
  lifecycle stage, so no stage claims it.
- The hand-written `add_test` lines are gone from `tests/CMakeLists.txt`; what
  remains there is build reality (sources, link libraries, defines), including
  the three `test_parity_gate.cpp` targets.
- `docs/MODELS.md` now has 39 rows and `test_manifest_sync` verifies it against
  the compiled table.

## Plan (TDD, one stage per commit) — all stages landed 2026-11-15

| # | deliverable | test (label) | state |
|---|---|---|---|
| 1 | `inkling_moe_constants.h` passport, naming-guard family token, `models/inkling-2-small.json` | `test_inkling_moe_constants` (fast) | **done, green** (186 assertions) |
| 2 | `inkling_moe_convert.cpp` + `--inkling` in `apps/lsxforge` | `test_inkling_moe_convert` (convert, synthetic) | **done, green** (445 assertions); real tree converted (366 GB, 10 min after the interleave fix) |
| 3 | topology reader + artifact layout (`inkling_moe_load.*`) | `test_inkling_moe_loader` (load, synthetic + real index) | **done, green** (topology half); the weight upload half is stage 5 work |
| 4 | `inkling_moe_kernels.cu` (attn-res mix, L2 clip, rel bias, sink, convs, softcapped SwiGLU, router) | `test_inkling_moe_kernels` (kernels) | **done, green** (1588 assertions, 16 host-reference cases) |
| 5 | weight upload (`inkling_moe_load.cpp`) + `inkling_moe_inference_pipeline.cu` | `test_inkling_moe_model` (model) | **done, green** (439 assertions): the pinned plane, the neutral expert store, the KV session, the decode scratch arena and the full 12-step layer body with the 8-expert staging and 3 batch GEMVs per layer |
| 6 | `inkling_moe_tool_format.cpp` (stage 4b, native `chat_template.jinja`) | `test_inkling_moe_tool_framing` (fast) | **done, green** (49 assertions) |
| 7 | `SelfCheck` (Paris), goldens | `test_inkling_moe_selfcheck`, `*_fsi/code/det_selfcheck` | **done, green**: SelfCheck x3 `323d30463a3d2b72`, FSI/CODE/DET 3/3 (template-derived goldens) |
| 7b | parity instrumentation + independent host references | `tools/inkling_layer_ref.py`, `tools/inkling_full_forward.py`, `tools/inkling_transformers_real.py` + `TraceForward` | **done**: the release's own modelling code on real weights agrees with the engine at 3.7e-4 |
| 8 | perf row | `test_perf_baseline_inkling_2_small` (perf) | **done, seeded**: 1.47 / 1.46 t/s at 2048, TTFT 14.5 s |
| 9 | report | `docs/model-reports/2026-11-15-Inkling-2-Small.md` | **done** and linked from the manifest row |

