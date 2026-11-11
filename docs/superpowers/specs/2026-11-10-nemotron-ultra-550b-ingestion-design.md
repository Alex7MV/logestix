# NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 — Ingestion Design

Date: 2026-11-10
Status: approved design (approach A — separate implementation lineage), pending implementation plan
Branch: `feat/nemotron-ultra-550b-ingest`

## Intent

Bring `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (`NemotronHForCausalLM`,
`model_type = nemotron_h`) into logestix as its **own implementation lineage**
inside the `nemotron` manifest family — its own frozen passport, its own
converter / loader / expert cache / kernels / pipeline / tool codec, sharing
nothing with the 30B Lightning pipeline except model-neutral infrastructure. Then
take it through the full lifecycle on the real weights: convert → load → kernels
→ SelfCheck → FSI/CODE/DET → tool round-trip → perf → report.

Operator decision (2026-11-10): **approach A**, not a shared-pipeline sibling.
The reason recorded here so a later reader does not "fix" it back: the 30B
pipeline is a perf-frontier row with seeded baselines and four gates; touching
it for a 550 B geometry that shares no widths would put those at risk for no
gain, and the operator wants the two lineages independently tunable. The repo's
own precedent for exactly this shape is `granite-switch` (family `granite`, own
namespace `granite_switch`, own TUs `granite_dense_switch_*`, 2 822 lines, no
pipeline sharing).

## Goals

- `kType = "nemotron-ultra-550b-a55b"` runs token-by-token decode from a tree
  converted in place at
  `<vault>/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/logestix/`.
- Every lifecycle stage green-or-declared-debt on the real weights, each with a
  named ctest target and recorded evidence.
- Numerics proven against the reference implementation, not only against
  "it says Paris": a per-layer BF16 torch oracle over a slice of the real
  checkpoint (the full 1.1 TB source cannot execute here — see Evidence tiers).
- The 30B Lightning and 30B NVFP4 lineages are **untouched**: no change to their
  TUs, their converted-layout schema, their goldens or their perf rows.

## Non-goals

- MTP / speculative decoding (`num_nextn_predict_layers = 1`): the reference
  drops `mtp.*` on load (`_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`);
  we cut and count the 1 037 tensors.
- Prefill chunk-scan: the sequential recurrent Mamba-2 form stands (as in the
  sibling), it is mathematically identical to the reference chunk scan.
- Context beyond the canonical 4096 `max_seq`: `max_position_embeddings` is
  262 144, but perf and parity run at the canonical 2048-token protocol.
- The retired stage 7 (token passport). Tool round-trip is in scope.

## Anti-goals

- Editing `nemotron_moe_*` (the 30B lineage) to accommodate Ultra geometry.
- Reaching into the 30B converter's or loader's internals from Ultra.
- Silently degrading precision to make the VRAM bill fit (Levers are measured
  and declared, never reflexive).

## Measured facts (source inventory, 2026-11-10)

225 shards, 1.1 TB BF16, 51 023 tensors, 550 B params / ~57 B active.
`layers_block_type` = 48 `mamba` + 48 `moe` + 12 `attention` (attn at
7,14,23,32,39,48,57,64,73,82,89,98); `mtp_layers_block_type = [attention, moe]`,
1 MTP layer.

| quantity | value | source |
|---|---|---|
| hidden `H` | 8192 | `hidden_size` |
| layers | 108 | `len(layers_block_type)` |
| vocab | 131072 | `vocab_size` |
| Mamba-2 | 256 heads × 64 ⇒ `d_inner` 16384; `conv_dim` 18432; `in_proj` 35072×8192; state 128; 8 groups ⇒ gated-norm group 2048 | config + shard header |
| attention | 64 Q / 2 KV, head_dim 128, **NoPE** | config + reference class |
| MoE | 512 routed, top-22, `I` 5120, shared `SI` 10240, `n_group` 8, `topk_group` 1, `routed_scaling_factor` 5.0, `norm_topk_prob` true, relu² | config + reference |
| latent MoE | `moe_latent_size` 2048 (`fc1_latent_proj` 8192→2048, `fc2_latent_proj` 2048→8192) | shard header |
| router bias | `e_score_correction_bias` F32, selects only | shard header |
| tokenizer | `tokenizer.json` md5 `e7f0d67fb6f0824ce61e6b85b7d0ba57` — byte-identical to the 30B checkpoints | md5 |

Converted-size arithmetic (NVFP4 = 4 bit + per-16 E4M3 scale; norms/router/conv
raw BF16-F32):

| component | bytes |
|---|---|
| embed + head (BF16, untied) | 4.29 GB |
| per mamba layer (in_proj 162 MB + out_proj 75 MB + raw) | 0.237 GB → **11.4 GB** |
| per attention layer (q/o 8192², k/v 256×8192) | 0.078 GB → **0.93 GB** |
| per MoE layer (router 8.4 MB + shared 94 MB + latent 18.9 MB) | 0.122 GB → **5.84 GB** |
| **dense resident graph** | **≈ 22.5 GB** |
| per routed expert (up 5120×2048 + down 2048×5120) | 11.80 MB |
| all 24 576 routed experts | **≈ 290 GB** |
| session: 48 mamba states F32 + KV @2048 | 403 MB + 50 MB |

Each generated token routes to 22×48 = 1 056 experts ⇒ **12.5 GB of expert
traffic per token**, plus ~11.4 GB of resident Mamba projections. The dense graph
(22.5 GB) leaves ~7 GB of expert slots on a 32 GB card (~590 of 24 576, 2.4 %
coverage), so expert reads are effectively all NVMe reads — the same regime as
`mistral-large-4-1t` (16.1 GiB pinned, 17 GB/token, 1.04 t/s) and
`beam-1.0-501b` (1.6 t/s). Expected throughput: **0.5–1.5 t/s**, not
30B-class 24 t/s. This is a structural property of the model on this card, and
the report must say so rather than read as a regression.

## Approaches

**A — separate implementation lineage (CHOSEN).** New frozen passport, new
converter/loader/cache/kernels/pipeline/tool-codec under `nemotron_moe_ultra_*`,
own namespace `lsxcommon::nemotron_ultra::frozen`, own format tag
`logestix-nemotron-ultra-nvfp4`. Cost: ~3 000 lines of a lineage that resembles
the 30B pipeline in outline (measured: `granite-switch` 2 822 lines for the same
shape). Buys: independent tunability, zero risk to the 30B's seeded baselines and
four gates, an index schema that can carry latent/grouped-router fields without
anyone having to keep the 30B reader compatible, and a clean
`test_model_independence_guard` story. Model-neutral layers (`lsx_nvfp4` — the
only NVFP4 path, `lsx_gemv` — the only GEMM path, `lsxcommon::attn`/`dense`,
`lsx_device_math.h`, `lsx_reduce.cuh`, neutral `ExpertStore<TEntry>`,
`ArrowTokenizerDb`, `ModelBase`) are still mandatory and still shared: A is about
the *model lineage*, not about forking neutral infrastructure.

**B — sibling leaf sharing `NemotronModelBase`.** Rejected by the operator. It
would have needed three surgical generalizations in the 30B pipeline (grouped
router, latent MoE, routed scale from the index), each a potential silent change
to a model whose baselines and goldens are already seeded.

**C — a variant of the 30B under the same format tag.** Rejected: the tag
carries a schema, and Ultra's schema grows latent-projection tensors and router
group fields that the 30B tree does not have. One tree cannot be two models
(manifest rule 10), and a tag that means "either shape" makes every loader
strictness check conditional.

## Design

### Files (new, all under the `nemotron` family token)

```
libs/lsxcommon/src/nemotron_moe_ultra_constants.h        frozen passport
libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_convert.h
libs/lsxcommon/src/nemotron_moe_ultra_convert.cpp        converter (own schema)
libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_load.h
libs/lsxcommon/src/nemotron_moe_ultra_load.cpp           index + device weights
libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_expert_cache.h
libs/lsxcommon/src/nemotron_moe_ultra_expert_cache.cpp   NuExpertEntry store + LRU
libs/lsxcommon/src/nemotron_moe_ultra_kernels.h
libs/lsxcommon/src/nemotron_moe_ultra_kernels.cu         device kernels
libs/lsxcommon/src/nemotron_moe_ultra_device_math.h      host reference math
libs/lsxcommon/include/lsxcommon/nemotron_moe_ultra_inference.h
libs/lsxcommon/src/nemotron_moe_ultra_inference.cpp      session alloc/reset
libs/lsxcommon/src/nemotron_moe_ultra_inference_pipeline.cu   per-token layer walk
libs/lsxcommon/src/nemotron_moe_ultra_model.h
libs/lsxcommon/src/nemotron_moe_ultra_model.cpp          leaf: kType, Load, Infer
libs/lsxcommon/include/lsxcommon/nemotron_ultra_tool_format.h
libs/lsxcommon/src/nemotron_ultra_tool_format.cpp        ChatML + XML tool codec
```

Naming: `test_code_naming_guard`'s grammar needs `ultra` added to the optional
variant vocabulary (`nemotron` family token already exists), and
`nemotron_ultra_tool_format` added to `AllowedCommon` as a family-level codec —
the same two edits `granite_switch` needed.

Manifest `models/nemotron-ultra-550b-a55b.json`: family `nemotron`, variant
null, `model_class` `NemotronUltra550BModel`, `model_header`
`nemotron_moe_ultra_model.h`, passport
`libs/lsxcommon/src/nemotron_moe_ultra_constants.h`, namespace
`lsxcommon::nemotron_ultra::frozen`, `format_tag`
`logestix-nemotron-ultra-nvfp4`, forge flag `--nemotron_ultra`, tokenizer kind 9,
vault dir `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`, env
`LSX_TEST_NEMOTRON_ULTRA_550B_A55B_ROOT`.

### Frozen passport (`nemotron_moe_ultra_constants.h`)

All values `static_assert`-cross-checked, one `frozen` namespace, leaf header:

`kType`, `kHidden=8192`, `kLayers=108`, `kVocab=131072`, `kMambaLayers=48`,
`kMoeLayers=48`, `kAttnLayers=12`, `kAttnLayerIdx` (the 12 indices above),
`kRmsEps=1e-5`, `kMambaHeads=256`, `kMambaHeadDim=64`, `kMambaDInner=16384`,
`kSsmStateSize=128`, `kSsmGroups=8`, `kMambaConvKernel=4`,
`kMambaConvDim=18432`, `kMambaInProj=35072`, `kMambaGroupSize=2048`, `kDtMin`,
`kHeads=64`, `kKVHeads=2`, `kHeadDim=128`, `kNumRouted=512`, `kTopK=22`,
`kMoeIntermediate=5120`, `kMoeLatent=2048`,
`kSharedIntermediate=10240`, `kRouterGroups=8`, `kRouterTopkGroup=1`,
`kRoutedScale=5.0f`. The **NoPE** fact is recorded here and restated at the
attention launch site, as in the sibling — it is the one defect this family's
predecessor shipped and it is silent on short prompts.

### Converted layout: `logestix-nemotron-ultra-nvfp4`

Same shape as the sibling, plus the latent tensors and the extra router fields:

- `pinned/global.arrow`: `embed.weight` BF16 [131072,8192], `head.weight`
  BF16 [131072,8192], `norm.weight` BF16 [8192].
- mamba layer: `norm.weight`; `mixer.in_proj.nvfp4` [35072,8192];
  `mixer.out_proj.nvfp4` [8192,16384]; `conv1d.weight` BF16 [18432,4];
  `conv1d.bias` BF16 [18432]; `A_log`/`D`/`dt_bias` F32 [256];
  `mixer.norm.weight` BF16 [16384].
- attention layer: `norm.weight`; `q_proj.nvfp4` [8192,8192];
  `k_proj`/`v_proj.nvfp4` [256,8192]; `o_proj.nvfp4` [8192,8192].
- moe layer: `norm.weight`; `gate.weight` BF16 [512,8192];
  `gate.e_score_correction_bias` F32 [512];
  `fc1_latent_proj.nvfp4` [2048,8192]; `fc2_latent_proj.nvfp4` [8192,2048];
  `shared_experts.up_proj.nvfp4` [10240,8192];
  `shared_experts.down_proj.nvfp4` [8192,10240].
- routed experts `experts/layer_N/exp_E.arrow`: `up_proj.nvfp4` [5120,2048],
  `down_proj.nvfp4` [2048,5120] — **latent-dimension**, per expert.
- index additionally carries `routed_scaling_factor`, `moe_latent_size`,
  `n_group`, `topk_group`, `mtp_tensors` (counted cut), `vram` profile.
- Never quantized: norms, biases, `A_log`/`D`/`dt_bias`, `conv1d.*`,
  `e_score_correction_bias`, router gate, embed/head.

### Kernel / math requirements (the parts the sibling never had)

1. **Grouped top-k router.** `group_scores[g] = Σ top2(scores + bias)` over each
   group's 64 experts; keep the best `topk_group` (=1) group; mask the rest to
   `-inf`; take top-22 (sorted=false in the reference, so selection is a set —
   the engine sorts by weight and breaks ties to the lowest index);
   `norm_topk_prob` ⇒ `w /= w.sum() + 1e-20`; `w *= 5.0`.
   Reference: `NemotronHTopkRouter.forward` (transformers 5.15.1 L728–753).
2. **Latent MoE.** `x → fc1_latent (8192→2048) → experts at 2048 (up 2048→5120,
   relu², down 5120→2048) → weighted sum in latent space → fc2_latent
   (2048→8192) → + shared_experts(x)` (shared expert runs at **H**, not latent).
   Reference: `NemotronHMoE.forward` L699–712.
3. **Mamba-2 at 256 heads / d_inner 16384 / gated-norm group 2048** — same
   recurrence as the sibling (softplus(dt+dt_bias), clamp to `kDtMin`, dA =
   exp(dt·A), state update, `+x·D`, silu-gated RMSNorm, `out_proj`), different
   widths. 8 state groups, `conv_dim` 18432.
4. **NoPE full attention**: q/k carry no rotation; GQA 32:1 over 2 KV heads;
   softmax in F32; `o_proj`. Reusing the sibling's kernel shape verbatim is
   fine; the passport constants differ.
5. **relu² MLP** for routed and shared experts (non-gated: up/down only).

### Converter

Own classifier and own writer. Three requirements beyond the sibling's:

- **Per-layer expert streaming.** The sibling accumulates every routed expert of
  the model in one map before writing; at 24 576 experts × 11.8 MB that is
  290 GB of records and would exhaust the host. Stream: layer → shards → that
  layer's 512 experts → write → free. Peak host cost is one layer (~6 GB of BF16
  source read, transient). Shards are opened per layer, so 225 shards × 48 layers
  of header parsing must be cheap — the index (`model.safetensors.index.json`)
  gives a direct `tensor name → shard` map, so no probe is needed.
- **Strictness**: `mtp.*` is the only cut (counted, 1 037 tensors); every other
  unknown name is an error naming the tensor. `fc1/fc2_latent_proj` are required
  when `moe_latent_size > 0` and forbidden otherwise.
- **Resumability**: conversion writes per-layer files, so a killed 275 GB run
  resumes at the next layer instead of from zero. A `--layers a:b` window exists
  for development (needed by the slice fixtures anyway).

### Evidence tiers (honest provenance)

The 1.1 TB BF16 source cannot be executed here: assembling
`NemotronHForCausalLM` needs ~1.1 TB of RAM, and no host in this tree has it (the
sibling's `tools/nemotron_nvfp4_oracle.py` recipe does not scale).

1. **Slice oracle (primary, new).** `tools/nemotron_ultra_oracle.py
   --layers 0,1,7` builds a real `NemotronHForCausalLM` whose block list is the
   selected real layers (mamba, moe, full-attention) plus the real
   embedding/norm_f/head, loads only those tensors (~4 GB BF16), and records
   hidden states for a fixed prompt at positions 0..N. The engine's per-layer
   dump is compared with it: `relL2 < 2e-2` and `cos > 0.99` per layer, the
   thresholds the sibling's real-weight run established (4-bit noise dominates).
   This is what actually proves the grouped router selection, the latent path,
   the 256-head Mamba widths and NoPE attention against the reference.
2. **Behavioural gates on the full model**: SelfCheck Paris ×3 bit-exact, tool
   round-trip, FSI/CODE/DET.
3. **Coherence + determinism**: three byte-identical runs; a ~200-token
   free-form generation that is readable. (The sibling's RoPE defect produced
   HTML-tag soup here, so this is a real detector.)

FSI/CODE/DET goldens are **template-derived** (the source cannot run) and are
labelled so in the report and in `docs/SELFCHECK_MATRIX.md` — never presented as
source parity. If the operator prefers, the alternative is declaring them debt;
template-derived-with-provenance is the recorded choice.

### VRAM plan and levers

Dense 22.5 GB + 403 MB mamba state + 50 MB KV + ~2 GiB reserve ≈ 24.9 GB of
32.6 GB ⇒ ~7 GB of expert slots (floor = `topk` = 22 slots ≈ 260 MB, ceiling from
the governor's `expert_slots_pct`; `LSX_NEMOTRON_ULTRA_EXPERT_CACHE_SLOTS`
overrides). Levers, documented and used only if measurement demands: embed/head
to NVFP4 (−3.7 GB, changes the logits). Not planned.

### Guards and registries

The manifest row generates the guard registries (`model_manifest.h`,
`model_targets.cmake`, `model_registry.inc`). Hand edits, all in the same
commits: `tests/CMakeLists.txt` wiring for the new targets,
`test_code_naming_guard` (variant token `ultra`, `AllowedCommon`
`nemotron_ultra_tool_format`), `test_model_suite_guard` (the Ultra suite table),
`test_ctest_labels_guard` (labels/locks/SKIP 4 come from the manifest overrides),
`docs/SELFCHECK_MATRIX.md` row, `docs/MODELS.md` regeneration.
`test_model_independence_guard` picks up the passport, namespace and lifecycle
TUs from the manifest — the new TUs must satisfy all seven lifecycle markers
(residency lock, streaming `Emit`, `Aborted()` poll, `TeardownContext`,
governor lease, output text, resident bytes).

### TDD plan (red → green per stage, one commit per stage)

| # | stage | target | asserts |
|---|---|---|---|
| 1 | passport | `test_nemotron_moe_ultra_constants` (fast) | every `static_assert`; the real `layers_block_type` decodes to 48/48/12 with the 12 attention indices; latent/group invariants |
| 2 | converter | `test_nemotron_moe_ultra_convert` (convert, ≤120 s) | synthetic Ultra checkpoint: index tag + new fields, latent tensors present/absent by config, NVFP4 scale geometry, 512 experts exported per layer, MTP counted, strict unknown-name rejection, layer-window resume |
| 3 | load | `test_nemotron_moe_ultra_loader` (fast) | frozen-geometry gate rejects a 30B-shaped tree and every mis-shaped tensor; latent tensors mandatory when `moe_latent_size>0`; synthetic tree always, real tree when present (SKIP 4) |
| 4 | math | `test_nemotron_moe_ultra_math` (fast) | host reference transcribed from `modeling_nemotron_h.py`: group scores, top-22 selection incl. tie cases, latent path, softplus/clamp, gated RMSNorm at group 2048 |
| 5 | expert cache | `test_nemotron_moe_ultra_expert_cache` (fast) | slot planning at 11.80 MB/expert, LRU eviction order, mmap views, no owning copies (R1/R2) |
| 6 | kernels | `test_nemotron_moe_ultra_kernels` (inference) | device vs host parity, bit-exact under `-fmad=false`: grouped router, latent MoE end-to-end, mamba conv/ssu/gated-norm at 16384, NoPE attn decode, relu² |
| 7 | model | `test_nemotron_moe_ultra_model` (fast) | factory resolution, `Load`, session sizes, expert-cache geometry 48×512, teardown |
| 8 | numerics | `test_nemotron_moe_ultra_layers` (inference) | slice oracle per-layer `relL2`/cos at pos 0..N (SKIP 4 when tree or fixture absent) |
| 9 | selfcheck | `test_nemotron_moe_ultra_selfcheck` (shared driver, SKIP 4) | Paris, ≥3 bit-exact cycles, tool round-trip |
| 10 | parity | `test_nemotron_moe_ultra_{fsi,code,det}_selfcheck` (shared drivers) | template-derived goldens; DET 3 runs byte-identical |
| 11 | tokenizer | `test_nemotron_moe_ultra_tokenizer_parity` (second target over the existing binary) | 22/22 against `tests/fixtures/nemotron/` — the checkpoint ships the byte-identical tokenizer |
| 12 | perf | `test_perf_baseline_nemotron` + new case (perf) | canonical 2048-token protocol, two runs agreeing <1 %, row in `tests/perf/perf_baselines.json` |
| 13 | report | `docs/model-reports/2026-11-DD-NVIDIA-Nemotron-3-Ultra-550B-A55B.md` | stage verdicts, measured bottleneck, golden provenance |

### Execution order

Convert (≈275 GB written, 1.1 TB read; expect 1–2 h at the measured 4.2 GB/s
write / 15 GB/s read) → loader on the real tree → kernels → slice oracle →
SelfCheck → parity → perf. One GPU workload at a time (`RESOURCE_LOCK gpu`), no
second `ctest` in flight during perf. The conversion runs in the background while
the synthetic-path stages are developed.

## What is left to the builder

- Whether the slice oracle needs a late attention layer (98) whose KV spans the
  prompt — decide by whether layer 7's cosine holds.
- Exact expert-slot count: measured, recorded in the report.
- Whether `Prewarm` should be skipped for a ~590-slot cache (it costs one full
  staging pass of 590 experts ≈ 7 GB; measure against TTFT).
- Whether the tool codec can be a near-copy of the sibling's own-family codec or
  needs its own framing — decided by the real-weight round-trip.

## Risks

- **Duplication cost** (~3 000 lines) is the price of A and is accepted; the
  drift risk between the two lineages is real, and the mitigation is that Ultra's
  numerics are pinned by its own slice oracle and gates, not by "same as 30B".
- **Host RAM in the converter** — per-layer streaming; if a shard layout defeats
  it, the fallback is an explicit two-pass (index → per-layer tensor list →
  stream) rather than caching records.
- **0.5–1.5 t/s**: FSI/CODE/DET and the 2048-token perf rows take minutes each.
  Budget the run; do not parallelize GPU work.
- **Grouped router is the highest-risk math** in this model; stage 4/8 exist for
  exactly that, and selection (not just weights) is pinned against a host
  reference.
- **24.9 GB of a 32.6 GB card** leaves little headroom: if slots cannot fit
  alongside a workable cache, the declared lever (head/embed NVFP4) changes every
  baseline, so it is an operator decision, not a silent adaptation.
