# Hy4-flash ingestion — design

Date: 2026-11-23. Branch: `feat/hy4-flash-ingestion`.
Source checkpoint: `/mnt/fast_storage/lsx_model/tencent/Hy4-flash` (tencent/Hy4-flash,
revision as shipped 2026-11-23). Operator-approved scope decisions are marked
**[approved]**.

## Intent

Ingest **Hy4-flash** — a 220B-parameter multimodal MoE (155.5B text backbone +
63.5B Engram tables + 0.43B ViT + 0.85B DFly2 drafter) — into logestix as a
**text-backbone** model, and carry it through every lifecycle stage: converter,
NVFP4 quantization, factory registration, tokenizer bridge, tool-calling bridge,
CUDA decode kernels, SelfCheck ("Paris"), FSI/CODE/DET parity gates, perf row,
model report. Hardware is one RTX 5090 (32 GiB, SM 12.0), 768 GB host RAM,
160 hardware threads — the regime the DeepSeek-V4-Flash/V4.1 contours already
run in (streamed routed experts, bounded host footprint).

## What the checkpoint is (stage-0 findings, measured)

Verified by reading `config.json`, `tokenizer.json`, `chat_template.jinja` and
the safetensors shard headers (1964 tensors, 440.5 GB of payload).

| property | value |
|---|---|
| architectures | `HYV4FlashForConditionalGeneration` / text `HYV4FlashForCausalLM`, `model_type` `hy_v4_flash` |
| backbone | 42 layers: 1 dense FFN (layer 0) + 41 MoE |
| hidden / vocab | 3072 / 120832 |
| residual | **iHC** identity Hyper-Connections, `hc_mult=4` streams, `hc_eps=1e-6`, `hc_magnitude=2.0`, **no sinkhorn, no comb matrix** |
| attention | MLA: `q_lora=1024`, `kv_lora=512`, `qk_nope=192`, `qk_rope=64`, `v_head=256`; **gated MLA** (`linear_gate` → `sigmoid`), **learnable per-head sink** |
| hybrid pattern | `layer_types`: DSA at layers `0,1` and `l ≡ 1 (mod 4)` → 12 layers (**32 heads**, θ=1e7); SWA everywhere else → 30 layers (**64 heads**, window 512, θ=1e4) |
| sparse indexer | `compressed_indexer` on the 12 DSA layers: ratio 4, gated pooling, `group_topk=512`, `index_topk=2048`, `skip_fine_select=true` |
| MoE | **LatentMoE**: `routed_expert_down_proj` 3072→1536, experts 1536→1536 (`gate_up` [512,3072,1536], `down` [512,1536,1536]), `routed_expert_up_proj` 1536→3072; one shared expert (3072→1536→3072); router = fp32 sigmoid + selection-only bias, top-16 of 512, renormalized, ×3.998; SwiGLU clamp ±10 |
| dense FFN | layer 0, intermediate 13824 |
| sandwich norms | `post_proj_norm` (all 42 layers), `post_fc2_norm` (dense layer 0), `combine_output_norm` (41 MoE layers) |
| rms eps | **1e-5** (not 1e-20 as in DeepSeek-V4.1, not 1e-6 as in `hy_v4`) |
| positions | interleaved MRoPE, sections `[12,10,10]`; for text-only input all three sections share one position, so it degenerates to plain rotate-half RoPE on the 64 rope dims |
| lm_head | **fp32** (`enable_lm_head_fp32`), untied |
| tokenizer | byte-level BPE, 120000 merges + 832 `added_tokens`, custom pre-tokenizer (isolated 1–3 digit numbers, CJK splits), markers `<｜hy_*:opensource｜>` |
| tool format | native Hunyuan: `<tool_calls:opensource>` / `<tool_call:opensource>{name}` + `<arg_key:opensource>`/`<arg_value:opensource>`, thinking via `<think:opensource>`, `reasoning_effort` `no_think`/`high` |
| ballast | ViT (27 layers, `vit.*`, perceive projector), DFly2 drafter (`model.dfly_layers.*`, confidence head, mtp fusion), Engram (4 layers: 1, 11, 21, 31 — 126.7 GB of tables) |

### Feasibility on this machine

* Pinned (non-expert, non-Engram) weights ≈ 13.7 GB BF16 → ≈ 5 GB NVFP4.
* Routed experts 297 GB BF16 → 21 000 records (41 × 512) ≈ **84 GB NVFP4**,
  4.0 MB per expert; streamed through `ExpertStore` + device slot cache.
* Engram cut → artifact ≈ 90 GB (724 GB free on `/mnt/fast_storage`).
* KV: MLA latent 576 × 2 B × 42 layers = **48 KB/token**; 8192 tokens = 400 MB.
* Decode read volume ≈ 656 experts × 4 MB ≈ 2.6 GB/token → PCIe-bound.

## Scope decisions

* **Engram is cut** **[approved]**, counted in the index `deferred` block and
  declared debt (DeepSeek-V4.1 precedent: the same mechanism, ~98 GiB, cut with
  a counted record). Its hash semantics exist in no published implementation —
  `transformers` carries `hy_v4` but not `hy_v4_flash` — so applying it would be
  a guess dressed as fidelity.
* **ViT and DFly2 are cut** and counted (stored-not-applied, repo policy).
* **IndexPool (compressed sparse indexer) is implemented** **[approved]**, not
  deferred. Sections below state the reconstruction and how it is measured.
* Prefill is token-by-token through the decode path (established contour;
  DeepSeek-V4.1 does the same).
* Dense linear matrices → NVFP4; norms, biases, router gate (fp32), sinks, iHC
  parameters (fp32), indexer scalars (fp32) keep source precision.

## Identity

| artifact | value |
|---|---|
| `kType` / factory key | `hy4-flash` |
| vault dir / alias | `Hy4-flash` → `hy4-flash` |
| manifest `family` / `variant` | `hy` / `flash` |
| class | `Hy4FlashModel` (`final : public ModelBase`) |
| files | `hy_moe_flash_*` (family `hy`, type `moe`, variant `flash` — the naming guard already accepts this without a scheme change) |
| namespace | `lsxcommon::hy4::frozen` |
| index format | `logestix-hy4-nvfp4` |
| forge flag | `--hy4_flash` |

The three already-registered `hy` models (`hy-memory-30b-a3b`,
`hy-simulmt-30b-a3b`) share nothing but the family stem: different geometry,
different architecture, own passport, own namespace, own class.

## Frozen geometry (passport)

`libs/lsxcommon/src/hy_moe_flash_constants.h` locks: `kHidden=3072`,
`kLayers=42`, `kVocab=120832`, `kDenseIntermediate=13824`,
`kMoeIntermediate=1536`, `kLatent=1536`, `kNumRouted=512`, `kTopK=16`,
`kSharedIntermediate=1536`, `kQLora=1024`, `kKvLora=512`, `kQkNope=192`,
`kQkRope=64`, `kVHead=256`, `kHeadsDsa=32`, `kHeadsSwa=64`, `kSwaWindow=512`,
`kHcMult=4`, `kHcRows=8`, indexer `kIndexNHeads=16`, `kIndexHeadDim=128`,
`kPoolRatio=4`, `kGroupTopk=512`, `kIndexTopk=2048`, `kRmsEps=1e-5f`,
`kHcEps=1e-6f`, `kHcMagnitude=2.0f`, `kRouteScale=3.998f`,
`kSwigluLimit=10.0f`, `kRopeThetaDsa=1e7f`, `kRopeThetaSwa=1e4f`.
Layer tables: `IsDsa(l)` (`l==0 || l==1 || l%4==1`), `IsSwa(l)`, `Heads(l)`,
`IsEngramCut(l)` (`l ∈ {1,11,21,31}`), each pinned by a `static_assert` against
the config's own lists.

## Semantics that silently break (each gets a test)

1. **iHC (independent form).** Streams are `[4][hidden]`, seeded from the
   embedding by broadcast. Per sublayer:
   `mix_r = (fn_r · flat) * rsqrt(mean(flat²)+eps)` for the 8 rows,
   `pre_j = sigmoid(mix_j*scale[0] + base_j) + eps`,
   `post_j = 2.0*sigmoid(mix_{4+j}*scale[1] + base_{4+j}) + eps`,
   `x = Σ_j pre_j · stream_j`; then `streams = post · y + streams`.
   At the head: `hc_head` collapses with its own `hc_fn/base/scale` (4 rows), then
   the shared final RMSNorm. All of it in **fp32**.
2. **Gated MLA.** `attn_out *= sigmoid(linear_gate(x))` elementwise over
   `heads*v_head`, before `o_proj`.
3. **Learnable sink in the softmax.** Per head, logits are augmented with one
   sink scalar, softmax runs over `[keys, sink]`, the sink column is dropped.
4. **Sandwich norms.** `post_proj_norm(attn_out)` before the residual add;
   `post_fc2_norm(mlp_out)` (dense) / `combine_output_norm(moe_out)` (MoE)
   before the residual add. Normalization of the *sublayer output*, not of the
   residual.
5. **LatentMoE order.** `lat = routed_expert_down_proj(x)`; per selected expert
   `h = swiglu_clamp(lat @ gate_up_e)`; weighted sum in fp32;
   `routed = routed_expert_up_proj(Σ)`; `out = combine_output_norm(routed) +
   shared_expert(x)` — the shared expert reads the full-width input, not the
   latent.
6. **Router.** fp32 `sigmoid`, `+ e_score_correction_bias` **for selection
   only**, top-16 (group mask is a no-op at `n_group=1`), renormalize, ×3.998.
7. **RoPE.** θ = 1e7 on the DSA layers, 1e4 on the SWA layers; rotate-half over
   the 64 rope dims only, applied to q and to the 64-dim k_rope of the latent.
8. **Attention scale** `256^-0.5 = 0.0625` (no YaRN mscale: `rope_type` is
   `default`).
9. **lm_head in fp32** with an untied embedding table.
10. **SWA window** = 511 preceding + current (512 total).

## IndexPool reconstruction (documented as a reconstruction)

The checkpoint publishes the tensors and the hyperparameters, not the code:

```
per token:   k_t   = token_key_norm(token_key_projection · x)              [128]
per group g: u_i   = k_{4g+i} + intra_group_ape[i]                         [4,128]
             v_i   = key_value_projection · u_i
             s_i   = sigmoid(key_gate_projection · u_i)
             p_g   = key_norm(Σ_i s_i ⊙ v_i)                               [128]
per query:   q_h   = query_projection · q_resid                            [16,128]
             w_h   = head_weight_projection · x                            [16]
             score_g = Σ_h w_h · (16^-0.5) · (128^-0.5) · relu(q_h · p_g)
select:      top-512 groups (causal, fully-complete groups only) → ≤2048 tokens
             ∪ the current group's already-decoded tokens
```

`skip_fine_select=true` means the pooled group score *is* the selection: no
token-level refinement pass. This reconstruction is validated two ways, both
committed: (a) unit tests of the pooling/selection math against a host reference
with synthetic weights, and (b) a **selection-recall measurement** at 4096 tokens
(the pooled selector against an exact token-level scorer on the same weights),
recorded in the model report. If the recall is poor, the fallback is dense causal
attention above the window, which is recorded as debt rather than silently
shipped.

## Artifact layout

`<vault>/Hy4-flash/logestix/`
* `model.index.json` — `format`, `model` (card name), geometry,
  `total_pinned_tensors`, `total_experts`, per-layer `{kind, heads, is_engram}`,
  `vram` block, `deferred: {engram, vision, dfly}`.
* `pinned/global.arrow` — `embed_tokens`, `lm_head`, final `norm`, `hc_head`.
* `pinned/layer_XXX.arrow` — attention (`q_a`, `q_a_norm`, `q_b`,
  `kv_a_proj_with_mqa`, `kv_a_norm`, `kv_b`, `o`, `linear_gate`, `sink`,
  `post_proj_norm`), iHC (`hc_attn_*`, `hc_mlp_*`), FFN (dense triple +
  `post_fc2_norm`, or router `gate`+`bias`, shared expert triple,
  `routed_down`/`routed_up` latent pair, `combine_output_norm`), and for DSA
  layers the eight indexer tensors. Dense matrices go into `dense_nvfp4/`
  records; norms/biases/scalars stay raw BF16/F32.
* `experts/layer_XXX/expert_NNNNN.arrow` — one record per expert
  (`gate_up` NVFP4 + `down` NVFP4 + global scales).

Host side follows R1/R2: `ExpertStore<Hy4ExpertEntry>` alias, mmap views, no
owning per-family container.

## Files

| file | responsibility |
|---|---|
| `libs/lsxcommon/src/hy_moe_flash_constants.h` | frozen passport + layer tables |
| `libs/lsxcommon/include/lsxcommon/hy_moe_flash_convert.h`, `src/hy_moe_flash_convert.cpp` | config parse, tensor classification, NVFP4 quantization, artifact write |
| `libs/lsxcommon/include/lsxcommon/hy_moe_flash_load.h`, `src/hy_moe_flash_load.cpp`, `src/hy_moe_flash_load.cu` | topology/geometry gates, pinned upload, expert store |
| `libs/lsxcommon/include/lsxcommon/hy_moe_flash_device_math.h` | device constants + inline helpers (iHC, clamp, rope tables) |
| `libs/lsxcommon/include/lsxcommon/hy_moe_flash_kernels.h`, `src/hy_moe_flash_kernels.cu` | iHC mix/pre/post, router, sandwich norms, latent projections, IndexPool pooling/selection, sink attention |
| `libs/lsxcommon/include/lsxcommon/hy_moe_flash_inference_pipeline.h`, `src/hy_moe_flash_inference_pipeline.cu` | the one-token decode step |
| `libs/lsxcommon/src/hy_moe_flash_model.{h,cpp}` | `IModel` (Load/Infer/SelfCheck/streaming/abort/tool bridge) |
| `libs/lsxcommon/include/lsxcommon/hy_moe_flash_tool_format.h`, `src/hy_moe_flash_tool_format.cpp` | Hunyuan-native conversation rendering + tool-call parsing |
| `apps/lsxforge/src/main.cpp` | `--hy4_flash` |
| `models/hy4-flash.json` | the manifest (single handwritten registry) |
| `tests/test_hy_moe_flash_{fixture.h,convert.cpp,loader.cpp,model.cpp,kernels.cu,tool_framing.cpp}` | stage tests |

## Acceptance

* `ctest --preset linux-debug` (fast guards) green.
* `ctest --test-dir build/linux-debug -R "hy4-flash" -j 1` green — every stage of
  the row.
* Real-weight SelfCheck ≥3 bit-exact cycles, top-1 `Paris`.
* FSI/CODE/DET goldens captured from the source and the gates **run**.
* Perf row seeded through the harness at the canonical 2048-token protocol.
* `docs/MODELS.md` regenerated, `SELFCHECK_MATRIX.md` row, `OPEN_TASKS.md` debt
  (Engram, ViT/DFly2, IndexPool reconstruction verdict), stage-9 report.

## Non-goals

* Vision input, Engram lookups, speculative decoding (DFly2).
* Multi-GPU, batched serving, >2048-token *dense* DSA (the indexer covers the
  sparse case).
