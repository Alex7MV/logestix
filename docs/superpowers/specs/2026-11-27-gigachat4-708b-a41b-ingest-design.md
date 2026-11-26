# GigaChat4-708B-A41B ingestion — design (stage 0 triage + plan)

Date: 2026-11-27. Branch: `feat/model-GigaChat4-708B-A41B`.
Source checkpoint: `/mnt/fast_storage/huggingface/GigaChat4-708B-A41B`
(`ai-sage/GigaChat4-708B-A41B`, FP8 inference release, 45 shards, 95 616 tensors,
720 GB). Read-only reference (not vendored): `modeling_gigachat4.py`,
`configuration_gigachat4.py`, `chat_template.jinja`, `tokenizer.json`,
`GigaChat_4.pdf`.

## Intent

Ingest **GigaChat4-708B-A41B** end to end through the Model Ingestion Protocol
(stages 1–9) as the big sibling of the already-shipped
`gigachat4-37b-a4b`: manifest → passport → NVFP4 converter → factory →
tokenizer bridge → tool bridge (4b) → decode path → SelfCheck → FSI/CODE/DET →
perf row → report. One RTX 5090 (32 GB, SM 12.0), 750 GB host RAM, 160 threads,
CUDA 13.5 headers on the 13.4 driver.

**Why this model.** The 37B proved the architecture (hybrid GatedDeltaNet +
DSA-sparse gated MLA + top-8 MoE). The 708B is the same architecture at 19× the
parameter count — the first GigaChat that cannot fit its expert set anywhere but
host RAM on this card, i.e. the first one that exercises the family's expert
cache as the *only* decode path rather than as a backup.

**Anti-goals.** Not a new architecture port (nothing about the model's math is
new — see §"Semantics"); not a throughput project (the card is PCIe-bound on
this expert set by construction); not MTP/speculative decoding (stored, never
applied, same policy as both siblings).

## Stage 0 — what the checkpoint is (measured)

Verified from `config.json`, the 45 safetensors shard headers (dtypes, shapes,
byte counts), `model_safetensors.index.json`, `tokenizer.json`,
`chat_template.jinja`, `generation_config.json`, and the two shipped `.py`
reference modules.

| property | 708B-A41B | 37B-A4B (shipped sibling) |
|---|---|---|
| architectures / model_type | `GigaChat4ForCausalLM` / `gigachat4` | same |
| layers | **64** (dense FFN 0–2, MoE 3–63 = **61** MoE layers) + MTP layer **64** | 32 (1 dense, 31 MoE) + MTP 32 |
| hidden / vocab | **7168** / 128 256 (`tie_word_embeddings=false`) | 2560 / 128 256 |
| hybrid pattern | `full_attention_layers = [3,7,…,63]` = **16** gated MLA + DSA; the other **48** layers are GatedDeltaNet (of which 0–2 carry the dense MLP) | 8 full + 24 GDN |
| MLA | heads **64** = kv_heads **64**, qk 128 nope + 64 rope, v_head 128, `q_lora` 1536, `kv_lora` 512 | heads 32 |
| DSA indexer | `index_n_heads` 64, `index_head_dim` 128, `index_topk` 2048, rope 64 (non-interleaved) — **identical** | identical |
| GDN | k_heads **32**, v_heads **64** (repeat 2), hk = hv = 128, conv_k 4 → qkvz **24 576**, conv **16 384**, ba **128** | 16 / 32 → 12 288 / 8 192 / 64 |
| MoE | **256** routed experts, top-8, `moe_intermediate` **2048**, 1 shared expert (2048), dense intermediate **18 432** | 192 / 768 / 8192 |
| norms | `GatedNorm`, plain scale `w` (**not** `1+w`), rank-16 gate computed from the **scaled** state; GDN output norm `gated_rmsnorm_sigmoid` | identical |
| residual | `pre_post` (add attn, then add MLP, each through its own gated norm) | identical |
| rope / logits | theta 1e5, YaRN factor 32 / orig 32 768 / beta 32-1, cos-sin factor 1.0, `attn_scale` 0.1308608, `final_logit_softcapping` 30 (tanh) | identical (verified by the shared kernel test) |
| router | FP32 sigmoid, selection-only `e_score_correction_bias`, renormalised top-k × 2.5, SwiGLU clamp L=10, shared expert added to the pre-MoE residual | identical |
| tokenizer | byte-level BPE, Llama-3 case-run pre-tokenizer → engine `tokenizer_kind = 3`; `tokenizer.json` **byte-identical** to the 37B's (`diff -q` empty) | — |
| tool format | chatml-v5 + GCML; `chat_template.jinja` **byte-identical** to the 37B's (44 706 B) | — |

`modeling_gigachat4.py`, `configuration_gigachat4.py`, `generation_config.json`,
`tokenizer_config.json` are also byte-identical to the 37B sibling's. **The 708B
is the same code with a bigger geometry.** That is the whole ingestion: no new
semantics, no new kernels, a new frozen passport plus the geometry plumbing.

### Measured payload inventory (why the artifact is ~410 GB)

| class | bytes | note |
|---|---|---|
| routed experts | **687.9 GB** (F8_E4M3 687.73 + F32 scales 0.17) | 61 layers × 256 experts × 3 matrices, 93 696 tensors incl. `weight_scale_inv` |
| GDN projections | 11.37 GB (FP8 11.27) | `in_proj_qkvz`, `out_proj` on 48 layers |
| shared experts | 5.37 GB **BF16** | 61 layers × 3 |
| MLA projections | 3.72 GB (BF16 2.0 = attention gate; FP8 1.72) | 16 layers + MTP |
| dense MLP | 3.17 GB **BF16** | layers 0–2 + MTP |
| embed + lm_head + MTP embed/head | 7.35 GB **BF16** | untied head |
| indexer | 0.26 GB (FP8 0.21 = `wq_b` only) | `wk`, `weights_proj`, `k_norm` are BF16 |
| norms / routers / GDN scalars | ~0.3 GB | BF16 / F32 |
| **total** | **719.7 GB**, 710.3 B params | per-param ≈ 1.013 B ⇒ FP8 body |

### Feasibility on THIS machine

* **Experts → NVFP4**: 687.73 B params ⇒ 343.9 GB packed + 43.0 GB E4M3 scale
  bytes = **386.9 GB**, i.e. **24.8 MB per expert** (3 matrices of 7168×2048).
  15 616 experts (61 × 256).
* **Dense → NVFP4 / kept BF16**: GDN 6.34 GB, MLA 0.97 GB + attention gate 1.88
  GB BF16, indexer `wq_b` 0.12 GB, shared experts 3.02 GB, dense MLP 1.34 GB,
  lm_head 1.84 GB, embed 1.84 GB (host-pinned), norms/scalars 0.35 GB, MTP
  ballast 3.9 GB ⇒ **≈ 410 GB artifact on disk**.
* **Disk**: 861 GB free on `/mnt/fast_storage` ⇒ 451 GB left after conversion.
  The source is never deleted by the converter; conversion streams tensor by
  tensor (peak host growth is one tensor, ≤ 30 MB) — *except* the expert map,
  see "converter memory" below.
* **Host RAM**: pinned expert store 386.9 GB + dense 20 GB + page cache ⇒ fits
  750 GB; `ulimit -l` is already `unlimited` on this host.
* **VRAM (32 GB)**: dense resident ≈ 15.9 GB, KV/state at a 4096-token session
  ≈ 5.8 GB (16 MLA layers × 4096 × 64 heads × 320 lanes × 4 B = 5.4 GB), so
  ≈ **6 GB of expert slots** (≈ 250 of 15 616) after a 2 GiB reserve.
* **Projected decode**: 61 MoE layers × top-8 = **488 expert touches per
  token** = 12.1 GB of H2D with a ~1.6 % slot hit rate ⇒ **≈ 2–3 t/s**
  (12.1 GB ÷ 25–40 GB/s effective PCIe 5.0 ×16). The 432B sibling measured
  11.3 t/s at 8.4 GB/token; the 1T Mistral measured 1.04 t/s on a 120-slot
  cache — 708B sits between them. This is a projection and the perf row is
  measured through the harness, never written by hand.
* **Conversion time**: 720 GB read + 700 GB of H2D dequantize work + 410 GB
  written ⇒ tens of minutes, not hours.

### Converter memory (one deliberate change to the shared core)

`ConvertGigaChat4Model` today accumulates **every** expert's packed payload in
an `absl::flat_hash_map` and writes them at the end. Correct for 19.7 GB
(37B), fatal at 386.9 GB. The shared core flushes each layer's experts as soon
as the layer is complete and frees them, so peak host growth is one layer
(256 × 24.8 MB = 6.4 GB) instead of the whole tree. The written files are
byte-identical; only the order of writes and the peak RSS change.

### Ballast to strip — and count

* MTP layer **64**: 11 tensors (own `embed_tokens`, `eh_proj`, `shared_head.head`,
  dense `mlp.{gate,up,down}_proj`), ~3.9 GB — **stored and counted, never
  applied**, exactly as both siblings.
* `dflash_draft/` (6.1 GB, separate 5-layer BF16 drafter, outside the index) —
  not ingested; recorded in the index `deferred` block.
* `GigaChat_4.pdf`, `README.md` — documentation, untouched.

### Semantics to verify before trusting output

There is **no new semantic** relative to the shipped 37B row: the same
`GcPipelineDims` variant knobs (offset-0 GatedNorm, gate-from-scaled-state,
GDN sigmoid output norm, logit softcap 30, `attn_scale` 0.1308608, DSA on 16
layers) apply, and the shared kernel tests already pin them. The one thing that
is *not* inherited is the geometry: `alpha_q = √(7168/1536) = 2.1602469`,
`alpha_kv = √(7168/512) = 3.7416574` (37B: 1.2909944 / 2.2360680), and every
width (`hidden`, `moe_intermediate`, GDN heads, layer map) changes. A single
constant left at the 37B value silently produces a model that answers but is
not this model, so the gate is: **every 37B constant either equals the 708B
value by derivation, or is re-derived and asserted in the new passport.**

## Approaches considered

**A. Copy the whole family (constant → converter → loader → model → kernels).**
Zero risk to the 37B row, but `gigachat_moe_*_inference_*.cu` is ~2 500 lines of
already-parameterised, already-verified device code and `tools/dedup_scan.py`
exists to catch exactly this. Rejected: it re-introduces the drift class the
shared layer was built to prevent.

**B. (chosen) New passport + new model TU + spec-parameterised family core.**
The architecture TUs stay shared and data-driven, as the 37B ingestion already
established ("One pipeline serves both GigaChat checkpoints. Everything a model
does differently is data here, never a compile-time branch"). The two frozen
touch points that remain hard-coded in the family core — the converter's
geometry gate/index strings and the loader's passport check — become explicit
*expectation* parameters, supplied by each model's own passport. New per-model
files: the passport, the converter entry point, the `IModel` subclass and the
tests. No kernel is duplicated and no `frozen::` value of another model is read
by the new row.

**C. Treat it as a new family** (`gigachat_moe_708b_*` everywhere, incl. its own
pipeline). Rejected: `test_model_independence_guard` forbids cross-family
borrowing but *expects* siblings to share the family layer; a duplicate pipeline
would be 2 500 lines of divergence for zero behaviour.

## Identity

| artifact | value |
|---|---|
| `model_type` / `kType` | `gigachat4-708b-a41b` |
| family / variant | `gigachat` / `708b` |
| class / header | `GigaChat4_708BMoeModel` in `gigachat_moe_708b_model.h` |
| passport | `libs/lsxcommon/src/gigachat_moe_708b_constants.h`, `namespace lsxcommon::gigachat4_708b::frozen` |
| format tag | `logestix-gigachat4-708b-nvfp4` (a distinct tag: a 708B row must never accept a 37B tree even before the geometry check) |
| forge flag | `--gigachat4-708b` |
| vault | `/mnt/fast_storage/huggingface/GigaChat4-708B-A41B/logestix` (`LSX_TEST_GIGACHAT4_708B_A41B_ROOT`) |
| tokenizer_kind | 3 (unchanged) |
| session cap | 4096 (perf protocol needs prompt + 2048 generated) |

## Files

New:

| file | role |
|---|---|
| `src/gigachat_moe_708b_constants.h` | frozen passport (all constants re-derived) + `static_assert`s |
| `include/.../gigachat_moe_708b_convert.h`, `src/gigachat_moe_708b_convert.cpp` | converter entry point: 708 geometry expectations + format tag + model name, delegating to the shared core |
| `include/.../gigachat_moe_708b_model.h`, `src/gigachat_moe_708b_model.cpp` | `IModel`: Load / Infer / Unload / Inspect / SelfCheck / Tokenizer + stage-4b bridge |
| `tests/test_gigachat_moe_708b_convert.cpp` | synthetic converter E2E + 708 defaults + frozen gate |
| `tests/test_gigachat_moe_708b_loader.cpp` | index parse + frozen-geometry gate accepts 708, rejects 37B |
| `tests/test_gigachat_moe_708b_weight_graph.cpp` | synthetic convert → device graph → one expert through the slot LRU |
| `tests/test_gigachat_moe_708b_model.cpp` | Load/Infer/streaming/abort on the synthetic tree |
| `tests/test_gigachat_moe_708b_kernels.cpp` | host reference math with the 708 passport (derived widths, alphas, scale) |
| `tests/test_gigachat_moe_708b_tokenizer_parity.cpp` | real-vault tokenizer ids (SKIP without the vault) + synthetic round-trip |
| `models/gigachat4-708b-a41b.json` | manifest (single source of truth) |
| `tests/fixtures/{fsi,code,det}/gigachat4-708b-a41b.json` | parity goldens (template-derived, 37B precedent) |
| `docs/model-reports/2026-11-27-GigaChat4-708B-A41B.md` | stage-9 report |
| `docs/open-tasks/live/models/gigachat4-708b-a41b.md` | open items |

Modified (additive, behaviour-preserving for the shipped rows):

| file | change |
|---|---|
| `src/gigachat_moe_37b_convert.cpp`, `.h` | extract the body into `ConvertGigaChat4ModelWithSpec(input, output, spec)`; add the per-layer expert flush; the 37B entry point passes the 37B spec (its synthetic E2E test proves the output is unchanged). **Also**: the `VramProfile` classifier lambdas must call the two-argument `classify_gigachat4_tensor(name, top.layers)` — the one-argument overload reads the 37B depth, which would classify every 708B layer ≥ 32 as MTP ballast and undercount the expert planes in the `vram` block that the governor consumes |
| `src/gigachat_moe_37b_load.cpp`, `.h` | `ValidateGigaChat4GeometryAgainst(idx, expect, error)` + a `Gc4FrozenExpect` struct; the existing name keeps the 37B passport |
| `src/gigachat_moe_37b_inference_indexer.{cu,h}`, `src/gigachat_moe_inference_pipeline.cu` | the indexer TU reads two values from the 37B passport (`kLinearONormEps`, `kIndexScoreScale`). Both are geometry-derived and identical here, but a new row must not read another model's passport, so they become fields of `GcDsaIndexerDims` fed from two new `GcPipelineDims` knobs (`index_onorm_eps`, `index_score_scale`), defaulted to today's values. The 37B keeps passing its own |
| `tests/test_code_naming_guard.cpp` | add `708b` to the variant vocabulary with an acceptance case |
| `tests/test_model_suite_guard.cpp` | tool verdict row + suite rows come from the manifest; the verdict row is hand-written (`test_gigachat_tool_framing_37b`, byte-identical template) |
| `apps/lsxforge/src/{main,cli}.{cpp,h}` | `--gigachat4-708b` flag → the 708B converter |
| `tools/fsi_capture.py` (+ `code_capture.py`, `det_capture.py`) | only if they gate on a model allowlist |
| `tests/CMakeLists.txt` | `add_executable` blocks for the new test TUs (the manifest supplies `add_test`) |
| `docs/MODELS.md` (generated), `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md` | catalogue + gate matrix + open items |
| `tests/perf/perf_baselines.json`, `tests/test_perf_baseline.cpp` | perf row seeded **through the harness** |

## TDD matrix (red first, one commit per stage)

| stage | test | what it pins |
|---|---|---|
| 1 convert | `test_gigachat_moe_708b_convert` (label `convert`, TIMEOUT 120, synthetic) | `ParseGigaChat4Config` defaults are the 708 passport (7168/64/256/2048/…); the index carries the 708 format tag + `model` name + `attn_scale`; a 37B-shaped config is rejected by the frozen gate and a synthetic one passes only under `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`; FP8→NVFP4 with scale tiles, BF16→NVFP4, raw pass-through, 3 projections per expert, MTP stored-and-counted, tokenizer arrows |
| 2 quant | same file | `base` reconstruction relL2 on a synthetic tensor; `global_scale != 1` (the hot-logits regression class) |
| 3 factory | `test_gigachat_moe_708b_loader` (label `fast`) | `LsxModelFactory` resolves `gigachat4-708b-a41b`; `ValidateGigaChat4GeometryAgainst` accepts the 708 index and **rejects** the 37B index and vice versa |
| 4 tokenizer | `test_gigachat_moe_708b_tokenizer_parity` (label `fast`) | `<s>`/`</s>`/role/message/think ids from the checkpoint; BPE round-trip on the control prompt; SKIP without the vault |
| 4b tools | reuse `test_gigachat_tool_framing_37b` | template + tokenizer are byte-identical (sha256 at stage 0); verdict row in `test_model_suite_guard.cpp` |
| 5 graph | `test_gigachat_moe_708b_weight_graph` | synthetic convert → `LoadGigaChat4Weights` → one expert through the `ExpertStore` LRU; indexer planes present |
| 5 math | `test_gigachat_moe_708b_kernels` (label `fast`) | the 708 derived widths (40 96/8 192/16 384/24 576/128), `alpha_q 2.1602469`, `alpha_kv 3.7416574`, `attn_scale 0.1308608`, `kSharedIntermediate 2048`; the shared variant math (offset-0 GatedNorm, sigmoid GDN norm, softcap, swiglu clamp) at these widths |
| 6 sandbox | shared kernel tests (no new kernel) | nothing new; the family's `full_attn_softmax`/norm rules are already pinned |
| 8 decode | `test_gigachat_moe_708b_model` | Load → Infer on a synthetic tree, one `Emit` per token through `req.token_callback`, pre-set `abort_flag` honoured, `InferenceResult` filled |
| 8 selfcheck | manifest `selfcheck` form → `test_selfcheck_driver.cpp` | ≥3 bit-exact cycles, `Paris`, streaming + abort |
| 8 parity | `test_gigachat_moe_708b_{fsi,code,det}_selfcheck` (shared `test_parity_gate.cpp`) | 3/3 each against the captured goldens |
| 8 perf | `test_perf_baseline_gigachat4_708b` | row at the canonical 2048-token protocol, measured |
| 9 report | — | publish-ready post + evidence |

## Conversion and real-weight run plan

1. `lsxforge --gigachat4-708b --input /mnt/fast_storage/huggingface/GigaChat4-708B-A41B
   --output /mnt/fast_storage/huggingface/GigaChat4-708B-A41B` (writes
   `.../logestix/`), on the release/AOCC tree, `LSX_EXPERT_PIN` default on.
   Expected ≈410 GB written; 451 GB stays free.
2. Real-weight gates on the **Release/AOCC** tree, one target at a time
   (`RESOURCE_LOCK "gpu"`): selfcheck → parity (fsi/code/det) → perf.
3. `docs/MODELS.md` regenerated after the manifest lands.

## Acceptance

* `ctest --preset linux-debug -j 24` green (fast guards) — includes
  `test_manifest_sync`, `test_model_suite_guard`,
  `test_model_independence_guard`, `test_ctest_labels_guard`,
  `test_code_naming_guard`, `test_expert_store_guard`.
* `ctest --test-dir build/linux-debug -R "gigachat" -j 1` green, **including
  the two shipped siblings** (the shared-core refactor must not move the 3.5 or
  37B bit-exact hashes).
* Real-weight: SelfCheck ≥3 bit-exact cycles + `Paris`, FSI/CODE/DET run (a red
  is recorded as debt, never rewritten), tool round-trip, perf row measured
  through the harness.
* `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .` clean and
  `--strict` clean.
* Stage-9 report written **after** the real-weight gates.

## Non-goals / declared debt

* MTP layer 64 and `dflash_draft`: stored and counted, not applied.
* No 1M-token context: session cap 4096 (KV is 5.4 GB at that cap on a 32 GB
  card); DSA above 2048 keys is exercised by the kernel tests.
* Batch > 1, multi-GPU, NCCL: out of scope for one card.
* Source-side full-model goldens: **infeasible here** (720 GB FP8 → 1.4 TB BF16
  against 750 GB of RAM). Parity is established by (a) template-derived
  FSI/CODE/DET goldens (37B precedent) and (b) the existing per-layer probe
  `tools/gigachat4_layer_probe.py` run on a two-layer prefix, which localises a
  defect to a stage instead of to "the answer is wrong".

## Open questions for the operator

1. **Throughput expectation.** The projection is 2–3 t/s (PCIe-bound; 12.1 GB
   of expert H2D per token). The perf row will be measured, not assumed — is
   that row still wanted at the canonical 2048-token protocol (~12–17 min per
   run), or is a shorter declared row acceptable?
2. **Disk headroom.** 861 GB free; the artifact takes ~410 GB. Proceed on
   `/mnt/fast_storage` (same volume as the source), or stage the output on `/`
   (851 GB free, 1.9 TB NVMe) and move it after?
