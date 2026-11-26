# GigaChat4-37B-A4B ingestion — design (stage 0 triage + plan)

Date: 2026-11-26. Branch: `feat/model-GigaChat4-37B-A4B`.
Source checkpoint: `/mnt/fast_storage/huggingface/GigaChat4-37B-A4B`
(`ai-sage/GigaChat4-37B-A4B`, FP8 release, 3 shards, 36 700 tensors, 39 GB).
Read-only reference (not vendored): `modeling_gigachat4.py`,
`configuration_gigachat4.py`, `chat_template.jinja`, `tokenizer.json`.

## Intent

Ingest **GigaChat4-37B-A4B** end to end through the Model Ingestion Protocol:
manifest → passport → NVFP4 converter → factory → tokenizer bridge → tool
bridge (4b) → CUDA decode kernels (including the new **DSA** top-k selection) →
SelfCheck → FSI/CODE/DET gates → perf row → report. One RTX 5090 (32 GiB,
SM 12.0), 768 GB host RAM, 160 threads; CUDA Toolkit 13.5 headers on a 13.4
driver (minor-version compatibility, as recorded in `AGENTS.md`).

## Stage 0 — what the checkpoint is (measured)

Verified from `config.json`, the safetensors shard headers, `tokenizer.json`,
`chat_template.jinja` and `modeling_gigachat4.py`.

| property | value |
|---|---|
| architectures | `GigaChat4ForCausalLM`, `model_type` `gigachat4` |
| layers | 32 (1 dense FFN layer `0`, 31 MoE layers) + **1 MTP layer (`32`)**, `num_nextn_predict_layers=1` |
| hidden / vocab | 2560 / 128 256 (`tie_word_embeddings=false`, head untied) |
| hybrid pattern | `full_attention_layers = [3,7,11,15,19,23,27,31]` (8 gated MLA + DSA), the other 24 are GatedDeltaNet (`GigaChat4GatedDeltaNet`) |
| MLA | heads 32 = kv_heads 32, qk 128 nope + 64 rope, v_head 128, `q_lora=1536`, `kv_lora=512`, `alpha_q=sqrt(2560/1536)`, `alpha_kv=sqrt(2560/512)`, **attention gate** `gate_proj` [4096,2560] BF16, `o_proj` |
| DSA indexer (8 layers) | `index_n_heads=64`, `index_head_dim=128`, `index_topk=2048`, `wq_b` [8192,1536] FP8 on the **alpha_q-scaled q latent**, `wk` [128,2560] BF16, `k_norm` (plain RMSNorm, 128, BF16), `weights_proj` [64,2560] BF16, **non-interleaved** RoPE on the first 64 lanes, `score = Σ_h w_h·relu(q_h·k)·128^-0.5`, `w` pre-scaled by `64^-0.5`, top-k over the causal-eligible set |
| GDN | k_heads 16, v_heads 32 (repeat 2), hk = hv = 128, conv_k 4, `in_proj_qkvz` [12288,2560] FP8, `in_proj_ba` [64,2560] BF16, `out_proj` [2560,4096] FP8, `A_log`/`dt_bias` [32], per-group qkvz order `[q|k|v|z]`, `linear_gating_type="gated_rmsnorm_sigmoid"` (**not** zero-centred), gate scale 2.0, o-norm weight [128] |
| norms | `norm_type="GatedNorm"` — plain RMSNorm scale `w` (**not** `1+w`), then `· 2.0 · sigmoid(Wdown·silu(Wup·xn))`, gate rank 16, applied to `input_layernorm`, `post_attention_layernorm`, `post_self_attn_layernorm`, `post_feedforward_layernorm`, `model.norm`, `q_a_layernorm`, `kv_a_layernorm` |
| residual | `pre_post`: `h += post_self_attn_norm(attn(input_norm(h)))`, then `h += post_feedforward_norm(mlp(post_attn_norm(h)))` |
| MoE | 192 routed experts, top-8, `moe_intermediate=768`, 1 shared expert (768), dense layer-0 intermediate 8192, router = **FP32 sigmoid + selection-only `e_score_correction_bias`**, renormalised top-k `×2.5`, group top-k degenerate (`n_group=topk_group=1`), SwiGLU clamp `L=10` (`gate≤L`, `up∈[-L,L]`) shared experts are added to the pre-MoE residual |
| rope | theta 1e5, YaRN factor 32 over `original_max_position_embeddings=32768`, beta 32/1, `mscale=mscale_all_dim=1.0` → cos/sin `attention_factor = get_mscale(32,1)/get_mscale(32,1) = 1.0`, and `self.scaling = (0.1·ln32+1)² / sqrt(192) = 1.813259/13.856406 = 0.1308608`; interleaved pairs |
| logits | **`final_logit_softcapping = 30.0`** (tanh in F32) — new for this family |
| tokenizer | byte-level BPE, 128 000 merges + 18 `added_tokens`, pretokenizer byte-identical to GigaChat3.5 (Llama-3 case-run split) → engine `tokenizer_kind = 3`; `<s>`=1, `</s>`=2, `<|role_sep|>`=128000, `<|message_sep|>`=128001, `<think>`/`</think>`=128014/128015, `｜GCML｜`=128013 (`special:false`) |
| tool format | `chatml v5` + GCML (`<｜GCML｜>tool_calls>`, `<｜GCML｜>invoke name="…">`, `<｜GCML｜>parameter name="…" string="true|false">`) — **identical** to the codec already in `gigachat_tool_format.{h,cpp}`; the GC4 template diff is attachments/text-block plumbing + a `reasoning` switch, not the wire format |
| ballast | MTP layer 32 (49 tensors, stored + counted, never applied); `dflash_draft/` (separate 5-layer BF16 drafter dir, outside the index) |

### Quantization split (from the `weight_scale_inv` inventory)

* **FP8 (E4M3 + block-128 F32 `weight_scale_inv`)** → dequant → NVFP4:
  `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`,
  `in_proj_qkvz`, `out_proj`, **`indexer.wq_b`**, and all routed experts
  (31 × 192 × 3 = 17 856 tensors).
* **BF16 → NVFP4 directly** (no scales in the source): `self_attn.gate_proj`,
  `mlp.shared_experts.{gate,up,down}_proj`, dense `mlp.{gate,up,down}_proj`.
* **kept at source precision**: every norm triple (`weight` + gate up/down),
  `mlp.gate.weight` (router, BF16), `e_score_correction_bias` (F32 on device),
  `indexer.wk` / `indexer.weights_proj` / `indexer.k_norm`, `in_proj_ba`,
  `conv1d`, `A_log`, `dt_bias`, `self_attn.norm`, `embed_tokens` (pinned host),
  `lm_head` (BF16 on device).

### Feasibility on this machine

* Routed experts: 31 × 192 × 3 × (2560×768) = 35.1 G params → NVFP4
  ≈ 983 KB packed + 123 KB scales per projection ⇒ **3.30 MB/expert**,
  **19.7 GB** for 5952 experts.
* Dense/pinned: embed + lm_head BF16 1.31 GB, everything else NVFP4 ≈ 0.9 GB.
* Artifact on disk ≈ **21 GB** (601 GB free on `/mnt/fast_storage`).
* Device residency: dense ≈ 2.1 GiB + expert slot cache ≈ 19.7 GiB ⇒ the whole
  expert set fits beside a 2 GiB reserve — the LRU saturates, decode becomes
  compute/read bound, not PCIe bound.
* KV: MLA F32 cache 8 × 32 × 192 × 4 B = **196 KB/token**, indexer keys
  8 × 128 × 2 B = 2 KB/token, GDN state 24 × 32 × 128 × 128 × 4 B = 50 MB
  (constant). At the 4096-token session cap ≈ 0.87 GB.
* `max_position_embeddings` is 1 048 576, but rope tables are built to the
  session capacity (4096) exactly as the sibling does.

### Semantics that silently break (each gets a test)

1. `GatedNorm` is **not** zero-centred: using `1+w` (the 3.5 kernel) scales the
   residual stream by ~2 and decodes garbage.
2. GDN output norm is `gated_rmsnorm_sigmoid` (no `1+w`, sigmoid activation
   `×2.0`); the 3.5 sibling uses the zero-centred variant.
3. `alpha_q` / `alpha_kv` scale the **q latent before** `q_b_proj` and the
   **kv latent before** `kv_b_proj` (and `alpha_q·q_latent` also feeds the
   indexer's `wq_b`).
4. DSA: indexer keys are cached **post-RoPE** per DSA layer; the selection is
   top-`2048` over the causal-eligible set and only then folded into the
   attention mask. At `pos < 2048` the selection is the identity (a test pins
   that equivalence — it is what makes short-context parity attainable).
5. `final_logit_softcapping = 30` (tanh) — must be applied before sampling and
   before `final_logits` is recorded, or every gate compares a different
   quantity than the served model.
6. Router: sigmoid, bias **only for selection**, renormalise over the selected
   weights, then `×2.5`; shared expert added to the pre-MoE residual.
7. RoPE: YaRN tables with `attention_factor = 1.0` (not the `get_mscale`
   value), and `attn_scale = 0.1308608` applied once. Interleaved pairs on the
   64 rope lanes of q/k; **non-interleaved** on the indexer.
8. SwiGLU clamp on both gate and up for routed, shared **and** dense MLPs.

## Scope decisions

* **New family row, same `family: "gigachat"`, `variant: "37b"`.** The
  architecture is *not* the 3.5 sibling's: DSA indexer, non-zero-centred
  norms, different GDN gating, logit softcap. It gets its own passport,
  converter, loader, model class, manifest row.
* **Family kernels are shared and parameterised, not duplicated.** The
  GigaChat3.5 pipeline/attention/norm TUs are extended with an explicit
  variant selector carried in `GcPipelineDims`/`GcMlaDims` (norm weight offset,
  rope theta/factor/attn-scale, DSA on/off, logit softcap, GDN gate offset).
  Rationale: the independence guard's contract is *unique frozen passport +
  no cross-family borrowing*, and the repo already sanctions a sibling sharing
  its pipeline TU (`inkling_moe_varloop`, whose Infer TU is its own and whose
  loop runs in the sibling's pipeline). Duplicating 2 500 lines would trip
  `dedup_scan` and re-introduce the drift class the shared layer exists to
  prevent. The 3.5 row keeps calling the same entry points with its own values
  (offset `1.0`, `factor=8`), and its bit-exact hash is re-proved after the
  refactor.
* **DSA is implemented, not deferred** — it is the model's core attention
  mechanism above 2048 tokens.
* **MTP and `dflash_draft` are cut and counted** (stored-not-applied policy).
* Dense linear matrices → NVFP4; norms, biases, router, indexer wk/wp, GDN
  scalars stay BF16/F32.
* Prefill is token-by-token through the same decode step (established contour).

## Identity

| artifact | value |
|---|---|
| `model_type` | `gigachat4-37b-a4b` |
| class / header | `GigaChat4MoeModel` in `gigachat_moe_37b_model.h` |
| passport | `libs/lsxcommon/src/gigachat_moe_37b_constants.h`, `namespace lsxcommon::gigachat4::frozen` |
| format tag | `logestix-gigachat4-nvfp4` |
| forge flag | `--gigachat4` |
| vault | `/mnt/fast_storage/huggingface/GigaChat4-37B-A4B/logestix` (`LSX_TEST_GIGACHAT4_37B_A4B_ROOT`) |
| tokenizer_kind | 3 |

## Files

New (`gigachat_moe_37b_*` = the variant slot; add `37b` to `SchemeRegex()`):

| file | role |
|---|---|
| `src/gigachat_moe_37b_constants.h` | frozen passport + `static_assert`s |
| `src/gigachat_moe_37b_convert.cpp`, `include/.../gigachat_moe_37b_convert.h` | converter (FP8→NVFP4, BF16→NVFP4, MTP ballast, tokenizer, index) |
| `src/gigachat_moe_37b_load.cpp`, `include/.../gigachat_moe_37b_load.h` | index parse + frozen-geometry gate + device weight graph (adds indexer weights) |
| `src/gigachat_moe_37b_model.cpp`, `include/.../gigachat_moe_37b_model.h` | `IModel`: Load/Infer/Unload/Inspect/SelfCheck/Tokenizer + 4b bridge |
| `src/gigachat_moe_37b_inference_indexer.cu`, `.h` | DSA indexer: q/k projections, k-norm, non-interleaved rope, index cache, scores, top-k select (subsystem `indexer`) |
| `tests/test_gigachat_moe_37b_convert.cpp`, `_loader.cpp`, `_load_fast.cpp`, `_model.cpp`, `_kernels.cpp`, `_indexer.cu`, `_selfcheck.cpp`, `test_gigachat_tool_framing_37b.cpp` | per-stage tests |
| `models/gigachat4-37b-a4b.json` | manifest (single source of truth) |

Modified (shared family layer, additive + parameterised):

* `gigachat_moe_norm_kernels.cu` / `gigachat_moe_kernels.h` — `w_offset`
  parameter (1.0 = zero-centred 3.5, 0.0 = 4.0) and the GDN gate offset.
* `gigachat_moe_inference_attn.cu` / `.h` — optional DSA mask in the softmax
  kernels, `alpha_q/alpha_kv` and rope/attn-scale from `GcMlaDims`, logit
  softcap entry point.
* `gigachat_moe_inference_pipeline.cu` / `.h` — variant knobs (norm offset,
  DSA step, softcap, gdn gate offset), per-layer indexer state.
* `gigachat_moe_device_math.h` — offset-aware norm element.
* `tests/test_code_naming_guard.cpp` — add `37b` (variant) and `indexer`
  (subsystem) tokens, with the acceptance case.
* `apps/lsxforge/src/{main,cli}.{cpp,h}` — `--gigachat4` flag.
* `tools/fsi_capture.py` (+ `code_capture.py`, `det_capture.py` if they gate on
  the model) — capture override.
* `docs/MODELS.md` (generated), `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_TASKS.md`,
  `docs/model-reports/2026-11-26-GigaChat4-37B-A4B.md`.

## TDD matrix (red first, one commit per stage)

| stage | test | what it pins |
|---|---|---|
| 1 convert | `test_gigachat_moe_37b_convert` (label `convert`, TIMEOUT 120, synthetic) | FP8→NVFP4 with scale tiles, BF16→NVFP4, raw pass-through, routed experts (3 proj/expert), MTP stored-and-counted, index schema (format tag, layer types, indexer dims, softcap, deferred dflash2), tokenizer arrows; synthetic geometry only via `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` |
| 2 quant | same file | `base` reconstruction relL2 on a synthetic tensor; global scale ≠ 1 (logits-scale regression class) |
| 3 factory | `test_gigachat_moe_37b_load_fast` (label `fast`) | `LsxModelFactory` resolves `gigachat4-37b-a4b` (+ alias `gigachat4-37b-a4b`), frozen-geometry mismatch fails Load |
| 4 tokenizer | `test_gigachat_moe_37b_tokenizer_parity` (label `fast`) | `<s>`/`</s>`/role/message/think ids from the checkpoint, BPE round-trip on the control prompt |
| 4b tools | `test_gigachat_tool_framing_37b` (label `fast`) | GCML render/parse round-trip on the synthetic `tests/data/` tokenizer; verdict row in `test_model_suite_guard.cpp` |
| 5 kernels | `test_gigachat_moe_37b_kernels` (label `fast` where host-only) | gated norm with `w_offset=0` vs host reference; GDN norm non-zero-centred; router sigmoid+bias+renorm ×2.5; swiglu clamp; softcap |
| 5 DSA | `test_gigachat_moe_37b_indexer.cu` | indexer q/k/rope/weights vs host reference; top-k selection; **at pos < 2048 selection ≡ full causal**, at pos ≥ 2048 mask matches the host reference; determinism |
| 5 decode | `test_gigachat_moe_37b_attn.cu` | MLA decode with/without mask against the host reference (tolerance class) |
| 8 selfcheck | `test_gigachat_moe_37b_selfcheck` (manifest `selfcheck` form, `SKIP_RETURN_CODE 4`) | ≥3 bit-exact cycles, "Paris", streaming + abort |
| 8 parity | `test_gigachat_moe_37b_{fsi,code,det}_selfcheck` (shared `test_parity_gate.cpp`) | 3/3 each against captured goldens |
| 8 perf | `test_perf_baseline_gigachat4` | seeded row at the canonical 2048-token protocol |
| 9 report | — | publish-ready post + evidence |

## Acceptance

* `ctest --preset linux-debug -j 24` green (fast guards, includes
  `test_model_suite_guard`, `test_model_independence_guard`,
  `test_ctest_labels_guard`, `test_code_naming_guard`,
  `test_manifest_sync`, `test_expert_store_guard`).
* `ctest --test-dir build/linux-debug -R "gigachat" -j 1` green (this row and
  the 3.5 sibling it shares TUs with — the 3.5 bit-exact SelfCheck hash must be
  unchanged by the variant refactor).
* Real-weight gates on the Release/AOCC tree: SelfCheck ≥3 bit-exact cycles +
  Paris, FSI/CODE/DET 3/3, tool round-trip, perf row measured through the
  harness (never hand-edited).
* `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .` clean.
* Report in `docs/model-reports/2026-11-26-GigaChat4-37B-A4B.md`; a SKIP or a
  debt stays labelled as such.

## Non-goals / declared debt

* MTP (layer 32) and `dflash_draft` speculative decoding: not applied.
* No 1M-token context: session cap 4096 (rope tables sized to it), as the 3.5
  sibling does.
* Batch > 1 prefill, multi-GPU, NCCL: out of scope.

## Open question for the operator

RSI/FSI/CODE/DET goldens can be captured two ways:

1. **Template-derived (3.5 precedent)** — golden `input_ids` + expected answer
   strings from the prompt set; the gate proves the converted model's answer,
   not source parity. Cheap (minutes), and exactly what the 3.5 row shipped.
2. **Source oracle** — dequantize the FP8 checkpoint to BF16 (74 GB, fits host
   RAM), run `modeling_gigachat4.py` on CPU for a short teacher-forced probe
   (head logits/top-1 on the control prompt + a few tokens), and compare against
   the engine. Orders of magnitude more expensive (hours), but it is the only
   way to catch "answers, but not *the model's* answers" (alpha/rope/softcap
   class errors) at ingestion time.
