# GLM-5.3-Flash Ingestion Design

Date: 2026-09-04
Status: Approved
Goal: Adapt GLM-5.3-Flash per the Model Ingestion Protocol (8 stages) with the
highest possible generation speed on the current hardware (RTX 5090 32GB + EPYC 9V74),
implemented via TDD.

## Invariant #1: Do not touch GLM-5.2

- All new files — `glm_moe_flash_*` only. The `glm_moe_*` files (GLM-5.2)
  are not edited at all.
- Reuse of existing primitives (`RunQwenNvfp4Gemm`, NVFP4 decode tables
  `shared_experts.cu`, hc GatedResidual qwen4_exp, expert cache) — by reference only.
  If a modified variant is needed — a new function/overload, not an edit.
- Legitimate touches of other files (mechanical, without changing semantics):
  - `libs/lsxcommon/src/model_registry.cpp` — one `Register<...>()` line
  - `tests/test_code_naming_guard.cpp` — allowlist for the `flash` variant
  - `libs/lsxcommon/CMakeLists.txt`, root `tests/CMakeLists.txt` — new targets
- Regression shield: 12 `test_glm_*` executables with bit-exact `final_logits`
  in CI will catch any accidental GLM-5.2 touch.

## Frozen geometry (compile-time, static_assert)

Source of truth: `/mnt/llm_models/models/GLM-5.3-Flash/config.json` (text_config,
model_type `glm5_next`) + safetensors inventory (2026-09-04, 62 shards, 306 GB,
FP8 e4m3 blockwise 128×128 with `weight_scale_inv`).

```
architectures = Glm5NextForConditionalGeneration (vision truncated, counted)
hidden = 4096, vocab = 154880
Layers: 45 main (0..44) + MTP nextn (45), num_nextn_predict_layers=1
KDA (linear_attention): 34 layers [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,
  25,26,28,29,30,32,33,34,36,37,38,40,41,42,44]
  64 heads x head_dim 128; short conv1d kernel=4 (q/k/v weights [8192,1,4]);
  A_log F32[64], dt_bias F32[8192]; beta b_proj BF16[64,4096];
  gate low-rank f_a/f_b, g_a/g_b (128 rank); o_norm RMS BF16[128]
DSA (deepseek_sparse_attention): 11 layers [3,7,11,15,19,23,27,31,35,39,43]
  MLA: q_lora=1536, kv_lora=512, qk_head_dim=256, qk_nope=256, qk_rope=0,
  v_head_dim=256, 64 heads; mla_use_nope=true
  Indexer: index_n_heads=32, index_head_dim=128, index_topk=2048, index_kpool=4,
  compress=true, always_select_tail=true, rope_interleave=true
Hyper-connections (mhc=true): hc_mult=4, eps=1e-6, sinkhorn_iters=20
  hc_attn_fn / hc_ffn_fn BF16[24,16384]; hc_attn_base / hc_ffn_base F32[24];
  hc_attn_scale / hc_ffn_scale F32[3]
MoE: first_k_dense_replace=3 (layers 0..2 dense MLP I=12288), MoE layers 3..45
  n_routed_experts=288 (I=2048), n_shared_experts=1 (I=2048)
  Router: scoring_func=sigmoid, topk_method=noaux_tc, n_group=1, topk_group=1,
  num_experts_per_tok=8, norm_topk_prob=true, routed_scaling_factor=2.5,
  moe_router_dtype=float32; gate BF16[288,4096] + e_score_correction_bias F32[288]
eos_token_id=[154820,154827,154829], pad_token_id=154820
swiglu_limit=10.0
MTP: store-not-apply (operator decision 2026-09-04)
```

## Operator decisions

| Question | Decision |
|---|---|
| MTP | Store-not-apply (bit-exact autoregressive decode) |
| Expert placement | Pinned-host store of all 43×288 experts + LRU device slots (mechanics of GLM-5.2 / Qwen3.5-35B-A3B) |
| Conversion | Layer-wise shards (dense part + 43 MoE-layer passes), no RAM OOM |
| Attention | Custom KDA kernels: GDN from qwen4_exp + short conv1d; DSA/Indexer from GLM-5.2 |

## Approach: hybrid clone of proven stacks

A composition of bit-exact components without modifying them:
- KDA layers: the `qwen_gated_delta_kernel` mechanics (qwen4_exp) + hc GatedResidual + a new
  short conv1d component.
- DSA layers: Indexer/IndexShare from `glm_moe_inference.cu` (GLM-5.2).
- MoE: sigmoid/noaux_tc/top-8 router (modeled on `qwen_moe_inference.cu`), FFN on
  the NVFP4 GEMM path, shared expert + coefficient combine.
- Quantization: FP8 (e4m3, scale_inv 128×128) → dequant FP32 → requant NVFP4
  (E2M1 nibbles + E4M3 per-block scales + global scale). Norms/biases/embed/lm_head/
  conv1d/hc/gates — FP16/BF16 without quantization (the protocol's precision invariant).

## Components (naming: family=glm, type=moe, variant=flash)

| File | Role |
|---|---|
| `libs/lsxcommon/src/glm_moe_flash_constants.h` | frozen geometry + static_assert |
| `libs/lsxcommon/src/glm_moe_flash_convert.cpp` + `include/lsxcommon/glm_moe_flash_convert.h` | `ConvertGlm5NextModel` — classifier, layer-wise shards, format `logestix-glm5next` |
| `libs/lsxcommon/src/glm_moe_flash_load.cpp` + `include/lsxcommon/glm_moe_flash_load.h` | pinned dense upload, expert store 43×288, LRU device cache |
| `libs/lsxcommon/src/glm_moe_flash_inference.cu` + `include/lsxcommon/glm_moe_flash_inference.h` | KDA kernel (conv1d + gated delta), DSA/Indexer, MoE FFN, hc GatedResidual, generation driver `RunGlmFlashGeneration` |
| `libs/lsxcommon/src/glm_moe_flash_device_math.h` | host references KDA/hc/router for parity tests |
| `libs/lsxcommon/src/glm_moe_flash_model.cpp` + `.h` | `Glm5NextFlashModel` (ModelBase), kType=`glm-5.3-flash` |

Additional (protocol steps 4, 7):
- Tokenizer: `BuildTokenizerArrowFiles` from `tokenizer.json` (BPE byte-level,
  bos/eos/pad from the config: eos 154820/154827/154829, pad 154820).
- Passport: `model-metadata/passport/passport.bin` via `lsxnetwork::BuildPassport`,
  `LsxTokenTranslator` as a private member of IoWorker (loaded after pin, before set_value).

## Metadata

Manifest and indices — via `lsxcommon::lsx_model_metadata(rel)` in
`/mnt/llm_models/models/model-metadata/` (Linux) / `D:\Programs\huggingface.co\model-metadata`
(Windows). No hardcoded paths in the converter.

## Generation driver

- A single end-to-end `cudaStream_t` (the protocol's stream invariant), no
  `cudaDeviceSynchronize` in the generation loop.
- KDA: recurrent per-head delta state, conv1d ring-buffer with stride arithmetic
  (the protocol's step invariant).
- DSA: KV cache + indexer top-2048 block selection.
- RMW in shared memory: thread 0 only under `__syncthreads()` (the sandbox invariant).
- Expert residency: LRU device slots, residency ensured before the layer GEMM.

## Tests (TDD — tests first, then implementation)

| File | Content |
|---|---|
| `tests/test_glm_moe_flash_convert.cpp` | architecture classifier, FP8→NVFP4 parity against a host-dequant reference, dense/MTP/vision policy |
| `tests/test_glm_moe_flash_loader.cpp` | roundtrip, strictness on missing tensors, LRU slots |
| `tests/test_glm_moe_flash_kernels.cu` | conv1d, KDA gated-delta parity with the host reference, hc GatedResidual, router sigmoid top-8 + scaling 2.5, indexer top-k |
| `tests/test_glm_moe_flash_selfcheck.cpp` | real selfcheck "The capital of France is" → "Paris", 3 bit-exact cycles, `LSX_TEST_GLM_SELFCHECK_CYCLES`, SKIP_RETURN_CODE 4 |
| `tests/test_code_naming_guard.cpp` | extend the registries (variant `flash`) |

Regression: full `ctest -R glm` (GLM-5.2 untouched), then the whole suite.

## Speed (phase 2, after a green selfcheck)

Iterative tuning on live hardware: occupancy selection for KDA/DSA kernels (48 warps/SM max,
Blackwell), NVFP4 GEMM tile variants, LRU hit-rate metrics, t/s measurement (`--benchmark`).
Speedups only via new overloads/kernels — GLM-5.2 is not touched. The final
metric is recorded in the commit's README report.

## Risks

- The FP8→NVFP4 requant changes the quantization noise relative to the original — mitigated by
  parity tests against a host reference and the selfcheck.
- Sinkhorn (hc, 20 iterations) on the decode path — computed in F32; if too costly,
  move it to precomputed per-layer scales (parity check).
- Indexer top-2048 at long context — the budget is pinned as a frozen constant,
  mitigated by a degenerate-case test.
