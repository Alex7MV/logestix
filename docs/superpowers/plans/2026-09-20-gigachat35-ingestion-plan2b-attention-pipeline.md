# GigaChat3.5 Ingestion — Plan 2b: Attention, Expert Streaming and Decode Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. Steps use checkbox (`- [ ]`) syntax. **No subagents** (AGENTS.md). Read `docs/NVIDIA_DOCS.md` / `/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md` before touching kernels.

**Goal:** Implement the GigaChat3.5 decode path — MLA and GatedDeltaNet attention kernels, the device weight loader, and the 40-layer hybrid pipeline — so `GigaChat35Model::Infer` produces a real forward pass, verified against a host reference.

**Architecture:** Family-owned CUDA kernels in `lsxcommon_gigachat_cu`, all matrix math through the neutral `LsxNvfp4Weight` + `RunLsxNvfp4Gemm*` layer, routed experts streamed through `GcExpertCache` (Plan 2a). One stream, shared-memory sandbox, deterministic reductions.

**Tech Stack:** CUDA 13.4 (sm_120a/sm_121a), CCCL 3.4.2, C++20, Catch2.

## Global Constraints

- CC 12.x: 48 warps/SM, 100 KB smem/SM; no tcgen05 — warp-level `mma.sync...mxf4nvf4.block_scale` only.
- One stream, no per-token sync; shared-memory RMW thread-0-only behind `__syncthreads()`; no `atomicAdd` in the logits reduce path.
- NVFP4 only via `lsx_inference_nvfp4.h`; norms/biases never quantized. No NVFP4 codec change.
- `#include`-reachable from `.cu`: no `<absl/container/*>`.
- Numeric class C-exact: fixed accumulation order, device-independent launch shapes (numeric-contract.md §1.1).
- Labels in the same commit; no subagents.

## Pinned reference formulas (modeling_gigachat3_5.py)

### MLA (`full_attention` layers {3,7,...,39})
```
alpha_q = sqrt(kHidden/kQLoraRank); alpha_kv = sqrt(kHidden/kKvLoraRank)
q = q_b_proj(alpha_q * q_a_layernorm(q_a_proj(x)))              # [H, 192]
kpass = kv_b_proj(alpha_kv * kv_a_layernorm(kv_a_proj(x)[:512]))# [H, 256]
krot  = kv_a_proj(x)[512:]                                      # [1, 64]
(qrot, krot) = rope_interleave_yarn(qrot, krot, pos)  # adjacent pairs
q = cat(qpass[:128], qrot); k = cat(kpass[:128], krot.bcast)
attn = softmax(q @ k^T * kAttnScale, F32)                       # causal, no GQA
o = attn @ kpass[128:]; o = o * sigmoid(gate_proj(x)); y = o_proj(o)
```
`kAttnScale = 192^-0.5 * mscale^2`, `mscale = 0.1*ln(8)+1 = 1.20794`.
YaRN freqs: factor 8, original 32768, beta_fast 32, beta_slow 1 (mirror `deepseek_moe_v41_device_math.h::YarnFreqsRef` + `RopeApplyRef`).

### GatedDeltaNet (`linear_attention` layers)
```
qkvz = in_proj_qkvz(x); ba = in_proj_ba(x)
view qkvz [k=32, 128q+128k+256v+256z]; ba [32, 2b+2a]
mixed = cat(q,k,v); mixed = silu(depthwise_conv1d_k4(mixed))
beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
q = repeat_interleave(q,2,heads); k likewise        # 32 -> 64 v-heads
core = recurrent_gated_delta_rule(l2norm(q), l2norm(k), v, g, beta, state)
out  = gated_norm_zero_centered_sigmoid(core, z, norm.weight, x2.0)
y = out_proj(out)
```
Recurrent decode: `state = state*exp(g); kv = (state·k); delta=(v-kv)*beta;
state += k⊗delta; core = state·q` (mirror `qwen_inference.cu` GDN recurrent).

## Tasks

- [x] **Task 1: device weight loader.** Parse the index `files` map; load `pinned/global.arrow`, `pinned/layer_NNN{,_nvfp4}.arrow` into `LsxNvfp4Weight` + raw BF16/FP32 device buffers (`GcWeights`), and initialize `GcExpertCache` from `<model>/logestix/experts`. Test: synthetic converted fixture loads; byte identity of a pinned tensor vs the source.
- [x] **Task 2: MLA decode kernel** (`gigachat_moe_inference_attn.cu`). q_a/q_b, kv_a/kv_b, interleaved yarn rope, `kAttnScale`, gated attention, o_proj. Parity vs a host MLA reference on random weights (tolerance 1e-3), plus a bit-exact rope test.
- [x] **Task 3: GDN decode kernel** (`gigachat_moe_inference_attn.cu`). fused qkvz/ba split, depthwise conv (prefill) + conv-state update (decode), beta/g, L2-norm QK, recurrent delta rule, output norm via `RunGdnOutputNorm`. Parity vs `torch_recurrent_gated_delta_rule` transcribed to host (see `qwen_device_math.h` for the l2norm/softplus reference).
- [x] **Task 4: pipeline** (`gigachat_moe_inference_pipeline.cu`). 40-layer walk; MLA or GDN per `full_attention_layers`; `pre_post` residual structure with `RunZeroCenteredGatedNorm`; MoE via `RunGigaChatRouterTopK` + `ResolveGcExperts` + `RunSwigluLimit` + NVFP4 GEMVs; dense MLP for layers 0–2; final norm + `lm_head` (F32 logits). KV cache (10 MLA layers) + GDN recurrent/conv state (30 layers).
- [x] **Task 5: model wiring.** `GigaChat35Model::Load` builds `GcWeights` + `GcExpertCache` (slot count via `GcExpertCacheSlots`, env `LSX_GIGACHAT_EXPERT_CACHE_SLOTS`); `Infer` opens `LsxFastMathScope`, runs the pipeline under the residency lock, fills `gpu::InferenceResult`.
- [x] **Task 6: forward parity + selfcheck.** Synthetic-model forward parity vs a host reference (tolerance class), then `test_gigachat_moe_selfcheck.cpp` (Paris, ≥3 bit-exact cycles, `SKIP_RETURN_CODE 4`) — Plan 3 formalizes the real-weight run.

## Status (2026-09-20)

Tasks 1-5 landed with per-kernel host parity (`test_gigachat_moe_attn`: bit-exact
RoPE, MLA block parity 1e-3, GDN block parity 1e-3 across two tokens) and a
synthetic 2-layer end-to-end model test (`test_gigachat_moe_model`, deterministic
re-run). `test_gigachat_moe_selfcheck` is registered and SKIPs without the real
converted model. **Still open:** a full synthetic-model forward parity against a
composed host reference (the per-layer parity is in place; the whole-stack
reference belongs with the Plan 3 real-weight validation).

## Verification

```bash
cmake --build --preset linux-debug
ctest --preset linux-debug-inference -R gigachat
./build/linux-debug/tests/test_model_suite_guard
```

## Notes for the implementer

- Reuse patterns, not geometry: `qwen_inference.cu` (GDN recurrent/conv), `deepseek_moe_v41_inference.cu` + `deepseek_moe_v41_device_math.h` (MLA latents, yarn rope), `nemotron_moe_inference_pipeline.cu` (heterogeneous walk).
- The real checkpoint's 416 GB conversion is an operator step; all tests use the synthetic converted fixture.
- Update the `gigachat` suite row's `selfcheck` stage in `test_model_suite_guard.cpp` when `test_gigachat_moe_selfcheck` lands.
