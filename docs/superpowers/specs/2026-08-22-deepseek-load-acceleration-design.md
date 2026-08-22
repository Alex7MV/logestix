# DeepSeek-V4-Flash Load Acceleration — Design

Date: 2026-08-22
Status: Approved (brainstorming session)
Etalon: GLM-5.2 fast load path (`docs/superpowers/specs/2026-08-22-glm-load-acceleration-design.md`)

## Problem

DeepSeek-V4-Flash weight loading is the legacy slow path:

- BF16→FP16 conversion runs as a **scalar host loop** per element (`upload_bf16_to_half`, `libs/lsxcommon/src/deepseek_inference_load.cu:30-52`) with a temporary pageable `std::vector<__half>` per tensor.
- H2D copies originate from **pageable memory**, degrading to synchronous staged copies.
- ~one `cudaMalloc` per tensor (~30+ tensors × 43 layers) instead of one arena per layer.
- Missing/broken layer files are **silently skipped**, leaving layers all-null (NaN/crash risk later at inference).
- Layer count hardcoded `kNumLayers = 43` (`deepseek_inference_load.cuh:23`) instead of topology-driven.
- Weights are re-loaded and freed on **every** `Infer()` call (`RunInferenceModel`, `deepseek_inference_pipeline.cu:289,822,832`), with a blocking `cudaDeviceSynchronize()` right after load (`:294`).

The GLM-5.2 model already solved all of these; this design ports that architecture 1:1 to DeepSeek.

## Goals

1. Fast load path: GPU in-place BF16→FP16 + pinned staging ring + per-layer device arena.
2. Strict error semantics: missing layer file or malformed tensor → fatal `cudaErrorInvalidValue`.
3. Topology-driven layer sizing.
4. Model-owned weights loaded once (`DeepSeekV4FlashModel::pw_`), no reload per Infer.
5. Remove post-load `cudaDeviceSynchronize`; single stream-scoped drain.

Non-goals: unifying loaders across GLM/DeepSeek/Qwen into one parameterized loader; accelerating Qwen (no fast-load analog yet); changing inference kernels.

## Architecture

### Component 1 — Shared fast-load primitives

- New header `libs/lsxcommon/src/fast_load_internal.h` (NVCC-safe, no absl containers): class `lsxcommon::fastload::PinnedStagingRing` moved from `glm_load_internal.h`. Same semantics: 2 pinned buffers × 8 MiB default chunk (min 1 MiB), 2 disable-timing events, parity guard via `cudaEventSynchronize`, RAII destructor draining both events before `cudaFreeHost`. Methods: `UploadBf16AsFp16(host_bf16, numel, d_dst, stream, bytes_acc)` and `UploadRaw(src, dst, bytes, stream, bytes_acc)`.
- Conversion kernel `bf16_to_half_in_place_kernel` + launcher `Bf16ToHalfInPlace(uint16_t*, int64_t, cudaStream_t)` move from `glm_inference_load.cu` into new `libs/lsxcommon/src/bf16_convert.cu` inside OBJECT lib `lsxcommon_cu`. Bit-exactness preserved (`bits << 16 → __float2half`); `-fmad=false` not required (kernel has no FMA ops).
- Public declarations in new `include/lsxcommon/fast_load.h`.
- GLM compatibility: `GlmBf16ToHalfInPlace` becomes a thin delegation to the shared symbol; `glm_load_internal.h` becomes a thin wrapper (or direct include) over `fast_load_internal.h`. All existing GLM tests must pass unchanged — regression guarantee for the refactor.

### Component 2 — DeepSeek fast loader (`deepseek_inference_load.cu`)

- Planning phase: before copying a layer, build a tensor plan (name → rows×cols×dtype), compute total size with 256-B alignment (`kArenaAlign`), then a single `cudaMalloc` device arena per layer. All layer `d_*` pointers become views into the arena. `LayerWeights` gains `void* d_arena; size_t arena_bytes;`; `Free()` releases only the arena when it is set.
- Uploads go through the ring: BF16 tensors via `UploadBf16AsFp16` (pinned staging + async H2D + in-place GPU conversion on the same stream); F32 / E4M3 / E8M0 raw bytes via `UploadRaw`. Host-pinned `h_gate_bias` / `h_tid2eid` unchanged (`cudaMallocHost` + memcpy).
- Topology-driven: layer count from `ModelTopology::num_layers` (parsed from `logestix/model.index.json`) instead of `kNumLayers=43`; hard check that the `compress_ratios` table size matches.
- Removed: host-scalar `upload_bf16_to_half`, per-tensor `cudaMalloc`, silent `continue` on missing/unreadable layer files.
- Invariant (stream): the whole load runs on a single `cudaStream_t`; synchronization only via ring event parity plus one final `cudaStreamSynchronize`.

### Component 3 — Weight ownership & pipeline

- `DeepSeekV4FlashModel` owns `gpu::PinnedWeights pw_;`: `Load()` parses topology and loads weights once; `Infer()` passes `pw_` into `gpu::RunInferenceModel(...)` instead of loading internally.
- `deepseek_inference_pipeline.cu`: remove local stack `PinnedWeights pw` and `INFER_CHECK(cudaDeviceSynchronize())`; `Load()` ends with one `cudaStreamSynchronize(stream)`.

## Error handling (strict, mirroring GLM)

- `ValidateJob` before each copy: tensor found by name/suffix, data size == expected rows×cols×dtype; otherwise return `cudaErrorInvalidValue` with an spdlog-error naming the tensor, free via `pw.Free()`, and drain pinned buffers through RAII (`StreamDrain` guard on every error exit).
- Missing layer file `<model>/logestix/pinned/layer_N.arrow` → fatal error (no skip).
- Shared-expert upload status is checked; failures are fatal (previously ignored).
- Geometry validation of embed/head/final_norm pointers after load, analogous to `GlmV5Model::Infer` checks.

## Testing (TDD — tests written first)

New `tests/test_deepseek_load_fast.cu` (tag `[deepseek][loadfast]`, compiled into lsxcommon_cu test target), ported from `tests/test_glm_load_fast.cu`:

1. Bit-exactness of GPU BF16→FP16 vs host reference `RefBf16ToHalf` at edge sizes {1, 7, 9, 4097}.
2. PinnedStagingRing chunking/reuse/raw-upload behavior.
3. Arena containment: every layer pointer inside `[d_arena, d_arena+arena_bytes)`; idempotent Free.

Extended `tests/test_deepseek_loader.cpp`:

4. Synthetic-model fixture (analogous to `glmtest::BuildSyntheticModel`) → fast load → D2H byte-exact verification of FP16/F32/raw-byte tensors.
5. `"DeepSeek loader errors on missing layer file"` → `cudaErrorInvalidValue`, empty layers after Free.
6. `"DeepSeek loader errors on malformed tensor"` → error + drained state.
7. Topology-driven sizing: num_layers comes from index.json, not the constant 43.

Regression gates: existing `ctest -R deepseek` and `ctest -R glm` suites stay green; DeepSeek SelfCheck determinism unaffected (loading only changes how bytes arrive, not what bytes arrive).

## Risks / notes

- Mixed dtypes per layer (BF16 + F8_E4M3/E8M0 raw + F32 HC): covered by ring's dual methods; F8 needs no conversion.
- absl-in-NVCC prohibition: `fast_load_internal.h` contains no absl includes; reachable from `.cu` TUs safely.
- GLM refactor is mechanical (move + alias) and guarded by the untouched GLM test suite.
