# GLM-5.2 Pinned-Weights Load Acceleration — Design

Date: 2026-08-22
Scope: `LoadGlmPinnedWeights` (`libs/lsxcommon/src/glm_inference_load.cpp`) only.
Approved approach: A — GPU in-place BF16→FP16 conversion + per-layer device arena + pinned staging ring.

## Problem

GLM load is dominated by three costs in the pinned-weights loader:

1. Scalar host BF16→FP16 conversion loop in `UploadBf16ToHalf`
   (~billions of elements across embed/lm_head + 79 layer files).
2. Pageable-memory H2D copies (every `cudaMemcpyAsync` from
   `std::vector` is effectively synchronous).
3. Hundreds of small synchronous `cudaMalloc` calls (one per tensor).

The expert index is already mmap-lazy and out of scope. The model is also
re-loaded on every `GlmV5Model::Infer()` call, which multiplies the cost.

## Requirements

- Bit-exact: converted FP16 weights must be bit-identical to the current
  host path (`bf16 bits << 16 → float → __float2half`, round-to-nearest).
  Existing `final_logits` baselines in `test_glm_selfcheck` stay valid.
- No on-disk format changes; Arrow layout, `PinnedSchema`, loader API
  signatures, inference kernels and callers unchanged.
- Strict error semantics (replaces silent skips):
  1. Missing `layer_<l>.arrow` for any `l` in topology → load error.
  2. Tensor found by suffix but empty/malformed (bad shape rank,
     `numel <= 0`, incomplete data) → load error.
  3. Absence of an optional tensor remains legitimate only where it is part
     of the heterogeneous layer grid (router only on MoE layers, indexer on
     indexer layers, MTP fields on layer 78). A required-for-type tensor
     missing is an error per rule 2.
- No hard speedup number required; wall-clock before/after is logged
  (spdlog) and compared on a real-model run.

## Architecture

Three replacements inside `LoadGlmPinnedWeights`:

### 1. GPU in-place conversion

BF16 and FP16 are both 2 bytes/element, so raw BF16 bytes are uploaded by
`cudaMemcpyAsync` directly into the final FP16 allocation, then the kernel
converts in place:

```cuda
// per element: float f = __uint_as_float(uint32_t(bits) << 16);
//              out = __float2half(f);   // RN — identical to host path
```

New unit: `GlmBf16ToHalfInPlace(uint16_t *d_buf, int64_t n, cudaStream_t)`
(kernel + launcher in `glm_inference_load.cu`, target `lsxcommon_glm_cu`;
declaration in `glm_inference_load.h`). Grid-stride; vectorized uint4 loads
(8 elements/thread) with element-wise ragged tail. Arena offsets are 256-byte
aligned so vectorized access is always safe.

In-place is race-free: each thread reads only the elements it writes.

### 2. Per-layer device arena

After parsing `layer_<l>.arrow` (records are already materialized in RAM via
`UnpackBatch`, as today), sum the FP16 footprint of all tensors of the layer
and issue ONE `cudaMalloc`; carve per-tensor pointers as offsets into the
arena. Global tensors (embed / lm_head / final_norm) keep individual
allocations (three total). Result: ~81 `cudaMalloc`s instead of ~1600+.

`GlmLayerWeights` gains `void *d_arena`. `Free()`: when `d_arena != nullptr`,
free only the arena and null all pointers; legacy per-pointer branch is kept
for hand-constructed structs. Layer weight pointers become arena views —
inference code is unaffected. Idempotent re-load (`pw.Free()` first) keeps
working unchanged. `resident_bytes` semantics unchanged (FP16 bytes).

### 3. Pinned staging ring

`PinnedStagingRing`: RAII helper, two pinned buffers of 8 MiB
(`cudaMallocHost`). Each tensor is uploaded in chunks:

```
TensorRecord.data (RAM)
  -> memcpy chunk N+1 -> pinned buf B      (host, overlapped)
  -> cudaMemcpyAsync(buf A -> arena off)   (true async: source is pinned)
  -> GlmBf16ToHalfInPlace(chunk, same stream)
```

Same-stream ordering guarantees the kernel never reads a chunk before its
copy completes; double buffering lets the host fill the next chunk while DMA
transfers the previous one. F32 tensors (`UploadRaw`, indexer k_bias) use the
same ring without the kernel. `h_gate_bias` stays `cudaMallocHost` + memcpy.

## Components

| Unit | Location | Change |
|---|---|---|
| `GlmBf16ToHalfInPlace` | `glm_inference_load.cu` (`lsxcommon_glm_cu`), decl in `glm_inference_load.h` | new |
| `GlmLayerWeights::d_arena` + arena-aware `Free()` | `glm_inference_load.h/.cpp` | new field |
| `PinnedStagingRing` (2 × 8 MiB, RAII) | anonymous namespace of `glm_inference_load.cpp` | new helper |
| `UploadBf16ToHalf` / `UploadRaw` | same file | rewritten over ring + arena |
| `Upload1D` / `Upload2D` / `LoadSingleLayer` | same file | signatures preserved; arena offsets, strict error checks inside |
| Load timing log | `LoadGlmPinnedWeights` | spdlog info: ms, bytes, GB/s |

Untouched: Arrow format, `PinnedSchema`, expert paths, `GlmExpertDb`,
inference kernels, all callers.

## Data Flow & Error Handling

Per-tensor flow: parse layer file → compute footprint → one arena alloc →
chunked ring upload → in-place convert kernel → next tensor.

Error rules:

- CUDA failure at any step: log, free that layer's/global arena, return the
  `cudaError*` as today. `Upload1D`/`Upload2D` keep surfacing failures with
  the tensor name; global-tensor failures remain fatal to the whole load.
- Before freeing the staging ring on any exit path:
  `cudaStreamSynchronize(stream)` — pinned buffers must outlive in-flight
  DMA. This is the only added synchronization; it is outside the generation
  hot path.
- Missing layer file / malformed tensor → error (see Requirements), with the
  offending path/tensor name logged — deterministic diagnostics for the
  lsxforge converter output instead of silent model degradation.

## Testing (TDD)

New `tests/test_glm_load_fast.cu` (Catch2, precedent:
`test_deepseek_inference_kernels.cu`):

1. Kernel bit-exactness vs the reference host scalar path on random BF16
   patterns including subnormals, ±inf, NaN payloads, round-to-even
   boundaries.
2. Tail/granularity: lengths not multiples of the uint4 vector width
   (1, 7, 9, 4097 elements) stay bit-exact.
3. Ring + chunking: a synthetic tensor larger than one chunk uploads through
   the ring and matches the reference on device; interleaved chunks preserve
   order.
4. Arena: all layer pointers fall inside one allocation after load;
   `Free()` releases exactly once; repeated `Load` is idempotent.
5. Errors: missing `layer_<l>.arrow` → error; tensor with empty shape →
   error (synthetic model built like the existing `test_glm_loader` helper).
6. Model-level bit-exactness regression: existing `test_glm_selfcheck`
   (SKIP without the real model) remains the arbiter of identical
   `final_logits`.

Performance: wall-clock timing log inside the loader; before/after captured
from real-model runs (SKIP without model).

## Out of Scope

- Expert index / mmap path acceleration.
- Caching loaded weights across `Infer()` calls (follow-up candidate).
- On-disk format change to pre-converted FP16 (lsxforge converter).
