---
name: nvfp4-blackwell-conventions
description: Use when writing or reviewing CUDA/Blackwell kernels in logestix — NVFP4 matrix math, stream/async-pipeline design, shared-memory reductions, CCCL/cub/thrust choice, tile programming, CompileIQ tuning, alignment and occupancy, or zero-copy Arrow/simdjson/libzmq paths. Also covers the mandatory NVIDIA docs mirror discipline before changing NVFP4 codec semantics.
---

# NVFP4 / Blackwell conventions

## Overview

Kernel-level rules for this repo, all failure-derived: non-deterministic decode
from shared-memory races, silent numeric drift from codec changes, flags and
formats that only work on one of the two supported toolchains.

**REQUIRED SUB-SKILL:** use `test-driven-development`. Performance work (nsys
attribution, budget tests, baselines) is the `perf-tuning-a-model` skill.

## Execution model

- **One stream, end to end.** The whole prefill→decode loop runs on a single
  `cudaStream_t`. Minimize `cudaDeviceSynchronize()`; there is none inside the
  token loop (final sync only). Use non-blocking `cudaStream_t`/`cudaEvent_t`
  and `cudaMemcpyAsync`, and cooperative groups for intra-grid sync.
- **Shared-memory sandbox.** Any read-modify-write in shared memory is
  `thread 0` only, behind hard `__syncthreads()`; after the barrier the other
  threads read only. This is what removes Independent-Thread-Scheduling
  nondeterminism (a shared-memory data race → NaN → nondeterministic
  generation). Reference: `full_attn_softmax_kernel`.
- **Deterministic reductions** for anything feeding `final_logits`: shared
  partials + a fixed sequential sum, not `atomicAdd`.

## NVFP4

- Implement matrix operations **natively in NVFP4** via CCCL + `std::mdspan`.
- All NVFP4 GEMM goes through the model-neutral layer
  `libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h` (`RunLsxNvfp4Gemm` +
  `LsxFastMathScope`). Never create a family-named copy of the dispatcher.
- Norms and biases are never quantized — they stay FP16/BF16.
- The weight codec (E2M1 nibbles + per-16 E4M3 scales + `global_scale`) and
  the activation quantizer are shared contracts; changing codec semantics
  (e.g. the e4-field zero check, the E4M3 encode rounding) requires updating
  the NVIDIA mirror **and** a regression test in the same change.
- The NVFP4 block scale is the **complete E4M3FN value table** over the valid
  unsigned domain (bit 7 clear) — the hardware-native `.ue4m3` scale operand
  of the Blackwell `mma ... kind::mxf4nvf4.block_scale` path (PTX §9.7.17.10).
  Semantics: `e=(b>>3)&0xF`, `m=b&0x7`; `e>=1 -> 2^(e-7)*(1+m/8)`;
  `e==0 -> m*2^-9` (**subnormal**, `0x00 -> 0`); `0x7F` reserved (NaN) and
  never emitted; max finite `0x7E = 448`. Over `0x00..0x7F` this is
  bit-identical to `cuda_fp8`/`torch.float8_e4m3fn`, so pre-quantized ModelOpt
  W4A16_NVFP4 block scales (which use subnormals — ~3M bytes in
  NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) decode exactly as NVIDIA reads
  them. Canonical codecs: `nvfp4_scale.h::FloatToE4m3` (RNE encode on the
  normal AND subnormal grid) and `E4m3ToFloat` (the device delegate is
  `nvfp4_codec.h::E4M3ScaleToFloat`). Historical deviations — do not
  reintroduce: pow2-only decode (dropped the mantissa), and an `e==0 -> 0`
  decode (dropped the subnormal blocks).
- Where each piece lives (`libs/lsxcommon/src/`): `nvfp4_codec.h` — the
  `__host__ __device__` E2M1 nibble decode (branchless immediates, never a
  `__constant__` LUT: a header-defined array is merged across object
  libraries and a TU can read another model's copy) and the
  `E4M3ScaleToFloat` delegate; `nvfp4_scale.h` — the host/device pair
  `FloatToE4m3` / `E4m3ToFloat` plus the E8M0→E4M3 pair remap (E8M0 is an
  exact power of two, so the pair helper stays mantissa-0 by construction);
  `lsx_kernels_quant.cuh` — the signed FP8 decode (`F8E4m3ToFloat`, for genuine
  FP8 tensors) and the E2M1 encode flavors. The two E4M3 flavors are **not**
  interchangeable: NVFP4 block scales are unsigned via
  `nvfp4_codec`/`nvfp4_scale`; real signed FP8 data uses `F8E4m3ToFloat`.
- **Every decode of a weight row ends with `* global_scale`.** The per-16 E4M3
  block scale is only half of the NVFP4 value (`w = e2m1 * e4m3 * gs`); a
  hand-written family kernel (logits/head kernels are the usual place) that
  applies the block scale and forgets the tensor scale is silently off by
  `1/gs` — gemma-4-26b's NVFP4 logits kernel shipped 7.01x hot logits that way
  (`gs = 0.1426`), and greedy argmax hid it because a positive scale preserves
  the ordering, while the logit row, temperature/top-k semantics and the
  SelfCheck logit evidence were all wrong. Unit tests for such a kernel must
  use a **non-unit** `global_scale`; a `gs = 1` fixture cannot see the defect.
- Model-neutral device helpers that families alias rather than copy:
  `lsx_device_math.h` (`ExpfGlibc`/`RsqrtfGlibc`, aliased by
  `glm_moe_device_math.h`) and `lsx_reduce.cuh` (`BlockMaxShfl`, aliased by
  `qwen_reduce.cuh`).

## CCCL / cub / thrust

- Prefer `cub::DeviceReduce` / `cub::BlockReduce` over hand-written reduction
  or sort loops unless you need extreme register tuning.
- On MSVC, use thrust as the fallback (it uses CUB/cuDPP internally); draft
  reductions through shared memory because CUB is not MSVC-compatible.
- `cub::BlockReduce` / `DeviceReduce` need CUDA 13.3+ on Linux — compile there.
- **Tile C++ for compute kernels**: use tile programming instead of manual
  shared-memory staging and explicit TMA intrinsics; the compiler lays the
  tiles out for Blackwell.

## Tuning and layout

- **CompileIQ**: enable the autotuning framework in CMake for hot kernels;
  `--compilation-database` is available on Linux (CUDA 13.3+), on MSVC it is
  passed via `-Xcompiler` when the host compiler supports it.
- **Alignment / occupancy**: check data alignment for CUDA structures and the
  shared-memory budget per block. Blackwell: **48 warps/SM max**.

## Zero-copy host interop

- **Arrow + CUDA**: use `arrow::cuda::CudaBuffer`; avoid redundant Host→Device
  copies; zero-copy through `CudaMemoryManager`.
- **simdjson + libzmq**: wrap `zmq::message_t` straight into
  `simdjson::ondemand::parser` — no intermediate `std::string`.
- **Arrow file reads are mmap-backed, always.** Open with
  `arrow::io::MemoryMappedFile::Open` (never `ReadableFile::Open`), then either
  keep raw pointers into the mapping or `std::move` the payload out before the
  mapping dies. A retained payload must come from a **view** reader —
  `LoadTensorBatchViews` / `LoadDenseNvfp4BatchViews` / `LoadSharedExpertBatchViews`
  / `ExpertStore::Load`. The `*::UnpackBatch` family returns **owning**
  `std::vector<uint8_t>` records (`TensorRecord`, `PackedExpert`,
  `DenseNvfp4Record`): correct only for a payload consumed in the same scope
  (converted to BF16/F32, uploaded, freed), never for one kept alive.
- **Verify zero-copy instead of assuming it.** Cast the columns
  (`dynamic_pointer_cast` to the concrete `Array` type) and check each buffer
  against the mapping bounds before storing its pointer — the `in_range` checks
  in `libs/lsxcommon/src/tokenizer.cpp` are the reference. Without them, an
  Arrow build that hands back heap buffers silently reintroduces the copy you
  thought you removed.
- **Routed-expert host memory goes through the neutral `ExpertStore`**
  (`include/lsxcommon/expert_store.h`), whose `max_open` bounds mapped file.
  Two properties of that store are load-bearing and easy to lose:
  (a) the window must cover one token's working set — the layers are visited
  once each per token, so a smaller window is a cyclic scan with a *provably
  zero* hit rate, not a smaller cache; and (b) eviction must not drain the
  stream. An evicted slot's mapping is retired with a CUDA event and released
  once the copies issued so far have executed (`RetireSlot`/`ReapRetired`), so
  never reintroduce a `cudaStreamSynchronize` on that path — it stalls on
  unrelated work and measured 20.3 ms per eviction against 0.072 ms.
  The full rule set, the two sanctioned storage shapes and the measured cost of
  the forbidden ones are in the `ingesting-a-model` skill, section **"Host-side
  weight storage — `ExpertStore` and Arrow zero-copy"**.

## Never guess an API

Unsure about an Arrow/CCCL signature? Read the `.h`/`.cuh` in the tree or
compile it. Do not invent APIs — the repo's third-party versions are pinned and
differ from upstream docs.

## NVIDIA docs (mandatory, MCP)

NVIDIA documentation is **not vendored** in this repo: the previous
`docs/NVIDIA_DOCS_*.md` export from the `nvidia-cuda-docs` MCP was removed to
avoid redistributing NVIDIA copyrighted material. The pointer is
`docs/NVIDIA_DOCS.md`.

1. Before changing an NVFP4/CUDA kernel, load the current NVIDIA documentation
   from the `nvidia-cuda-docs` MCP (PTX ISA, CUDA C++ Programming Guide,
   `cuda_fp4`/`cuda_fp8` semantics, CUTLASS, TMA, tcgen05, CC 12.x limits) and
   reconcile format semantics and hardware limits. Do not rely on stale
   excerpts.
2. If the MCP is not available, stop and obtain access to the current
   documentation before proceeding.
3. When the MCP yields **new** facts the code depends on, pin the derived
   invariant in-code (comment/regression test) in the **same commit** — there
   is no offline mirror to extend.
4. Never diverge silently: a codec semantics change must update the matching
   in-code invariant and its regression test.

## `__device__` variables in host code

Evaluating a `__device__` variable from **host** code does not give you a device
address on this toolchain (MSVC host compiler + nvcc 13.4). The expression
resolves to a *host* pointer — `0x7ff6…` on Windows — and the kernel it was
handed writes straight into the process's address space:

```
Invalid __global__ write of size 4 bytes
  at ...::router_score_kernel(...) in gigachat_moe_kernels.cu:70   // score[e] = s
  Access to 0x7ff67766b220 is out of bounds
  (host_score=0x7FF60ABEB220, sym_score=0x1307600000)              // cudaGetSymbolAddress
```

- Never pass a `__device__` symbol as a kernel argument or dereference one in
  host code. Resolve the address with `cudaGetSymbolAddress`, cache it, and
  pass that. This is also what the runtime requires before the symbol can be
  used in a launch.
- For a device flag that host code must branch on (a debug switch, a mode
  selector), keep a plain host mirror and write it from the same value used for
  the symbol upload, so the two cannot diverge. A `static` initialised from
  `getenv` at load time is *not* equivalent if the env can be set later.
- The trap is independent of linkage: the same fault occurs with the variable
  at namespace scope. "It's in an anonymous namespace" is not the explanation.
- Precedent, both fixed: `gigachat_moe_kernels.cu` (`RouterScratchDev`) and
  `glm_moe_flash_inference.cu` (`flash_dbg_on_host`).

## Common mistakes

- A second stream for convenience, then a `cudaDeviceSynchronize` to "make it
  correct" — breaks the single-stream invariant and the perf model.
- `atomicAdd` in a reduction feeding `final_logits` → nondeterministic decode.
- A family-named NVFP4 GEMM wrapper instead of the neutral layer.
- Decoding a weight row without the table/tensor `global_scale`, or testing
  that kernel only with `global_scale = 1` (see the NVFP4 bullet above).
- Quantizing norms/biases "for consistency".
- Changing codec rounding and updating only the code, not the mirror + test.
- Assuming CUB works identically under MSVC.
- Passing a `__device__` symbol to a kernel launch, or branching on one from
  host code — see the section above.
