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
  (e.g. a zero-check on the e4 field, ceil-rounding of the pow2 E4M3 scale)
  requires updating the NVIDIA mirror **and** a regression test in the same
  change.

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
- Quantizing norms/biases "for consistency".
- Changing codec rounding and updating only the code, not the mirror + test.
- Assuming CUB works identically under MSVC.
- Passing a `__device__` symbol to a kernel launch, or branching on one from
  host code — see the section above.
