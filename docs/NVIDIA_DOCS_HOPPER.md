# NVIDIA Docs Reference — CUDA / Hopper (CC 9.0) Offline Mirror

> Source: NVIDIA CUDA documentation (CUDA C++ Programming Guide, PTX ISA,
> CUDA headers, CUTLASS notes), https://docs.nvidia.com/. Excerpts are
> copyright (c) NVIDIA Corporation and are reproduced here for in-repo
> reference with attribution. NVIDIA, CUDA, and PTX are trademarks of
> NVIDIA Corporation. See `THIRD_PARTY_NOTICES` section 9.

Offline mirror of nvidia-cuda-docs MCP excerpts covering the Hopper
generation (sm_90 / sm_90a, CUDA Toolkit 13.x), created 2026-09-09 by
explicit operator request, mirroring the structure of
`docs/NVIDIA_DOCS_NVFP4_MEMORY.md` (the Blackwell reference). Sections cite
their source (Programming Guide §, PTX ISA §, cuda_fp8.h docs).

**Repo status (critical context)**: per AGENTS.md, Hopper (sm_90) is a
FORBIDDEN build target — the project compiles exclusively for Blackwell
(`120-real;121-real`). This document is reference-only: for understanding third-party
Hopper-era kernels (CUTLASS, FlashAttention-3, TensorRT-LLM code paths),
for porting comparisons, and because several Blackwell features are
evolutionary descendants of Hopper introductions (TMA, mbarrier, clusters,
programmatic dependent launch).
Nothing in this doc changes the build: never add `90-real` or `sm_90a`
targets to CMake.

A recurring theme: **many Hopper MMA features are `sm_90a`
architecture-specific** and do NOT carry forward to Blackwell consumer dies
— the mirror image of the Blackwell situation where `tcgen05` tensor-memory
MMA skips sm_120/sm_121 (see Blackwell reference §B4).

## H1. Compute Capability 9.0 hardware limits

(Programming Guide, "Compute Capabilities" appendix, Tables 30/31;
re-verified via nvidia-cuda-docs MCP 2026-09-09.)

| Limit | CC 9.0 value | (Blackwell CC 12.x for contrast) |
|---|---|---|
| Max resident warps per SM | **64** | 48 |
| Max resident threads per SM | **2048** | 1536 |
| Max resident blocks per SM | **32** | 24 |
| Registers per SM / per block | 64 K × 32-bit | 64 K |
| Max registers per thread | 255 | 255 |
| Shared memory per SM / per block max | **228 KB / 227 KB** | 100 KB / 99 KB |
| Shared memory banks | 32 | 32 |
| Warp size | 32 | 32 |
| Max threads per block (x*y) / z | 1024 / 64 | 1024 / 64 |
| Max grid x-dim / y,z-dim | 2^31−1 / 65535 | same |
| Max resident grids per device (concurrent kernels) | 128 | 128 |
| Constant memory | 64 KB (8 KB working set/SM) | same |
| Local memory per thread | 512 KB | 512 KB |
| Ratio of FP32 to FP64 throughput | 2:1 | 64:1 (CC 12.x) |

Key contrasts for occupancy math when reading/porting Hopper kernels:
Hopper allows **more** warps (64 vs 48) and **much more** shared memory
(228 KB vs 100 KB per SM) than consumer Blackwell. A Hopper kernel that
stages a 200 KB SMEM pipeline (typical wgmma producer/consumer pattern)
cannot be transplanted to sm_120 without re-tiling — the Blackwell consumer
budget is under half that. Green contexts: minimum SM partition size 8 on
CC 9.0.

## H2. Alternate FP data formats on Hopper

(PTX ISA §5.2.3 "Alternate Floating-Point Data Formats"; cuda_fp8.h docs.)

Hopper natively supports the 8-bit formats (and 16-bit fp6 for tcgen05-era
PTX, not hardware MMA on Hopper).

| Format | Bits (E/M/S) | Inf | NaN encodings | Notes |
|---|---|---|---|---|
| `e4m3` | 4/3/+sign | none | **only `0x7f`, `0xff`** | FP8, native wgmma type |
| `e5m2` | 5/2/+sign | yes | IEEE-like | FP8, native wgmma type |
| `ue8m0` | 8/0/unsigned | none | **only `0xff`** | scale format (MX-era PTX) |

`cuda_fp8.h` conversion semantics (relevant when porting Hopper FP8 paths
to the repo's NVFP4 scale machinery):

- `__NV_SATFINITE`: out-of-range converts to **MAXNORM of the same sign**
  (identical flavor rule as the FP4 path in the Blackwell reference §B3).
- `__NV_NOSAT` on e4m3: values beyond MAXNORM become **NaN** (e4m3 has the
  NaN slot; E2M1 does not — FP4 NaN inputs map to +MAXNORM instead).
- Round-to-nearest-even is the hardware conversion default.

Tensor Core input data types, CC 9.0 (Programming Guide Table 33,
verified via MCP 2026-09-09):

| Type | CC 9.0 | Notes |
|---|---|---|
| FP64 | Yes | datacenter feature, back on Hopper (absent 8.6/8.7) |
| TF32 / BF16 / FP16 | Yes | |
| FP8 (e4m3/e5m2) | Yes | first generation with native FP8 MMA |
| FP6 / FP4 | — | start at Blackwell (CC 10.x / 12.x) |
| INT8 | Yes | |
| INT4 | — | **dropped** (Turing/Ampere had it; CC 9.0 does not) |

Consequences: there is **no FP4 hardware path anywhere in Hopper** —
the E2M1/NVFP4 formats of the Blackwell reference have no Hopper MMA path.

## H3. Warpgroup MMA — `wgmma.mma_async` (PTX ISA §9.7.16, sm_90a)

The defining Hopper MMA feature. A **warpgroup** (128 threads = 4 warps)
collectively issues one asynchronous `D = A·B + D` matrix
multiply-accumulate.

Programming model:

- **Fence**: `wgmma.fence.sync.aligned` establishes ordering between prior
  warpgroup register accesses and subsequent `wgmma.mma_async` accesses.
  Mandatory before the first wgmma and between a register write and a wgmma
  touching the same registers (except accumulator-only reuse across wgmma of
  the same shape). Issued by all 4 warps; divergence → undefined behavior.
- **Commit/Wait**: `wgmma.commit_group.sync.aligned` /
  `wgmma.wait_group.sync.aligned N` order and drain the async groups.
- **Operand sourcing**: B always from shared memory via a 64-bit **matrix
  descriptor** (contents must be identical across all warps of the
  warpgroup); A from either an SMEM descriptor or registers.
  Accumulator lives in registers (RMEM) — unlike Blackwell tcgen05, where
  the accumulator lives in **tensor memory**.
- **Transposes/negates**: `imm-trans-a`/`imm-trans-b` (0/1) only for
  `.f16`/`.bf16` SMEM-descriptor operands; `imm-scale-a`/`imm-scale-b`
  (±1) element-wise negation for FP variants.
- **Sparse support**: structured sparsity with `sp-meta`/`sp-sel` metadata.

Shape/type matrix (dense + sparse sp-sel conventions):

| Shape | `.atype`/`.btype` | Element size | K per instruction |
|---|---|---|---|
| `.m64nNk16` | `.tf32` | 32-bit | 16 |
| `.m64nNk32` | `.f16` / `.bf16` | 16-bit | 32 |
| `.m64nNk32` | `.e4m3` / `.e5m2` | 8-bit | 32 |
| `.m64nNk64` | `.s8` / `.u8` | 8-bit | 64 |

Precision semantics (PTX ISA notes):

- FP16/BF16/TF32: element-wise multiply at least single precision;
  `.dtype .f32` accumulates at least single precision; `.dtype .f16`
  accumulates at least half precision. Accumulation order, rounding and
  subnormal handling are **unspecified** — wgmma output is NOT bit-stable
  contract material (contrast: the repo pins bit-exact logits via
  `-fmad=false` warp-level paths, Blackwell reference §3).
- **FP8 accumulation caveat (PTX ISA 9.2 semantic clarification)**: for
  `.e4m3`/`.e5m2` with `.dtype .f32`, the current implementation accumulates
  at *higher than half precision but lower than single precision*.

Porting note for this repo: Blackwell consumer (sm_120/sm_121) has **no**
`wgmma` tensor-MMA route either (`tcgen05` skips it, and wgmma is sm_90a
territory); the repo's warp-level
`mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale` is the correct
CC 12.x path — Blackwell reference §B4.

## H4. TMA — Tensor Memory Accelerator (Programming Guide "TMA"; PTX ISA `cp.async.bulk`)

Introduced in compute capability 9.0 (Hopper) — TMA is a Hopper
first-generation feature that Blackwell inherits (the repo already uses it
on sm_120: probe `ProbeTmaBulkCopy` passes, Blackwell reference §B4/§B6).

- Terminology: *bulk-asynchronous copies* (1-D contiguous, no descriptor)
  vs *bulk-tensor asynchronous copies* (multi-D up to 5-D, needs a tensor
  map created host-side via `cuTensorMapEncodeTiled` / `cuTensorMapEncode`).
- Tensor map passes to the kernel as a **`const` parameter annotated
  `__grid_constant__`**; device copies tiles gmem↔smem (and smem→gmem,
  smem→distributed-smem within a cluster).
- Copy eligibility: barrier in shared memory, data 16-byte aligned, source
  global, destination shared.
- **Multicast** (gmem → SMEM of multiple cluster blocks) is *optimized for
  `sm_90a`* and may be significantly slower on other targets. On Hopper
  datacenter dies it is the intended fast path; on the repo's Blackwell
  consumer targets avoid assuming multicast speedups.
- Async-proxy operation; completion via mbarrier (`expect_tx` tx-count or
  `cp.async.bulk.commit_group`/`wait_group` variants). The issuing thread
  keeps computing — never wrap TMA issues in host syncs.

## H5. Thread block clusters and Distributed Shared Memory (CC 9.0+)

(Programming Guide "Thread Block Clusters" / "Distributed Shared Memory";
Cooperative Groups `cluster_group` API.)

- Clusters are launched with a cluster dims grid attribute; **max cluster
  size is hardware-dependent and varies between devices** (Programming
  Guide §1.2.2.1.1 — no portable constant is published; query
  `cudaOccupancyMaxActiveClusters` / device props at runtime). All blocks
  of a cluster are co-scheduled onto the same GPC.
- **DSMEM**: threads of a cluster can read, write, or atomically access
  shared memory of any block in the cluster
  (`cluster.map_shared_rank(addr, rank)` /
  `this_cluster().map_shared_rank`), address space =
  "Distributed Shared Memory".
- DSMEM capacity = blocks-per-cluster × shared-memory-per-block; per-block
  static/dynamic smem limits (H1) are unchanged.
- Correctness contract: remote reads require all blocks resident —
  `cluster.sync()` before assuming visibility, and a block must not exit
  while remote blocks may still read its SMEM.
- TMA interaction: bulk-tensor copies may target DSMEM of a peer block in
  the cluster (H4).

## H6. Asynchronous barriers — mbarrier (PTX ISA §9.7.14.16)

Hopper-era synchronization primitive that TMA and wgmma completion hang
off. Facts worth keeping pinned:

- Opaque `.b64` object, 8-byte alignment, lives in `.shared` memory;
  user-defined count — only limited by SMEM capacity (unlike the limited
  hardware `bar.sync` barriers per CTA).
- Tracks per phase: pending/expected arrival counts and **tx-count**
  (pending async transactions). Operations:
  `mbarrier.init`, `arrive`, `arrive_drop`, `expect_tx`, `complete_tx`,
  `test_wait`, `try_wait`, `pending_count`.
- Phases flip when expected arrivals complete; `try_wait` with explicit
  **parity** (`mbarrier_try_wait_parity(bar, phaseParity)`, parity ∈ {0,1},
  init phase has parity 0) allows single-thread-arrive + all-threads-wait
  pipelines — the canonical producer/consumer pattern under TMA loads.
- Warp-divergence caveat: a converged warp's arrive updates the barrier
  once; a fully diverged warp applies 32 updates — re-converge
  (`__syncwarp`) before arrive-on. (Mirrors the repo's shared-memory RMW
  discipline: Model Ingestion Protocol step 6.)
- Any mbarrier op except `init` on an uninitialized object, or any
  non-mbarrier op on an initialized one, is UB.

## H7. Programmatic Dependent Launch (CC 9.0+)

(Programming Guide §"Programmatic Dependent Launch and Synchronization".)

- A secondary kernel launched in the same stream with attribute
  `cudaLaunchAttributeProgrammaticStreamSerialization` may start **before**
  the primary kernel finishes: the primary calls
  `cudaTriggerProgrammaticLaunchCompletion()` when its dependents may
  launch (implicit at kernel exit if never called).
- The secondary MUST call `cudaGridDependencySynchronize()` before touching
  results of the primary — thread blocks can launch before primary writes
  are visible/flushed.
- Concurrency is **opportunistic**, not guaranteed; relying on overlap for
  correctness can deadlock. Useful for hiding the launch latency + preamble
  (buffer zeroing, constant loads) of dependent kernels.
- Relevance to the repo's stream invariant: the "one end-to-end
  cudaStream_t, no per-token host sync" rule (Model Ingestion Protocol
  step 5) already excludes per-kernel host round-trips; PDL is the
  device-side complement for kernel-chained graphs, available on both
  Hopper and Blackwell.

## H8. Target qualifiers: sm_90 vs sm_90a (PTX ISA §"Target Architecture")

(PTX ISA "Semantic" / target-architecture table; re-verified 2026-09-09.)

- Onion-layer model: later generations retain features of earlier ones.
- **`sm_90a`** = architecture-SPECIFIC features (wgmma and friends):
  supported only on sm_90; code compiled for it does NOT run on later
  devices and does not follow the onion model.
- **`sm_90f` does not exist**: family-suffix (`f`) targets start with
  sm_10x (sm_100f, sm_103f) and sm_11x/sm_12x. Hopper has no
  family-forward-compatible tier — Hopper-only features are locked to
  `a`-targets, unlike Blackwell datacenter where `sm_100f` gives a
  forward-compatible path within the sm_10x family.
- Practical consequence (mirrors Blackwell reference §B4's tcgen05
  conclusion): any `sm_90a`-gated kernel (wgmma, Hopper multicast-tuned
  paths) is dead code for this repo's sm_120/sm_121 targets. Portable
  Hopper-era techniques that DO carry to the repo: TMA bulk/bulk-tensor
  copies (§H4), mbarrier pipelines (§H6), clusters/DSMEM (§H5), PDL (§H7).

## H9. Hopper → Blackwell porting cheat sheet

Distilled cross-references for reading Hopper-era code against this repo's
Blackwell (CC 12.x) reality:

| Hopper concept | Blackwell CC 12.x counterpart / status |
|---|---|
| `wgmma.mma_async` (sm_90a) | absent; use warp-level `mma.sync` block-scale (`mxf4nvf4`) — no tcgen05 on sm_120/121 |
| Accumulator in registers (RMEM) | tcgen05 would use tensor memory, but consumer dies lack it — RMEM stays |
| FP8 (e4m3/e5m2) MMA | FP8 still available; NVFP4 (E2M1+UE4M3/UE8M0 scales) is the repo's production path |
| No FP4 hardware | NVFP4 TC path via `mma.sync...mxf4nvf4` / cuBLASLt (Blackwell ref §B4/§B5) |
| TMA multicast fast path (sm_90a-tuned) | present but deoptimized off sm_90a; avoid multicast assumptions |
| 228 KB SMEM/SM, 64 warps/SM | 100 KB SMEM/SM, 48 warps/SM — re-tile pipelines |
| mbarrier, clusters/DSMEM, PDL, `cudaMallocAsync` | carried forward unchanged (Blackwell ref §2, §B6) |
