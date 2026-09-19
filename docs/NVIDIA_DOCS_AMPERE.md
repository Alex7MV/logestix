# NVIDIA Docs Reference — CUDA / Ampere (CC 8.6 / 8.7) Offline Mirror

> Source: NVIDIA CUDA documentation (CUDA C++ Programming Guide, PTX ISA,
> CUDA headers, CUTLASS notes), https://docs.nvidia.com/. Excerpts are
> copyright (c) NVIDIA Corporation and are reproduced here for in-repo
> reference with attribution. NVIDIA, CUDA, and PTX are trademarks of
> NVIDIA Corporation. See `THIRD_PARTY_NOTICES` section 9.

Offline mirror of nvidia-cuda-docs MCP excerpts covering the Ampere
generation, focused on the two consumer/embedded dies the operator asked
to pin: **CC 8.6** (sm_86, desktop GA102) and **CC 8.7** (sm_87,
mobile/embedded GA10B — Jetson Orin family). Created 2026-09-09 by
explicit operator request, mirroring the structure of
`docs/NVIDIA_DOCS_NVFP4_MEMORY.md` (Blackwell) and
`docs/NVIDIA_DOCS_HOPPER.md`. Sections cite their source (Programming
Guide appendix tables, PTX ISA §, CUDA C++ Programming Guide).

**Repo status (critical context)**: per AGENTS.md, Ampere (sm_80) is a
FORBIDDEN build target — the project compiles exclusively for Blackwell
(`120-real;121-real`). This document is reference-only: for reading
Ampere-era third-party kernels (CUTLASS 2.x, FlashAttention-2, older
TensorRT-LLM paths) and understanding which techniques originated here.
Never add `80-real`/`86-real`/`87-real` targets to CMake.

Ampere is the generation that introduced **warp-level async copies
(`cp.async`)** — the ancestor of TMA (Hopper) and the mbarrier pipelines
this repo's Blackwell kernels rely on indirectly.

## A1. Compute Capability 8.6 / 8.7 hardware limits

(Programming Guide, "Compute Capabilities" appendix, Tables 30/31/32;
re-verified via nvidia-cuda-docs MCP 2026-09-09.)

| Limit | CC 8.6 | CC 8.7 | (Blackwell CC 12.x contrast) |
|---|---|---|---|
| Max resident warps per SM | 48 | 48 | 48 |
| Max resident threads per SM | 1536 | 1536 | 1536 |
| Max resident blocks per SM | **16** | **16** | 24 |
| Registers per SM / per block | 64 K × 32-bit | 64 K | 64 K |
| Max registers per thread | 255 | 255 | 255 |
| Shared memory per SM / per block max | **100 KB / 99 KB** | **164 KB / 163 KB** | 100 KB / 99 KB |
| Unified data cache (L1/smem) size | 128 KB | 192 KB | 128 KB |
| SMEM carve-out options (KB) | 0, 8, 16, 32, 64, 100 | 0, 8, 16, 32, 64, 100, 132, 164 | 0..100 |
| Shared memory banks | 32 | 32 | 32 |
| Warp size / max block | 32 / 1024 threads (z≤64) | same | same |
| Max grid x-dim / y,z-dim | 2^31−1 / 65535 | same | same |
| Max resident grids per device | 128 | 128 | 128 |
| Constant memory | 64 KB (8 KB working set/SM) | same | same |
| Local memory per thread | 512 KB | 512 KB | 512 KB |
| FP32:FP64 throughput (non-tensor) | 64:1 | 64:1 | 64:1 |
| Texture cache working set per SM | 28–128 KB | 28–192 KB | 28–128 KB |

Key reading of the table:

- **sm_87 is sm_86 with more L1/smem**: GA10B carries the GA100-class
  192 KB unified cache (up to 164 KB SMEM per block, same carve-out ladder
  as CC 8.0), while GA102 (sm_86) is capped at 128 KB unified / 100 KB
  SMEM — identical to consumer Blackwell's budget. A kernel sized for the
  100 KB smem tier runs on both; anything relying on >100 KB SMEM is
  sm_87-only within Ampere.
- sm_86/87 drop the CC 8.0 datacenter features: no FP64 tensor cores, and
  the FP32:FP64 ratio is 64:1 (vs 2:1 on GA100).
- Occupancy ceiling differs from Blackwell in *blocks*, not warps:
  16 resident blocks/SM on Ampere vs 24 on CC 12.x — small-block kernels
  with many blocks per SM lose co-residency when ported.

## A2. Tensor Core data types on CC 8.6 / 8.7

(Programming Guide Table 33 "Input Data Types Supported by Tensor Core
Acceleration per Compute Capability".)

| Type | CC 8.6 | CC 8.7 | Notes |
|---|---|---|---|
| FP64 tensor | — | — | FP64 TC only on CC 8.0 (GA100) and 9.0 |
| TF32 | Yes | Yes | introduced with Ampere |
| BF16 | Yes | Yes | introduced with Ampere (Turing has none) |
| FP16 | Yes | Yes | carried from Turing |
| FP8 / FP6 / FP4 | — | — | FP8 starts at CC 8.9 (Ada) / 9.0 (Hopper) |
| INT8 / INT4 | Yes | Yes | carried from Turing |

Consequences: there is **no FP8 and no FP4 hardware path anywhere in
Ampere** — everything this repo quantizes (NVFP4, E4M3 scales) would be a
software emulation on sm_86/87. TF32 and BF16 MMA are the widest Ampere
inputs; this is why the BF16→FP16 in-place conversion (`GlmBf16ToHalfInPlace`,
host fast path in `glm_moe_load.cpp`) targets 16-bit types, not 8/4-bit,
when reasoning about older hardware.

## A3. `cp.async` — warp-level asynchronous copies (PTX ISA §9.7.9.26)

The defining Ampere data-movement feature (PTX ISA 7.0, `sm_80+`).
Asynchronous gmem→smem copy issued per-thread, tracked in *cp.async-groups*.

- Syntax variants: `cp.async.ca.shared.global [dst], [src], 4|8|16;` and
  `cp.async.cg.shared.global [dst], [src], 16;` — `.ca` (cache-all) for
  4/8/16 bytes, `.cg` (cache-global, L1-bypass) for 16 bytes only.
- Alignment: data aligned to **at least 4 bytes** suffices on Ampere+
  (contrast: the Hopper+ TMA barrier overloads require 16-byte alignment
  and shared-memory barrier — see Hopper reference §H4). The libcu++
  backend picks `cp.async` automatically on Ampere+ under exactly these
  conditions.
- Grouping/waiting:
  - `cp.async.commit_group` closes the current cp.async-group;
  - `cp.async.wait_group N` waits until ≤ N most recent groups are
    pending (`wait_group 0` = all);
  - `cp.async.wait_all` ≡ `commit_group; wait_group 0`.
- Visibility rule (pinned): writes from a `cp.async` are visible to the
  issuing thread only after the matching `wait_all` / `wait_group`, or
  after `mbarrier.test_wait` returns true on an mbarrier tracking the
  operation. Two `cp.async` ops with no wait/mbarrier between them have
  **no ordering** — the classic silent-corruption trap when hand-rolling
  double buffering.
- Empty cp.async-groups are trivially complete.

`mbarrier` instructions (PTX 7.0) exist on Ampere and can track cp.async
completion via `mbarrier.arrive` tx-count — the mbarrier facts pinned in
the Hopper reference §H6 apply to sm_80+ already; only the *TMA*
completion overloads are Hopper+.

## A4. Warp-level MMA on Ampere (`mma.sync`, PTX ISA §9.7.15)

Ampere's tensor-core path is the warp-level `mma.sync.aligned.m16n8k*`
family (accumulator in registers) — the same instruction family this repo
uses on Blackwell consumer dies for NVFP4
(`mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale`, Blackwell
reference §B4). Canonical Ampere shapes:

| Shape | Type | Notes |
|---|---|---|
| `m16n8k8` | `.tf32` | K=8 (tf32 is 32-bit) |
| `m16n8k16` | `.f16` / `.bf16` | K=16 |
| `m16n8k8` | `.s8`/`.u8` dense family | integer path |
| `m16n8k8` | `.f64` | only CC 8.0; absent on 8.6/8.7 |

Precision semantics: element-wise multiply at least single precision;
`.f32` accumulator at least single precision; `.f16` accumulator at least
half; accumulation order/rounding/subnormals unspecified — same
non-bit-stable contract as wgmma (Hopper reference §H3).

**Structured sparsity** (introduced with Ampere): `mma.sp` doubles
effective throughput by skipping zeros in A. Metadata layouts
(re-verified via MCP 2026-09-09):

- sparse `m16n8k16` `.f16`/`.bf16`: A fragment = 2 `.b32` regs (2 nonzero
  f16 elements each out of 4 consecutive); metadata = one `.b32` holding
  **16 × 2-bit** indices, one per 4-wide chunk of A;
- sparse `m16n8k16` `.tf32`: A fragment = 4 `.b32` regs (1 nonzero tf32
  out of 2 consecutive elements each); metadata = one `.b32` holding
  **8 × 4-bit** indices per 2-wide chunk;
- `sp-sel` selects which thread pair (T0/T1 vs T2/T3) of each group owns
  the metadata path.

Fragment addressing convention (all shapes, reused by the repo's Blackwell
fragment math): `groupID = laneid >> 2`, `threadID_in_group = laneid % 4`;
rows `groupID` (+8 for even/odd halves), columns offset by
`threadID_in_group` × elements-per-thread.

## A5. `ldmatrix` — SMEM fragment loads (PTX ISA §9.7.15.4)

Bridge between cp.async-staged SMEM tiles and mma fragments.

- `ldmatrix.sync.aligned.m8n8.x1|x2|x4.shared.b16 {r}, [addr];` — a group
  of four consecutive threads loads one 8×8×16-bit matrix row (16 bytes);
  `.x4` covers 4 matrices per warp (threads 0–31 each supply one row
  address in four groups of 8).
- `.trans` loads column-major (mandatory for the 16×16 8-bit variant).
- Requires `sm_75+`; `::cta` sub-qualifier (PTX 7.8) constrains the
  address-space probe. Treated as a **weak memory operation** in the CUDA
  memory consistency model — order against cp.async writes only via the
  documented wait/mbarrier mechanisms (§A3), not via ldmatrix itself.
- PTX 8.6 added `.m16n16`/`.m8n16` shapes and `.b8` type (with 4-bit/6-bit
  source formats zero-padded into 8-bit elements) — relevant when reading
  modern kernels, not available to code targeting plain sm_86 PTX 7.x.

## A6. `cuda::memcpy_async` + `cuda::pipeline` (CUDA C++ Programming Guide "Asynchronous Data Copies")

Host-language wrapper over the same hardware path, usable on Ampere+:

- `cuda::memcpy_async(dst, src, cuda::aligned_size_t<N>(bytes), pipe)`
  issues the copy; on CC 8.0+ it lowers to `cp.async` when gmem→smem and
  alignment ≥ 4 bytes (large aligned chunks should use 16-byte alignment
  to hit the `.cg`/16B path).
- `cuda::pipeline<cuda::thread_scope_thread>` provides
  `producer_acquire / producer_commit / consumer_wait / consumer_release`
  — the rotate-buffer multi-stage pattern (canonical sample:
  `MatrixMulAsyncCopyMultiStageLargeChunk`, 4 stages of BLOCK_SIZE×
  BLOCK_SIZE float4-aligned tiles) overlaps load and compute without extra
  `__syncthreads` between stages (stages > 1 ⇒ next iteration writes a
  different buffer).
- With CC 8.0+ the pipeline may be backed by `cuda::barrier`
  (`cuda/barrier` header, `__CUDA_ARCH__ >= 700` guard in samples).
- This is the portable fallback of the Hopper/Blackwell mbarrier+
  TMA pipelines (Hopper ref §H4/§H6); same invariant applies: never mix
  unsynchronized cp.async/memcpy_async writes with reads.

## A7. L2 cache persistence (CUDA C++ Programming Guide §4.13; re-verified via nvidia-cuda-docs MCP 2026-09-09)

Ampere introduced the persisting-L2 window API. An **access policy window**
specifies a contiguous global-memory region and an L2 persistence property
for accesses within it.

Setup (full canonical sequence):

```cuda
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);   // set-aside (≤ prop.persistingL2CacheMaxSize)
stream_attribute.accessPolicyWindow.base_ptr  = ptr;
stream_attribute.accessPolicyWindow.num_bytes = window_size; // ≤ prop.accessPolicyMaxWindowSize
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

- The window can also be bound to a CUDA Graph kernel node via
  `cudaGraphKernelNodeSetAttribute(..., cudaKernelNodeAttributeAccessPolicyWindow, ...)`.
- **`hitRatio` semantics**: the fraction of accesses receiving `hitProp`
  is *random with probability ≈ hitRatio* (distribution is
  architecture/extent dependent). Canonical example: 16 KB set-aside +
  32 KB window — `hitRatio 0.5` randomly designates 16 KB of the window
  as persisting; `hitRatio 1.0` forces LRU eviction churn to keep the
  most-recent 16 KB. Below-1.0 values prevent thrashing and let concurrent
  streams' windows coexist (two 16 KB windows with hitRatio 0.5 coexist in
  a 16 KB set-aside; with 1.0 they evict each other).
- Three access properties:
  - `cudaAccessPropertyStreaming` — preferentially **evicted**;
  - `cudaAccessPropertyPersisting` — preferentially **retained** in the
    set-aside L2 portion;
  - `cudaAccessPropertyNormal` — **resets** a prior persisting status.
- Reset-to-normal (pinned — three ways):
  1. re-mark the region `cudaAccessPropertyNormal`;
  2. `cudaCtxResetPersistingL2Cache()` (all lines);
  3. eventual automatic reset of untouched lines — **reliance on it is
     strongly discouraged** (undetermined timing); persisting lines can
     survive long after use and starve later normal kernels.
- Disabling: set `num_bytes = 0` and overwrite the stream attribute, then
  `cudaCtxResetPersistingL2Cache()`.
- Sizing convention from the official example:
  `size = min(l2CacheSize * 0.75, persistingL2CacheMaxSize)`.
- Practicality on the dies this doc pins: desktop GA102 carries a small L2
  (few MB) — persisting windows matter far less than on GA100 (40 MB) /
  H100 (50 MB); treat the mechanism as datacenter-oriented.

## A8. Ampere → Blackwell (CC 12.x) porting cheat sheet

Cross-references for reading Ampere-era code against this repo's Blackwell
reality:

| Ampere concept | Blackwell CC 12.x counterpart / status |
|---|---|
| `cp.async` 4/8/16 B gmem→smem | still present; superseded by TMA bulk copies (Hopper ref §H4) for tiles |
| mma.sync m16n8k* (f16/bf16/tf32/int8) | same family; FP4 via `mxf4nvf4` block-scale (Blackwell ref §B4) |
| Structured sparsity `mma.sp` 2-bit/4-bit metadata | carried forward; NVFP4 sparse variants use UE4M3/UE8M0 SF |
| `ldmatrix` | still the fragment bridge for warp-level MMA |
| `cuda::pipeline` multi-stage | mbarrier + TMA pipelines preferred (Hopper ref §H4/§H6) |
| L2 persisting window | unchanged API; small L2 on GA102-class — little use |
| No FP8/FP4 hardware | NVFP4 is Blackwell-native; Ampere needs software emulation |
| 16 resident blocks/SM, 100 KB smem (sm_86) | 24 blocks/SM, 100 KB smem (sm_120) — block-count headroom improves |
| bf16 fast path | bf16 TC carried forward; repo keeps norms/biases BF16/FP16 (Ingestion Protocol step 2) |
