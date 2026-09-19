# NVIDIA Docs Reference — CUDA / Rubin (CC 11.0) Offline Mirror

> Source: NVIDIA CUDA documentation (CUDA C++ Programming Guide, PTX ISA,
> CUDA headers, CUTLASS notes), https://docs.nvidia.com/. Excerpts are
> copyright (c) NVIDIA Corporation and are reproduced here for in-repo
> reference with attribution. NVIDIA, CUDA, and PTX are trademarks of
> NVIDIA Corporation. See `THIRD_PARTY_NOTICES` section 9.

Offline mirror of nvidia-cuda-docs MCP excerpts covering the Rubin
generation (CC 11.0, targets `sm_110` / `sm_110a` / `sm_110f`, CUDA
Toolkit 13.x). Created 2026-09-09 by explicit operator request, mirroring
the structure of `docs/NVIDIA_DOCS_NVFP4_MEMORY.md` (Blackwell),
`NVIDIA_DOCS_HOPPER.md` and `NVIDIA_DOCS_AMPERE.md`. Sections cite their
source (Programming Guide appendix tables, PTX ISA §, CUTLASS notes).

**Repo status (critical context)**: this repo targets consumer Blackwell
exclusively (`120-real;121-real` per AGENTS.md; Hopper/Ada/Ampere
forbidden). Rubin (sm_110) is a datacenter-class successor generation —
this document is reference-only: forward-planning context for the NVFP4
stack (the repo's codec survives into Rubin *hardware MMA*), and a map of
which consumer-Blackwell limitations disappear on sm_110. Never add
`110-real` targets to CMake.

**Lineage caveat (pinned)**: PTX ISA 9.0 renamed targets
`sm_101/sm_101a/sm_101f` → `sm_110/sm_110a/sm_110f`. Older CUDA headers
and third-party code may reference either name — they are the same target
line. Per Table 61, the **sm_11x family** is `sm_110f` (+ sm_101f legacy)
and future targets.

## R1. Compute Capability 11.0 hardware limits

(Programming Guide, "Compute Capabilities" appendix, Tables 30/31;
re-verified via nvidia-cuda-docs MCP 2026-09-09. Blank cells in the docs'
tables inherit the previous column's value — the mapping below follows
that convention and is consistent with the independently verified CC 9.0
and CC 12.x values.)

| Limit | CC 11.0 | (CC 9.0) | (CC 12.x consumer) |
|---|---|---|---|
| Max resident warps per SM | **64** | 64 | 48 |
| Max resident threads per SM | **2048** | 2048 | 1536 |
| Max resident blocks per SM | **32** | 32 | 24 |
| Registers per SM / per block | 64 K × 32-bit | 64 K | 64 K |
| Max registers per thread | 255 | 255 | 255 |
| Shared memory per SM / per block max | **228 KB / 227 KB** | 228 / 227 | 100 KB / 99 KB |
| Shared memory banks | 32 | 32 | 32 |
| Warp size / max block | 32 / 1024 threads (z≤64) | same | same |
| Max grid x-dim / y,z-dim | 2^31−1 / 65535 | same | same |
| Max resident grids per device | 128 | 128 | 128 |
| Constant memory | 64 KB (8 KB working set/SM) | same | same |
| Local memory per thread | 512 KB | 512 KB | 512 KB |
| FP32:FP64 throughput (non-tensor) | 2:1 per table | 2:1 | 64:1 |
| Texture cache working set per SM | 28–256 KB | 28–256 KB | 28–128 KB |
| Green contexts: min SM partition (useFlags 0) | 8 | 8 | 8 |

Key reading: CC 11.0 is **datacenter-class geometry** — Hopper-like warp
and SMEM budgets (64 warps/SM, 228 KB SMEM/SM), far above consumer
Blackwell (48 warps, 100 KB). A pipeline re-tiled down to sm_120 budgets
(Hopper reference §H9) does not need re-tiling on Rubin — but consumer
Blackwell does not get that relief. The FP32:FP64 row: Table 33 shows **no
FP64 tensor cores on CC 11.0** (FP64 TC is CC 8.0/9.0/10.0-only), so the
inherited 2:1 ratio concerns non-tensor FP64 only; treat the FP64 column
with caution (docs' blank-cell inheritance artifact).

## R2. Tensor Core input data types, CC 11.0

(Programming Guide Table 33, verified via MCP 2026-09-09.)

| Type | CC 11.0 | Notes |
|---|---|---|
| FP64 | — | tensor-FP64 ends with CC 10.0 |
| TF32 / BF16 / FP16 / FP8 | Yes | |
| FP6 (e2m3/e3m2) | **Yes** | |
| FP4 (E2M1) | **Yes** | NVFP4/MXFP4 hardware MMA — see R3 |
| INT8 | Yes | `.kind::i8` on sm_110a (PTX 9.0) |
| INT4 | — | dropped after Ampere |

Full narrow-precision stack carried from Blackwell datacenter: NVFP4,
MXFP4, MXFP6, MXFP8 element+scale types (CUTLASS `float_subbyte.h`). The
repo's NVFP4 flavor (E2M1 + UE4M3 block scales, K-block 16) has a native
tensor-memory MMA path on sm_110a — see R3.

## R3. tcgen05 tensor-memory MMA on Rubin (PTX ISA §9.7.17)

The headline difference from consumer Blackwell: **tcgen05 works on
sm_110**. Consumer dies (sm_120/sm_121) have no tensor-memory MMA
(Blackwell reference §B4); datacenter sm_100/sm_103 and **sm_110** do.

Availability matrix for `tcgen05.mma` (Target ISA Notes, re-verified via
MCP 2026-09-09):

| Feature | Targets |
|---|---|
| `tcgen05.mma` base + block-scale variants | `sm_100a`, `sm_101a`→**`sm_110a`**, `sm_103a`; family tier `sm_100f`/`sm_110f` (PTX 8.8+, non-FP4 kinds) |
| `.kind::mxf4nvf4` (NVFP4) + `.kind::mxf4` | `sm_100a`, `sm_103a`, **`sm_110a`** |
| `.block16` / `.block32` aliases | **`sm_100f` or `sm_110f`** — family-tier, forward-compatible within sm_10x/sm_11x |
| `.scale_vec::4X` with `.ue8m0` for `mxf4nvf4` | PTX 9.1 addition |
| `.kind::i8` | `sm_100a`, **`sm_110a`** (PTX 9.0) |
| `tcgen05.mma.ws` / `.ws.sp` | `sm_100a`; `sm_100f` (PTX 8.8); **`sm_110f`** (PTX 9.0) |
| `tcgen05.ld.red` | `sm_110a`; `sm_110f`, `sm_103f` (PTX 8.8) |

Consequences for the repo's NVFP4 layer:

- On sm_110a, `tcgen05.mma...kind::mxf4nvf4.block_scale.scale_vec::4X`
  (PTX API block confirmed for SM_100a/SM_103a/**SM_110a**,
  `.cta_group::1/2`) is the hardware route for E2M1 + UE4M3 vec-16 — the
  exact semantics the repo's scale machinery implements (Blackwell
  reference §1/§B4). The repo's warp-level `mma.sync...mxf4nvf4` fallback
  (needed on sm_120) would be superseded by the tensor-memory path.
- `.block16` (alias of `.scale_vec::4X`, K-block 16) is explicitly
  `sm_100f`/`sm_110f` — on Rubin the NVFP4 block-16 flavor is
  **family-forward-compatible**, not `a`-target-locked.
- Accumulator lives in **tensor memory** (`tcgen05.alloc/dealloc`,
  `tcgen05.ld/st`, tmem↔rmem copies); PTX 9.0 semantic rule: all tcgen05
  instructions in a kernel must use the same `.cta_group`.
- CUTLASS support surface for this generation: FP4/FP6/FP8 + NVFP4/MXFP4/
  MXFP6/MXFP8, TMEM-based accumulator double buffering, sm100-style
  warp-specialized kernel layers (reused for sm_110).
- For contrast: warp-level `mma` with `.e2m1`/`.kind::mxf4nvf4` on
  consumer sm_120a/sm_121a (PTX 8.7) remains the consumer-Blackwell-only
  route — Rubin code and consumer-Blackwell code do NOT share an MMA path.

## R4. Cluster Launch Control and scheduling

(PTX ISA / CUTLASS notes.)

- **CLC (Cluster Launch Control)** — Blackwell datacenter feature for
  dynamic persistent tile scheduling; CUTLASS sm100 tile schedulers
  (dynamic persistent + stream-K) are built on it. CUTLASS pins the
  *flexible preferred/fallback cluster shapes* to `sm100a` kernels;
  hardware rasterizes a single cluster shape elsewhere — do not assume
  CLC flexibility carries to other targets without an `a`/`f` qualifier.
- `.blocksareclusters` directive (PTX 9.0): maps CUDA thread blocks onto
  clusters grid-wide.
- Multimem family grows in PTX 9.1/9.3 (`multimem.cp.async.bulk`,
  `multimem.cp.reduce.async.bulk`, then `multimem.st.async` /
  `multimem.red.async`) — NVLink-space multicast/collectives, NVLink
  fabric domain (Rubin ships NVLink 6 class interconnect; outside CUDA
  docs' scope).

## R5. Carried-forward platform features (all CC 9.0+)

Unchanged from Hopper/Blackwell (cross-references):

- TMA bulk / bulk-tensor copies (`__grid_constant__` tensor map) — Hopper
  ref §H4; on sm_110 the TMA completion machinery is the same mbarrier
  tx-count model.
- PTX 9.2 adds `.ignore_oob` for `cp.async.bulk` (out-of-bounds tolerance
  for tile-edge copies).
- Thread block clusters + DSMEM — Hopper ref §H5 (max cluster size is
  hardware-dependent; query at runtime).
- mbarrier — Hopper ref §H6; PTX 9.3 adds `.layout` qualifier +
  `mbarrier.check_layout` + report-predicate variants.
- Programmatic Dependent Launch — Hopper ref §H7 (CUTLASS enables PDL for
  back-to-back GEMM epilogue/prologue overlap).
- Stream-ordered memory (`cudaMallocAsync` pools) — Blackwell ref §2.

## R6. Rubin → consumer Blackwell (this repo) cheat sheet

What the sm_120/sm_121 targets gain/lose relative to Rubin:

| Rubin (CC 11.0) concept | CC 12.x consumer status |
|---|---|
| tcgen05 tensor-memory MMA (tmem accumulator) | **absent** on sm_120/121; warp-level `mma.sync...mxf4nvf4.block_scale` stays |
| tcgen05 NVFP4 `.block16` via family tier `sm_110f` | no tcgen05 at all → repo's warp-level path (Blackwell ref §B4) |
| 228 KB SMEM/SM, 64 warps/SM, 2048 threads/SM | 100 KB / 48 / 1536 — occupancy budgets differ ~2× |
| 32 resident blocks/SM | 24 |
| FP6 tensor cores | present on CC 12.x too (Table 33), unused by the repo |
| INT4 tensor cores | absent on both (dropped after Ampere) |
| CLC dynamic persistent scheduling | CUTLASS pins flexible-cluster flavor to sm100a; not assumed on sm_120 |
| TMA/mbarrier/clusters/PDL/pools | identical, carried forward |
