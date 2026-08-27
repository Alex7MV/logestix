# NVIDIA Docs Reference — CUDA / Blackwell / NVFP4 / Memory

Offline mirror of nvidia-cuda-docs MCP excerpts needed by this project
(CUDA Toolkit 13.x, Blackwell sm_120/sm_121, NVFP4 inference). Kept in-repo
because not every dev box has MCP access. Fixed 2026-08-26; sections cite
their source (Programming Guide §, PTX ISA §, cuda_fp4/cuda_fp8 headers,
cuBLASLt samples).

Part A covers the NVFP4 codec and stream-ordered memory used by the
`fix/nvfp4-scale-dedup` iteration; Part B is the broader Blackwell/CUDA
reference.

## 1. NVFP4 quantization recipe (Programming Guide, "1D Block Quantization")

For a block of k elements (NVFP4: k = 16; partial blocks treat missing values
as zero):

1. Block absolute maximum `Amax(D^b_fp32)`.
2. Block scaling factor in fp32: `S^b = Amax(D^b) / Amax(e2m1)` where
   `Amax(e2m1) = 6.0`.
3. **FP4 with UE4M3 scales**: `S^b_e4m3 = e4m3(S^b)`; conversion factor
   `R^b = 1 / fp32(S^b_e4m3)`; each element is stored as
   `DType(d_fp32 * R^b)`.
   Dequant therefore is `E2M1(nibble) * fp32(S^b_e4m3) [ * global_scale ]`.
4. Tensor-wide pre-scale (`scale_D^in = Amax(e2m1)*Amax(e4m3)/Amax(D_temp)`)
   is what logestix stores as the separate float32 `global_scale`.

**logestix deviation (deliberate, documented)**: block scales are
power-of-two-only E4M3 payloads — exponent bits only, rounded UP (ceil), so
the block maximum can never saturate. Canonical codecs:
`lsxcommon/nvfp4_scale.h` (`FloatToE4m3P2` ceil encode with exact-pow2
identity, clamp e4∈[1,14]) and its delegate `nvfp4_codec.h::E4M3ScaleToFloat`.
Historical note: a bare `frexpf`-based encode doubles the scale on exact
powers of two (`frexp(1.0) = (0.5, 1) → 2·2^0`) — this was a real defect fixed
by this branch; do not reintroduce.

**Scale-byte zero semantics (regression-pinned)**: the zero test is on the
e4 FIELD (`(byte >> 3) & 0xF == 0`), never the whole byte. Bytes `0x80..0x87`
have a zero exponent field and decode to exactly 0. NVIDIA docs agree this is
the only sound reading: per cuda_fp8 E4M3 semantics those bytes are −0 /
negative subnormals (−m·2^-9), i.e. never legitimate scale factors, and the
hardware NVFP4 scale type is **unsigned** `float_ue4m3_t` (Blackwell
tcgen05-mma narrow-precision table; `nv_float4_t` → SF = UE4M3, vector size
16 / `.block16` — matching the repo's per-16 blocks). A whole-byte zero check
instead decoded `0x82` as 2^-7 and silently corrupted every GEMM whose stored
scale pattern contained such bytes.

## Hardware scale-factor facts (PTX ISA tcgen05 / Blackwell)

- Block-scaled MMA: `D = C + (A × SFA) · (B × SFB)`, scale factors applied
  along GEMM-K every SV elements (SV = 16 or 32).
- Valid SF element types: UE8M0 for `mxf8f6f4`/`mxf4`; for NVFP4
  (`kind::mxf4nvf4`, E2M1 elements) either UE8M0 or **UE4M3** with
  `.scale_vec::4X` / `.block16` (K-block of 16).
- UE4M3/UE8M0 are UNSIGNED (no sign bit); repo's pow2-E4M3 payload keeps the
  byte layout but only ever sets exponent bits via the canonical encoder.
- Repo's pow2-only scale flavor is an internal deviation from full UE4M3
  (mantissa bits unused by encode AND ignored by decode — both directions
  consistent).

### E2M1 value levels

sign(1) | exp(2, bias=1) | mant(1); no Inf/NaN encoding:

| nibble | value | nibble | value |
|---|---|---|---|
| 0x0/0x8 | ±0.0 | 0x4/0xC | ±2.0 |
| 0x1/0x9 | ±0.5 | 0x5/0xD | ±3.0 |
| 0x2/0xA | ±1.0 | 0x6/0xE | ±4.0 |
| 0x3/0xB | ±1.5 | 0x7/0xF | ±6.0 |

Canonical decode: `nvfp4_codec.h::E2M1Value` (inline __host__ __device__
function with immediates — never convert to a header-defined `__constant__`
array; see the ODR/constant-bank comment at the top of that header).

### Packing order (PTX cvt.rn.satfinite.e2m1x2.f32)

Value from input `a` goes to the UPPER 4 bits of the result byte, input `b`
to the LOWER 4 bits ⇒ element i of a byte: even index → low nibble,
odd index → high nibble ("lo nibble first"). All repo packers/unpackers
(`quantize_pack_perblock_kernel`, `unpack_nibbles8`, test hosts) follow this.

## 2. Stream-ordered memory management (Guide §4.3.2)

- `cudaMallocAsync(ptr, size, stream)` / `cudaFreeAsync(ptr, stream)` tie
  allocation/free to a stream: no host blocking, and memory pools reuse
  blocks to cut overhead and fragmentation.
- Allocation location follows the supplied memory pool or stream, NOT the
  current device/context.
- Cross-stream use requires ordering (events / `cudaStreamWaitEvent`) before
  access and before free.
- Mixing APIs is legal both ways:
  - `cudaMalloc` memory may be freed with `cudaFreeAsync`;
  - `cudaMallocAsync` memory may be freed with `cudaFree`, but the driver
    then assumes all accesses completed — sync first
    (`cudaStreamSynchronize` / `cudaDeviceSynchronize`).

**Project application (test code)**: unit tests own short-lived buffers and
already synchronize (`cudaDeviceSynchronize`) before teardown, so plain
`cudaMalloc`/`cudaFree` wrapped in the `DeviceBuffer` RAII helper
(`tests/test_blackwell_tc_mma.cu`) is sufficient and correct;
`cudaMallocAsync` pools would add nothing for these sizes/lifetimes.
Revisit only if tests start allocating per-iteration inside long loops.

## 3. Related repo invariants touched by this iteration

- E4M3 full decode (with mantissa, subnormals, NaN slot 0x7F/0xFF) lives ONCE
  in `lsx_kernels_quant.cuh::F8E4m3ToFloat`; the pow2 scale decode is the
  separate single-source flavor in `nvfp4_scale.h`. The two flavors are NOT
  interchangeable.
- E2M1 encode has two documented tie flavors in `lsx_kernels_quant.cuh`
  (`QuantizeE2m1TieUp` — runtime act-quant; `QuantizeE2m1Nearest` — offline
  converter). PTX hardware `cvt.rn` uses RNE (ties-to-even), which matches
  NEITHER flavor exactly — never mix hardware-cvt results into these paths.

---

# Part B — CUDA / Blackwell offline reference

## B1. Compute Capability 12.x (Blackwell consumer) hardware limits

(Programming Guide, "Compute Capabilities" appendix, Table 31/32.)

| Limit | CC 12.x value |
|---|---|
| Max resident warps per SM | **48** |
| Max resident threads per SM | **1536** |
| Max resident blocks per SM | 24 |
| Registers per SM / per block | 64 K × 32-bit |
| Max registers per thread | 255 |
| Shared memory per SM / per block max | **100 KB** / 99 KB |
| Shared memory banks | 32 |
| Warp size | 32 |
| Max threads per block (x*y) / z | 1024 / 64 |
| Max grid x-dim / y,z-dim | 2^31−1 / 65535 |
| Max resident grids per device (concurrent kernels) | 128 |
| Constant memory | 64 KB (8 KB working set/SM) |
| Local memory per thread | 512 KB |

Occupancy math for this repo MUST use 48 warps/SM and the 100 KB smem budget
(e.g. a kernel with >99 KB dynamic smem per block cannot launch; >48 warps
never co-reside). AGENTS.md's "48 warps/SM" is confirmed here.

## B2. Alternate FP data formats (PTX ISA §5.2.3 "Alternate Floating-Point Data Formats")

| Format | Bits (E/M/S) | Inf | NaN encodings | Storage |
|---|---|---|---|---|
| `e4m3` | 4/3/+sign | none | **only `0x7f`, `0xff`** | `.b8` |
| `e5m2` | 5/2/+sign | yes | IEEE-like | `.b8` |
| `e2m1` | 2/1/+sign | none | **none** | packed `e2m1x2` in `.b8` |
| `e2m3` / `e3m2` | 6-bit | none | none | packed `.b16`, 2 MSB zero-padded |
| `ue8m0` | 8/0/unsigned | none | **only `0xff`** | packed `ue8m0x2` in `.b16` |
| `ue4m3` | 4/3/**unsigned (7 bit)** | none | **only `0x7f`** | `.b8`, **MSB padded with zero** |

Consequences for this repo:
- A byte with MSB set (`0x80..0xff`) is NOT a valid UE4M3 scale factor; the
  repo's pow2-E4M3 payload reuses that bit space for its own layout, so scale
  bytes must always be interpreted through the canonical codecs
  (`nvfp4_scale.h`), never via raw numeric casts.
- `cuda_fp8.h` full-E4M3 decode of `0x80..0x87`: −0 and negative subnormals
  (`−m·2^−9`); NaN only at `0x7f`/`0xff`. Neither yields positive powers of
  two — field-zero → exactly-0 scale decoding (see Part A) stays the contract.

## B3. cuda_fp4.h conversion API semantics

(`__nv_cvt_float_to_fp4`, `__nv_cvt_float2_to_fp4x2`, … in `cuda_fp4.h`.)

- Rounding: caller-selected mode; hardware default flavor is
  round-to-nearest-even.
- Saturation (`__NV_SATFINITE`): out-of-range values saturate to **MAXNORM of
  the same sign** (E2M1 MAXNORM = ±6).
- **NaN inputs produce +MAXNORM (+6)**, not a NaN payload (E2M1 has none).

## B4. Blackwell tcgen05 block-scaled MMA (PTX ISA §9.7.17.10)

- Semantics: `D = C + (A × SFA) · (B × SFB)`; scale factors apply along GEMM-K
  every SV elements (16 or 32 dense).
- Scale-factor shapes vs `.scale_vectorsize`:
  - `.scale_vec::1X`/`.block32`: SFA M×1, SFB 1×N;
  - `.scale_vec::2X`/`.block32`: M×2 / 2×N;
  - `.scale_vec::4X`/`.block16`: M×4 / 4×N;
  - `.block16` = alias of `.scale_vec::4X`; `.block32` = `::1X` or `::2X`.
- Element/SF type combos:
  - `kind::mxf8f6f4` (E4M3/E5M2/E2M3/E3M2/E2M1) → SF = **UE8M0**, vec 32;
  - `kind::mxf4` (E2M1) → SF = UE8M0, vec 32;
  - `kind::mxf4nvf4` (E2M1) → SF = UE8M0 (vec 32) OR **UE4M3 (vec 16,
    `.block16`)** ← NVFP4 path used by this repo's TC kernels.
- CUTLASS naming: `nv_float4_t` pairs E2M1 with `float_ue4m3_t` (SF vector 16
  dense / 32 sparse) and is marked **not OCP-compliant**; OCP MX types pair
  any-FP4/6/8 with `float_ue8m0_t` vec 32.
- Only `float_ue8m0_t` and `float_ue4m3_t` can serve as tcgen05 scale-factor
  operand types on Blackwell.
- **tcgen05.mma availability vs consumer Blackwell** (PTX ISA §9.7.17.10,
  re-verified via nvidia-cuda-docs MCP 2026-08-27): every `tcgen05.mma`
  tensor-memory form targets SM_100a/SM_100f + SM_103a/SM_103f +
  SM_110a/SM_110f ONLY — sm_120a/sm_121a are NOT in that set. Consumer
  Blackwell therefore has no tensor-memory MMA; the repo's warp-level
  `mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale` path
  (`qwen_tc_gemm.cu`, empirically validated on RTX 5060 Ti / 5090) is the
  correct NVFP4 MMA route for CC 12.x. TMA bulk-tensor copies ARE available on
  sm_120 (probe `ProbeTmaBulkCopy` passes); only the tcgen05 tensor-memory
  compute side is absent.

## B5. cuBLASLt NVFP4 matmul (official sample `LtNvfp4Matmul`)

Facts to mirror when touching `libs/lsxcommon/src/benchmark_gpu.cu` or any
cuBLASLt-based FP4 path:

- Operands A/B: packed `__nv_fp4_e2m1`; per-tensor scales:
  `__nv_fp8_e4m3` device pointers (`a_scale`, `b_scale`); accumulate/output
  in bf16 (`C`), optional FP4 output `D` with its own `c_scale`/`d_scale`/
  `d_out_scale`.
- Scaling application controlled per-matrix via
  `cublasLtMatmulMatrixScale_t` mode flags (AScaleMode/BScaleMode/CScaleMode/
  DScaleMode/DOutScaleMode).
- Workspace required to enable split-K algorithms; α/β pointer mode set to
  host for α while β stays host; tensor-core ops are allowed implicitly —
  the global cublas handle's math-mode config is bypassed.

## B6. TMA / bulk-asynchronous copies (Programming Guide "Tensor Memory Accelerator", libcu++ pipelining notes)

- Terminology: *bulk-asynchronous copies* (1-D contiguous, no descriptor)
  vs *bulk-tensor asynchronous copies* (multi-D, needs a tensor map).
- Tensor map: created host-side via `cuTensorMapEncodeTiled`
  (`CUtensorMap`), passed to the kernel as a **`const` parameter annotated
  `__grid_constant__`**; device then copies tiles gmem↔smem.
- TMA eligibility (`cp.async.bulk` via barrier overload): barrier in shared
  memory, data 16-byte aligned, source global, destination shared. On
  Ampere+ plain `cp.async` needs only ≥4-byte alignment (gmem→smem).
- Up to 5-D arrays supported; copies may also go smem→gmem and
  smem→distributed-smem (cluster); multicast is optimized for `sm_90a` and
  may be significantly slower on other targets — avoid assuming multicast
  speedups on sm_120/sm_121.
- Transfers are async-proxy operations; completion uses the PTX async
  completion mechanisms (mbarrier etc.). The initiating thread keeps
  computing — do not add host syncs around TMA issues.

## B7. Stream-ordered memory management

See Part A §2 (kept there): `cudaMallocAsync`/`cudaFreeAsync` semantics,
pool/device selection rules, cross-API mixing rules, and the repo decision
that unit tests use RAII `cudaMalloc`/`cudaFree` (sync-before-free).
