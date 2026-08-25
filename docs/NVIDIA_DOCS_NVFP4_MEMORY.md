# NVIDIA Docs Reference — NVFP4 Codec & Device Memory (fixed 2026-08-26)

Authoritative excerpts used by the `fix/nvfp4-scale-dedup` audit/iteration.
Sources: CUDA C++ Programming Guide §4.3.2 (Memory Management) and the PTX ISA
`cvt` / block-quantization sections, retrieved via the nvidia-cuda-docs MCP.

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
