#include "lsxcommon/shared_experts.h"
#include "lsxcommon/arrow_validate.h"
#include "lsxcommon/schema.h"
#include "nvfp4_codec.h"
#include "lsx_kernels_quant.cuh"

#include <cfloat>
#include <cstdint>
#include <memory>

//==============================================================================
// Blackwell (CC 12.0) NVFP4 shared-expert dequantization + zero-copy Arrow
// ingest.
//
// On-disk layout (pinned/shared_experts.arrow, one record per layer):
//   w1/w2/w3       packed E2M1 nibbles, 2 elems/byte, lo nibble first
//   w1/w2/w3 scale per-block E4M3 descriptor (1 byte per 16 elements)
//   global_scale   float32 multiplier
//   dequant:       value = E2M1(nibble) * E4M3(block_scale) * global_scale
//==============================================================================

namespace lsxcommon {
namespace gpu {
namespace {

// Decode primitives live in the shared nvfp4_codec.h (kE2M1Table /
// E4M3ScaleToFloat); alias them into this anonymous namespace so existing
// call sites compile unchanged.
using nvfp4::E2M1Value;
using nvfp4::E4M3ScaleToFloat;



// NVFP4 E4M3/E2M1 decode primitives now live in nvfp4_codec.h (aliased above).


__device__ __forceinline__ uint32_t half2_raw_bits(__half2 h)
{
    uint32_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Unpacks 8 contiguous nibbles from `u` (bytes lo-nibble-first) into 4
// uint32 words, each word being 4 consecutive FP16 outputs (2 half2).
__device__ __forceinline__ void unpack_nibbles8(uint32_t u, float scale, uint32_t* w)
{
    uint32_t n0 =  u        & 0x0F;
    uint32_t n1 = (u >> 4)  & 0x0F;
    uint32_t n2 = (u >> 8)  & 0x0F;
    uint32_t n3 = (u >> 12) & 0x0F;
    uint32_t n4 = (u >> 16) & 0x0F;
    uint32_t n5 = (u >> 20) & 0x0F;
    uint32_t n6 = (u >> 24) & 0x0F;
    uint32_t n7 =  u >> 28;

    __half2 h01 = __floats2half2_rn(E2M1Value(n0) * scale, E2M1Value(n1) * scale);
    __half2 h23 = __floats2half2_rn(E2M1Value(n2) * scale, E2M1Value(n3) * scale);
    __half2 h45 = __floats2half2_rn(E2M1Value(n4) * scale, E2M1Value(n5) * scale);
    __half2 h67 = __floats2half2_rn(E2M1Value(n6) * scale, E2M1Value(n7) * scale);

    w[0] = half2_raw_bits(h01);
    w[1] = half2_raw_bits(h23);
    w[2] = half2_raw_bits(h45);
    w[3] = half2_raw_bits(h67);
}

template<int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM)
__global__ void dequant_shared_nvfp4_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float                        global_scale,
    __half*       __restrict__ output,
    int64_t                     num_elements)
{
    constexpr int ELEMS_PER_THREAD = 32;   // 4 × uint4, 2 scale blocks
    constexpr int ELEMS_PER_BLOCK  = BLOCK_DIM * ELEMS_PER_THREAD;

    const int64_t chunk = static_cast<int64_t>(blockIdx.x) * ELEMS_PER_BLOCK;
    if (chunk >= num_elements) return;

    const int64_t base = chunk + static_cast<int64_t>(threadIdx.x) * ELEMS_PER_THREAD;

    // ---- Fast path: the thread's 32 elements are fully in range ----
    if (base + ELEMS_PER_THREAD <= num_elements) {
        const uint4 p = *reinterpret_cast<const uint4*>(packed + (base >> 1));
        // Two E4M3 scale descriptors for elems [base, base+32)
        const uint16_t s2 = *reinterpret_cast<const uint16_t*>(scales + (base >> 4));
        const float scl_lo = nvfp4::E4M3ScaleToFloat(static_cast<uint8_t>(s2)) * global_scale;
        const float scl_hi = nvfp4::E4M3ScaleToFloat(static_cast<uint8_t>(s2 >> 8)) * global_scale;

        uint32_t w[16];
        unpack_nibbles8(p.x, scl_lo, &w[0]);
        unpack_nibbles8(p.y, scl_lo, &w[4]);
        unpack_nibbles8(p.z, scl_hi, &w[8]);
        unpack_nibbles8(p.w, scl_hi, &w[12]);

        uint4* out4 = reinterpret_cast<uint4*>(output + base);
        out4[0] = make_uint4(w[0], w[1], w[2], w[3]);
        out4[1] = make_uint4(w[4], w[5], w[6], w[7]);
        out4[2] = make_uint4(w[8], w[9], w[10], w[11]);
        out4[3] = make_uint4(w[12], w[13], w[14], w[15]);
        return;
    }

    // ---- Ragged tail: scalar fallback with per-element bounds ----
    for (int64_t i = base; i < num_elements; ++i) {
        const int64_t byte_idx = i >> 1;
        const uint8_t nibble = (i & 1)
            ? (packed[byte_idx] >> 4)
            : (packed[byte_idx] & 0x0F);
        const float scale = nvfp4::E4M3ScaleToFloat(scales[i >> 4]) * global_scale;
        output[i] = __float2half(E2M1Value(nibble) * scale);
    }
}

// Routed-expert NVFP4 dequant: E2M1 nibbles (2 elems/byte) with a paired
// F8_E8M0 scale per 32 elements. Row-major layout [rows, cols], packed bytes
// [rows, cols/2], scales [rows, cols/32]; cols is always a multiple of 32, so
// the flat index i maps to scale[i >> 5] without row-boundary crossings.
template<int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM)
__global__ void dequant_fp4_e8m0_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    __half*        __restrict__ output,
    int64_t                     num_elements)
{
    constexpr int ELEMS_PER_THREAD = 32;   // exactly one F8_E8M0 scale block
    constexpr int ELEMS_PER_BLOCK  = BLOCK_DIM * ELEMS_PER_THREAD;

    const int64_t chunk = static_cast<int64_t>(blockIdx.x) * ELEMS_PER_BLOCK;
    if (chunk >= num_elements) return;

    const int64_t base = chunk + static_cast<int64_t>(threadIdx.x) * ELEMS_PER_THREAD;

    if (base + ELEMS_PER_THREAD <= num_elements) {
        const uint4 p = *reinterpret_cast<const uint4*>(packed + (base >> 1));
        const float scl = kernels::E8m0ClampedToFloat(scales[base >> 5]);

        uint32_t w[16];
        unpack_nibbles8(p.x, scl, &w[0]);
        unpack_nibbles8(p.y, scl, &w[4]);
        unpack_nibbles8(p.z, scl, &w[8]);
        unpack_nibbles8(p.w, scl, &w[12]);

        uint4* out4 = reinterpret_cast<uint4*>(output + base);
        out4[0] = make_uint4(w[0], w[1], w[2], w[3]);
        out4[1] = make_uint4(w[4], w[5], w[6], w[7]);
        out4[2] = make_uint4(w[8], w[9], w[10], w[11]);
        out4[3] = make_uint4(w[12], w[13], w[14], w[15]);
        return;
    }

    for (int64_t i = base; i < num_elements; ++i) {
        const int64_t byte_idx = i >> 1;
        const uint8_t nibble = (i & 1)
            ? (packed[byte_idx] >> 4)
            : (packed[byte_idx] & 0x0F);
        output[i] = __float2half(E2M1Value(nibble)
                                 * kernels::E8m0ClampedToFloat(scales[i >> 5]));
    }
}

// F8_E4M3 × F8_E8M0 dequant kernel (per-128x128 tile scales).
// Same format as attention weights. 1 byte per element (not packed).
// Vectorized fast path: 16 elements per thread via one uint4 byte load.
// C is a multiple of 128, so a 16-element group never crosses a row or a
// 128-column scale-tile boundary — r/sr/sc are loop-invariant per thread.
__global__ void dequant_f8_e8m0_kernel(
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ scales,
    __half* __restrict__ out, int64_t n,
    int64_t C, int64_t SC, int64_t block_r, int64_t block_c)
{
    constexpr int ELEMS = 16;
    const int64_t base = (static_cast<int64_t>(blockIdx.x) * blockDim.x
                          + threadIdx.x) * ELEMS;
    if (base + ELEMS <= n) {
        const int64_t r = base / C;
        const int64_t c = base - r * C;          // multiple of 16
        const int64_t sr = r / block_r;
        const int64_t sc = c / block_c;
        const float sv = kernels::E8m0ClampedToFloat(scales[sr * SC + sc]);
        const uint4 ww = *reinterpret_cast<const uint4*>(w + base);
        const uint8_t bytes[ELEMS] = {
            uint8_t(ww.x & 0xFF), uint8_t((ww.x >> 8) & 0xFF),
            uint8_t((ww.x >> 16) & 0xFF), uint8_t(ww.x >> 24),
            uint8_t(ww.y & 0xFF), uint8_t((ww.y >> 8) & 0xFF),
            uint8_t((ww.y >> 16) & 0xFF), uint8_t(ww.y >> 24),
            uint8_t(ww.z & 0xFF), uint8_t((ww.z >> 8) & 0xFF),
            uint8_t((ww.z >> 16) & 0xFF), uint8_t(ww.z >> 24),
            uint8_t(ww.w & 0xFF), uint8_t((ww.w >> 8) & 0xFF),
            uint8_t((ww.w >> 16) & 0xFF), uint8_t(ww.w >> 24)};
        __half vals[ELEMS];
        #pragma unroll
        for (int j = 0; j < ELEMS; ++j)
            vals[j] = __float2half(kernels::F8E4m3ToFloat(bytes[j]) * sv);
        uint4* out4 = reinterpret_cast<uint4*>(out + base);
        out4[0] = *reinterpret_cast<const uint4*>(vals);
        out4[1] = *reinterpret_cast<const uint4*>(vals + 8);
        return;
    }
    for (int64_t i = base; i < n; ++i) {
        int64_t r = i / C;
        int64_t c = i % C;
        int64_t sr = r / block_r;
        int64_t sc = c / block_c;
        float v = kernels::F8E4m3ToFloat(w[i]) * kernels::E8m0ClampedToFloat(scales[sr * SC + sc]);
        out[i] = __float2half(v);
    }
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Host-side zero-copy Arrow ingest
//------------------------------------------------------------------------------
arrow::Status load_shared_experts_metadata(
    const std::shared_ptr<arrow::io::MemoryMappedFile>& mmap,
    SharedExpertDb& db)
{
    if (!mmap)
        return arrow::Status::Invalid("shared_experts: null MemoryMappedFile");

    db = SharedExpertDb{};
    ARROW_ASSIGN_OR_RAISE(int64_t file_size, mmap->GetSize());
    db.file_size = file_size;

    // Base of the mapped region — ReadAt(0,1) returns a zero-copy buffer
    // pointing at the start of the mapping.
    ARROW_ASSIGN_OR_RAISE(auto base_buf, mmap->ReadAt(0, 1));
    db.file_base = base_buf->data();

    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap.get());
    if (!reader_res.ok()) return reader_res.status();
    auto reader = *reader_res;
    if (reader->num_record_batches() == 0)
        return arrow::Status::Invalid("shared_experts: empty file (no batches)");
    auto batch_res = reader->ReadRecordBatch(0);
    if (!batch_res.ok()) return batch_res.status();
    auto batch = *batch_res;

    if (batch->num_columns() != SharedExpertSchema::kNumFields)
        return arrow::Status::Invalid(
            "shared_experts: expected 9 columns, got ",
            std::to_string(batch->num_columns()));

    const int64_t n = batch->num_rows();
    if (n <= 0)
        return arrow::Status::Invalid("shared_experts: no rows");

    auto w1 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        batch->column(SharedExpertSchema::kIdxW1Weight));
    auto w2 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        batch->column(SharedExpertSchema::kIdxW2Weight));
    auto w3 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        batch->column(SharedExpertSchema::kIdxW3Weight));
    auto s1 = std::dynamic_pointer_cast<arrow::ListArray>(
        batch->column(SharedExpertSchema::kIdxW1Scale));
    auto s2 = std::dynamic_pointer_cast<arrow::ListArray>(
        batch->column(SharedExpertSchema::kIdxW2Scale));
    auto s3 = std::dynamic_pointer_cast<arrow::ListArray>(
        batch->column(SharedExpertSchema::kIdxW3Scale));
    auto gs1 = std::dynamic_pointer_cast<arrow::FloatArray>(
        batch->column(SharedExpertSchema::kIdxW1GlobalScale));
    auto gs2 = std::dynamic_pointer_cast<arrow::FloatArray>(
        batch->column(SharedExpertSchema::kIdxW2GlobalScale));
    auto gs3 = std::dynamic_pointer_cast<arrow::FloatArray>(
        batch->column(SharedExpertSchema::kIdxW3GlobalScale));

    if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3 || !gs1 || !gs2 || !gs3)
        return arrow::Status::TypeError(
            "shared_experts: column type mismatch");

    db.w1.resize(static_cast<size_t>(n));
    db.w2.resize(static_cast<size_t>(n));
    db.w3.resize(static_cast<size_t>(n));

    auto extract_weights = [&](const std::shared_ptr<arrow::LargeBinaryArray>& arr,
                               std::vector<SharedExpertDescriptor>& out,
                               int64_t row, int layer) -> arrow::Status {
        auto view = arr->GetView(row);
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(view.data());
        if (view.size() > 0 &&
            (ptr < db.file_base || ptr + view.size() > db.file_base + db.file_size))
            return arrow::Status::Invalid(
                "shared_experts: packed buffer not backed by the mmap "
                "(zero-copy read required)");
        out[static_cast<size_t>(row)].layer         = layer;
        out[static_cast<size_t>(row)].host_ptr      = ptr;
        out[static_cast<size_t>(row)].packed_offset = ptr - db.file_base;
        out[static_cast<size_t>(row)].packed_bytes  = view.size();
        out[static_cast<size_t>(row)].num_elements  = view.size() * 2;
        return arrow::Status::OK();
    };

    auto extract_scales = [&](const std::shared_ptr<arrow::ListArray>& lst,
                              std::vector<SharedExpertDescriptor>& out,
                              int64_t row) -> arrow::Status {
        auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
        auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
        if (!u8 || !offs)
            return arrow::Status::TypeError(
                "shared_experts: scale column not list<uint8>");
        ARROW_RETURN_NOT_OK(arrow_check::CheckListSlice(
            *offs, row, u8->length(), "shared-expert scale list",
            std::to_string(row)));
        int64_t begin = offs->Value(row);
        int64_t end   = offs->Value(row + 1);
        const uint8_t* ptr = u8->raw_values() + begin;
        if (end > begin &&
            (ptr < db.file_base || ptr + (end - begin) > db.file_base + db.file_size))
            return arrow::Status::Invalid(
                "shared_experts: scale buffer not backed by the mmap");
        out[static_cast<size_t>(row)].scale_host_ptr = ptr;
        out[static_cast<size_t>(row)].scale_offset   = ptr - db.file_base;
        out[static_cast<size_t>(row)].scale_bytes    = end - begin;
        return arrow::Status::OK();
    };

    for (int64_t i = 0; i < n; ++i) {
        ARROW_RETURN_NOT_OK(extract_weights(w1, db.w1, i, static_cast<int>(i)));
        ARROW_RETURN_NOT_OK(extract_weights(w2, db.w2, i, static_cast<int>(i)));
        ARROW_RETURN_NOT_OK(extract_weights(w3, db.w3, i, static_cast<int>(i)));
        ARROW_RETURN_NOT_OK(extract_scales(s1, db.w1, i));
        ARROW_RETURN_NOT_OK(extract_scales(s2, db.w2, i));
        ARROW_RETURN_NOT_OK(extract_scales(s3, db.w3, i));
        ARROW_RETURN_NOT_OK(arrow_check::CheckExpertScaleCount(
            db.w1[static_cast<size_t>(i)].packed_bytes,
            db.w1[static_cast<size_t>(i)].scale_bytes, kVFP4BlockSize,
            "shared_experts.w1[" + std::to_string(i) + "]"));
        ARROW_RETURN_NOT_OK(arrow_check::CheckExpertScaleCount(
            db.w2[static_cast<size_t>(i)].packed_bytes,
            db.w2[static_cast<size_t>(i)].scale_bytes, kVFP4BlockSize,
            "shared_experts.w2[" + std::to_string(i) + "]"));
        ARROW_RETURN_NOT_OK(arrow_check::CheckExpertScaleCount(
            db.w3[static_cast<size_t>(i)].packed_bytes,
            db.w3[static_cast<size_t>(i)].scale_bytes, kVFP4BlockSize,
            "shared_experts.w3[" + std::to_string(i) + "]"));
        db.w1[static_cast<size_t>(i)].global_scale = gs1->Value(i);
        db.w2[static_cast<size_t>(i)].global_scale = gs2->Value(i);
        db.w3[static_cast<size_t>(i)].global_scale = gs3->Value(i);
    }

    db.mmap = mmap;   // keep the mapping alive for the cached host pointers
    return arrow::Status::OK();
}

//------------------------------------------------------------------------------
// Dequant launch wrapper
//------------------------------------------------------------------------------
cudaError_t dequant_shared_nvfp4(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    float global_scale,
    __half* d_output, int64_t num_elements)
{
    if (!d_packed || !d_scales || !d_output)
        return cudaErrorInvalidValue;
    if (num_elements <= 0 || packed_bytes <= 0 || scales_bytes <= 0)
        return cudaErrorInvalidValue;

    const int64_t need_packed = (num_elements + 1) / 2;
    const int64_t need_scales = (num_elements + kVFP4BlockSize - 1) / kVFP4BlockSize;
    if (packed_bytes < need_packed)
        return cudaErrorInvalidValue;   // packed buffer too small
    if (scales_bytes < need_scales)
        return cudaErrorInvalidValue;   // scale descriptor buffer too small

    constexpr int BLOCK_DIM = 256;
    constexpr int ELEMS_PER_THREAD = 32;
    const int64_t elems_per_block = static_cast<int64_t>(BLOCK_DIM) * ELEMS_PER_THREAD;
    const int blocks = static_cast<int>(
        (num_elements + elems_per_block - 1) / elems_per_block);

    dequant_shared_nvfp4_kernel<BLOCK_DIM><<<blocks, BLOCK_DIM, 0, stream>>>(
        d_packed, d_scales, global_scale, d_output, num_elements);

    return cudaGetLastError();
}

// FP32 output version of the NVFP4 (E2M1 nibble + E4M3 per-16-block scale +
// global_scale) dequant kernel. NOT currently used at runtime: verified
// against the real DeepSeek-V4-Flash safetensors headers that expert
// weights actually go through converter.cpp's F8_E4M3/I8-passthrough path
// (F8_E8M0-tiled scales, not E4M3-per-16 + global_scale), so this format
// only applies to weights that go through fresh GPU NVFP4 quantization.
// Kept for that case and to mirror the existing (also currently-unused)
// half-precision dequant_shared_nvfp4 above.
template<int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM)
__global__ void dequant_shared_nvfp4_f32_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float                        global_scale,
    float*        __restrict__ output,
    int64_t                     num_elements)
{
    constexpr int ELEMS_PER_THREAD = 32;
    constexpr int ELEMS_PER_BLOCK  = BLOCK_DIM * ELEMS_PER_THREAD;
    const int64_t chunk = static_cast<int64_t>(blockIdx.x) * ELEMS_PER_BLOCK;
    if (chunk >= num_elements) return;
    const int64_t base = chunk + static_cast<int64_t>(threadIdx.x) * ELEMS_PER_THREAD;
    const int64_t end = min(base + static_cast<int64_t>(ELEMS_PER_THREAD), num_elements);

    for (int64_t i = base; i < end; ++i) {
        const int64_t byte_idx = i >> 1;
        const uint8_t nibble = (i & 1)
            ? (packed[byte_idx] >> 4)
            : (packed[byte_idx] & 0x0F);
        const float scale = nvfp4::E4M3ScaleToFloat(scales[i >> 4]) * global_scale;
        output[i] = E2M1Value(nibble) * scale;
    }
}

// FP32-output NVFP4 dequant wrapper (E2M1 + E4M3-per-16 + global_scale).
cudaError_t dequant_shared_nvfp4_f32(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    float global_scale,
    float* d_output, int64_t num_elements)
{
    if (!d_packed || !d_scales || !d_output)
        return cudaErrorInvalidValue;
    if (num_elements <= 0 || packed_bytes <= 0 || scales_bytes <= 0)
        return cudaErrorInvalidValue;

    const int64_t need_packed = (num_elements + 1) / 2;
    const int64_t need_scales = (num_elements + kVFP4BlockSize - 1) / kVFP4BlockSize;
    if (packed_bytes < need_packed)
        return cudaErrorInvalidValue;
    if (scales_bytes < need_scales)
        return cudaErrorInvalidValue;

    constexpr int BLOCK_DIM = 256;
    constexpr int ELEMS_PER_THREAD = 32;
    const int64_t elems_per_block = static_cast<int64_t>(BLOCK_DIM) * ELEMS_PER_THREAD;
    const int blocks = static_cast<int>(
        (num_elements + elems_per_block - 1) / elems_per_block);

    dequant_shared_nvfp4_f32_kernel<BLOCK_DIM><<<blocks, BLOCK_DIM, 0, stream>>>(
        d_packed, d_scales, global_scale, d_output, num_elements);

    return cudaGetLastError();
}

// FP32 output version of the fp4 dequant kernel
template<int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM)
__global__ void dequant_fp4_e8m0_f32_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float*          __restrict__ output,
    int64_t                       num_elements)
{
    constexpr int ELEMS_PER_THREAD = 32;
    constexpr int ELEMS_PER_BLOCK  = BLOCK_DIM * ELEMS_PER_THREAD;
    const int64_t chunk = static_cast<int64_t>(blockIdx.x) * ELEMS_PER_BLOCK;
    if (chunk >= num_elements) return;
    const int64_t base = chunk + static_cast<int64_t>(threadIdx.x) * ELEMS_PER_THREAD;

    if (base + ELEMS_PER_THREAD <= num_elements) {
        const uint4 p = *reinterpret_cast<const uint4*>(packed + (base >> 1));
        const float scl = kernels::E8m0ClampedToFloat(scales[base >> 5]);
        // Unpack 32 nibbles into 32 floats
        uint32_t u = p.x;
        for (int i = 0; i < 8; ++i) {
            output[base + i] = E2M1Value(u & 0x0F) * scl;
            u >>= 4;
        }
        u = p.y;
        for (int i = 8; i < 16; ++i) {
            output[base + i] = E2M1Value(u & 0x0F) * scl;
            u >>= 4;
        }
        u = p.z;
        for (int i = 16; i < 24; ++i) {
            output[base + i] = E2M1Value(u & 0x0F) * scl;
            u >>= 4;
        }
        u = p.w;
        for (int i = 24; i < 32; ++i) {
            output[base + i] = E2M1Value(u & 0x0F) * scl;
            u >>= 4;
        }
        return;
    }
    for (int64_t i = base; i < num_elements; ++i) {
        const int64_t byte_idx = i >> 1;
        const uint8_t nibble = (i & 1)
            ? (packed[byte_idx] >> 4)
            : (packed[byte_idx] & 0x0F);
        output[i] = E2M1Value(nibble) * kernels::E8m0ClampedToFloat(scales[i >> 5]);
    }
}

// FP32 output version of the F8 dequant kernel (vectorized, see the __half
// variant above for the access-pattern rationale).
__global__ void dequant_f8_e8m0_f32_kernel(
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ scales,
    float* __restrict__ out, int64_t n,
    int64_t C, int64_t SC, int64_t block_r, int64_t block_c)
{
    constexpr int ELEMS = 16;
    const int64_t base = (static_cast<int64_t>(blockIdx.x) * blockDim.x
                          + threadIdx.x) * ELEMS;
    if (base + ELEMS <= n) {
        const int64_t r = base / C;
        const int64_t c = base - r * C;
        const int64_t sr = r / block_r;
        const int64_t sc = c / block_c;
        const float sv = kernels::E8m0ClampedToFloat(scales[sr * SC + sc]);
        const uint4 ww = *reinterpret_cast<const uint4*>(w + base);
        const uint8_t bytes[ELEMS] = {
            uint8_t(ww.x & 0xFF), uint8_t((ww.x >> 8) & 0xFF),
            uint8_t((ww.x >> 16) & 0xFF), uint8_t(ww.x >> 24),
            uint8_t(ww.y & 0xFF), uint8_t((ww.y >> 8) & 0xFF),
            uint8_t((ww.y >> 16) & 0xFF), uint8_t(ww.y >> 24),
            uint8_t(ww.z & 0xFF), uint8_t((ww.z >> 8) & 0xFF),
            uint8_t((ww.z >> 16) & 0xFF), uint8_t(ww.z >> 24),
            uint8_t(ww.w & 0xFF), uint8_t((ww.w >> 8) & 0xFF),
            uint8_t((ww.w >> 16) & 0xFF), uint8_t(ww.w >> 24)};
        float vals[ELEMS];
        #pragma unroll
        for (int j = 0; j < ELEMS; ++j)
            vals[j] = kernels::F8E4m3ToFloat(bytes[j]) * sv;
        float4* out4 = reinterpret_cast<float4*>(out + base);
        out4[0] = *reinterpret_cast<const float4*>(vals);
        out4[1] = *reinterpret_cast<const float4*>(vals + 4);
        out4[2] = *reinterpret_cast<const float4*>(vals + 8);
        out4[3] = *reinterpret_cast<const float4*>(vals + 12);
        return;
    }
    for (int64_t i = base; i < n; ++i) {
        int64_t r = i / C;
        int64_t c = i % C;
        int64_t sr = r / block_r;
        int64_t sc = c / block_c;
        out[i] = kernels::F8E4m3ToFloat(w[i]) * kernels::E8m0ClampedToFloat(scales[sr * SC + sc]);
    }
}

// Routed-expert fp4 dequant (FP32 output — no FP16 overflow)
cudaError_t dequant_fp4_e8m0(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    float* d_output, int64_t num_elements)
{
    if (!d_packed || !d_scales || !d_output)
        return cudaErrorInvalidValue;
    if (num_elements <= 0 || packed_bytes <= 0 || scales_bytes <= 0)
        return cudaErrorInvalidValue;
    const int64_t need_packed = (num_elements + 1) / 2;
    const int64_t need_scales = (num_elements + 31) / 32;
    if (packed_bytes < need_packed) return cudaErrorInvalidValue;
    if (scales_bytes < need_scales) return cudaErrorInvalidValue;
    constexpr int BLOCK_DIM = 256;
    constexpr int ELEMS_PER_THREAD = 32;
    const int64_t elems_per_block = static_cast<int64_t>(BLOCK_DIM) * ELEMS_PER_THREAD;
    const int blocks = static_cast<int>(
        (num_elements + elems_per_block - 1) / elems_per_block);
    dequant_fp4_e8m0_f32_kernel<BLOCK_DIM><<<blocks, BLOCK_DIM, 0, stream>>>(
        d_packed, d_scales, d_output, num_elements);
    return cudaGetLastError();
}

// Legacy FP16 version (kept for compatibility)
cudaError_t dequant_fp4_e8m0_half(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    __half* d_output, int64_t num_elements)
{
    if (!d_packed || !d_scales || !d_output)
        return cudaErrorInvalidValue;
    if (num_elements <= 0 || packed_bytes <= 0 || scales_bytes <= 0)
        return cudaErrorInvalidValue;
    const int64_t need_packed = (num_elements + 1) / 2;
    const int64_t need_scales = (num_elements + 31) / 32;
    if (packed_bytes < need_packed) return cudaErrorInvalidValue;
    if (scales_bytes < need_scales) return cudaErrorInvalidValue;
    constexpr int BLOCK_DIM = 256;
    constexpr int ELEMS_PER_THREAD = 32;
    const int64_t elems_per_block = static_cast<int64_t>(BLOCK_DIM) * ELEMS_PER_THREAD;
    const int blocks = static_cast<int>(
        (num_elements + elems_per_block - 1) / elems_per_block);
    dequant_fp4_e8m0_kernel<BLOCK_DIM><<<blocks, BLOCK_DIM, 0, stream>>>(
        d_packed, d_scales, d_output, num_elements);
    return cudaGetLastError();
}

// ── E8M0 → UE4M3 repack (TC W4A4 prep, F-12) ──────────────────────────────
// One output byte per 16 elements: output block j covers elements
// [16j, 16j+16) and reads source scale (16j) >> 5 == j >> 1 — both halves of
// the 32-element E8M0 block share one source byte (cols % 32 == 0 keeps row
// boundaries out of the flat mapping).
__global__ void repack_e8m0_ue4m3_kernel(const uint8_t* __restrict__ src,
                                         int64_t n_out,
                                         uint8_t* __restrict__ dst)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= n_out) return;
    const unsigned e8 = src[i >> 1];
    // Repo pow2-E4M3 byte layout: exponent in bits [6:3], decode
    // 2^(((b >> 3) & 15) - 7) (nvfp4_codec.h E4M3ScaleToFloat). Map the
    // E8M0 exponent e8 onto it: e4 = e8 - 120, stored shifted << 3.
    const unsigned e4 = e8 > 135u ? 15u
                      : (e8 > 120u ? e8 - 120u : 0u);
    dst[i] = static_cast<uint8_t>(e4 << 3);
}

cudaError_t RepackE8m0ToUe4m3(cudaStream_t stream,
                              const uint8_t *d_e8m0, int64_t num_elements,
                              uint8_t *d_ue4m3)
{
    if (!d_e8m0 || !d_ue4m3 || num_elements <= 0 || (num_elements & 31))
        return cudaErrorInvalidValue;
    const int64_t n_out = num_elements / 16;
    const int block = 256;
    const int blocks = static_cast<int>((n_out + block - 1) / block);
    repack_e8m0_ue4m3_kernel<<<blocks, block, 0, stream>>>(d_e8m0, n_out,
                                                           d_ue4m3);
    return cudaGetLastError();
}

bool ValidateE8m0Range(const uint8_t *h_scales, int64_t scale_bytes,
                       int64_t *bad_offset)
{
    if (!h_scales || scale_bytes <= 0) return false;
    for (int64_t i = 0; i < scale_bytes; ++i) {
        if (h_scales[i] > 135) {
            if (bad_offset) *bad_offset = i;
            return false;
        }
    }
    return true;
}

// FP16 + fused per-128 activation scale (see shared_experts.h).
// Kernel definition lives at the bottom of this file (after the F8 kernels).
template<int BLOCK_DIM>
__global__ void dequant_fp4_e8m0_actscale_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    const float* __restrict__ act_scales,
    int K,
    __half* __restrict__ output,
    int64_t num_elements);

cudaError_t dequant_fp4_e8m0_half_actscale(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    const float* d_act_scales, int K,
    __half* d_output, int64_t num_elements)
{
    if (!d_packed || !d_scales || !d_act_scales || !d_output)
        return cudaErrorInvalidValue;
    if (num_elements <= 0 || packed_bytes <= 0 || scales_bytes <= 0)
        return cudaErrorInvalidValue;
    if (K <= 0 || (K % 128) != 0)
        return cudaErrorInvalidValue;
    const int64_t need_packed = (num_elements + 1) / 2;
    const int64_t need_scales = (num_elements + 31) / 32;
    if (packed_bytes < need_packed) return cudaErrorInvalidValue;
    if (scales_bytes < need_scales) return cudaErrorInvalidValue;
    constexpr int BLOCK_DIM = 256;
    constexpr int ELEMS_PER_THREAD = 32;
    const int64_t elems_per_block = static_cast<int64_t>(BLOCK_DIM) * ELEMS_PER_THREAD;
    const int blocks = static_cast<int>(
        (num_elements + elems_per_block - 1) / elems_per_block);
    dequant_fp4_e8m0_actscale_kernel<BLOCK_DIM><<<blocks, BLOCK_DIM, 0, stream>>>(
        d_packed, d_scales, d_act_scales, K, d_output, num_elements);
    return cudaGetLastError();
}

// Shared-expert F8 dequant (FP32 output)
cudaError_t dequant_shared_f8(
    cudaStream_t stream,
    const uint8_t* d_weights, int64_t n_bytes,
    const uint8_t* d_scales, int64_t scale_bytes,
    float* d_output, int64_t num_elements, int64_t cols)
{
    if (!d_weights || !d_scales || !d_output)
        return cudaErrorInvalidValue;
    if (num_elements <= 0 || n_bytes <= 0)
        return cudaErrorInvalidValue;
    if (n_bytes < num_elements)
        return cudaErrorInvalidValue;
    int block = 256;
    // Vectorized kernel: 16 elements per thread.
    int grid = static_cast<int>((num_elements + block * 16 - 1) / (block * 16));
    dequant_f8_e8m0_f32_kernel<<<grid, block, 0, stream>>>(
        d_weights, d_scales, d_output, num_elements,
        cols, cols / 128, 128, 128);
    return cudaGetLastError();
}

// Legacy FP16 version
cudaError_t dequant_shared_f8_half(
    cudaStream_t stream,
    const uint8_t* d_weights, int64_t n_bytes,
    const uint8_t* d_scales, int64_t scale_bytes,
    __half* d_output, int64_t num_elements, int64_t cols)
{
    if (!d_weights || !d_scales || !d_output)
        return cudaErrorInvalidValue;
    if (num_elements <= 0 || n_bytes <= 0)
        return cudaErrorInvalidValue;
    if (n_bytes < num_elements)
        return cudaErrorInvalidValue;
    int block = 256;
    // Vectorized kernel: 16 elements per thread.
    int grid = static_cast<int>((num_elements + block * 16 - 1) / (block * 16));
    dequant_f8_e8m0_kernel<<<grid, block, 0, stream>>>(
        d_weights, d_scales, d_output, num_elements,
        cols, cols / 128, 128, 128);
    return cudaGetLastError();
}

// FP16-output routed-expert dequant with the per-128 activation scale fused
// in (see shared_experts.h). Same vectorized structure as dequant_fp4_e8m0_kernel.
template<int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM)
__global__ void dequant_fp4_e8m0_actscale_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    const float* __restrict__ act_scales,
    int K,
    __half* __restrict__ output,
    int64_t num_elements)
{
    constexpr int ELEMS_PER_THREAD = 32;   // exactly one F8_E8M0 scale block
    constexpr int ELEMS_PER_BLOCK  = BLOCK_DIM * ELEMS_PER_THREAD;

    const int64_t chunk = static_cast<int64_t>(blockIdx.x) * ELEMS_PER_BLOCK;
    if (chunk >= num_elements) return;

    const int64_t base = chunk + static_cast<int64_t>(threadIdx.x) * ELEMS_PER_THREAD;

    if (base + ELEMS_PER_THREAD <= num_elements) {
        const uint4 p = *reinterpret_cast<const uint4*>(packed + (base >> 1));
        const float scl = kernels::E8m0ClampedToFloat(scales[base >> 5]);
        // K % 128 == 0 and base % 32 == 0 → the whole 32-element group shares
        // one activation scale (worst case k0%128 == 96, 96+31 == 127).
        const float ascl = act_scales[(base % K) >> 7];

        uint32_t w[16];
        unpack_nibbles8(p.x, scl * ascl, &w[0]);
        unpack_nibbles8(p.y, scl * ascl, &w[4]);
        unpack_nibbles8(p.z, scl * ascl, &w[8]);
        unpack_nibbles8(p.w, scl * ascl, &w[12]);

        uint4* out4 = reinterpret_cast<uint4*>(output + base);
        out4[0] = make_uint4(w[0], w[1], w[2], w[3]);
        out4[1] = make_uint4(w[4], w[5], w[6], w[7]);
        out4[2] = make_uint4(w[8], w[9], w[10], w[11]);
        out4[3] = make_uint4(w[12], w[13], w[14], w[15]);
        return;
    }

    for (int64_t i = base; i < num_elements; ++i) {
        const int64_t byte_idx = i >> 1;
        const uint8_t nibble = (i & 1)
            ? (packed[byte_idx] >> 4)
            : (packed[byte_idx] & 0x0F);
        output[i] = __float2half(E2M1Value(nibble)
                                 * kernels::E8m0ClampedToFloat(scales[i >> 5])
                                 * act_scales[(i % K) >> 7]);
    }
}

} // namespace gpu
} // namespace lsxcommon
