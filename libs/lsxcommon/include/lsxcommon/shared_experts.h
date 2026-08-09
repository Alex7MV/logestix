#pragma once

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

//==============================================================================
// Shared Expert structural module — Blackwell (CC 12.0)
//
// Two responsibilities:
//   1. Host-side zero-copy Arrow ingest of the NVFP4-packed shared expert
//      weights (per layer), caching exact raw byte offsets + lengths into a
//      structural descriptor registry (SharedExpertDb).
//   2. GPU dequantization of the packed E2M1 nibbles back to FP16 using the
//      per-block E4M3 scale descriptors and the global float multiplier.
//
// No dynamic memory management happens inside the execution loop — the kernel
// only touches pre-allocated VRAM arenas.
//==============================================================================

namespace lsxcommon {
namespace gpu {

// NVFP4 block size (elements per per-block E4M3 scale descriptor).
constexpr int kVFP4BlockSize = 16;

struct SharedExpertDescriptor {
    int      layer = 0;
    // Packed NVFP4 payload (E2M1 nibbles).
    int64_t  packed_offset = 0;    // raw byte offset relative to file start
    int64_t  packed_bytes  = 0;    // packed payload length in bytes
    const uint8_t* host_ptr = nullptr;  // direct pointer into the mmap
    // Per-block E4M3 scale descriptors.
    int64_t  scale_offset = 0;
    int64_t  scale_bytes  = 0;
    const uint8_t* scale_host_ptr = nullptr;
    // Element count derived from the packed payload (2 elems per byte).
    int64_t  num_elements = 0;
    // Global float multiplier (dequant: value = E2M1 * block_scale * gs).
    float    global_scale = 1.0f;
};

struct SharedExpertDb {
    std::vector<SharedExpertDescriptor> w1;   // index == layer
    std::vector<SharedExpertDescriptor> w2;
    std::vector<SharedExpertDescriptor> w3;
    std::shared_ptr<arrow::io::MemoryMappedFile> mmap;  // keeps the mapping alive
    int64_t       file_size = 0;
    const uint8_t* file_base = nullptr;  // start of the mapped region
    std::string   error_msg;

    bool empty() const { return w1.empty() && w2.empty() && w3.empty(); }
    int  num_layers() const { return static_cast<int>(w1.size()); }
};

// Parses a pinned/shared_experts.arrow MemoryMappedFile, extracts the
// per-layer shared expert blocks (w1/w2/w3), and caches raw byte offsets,
// lengths, element counts and global scales into `db`. The `db` keeps the
// mmap alive so the cached host pointers stay valid.
arrow::Status load_shared_experts_metadata(
    const std::shared_ptr<arrow::io::MemoryMappedFile>& mmap,
    SharedExpertDb& db);

// Dequantizes one NVFP4-packed weight tensor:
//   output[i] = E2M1(nibble_i) * E4M3(block_scale) * global_scale
// Returns cudaErrorInvalidValue on null pointers / size mismatches without
// launching. On launch, returns cudaGetLastError().
cudaError_t dequant_shared_nvfp4(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    float global_scale,
    __half* d_output, int64_t num_elements);

// FP32-output variant of the above. NOT currently wired into any live
// dequant path — verified against the real DeepSeek-V4-Flash safetensors
// headers that both routed- and shared-expert weights actually go through
// converter.cpp's passthrough path (F8_E4M3/I8 + F8_E8M0-tiled scale, not
// E4M3-per-16 + global_scale), so this format only applies to weights that
// go through fresh GPU NVFP4 quantization (see quantize_tensor_gpu).
cudaError_t dequant_shared_nvfp4_f32(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    float global_scale,
    float* d_output, int64_t num_elements);

// Dequantizes one routed-expert fp4-packed weight tensor:
//   output[i] = E2M1(nibble_i) * F8_E8M0(scale[i >> 5])
// Row-major [rows, cols] with packed bytes [rows, cols/2] and scales
// [rows, cols/32]. cols must be a multiple of 32. Returns cudaErrorInvalidValue
// on null pointers / size mismatches without launching.
cudaError_t dequant_fp4_e8m0(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    float* d_output, int64_t num_elements);

cudaError_t dequant_fp4_e8m0_half(
    cudaStream_t stream,
    const uint8_t* d_packed, int64_t packed_bytes,
    const uint8_t* d_scales, int64_t scales_bytes,
    __half* d_output, int64_t num_elements);

// Dequantizes one shared-expert F8 weight tensor:
//   output[i] = F8_E4M3(w[i]) * F8_E8M0(scale[r/128, c/128])
// 1 byte per element, per-128x128 tile E8M0 scales (same as attention weights).
// cols = number of columns in the [rows, cols] weight matrix.
cudaError_t dequant_shared_f8(
    cudaStream_t stream,
    const uint8_t* d_weights, int64_t n_bytes,
    const uint8_t* d_scales, int64_t scale_bytes,
    float* d_output, int64_t num_elements, int64_t cols);

cudaError_t dequant_shared_f8_half(
    cudaStream_t stream,
    const uint8_t* d_weights, int64_t n_bytes,
    const uint8_t* d_scales, int64_t scale_bytes,
    __half* d_output, int64_t num_elements, int64_t cols);

} // namespace gpu
} // namespace lsxcommon
