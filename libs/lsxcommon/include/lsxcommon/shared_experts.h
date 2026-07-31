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

} // namespace gpu
} // namespace lsxcommon
