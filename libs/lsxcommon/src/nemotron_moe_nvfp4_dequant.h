#pragma once

// Offline dequant for the PRE-QUANTIZED Nemotron-3.5-Lightning source
// (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, Model Ingestion Protocol
// stage 2).
//
// Two source encodings must be undone, and NEITHER may be confused with the
// repo's own runtime NVFP4 codec:
//
//   * `NemotronNvfp4ToBf16` — the vault's W4A16_NVFP4: U8 packed E2M1 nibbles
//     + a FULL OCP E4M3 block scale per 16 + an F32 per-tensor weight_scale_2.
//     The block scale carries MANTISSA bits; the repo's runtime decoder
//     (`nvfp4_codec.h::E4M3ScaleToFloat`) reads the exponent field only. This
//     kernel therefore uses the COMPLETE E4M3 decode, never that helper.
//     Measured on the real tensor backbone.layers.1.mixer.experts.0.down_proj:
//     a byte passthrough loses ~30% relative L2 on the expert weights.
//
//   * `NemotronFp8ToBf16` — the vault's per-tensor FP8: F8_E4M3 weights + one
//     SCALAR F32 weight_scale (shape []), plus an F32 input_scale for the
//     activations (which the converter ignores - the converted layout runs
//     FP16 activations).
//
// Both write BF16 and are streaming (one element per thread, grid-stride), so
// the converter's peak device footprint is ONE tensor, never one expert set.

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace nemotron_nvfp4 {

// Logical weight matrix [rows, cols] laid out row-major, exactly as the
// runtime expects (`packed` holds cols/2 bytes per row).
struct Nvfp4SourceView {
    const uint8_t *packed = nullptr;        // U8 nibbles, 2 per byte
    const uint8_t *block_scale = nullptr;   // F8_E4M3, one per 16 elements
    float global_scale = 1.0f;              // F32 weight_scale_2
    int64_t rows = 0;
    int64_t cols = 0;                       // LOGICAL columns (even)
};

struct Fp8SourceView {
    const uint8_t *weight = nullptr;        // F8_E4M3
    const float *weight_scale = nullptr;    // F32, one scalar (per-tensor)
    int64_t rows = 0;
    int64_t cols = 0;
};

// Host-side codec mirror of the two kernels, for tests and for the converter's
// own bounds validation. `out` receives rows*cols BF16 values.
void Nvfp4SourceToBf16Host(const Nvfp4SourceView &v,
                           std::vector<uint8_t> &out_bf16,
                           std::string &error);
void Fp8SourceToBf16Host(const Fp8SourceView &v,
                         std::vector<uint8_t> &out_bf16,
                         std::string &error);

// GPU entry points. `out_bf16` receives rows*cols*2 bytes. Both return false on
// any CUDA failure and leave nothing allocated.
bool Nvfp4SourceToBf16Gpu(const Nvfp4SourceView &v, void *out_bf16,
                          cudaStream_t stream, std::string &error);
bool Fp8SourceToBf16Gpu(const Fp8SourceView &v, void *out_bf16,
                        cudaStream_t stream, std::string &error);

} // namespace nemotron_nvfp4
} // namespace lsxcommon