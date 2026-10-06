// Offline dequant of the PRE-QUANTIZED Nemotron-3.5-Lightning source
// (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4). See
// src/nemotron_moe_nvfp4_dequant.h for the contract and the measured
// justification.
//
// The source block scales are COMPLETE E4M3FN — all three mantissa bits, and
// subnormal (e4 field 0) bytes on ~0.16% of the source blocks. The repo's
// canonical codec (`nvfp4_codec.h::E4M3ScaleToFloat`, delegating to
// `nvfp4_scale.h::E4m3ToFloat`) now decodes exactly that value table, so these
// kernels use it directly instead of a private decode. The converter output
// still goes through quantize_tensor_gpu, so the stored tree carries the
// repo's own (E4M3FN) scales.
#include "nemotron_moe_nvfp4_dequant.h"

#include "nvfp4_codec.h"
#include "lsx_kernels_quant.cuh"  // kernels::F8E4m3ToFloat (signed FP8 decode)

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace lsxcommon {
namespace nemotron_nvfp4 {
namespace {

// E2M1 nibble decode and the E4M3FN block-scale decode are the model-neutral
// canonical codecs (single source in nvfp4_codec.h / nvfp4_scale.h); this TU
// only composes them.
using lsxcommon::gpu::nvfp4::E2M1Value;
using lsxcommon::gpu::nvfp4::E4M3ScaleToFloat;

__global__ void nvfp4_source_to_bf16_kernel(const uint8_t *__restrict__ packed,
                                            const uint8_t *__restrict__ bscale,
                                            float global, __nv_bfloat16 *__restrict__ out,
                                            long long numel)
{
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= numel) return;
    // PTX cvt.rn.satfinite.e2m1x2.f32 order: even element -> LOW nibble.
    const uint8_t byte = packed[i >> 1];
    const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
    const float s = E4M3ScaleToFloat(bscale[i >> 4]);
    out[i] = __float2bfloat16(E2M1Value(nib) * s * global);
}

__global__ void fp8_source_to_bf16_kernel(const uint8_t *__restrict__ weight,
                                          const float *__restrict__ scale,
                                          __nv_bfloat16 *__restrict__ out,
                                          long long numel)
{
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= numel) return;
    const float v = lsxcommon::kernels::F8E4m3ToFloat(weight[i]);
    out[i] = __float2bfloat16(v * scale[0]);  // per-tensor FP8: ONE scalar
}

struct DevBuf {
    void *p = nullptr;
    size_t n = 0;
    ~DevBuf() { if (p) cudaFree(p); }
};

bool UploadHost(const void *src, size_t bytes, DevBuf &d, cudaStream_t stream,
                std::string &error, const char *what)
{
    if (bytes == 0) return true;
    if (cudaMalloc(&d.p, bytes) != cudaSuccess) {
        error = std::string("cudaMalloc failed for ") + what;
        return false;
    }
    d.n = bytes;
    if (cudaMemcpyAsync(d.p, src, bytes, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
        error = std::string("H2D copy failed for ") + what;
        return false;
    }
    return true;
}

bool Finish(void *out_bf16, cudaStream_t stream, std::string &error)
{
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        error = "dequant stream sync failed";
        return false;
    }
    return out_bf16 != nullptr;
}

} // namespace

void Nvfp4SourceToBf16Host(const Nvfp4SourceView &v,
                           std::vector<uint8_t> &out_bf16,
                           std::string &error)
{
    error.clear();
    out_bf16.clear();
    if (v.rows <= 0 || v.cols <= 0 || (v.cols & 1) != 0) {
        error = "NVFP4 source geometry must be positive with even cols";
        return;
    }
    if (!v.packed || !v.block_scale) {
        error = "NVFP4 source view is missing packed/block_scale";
        return;
    }
    const int64_t numel = v.rows * v.cols;
    out_bf16.resize(static_cast<size_t>(numel) * 2);
    auto *dst = reinterpret_cast<__nv_bfloat16 *>(out_bf16.data());
    for (int64_t i = 0; i < numel; ++i) {
        const uint8_t byte = v.packed[i >> 1];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        const float f = E2M1Value(nib) * E4M3ScaleToFloat(v.block_scale[i >> 4]) *
                        v.global_scale;
        dst[i] = __float2bfloat16(f);
    }
}

void Fp8SourceToBf16Host(const Fp8SourceView &v,
                         std::vector<uint8_t> &out_bf16,
                         std::string &error)
{
    error.clear();
    out_bf16.clear();
    if (v.rows <= 0 || v.cols <= 0) {
        error = "FP8 source geometry must be positive";
        return;
    }
    if (!v.weight || !v.weight_scale) {
        error = "FP8 source view is missing weight/weight_scale";
        return;
    }
    const int64_t numel = v.rows * v.cols;
    out_bf16.resize(static_cast<size_t>(numel) * 2);
    auto *dst = reinterpret_cast<__nv_bfloat16 *>(out_bf16.data());
    for (int64_t i = 0; i < numel; ++i) {
        const float f =
            lsxcommon::kernels::F8E4m3ToFloat(v.weight[i]) * v.weight_scale[0];
        dst[i] = __float2bfloat16(f);
    }
}

bool Nvfp4SourceToBf16Gpu(const Nvfp4SourceView &v, void *out_bf16,
                          cudaStream_t stream, std::string &error)
{
    error.clear();
    if (v.rows <= 0 || v.cols <= 0 || (v.cols & 1) != 0) {
        error = "NVFP4 source geometry must be positive with even cols";
        return false;
    }
    if (!v.packed || !v.block_scale) {
        error = "NVFP4 source view is missing packed/block_scale";
        return false;
    }
    const int64_t numel = v.rows * v.cols;
    DevBuf d_packed, d_scale;
    if (!UploadHost(v.packed, static_cast<size_t>((numel + 1) / 2), d_packed,
                    stream, error, "NVFP4 packed nibbles") ||
        !UploadHost(v.block_scale, static_cast<size_t>((numel + 15) / 16),
                    d_scale, stream, error, "NVFP4 block scales"))
        return false;
    const int threads = 256;
    const int64_t blocks = (numel + threads - 1) / threads;
    nvfp4_source_to_bf16_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                                  stream>>>(
        static_cast<const uint8_t *>(d_packed.p),
        static_cast<const uint8_t *>(d_scale.p), v.global_scale,
        static_cast<__nv_bfloat16 *>(out_bf16), numel);
    if (cudaGetLastError() != cudaSuccess) {
        error = "nvfp4_source_to_bf16_kernel launch failed";
        return false;
    }
    return Finish(out_bf16, stream, error);
}

bool Fp8SourceToBf16Gpu(const Fp8SourceView &v, void *out_bf16,
                        cudaStream_t stream, std::string &error)
{
    error.clear();
    if (v.rows <= 0 || v.cols <= 0) {
        error = "FP8 source geometry must be positive";
        return false;
    }
    if (!v.weight || !v.weight_scale) {
        error = "FP8 source view is missing weight/weight_scale";
        return false;
    }
    const int64_t numel = v.rows * v.cols;
    DevBuf d_w, d_s;
    if (!UploadHost(v.weight, static_cast<size_t>(numel), d_w, stream, error,
                    "FP8 weights") ||
        !UploadHost(v.weight_scale, sizeof(float), d_s, stream, error,
                    "FP8 per-tensor scale"))
        return false;
    const int threads = 256;
    const int64_t blocks = (numel + threads - 1) / threads;
    fp8_source_to_bf16_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                                stream>>>(
        static_cast<const uint8_t *>(d_w.p), static_cast<const float *>(d_s.p),
        static_cast<__nv_bfloat16 *>(out_bf16), numel);
    if (cudaGetLastError() != cudaSuccess) {
        error = "fp8_source_to_bf16_kernel launch failed";
        return false;
    }
    return Finish(out_bf16, stream, error);
}

} // namespace nemotron_nvfp4
} // namespace lsxcommon