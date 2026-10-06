// Offline dequant of the PRE-QUANTIZED Nemotron-3.5-Lightning source
// (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4). See
// src/nemotron_moe_nvfp4_dequant.h for the contract and the measured
// justification.
//
// The one thing this file must never do is reuse the repo's runtime NVFP4
// decoder (`nvfp4_codec.h::E4M3ScaleToFloat`) on the SOURCE block scales: that
// helper reads the e4 exponent field only, while ModelOpt's W4A16_NVFP4 writes
// a COMPLETE OCP E4M3 with all three mantissa bits in play. On the real tensor
// backbone.layers.1.mixer.experts.0.down_proj that mistake costs ~30% relative
// L2 on the expert weights, so the kernels below do the full E4M3 decode via
// cuda_fp8 and only the CONVERTER's output (which goes through
// quantize_tensor_gpu) is in the repo's pow2 flavor.
#include "nemotron_moe_nvfp4_dequant.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace lsxcommon {
namespace nemotron_nvfp4 {
namespace {

// E2M1 nibble decode. Bit-identical to nvfp4_codec.h::E2M1Value (the same
// {0, .5, 1, 1.5, 2, 3, 4, 6} level table, sign in bit 3), re-expressed here so
// this translation unit does not depend on the runtime decoder's E4M3 flavor.
__host__ __device__ __forceinline__ float E2m1(int nib)
{
    constexpr float kLv[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kLv[nib & 7];
    return (nib & 8) ? -v : v;
}

// COMPLETE OCP E4M3FN decode (mantissa bits included), host side.
float E4m3Full(uint8_t b)
{
    const float sign = (b >> 7) ? -1.0f : 1.0f;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0) return sign * static_cast<float>(m) * 0.001953125f;  // m*2^-9
    return sign * (1.0f + static_cast<float>(m) / 8.0f) *
           static_cast<float>(1 << (e - 7));
}

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
    // Full E4M3, NOT the repo's pow2-only E4M3ScaleToFloat.
    const float s = __half2float(__half(__nv_cvt_fp8_to_halfraw(
        bscale[i >> 4], __NV_E4M3)));
    out[i] = __float2bfloat16(E2m1(nib) * s * global);
}

__global__ void fp8_source_to_bf16_kernel(const uint8_t *__restrict__ weight,
                                          const float *__restrict__ scale,
                                          __nv_bfloat16 *__restrict__ out,
                                          long long numel)
{
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= numel) return;
    const float v = __half2float(__half(__nv_cvt_fp8_to_halfraw(
        weight[i], __NV_E4M3)));
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
        const float f = E2m1(nib) * E4m3Full(v.block_scale[i >> 4]) *
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
        const float f = E4m3Full(v.weight[i]) * v.weight_scale[0];
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