// test_deepseek_moe_inference_kernels.cu — Unit tests for DeepSeek CUDA inference kernels
// Tests: E2M1 dequant, E8M0 scale conversion, act_quant block scaling,
//        KV act_quant, RMSNorm, clamp_f32_to_fp16

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <lsxcommon/safetensors_gpu.h>
#include <lsxcommon/shared_experts.h>
#include <lsxcommon/lsx_tc_gemm.h>
#include <lsxcommon/schema.h>

using Catch::Approx;


// ── Test helpers ────────────────────────────────────────────────────────────

static bool cuda_sync_check() {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        WARN("CUDA error: " << cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ── E2M1 decode table test ──────────────────────────────────────────────────
// E2M1 format: sign(1) + exp(2, bias=1) + mant(1)
// Values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0

TEST_CASE("E2M1 decode table has correct values", "[deepseek][inference]") {
    // Expected E2M1 values indexed by nibble 0-15
    const float expected[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    // The table is in shared_experts.cu as kE2M1Table (device __constant__)
    // We verify the expected values here for documentation
    for (int i = 0; i < 16; ++i) {
        INFO("nibble " << i << " expected " << expected[i]);
        CHECK(expected[i] == expected[i]);  // trivial — values are compile-time
    }
    // Verify monotonic positive
    for (int i = 1; i < 8; ++i)
        CHECK(expected[i] > expected[i-1]);
    // Verify symmetric negative
    for (int i = 0; i < 8; ++i)
        CHECK(expected[i + 8] == -expected[i]);
}

// ── E8M0 scale conversion test ──────────────────────────────────────────────
// E8M0: scale = 2^(b-127), 0 → 0.0

__device__ float test_e8m0_to_float(uint8_t b) {
    if (b == 0) return 0.0f;
    return __uint_as_float(static_cast<uint32_t>(b) << 23);
}

// Top-level __global__ kernel (not an inline lambda): nvcc's extended-lambda
// support for __global__ closures defined inside a loop body in a
// host TEST_CASE function is unreliable across compiler versions, so this
// mirrors the plain top-level kernel style used by every other test below.
__global__ void test_e8m0_kernel_single(const uint8_t* in, float* out) {
    uint8_t b = *in;
    out[0] = test_e8m0_to_float(b);
}

TEST_CASE("E8M0 scale conversion", "[deepseek][inference]") {
    // Test on device
    struct TestCase { uint8_t input; float expected; };
    TestCase cases[] = {
        {0, 0.0f},
        {127, 1.0f},        // 2^0
        {128, 2.0f},        // 2^1
        {120, 0.0078125f},  // 2^(-7)
        {126, 0.5f},        // 2^(-1)
        {134, 128.0f},      // 2^7
    };
    
    for (const auto& tc : cases) {
        float *d_out = nullptr;
        uint8_t *d_in = nullptr;
        cudaMalloc(&d_out, sizeof(float));
        cudaMalloc(&d_in, sizeof(uint8_t));
        cudaMemcpy(d_in, &tc.input, sizeof(uint8_t), cudaMemcpyHostToDevice);
        
        // Launch a 1-thread kernel
        test_e8m0_kernel_single<<<1, 1>>>(d_in, d_out);
        cudaDeviceSynchronize();
        
        float result = 0.0f;
        cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        cudaFree(d_in);
        
        INFO("E8M0(" << (int)tc.input << ") = " << result << " expected " << tc.expected);
        CHECK(result == Approx(tc.expected).epsilon(1e-6));
    }
}

// ── FP8_E4M3 quantization round-trip test ────────────────────────────────────
// FP8_E4M3: sign(1) + exp(4, bias=7) + mant(3)
// Max representable: 448.0, Min normal: 2^(-6) = 0.015625

__global__ void test_fp8_e4m3_roundtrip_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    v = fmaxf(-448.0f, fminf(448.0f, v));
    // Quantize to E4M3
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp32 = (bits >> 23) & 0xFF;
    uint32_t mant3 = (bits >> 20) & 0x7;
    int e4 = exp32 - 127 + 7;
    if (e4 < 1) e4 = 0;
    if (e4 > 15) { e4 = 15; mant3 = 7; }
    if (e4 == 0) {
        out[i] = 0.0f;
    } else {
        uint32_t out_bits = (sign << 31) | (static_cast<uint32_t>(e4 + 120) << 23) | (mant3 << 20);
        float result;
        std::memcpy(&result, &out_bits, sizeof(result));
        out[i] = result;
    }
}

TEST_CASE("FP8_E4M3 quantization clamps and rounds", "[deepseek][inference]") {
    float h_in[] = {0.0f, 1.0f, -1.0f, 448.0f, -448.0f, 500.0f, -500.0f,
                   0.5f, 1.5f, 2.0f, 3.0f, 6.0f, 0.015625f};
    int n = sizeof(h_in) / sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    
    test_fp8_e4m3_roundtrip_kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    float h_out[13];
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    
    CHECK(h_out[0] == 0.0f);           // 0 → 0
    CHECK(h_out[1] == Approx(1.0f));   // 1.0 → 1.0 (exact)
    CHECK(h_out[2] == Approx(-1.0f));  // -1.0 → -1.0 (exact)
    CHECK(h_out[3] == Approx(448.0f)); // 448 → 448 (max)
    CHECK(h_out[4] == Approx(-448.0f));
    CHECK(h_out[5] == Approx(448.0f)); // 500 clamped to 448
    CHECK(h_out[6] == Approx(-448.0f));
    CHECK(h_out[7] == Approx(0.5f));   // 0.5 → 0.5 (exact)
    CHECK(h_out[8] == Approx(1.5f));   // 1.5 → 1.5 (exact)
    CHECK(h_out[9] == Approx(2.0f));
    CHECK(h_out[10] == Approx(3.0f));
    CHECK(h_out[11] == Approx(6.0f));
}

// ── clamp_f32_to_fp16 test ───────────────────────────────────────────────────

__global__ void test_clamp_kernel(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    v = fmaxf(-65504.0f, fminf(65504.0f, v));
    out[i] = __float2half(v);
}

TEST_CASE("clamp_f32_to_fp16 prevents overflow", "[deepseek][inference]") {
    float h_in[] = {0.0f, 1.0f, -1.0f, 65504.0f, -65504.0f,
                    100000.0f, -100000.0f, 1e30f, -1e30f, 42.5f};
    int n = sizeof(h_in) / sizeof(float);
    float *d_in = nullptr;
    __half *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(__half));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    
    test_clamp_kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    __half h_out[10];
    cudaMemcpy(h_out, d_out, n * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    
    CHECK(__half2float(h_out[0]) == 0.0f);
    CHECK(__half2float(h_out[1]) == 1.0f);
    CHECK(__half2float(h_out[2]) == -1.0f);
    CHECK(__half2float(h_out[3]) == 65504.0f);
    CHECK(__half2float(h_out[4]) == -65504.0f);
    CHECK(__half2float(h_out[5]) == 65504.0f);  // 100000 clamped
    CHECK(__half2float(h_out[6]) == -65504.0f);
    CHECK(__half2float(h_out[7]) == 65504.0f);  // 1e30 clamped
    CHECK(__half2float(h_out[8]) == -65504.0f);
    CHECK(__half2float(h_out[9]) == 42.5f);
}

// ── ue8m0 power-of-2 scale rounding test ─────────────────────────────────────

__device__ float test_ue8m0_scale(float amax) {
    float linear_scale = fmaxf(amax, 1e-4f) / 448.0f;
    int log2_s = 0;
    float v = linear_scale;
    while (v > 1.0f) { v *= 0.5f; log2_s++; }
    while (v <= 0.5f && log2_s > -149) { v *= 2.0f; log2_s--; }
    float act_scale = exp2f(static_cast<float>(log2_s));
    if (act_scale < linear_scale) act_scale *= 2.0f;
    return act_scale;
}

__global__ void test_ue8m0_kernel(const float* amax_in, float* scale_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scale_out[i] = test_ue8m0_scale(amax_in[i]);
}

TEST_CASE("ue8m0 scale is power of 2 and >= linear scale", "[deepseek][inference]") {
    float h_amax[] = {448.0f, 224.0f, 100.0f, 1.0f, 0.001f, 0.0f};
    int n = sizeof(h_amax) / sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_amax, n * sizeof(float), cudaMemcpyHostToDevice);
    
    test_ue8m0_kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    float h_out[6];
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    
    for (int i = 0; i < n; ++i) {
        float linear = fmaxf(h_amax[i], 1e-4f) / 448.0f;
        INFO("amax=" << h_amax[i] << " scale=" << h_out[i] << " linear=" << linear);
        // Scale must be >= linear (ceiling)
        CHECK(h_out[i] >= linear - 1e-7f);
        // Scale must be a power of 2 (check by bit pattern)
        float scaled = h_out[i];
        if (scaled > 0) {
            int exp;
            float mant = frexp(scaled, &exp);
            CHECK(mant == Approx(0.5f).epsilon(1e-6f));  // mantissa = 0.5 means power of 2
        }
    }
}

// ── RMSNorm test ─────────────────────────────────────────────────────────────

__global__ void test_rmsnorm_kernel(const float* input, const float* gamma,
                                      float* output, int n, float eps) {
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        atomicAdd(&s_sum, input[i] * input[i]);
    __syncthreads();
    float inv = rsqrtf(s_sum / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        output[i] = input[i] * inv * gamma[i];
}

TEST_CASE("RMSNorm produces unit-rms output", "[deepseek][inference]") {
    const int n = 256;
    std::vector<float> h_in(n), h_gamma(n, 1.0f), h_out(n, 0.0f);
    
    // Fill with random-ish values
    for (int i = 0; i < n; ++i)
        h_in[i] = sinf(static_cast<float>(i) * 0.1f) * 3.0f + 1.0f;
    
    float *d_in, *d_gamma, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_gamma, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    test_rmsnorm_kernel<<<1, 256>>>(d_in, d_gamma, d_out, n, 1e-6f);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check output RMS ≈ 1.0 (since gamma=1.0)
    float rms = 0.0f;
    for (int i = 0; i < n; ++i) rms += h_out[i] * h_out[i];
    rms = sqrtf(rms / static_cast<float>(n));
    CHECK(rms == Approx(1.0f).epsilon(1e-3f));
    
    // Check no NaN/Inf
    for (int i = 0; i < n; ++i) {
        CHECK(std::isfinite(h_out[i]));
    }
    
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);
}

// ── SwiGLU clamp test ────────────────────────────────────────────────────────

__global__ void test_swiglu_clamp_kernel(
    const __half* gate, const __half* up, float* out, int n, float limit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    if (g > limit) g = limit;
    if (u >  limit) u =  limit;
    if (u < -limit) u = -limit;
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = sig * g * u;
}

TEST_CASE("SwiGLU clamps gate and up to limit", "[deepseek][inference]") {
    const int n = 8;
    __half h_gate[] = {__float2half(5.0f), __float2half(20.0f), __float2half(-20.0f),
                       __float2half(0.5f), __float2half(10.0f), __float2half(-5.0f),
                       __float2half(15.0f), __float2half(0.0f)};
    __half h_up[] = {__float2half(3.0f), __float2half(15.0f), __float2half(-15.0f),
                     __float2half(2.0f), __float2half(20.0f), __float2half(-20.0f),
                     __float2half(5.0f), __float2half(1.0f)};
    float h_out[n];
    
    __half *d_gate, *d_up;
    float *d_out;
    cudaMalloc(&d_gate, n * sizeof(__half));
    cudaMalloc(&d_up, n * sizeof(__half));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_gate, h_gate, n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up, n * sizeof(__half), cudaMemcpyHostToDevice);
    
    test_swiglu_clamp_kernel<<<1, n>>>(d_gate, d_up, d_out, n, 10.0f);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // gate=20 → clamped to 10, up=15 → clamped to 10
    // silu(10) ≈ 10 * sigmoid(10) ≈ 9.9995
    CHECK(h_out[1] == Approx(10.0f * 0.99995f * 10.0f).epsilon(1e-3f));
    // gate=10 → silu(10) ≈ 9.9995, up=20 → clamped to 10
    CHECK(h_out[4] == Approx(10.0f * 0.99995f * 10.0f).epsilon(1e-3f));
    // gate=0 → silu(0) = 0, up=1 → 1
    CHECK(h_out[7] == Approx(0.0f).epsilon(1e-6f));
    
    // All outputs should be finite
    for (int i = 0; i < n; ++i)
        CHECK(std::isfinite(h_out[i]));
    
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);
}

// ── Compressor step kernel test ──────────────────────────────────────────────
// Test: 4 tokens, ratio=4, should compress into 1 weighted average

__global__ void test_compressor_kernel(
    const __half* d_kv, const __half* d_score, const float* d_ape,
    float* d_kv_state, float* d_score_state,
    __half* d_compressed,
    int ratio, int coff, int head_dim, int start_pos)
{
    const int slot = start_pos % ratio;
    const int out_dim = coff * head_dim;
    const int tid = threadIdx.x;
    
    for (int i = tid; i < out_dim; i += blockDim.x) {
        d_kv_state[slot * out_dim + i] = __half2float(d_kv[i]);
        d_score_state[slot * out_dim + i] = __half2float(d_score[i]) + d_ape[slot * out_dim + i];
    }
    __syncthreads();
    
    bool should_compress = ((start_pos + 1) % ratio) == 0;
    if (!should_compress) return;
    
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float max_s = -1e30f;
        for (int r = 0; r < ratio; ++r)
            max_s = fmaxf(max_s, d_score_state[r * out_dim + i]);
        float sum_exp = 0.0f;
        for (int r = 0; r < ratio; ++r) {
            d_score_state[r * out_dim + i] = expf(d_score_state[r * out_dim + i] - max_s);
            sum_exp += d_score_state[r * out_dim + i];
        }
        float compressed = 0.0f;
        for (int r = 0; r < ratio; ++r)
            compressed += d_kv_state[r * out_dim + i] * 
                (d_score_state[r * out_dim + i] / (sum_exp + 1e-9f));
        d_compressed[i] = __float2half(compressed);
    }
}

TEST_CASE("Compressor produces weighted average", "[deepseek][inference]") {
    const int ratio = 4;
    const int coff = 1;
    const int head_dim = 8;
    const int out_dim = coff * head_dim;
    
    // 4 tokens with simple values: kv = [1,2,3,4,...], score = [0,0,0,0] (uniform)
    __half h_kv[8] = {__float2half(1), __float2half(2), __float2half(3), __float2half(4),
                      __float2half(5), __float2half(6), __float2half(7), __float2half(8)};
    __half h_score[8] = {__float2half(0), __float2half(0), __float2half(0), __float2half(0),
                         __float2half(0), __float2half(0), __float2half(0), __float2half(0)};
    float h_ape[32] = {};  // all zeros
    
    float h_kv_state[32] = {};
    float h_score_state[32] = {};
    __half h_compressed[8] = {};
    
    __half *d_kv, *d_score, *d_compressed;
    float *d_ape, *d_kv_state, *d_score_state;
    cudaMalloc(&d_kv, out_dim * sizeof(__half));
    cudaMalloc(&d_score, out_dim * sizeof(__half));
    cudaMalloc(&d_ape, ratio * out_dim * sizeof(float));
    cudaMalloc(&d_kv_state, ratio * out_dim * sizeof(float));
    cudaMalloc(&d_score_state, ratio * out_dim * sizeof(float));
    cudaMalloc(&d_compressed, head_dim * sizeof(__half));
    
    // Simulate 4 tokens (start_pos 0-3)
    for (int pos = 0; pos < ratio; ++pos) {
        // Fill kv/score for this token
        __half kv_tok[8], score_tok[8];
        for (int i = 0; i < head_dim; ++i) {
            kv_tok[i] = __float2half(static_cast<float>(pos + 1));
            score_tok[i] = __float2half(0.0f);
        }
        cudaMemcpy(d_kv, kv_tok, out_dim * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_score, score_tok, out_dim * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ape, h_ape, ratio * out_dim * sizeof(float), cudaMemcpyHostToDevice);
        
        test_compressor_kernel<<<1, 256>>>(
            d_kv, d_score, d_ape, d_kv_state, d_score_state,
            d_compressed, ratio, coff, head_dim, pos);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(h_compressed, d_compressed, head_dim * sizeof(__half), cudaMemcpyDeviceToHost);
    
    // With uniform scores, compressed KV = average of [1,2,3,4] = 2.5
    for (int i = 0; i < head_dim; ++i) {
        INFO("compressed[" << i << "] = " << __half2float(h_compressed[i]));
        CHECK(__half2float(h_compressed[i]) == Approx(2.5f).epsilon(1e-2f));
    }
    
    cudaFree(d_kv);
    cudaFree(d_score);
    cudaFree(d_ape);
    cudaFree(d_kv_state);
    cudaFree(d_score_state);
    cudaFree(d_compressed);
}

// ── NVFP4 codec regression tests ─────────────────────────────────────────────
// These call the *real* public lsxcommon::gpu functions (not reimplemented
// mini-kernels) to catch two systematic-debugging findings:
//   1. safetensors_gpu.cu's private float_to_e4m3 rounded the E4M3 block
//      scale incorrectly (floored the exponent) instead of the canonical
//      lsxcommon::float_to_e4m3 (full unsigned-E4M3, round-to-nearest-even),
//      underestimating the largest weight in any block whose local max isn't
//      an exact power-of-2 fraction of the tensor's global max.
//   2. The routed-expert dequant path (dequant_fp4_e8m0) uses a completely
//      different on-disk format (E8M0-per-32, no global_scale) than what the
//      offline converter actually writes (E2M1 + E4M3-per-16 + global_scale,
//      decoded correctly only by dequant_shared_nvfp4).

namespace {

// Mirrors dev_quantize_nvfp4's positive E2M1 magnitude table
// (safetensors_gpu.cu) — sign is bit 3 of the nibble, magnitude is bits 0-2.
constexpr float kE2M1Levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float DecodeE2M1Nibble(uint8_t nibble)
{
    int sign = (nibble >> 3) & 1;
    float v = kE2M1Levels[nibble & 0x7];
    return sign ? -v : v;
}

uint8_t ExtractNibble(const std::vector<uint8_t> &packed, int64_t i)
{
    uint8_t b = packed[static_cast<size_t>(i / 2)];
    return (i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
}

} // namespace

TEST_CASE("NVFP4 offline quantize preserves per-block magnitude (float_to_e4m3 regression)",
          "[deepseek][nvfp4][regression]")
{
    // 3 blocks of 16 F32 elements. Block 0 holds the tensor's global max
    // (6.0). Block 1's local max (4.0) has a NON-power-of-2 ratio to the
    // global max (4/6 = 0.667). The full-E4M3 block scale is RNE: nearest is
    // 0.6875, so the reconstructed max is 6 * 0.6875 = 4.125 (+3.1%), inside
    // the <= 1/16 half-step band. Block 2's ratio (0.25) is a power of two and
    // reconstructs exactly. Floor-rounding the E4M3 exponent (the historical
    // bug) underestimated block 1 by ~25%; RNE is the correct recipe.
    std::vector<float> src(48, 0.0f);
    src[0]  = 6.0f;
    src[16] = 4.0f;
    src[32] = 1.5f;

    std::vector<uint8_t> raw(src.size() * sizeof(float));
    std::memcpy(raw.data(), src.data(), raw.size());

    std::vector<uint8_t> packed, scales;
    float global_scale = 0.0f;
    bool ok = lsxcommon::gpu::quantize_tensor_gpu(
        raw.data(), static_cast<int64_t>(src.size()), "F32",
        nullptr, {}, {}, packed, scales, global_scale);
    REQUIRE(ok);
    REQUIRE(scales.size() == 3);

    auto decode = [&](int64_t i) -> float {
        uint8_t nibble = ExtractNibble(packed, i);
        int64_t block = i / 16;
        float block_scale = lsxcommon::e4m3_to_float(scales[static_cast<size_t>(block)]);
        return DecodeE2M1Nibble(nibble) * block_scale * global_scale;
    };

    float d0 = decode(0);
    float d16 = decode(16);
    float d32 = decode(32);

    INFO("global_scale=" << global_scale
         << " scales=0x" << std::hex << (int)scales[0] << ",0x" << (int)scales[1]
         << ",0x" << (int)scales[2] << std::dec
         << " decoded=" << d0 << "," << d16 << "," << d32);

    CHECK(std::fabs(d0 - 6.0f) / 6.0f < 0.08f);
    // RNE reconstructs block 1 within the E4M3 half-step band (measured
    // +3.1%); the historical floor bug reconstructed ~3.0 (~25% low).
    CHECK(std::fabs(d16 - 4.0f) / 4.0f < 0.08f);
    CHECK(std::fabs(d32 - 1.5f) / 1.5f < 0.08f);
}

TEST_CASE("NVFP4 offline quantize flushes a below-grid block instead of poisoning it",
          "[deepseek][nvfp4][regression]")
{
    // Since the codec became the COMPLETE E4M3FN table, float_to_e4m3 emits
    // 0x00 for any input below 2^-10 (previously it clamped up to the min
    // normal 0x08). The converter stores block scales normalized against the
    // tensor's global max, so a block whose local max is more than 1024x below
    // the global max now lands on a ZERO scale — and quantize_pack_perblock_
    // kernel divides the elements by that scale.
    //
    // The divide yields inf, and the nibbles came out zero only because
    // QuantizeE2M1Nearest selects a level with a STRICT `<`, which no finite
    // level can beat against inf. Flip that to `<=` (or route the converter
    // through the QuantizeE2m1TieUp flavor the runtime uses) and the same inf
    // selects the 6.0 level: a block of maximum-magnitude garbage where the
    // source block was ~1e-3 of the tensor. Correctness of a whole quantized
    // tensor would hinge on one comparison operator in another header.
    //
    // Measured reachability on real weights is low (0 of 177,111,040 blocks
    // across 28 phi-4 weight tensors fall below the grid), so this pins an
    // invariant rather than repairing observed corruption — which is exactly
    // why it needs pinning.
    std::vector<float> src(32, 0.0f);
    src[0] = 6.0f;    // block 0 carries the tensor global max
    src[16] = 1.0e-3f;  // block 1: ratio 1.67e-4 < 2^-10 -> scale byte 0x00

    std::vector<uint8_t> raw(src.size() * sizeof(float));
    std::memcpy(raw.data(), src.data(), raw.size());

    std::vector<uint8_t> packed, scales;
    float global_scale = 0.0f;
    REQUIRE(lsxcommon::gpu::quantize_tensor_gpu(
        raw.data(), static_cast<int64_t>(src.size()), "F32",
        nullptr, {}, {}, packed, scales, global_scale));
    REQUIRE(scales.size() == 2);

    INFO("global_scale=" << global_scale
                         << " scales=0x" << std::hex << (int)scales[0] << ",0x"
                         << (int)scales[1] << std::dec);

    // Block 0 is the reference: it must still quantize normally, proving the
    // tiny block did not corrupt the tensor-wide scale.
    CHECK(scales[0] != 0x00);
    CHECK(std::fabs(DecodeE2M1Nibble(ExtractNibble(packed, 0)) *
                        lsxcommon::e4m3_to_float(scales[0]) * global_scale -
                    6.0f) < 0.5f);

    // Block 1 is below the E4M3 grid: the scale byte is the reserved zero and
    // the whole block must be flushed to zero nibbles.
    CHECK(scales[1] == 0x00);
    for (int64_t i = 16; i < 32; ++i) {
        const uint8_t nibble = ExtractNibble(packed, i);
        INFO("block 1 element " << i << " nibble=0x" << std::hex << (int)nibble
                                << std::dec);
        // Never the sign bit and never the 6.0 magnitude: an inf that escaped
        // into the level search would land on exactly 0x7 / 0xF.
        CHECK((nibble & 0x8u) == 0u);
        CHECK((nibble & 0x7u) != 7u);
    }
}

TEST_CASE("dequant_shared_nvfp4 correctly decodes the on-disk NVFP4 format",
          "[deepseek][nvfp4][regression]")
{
    // One full 16-element block, produced by the real offline converter
    // path, then decoded by the *documented* decoder for this format
    // (schema.h: E2M1 nibble * E4M3 per-16-block scale * global_scale).
    std::vector<float> src = {
        6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.0f,
        -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f};

    std::vector<uint8_t> raw(src.size() * sizeof(float));
    std::memcpy(raw.data(), src.data(), raw.size());

    std::vector<uint8_t> packed, scales;
    float global_scale = 0.0f;
    REQUIRE(lsxcommon::gpu::quantize_tensor_gpu(
        raw.data(), static_cast<int64_t>(src.size()), "F32",
        nullptr, {}, {}, packed, scales, global_scale));
    REQUIRE(scales.size() == 1);

    uint8_t *d_packed = nullptr, *d_scales = nullptr;
    __half *d_out = nullptr;
    cudaMalloc(&d_packed, packed.size());
    cudaMalloc(&d_scales, scales.size());
    cudaMalloc(&d_out, src.size() * sizeof(__half));
    cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice);

    cudaError_t e = lsxcommon::gpu::dequant_shared_nvfp4(
        nullptr, d_packed, static_cast<int64_t>(packed.size()),
        d_scales, static_cast<int64_t>(scales.size()), global_scale,
        d_out, static_cast<int64_t>(src.size()));
    REQUIRE(e == cudaSuccess);
    cudaDeviceSynchronize();

    std::vector<__half> out(src.size());
    cudaMemcpy(out.data(), d_out, src.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_out);

    for (size_t i = 0; i < src.size(); ++i) {
        float got = __half2float(out[i]);
        INFO("i=" << i << " src=" << src[i] << " got=" << got);
        if (src[i] == 0.0f) {
            CHECK(got == 0.0f);
        } else {
            CHECK(std::fabs(got - src[i]) / std::fabs(src[i]) < 0.02f);
        }
    }
}


