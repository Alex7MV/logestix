#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/diagnostics.h"

#include <arrow/gpu/cuda_api.h>

#include <cuda_runtime.h>

#include <cstring>
#include <cstdint>
#include <mutex>
#include <algorithm>
#include <vector>
#include <cmath>

namespace lsxcommon {
namespace gpu {

namespace {

constexpr int64_t kBlockSize = 16;

// ── Singleton Arrow CUDA context ─────────────────────────────────────────────

arrow::Result<std::shared_ptr<arrow::cuda::CudaContext>> get_cuda_ctx()
{
    static std::shared_ptr<arrow::cuda::CudaContext> ctx;
    static std::once_flag init;
    static arrow::Status status;

    std::call_once(init, [] {
        log::debug("[DIAG] CUDA context init: before CudaDeviceManager::Instance()");
        diag::LogArrowSystemState("pre-cuda-init");
        auto mgr = arrow::cuda::CudaDeviceManager::Instance();
        if (!mgr.ok()) { status = mgr.status(); return; }
        diag::LogArrowSystemState("post-cuda-init");
        auto result = (*mgr)->GetContext(0);
        if (!result.ok()) { status = result.status(); return; }
        auto ctx_ptr = *result;
        log::debug(std::string("[DIAG] CUDA context initialized: device=")
            + ctx_ptr->device()->ToString());
        ctx = std::move(ctx_ptr);
    });

    if (!ctx) return status;
    return ctx;
}

// ── Device helpers ───────────────────────────────────────────────────────────

__device__ float dev_bf16_to_float(const uint8_t *p)
{
    uint16_t u;
    memcpy(&u, p, sizeof(u));
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

__device__ float dev_f8e4m3_to_float(uint8_t b)
{
    if (!b || b == 0x80) return 0.0f;
    int s = (b >> 7) & 1;
    int e = (b >> 3) & 0x0F;
    int m = b & 0x07;
    uint32_t f32 = (static_cast<uint32_t>(s) << 31)
                 | (static_cast<uint32_t>(e + 127 - 7) << 23)
                 | (static_cast<uint32_t>(m) << 20);
    float v;
    memcpy(&v, &f32, sizeof(v));
    return v;
}

__device__ float dev_f8e8m0_to_float(uint8_t b)
{
    if (!b) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

__device__ float dev_i8_to_float(uint8_t b)
{
    return static_cast<float>(static_cast<int8_t>(b));
}

__device__ uint8_t dev_quantize_nvfp4(float val)
{
    if (val == 0.0f) return 0;
    int sign = val < 0 ? 1 : 0;
    float abs_val = fabsf(val);

    // NVFP4 E2M1 positive magnitudes (must match the E2M1 decode used by
    // the dequant kernels — e.g. nvfp4_to_float in testmodel_gpu.cu)
    static const float levels[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    int best = 0;
    float best_diff = fabsf(abs_val - levels[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(abs_val - levels[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return static_cast<uint8_t>((sign << 3) | best);
}

// ── Kernels ──────────────────────────────────────────────────────────────────

__global__ void convert_kernel(const uint8_t *input, float *output,
                               int64_t n, int dtype_code,
                               const uint8_t *e8m0_scales,
                               int64_t C, int64_t SC, int64_t block_r,
                               int64_t block_c)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (; i < n; i += stride) {
        float v;
        switch (dtype_code) {
        case 0: v = dev_bf16_to_float(input + i * 2); break;
        case 1: v = dev_f8e4m3_to_float(input[i]);    break;
        case 2: v = dev_f8e8m0_to_float(input[i]);    break;
        case 3: memcpy(&v, input + i * 4, sizeof(v)); break;
        case 4: v = dev_i8_to_float(input[i]);         break;
        default: v = 0.0f;
        }
        if (e8m0_scales) {
            // 2D tile layout: scale[SR,SC] over weight[R,C] with per-tile
            // block_r = R/SR, block_c = C/SC. Element (r,c) -> tile index.
            int64_t r = i / C;
            int64_t c = i % C;
            int64_t sr = r / block_r;
            int64_t sc = c / block_c;
            v *= dev_f8e8m0_to_float(e8m0_scales[sr * SC + sc]);
        }
        output[i] = v;
    }
}

__global__ void block_maxabs_kernel(const float *input, float *block_max,
                                     int64_t nblocks, int64_t numel)
{
    __shared__ float sdata[16];

    int64_t blk = blockIdx.x;
    int64_t base = blk * kBlockSize;
    int64_t idx = base + threadIdx.x;

    sdata[threadIdx.x] = (idx < numel) ? fabsf(input[idx]) : 0.0f;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x],
                                        sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        block_max[blk] = sdata[0];
}

__global__ void quantize_pack_perblock_kernel(const float *input,
                                               uint8_t *output,
                                               const float *block_scales,
                                               int64_t n)
{
    int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (i >= n) return;

    int64_t blk0 = i / kBlockSize;
    int64_t blk1 = (i + 1) / kBlockSize;

    float s0 = block_scales[blk0];
    float s1 = block_scales[blk1];

    uint8_t lo = dev_quantize_nvfp4(input[i] / s0);
    uint8_t hi = (i + 1 < n) ? dev_quantize_nvfp4(input[i + 1] / s1) : 0;
    output[i / 2] = (hi << 4) | lo;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

int dtype_code(const std::string &dtype)
{
    if (dtype == "BF16") return 0;
    if (dtype == "F8_E4M3") return 1;
    if (dtype == "F8_E8M0") return 2;
    if (dtype == "F32") return 3;
    if (dtype == "I8" || dtype == "U8") return 4;
    return -1;
}

int64_t dtype_byte_size(int dc)
{
    switch (dc) {
    case 0: return 2;
    case 1: return 1;
    case 2: return 1;
    case 3: return 4;
    case 4: return 1;
    default: return 0;
    }
}

float e4m3_p2_to_float(uint8_t b)
{
    if (b == 0) return 0.0f;
    int e_stored = (b >> 3) & 0x0F;
    int exp_unb = e_stored - 7;
    return std::ldexp(1.0f, exp_unb);
}

uint8_t float_to_e4m3_p2(float v)
{
    if (v == 0.0f) return 0;
    int e;
    float m = std::frexp(v, &e);
    int e4 = e - 1 + 7;
    if (e4 < 1) e4 = 1;
    if (e4 > 14) e4 = 14;
    return static_cast<uint8_t>(e4 << 3);
}

} // anonymous namespace

bool quantize_tensor_gpu(const uint8_t *raw_data, int64_t numel,
                         const std::string &dtype,
                         const uint8_t *e8m0_scales,
                         const std::vector<int64_t> &tensor_shape,
                         const std::vector<int64_t> &e8m0_scale_shape,
                         std::vector<uint8_t> &packed,
                         std::vector<uint8_t> &scales_e4m3,
                         float &global_scale)
{
    int dc = dtype_code(dtype);
    if (dc < 0) return false;

    auto maybe_ctx = get_cuda_ctx();
    if (!maybe_ctx.ok()) return false;
    auto ctx = *maybe_ctx;
    (void)ctx;

    int64_t raw_size = numel * dtype_byte_size(dc);
    int64_t npacked = (numel + 1) / 2;
    int64_t nblocks = (numel + kBlockSize - 1) / kBlockSize;

    // Resolve the E8M0 2D tile layout (scale[SR,SC] over weight[R,C]).
    int64_t C = 1, SC = 1, block_r = 1, block_c = 1;
    int64_t scale_numel = 0;
    bool use_e8m0 = (e8m0_scales != nullptr)
        && tensor_shape.size() == 2 && e8m0_scale_shape.size() == 2;
    if (use_e8m0) {
        int64_t R = tensor_shape[0];
        C = tensor_shape[1];
        int64_t SR = e8m0_scale_shape[0];
        SC = e8m0_scale_shape[1];
        block_r = (SR > 0) ? R / SR : 1;
        block_c = (SC > 0) ? C / SC : 1;
        scale_numel = SR * SC;
        if (block_r <= 0 || block_c <= 0 || scale_numel <= 0)
            use_e8m0 = false;
    }

    uint8_t *d_raw = nullptr;
    float *d_float = nullptr;
    uint8_t *d_packed = nullptr;
    float *d_block_max = nullptr;
    float *d_block_scales = nullptr;
    uint8_t *d_e8m0 = nullptr;

    auto alloc_dev = [](auto &ptr, int64_t sz) -> bool {
        if (sz <= 0) { ptr = nullptr; return true; }
        return cudaMalloc(&ptr, static_cast<size_t>(sz)) == cudaSuccess;
    };
    if (!alloc_dev(d_raw, raw_size)) return false;
    if (!alloc_dev(d_float, numel * sizeof(float))) { cudaFree(d_raw); return false; }
    if (!alloc_dev(d_packed, npacked)) { cudaFree(d_raw); cudaFree(d_float); return false; }
    if (!alloc_dev(d_block_max, nblocks * sizeof(float))) { cudaFree(d_raw); cudaFree(d_float); cudaFree(d_packed); return false; }
    if (!alloc_dev(d_block_scales, nblocks * sizeof(float))) { cudaFree(d_raw); cudaFree(d_float); cudaFree(d_packed); cudaFree(d_block_max); return false; }
    if (use_e8m0 && !alloc_dev(d_e8m0, scale_numel)) { cudaFree(d_raw); cudaFree(d_float); cudaFree(d_packed); cudaFree(d_block_max); cudaFree(d_block_scales); return false; }

    struct CudaMemGuard {
        uint8_t *&r; float *&f; uint8_t *&p; float *&bm; float *&bs; uint8_t *&e;
        ~CudaMemGuard() {
            if (e) cudaFree(e);
            if (bs) cudaFree(bs);
            if (bm) cudaFree(bm);
            if (p) cudaFree(p);
            if (f) cudaFree(f);
            if (r) cudaFree(r);
        }
    } cmg{d_raw, d_float, d_packed, d_block_max, d_block_scales, d_e8m0};

    // ── Async stream ────────────────────────────────────────────────────────
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } sg{stream};

    // ── Async pipeline on stream ────────────────────────────────────────────

    // 1. Copy raw input host → device
    cudaMemcpyAsync(d_raw, raw_data, raw_size, cudaMemcpyHostToDevice, stream);
    if (use_e8m0)
        cudaMemcpyAsync(d_e8m0, e8m0_scales, scale_numel,
                        cudaMemcpyHostToDevice, stream);

    // 2. Convert raw → float32 (applying E8M0 block scales if present)
    convert_kernel<<<512, 256, 0, stream>>>(
        d_raw, d_float, numel, dc,
        use_e8m0 ? d_e8m0 : nullptr, C, SC, block_r, block_c);
    if (cudaGetLastError() != cudaSuccess) return false;

    // 3. Per-block max_abs
    block_maxabs_kernel<<<static_cast<int>(nblocks), kBlockSize, 0, stream>>>(
        d_float, d_block_max, nblocks, numel);
    if (cudaGetLastError() != cudaSuccess) return false;

    // 4. Sync stream so block_max is ready for host readback
    cudaStreamSynchronize(stream);

    // 5. Read back block_max → host, compute E4M3 scales and global max
    float global_max_abs = 0.0f;
    {
        std::vector<float> host_block_max(static_cast<std::size_t>(nblocks));
        cudaMemcpy(host_block_max.data(), d_block_max,
                   nblocks * sizeof(float), cudaMemcpyDeviceToHost);

        for (int64_t i = 0; i < nblocks; i++) {
            float v = host_block_max[static_cast<std::size_t>(i)];
            if (v > global_max_abs) global_max_abs = v;
        }

        if (!std::isfinite(global_max_abs) || global_max_abs == 0.0f)
            global_max_abs = 1.0f / 128.0f;
        // E2M1 max magnitude is 6.0 — map the global max onto the top level.
        // Per-block E4M3 scales are stored relative to the global max so that
        //   dequant = E2M1(q) * e4m3_block_scale * global_scale  ≈  input
        // (i.e. the quantizer divides by the same product).
        global_scale = global_max_abs / 6.0f;

        scales_e4m3.assign(static_cast<std::size_t>(nblocks), 0);
        std::vector<float> host_block_scales(static_cast<std::size_t>(nblocks));
        for (int64_t i = 0; i < nblocks; i++) {
            float max_val = host_block_max[static_cast<std::size_t>(i)];
            float s = (max_val == 0.0f)
                ? (1.0f / 128.0f)
                : (max_val / global_max_abs);
            uint8_t e4 = float_to_e4m3_p2(s);
            scales_e4m3[static_cast<std::size_t>(i)] = e4;
            // Device-side block scales feed the quantizer, which divides by the
            // full scale product (e4m3 * global_scale) — matching the dequant.
            host_block_scales[static_cast<std::size_t>(i)] =
                e4m3_p2_to_float(e4) * global_scale;
        }

        // Upload float scales to device (async)
        cudaMemcpyAsync(d_block_scales, host_block_scales.data(),
                        nblocks * sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    // 7. Per-block quantize + pack
    quantize_pack_perblock_kernel<<<512, 256, 0, stream>>>(
        d_float, d_packed, d_block_scales, numel);
    if (cudaGetLastError() != cudaSuccess) return false;

    // 8. Copy packed data device → host (async)
    packed.resize(static_cast<std::size_t>(npacked));
    cudaMemcpyAsync(packed.data(), d_packed, npacked,
                    cudaMemcpyDeviceToHost, stream);

    // 9. Final sync
    cudaStreamSynchronize(stream);

    return true;
}

} // namespace gpu
} // namespace lsxcommon
