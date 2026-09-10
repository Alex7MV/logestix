#include "lsxcommon/perf_calibration.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/log.h"

namespace lsxcommon {
namespace perf {

using lsx_nvfp4::RunLsxNvfp4Gemm;
using lsx_nvfp4::SetLsxSeq1Gemv;
using lsx_nvfp4::SetLsxTcFastMath;

const std::vector<CalibrationShape> &RepresentativeShapes()
{
    static const std::vector<CalibrationShape> kShapes = {
        {4096, 4096},    // attn qkv/o class (~8 MiB)
        {2048, 5632},    // mlp gate/up class (~28 MiB)
        {2048, 2048},    // expert w2 class (~4 MiB)
        {4096, 151936},  // LM-head class (~297 MiB, GEMV ineligible)
    };
    return kShapes;
}

namespace {

double TimeGemmPath(int reps, bool use_gemv, bool use_tc, const __half *x,
                    const uint8_t *packed, const uint8_t *scales, float gs,
                    __half *y, int K, int N, cudaStream_t stream)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    SetLsxTcFastMath(use_tc);
    SetLsxSeq1Gemv(use_gemv);
    for (int i = 0; i < 5; ++i) {  // warmup
        RunLsxNvfp4Gemm(stream, x, packed, scales, gs, y, 1, K, N);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(a, stream);
    for (int i = 0; i < reps; ++i)
        RunLsxNvfp4Gemm(stream, x, packed, scales, gs, y, 1, K, N);
    cudaEventRecord(b, stream);
    cudaStreamSynchronize(stream);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / reps;
}

} // namespace

bool CalibrateGemvLimitMib(int &out_mib)
{
    // Save/restore the process routing state around the measurement.
    const bool prev_tc = lsx_nvfp4::LsxTcFastMath();
    const bool prev_gemv = lsx_nvfp4::LsxSeq1Gemv();

    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
        cudaSuccess)
        return false;

    int best_mib = 8;
    for (const CalibrationShape &sh : RepresentativeShapes()) {
        const int64_t packed_bytes = int64_t{sh.K / 2} * sh.N;
        const int64_t scale_bytes = int64_t{(sh.K + 15) / 16} * sh.N;
        uint8_t *packed = nullptr, *scales = nullptr;
        __half *x = nullptr, *y = nullptr;
        const bool ok =
            cudaMalloc(&packed, packed_bytes) == cudaSuccess &&
            cudaMalloc(&scales, scale_bytes) == cudaSuccess &&
            cudaMalloc(&x, sizeof(__half) * sh.K) == cudaSuccess &&
            cudaMalloc(&y, sizeof(__half) * sh.N) == cudaSuccess;
        if (!ok) {
            cudaFree(packed);
            cudaFree(scales);
            cudaFree(x);
            cudaFree(y);
            cudaStreamDestroy(stream);
            lsx_nvfp4::SetLsxTcFastMath(prev_tc);
            lsx_nvfp4::SetLsxSeq1Gemv(prev_gemv);
            return false;
        }
        cudaMemsetAsync(packed, 0x88, packed_bytes, stream);  // mid-level E2M1
        cudaMemsetAsync(scales, 0x38, scale_bytes, stream);   // E4M3 ~ 1.0
        cudaMemsetAsync(x, 0, sizeof(__half) * sh.K, stream);
        cudaStreamSynchronize(stream);

        const double gemv_ms =
            TimeGemmPath(50, true, true, x, packed, scales, 1.0f, y, sh.K,
                         sh.N, stream);
        const double tc_ms =
            TimeGemmPath(50, false, true, x, packed, scales, 1.0f, y, sh.K,
                         sh.N, stream);
        const int64_t total_bytes = packed_bytes + scale_bytes;
        if (gemv_ms < tc_ms && total_bytes <= int64_t{64} << 20)
            best_mib = std::max(best_mib,
                                static_cast<int>(total_bytes >> 20));
        cudaFree(packed);
        cudaFree(scales);
        cudaFree(x);
        cudaFree(y);
    }
    cudaStreamDestroy(stream);
    lsx_nvfp4::SetLsxTcFastMath(prev_tc);
    lsx_nvfp4::SetLsxSeq1Gemv(prev_gemv);
    out_mib = std::clamp(best_mib, 8, 64);
    return true;
}

} // namespace perf
} // namespace lsxcommon
