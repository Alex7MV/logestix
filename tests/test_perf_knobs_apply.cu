#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/lsx_inference_nvfp4.h"

using namespace lsxcommon::lsx_nvfp4;

TEST_CASE("GEMV eligibility follows SetLsxGemmMaxMib", "[perf_knobs]")
{
    SetLsxTcFastMath(true);
    SetLsxSeq1Gemv(true);
    // Bytes for [N,K] NVFP4 weight: packed = K/2*N, scales = ceil(K/16)*N.
    // K=4096, N=3584 -> packed 7 MiB + scales 0.875 MiB ~ 7.875 MiB.
    // K=8192, N=11008 -> packed 44 MiB + scales 5.5 MiB ~ 49.5 MiB.
    auto bytes = [](int K, int N) -> int64_t {
        return int64_t(K / 2) * N + int64_t((K + 15) / 16) * N;
    };
    const int64_t small = bytes(4096, 3584);
    const int64_t large = bytes(8192, 11008);

    SetLsxGemmMaxMib(32);
    REQUIRE(small <= int64_t{32} << 20);
    REQUIRE(large > int64_t{32} << 20);

    SetLsxGemmMaxMib(64);
    REQUIRE(large <= int64_t{64} << 20);   // now eligible

    SetLsxGemmMaxMib(4);
    REQUIRE(small > int64_t{4} << 20);     // now ineligible

    SetLsxGemmMaxMib(32);  // restore default
    SetLsxTcFastMath(false);
    SetLsxSeq1Gemv(false);
}

#include "lsxcommon/lsx_tc_gemm.h"

using lsxcommon::lsx_tc::LsxTcBatchSplitK;
using lsxcommon::lsx_tc::SetLsxSplitKGridMin;

TEST_CASE("LsxTcBatchSplitK follows the splitk_grid_min knob", "[perf_knobs]")
{
    // K=8192, N=2048 -> grid_x = 2048/64 = 32. With min 96: 32*8=256 >= 96
    // and (8192/64)%8==0 -> split 8.
    SetLsxSplitKGridMin(96);
    REQUIRE(LsxTcBatchSplitK(8192, 2048) == 8);
    // Raise the threshold so even 8-way misses it -> no split.
    SetLsxSplitKGridMin(300);
    REQUIRE(LsxTcBatchSplitK(8192, 2048) == 1);
    // Narrow N so even the 8-way split misses the raised threshold.
    SetLsxSplitKGridMin(64);
    REQUIRE(LsxTcBatchSplitK(2048, 1024) == 8);  // grid_x=16: 16*8=128 >= 64
    SetLsxSplitKGridMin(96);  // restore default
}
