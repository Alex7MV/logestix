// Bit-exact parity across KV layouts, at the decode-step level.
//
// CHARACTERIZATION TEST - there is no red phase and faking one would be theatre.
// The migration changes addresses only: for each (t, h) the scalar operation order,
// the values and the explicitly-rounded intrinsics (__fadd_rn/__fmul_rn) are
// unchanged, and the split-K combine sees partials computed identically. So the
// two layouts must agree bit for bit, and this checks that rather than assuming
// it. It exists to catch any FUTURE edit that makes the layout observable in the
// numbers: a reassociated reduction, a traversal order that coalescing changes,
// an FMA contraction one layout happens to elide.
//
// It runs on the synthetic fixture rather than real weights on purpose. Parity is
// a property of the addressing, not of the weights, and this exercises exactly the
// kernels that changed - the full XingDecodeStep over three layers. It also has to:
// two resident Xing instances need ~30 GiB between them, and the second expert-slab
// allocation dies of OOM on a 32 GiB card. Real-weight coverage of the new default
// is the FSI/CODE/DET golden gates, which now run head-major.
#include <catch2/catch_test_macros.hpp>

#include "test_xing_moe_fixture.h"

#include "lsxcommon/log.h"
#include "xing_moe_constants.h"
#include "lsxcommon/xing_moe_convert.h"
#include "lsxcommon/xing_moe_inference.h"
#include "lsxcommon/xing_moe_load.h"

#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace frozen = lsxcommon::xing::frozen;
using namespace lsxcommon;
namespace fs = std::filesystem;

namespace
{
// Enough positions that the split-K path runs with several chunks per head: the
// parity claim is about long-context addressing, and a single-chunk comparison
// would leave most of the stride untested.
constexpr int kParityCapacity = 512;
constexpr int kParitySteps = 6;

struct Fixture
{
    fs::path out;
    xing::XingPinnedWeights pw;
    std::vector<float> rc, rs;
    float *d_rc = nullptr;
    float *d_rs = nullptr;
    cudaStream_t stream = nullptr;

    ~Fixture()
    {
        pw.Free();
        if (d_rc) cudaFree(d_rc);
        if (d_rs) cudaFree(d_rs);
        if (stream) cudaStreamDestroy(stream);
    }
};

Fixture BuildFixture()
{
    Fixture f;
    auto in = fs::temp_directory_path() / "xing_kvparity_in";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::create_directories(in);
    xingtest::WriteXingConfig(in);
    xingtest::WriteSafetensors(
        in, "model-00001-of-00001.safetensors",
        xingtest::BuildSyntheticXingRealGeometry(3, 4));
    f.out = fs::temp_directory_path() / "xing_kvparity_out";
    fs::remove_all(f.out, ec);
    REQUIRE(xing::ConvertXingModel(in.string(), f.out).ok());
    REQUIRE(cudaStreamCreate(&f.stream) == cudaSuccess);
    std::string error;
    const cudaError_t err =
        xing::LoadXingWeights(f.out.string(), f.pw, f.stream, error);
    INFO("loader error: " << error);
    REQUIRE(err == cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kParityCapacity, f.rc, f.rs));
    REQUIRE(xing::UploadXingYarnTables(f.stream, kParityCapacity, f.rc, f.rs,
                                       &f.d_rc, &f.d_rs) == cudaSuccess);
    return f;
}

// Runs `steps` decode steps from a clean session and returns the final logits
// plus the KV cache contents. The cache is part of the comparison on purpose: the
// write path is one of the migrated sites, and logits alone could agree while the
// cache silently diverges for positions nothing reads back yet.
std::vector<float> RunLayout(Fixture &f, const xing::XingDims &dims,
                             KvCacheLayout layout, std::vector<float> *kv_out)
{
    xing::XingSession s;
    REQUIRE(xing::AllocateXingSession(s, dims, kParityCapacity, layout,
                                      f.stream) == cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kParityCapacity, f.rc, f.rs));
    std::vector<float> logits(frozen::kVocab, 0.0f);
    for (int step = 0; step < kParitySteps; ++step)
    {
        const cudaError_t st = xing::XingDecodeStep(
            f.pw, dims, s, 1000 + step, f.d_rc, f.d_rs, 0, f.stream, -1);
        INFO("layout " << static_cast<int>(layout) << " step " << step
                       << " rc=" << static_cast<int>(st));
        REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
        xing::XingReadLogits(s, logits.data(), f.stream);
        REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
    }
    if (kv_out)
    {
        // XingCopyLayerKV gathers one position's row for one layer; take layer 0
        // and the last position written, which is the row most likely to sit on a
        // stride boundary.
        kv_out->assign(static_cast<size_t>(frozen::kHeads) * frozen::kQkHeadDim,
                       0.0f);
        std::vector<float> v(static_cast<size_t>(frozen::kHeads) * frozen::kVHeadDim,
                             0.0f);
        REQUIRE(xing::XingCopyLayerKV(s, 0, kParitySteps - 1, kv_out->data(),
                                      v.data(), f.stream) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
        kv_out->insert(kv_out->end(), v.begin(), v.end());
    }
    s.Free();
    return logits;
}
}  // namespace

TEST_CASE("Xing decode step is bit-identical across KV layouts",
          "[xing][kv_layout]")
{
    Fixture f = BuildFixture();
    const auto dims = xing::XingDimsFromWeights(f.pw);

    std::vector<float> kv_head, kv_pos;
    const auto head = RunLayout(f, dims, KvCacheLayout::kHeadMajor, &kv_head);
    const auto pos = RunLayout(f, dims, KvCacheLayout::kPosMajor, &kv_pos);

    REQUIRE(head.size() == pos.size());
    REQUIRE(!head.empty());
    CHECK(std::memcmp(head.data(), pos.data(), head.size() * sizeof(float)) == 0);

    REQUIRE(kv_head.size() == kv_pos.size());
    REQUIRE(!kv_head.empty());
    CHECK(std::memcmp(kv_head.data(), kv_pos.data(),
                      kv_head.size() * sizeof(float)) == 0);
}
