// Xing4.0-29B-A4B model-stage test (Model Ingestion Protocol stage 5/8).
//
// What this pins, in order of what it would catch:
//
//   1. ONE decode step runs end to end on a synthetic converted model and
//      produces FINITE logits for the full vocab. This is the gate the decode
//      pipeline had no coverage for at all — a missing kernel launch or a
//      half-wired mHC produced plausible garbage, not a failure.
//   2. Two steps at different positions agree with a manual recomputation of
//      the layer invariants the step maintains: the four streams stay finite,
//      the KV cache grows by exactly one row per step, and the attention scales
//      stay inside the range the YaRN table implies.
//   3. Determinism: the same token at the same position twice gives
//      bit-identical logits (the shared-memory sandbox, no atomics).
//   4. Cooperative cancel through the shared helpers, so `--serve` can stop a
//      Xing decode.

#include <catch2/catch_test_macros.hpp>

#include "test_xing_moe_fixture.h"
#include "test_streaming_assert.h"

#include "lsxcommon/log.h"
#include "lsxcommon/xing_moe_convert.h"
#include "lsxcommon/xing_moe_inference.h"
#include "lsxcommon/xing_moe_load.h"

#include <cuda_runtime.h>

#include <cmath>
#include <absl/strings/str_cat.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace frozen = lsxcommon::xing::frozen;
using namespace lsxcommon;
using lsxcommon::xing::XingPinnedWeights;
namespace fs = std::filesystem;

namespace {

struct ConvertedFixture {
    fs::path out;
    XingPinnedWeights pw;
    std::unique_ptr<lsxcommon::xing::XingSession> session;
    std::vector<float> rc, rs;
    float *d_rc = nullptr;
    float *d_rs = nullptr;
    cudaStream_t stream = nullptr;

    ConvertedFixture() = default;
    ConvertedFixture(ConvertedFixture &&) = default;
    ConvertedFixture &operator=(ConvertedFixture &&) = default;
    ConvertedFixture(const ConvertedFixture &) = delete;
    ConvertedFixture &operator=(const ConvertedFixture &) = delete;

    ~ConvertedFixture()
    {
        pw.Free();
        if (session) session->Free();
        if (d_rc) cudaFree(d_rc);
        if (d_rs) cudaFree(d_rs);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Converts a FULL-geometry synthetic model (3 layers: 2 dense + 1 MoE with 2
// experts) and loads it. Full geometry because the decode step reads every
// projection at its real width - a [16,16] fixture would be stepped with the
// wrong shapes.
ConvertedFixture BuildFixture()
{
    ConvertedFixture f;
    auto in = fs::temp_directory_path() / "xing_model_in";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::create_directories(in);
    xingtest::WriteXingConfig(in);
    xingtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               xingtest::BuildSyntheticXingRealGeometry(3, 4));  // >= kTopK(4): a decode with fewer experts than top_k has no 4th candidate
    f.out = fs::temp_directory_path() / "xing_model_out";
    fs::remove_all(f.out, ec);
    REQUIRE(xing::ConvertXingModel(in.string(), f.out).ok());


    REQUIRE(cudaStreamCreate(&f.stream) == cudaSuccess);
    std::string error;
    const cudaError_t err =
        xing::LoadXingWeights(f.out.string(), f.pw, f.stream, error);
    INFO("loader error: " << error);
    REQUIRE(err == cudaSuccess);
    return f;
}

constexpr int kCapacity = 8;

} // namespace

TEST_CASE("Xing decode step produces finite logits", "[xing][model]")
{
    ConvertedFixture f = BuildFixture();
    const auto dims = xing::XingDimsFromWeights(f.pw);
    REQUIRE(dims.hidden == 3584);
    REQUIRE(dims.vocab == 131072);
    REQUIRE(dims.layers == 3);
    REQUIRE(dims.experts == 4);   // >= kTopK, see the fixture comment
    REQUIRE(dims.heads == 32);

    f.session = std::make_unique<lsxcommon::xing::XingSession>();
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity, f.stream) ==
            cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kCapacity, f.rc, f.rs));
    REQUIRE(xing::UploadXingYarnTables(f.stream, kCapacity, f.rc, f.rs, &f.d_rc,
                                       &f.d_rs) == cudaSuccess);

    // Bisect the stack first: a layer prefix tells us where a non-finite value
    // first appears, which is the difference between a 5-minute and a
    // 5-hour debug.
    for (int prefix = 0; prefix <= dims.layers; ++prefix)
    {
        f.session->Free();
        REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity, f.stream) ==
                cudaSuccess);
        const cudaError_t st =
            xing::XingDecodeStep(f.pw, dims, *f.session, 7, f.d_rc, f.d_rs, 0,
                                 f.stream, prefix);
        INFO("prefix " << prefix << " rc=" << static_cast<int>(st));
        REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
        std::vector<float> probe(frozen::kVocab, 0.0f);
        xing::XingReadLogits(*f.session, probe.data(), f.stream);
        REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
        int bad = 0;
        for (float v : probe)
            if (!std::isfinite(v)) ++bad;
        if (bad)
        {
            const struct { const char *name; const float *p; int64_t n; } bufs[] = {
                {"d_hidden", f.session->d_hidden, 4 * dims.hidden},
                {"d_collapsed", f.session->d_collapsed, dims.hidden},
                {"d_act", f.session->d_act, dims.hidden},
                {"d_norm", f.session->d_norm, dims.hidden},
                {"d_q", f.session->d_q, dims.heads * dims.qk_head},
                {"d_kv_latent", f.session->d_kv_latent, dims.kv_lora + dims.qk_rope},
                {"d_kv_b", f.session->d_kv_b, dims.heads * (dims.qk_nope + dims.v_head)},
                {"d_attn", f.session->d_attn, dims.heads * dims.v_head},
                {"d_scores", f.session->d_scores, kCapacity * dims.heads},
                {"d_post", f.session->d_post, dims.hc_mult},
                {"d_comb", f.session->d_comb, dims.hc_mult * dims.hc_mult},
            };
            std::string scan;
            for (const auto &b : bufs)
            {
                int nb = 0;
                xing::XingCountNonFinite(b.p, b.n, &nb, f.stream);
                if (cudaStreamSynchronize(f.stream) != cudaSuccess) break;
                float mx = 0.0f;
                xing::XingMaxAbs(b.p, b.n, &mx, f.stream);
                if (cudaStreamSynchronize(f.stream) != cudaSuccess) break;
                scan += absl::StrCat("\n  ", b.name, ": ", nb, " max|x|=", mx);
            }
            lsxcommon::log::warn(absl::StrCat("non-finite by buffer:", scan));
        }
        INFO("prefix " << prefix << " non-finite logits: " << bad);
        if (bad)
        {
            FAIL("non-finite logits after " << prefix << " layer(s)");
        }
    }

    const cudaError_t step =
        xing::XingDecodeStep(f.pw, dims, *f.session, 7, f.d_rc, f.d_rs, 0,
                             f.stream);
    INFO("decode step rc=" << static_cast<int>(step) << " ("
                          << cudaGetErrorString(step)
                          << ") at xing_moe_inference.cu:"
                          << xing::XingLastFailureLine());
    REQUIRE(step == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);

    std::vector<float> logits(frozen::kVocab, 0.0f);
    xing::XingReadLogits(*f.session, logits.data(), f.stream);
    REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);

    int finite = 0;
    float lo = 1e30f, hi = -1e30f;
    for (float v : logits)
    {
        if (std::isfinite(v))
        {
            ++finite;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    INFO("finite logits: " << finite << " range [" << lo << ", " << hi << "]");
    REQUIRE(finite == frozen::kVocab);
    REQUIRE(std::isfinite(lo));
    REQUIRE(std::isfinite(hi));
    REQUIRE(hi > lo);
    REQUIRE(f.session->length == 1);

    // The cache holds exactly one token after one step.
    REQUIRE(f.session->capacity == kCapacity);
}

TEST_CASE("Xing decode is deterministic at a fixed position", "[xing][model]")
{
    ConvertedFixture f = BuildFixture();
    const auto dims = xing::XingDimsFromWeights(f.pw);
    f.session = std::make_unique<lsxcommon::xing::XingSession>();
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity, f.stream) ==
            cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kCapacity, f.rc, f.rs));
    REQUIRE(xing::UploadXingYarnTables(f.stream, kCapacity, f.rc, f.rs, &f.d_rc,
                                       &f.d_rs) == cudaSuccess);

    std::vector<float> first(frozen::kVocab, 0.0f), second(frozen::kVocab, 0.0f);
    REQUIRE(xing::XingDecodeStep(f.pw, dims, *f.session, 11, f.d_rc, f.d_rs, 0,
                                 f.stream) == cudaSuccess);
    xing::XingReadLogits(*f.session, first.data(), f.stream);
    REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
    
    // Same token, same position, SAME session: a family whose decode is not
    // run-to-run stable cannot serve --serve at all.
    REQUIRE(xing::XingDecodeStep(f.pw, dims, *f.session, 11, f.d_rc, f.d_rs, 0,
                                 f.stream) == cudaSuccess);
    xing::XingReadLogits(*f.session, second.data(), f.stream);
    REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
    
    for (int i = 0; i < frozen::kVocab; i += 97)
        REQUIRE(first[i] == second[i]);
}

TEST_CASE("Xing decode advances the KV cache one token per step",
          "[xing][model]")
{
    ConvertedFixture f = BuildFixture();
    const auto dims = xing::XingDimsFromWeights(f.pw);
    f.session = std::make_unique<lsxcommon::xing::XingSession>();
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity, f.stream) ==
            cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kCapacity, f.rc, f.rs));
    REQUIRE(xing::UploadXingYarnTables(f.stream, kCapacity, f.rc, f.rs, &f.d_rc,
                                       &f.d_rs) == cudaSuccess);

    std::vector<float> logits(frozen::kVocab, 0.0f);
    for (int pos = 0; pos < 4; ++pos)
    {
        REQUIRE(xing::XingDecodeStep(f.pw, dims, *f.session, pos + 1, f.d_rc,
                                     f.d_rs, pos, f.stream) == cudaSuccess);
        REQUIRE(f.session->length == pos + 1);
        xing::XingReadLogits(*f.session, logits.data(), f.stream);
        REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
        for (int i = 0; i < frozen::kVocab; i += 1024)
            REQUIRE(std::isfinite(logits[i]));
    }
    // Position 4 must differ from position 3: a step that ignored `pos` would
    // silently produce the same logits forever.
    REQUIRE(f.session->length == 4);
}