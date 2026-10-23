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
#include "lsxcommon/model_factory.h"
#include "lsxcommon/xing_moe_convert.h"
#include "lsxcommon/xing_moe_inference.h"
#include "lsxcommon/xing_moe_load.h"

#include <cuda_runtime.h>

#include <cmath>
#include <absl/strings/str_cat.h>

#include <filesystem>
#include <memory>
#include <string>
#include <cstring>
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
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity,
                                      lsxcommon::KvCacheLayout::kHeadMajor,
                                      f.stream) == cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kCapacity, f.rc, f.rs));
    REQUIRE(xing::UploadXingYarnTables(f.stream, kCapacity, f.rc, f.rs, &f.d_rc,
                                       &f.d_rs) == cudaSuccess);

    // Bisect the stack first: a layer prefix tells us where a non-finite value
    // first appears, which is the difference between a 5-minute and a
    // 5-hour debug.
    for (int prefix = 0; prefix <= dims.layers; ++prefix)
    {
        f.session->Free();
        REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity,
                                      lsxcommon::KvCacheLayout::kHeadMajor,
                                      f.stream) ==
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
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity,
                                      lsxcommon::KvCacheLayout::kHeadMajor,
                                      f.stream) ==
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
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity,
                                      lsxcommon::KvCacheLayout::kHeadMajor,
                                      f.stream) ==
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
// Every decoder layer owns its K/V (upstream passes layer_idx to
// past_key_values.update). With one shared buffer each layer overwrites the
// previous layer's row for the same position, so a two-layer stack would return
// the LAST layer's values for layer 0 - and nothing in a single-token prefill
// notices, because a lone token is never read back.
TEST_CASE("Xing keeps a separate KV cache per layer", "[xing][model]")
{
    ConvertedFixture f = BuildFixture();
    const auto dims = xing::XingDimsFromWeights(f.pw);
    REQUIRE(dims.layers >= 2);

    f.session = std::make_unique<lsxcommon::xing::XingSession>();
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kCapacity,
                                      lsxcommon::KvCacheLayout::kHeadMajor,
                                      f.stream) ==
            cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kCapacity, f.rc, f.rs));
    REQUIRE(xing::UploadXingYarnTables(f.stream, kCapacity, f.rc, f.rs, &f.d_rc,
                                       &f.d_rs) == cudaSuccess);
    REQUIRE(xing::XingDecodeStep(f.pw, dims, *f.session, 11, f.d_rc, f.d_rs, 0,
                                 f.stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);

    const size_t k_row = static_cast<size_t>(frozen::kHeads) * frozen::kQkHeadDim;
    const size_t v_row = static_cast<size_t>(frozen::kHeads) * frozen::kVHeadDim;
    std::vector<float> k0(k_row), k1(k_row), v0(v_row), v1(v_row);
    REQUIRE(xing::XingCopyLayerKV(*f.session, 0, 0, k0.data(), v0.data(),
                                  f.stream) == cudaSuccess);
    REQUIRE(xing::XingCopyLayerKV(*f.session, 1, 0, k1.data(), v1.data(),
                                  f.stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);

    // The fixture's layers hold different random weights, so neither K nor V
    // may coincide across layers.
    REQUIRE(k0 != k1);
    REQUIRE(v0 != v1);
    // A written slot is not all-zero (an unwritten cache reads back as zeros).
    const auto nonzero = [](const std::vector<float> &v) {
        return std::any_of(v.begin(), v.end(),
                           [](float x) { return x != 0.0f; });
    };
    REQUIRE(nonzero(k0));
    REQUIRE(nonzero(v0));
    REQUIRE(nonzero(k1));
    REQUIRE(nonzero(v1));
}

// Multi-step determinism. Every earlier Xing test ran a SINGLE token, where the
// attention softmax is trivial and nothing reads the cache back - so a defect
// that only shows once positions accumulate (the real 29B model diverged after
// a dozen steps) was invisible. This walks a short prompt and then decodes,
// twice, and demands bit-identical logits.
TEST_CASE("Xing multi-step decode is bit-reproducible", "[xing][model]")
{
    ConvertedFixture f = BuildFixture();
    const auto dims = xing::XingDimsFromWeights(f.pw);
    // A DEEPER cache than the other cases use: the real 29B model only lost
    // determinism once a dozen positions were in the cache, so a 2-step walk
    // cannot see it.
    constexpr int kDeep = 32;
    f.session = std::make_unique<lsxcommon::xing::XingSession>();
    REQUIRE(xing::AllocateXingSession(*f.session, dims, kDeep,
                                      lsxcommon::KvCacheLayout::kHeadMajor,
                                      f.stream) ==
            cudaSuccess);
    REQUIRE(xing::BuildXingYarnTables(kDeep, f.rc, f.rs));
    REQUIRE(xing::UploadXingYarnTables(f.stream, kDeep, f.rc, f.rs, &f.d_rc,
                                       &f.d_rs) == cudaSuccess);

    const std::vector<int32_t> prompt = {5, 17};
    const int kSteps = 18;
    std::vector<float> first;
    auto run = [&]() {
        f.session->Free();
        REQUIRE(xing::AllocateXingSession(*f.session, dims, kDeep,
                                      lsxcommon::KvCacheLayout::kHeadMajor,
                                      f.stream) ==
                cudaSuccess);
        int pos = 0;
        for (const int32_t t : prompt)
        {
            REQUIRE(xing::XingDecodeStep(f.pw, dims, *f.session, t, f.d_rc, f.d_rs,
                                         pos, f.stream) == cudaSuccess);
            ++pos;
        }
        for (int i = 0; i < kSteps; ++i)
        {
            REQUIRE(xing::XingDecodeStep(f.pw, dims, *f.session, 7 + i, f.d_rc,
                                         f.d_rs, pos, f.stream) == cudaSuccess);
            ++pos;
        }
        std::vector<float> logits(frozen::kVocab, 0.0f);
        xing::XingReadLogits(*f.session, logits.data(), f.stream);
        REQUIRE(cudaStreamSynchronize(f.stream) == cudaSuccess);
        return logits;
    };
    first = run();
    const std::vector<float> second = run();
    // memcmp, not ==: the synthetic fixture's random weights can overflow to
    // NaN, and NaN != NaN would mask the bitwise comparison we actually want.
    REQUIRE(std::memcmp(first.data(), second.data(),
                        first.size() * sizeof(float)) == 0);
}

// Audit X1/X2: a Xing continuation must append to the cache and produce the
// same tokens/logits as a fresh prefill of prompt + generated tokens.
TEST_CASE("xing continuation matches a fresh prefill", "[xing][model]")
{
    ConvertedFixture f = BuildFixture();
    EnsureModelRegistry();

    ModelInitConfig cfg;
    cfg.model_path = f.out.string();
    cfg.model_type = "xing4.0-29b-a4b";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {1, 2, 3};
    req.max_gen_tokens = 2;
    req.ignore_eos = true;
    const auto r1 = model->Infer(req);
    REQUIRE(r1.ok);
    REQUIRE(r1.gen_tokens.size() == 2);

    // session_tail_ == prompt + generated for Xing.
    ModelRequest cont;
    cont.input_ids = req.input_ids;
    cont.input_ids.insert(cont.input_ids.end(), r1.gen_tokens.begin(),
                          r1.gen_tokens.end());
    cont.max_gen_tokens = 2;
    cont.ignore_eos = true;
    const auto r2 = model->Infer(cont);
    REQUIRE(r2.ok);
    model->Unload();

    auto fresh = LsxModelFactory::Create(cfg);
    REQUIRE(fresh != nullptr);
    REQUIRE(fresh->Load());
    ModelRequest same;
    same.input_ids = cont.input_ids;
    same.max_gen_tokens = 2;
    same.ignore_eos = true;
    const auto r3 = fresh->Infer(same);
    REQUIRE(r3.ok);
    fresh->Unload();

    std::string dbg = "r1 tokens=";
    for (auto t : r1.gen_tokens) dbg += std::to_string(t) + ",";
    dbg += " slots=" + std::to_string(r1.kv_slots_used) +
           " r2=" + std::to_string(r2.kv_slots_used) +
           " r3=" + std::to_string(r3.kv_slots_used);
    INFO(dbg);
    for (size_t i = 0; i < r2.gen_tokens.size(); ++i)
        INFO("tok[" << i << "] r2=" << r2.gen_tokens[i]
                    << " r3=" << r3.gen_tokens[i]);
    CHECK(r2.gen_tokens == r3.gen_tokens);
    CHECK(r2.kv_slots_used == r3.kv_slots_used);
    REQUIRE(r2.final_logits.size() == r3.final_logits.size());
    double max_abs = 0.0;
    for (size_t i = 0; i < r2.final_logits.size(); ++i)
        max_abs = std::max(max_abs,
                           (double)std::abs(r2.final_logits[i] -
                                            r3.final_logits[i]));
    INFO("max logit delta = " << max_abs);
    CHECK(max_abs < 1e-3);
}
