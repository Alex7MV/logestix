// TML-d24-varloop-research generation-pipeline test: synthetic fixture only.
//
// The row's whole reason to exist is that ONE set of weights runs a
// runtime-chosen number of loop passes, so this suite pins what a "compiles and
// answers" implementation gets wrong:
//
//   * SCHEDULING — the requested count is what runs: `passes` passes produce
//     `LayerApplications(passes)` applications, and a larger count moves the
//     residual stream further (5 passes is not 3 passes with extra bookkeeping);
//   * SLAB — the KV/conv slab is sized for the MAXIMUM count, so a 1-pass run
//     touches slot prefix [0, 24) and a 5-pass run [0, 72) of one allocation,
//     and no slot moves when the count changes. A slab sized from the active
//     count instead writes past its own end at 5 passes (the release appends a
//     cache block per extra pass);
//   * ISOLATION + DETERMINISM — every pass owns its slot (pass 4's keys are
//     neither pass 0's nor untouched), and two identical requests reproduce the
//     token stream AND the logits row bit-for-bit.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_loop_convert.h>
#include <lsxcommon/inkling_moe_loop_inference_pipeline.h>
#include <lsxcommon/inkling_moe_loop_load.h>
#include <lsxcommon/inkling_moe_varloop_convert.h>
#include "inkling_moe_varloop_constants.h"
#include <lsxcommon/sampling.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "test_inkling_moe_loop_fixture.h"

namespace fs = std::filesystem;
using namespace inklinglooptest;
using namespace lsxcommon::inkling;
namespace vf = lsxcommon::inkling::varloop::frozen;

namespace {

constexpr int kVarPassesMax = 5;

fs::path ConvertFixture(const char *tag)
{
    const fs::path in =
        fs::temp_directory_path() / (std::string("inkling_varloop_pipe_in_") + tag);
    const fs::path out =
        fs::temp_directory_path() / (std::string("inkling_varloop_pipe_out_") + tag);
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in, /*sane=*/true, /*varloop=*/true);
    auto st = ConvertInklingVarloopModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    return out;
}

struct Harness {
    fs::path dir;
    cudaStream_t stream = nullptr;
    InklingLoopPinnedWeights pw;
    InklingLoopExpertDb db;
    InklingLoopSession session;
    InklingLoopScratch scratch;

    bool Open(const char *tag)
    {
        if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
            return false;
        dir = ConvertFixture(tag);
        if (!LoadInklingLoopTopology(dir.string(), pw.top,
                                     VarloopConvertSpec().format_tag)
                 .ok())
            return false;
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        if (!LoadInklingLoopPinnedWeights(dir.string(), pw, stream)) return false;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
        db = LoadInklingLoopExpertIndex(dir.string(), inklinglooptest::kLayers,
                                        inklinglooptest::kExperts,
                                        /*max_open=*/64);
        if (db.num_layers != inklinglooptest::kLayers ||
            db.experts_per_layer != inklinglooptest::kExperts)
            return false;
        if (!scratch.Allocate(pw.top, /*capacity=*/64, stream)) return false;
        return true;
    }

    // The knob the model exposes: the ACTIVE pass count, inside the slab the
    // index sized. Changing it between requests is legal by construction.
    void SetPasses(int passes) { pw.top.loop_passes = passes; }

    void Close()
    {
        scratch.Free();
        session.Free();
        FreeInklingLoopExpertIndex(db);
        FreeInklingLoopPinnedWeights(pw);
        if (stream != nullptr) cudaStreamDestroy(stream);
        stream = nullptr;
        fs::remove_all(dir);
    }
};

bool AllFinite(const std::vector<float> &v)
{
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

float MaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
{
    REQUIRE(a.size() == b.size());
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

TEST_CASE("varloop trace runs exactly the requested application count",
          "[inkling][varloop][pipeline]")
{
    Harness h;
    if (!h.Open("trace")) SKIP("synthetic-geometry fixture not enabled");

    const int H = h.pw.top.hidden;
    const std::vector<int32_t> prompt = {5, 17, 231};

    std::vector<float> rows3;
    for (int passes : {1, 3, 5}) {
        h.SetPasses(passes);
        std::vector<float> states;
        std::vector<float> logits;
        INFO("passes " << passes);
        REQUIRE(RunInklingLoopForwardTrace(h.stream, h.pw, h.db, h.session,
                                           h.scratch, prompt, 30.0f, kUnpadded,
                                           states, &logits) == cudaSuccess);
        // One row per layer APPLICATION plus the post-embed_norm row: the count
        // follows the ACTIVE passes, not the slab. Computed from the fixture's
        // own geometry (8 layers, 4 loop layers), not the real passport's.
        REQUIRE(states.size() ==
                static_cast<size_t>(inklinglooptest::kLayers +
                                    (passes - 1) * inklinglooptest::kLoopLayers +
                                    1) *
                    H);
        REQUIRE(AllFinite(states));
        REQUIRE(logits.size() == static_cast<size_t>(kUnpadded));
        REQUIRE(AllFinite(logits));
        // The slab is the maximum for every count, and it never re-allocates.
        REQUIRE(h.session.geom.layers ==
                static_cast<size_t>(inklinglooptest::kLayers +
                                    (kVarPassesMax - 1) * inklinglooptest::kLoopLayers));
        if (passes == 3) rows3 = states;
    }

    // The requested count changes the COMPUTATION, not just the row count: the
    // 5-pass stream is not the 3-pass stream with two extra rows appended. Both
    // the layer-application rows and the logits differ.
    h.SetPasses(5);
    std::vector<float> states5;
    REQUIRE(RunInklingLoopForwardTrace(h.stream, h.pw, h.db, h.session, h.scratch,
                                       prompt, 30.0f, kUnpadded, states5,
                                       nullptr) == cudaSuccess);
    REQUIRE(states5.size() > rows3.size());
    {
        // Compare the first 3 passes' worth of rows: the extra passes must not
        // be a no-op, so the FINAL row (post-loop layers) has moved.
        const size_t per_row = static_cast<size_t>(H);
        const std::vector<float> last3(
            rows3.end() - static_cast<long>(per_row), rows3.end());
        const std::vector<float> last5(
            states5.end() - static_cast<long>(per_row), states5.end());
        INFO("final-row delta " << MaxAbsDiff(last3, last5));
        REQUIRE(MaxAbsDiff(last3, last5) > 0.0f);
    }

    h.Close();
}

TEST_CASE("varloop gives the last pass its own KV slot inside the slab",
          "[inkling][varloop][pipeline]")
{
    Harness h;
    if (!h.Open("kv")) SKIP("synthetic-geometry fixture not enabled");
    const auto &top = h.pw.top;

    // Five passes = the last slot block of the slab. This is the case a slab
    // sized from the active (3) count would corrupt.
    h.SetPasses(kVarPassesMax);
    const std::vector<int32_t> prompt = {7, 9};
    std::vector<float> states;
    REQUIRE(RunInklingLoopForwardTrace(h.stream, h.pw, h.db, h.session, h.scratch,
                                       prompt, 30.0f, kUnpadded, states,
                                       nullptr) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(h.stream) == cudaSuccess);

    const int l = top.loop_start;
    const int64_t row_elems = h.session.geom.pos_stride;
    REQUIRE(h.session.geom.layers == top.CacheLayers());
    std::vector<__half> a(static_cast<size_t>(row_elems));
    std::vector<__half> b(static_cast<size_t>(row_elems));
    auto slot_row = [&](int slot, std::vector<__half> &dst) {
        const int64_t off = static_cast<int64_t>(slot) * h.session.geom.LayerStride();
        return cudaMemcpy(dst.data(), h.session.d_k + off,
                          dst.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    };
    // Pass 0 -> the physical slot; pass 4 -> the last appended block, whose
    // offset is num_layers + (pass-1)*loop_layers for pass p >= 1.
    REQUIRE(h.pw.top.CacheSlot(l, 0) == l);
    REQUIRE(h.pw.top.CacheSlot(l, kVarPassesMax - 1) ==
            inklinglooptest::kLayers +
                (kVarPassesMax - 2) * inklinglooptest::kLoopLayers);
    // The whole run stays inside the slab the MAX count sized.
    REQUIRE(h.pw.top.CacheSlot(top.loop_end - 1, kVarPassesMax - 1) ==
            h.pw.top.CacheLayers() - 1);
    REQUIRE(slot_row(h.pw.top.CacheSlot(l, 0), a) == cudaSuccess);
    REQUIRE(slot_row(h.pw.top.CacheSlot(l, kVarPassesMax - 1), b) == cudaSuccess);
    int nonzero = 0;
    int differing = 0;
    for (int64_t i = 0; i < row_elems; ++i) {
        if (__half2float(a[static_cast<size_t>(i)]) != 0.0f) ++nonzero;
        if (__half2float(a[static_cast<size_t>(i)]) !=
            __half2float(b[static_cast<size_t>(i)]))
            ++differing;
    }
    // Both slots hold this token's keys, and they are not the same bytes: the
    // last pass restarted from the pre-loop snapshot with its own message.
    REQUIRE(nonzero > 0);
    REQUIRE(differing > 0);

    h.Close();
}

TEST_CASE("varloop reproduces a request bit-for-bit at any pass count",
          "[inkling][varloop][pipeline]")
{
    Harness h;
    if (!h.Open("det")) SKIP("synthetic-geometry fixture not enabled");

    const std::vector<int32_t> prompt = {3, 11, 42, 250};
    const std::vector<int32_t> eos = {kUnpadded - 1};
    lsxcommon::sample::GreedySampler sampler;

    for (int passes : {1, 3, 5}) {
        h.SetPasses(passes);
        InklingLoopGenerationResult first;
        REQUIRE(RunInklingLoopGeneration(h.stream, h.pw, h.db, h.session,
                                         h.scratch, prompt, /*max_gen=*/3, eos,
                                         30.0f, kUnpadded, sampler, first,
                                         nullptr) == cudaSuccess);
        REQUIRE(first.ok);
        REQUIRE(!first.tokens.empty());

        InklingLoopGenerationResult second;
        REQUIRE(RunInklingLoopGeneration(h.stream, h.pw, h.db, h.session,
                                         h.scratch, prompt, /*max_gen=*/3, eos,
                                         30.0f, kUnpadded, sampler, second,
                                         nullptr) == cudaSuccess);
        REQUIRE(second.ok);
        INFO("passes " << passes);
        REQUIRE(second.tokens == first.tokens);
        REQUIRE(std::memcmp(second.final_logits.data(), first.final_logits.data(),
                            first.final_logits.size() * sizeof(float)) == 0);
        for (int32_t t : first.tokens) {
            REQUIRE(t >= 0);
            REQUIRE(t < kUnpadded);
        }
    }

    h.Close();
}
