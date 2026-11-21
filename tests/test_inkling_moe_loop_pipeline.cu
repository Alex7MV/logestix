// Inkling-2-Mini generation-pipeline test: synthetic converted fixture only.
//
// The loop is the whole delta over Inkling-2-Small, so this suite pins the
// three properties a "compiles and answers" implementation can get wrong while
// still producing fluent text:
//
//   * SCHEDULING — every application runs, in the reference's order, and the
//     extra passes are NOT composed on the previous pass's output: they restart
//     from the pre-loop snapshots and receive messages, so their rows differ
//     from pass 0 at the same physical layer;
//   * PER-PASS STATE — each pass writes its own KV slot (CacheSlot(l, p)), so
//     pass 1's keys are neither pass 0's nor untouched;
//   * SEQUENCE ISOLATION — two identical requests reproduce the token stream
//     AND the logits row bit-for-bit (the conv rings and KV bookkeeping are
//     per-sequence state and are reset at every request start).
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_loop_convert.h>
#include <lsxcommon/inkling_moe_loop_inference_pipeline.h>
#include <lsxcommon/inkling_moe_loop_load.h>
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

namespace {

fs::path ConvertFixture(const char *tag)
{
    const fs::path in =
        fs::temp_directory_path() / (std::string("inkling_loop_pipe_in_") + tag);
    const fs::path out =
        fs::temp_directory_path() / (std::string("inkling_loop_pipe_out_") + tag);
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in, /*sane=*/true);
    auto st = ConvertInklingLoopModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    return out;
}

// Everything one decode session needs, built from a converted synthetic tree.
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
        if (!LoadInklingLoopTopology(dir.string(), pw.top).ok()) return false;
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        if (!LoadInklingLoopPinnedWeights(dir.string(), pw, stream)) return false;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
        db = LoadInklingLoopExpertIndex(dir.string(), kLayers, kExperts,
                                        /*max_open=*/64);
        if (db.num_layers != kLayers || db.experts_per_layer != kExperts)
            return false;
        if (!scratch.Allocate(pw.top, /*capacity=*/kwarm_capacity, stream))
            return false;
        return true;
    }

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

    static constexpr int64_t kwarm_capacity = 32;
};

// Row of `out_states` holding the residual stream after the application of
// physical layer `l` in loop pass `p`.
int TraceRow(const InklingLoopTopology &top, int l, int p)
{
    if (!top.IsLoop(l)) return 1 + l;
    return 1 + top.loop_start + p * top.LoopLayers() + (l - top.loop_start);
}

bool AllFinite(const std::vector<float> &v)
{
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace

TEST_CASE("inkling loop trace runs every application in execution order",
          "[inkling][pipeline]")
{
    Harness h;
    if (!h.Open("trace")) SKIP("synthetic-geometry fixture not enabled");
    const InklingLoopTopology &top = h.pw.top;
    const int H = top.hidden;

    const std::vector<int32_t> prompt = {5, 17, 231};
    std::vector<float> states;
    std::vector<float> logits;
    cudaError_t e = RunInklingLoopForwardTrace(h.stream, h.pw, h.db, h.session,
                                               h.scratch, prompt, 30.0f,
                                               kUnpadded, states, &logits);
    REQUIRE(e == cudaSuccess);

    // (CacheLayers() + 1) rows: row 0 is post-embed_norm, then one row per
    // application (pre-loop, every pass of the loop core, post-loop).
    REQUIRE(top.CacheLayers() == kLayers + (kLoopPasses - 1) * kLoopLayers);
    REQUIRE(states.size() ==
            static_cast<size_t>(top.CacheLayers() + 1) * H);
    REQUIRE(AllFinite(states));

    auto row = [&](int idx) {
        return std::vector<float>(states.begin() + static_cast<int64_t>(idx) * H,
                                  states.begin() +
                                      static_cast<int64_t>(idx + 1) * H);
    };
    auto max_abs_diff = [&](int a, int b) {
        const auto ra = row(a);
        const auto rb = row(b);
        float m = 0.0f;
        for (int i = 0; i < H; ++i)
            m = std::max(m, std::fabs(ra[static_cast<size_t>(i)] -
                                      rb[static_cast<size_t>(i)]));
        return m;
    };

    // A layer application must move the residual stream.
    REQUIRE(max_abs_diff(0, 1) > 1e-4f);
    // Post-embed_norm is not the hidden entering layer 0's successor: every
    // application differs from the one before it in the pre-loop stack.
    for (int l = 0; l + 1 < top.num_layers; ++l)
        REQUIRE(max_abs_diff(TraceRow(top, l, 0), TraceRow(top, l + 1, 0)) >
                1e-6f);

    // The extra passes restart from the pre-loop snapshots and are driven by
    // messages, so at the SAME physical layer their row must differ from pass
    // 0's — at EVERY loop layer, because the gate is applied per layer and each
    // layer reads its own message query row.
    //
    // The comparison is `> 0` rather than a magnitude threshold on purpose: the
    // kernels are deterministic and a pass that re-runs the layer with the gate
    // bypassed reproduces the row BIT-FOR-BIT. The synthetic fixture's weights
    // make the gate's effect on the residual small in absolute terms (it acts
    // through attention logits), so a magnitude threshold here would pin the
    // fixture's scale instead of the scheduling.
    for (int b = 1; b < kLoopPasses; ++b) {
        float total = 0.0f;
        for (int l = top.loop_start; l < top.loop_end; ++l) {
            const float d = max_abs_diff(TraceRow(top, l, 0), TraceRow(top, l, b));
            INFO("pass " << b << " layer " << l << " delta " << d);
            REQUIRE(d > 0.0f);
            total += d;
        }
        INFO("pass " << b << " total delta " << total);
        REQUIRE(total > 1e-6f);
    }

    // Logits are the sliced, softcapped row.
    REQUIRE(logits.size() == static_cast<size_t>(kUnpadded));
    REQUIRE(AllFinite(logits));
    float max_abs = 0.0f;
    for (float v : logits) max_abs = std::max(max_abs, std::fabs(v));
    REQUIRE(max_abs <= 30.0f + 1e-3f);  // the logit softcap bounds the row

    h.Close();
}

TEST_CASE("inkling loop gives every pass its own KV slot",
          "[inkling][pipeline]")
{
    Harness h;
    if (!h.Open("kv")) SKIP("synthetic-geometry fixture not enabled");
    const InklingLoopTopology &top = h.pw.top;

    const std::vector<int32_t> prompt = {7, 9};
    std::vector<float> states;
    REQUIRE(RunInklingLoopForwardTrace(h.stream, h.pw, h.db, h.session, h.scratch,
                                       prompt, 30.0f, kUnpadded, states,
                                       nullptr) == cudaSuccess);

    REQUIRE(cudaStreamSynchronize(h.stream) == cudaSuccess);
    // Pass 0 must use the physical slot and pass p the appended one; both must
    // hold this token's keys, and they must not be the same bytes (the pass
    // restarted from the pre-loop hidden, so its k/v differ).
    const int l = top.loop_start;  // first loop layer
    const int64_t row_elems = h.session.geom.pos_stride;
    REQUIRE(h.session.geom.layers == top.CacheLayers());
    std::vector<__half> a(static_cast<size_t>(row_elems));
    std::vector<__half> b(static_cast<size_t>(row_elems));
    auto slot_row = [&](int slot, std::vector<__half> &dst) {
        const int64_t off = static_cast<int64_t>(slot) * h.session.geom.LayerStride();
        return cudaMemcpy(dst.data(), h.session.d_k + off,
                          dst.size() * sizeof(__half),
                          cudaMemcpyDeviceToHost);
    };
    REQUIRE(slot_row(top.CacheSlot(l, 0), a) == cudaSuccess);
    REQUIRE(slot_row(top.CacheSlot(l, 1), b) == cudaSuccess);
    int nonzero = 0;
    int differing = 0;
    for (int64_t i = 0; i < row_elems; ++i) {
        if (__half2float(a[static_cast<size_t>(i)]) != 0.0f) ++nonzero;
        if (__half2float(a[static_cast<size_t>(i)]) !=
            __half2float(b[static_cast<size_t>(i)]))
            ++differing;
    }
    REQUIRE(nonzero > 0);
    REQUIRE(differing > 0);
    // Pass 1 stores at CacheSlot(l,1) = num_layers + (l - loop_start), i.e.
    // strictly past the physical stack.
    REQUIRE(top.CacheSlot(l, 1) >= top.num_layers);
    REQUIRE(top.CacheSlot(l, 1) == top.num_layers + (l - top.loop_start));

    h.Close();
}

TEST_CASE("inkling loop reproduces a request bit-for-bit after a reset",
          "[inkling][pipeline]")
{
    Harness h;
    if (!h.Open("det")) SKIP("synthetic-geometry fixture not enabled");

    const std::vector<int32_t> prompt = {3, 11, 42, 250};
    const std::vector<int32_t> eos = {kUnpadded - 1};
    lsxcommon::sample::GreedySampler sampler;

    InklingLoopGenerationResult first;
    cudaError_t e = RunInklingLoopGeneration(h.stream, h.pw, h.db, h.session,
                                             h.scratch, prompt, /*max_gen=*/4,
                                             eos, 30.0f, kUnpadded, sampler,
                                             first, nullptr);
    REQUIRE(e == cudaSuccess);
    REQUIRE(first.ok);
    REQUIRE(!first.tokens.empty());
    REQUIRE(static_cast<int>(first.tokens.size()) <= 4);

    InklingLoopGenerationResult second;
    e = RunInklingLoopGeneration(h.stream, h.pw, h.db, h.session, h.scratch,
                                 prompt, /*max_gen=*/4, eos, 30.0f, kUnpadded,
                                 sampler, second, nullptr);
    REQUIRE(e == cudaSuccess);
    REQUIRE(second.ok);
    REQUIRE(second.tokens == first.tokens);
    REQUIRE(second.final_logits.size() == first.final_logits.size());
    // Bit-exact, not merely close: any leaked conv/KV state between the two
    // requests shows up as a different row.
    REQUIRE(std::memcmp(second.final_logits.data(), first.final_logits.data(),
                        first.final_logits.size() * sizeof(float)) == 0);
    for (int32_t t : first.tokens) {
        REQUIRE(t >= 0);
        REQUIRE(t < kUnpadded);
    }
    h.Close();
}
