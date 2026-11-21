// Inkling-2-Mini loader-stage test: synthetic converted fixture only (the
// vault integration belongs to the selfcheck/perf suites). Pins the topology
// reader (incl. the loop fields and per-pass cache slots), the frozen-geometry
// cross-check, the pinned upload (controllers on loop layers only, merged
// message queries), and that Free releases every plane.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_loop_convert.h>
#include <lsxcommon/inkling_moe_loop_load.h>
#include <lsxcommon/schema.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "test_inkling_moe_loop_fixture.h"

namespace fs = std::filesystem;
using namespace inklinglooptest;

namespace {

// Converts the synthetic checkpoint once per case into a fresh directory.
fs::path ConvertFixture(const char *tag)
{
    const fs::path in = fs::temp_directory_path() / (std::string("inkling_loop_load_in_") + tag);
    const fs::path out = fs::temp_directory_path() / (std::string("inkling_loop_load_out_") + tag);
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);
    auto st = lsxcommon::inkling::ConvertInklingLoopModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    return out;
}

}  // namespace

TEST_CASE("inkling loop topology reader reads a converted tree",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertFixture("top");

    lsxcommon::inkling::InklingLoopTopology top;
    auto ts = lsxcommon::inkling::LoadInklingLoopTopology(out.string(), top);
    if (!ts.ok()) FAIL(ts.ToString());
    REQUIRE(top.num_layers == kLayers);
    REQUIRE(top.hidden == kHidden);
    REQUIRE(top.num_heads == kHeads);
    REQUIRE(top.num_kv_heads == kKvHeads);
    REQUIRE(top.head_dim == kHeadDim);
    REQUIRE(top.num_experts == kExperts);
    REQUIRE(top.top_k == 2);
    REQUIRE(top.moe_intermediate == kMoeInter);
    REQUIRE(top.d_rel == kDRel);
    REQUIRE(top.rel_extent == kRelExtent);
    REQUIRE(top.sliding_window == kWindow);
    REQUIRE(top.conv_kernel == kConvK);
    REQUIRE(top.attn_res_block == kAttnResBlock);
    REQUIRE(top.attn_scale_denom == static_cast<double>(kHeadDim));
    REQUIRE(top.eos_id == 200006);
    REQUIRE(top.stop_id_alt == 199999);
    REQUIRE(top.bos_id == -1);
    REQUIRE(top.pad_id == 199999);
    REQUIRE(top.full_layers.size() == 1);
    REQUIRE(top.full_layers[0] == kFullLayer);
    REQUIRE(top.sliding_layers.size() == 7);
    REQUIRE(top.IsFull(kFullLayer));
    REQUIRE_FALSE(top.IsFull(4));
    REQUIRE(top.IsBlockBoundary(0));
    REQUIRE_FALSE(top.IsBlockBoundary(1));
    REQUIRE(top.expert_nvfp4.size() == static_cast<size_t>(kLayers));
    for (int l = 0; l < kLayers; ++l)
        REQUIRE(top.expert_nvfp4[static_cast<size_t>(l)] == (l < kNovfp4Layers));

    // ── the loop core ─────────────────────────────────────────────────────
    REQUIRE(top.loop_start == kLoopStart);
    REQUIRE(top.loop_end == kLoopEnd);
    REQUIRE(top.loop_passes == kLoopPasses);
    REQUIRE(top.loop_gate_rank == kLoopGateRank);
    REQUIRE(top.LoopLayers() == kLoopLayers);
    REQUIRE(top.CacheLayers() == kLayers + (kLoopPasses - 1) * kLoopLayers);
    REQUIRE_FALSE(top.IsLoop(kLoopStart - 1));
    REQUIRE(top.IsLoop(kLoopStart));
    REQUIRE(top.IsLoop(kLoopEnd - 1));
    REQUIRE_FALSE(top.IsLoop(kLoopEnd));
    // Per-pass cache slots: pass 0 is the physical slot; extra passes append.
    for (int l = 0; l < kLayers; ++l) REQUIRE(top.CacheSlot(l, 0) == l);
    REQUIRE(top.CacheSlot(kLoopStart, 1) == kLayers);
    REQUIRE(top.CacheSlot(kLoopEnd - 1, 1) == kLayers + kLoopLayers - 1);
    fs::remove_all(out);
}

TEST_CASE("inkling loop topology rejects a wrong format tag", "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertFixture("fmt");
    // Corrupt the format tag: the loader must refuse a foreign tree.
    {
        const fs::path idx = out / "logestix" / "model.index.json";
        std::ifstream f(idx);
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        const std::string from = "logestix-inkling-2-mini";
        const auto pos = json.find(from);
        REQUIRE(pos != std::string::npos);
        json.replace(pos, from.size(), "logestix-inkling-2-small");
        std::ofstream g(idx);
        g << json;
    }
    lsxcommon::inkling::InklingLoopTopology top;
    auto ts = lsxcommon::inkling::LoadInklingLoopTopology(out.string(), top);
    REQUIRE_FALSE(ts.ok());
    fs::remove_all(out);
}

TEST_CASE("inkling loop pinned upload carries controllers and message queries",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertFixture("pin");

    lsxcommon::inkling::InklingLoopTopology top;
    REQUIRE(lsxcommon::inkling::LoadInklingLoopTopology(out.string(), top).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::inkling::InklingLoopPinnedWeights pw;
    pw.top = top;
    REQUIRE(lsxcommon::inkling::LoadInklingLoopPinnedWeights(out.string(), pw,
                                                             stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    REQUIRE(pw.embed != nullptr);
    REQUIRE(pw.head != nullptr);
    REQUIRE(pw.embed_rows == kVocab);
    REQUIRE(pw.embed_cols == kHidden);
    REQUIRE(pw.message_queries != nullptr);
    REQUIRE(pw.resident_bytes > 0);

    // The merged message queries must arrive in [q,k,v,mlp] order: copy them
    // back and compare against the fixture's source rows.
    {
        std::vector<float> mq(static_cast<size_t>(kLoopLayers) * 4 * kHidden);
        REQUIRE(cudaMemcpy(mq.data(), pw.message_queries,
                           mq.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
                cudaSuccess);
        auto src_q = synthtest::MakeF32Values(2 * kLoopLayers * kHidden,
                                              [](int64_t i) {
                                                  return 0.001f * static_cast<float>(i + 1);
                                              });
        auto src_kv = synthtest::MakeF32Values(2 * kLoopLayers * kHidden,
                                               [](int64_t i) {
                                                   return -0.001f * static_cast<float>(i + 1);
                                               });
        for (int i = 0; i < kLoopLayers; ++i) {
            for (int h = 0; h < kHidden; ++h) {
                const size_t row = static_cast<size_t>(i * 4 * kHidden + h);
                const float q = *reinterpret_cast<const float *>(
                    src_q.data() + static_cast<size_t>(2 * i * kHidden + h) * 4);
                const float mlp = *reinterpret_cast<const float *>(
                    src_q.data() + static_cast<size_t>((2 * i + 1) * kHidden + h) * 4);
                const float k = *reinterpret_cast<const float *>(
                    src_kv.data() + static_cast<size_t>(2 * i * kHidden + h) * 4);
                const float v = *reinterpret_cast<const float *>(
                    src_kv.data() + static_cast<size_t>((2 * i + 1) * kHidden + h) * 4);
                INFO("loop layer " << i << " hidden " << h);
                REQUIRE(mq[row] == q);
                REQUIRE(mq[row + static_cast<size_t>(kHidden)] == k);
                REQUIRE(mq[row + static_cast<size_t>(2 * kHidden)] == v);
                REQUIRE(mq[row + static_cast<size_t>(3 * kHidden)] == mlp);
            }
        }
    }

    // Controllers: present on loop layers, null elsewhere; shapes pinned.
    for (int l = 0; l < kLayers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        REQUIRE(lw.loop == IsLoop(l));
        for (int m = 0; m < 4; ++m) {
            const auto &g = lw.msg[m];
            INFO("layer " << l << " message " << m);
            if (IsLoop(l)) {
                REQUIRE(g.down.d != nullptr);
                REQUIRE(g.down.rows == kLoopGateRank);
                REQUIRE(g.down.cols == 2 * kHidden);
                REQUIRE(g.forget != nullptr);
                REQUIRE(g.gain != nullptr);
            } else {
                REQUIRE(g.down.d == nullptr);
                REQUIRE(g.forget == nullptr);
                REQUIRE(g.gain == nullptr);
            }
        }
    }

    // Free releases every plane: a second load/free cycle must not grow the
    // device footprint (cudaMemGetInfo delta across the cycle ~ 0).
    size_t free_before = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);
    lsxcommon::inkling::FreeInklingLoopPinnedWeights(pw);
    REQUIRE(pw.embed == nullptr);
    REQUIRE(pw.message_queries == nullptr);
    REQUIRE(pw.layers.empty());
    REQUIRE(pw.resident_bytes == 0);
    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
    // Allow a small allocator slack, but not a whole plane.
    REQUIRE(free_after + (64u << 20) >= free_before);

    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaStreamDestroy(stream);
    fs::remove_all(out);
}

TEST_CASE("inkling loop expert store reads both codecs as views",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertFixture("store");

    auto db = lsxcommon::inkling::LoadInklingLoopExpertIndex(
        out.string(), kLayers, kExperts, /*max_open=*/64);
    REQUIRE(db.num_layers == kLayers);
    REQUIRE(db.experts_per_layer == kExperts);

    // NVFP4 layer 0: packed nibbles, per-expert global scale amax/2688.
    {
        db.EnsureResident(0, 2, nullptr);
        const auto &entry = db.entries[static_cast<size_t>(0 * kExperts + 2)];
        REQUIRE(entry.host_w1 != nullptr);
        REQUIRE(entry.host_w2 != nullptr);
        REQUIRE(entry.host_w3 != nullptr);
        REQUIRE(entry.w1_bytes ==
                static_cast<int64_t>(kMoeInter) * kHidden / 2);
        REQUIRE(entry.w2_bytes ==
                static_cast<int64_t>(kHidden) * kMoeInter / 2);
        const float expect = (0.08f + 0.001f * 2.0f) / 2688.0f;
        REQUIRE(std::fabs(entry.w1_global_scale - expect) < 1e-9f);
        REQUIRE(std::fabs(entry.w3_global_scale - expect) < 1e-9f);
    }
    // MXFP8 layer kNovfp4Layers: one byte per element, global scale 1.
    {
        db.EnsureResident(kNovfp4Layers, 1, nullptr);
        const auto &entry =
            db.entries[static_cast<size_t>(kNovfp4Layers * kExperts + 1)];
        REQUIRE(entry.host_w1 != nullptr);
        REQUIRE(entry.w1_bytes ==
                static_cast<int64_t>(kMoeInter) * kHidden);
        REQUIRE(entry.w1_global_scale == 1.0f);
        REQUIRE(entry.w2_global_scale == 1.0f);
    }
    lsxcommon::inkling::FreeInklingLoopExpertIndex(db);
    fs::remove_all(out);
}