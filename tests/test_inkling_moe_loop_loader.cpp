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
        const fs::path idx = out / "model.index.json";
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

// The device-resident slab must hold a byte-exact copy of every expert block:
// the GEMV then reads it in place, so a misplaced byte becomes wrong text with
// no staging step left to catch it. This copies each block back and compares it
// against the store's planes.
TEST_CASE("inkling loop device expert slab is a byte-exact copy of the store",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertFixture("devslab");

    auto db = lsxcommon::inkling::LoadInklingLoopExpertIndex(
        out.string(), kLayers, kExperts, /*max_open=*/4096,
        /*warm_on_load=*/true);
    lsxcommon::inkling::InklingLoopTopology top;
    REQUIRE(lsxcommon::inkling::LoadInklingLoopTopology(out.string(), top)
                .ok());

    auto cache = lsxcommon::inkling::LoadInklingLoopDeviceExperts(
        db, top, kMoeInter, kHidden, /*stream=*/nullptr);
    REQUIRE(cache.Enabled());
    REQUIRE(cache.experts ==
            static_cast<size_t>(kLayers) * static_cast<size_t>(kExperts));
    REQUIRE(cache.total_bytes > 0);

    for (int layer = 0; layer < kLayers; ++layer) {
        const bool mxfp8 = top.ExpertIsFp8(layer);
        const auto off = lsxcommon::inkling::InklingLoopExpertPlaneOffsets(
            kMoeInter, kHidden, mxfp8);
        REQUIRE(cache.block_bytes[static_cast<size_t>(layer)] ==
                off.block_bytes);
        for (int e = 0; e < kExperts; ++e) {
            db.EnsureResident(layer, e, nullptr);
            const auto &en =
                db.entries[static_cast<size_t>(layer * kExperts + e)];
            const int64_t at = cache.ExpertOffset(layer, e);
            REQUIRE(at >= 0);
            // One plane at a time is enough to catch an off-by-one in the
            // layer/plane arithmetic, and keeps the fixture cheap.
            for (const auto &probe :
                 {std::pair<int64_t, const uint8_t *>{off.w1, en.host_w1},
                  std::pair<int64_t, const uint8_t *>{off.w2, en.host_w2},
                  std::pair<int64_t, const uint8_t *>{off.w3, en.host_w3},
                  std::pair<int64_t, const uint8_t *>{off.w1_scales,
                                                      en.host_w1_scales},
                  std::pair<int64_t, const uint8_t *>{off.w2_scales,
                                                      en.host_w2_scales},
                  std::pair<int64_t, const uint8_t *>{off.w3_scales,
                                                      en.host_w3_scales}}) {
                const int64_t bytes = mxfp8
                    ? (probe.first == off.w1_scales || probe.first == off.w2_scales ||
                       probe.first == off.w3_scales ? int64_t(kMoeInter) * kHidden / 16
                                                    : int64_t(kMoeInter) * kHidden)
                    : (probe.first == off.w1_scales || probe.first == off.w2_scales ||
                       probe.first == off.w3_scales ? int64_t(kMoeInter) * kHidden / 16
                                                    : int64_t(kMoeInter) * kHidden / 2);
                REQUIRE(probe.second != nullptr);
                std::vector<uint8_t> got(static_cast<size_t>(bytes));
                REQUIRE(cudaMemcpy(got.data(), cache.base + at + probe.first,
                                   static_cast<size_t>(bytes),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                INFO("layer " << layer << " expert " << e << " plane at "
                              << probe.first);
                REQUIRE(std::memcmp(got.data(), probe.second,
                                    static_cast<size_t>(bytes)) == 0);
            }
        }
    }
    cache.Free();
    REQUIRE_FALSE(cache.Enabled());
    lsxcommon::inkling::FreeInklingLoopExpertIndex(db);
    fs::remove_all(out);
}

// The single-DMA staging path (StageExperts) copies ONE aligned block per
// expert instead of six planes, which is only legal if the pinned store lays an
// expert out contiguously in exactly the order the device slab expects. That is
// a cross-header contract — ExpertStore::PinChunked decides the offsets, the
// inference pipeline derives the pointer planes from them — so it is pinned
// here: if either side changes, this fails instead of silently multiplying
// padding bytes into a weight plane.
TEST_CASE("inkling loop pinned expert store is one contiguous block per expert",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path out = ConvertFixture("pin");

    // PinChunked only sees entries that are already parsed, so the pinning
    // loader warms first (the vault path does the same).
    auto db = lsxcommon::inkling::LoadInklingLoopExpertIndex(
        out.string(), kLayers, kExperts, /*max_open=*/4096,
        /*warm_on_load=*/true);
    REQUIRE(db.PinChunked<lsxcommon::inkling::InklingLoopExpertDb::
                              PinLayout::kStream>());

    for (int layer : {0, kNovfp4Layers}) {
        const auto offsets = lsxcommon::inkling::InklingLoopExpertPlaneOffsets(
            kMoeInter, kHidden, /*mxfp8=*/layer >= kNovfp4Layers);
        for (int e = 0; e < kExperts; ++e) {
            db.EnsureResident(layer, e, nullptr);
            const auto &entry =
                db.entries[static_cast<size_t>(layer * kExperts + e)];
            REQUIRE(entry.host_w1 != nullptr);
            INFO("layer " << layer << " expert " << e);
            REQUIRE(entry.w1_bytes ==
                    static_cast<int64_t>(kMoeInter) * kHidden /
                        (layer < kNovfp4Layers ? 2 : 1));
            // The pinned block starts at host_w1 and every other plane sits at
            // its documented offset from that base.
            REQUIRE(entry.host_w3 == entry.host_w1 + offsets.w3);
            REQUIRE(entry.host_w2 == entry.host_w1 + offsets.w2);
            REQUIRE(entry.host_w1_scales == entry.host_w1 + offsets.w1_scales);
            REQUIRE(entry.host_w3_scales == entry.host_w1 + offsets.w3_scales);
            REQUIRE(entry.host_w2_scales == entry.host_w1 + offsets.w2_scales);
            // ... and the last plane ends inside the block, with less than one
            // alignment step of slack, so one DMA of `block_bytes` carries every
            // byte StageExperts reads.
            INFO("w2_scales " << offsets.w2_scales << " + "
                              << entry.w2_scales_bytes << " vs block "
                              << offsets.block_bytes);
            REQUIRE(offsets.w2_scales + entry.w2_scales_bytes <=
                    offsets.block_bytes);
            REQUIRE(offsets.block_bytes -
                        (offsets.w2_scales + entry.w2_scales_bytes) <
                    256);
        }
    }
    lsxcommon::inkling::FreeInklingLoopExpertIndex(db);
    fs::remove_all(out);
}