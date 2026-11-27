// Inkling-2-Small loader-stage test.
//
// Two halves:
//   * a synthetic half that pins the topology reader against a converted tree
//     written by the converter under LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1;
//   * a real-weight half that reads ONE light file (the converted
//     `logestix/model.index.json`) plus the byte sizes of two expert records,
//     and SKIPs when the vault is absent. No weight is loaded into VRAM here.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_convert.h>
#include <lsxcommon/inkling_moe_load.h>
#include <lsxcommon/schema.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "test_inkling_fixture.h"

#include "../libs/lsxcommon/src/inkling_moe_constants.h"

namespace fs = std::filesystem;
using namespace inklingtest;

// ── A3.1 probe: array new/delete interposition ─────────────────────────────
// Tracks live array allocations: pointer → byte size, plus a per-size live
// count. Global interposition affects the whole test binary, so the probe
// reads DELTAS around load/free cycles, never absolute counts. Both delete
// forms are intercepted (the unsized one looks the size up in the map), so
// the accounting is exact regardless of which form the compiler emits.
namespace {
std::mutex g_arr_mutex;
std::map<void *, std::size_t> g_arr_sizes;
std::map<std::size_t, std::int64_t> g_arr_live;

std::int64_t LiveIntArrayCount(std::size_t bytes)
{
    std::lock_guard<std::mutex> lk(g_arr_mutex);
    auto it = g_arr_live.find(bytes);
    return it == g_arr_live.end() ? 0 : it->second;
}

void ArrFree(void *p)
{
    if (!p) return;
    {
        std::lock_guard<std::mutex> lk(g_arr_mutex);
        auto it = g_arr_sizes.find(p);
        if (it != g_arr_sizes.end()) {
            g_arr_live[it->second] -= 1;
            g_arr_sizes.erase(it);
        }
    }
    std::free(p);
}
}  // namespace

void *operator new[](std::size_t n)
{
    void *p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    std::lock_guard<std::mutex> lk(g_arr_mutex);
    g_arr_sizes[p] = n;
    g_arr_live[n] += 1;
    return p;
}

void operator delete[](void *p) noexcept { ArrFree(p); }

void operator delete[](void *p, std::size_t) noexcept { ArrFree(p); }

namespace {

std::string VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_INKLING_2_SMALL_ROOT"))
        return env;
    return "/mnt/fast_storage/lsx_model/thinkingmachines/Inkling-2-Small";
}

}  // namespace

TEST_CASE("inkling topology reader reads a converted tree",
          "[inkling][load]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path in = fs::temp_directory_path() / "inkling_load_in";
    const fs::path out = fs::temp_directory_path() / "inkling_load_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);

    auto st = lsxcommon::inkling::ConvertInklingModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());

    lsxcommon::inkling::InklingTopology top;
    auto ts = lsxcommon::inkling::LoadInklingTopology(out.string(), top);
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
    REQUIRE(top.pad_id == -1);
    REQUIRE(top.full_layers.size() == 1);
    REQUIRE(top.full_layers[0] == kFullLayer);
    REQUIRE(top.sliding_layers.size() == 5);
    REQUIRE(top.IsFull(kFullLayer));
    REQUIRE_FALSE(top.IsFull(4));
    REQUIRE(top.IsBlockBoundary(0));
    REQUIRE_FALSE(top.IsBlockBoundary(1));
    REQUIRE(top.FullLayerIndex(kFullLayer) == 0);
    REQUIRE(top.FullLayerIndex(0) == -1);
    REQUIRE(top.expert_nvfp4.size() == static_cast<size_t>(kLayers));
    for (int l = 0; l < kLayers; ++l)
        REQUIRE(top.expert_nvfp4[static_cast<size_t>(l)] == (l < kNovfp4Layers));

    // Two expert encodings, one record SCHEMA: NVFP4 layers pack nibbles (two
    // elements per byte), MXFP8 layers keep one E4M3 byte per element with the
    // source's per-32 ue8m0 exponent duplicated to a per-16 count.
    for (int l : {0, kLayers - 1}) {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            out / "experts" / ("layer_" + std::to_string(l)) /
            "exp_0.arrow");
        REQUIRE(batch.ok());
        // scale_block = 0: layer 4 is one of the MXFP8 layers, whose plane is
        // one E4M3 byte per element (not packed nibbles), so the per-16
        // heuristic would reject it by 2x. The exact plane sizes are asserted
        // below instead, which is the stronger check.
        const bool fp8_layer = (l >= kNovfp4Layers);
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch, fp8_layer ? 0 : 16);
        if (!pe.ok()) FAIL(pe.status().ToString());
        // NVFP4 packs two elements per byte; the MXFP8 plane is one byte each.
        const size_t gate_bytes = fp8_layer
                                      ? static_cast<size_t>(kMoeInter) * kHidden
                                      : static_cast<size_t>(kMoeInter) * kHidden / 2;
        const size_t down_bytes = fp8_layer
                                      ? static_cast<size_t>(kHidden) * kMoeInter
                                      : static_cast<size_t>(kHidden) * kMoeInter / 2;
        REQUIRE(pe->w1_packed.size() == gate_bytes);
        REQUIRE(pe->w3_packed.size() == gate_bytes);
        REQUIRE(pe->w2_packed.size() == down_bytes);
        REQUIRE(pe->w1_scales.size() ==
                static_cast<size_t>(kMoeInter) * (kHidden / 16));
        REQUIRE(pe->w2_scales.size() ==
                static_cast<size_t>(kHidden) * (kMoeInter / 16));
    }
}

TEST_CASE("inkling pinned load releases the query-offset array (A3.1)",
          "[inkling][load][leak]")
{
    // LoadInklingPinnedWeights allocates pw.attn_query_offset with
    // new int[num_layers]; FreeInklingPinnedWeights must release it.
    // The array is tiny (kLayers*4 B), so RSS probes cannot see it —
    // interpose the array new/delete pair in this TU and count live
    // allocations of exactly that size across load/free cycles.
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path in = fs::temp_directory_path() / "inkling_leak_in";
    const fs::path out = fs::temp_directory_path() / "inkling_leak_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);
    auto st = lsxcommon::inkling::ConvertInklingModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());

    lsxcommon::inkling::InklingTopology top;
    REQUIRE(lsxcommon::inkling::LoadInklingTopology(out.string(), top).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    const size_t want_bytes = static_cast<size_t>(top.num_layers) * sizeof(int);
    // Warm-up cycle: settle any one-time allocations of the same size.
    {
        lsxcommon::inkling::InklingPinnedWeights warm;
        warm.top = top;
        REQUIRE(lsxcommon::inkling::LoadInklingPinnedWeights(out.string(), warm,
                                                             stream));
        lsxcommon::inkling::FreeInklingPinnedWeights(warm);
    }
    const std::int64_t baseline = LiveIntArrayCount(want_bytes);
    for (int i = 0; i < 3; ++i) {
        lsxcommon::inkling::InklingPinnedWeights pw;
        pw.top = top;
        REQUIRE(lsxcommon::inkling::LoadInklingPinnedWeights(out.string(), pw,
                                                             stream));
        REQUIRE(pw.attn_query_offset != nullptr);
        lsxcommon::inkling::FreeInklingPinnedWeights(pw);
        INFO("cycle " << i << ": live int[" << top.num_layers
                      << "] delta = "
                      << LiveIntArrayCount(want_bytes) - baseline);
        REQUIRE(LiveIntArrayCount(want_bytes) == baseline);
    }
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaStreamDestroy(stream);
    fs::remove_all(in);
    fs::remove_all(out);
}

TEST_CASE("inkling real vault expert store reads both codecs", "[inkling][load]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "model.index.json"))
        SKIP("converted Inkling-2-Small tree absent: " + root.string());

    lsxcommon::inkling::InklingTopology top;
    auto st = lsxcommon::inkling::LoadInklingTopology(root.string(), top);
    if (!st.ok()) FAIL(st.ToString());

    auto db = lsxcommon::inkling::LoadInklingExpertIndex(
        root.string(), top.num_layers, top.num_experts, 64, false);
    REQUIRE(db.error_msg.empty());
    for (int layer : {0, top.num_layers - 1}) {
        CAPTURE(layer);
        db.EnsureResident(layer, 84, nullptr);
        const auto &e = db.entries[static_cast<size_t>(layer) *
                                       static_cast<size_t>(top.num_experts) + 84];
        // Both codecs must be readable through the same store: the NVFP4 plane
        // packs nibbles, the MXFP8 plane is one E4M3 byte per element.
        REQUIRE(e.host_w1 != nullptr);
        REQUIRE(e.host_w2 != nullptr);
        REQUIRE(e.host_w3 != nullptr);
        REQUIRE(e.host_w1_scales != nullptr);
        REQUIRE(e.host_w2_scales != nullptr);
        REQUIRE(e.host_w3_scales != nullptr);
        const bool fp8 = top.ExpertIsFp8(layer);
        CAPTURE(fp8);
        const int64_t I = top.moe_intermediate;
        const int64_t H = top.hidden;
        REQUIRE(e.w1_bytes == (fp8 ? I * H : I * H / 2));
        REQUIRE(e.w2_bytes == (fp8 ? H * I : H * I / 2));
        REQUIRE(e.w1_scales_bytes == I * (H / 16));
        REQUIRE(e.w2_scales_bytes == H * (I / 16));
    }
    lsxcommon::inkling::FreeInklingExpertIndex(db);
}

TEST_CASE("inkling real vault index matches the frozen passport",
          "[inkling][load]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "model.index.json"))
        SKIP("converted Inkling-2-Small tree absent: " + root.string());

    lsxcommon::inkling::InklingTopology top;
    auto st = lsxcommon::inkling::LoadInklingTopology(root.string(), top);
    if (!st.ok()) FAIL(st.ToString());

    REQUIRE(top.format == "logestix-inkling-2-small");
    REQUIRE(top.source_model_type == "inkling2");
    REQUIRE(top.tokenizer_kind == 2);

    // The whole point of the passport: the real tree must match it exactly.
    REQUIRE(lsxcommon::inkling::InklingTopologyMatchesFrozen(top));

    namespace frozen = lsxcommon::inkling::frozen;
    REQUIRE(top.num_layers == frozen::kLayers);
    REQUIRE(top.hidden == frozen::kHidden);
    REQUIRE(top.vocab == frozen::kVocab);
    REQUIRE(top.unpadded_vocab == frozen::kBaseVocab);
    REQUIRE(top.num_heads == frozen::kHeads);
    REQUIRE(top.num_kv_heads == frozen::kKvHeads);
    REQUIRE(top.head_dim == frozen::kHeadDim);
    REQUIRE(top.d_rel == frozen::kDRel);
    REQUIRE(top.rel_extent == frozen::kRelExtent);
    REQUIRE(top.sliding_window == frozen::kSlidingWindow);
    REQUIRE(top.conv_kernel == frozen::kConvKernel);
    REQUIRE(top.attn_res_block == frozen::kAttnResBlock);
    REQUIRE(top.num_experts == frozen::kExperts);
    REQUIRE(top.top_k == frozen::kTopK);
    REQUIRE(top.moe_intermediate == frozen::kMoeIntermediate);
    REQUIRE(top.shared_experts == 0);
    REQUIRE(top.attn_scale_denom == frozen::kAttnScaleDenom);
    REQUIRE(top.qk_l2_clip == frozen::kQkL2Clip);
    REQUIRE(top.rel_l2_clip == frozen::kRelL2Clip);
    REQUIRE(top.log_scaling_n_floor == frozen::kLogScaleFloor);
    REQUIRE(top.eos_id == frozen::kEosId);
    REQUIRE(top.bos_id == -1);
    REQUIRE(top.full_layers.size() == static_cast<size_t>(frozen::kFullLayerCount));
    for (size_t i = 0; i < frozen::kFullLayerCount; ++i)
        REQUIRE(top.full_layers[i] == frozen::kFullLayers[i]);
    REQUIRE(top.expert_nvfp4.size() == static_cast<size_t>(frozen::kLayers));
    int nvfp4_layers = 0;
    for (bool b : top.expert_nvfp4) nvfp4_layers += b ? 1 : 0;
    REQUIRE(nvfp4_layers == frozen::kNovfp4ExpertLayerCount);

    // One expert record of an NVFP4-source layer: the pass-through must keep
    // the checkpoint's own byte count (I*H/2 packed, I*H/16 scales per plane).
    const fs::path dir = root / "experts" / "layer_0" / "exp_0.arrow";
    REQUIRE(fs::exists(dir));
    auto batch = lsxcommon::ReadRecordBatchFromFile(dir);
    if (!batch.ok()) FAIL(batch.status().ToString());
    auto pe = lsxcommon::ExpertSchema::UnpackBatch(*batch, lsxcommon::kNVFP4BlockSize);
    if (!pe.ok()) FAIL(pe.status().ToString());
    REQUIRE(pe->w1_packed.size() ==
            static_cast<size_t>(frozen::kMoeIntermediate) * (frozen::kHidden / 2));
    REQUIRE(pe->w2_packed.size() ==
            static_cast<size_t>(frozen::kHidden) * (frozen::kMoeIntermediate / 2));
    REQUIRE(pe->w1_global_scale > 0.0f);
    REQUIRE(pe->w2_global_scale > 0.0f);
    // The per-expert global scale is amax/2688, i.e. ~1e-5 for these weights.
    REQUIRE(pe->w1_global_scale < 1.0f);
    REQUIRE(pe->w2_global_scale < 1.0f);
}
