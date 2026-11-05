// Kimi-K3.1-Pro loader stage: the shared K3 pinned graph plus the variant's
// Gated-DSA indexer weights, exercised on a synthetic converted tree (never
// the vault).
//
// `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` is set by this target's ctest ENVIRONMENT
// property and enforced inside the converter; the test deliberately does not
// read it (the suite guard forbids getenv in a loader-stage test, because that
// is how a loader test would resolve a real vault root).
//
// The loader must stay ONE loader. A Kimi-K3 tree loads exactly as before
// (`dsa == false`, every indexer pointer null); a Kimi-K3.1-Pro tree loads the
// five BF16 indexer tensors per full-attention layer as FP16 device copies and
// counts their bytes in `resident_bytes`. The variant cross-check is the
// frozen passport, not the tree: a synthetic tree never passes it.
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "kimi_moe_k3_constants.h"
#include "kimi_moe_pro_constants.h"
#include "lsxcommon/kimi_moe_k3_convert.h"
#include "lsxcommon/kimi_moe_k3_load.h"
#include "lsxcommon/kimi_moe_pro_convert.h"
#include "test_kimi_pro_fixture.h"

namespace fs = std::filesystem;
using namespace lsxcommon;

namespace {

bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

fs::path BuildPro(const std::string &tag)
{
    const fs::path in = fs::temp_directory_path() / ("kimi_pro_loader_in_" + tag);
    const fs::path out = fs::temp_directory_path() / ("kimi_pro_loader_out_" + tag);
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    kimiprotest::WriteCheckpoint(in, kimiprotest::Variant::kPro);
    const auto st = kimi_pro::ConvertKimiK31ProModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    return out;
}

fs::path BuildK3(const std::string &tag)
{
    const fs::path in = fs::temp_directory_path() / ("kimi_k3_loader_in_" + tag);
    const fs::path out = fs::temp_directory_path() / ("kimi_k3_loader_out_" + tag);
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    kimik3test::WriteCheckpoint(in);
    const auto st = kimi_k3::ConvertKimiK3Model(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    return out;
}

// The fixture writes BF16 bytes; the loader expands them to FP16 (exact for
// every value the fixture produces: BF16 has 7 explicit mantissa bits, FP16 has
// 10), so the expected device value can be derived from the fixture bytes.
float Bf16Value(const std::vector<uint8_t> &bytes, size_t i)
{
    const uint16_t b =
        static_cast<uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8));
    const uint32_t f = static_cast<uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &f, 4);
    return v;
}

std::vector<uint8_t> IndexerBytes(int64_t n, float base)
{
    return synthtest::MakeBf16Values(n, [base](int64_t i) {
        return base * (1.0f + static_cast<float>(i % 11));
    });
}

std::vector<__half> ReadBack(const __half *d, size_t n)
{
    std::vector<__half> out(n);
    if (!d) return out;
    cudaMemcpy(out.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost);
    return out;
}

// A topology equal to the frozen Kimi-K3 passport. The DSA block is filled by
// the caller so the cross-check can be exercised without a real checkpoint.
kimi_k3::KimiK3Topology FrozenTopology()
{
    using namespace kimi_k3::frozen;
    kimi_k3::KimiK3Topology t;
    t.format = kFormatTag;
    t.num_layers = kLayers;
    t.dense_layers = kDenseLayers;
    t.hidden = kHidden;
    t.vocab = kVocab;
    t.attn_res_block = kAttnResBlock;
    t.mla_heads = kMlaHeads;
    t.kv_heads = kMlaKVHeads;
    t.q_lora = kQLoraRank;
    t.kv_lora = kKvLoraRank;
    t.qk_nope = kQkNopeHeadDim;
    t.qk_rope = kQkRopeHeadDim;
    t.v_head = kVHeadDim;
    t.kda_heads = kKdaHeads;
    t.kda_head_dim = kKdaHeadDim;
    t.conv_kernel = kKdaConvKernel;
    t.num_experts = kExperts;
    t.top_k = kTopK;
    t.moe_intermediate = kMoeIntermediate;
    t.latent = kLatentDim;
    t.shared_experts = kSharedExperts;
    t.shared_intermediate = kSharedIntermediate;
    t.mla_layers.assign(kMlaLayers.begin(), kMlaLayers.end());
    t.kda_layers.assign(1, 0);
    return t;
}

void FillDsaBlock(kimi_k3::KimiK3Topology &t)
{
    using namespace kimi_pro::frozen;
    t.dsa = true;
    t.dsa_layers.assign(kDsaLayers.begin(), kDsaLayers.end());
    t.index_heads = kIndexHeads;
    t.index_head_dim = kIndexHeadDim;
    t.index_topk = kIndexTopK;
    t.index_rope_dim = kIndexRopeDim;
    t.index_norm_eps = kIndexNormEps;
}

// Writes a copy of the tree's index JSON with one textual substitution, so a
// hand-broken index can be fed to the loader without touching the tree.
fs::path IndexWith(const fs::path &tree, const std::string &from,
                   const std::string &to, const std::string &tag)
{
    const fs::path src = tree / "logestix" / "model.index.json";
    std::ifstream in(src);
    REQUIRE(in.is_open());
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    const auto pos = json.find(from);
    REQUIRE(pos != std::string::npos);
    json.replace(pos, from.size(), to);
    const fs::path dir = fs::temp_directory_path() / ("kimi_pro_badidx_" + tag);
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "logestix");
    std::ofstream out(dir / "logestix" / "model.index.json");
    out << json;
    return dir;
}

}  // namespace

TEST_CASE("kimi-k3.1-pro loader reads the DSA topology", "[kimi-pro][loader]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path root = BuildPro("top");

    kimi_k3::KimiK3Topology top;
    REQUIRE(kimi_k3::LoadKimiK3Topology(root.string(), top).ok());
    CHECK(top.format == "logestix-kimi-k31-pro-nvfp4");
    CHECK(top.dsa);
    CHECK(top.dsa_layers == std::vector<int>{2, 3});
    CHECK(top.dsa_layers == top.mla_layers);
    CHECK(top.index_heads == kimiprotest::kIndexHeads);
    CHECK(top.index_head_dim == kimiprotest::kIndexHeadDim);
    CHECK(top.index_topk == kimiprotest::kIndexTopK);
    CHECK(top.index_rope_dim == kimiprotest::kIndexRopeDim);
    // The indexer key LayerNorm's eps travels with the tree: it is the
    // arithmetic, not a runtime knob.
    CHECK(top.index_norm_eps == kimi_pro::frozen::kIndexNormEps);
    CHECK(top.IsDsa(2));
    CHECK(top.IsDsa(3));
    CHECK_FALSE(top.IsDsa(0));
    CHECK_FALSE(top.IsDsa(1));
    // Synthetic geometry must NOT pass the frozen cross-check.
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(top, true,
                                                     kimiprotest::kIndexTopK));
}

TEST_CASE("kimi-k3.1-pro loader rejects an index whose DSA table is not the MLA table",
          "[kimi-pro][loader]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path root = BuildPro("dsa_table");

    // The indexer runs on every full-attention layer or not at all; a partial
    // table means the tree was written by something that did not understand it.
    const fs::path broken =
        IndexWith(root, "\"dsa_layers\": [2,3]", "\"dsa_layers\": [2]",
                  "dsa_table");
    kimi_k3::KimiK3Topology top;
    const auto st = kimi_k3::LoadKimiK3Topology(broken.string(), top);
    CHECK_FALSE(st.ok());
    CHECK(top.error_msg.find("dsa_layers") != std::string::npos);
}

TEST_CASE("kimi-k3.1-pro loader pins the five indexer weights per DSA layer",
          "[kimi-pro][loader]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path root = BuildPro("pin");

    kimi_k3::KimiK3PinnedWeights pw;
    REQUIRE(kimi_k3::LoadKimiK3PinnedWeights(root.string(), pw, 0));
    CHECK(pw.top.dsa);

    // DSA layers carry all five tensors; the KDA layers carry none.
    for (int l : {2, 3}) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        INFO("DSA layer " << l);
        REQUIRE(lw.idx_wqb != nullptr);
        REQUIRE(lw.idx_wk != nullptr);
        REQUIRE(lw.idx_knorm_w != nullptr);
        REQUIRE(lw.idx_knorm_b != nullptr);
        REQUIRE(lw.idx_wproj != nullptr);
        // Content is the oracle: the loader must have uploaded *these* tensors.
        // A wrong source (wq_b vs wk vs weights_proj) or a wrong element count
        // shows up immediately; the fixture's patterns differ per tensor.
        const auto want_wqb = IndexerBytes(
            static_cast<int64_t>(kimiprotest::kIndexWqBRows) *
                kimiprotest::kIndexQLora, 0.02f);
        const auto want_wk = IndexerBytes(
            static_cast<int64_t>(kimiprotest::kIndexHeadDim) *
                kimik3test::kHidden, 0.03f);
        const auto want_wproj = IndexerBytes(
            static_cast<int64_t>(kimiprotest::kIndexHeads) *
                kimik3test::kHidden, 0.04f);
        const auto wqb = ReadBack(lw.idx_wqb, want_wqb.size() / 2);
        const auto wk = ReadBack(lw.idx_wk, want_wk.size() / 2);
        const auto wproj = ReadBack(lw.idx_wproj, want_wproj.size() / 2);
        for (size_t i : {size_t{0}, wqb.size() / 3, wqb.size() - 1})
            CHECK(__half2float(wqb[i]) == Bf16Value(want_wqb, i));
        for (size_t i : {size_t{0}, wk.size() / 2, wk.size() - 1})
            CHECK(__half2float(wk[i]) == Bf16Value(want_wk, i));
        for (size_t i : {size_t{0}, wproj.size() - 1})
            CHECK(__half2float(wproj[i]) == Bf16Value(want_wproj, i));
        // Content: the fixture writes k_norm.weight = 1 and k_norm.bias = 0, so
        // a weight/bias swap (the classic copy-paste bug) is visible here.
        const auto w = ReadBack(lw.idx_knorm_w, kimiprotest::kIndexHeadDim);
        const auto b = ReadBack(lw.idx_knorm_b, kimiprotest::kIndexHeadDim);
        for (int i = 0; i < kimiprotest::kIndexHeadDim; ++i) {
            CHECK(__half2float(w[static_cast<size_t>(i)]) == 1.0f);
            CHECK(__half2float(b[static_cast<size_t>(i)]) == 0.0f);
        }
    }
    for (int l : {0, 1}) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        INFO("KDA layer " << l);
        CHECK(lw.idx_wqb == nullptr);
        CHECK(lw.idx_wk == nullptr);
        CHECK(lw.idx_knorm_w == nullptr);
        CHECK(lw.idx_knorm_b == nullptr);
        CHECK(lw.idx_wproj == nullptr);
    }

    kimi_k3::FreeKimiK3PinnedWeights(pw);
    // Everything the loader owns is released (checked on the reset struct).
    for (const auto &lw : pw.layers) {
        CHECK(lw.idx_wqb == nullptr);
        CHECK(lw.idx_wproj == nullptr);
    }
    CHECK(pw.resident_bytes == 0);
}

TEST_CASE("kimi-k3.1-pro loader counts the indexer bytes it allocated",
          "[kimi-pro][loader]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path pro = BuildPro("bytes_pro");
    const fs::path k3 = BuildK3("bytes_k3");

    kimi_k3::KimiK3PinnedWeights pro_pw, k3_pw;
    REQUIRE(kimi_k3::LoadKimiK3PinnedWeights(pro.string(), pro_pw, 0));
    REQUIRE(kimi_k3::LoadKimiK3PinnedWeights(k3.string(), k3_pw, 0));
    // The two trees differ by exactly the indexer plane: 2 DSA layers x five
    // BF16 tensors (FP16 on device, same byte count).
    const unsigned long long per_layer =
        2ull * (static_cast<unsigned long long>(kimiprotest::kIndexWqBRows) *
                    kimiprotest::kIndexQLora +
                static_cast<unsigned long long>(kimiprotest::kIndexHeadDim) *
                    kimik3test::kHidden +
                2ull * kimiprotest::kIndexHeadDim +
                static_cast<unsigned long long>(kimiprotest::kIndexHeads) *
                    kimik3test::kHidden);
    CHECK(pro_pw.resident_bytes == k3_pw.resident_bytes + 2ull * per_layer);
    kimi_k3::FreeKimiK3PinnedWeights(pro_pw);
    kimi_k3::FreeKimiK3PinnedWeights(k3_pw);
}

TEST_CASE("kimi-k3 loader leaves every indexer pointer null without the variant",
          "[kimi-pro][loader]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path root = BuildK3("k3_null");

    kimi_k3::KimiK3Topology top;
    REQUIRE(kimi_k3::LoadKimiK3Topology(root.string(), top).ok());
    CHECK_FALSE(top.dsa);
    CHECK(top.dsa_layers.empty());
    CHECK(top.index_heads == 0);
    CHECK(top.index_head_dim == 0);
    CHECK(top.index_topk == 0);
    CHECK(top.index_rope_dim == 0);
    CHECK_FALSE(top.IsDsa(2));

    kimi_k3::KimiK3PinnedWeights pw;
    REQUIRE(kimi_k3::LoadKimiK3PinnedWeights(root.string(), pw, 0));
    for (const auto &lw : pw.layers) {
        CHECK(lw.idx_wqb == nullptr);
        CHECK(lw.idx_wk == nullptr);
        CHECK(lw.idx_knorm_w == nullptr);
        CHECK(lw.idx_knorm_b == nullptr);
        CHECK(lw.idx_wproj == nullptr);
    }
    kimi_k3::FreeKimiK3PinnedWeights(pw);
    // A second free on a reset struct is a no-op, not a double free.
    kimi_k3::FreeKimiK3PinnedWeights(pw);
    CHECK(pw.layers.empty());
}

TEST_CASE("kimi-k3.1-pro frozen cross-check distinguishes the variant",
          "[kimi-pro][loader][unit]")
{
    using namespace kimi_pro::frozen;

    // A Kimi-K3 topology: the one-argument form passes, the DSA form does not.
    auto k3 = FrozenTopology();
    CHECK(kimi_k3::KimiK3TopologyMatchesFrozen(k3));
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(k3, true, kIndexTopK));

    // The same tree with the variant's DSA block: only the DSA form passes.
    auto pro = k3;
    FillDsaBlock(pro);
    CHECK(kimi_k3::KimiK3TopologyMatchesFrozen(pro, true, kIndexTopK));
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(pro));
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(pro, false, kIndexTopK));
    // A different top-k is a different variant (a dsa_config edit must fail).
    CHECK_FALSE(
        kimi_k3::KimiK3TopologyMatchesFrozen(pro, true, kIndexTopK - 1));
    // The DSA table must still be the MLA table.
    auto partial = pro;
    partial.dsa_layers.pop_back();
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(partial, true, kIndexTopK));
    // The indexer key LayerNorm eps is arithmetic, not a knob: a tree that
    // writes a different one was built by a different recipe.
    auto eps = pro;
    eps.index_norm_eps = 1e-6;
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(eps, true, kIndexTopK));
    // A DSA tree must state its eps at all (a missing field parses as 0).
    auto no_eps = pro;
    no_eps.index_norm_eps = 0.0;
    CHECK_FALSE(
        kimi_k3::KimiK3TopologyMatchesFrozen(no_eps, true, kIndexTopK));
    // Kimi-K3 geometry is shared: breaking it fails both forms.
    auto broken = pro;
    broken.hidden = kHidden - 1;
    CHECK_FALSE(kimi_k3::KimiK3TopologyMatchesFrozen(broken, true, kIndexTopK));
}

// The DSA variant inherits the parent's expert working set, so the routed-expert
// window floor is the same 1472 and the same clamp applies (see
// tests/test_kimi_k3_loader.cpp for the measurement behind the floor).
TEST_CASE("kimi-k3.1-pro expert window floor matches the parent's",
          "[kimi-pro][loader][unit]")
{
    using namespace kimi_pro::frozen;
    auto pro = FrozenTopology();
    FillDsaBlock(pro);
    CHECK(kimi_k3::KimiK3ExpertWindowFloor(pro) ==
          static_cast<size_t>((kLayers - kDenseLayers) * kTopK));
    CHECK(kimi_k3::KimiK3ExpertWindowFloor(pro) == 1472);
    CHECK(kimi_k3::KimiK3ResolveExpertWindow(pro, 512) == 1472);
    CHECK(kimi_k3::KimiK3ResolveExpertWindow(pro, 4096) == 4096);
}
