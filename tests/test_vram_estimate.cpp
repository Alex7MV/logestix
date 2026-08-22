#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/vram_estimate.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using lsxcommon::VramGeometry;
using lsxcommon::VramProfile;
using lsxcommon::VramTensorClass;
using lsxcommon::VramTensorInfo;
using lsxcommon::VramTensorPolicy;

namespace {

// Capture-free lambdas convert implicitly to the plain function pointers
// stored in VramTensorPolicy.
constexpr auto kAllNvfp4 = +[](const std::string &) { return true; };
constexpr auto kNoneNvfp4 = +[](const std::string &) { return false; };
constexpr auto kAllRouted = +[](const std::string &) {
    return VramTensorClass::RoutedExpert;
};
constexpr auto kAllResident = +[](const std::string &) {
    return VramTensorClass::Resident;
};

VramGeometry BaseGeom()
{
    VramGeometry g;
    g.hidden_dim = 4096;
    g.kv_dim = 512;
    g.num_attn_layers = 1;
    g.num_moe_layers = 1;
    g.experts_per_layer = 1;
    g.context_length = 2048;
    return g;
}

} // namespace

TEST_CASE("NVFP4 packing math: numel/2 packed + numel/16 scales", "[vram]")
{
    // 16384 elements: 8192 bytes packed E2M1 + 1024 bytes E4M3 scales.
    std::vector<VramTensorInfo> tensors = {
        {"layers.0.ffn.experts.0.w1.weight", 16384, 32768},
    };
    auto p = lsxcommon::EstimateVram(tensors, {kAllNvfp4, kAllRouted}, BaseGeom());
    CHECK(p.min_bytes == 0);          // routed expert -> not resident
    CHECK(p.optimal_bytes == p.min_bytes + 37748736ull); // kv+ws @ ctx=2048, 1 attn layer
    CHECK(p.max_bytes == 9216ull + 37748736ull);         // 16384/2 + 16384/16
}

TEST_CASE("Non-quantized tensors keep raw dtype size", "[vram]")
{
    std::vector<VramTensorInfo> tensors = {
        {"model.embed_tokens.weight", 1000, 2000}, // BF16, resident
    };
    auto p = lsxcommon::EstimateVram(tensors, {kNoneNvfp4, kAllResident}, BaseGeom());
    CHECK(p.min_bytes == 2000);
    CHECK(p.max_bytes == p.optimal_bytes); // no routed experts
}

TEST_CASE("Hierarchy invariant min <= optimal <= max", "[vram]")
{
    std::vector<VramTensorInfo> tensors = {
        {"model.embed_tokens.weight", 1000, 2000},
        {"layers.0.ffn.experts.0.w1.weight", 16384, 32768},
    };
    auto p = lsxcommon::EstimateVram(tensors, {kAllNvfp4, kAllRouted}, BaseGeom());
    CHECK(p.min_bytes <= p.optimal_bytes);
    CHECK(p.optimal_bytes <= p.max_bytes);
}

TEST_CASE("KV cache scales with context, workspace stays chunk-bounded", "[vram]")
{
    std::vector<VramTensorInfo> tensors = {
        {"model.embed_tokens.weight", 1000, 2000},
    };
    auto g = BaseGeom();
    g.context_length = 2048;
    auto p1 = lsxcommon::EstimateVram(tensors, {kNoneNvfp4, kAllResident}, g);
    g.context_length = 4096;
    auto p2 = lsxcommon::EstimateVram(tensors, {kNoneNvfp4, kAllResident}, g);
    // kv = ctx*attn*kv_dim*2*2 scales linearly with ctx; the workspace is
    // computed from a fixed 2048-token prefill chunk and does not grow.
    // kv delta: 2048 extra tokens * 1 attn layer * 512 kv_dim * 2(K+V) * 2(FP16)
    constexpr std::uint64_t kKvDelta =
        2048ull * 1 * 512 * 2 * 2;   // 4 MiB
    CHECK(p2.optimal_bytes - p1.optimal_bytes == kKvDelta);
    CHECK(p2.context_tokens == 4096);

    // Workspace component (fixed): 2048 * hidden(4096) * 1 * 2 * 2 = 32 MiB.
    CHECK(p1.optimal_bytes - p1.min_bytes - kKvDelta ==
          2048ull * 4096 * 1 * 2 * 2);
}

TEST_CASE("Empty inventory yields zero profile", "[vram]")
{
    std::vector<VramTensorInfo> empty;
    auto p = lsxcommon::EstimateVram(empty, {kAllNvfp4, kAllRouted}, BaseGeom());
    CHECK(p.min_bytes == 0);
    CHECK(p.optimal_bytes == 0);
    CHECK(p.max_bytes == 0);
}

TEST_CASE("Zero context_length falls back to 2048", "[vram]")
{
    auto g = BaseGeom();
    g.context_length = 0;
    std::vector<VramTensorInfo> empty;
    auto p = lsxcommon::EstimateVram(empty, {kNoneNvfp4, kAllResident}, g);
    CHECK(p.context_tokens == 2048);
}

TEST_CASE("Compressed KV ratios shrink the stored token count", "[vram]")
{
    std::vector<VramTensorInfo> tensors = {
        {"model.embed_tokens.weight", 1000, 2000},
    };
    auto g = BaseGeom();   // kv_dim=512, 1 attn layer, ctx=2048
    // Full-context layer + /4 layer + /128 layer:
    // tokens = 2048 + 512 + 16 = 2576; KV bytes = tokens*512*2*2.
    g.kv_compress_ratios = {0, 4, 128};
    auto p = lsxcommon::EstimateVram(tensors, {kNoneNvfp4, kAllResident}, g);
    const std::uint64_t kv =
        (2048ull + 2048ull / 4 + 2048ull / 128) * 512 * 2 * 2;
    const std::uint64_t ws = 2048ull * 4096 * 1 * 2 * 2;
    CHECK(p.optimal_bytes - p.min_bytes == kv + ws);

    // Empty ratios keep the legacy full-context behavior for all layers.
    auto g2 = BaseGeom();
    auto p2 = lsxcommon::EstimateVram(tensors, {kNoneNvfp4, kAllResident}, g2);
    CHECK(p2.optimal_bytes - p2.min_bytes ==
          2048ull * 1 * 512 * 2 * 2 + ws);
}

TEST_CASE("kEmptyVramProfile is all zeros", "[vram]")
{
    CHECK(lsxcommon::kEmptyVramProfile.min_bytes == 0);
    CHECK(lsxcommon::kEmptyVramProfile.context_tokens == 0);
}

namespace {

fs::path WriteTempConfig(const std::string &json)
{
    static int counter = 0;   // portable unique-enough temp name per process
    auto p = fs::temp_directory_path() /
             ("logestix_vram_cfg_" + std::to_string(counter++) + ".json");
    std::ofstream f(p, std::ios::trunc);
    f << json;
    return p;
}

} // namespace

TEST_CASE("ReadContextLength: max_position_embeddings at root", "[vram]")
{
    auto p = WriteTempConfig(R"({"hidden_size": 4096,
        "max_position_embeddings": 163840})");
    CHECK(lsxcommon::ReadContextLength(p.string()) == 163840);
    fs::remove(p);
}

TEST_CASE("ReadContextLength: nested under text_config (Qwen)", "[vram]")
{
    auto p = WriteTempConfig(R"({"text_config": {
        "max_position_embeddings": 262144}})");
    CHECK(lsxcommon::ReadContextLength(p.string()) == 262144);
    fs::remove(p);
}

TEST_CASE("ReadContextLength: context_length / seq_length fallbacks", "[vram]")
{
    auto p1 = WriteTempConfig(R"({"context_length": 4096})");
    CHECK(lsxcommon::ReadContextLength(p1.string()) == 4096);
    fs::remove(p1);

    auto p2 = WriteTempConfig(R"({"seq_length": 8192})");
    CHECK(lsxcommon::ReadContextLength(p2.string()) == 8192);
    fs::remove(p2);
}

TEST_CASE("ReadContextLength: missing field / missing file -> 2048", "[vram]")
{
    auto p = WriteTempConfig(R"({"hidden_size": 4096})");
    CHECK(lsxcommon::ReadContextLength(p.string()) == 2048);
    fs::remove(p);

    CHECK(lsxcommon::ReadContextLength("Z:/definitely/not/here.json") == 2048);
}

TEST_CASE("FormatVramJson / ParseVramBlock round-trip", "[vram]")
{
    VramProfile in;
    in.min_bytes = 8734629888ull;
    in.optimal_bytes = 9126804480ull;
    in.max_bytes = 412316860416ull;
    in.context_tokens = 163840;

    std::string json = "{" + lsxcommon::FormatVramJson(in) + "}";

    VramProfile out;
    std::string err;
    REQUIRE(lsxcommon::ParseVramBlock(json, out, err));
    CHECK(err.empty());
    CHECK(out.min_bytes == in.min_bytes);
    CHECK(out.optimal_bytes == in.optimal_bytes);
    CHECK(out.max_bytes == in.max_bytes);
    CHECK(out.context_tokens == in.context_tokens);
}

TEST_CASE("ParseVramBlock rejects missing vram block", "[vram]")
{
    VramProfile out;
    std::string err;
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"format": "logestix-glm", "num_layers": 79})", out, err));
    CHECK(err.find("vram") != std::string::npos);
}

TEST_CASE("ParseVramBlock rejects malformed vram block", "[vram]")
{
    VramProfile out;
    std::string err;
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"vram": {"min_bytes": 1}})", out, err));
    CHECK(err.find("malformed") != std::string::npos);
}

TEST_CASE("ParseVramBlock rejects inconsistent values", "[vram]")
{
    VramProfile out;
    std::string err;
    // max < optimal
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"vram": {"min_bytes": 10, "optimal_bytes": 30,
            "max_bytes": 20, "context_tokens": 2048}})", out, err));
    CHECK(err.find("inconsistent") != std::string::npos);

    // context_tokens <= 0
    REQUIRE_FALSE(lsxcommon::ParseVramBlock(
        R"({"vram": {"min_bytes": 10, "optimal_bytes": 20,
            "max_bytes": 30, "context_tokens": 0}})", out, err));
}

TEST_CASE("ParseVramBlock rejects invalid JSON document", "[vram]")
{
    VramProfile out;
    std::string err;
    REQUIRE_FALSE(lsxcommon::ParseVramBlock("not json {{{", out, err));
    CHECK_FALSE(err.empty());
}
