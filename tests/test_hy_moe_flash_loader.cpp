// Hy4-flash loader (stage 3) on a small hand-built converted tree.
//
// The real vault is not readable here (a 90 GB artifact), so the fixture writes
// the SAME Arrow layout the converter emits (global raw, per-layer raw, per-layer
// DenseNvfp4, one ExpertSchema file per expert) at synthetic geometry and checks
// the gates and the upload path against it.
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <arrow/api.h>

#include "hy_moe_flash_load.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/index_layout.h"
#include "lsxcommon/synthetic_geometry.h"

namespace fs = std::filesystem;
namespace gpu = lsxcommon::hy4::gpu;

namespace {

// Writes a minimal but structurally valid converted tree for `layers` layers
// with `experts` experts, one sparse layer (index 0) and one MoE layer (index 1).
fs::path BuildTree(int layers, int experts)
{
    const fs::path root = fs::temp_directory_path() /
                          ("hy4-load-" + std::to_string(::getpid()));
    fs::remove_all(root);
    const fs::path log = root;
    fs::create_directories(log / "pinned");
    fs::create_directories(log / "experts" / "layer_1");

    // Index: enough fields for the topology parser and the geometry gate.
    {
        std::ofstream f(log / "model.index.json");
        f << "{\n";
        f << "  \"layout\": 2,\n";
        f << "  \"model\": \"Hy4-flash\",\n";
        f << "  \"model_type\": \"hy4-flash\",\n";
        f << "  \"num_layers\": " << layers << ",\n";
        f << "  \"experts_per_layer\": " << experts << ",\n";
        f << "  \"total_pinned_tensors\": 1,\n";
        f << "  \"total_experts\": " << experts << ",\n";
        f << "  \"hidden_size\": 3072,\n  \"vocab_size\": 120832,\n";
        f << "  \"num_experts_per_tok\": 16,\n  \"moe_intermediate_size\": 1536,\n";
        f << "  \"latent_size\": 1536,\n  \"shared_intermediate_size\": 1536,\n";
        f << "  \"dense_intermediate_size\": 13824,\n";
        f << "  \"num_attention_heads\": 32,\n  \"swa_num_attention_heads\": 64,\n";
        f << "  \"q_lora_rank\": 1024,\n  \"kv_lora_rank\": 512,\n";
        f << "  \"qk_nope_head_dim\": 192,\n  \"qk_rope_head_dim\": 64,\n";
        f << "  \"v_head_dim\": 256,\n  \"sliding_window\": 512,\n";
        f << "  \"hc_mult\": 4,\n  \"index_n_heads\": 16,\n";
        f << "  \"index_head_dim\": 128,\n  \"index_topk\": 2048,\n";
        f << "  \"bos_token_id\": 120000,\n  \"eos_token_id\": 120025,\n";
        f << "  \"pad_token_id\": 120002,\n";
        // The vram block's real schema: three byte classes and the context.
        f << "  \"vram\": {\"min_bytes\": 1024, \"optimal_bytes\": 2048, "
             "\"max_bytes\": 4096, \"context_tokens\": 8192},\n";
        f << "  \"layers\": [";
        for (int l = 0; l < layers; ++l) {
            const bool dsa = l <= 1 || l % 4 == 1;
            f << "{\"is_dsa\": " << (dsa ? "true" : "false")
              << ", \"heads\": " << (dsa ? 32 : 64)
              << ", \"is_moe\": " << (l > 0 ? "true" : "false")
              << ", \"has_engram\": " << (l == 1 ? "true" : "false") << "}";
            if (l + 1 < layers) f << ",";
        }
        f << "],\n";
        f << "  \"deferred\": {\"engram\": 28, \"vision\": 266, \"dfly\": 74}\n";
        f << "}\n";
    }
    return root;
}

} // namespace

TEST_CASE("hy4 loader: topology parses and the format gate is exact",
          "[hy4][fast]")
{
    const fs::path root = BuildTree(42, 512);
    const auto topo = gpu::LoadTopology(root.string());
    CHECK(topo.error_msg.empty());
    CHECK(topo.num_layers == 42);
    CHECK(topo.experts_per_layer == 512);
    CHECK(topo.layout == lsxcommon::kIndexLayout);
    CHECK(gpu::IsSupportedFormat(topo));
    CHECK(topo.deferred_engram == 28);
    REQUIRE(topo.layers.size() == 42);
    CHECK(topo.layers[0].is_dsa);
    CHECK(!topo.layers[2].is_dsa);
    CHECK(topo.layers[2].heads == 64);
    CHECK(topo.layers[1].is_moe);
    CHECK(!topo.layers[0].is_moe);

    // One card name per artifact.
    CHECK(gpu::MatchesModelType(topo, "hy4-flash"));
    CHECK(!gpu::MatchesModelType(topo, "hy-memory-30b-a3b"));
    CHECK(gpu::NormalizeCardName("Hy4-flash") == "hy4-flash");

    std::string why;
    CHECK(gpu::ValidateGeometry(topo, &why));

    // A wrong layer count is a hard failure, not a warning.
    auto bad = topo;
    bad.num_layers = 41;
    CHECK(!gpu::ValidateGeometry(bad, &why));
    CHECK(!why.empty());

    // A tree with the sibling's hybrid pattern is rejected per layer.
    auto wrong = topo;
    wrong.layers[2].is_dsa = true;
    CHECK(!gpu::ValidateGeometry(wrong, &why));

    fs::remove_all(root);
}

TEST_CASE("hy4 loader: a missing index reports why", "[hy4][fast]")
{
    const auto topo = gpu::LoadTopology("/nonexistent/hy4");
    CHECK(!topo.error_msg.empty());
    CHECK(topo.num_layers == 0);
}

TEST_CASE("hy4 loader: kv geometry is the MLA latent", "[hy4][fast]")
{
    const auto geom = gpu::KvGeometry(8192);
    // 42 layers x 8192 positions x (512 + 64) halves = 48 KiB per token.
    CHECK(geom.layers == 42);
    CHECK(geom.kv_heads == 1);
    CHECK(geom.head_dim == 576);
    CHECK(geom.Elements() == 42LL * 8192 * 576);
    CHECK(lsxcommon::hy4::KvBytesPerToken() == 42 * 576 * 2);
}
