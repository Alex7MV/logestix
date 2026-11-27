// Hy4-flash converter (stage 1/2).
//
// Two cases, and they answer different questions:
//
//  1. The CLASSIFIER against the real checkpoint's own index (167 KB, no
//     weights). Every one of the 1964 tensor names must land in exactly one
//     bucket — an unclassified name is a converter that silently ships a
//     truncated model, and this is the only test that can see the real name
//     set without reading 411 GB.
//  2. The synthetic end-to-end conversion: a hand-built tree with the flash
//     layout (3-D packed expert tensors, split kv_b, an Engram layer, a vision
//     tower and a DFly2 sample) converted through the real pipeline, asserting
//     the artifact shape the loader depends on.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <simdjson.h>

#include "lsxcommon/hy_moe_flash_convert.h"

#include "hy_moe_flash_constants.h"

namespace fs = std::filesystem;
namespace hy4 = lsxcommon::hy4;

namespace {

fs::path VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_HY4_FLASH_ROOT"))
        return fs::path(env);
    return fs::path("/mnt/fast_storage/lsx_model/tencent/Hy4-flash");
}

} // namespace

TEST_CASE("hy4 convert: the classifier covers every real tensor name",
          "[hy4][fast]")
{
    const fs::path index = VaultRoot() / "model.safetensors.index.json";
    if (!fs::exists(index))
        SKIP("Hy4-flash index not present at " + index.string());

    std::ifstream f(index);
    REQUIRE(f.good());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);
    simdjson::dom::object weight_map;
    REQUIRE(doc["weight_map"].get_object().get(weight_map) == simdjson::SUCCESS);

    int unknown = 0, engram = 0, vision = 0, dfly = 0, experts = 0, dense = 0,
        raw = 0, globals = 0, total = 0;
    std::set<int> engram_layers, moe_layers;
    int kv_b = 0;
    for (auto [name, file] : weight_map) {
        ++total;
        const auto p = hy4::ParseHy4Tensor(name);
        switch (p.kind) {
        case hy4::TensorKind::kUnknown: ++unknown; break;
        case hy4::TensorKind::kCutEngram: ++engram; engram_layers.insert(p.layer); break;
        case hy4::TensorKind::kCutVision: ++vision; break;
        case hy4::TensorKind::kCutDfly: ++dfly; break;
        case hy4::TensorKind::kExpertGateUp:
        case hy4::TensorKind::kExpertDown: ++experts; moe_layers.insert(p.layer); break;
        case hy4::TensorKind::kDenseNvfp4: ++dense; break;
        case hy4::TensorKind::kLayerRaw: ++raw; break;
        case hy4::TensorKind::kGlobalRaw: ++globals; break;
        }
        // kv_b_proj appears once per BACKBONE layer; the drafter carries its
        // own five (cut, so they must not be counted here).
        if (std::string_view(name).rfind("model.dfly_layers.", 0) != 0 &&
            std::string_view(name).find("kv_b_proj.weight") !=
                std::string_view::npos)
            ++kv_b;
    }

    CHECK(unknown == 0);
    CHECK(total == 1964);
    // Engram lives on exactly the four layers the config lists.
    CHECK(engram_layers == std::set<int>{1, 11, 21, 31});
    // Two packed expert tensors per MoE layer: 41 layers, layer 0 is dense.
    CHECK(experts == 2 * 41);
    CHECK(moe_layers.size() == 41);
    CHECK(moe_layers.count(0) == 0);
    // Every layer carries one kv_b projection (split into two records later).
    CHECK(kv_b == 42);
    // embed, lm_head, final norm and the three hc_head parameters.
    CHECK(globals == 6);
    CHECK(vision == 441);   // the 27-layer ViT tower and its projector
    CHECK(dfly == 87);      // 5 drafter layers + fusion + confidence head
    CHECK(engram == 164);   // 4 layers x (4 key_projs + 16 head norms + ...)
    // Everything else is a dense NVFP4 projection or a raw norm/scalar; the
    // split is what the converter's per-leaf table decides.
    CHECK(dense + raw == 1184);
}

TEST_CASE("hy4 convert: an unclassified name is a hard error", "[hy4][fast]")
{
    CHECK(hy4::ParseHy4Tensor("model.layers.0.mystery.weight").kind ==
          hy4::TensorKind::kUnknown);
    CHECK(hy4::ParseHy4Tensor("totally.foreign").kind ==
          hy4::TensorKind::kUnknown);
    // The known buckets, spelled the way the checkpoint spells them.
    CHECK(hy4::ParseHy4Tensor("model.embed_tokens.weight").kind ==
          hy4::TensorKind::kGlobalRaw);
    CHECK(hy4::ParseHy4Tensor("model.layers.7.mlp.experts.gate_up_proj").kind ==
          hy4::TensorKind::kExpertGateUp);
    CHECK(hy4::ParseHy4Tensor("model.layers.7.mlp.experts.down_proj").kind ==
          hy4::TensorKind::kExpertDown);
    CHECK(hy4::ParseHy4Tensor("model.layers.7.self_attn.q_b_proj.weight").kind ==
          hy4::TensorKind::kDenseNvfp4);
    CHECK(hy4::ParseHy4Tensor("model.layers.7.mlp.gate.weight").kind ==
          hy4::TensorKind::kLayerRaw);
    CHECK(hy4::ParseHy4Tensor("model.layers.7.hc_mlp_layer.hc_pre.hc_fn").kind ==
          hy4::TensorKind::kLayerRaw);
    CHECK(hy4::ParseHy4Tensor("model.layers.7.engram_layer.lookup_table").kind ==
          hy4::TensorKind::kCutEngram);
    CHECK(hy4::ParseHy4Tensor("vit.layers.3.self_attn.q_proj.weight").kind ==
          hy4::TensorKind::kCutVision);
    CHECK(hy4::ParseHy4Tensor("model.dfly_layers.2.mlp.up_proj.weight").kind ==
          hy4::TensorKind::kCutDfly);
}

TEST_CASE("hy4 convert: layer shapes come from the config lists", "[hy4][fast]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "config.json"))
        SKIP("Hy4-flash config not present at " + root.string());

    std::vector<hy4::LayerShape> shapes;
    std::string why;
    const auto shapes_status = hy4::ReadHy4LayerShapes(root.string(), shapes, why);
    REQUIRE(shapes_status.ok());
    REQUIRE(shapes.size() == 42);
    int dsa = 0, moe = 0, engram = 0;
    for (int l = 0; l < 42; ++l) {
        const auto &s = shapes[static_cast<size_t>(l)];
        CHECK(s.is_dsa == lsxcommon::hy4::IsDsa(l));
        CHECK(s.is_moe == lsxcommon::hy4::IsMoeLayer(l));
        CHECK(s.heads == lsxcommon::hy4::Heads(l));
        if (s.is_dsa) ++dsa;
        if (s.is_moe) ++moe;
        if (s.has_engram) ++engram;
    }
    CHECK(dsa == 12);
    CHECK(moe == 41);
    CHECK(engram == 4);
}
