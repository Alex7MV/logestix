// TML-d24-varloop-research synthetic end-to-end converter test.
//
// The varloop checkpoint is Inkling-2-Mini's architecture with
// `loop_config.is_varloop: true`, so the conversion is the SAME pipeline: the
// artifact must be byte-layout identical to the parent's and differ only in the
// three things the capability needs recorded —
//   * the format tag (`logestix-inkling-2-varloop`), so the two trees can never
//     be loaded as each other;
//   * `model` — the card name path auto-detection normalizes into the factory
//     key;
//   * `varloop` + `loop_passes_max` — the trained range's upper bound, which the
//     checkpoint's config does NOT carry (it states only the post-trained
//     default). The loader sizes the KV/conv slab from it.
//
// The other half of the contract is rejection: a varloop checkpoint fed to the
// fixed-pass pipeline, or a fixed-pass checkpoint fed to this one, is an error.
// Accepting either silently produces a model whose pass count is wrong — which
// still answers fluently, so no downstream gate would catch it.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"
#include <lsxcommon/inkling_moe_loop_convert.h>
#include <lsxcommon/inkling_moe_varloop_convert.h>
#include <lsxcommon/schema.h>
#include <simdjson.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_inkling_moe_loop_fixture.h"

namespace fs = std::filesystem;
using namespace inklinglooptest;

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

// A config-only checkpoint directory: enough for ParseInklingLoopConfig, which
// is what decides the varloop verdict before any weight is looked at.
fs::path ConfigOnlyFixture(const char *tag, bool varloop)
{
    const fs::path dir = fs::temp_directory_path() /
                         (std::string("inkling_varloop_cfg_") + tag);
    fs::remove_all(dir);
    fs::create_directories(dir);
    WriteConfig(dir, varloop);
    WriteQuantConfig(dir);
    return dir;
}

}  // namespace

TEST_CASE("varloop convert spec carries this model's own passport numbers",
          "[inkling][varloop][convert]")
{
    using namespace lsxcommon::inkling;
    const InklingLoopConvertSpec spec = VarloopConvertSpec();
    CHECK(spec.varloop);
    CHECK(spec.max_passes == 5);
    CHECK(spec.format_tag == "logestix-inkling-2-varloop");
    // `model` is the CARD name: gpu::detect_model_type lower-cases it into the
    // factory key, so it must normalize to exactly the manifest's model_type.
    CHECK(spec.model_name == "TML-d24-varloop-research");
    // The geometry the varloop row validates against is its own passport's.
    CHECK(spec.frozen.hidden == 2304);
    CHECK(spec.frozen.layers == 24);
    CHECK(spec.frozen.loop_start == 6);
    CHECK(spec.frozen.loop_end == 18);
    CHECK(spec.frozen.nvfp4_expert_layers == 20);
    CHECK(spec.frozen.full_layers == std::vector<int>{5, 11, 17, 23});
}

TEST_CASE("varloop parse accepts is_varloop and records the trained ceiling",
          "[inkling][varloop][convert]")
{
    using namespace lsxcommon::inkling;
    const fs::path dir = ConfigOnlyFixture("var", /*varloop=*/true);
    const InklingLoopConvertTopology top =
        ParseInklingLoopConfig(dir.string(), VarloopConvertSpec());
    INFO(top.error_msg);
    REQUIRE(top.error_msg.empty());
    CHECK(top.varloop);
    CHECK(top.loop_passes == kLoopPasses);  // the checkpoint's declared default
    CHECK(top.loop_passes_max == 5);        // from the model card, not the config
}

TEST_CASE("a varloop checkpoint is rejected by the fixed-pass pipeline",
          "[inkling][varloop][convert]")
{
    using namespace lsxcommon::inkling;
    const fs::path dir = ConfigOnlyFixture("var_mini", /*varloop=*/true);
    const InklingLoopConvertTopology top = ParseInklingLoopConfig(dir.string());
    // Inkling-2-Mini's weights were trained at exactly 3 passes; running them at
    // another count is not this checkpoint's model, so the fixed pipeline must
    // refuse rather than publish a tree that claims a range it does not have.
    CHECK_FALSE(top.error_msg.empty());
    CHECK(top.error_msg.find("is_varloop") != std::string::npos);
}

TEST_CASE("a fixed-pass checkpoint is rejected by the varloop pipeline",
          "[inkling][varloop][convert]")
{
    using namespace lsxcommon::inkling;
    const fs::path dir = ConfigOnlyFixture("fixed", /*varloop=*/false);
    const InklingLoopConvertTopology top =
        ParseInklingLoopConfig(dir.string(), VarloopConvertSpec());
    CHECK_FALSE(top.error_msg.empty());
    CHECK(top.error_msg.find("is_varloop") != std::string::npos);
    // ...and the fixed pipeline still parses its own tree.
    const InklingLoopConvertTopology ok = ParseInklingLoopConfig(dir.string());
    CHECK(ok.error_msg.empty());
    CHECK_FALSE(ok.varloop);
    // A non-varloop tree records no ceiling: its only legal count is its own.
    CHECK(ok.loop_passes_max == ok.loop_passes);
}

TEST_CASE("a ceiling below the declared pass count is an error",
          "[inkling][varloop][convert]")
{
    using namespace lsxcommon::inkling;
    const fs::path dir = ConfigOnlyFixture("ceiling", /*varloop=*/true);
    InklingLoopConvertSpec spec = VarloopConvertSpec();
    spec.max_passes = kLoopPasses - 1;  // 1 pass declared, ceiling below it
    const InklingLoopConvertTopology top = ParseInklingLoopConfig(dir.string(), spec);
    CHECK_FALSE(top.error_msg.empty());
    CHECK(top.error_msg.find("loop_passes_max") != std::string::npos);
}

TEST_CASE("varloop synthetic convert emits logestix-inkling-2-varloop",
          "[inkling][varloop][convert]")
{
    using namespace lsxcommon::inkling;
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_varloop_synth_in";
    const fs::path out = fs::temp_directory_path() / "inkling_varloop_synth_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in, /*sane=*/false, /*varloop=*/true);

    auto st = ConvertInklingVarloopModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const fs::path idx = base / "model.index.json";
    REQUIRE(fs::exists(idx));

    const std::string json = ReadFile(idx);
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    auto doc = parser.iterate(pad);

    std::string_view fmt;
    REQUIRE(doc["format"].get_string().get(fmt) == simdjson::SUCCESS);
    REQUIRE(fmt == std::to_string(lsxcommon::kIndexLayout));

    std::string_view card;
    REQUIRE(doc["model"].get_string().get(card) == simdjson::SUCCESS);
    REQUIRE(card == "TML-d24-varloop-research");
    {
        std::string normalized(card);
        for (char &c : normalized)
            if (c == ' ') c = '-';
            else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        REQUIRE(normalized == "tml-d24-varloop-research");
    }

    bool varloop = false;
    REQUIRE(doc["varloop"].get_bool().get(varloop) == simdjson::SUCCESS);
    CHECK(varloop);

    int64_t passes = 0, passes_max = 0, loop_start = 0, loop_end = 0;
    REQUIRE(doc["loop_passes"].get_int64().get(passes) == simdjson::SUCCESS);
    REQUIRE(doc["loop_passes_max"].get_int64().get(passes_max) == simdjson::SUCCESS);
    REQUIRE(doc["loop_start"].get_int64().get(loop_start) == simdjson::SUCCESS);
    REQUIRE(doc["loop_end"].get_int64().get(loop_end) == simdjson::SUCCESS);
    // The declared default is the checkpoint's; the ceiling is the model card's.
    CHECK(passes == kLoopPasses);
    CHECK(passes_max == 5);
    CHECK(loop_start == kLoopStart);
    CHECK(loop_end == kLoopEnd);

    // The artifact layout is the parent's: same expert planes, same pinned
    // tensors, same stripped ballast. A varloop conversion that "improved" any
    // of this would be a second implementation.
    REQUIRE(fs::exists(base / "pinned" / "global.arrow"));
    REQUIRE(fs::exists(base / "pinned" / "layer_0.arrow"));
    REQUIRE(fs::exists(base / "experts" / "layer_0"));
    int64_t skipped_vis = -1;
    REQUIRE(doc["skipped_visual_tensors"].get_int64().get(skipped_vis) ==
            simdjson::SUCCESS);
    CHECK(skipped_vis == 2);
    int64_t experts = 0;
    REQUIRE(doc["num_experts"].get_int64().get(experts) == simdjson::SUCCESS);
    CHECK(experts == kExperts);
    // The mixed codec split survives the varloop pipeline unchanged.
    std::vector<bool> codecs;
    for (auto v : doc["expert_nvfp4"].get_array())
        codecs.push_back(v.get_bool().value());
    REQUIRE(codecs.size() == static_cast<size_t>(kLayers));
    for (int l = 0; l < kLayers; ++l)
        CHECK(codecs[static_cast<size_t>(l)] == (l < kNovfp4Layers));
}

TEST_CASE("the varloop converter still rejects a fixed-pass checkpoint",
          "[inkling][varloop][convert]")
{
    using namespace lsxcommon::inkling;
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_varloop_fixed_in";
    const fs::path out = fs::temp_directory_path() / "inkling_varloop_fixed_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in, /*sane=*/false, /*varloop=*/false);
    auto st = ConvertInklingVarloopModel(in.string(), out.string());
    REQUIRE_FALSE(st.ok());
    CHECK(st.ToString().find("is_varloop") != std::string::npos);
}
