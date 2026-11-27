// TML-d24-varloop-research model-stage test: synthetic converted tree only.
//
// Three things a working-but-wrong implementation gets wrong, and which the
// real-weight gates would only catch much later:
//
//   * IDENTITY — the row registers under its own kType and refuses the sibling's
//     artifact (format tag) instead of quietly decoding a fixed-pass tree with
//     a runtime pass knob;
//   * THE PASS KNOB — the active count is the checkpoint's declared default when
//     nothing is requested, and the operator override clamps into the TRAINED
//     range (1..5) rather than running a count the weights never saw;
//   * THE TOOL BRIDGE (stage 4b) — the row answers the conversation surface with
//     the family's verified codec (`<|content_invoke_tool_json|>` payloads), not
//     with the generic fallback.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "inkling_moe_varloop_constants.h"
#include "inkling_moe_varloop_model.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/inkling_moe_loop_tool_format.h"
#include "lsxcommon/inkling_moe_varloop_convert.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"

#include "test_inkling_moe_loop_fixture.h"

namespace fs = std::filesystem;
using namespace lsxcommon;
namespace vf = lsxcommon::inkling::varloop::frozen;
namespace lt = lsxcommon::inkling::loop::tools;

namespace {

constexpr const char *kTokFixture =
    TOKENIZER_TEST_DATA_DIR "/synthetic_inkling_loop_tok.json";
constexpr int kVarPassesMax = 5;  // this row's trained ceiling

fs::path ConvertVarloopFixture(const char *tag)
{
    const fs::path in = fs::temp_directory_path() /
                        (std::string("inkling_varloop_model_in_") + tag);
    const fs::path out = fs::temp_directory_path() /
                         (std::string("inkling_varloop_model_out_") + tag);
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    // `sane` values: the model test decodes, so the fixture cannot be the raw
    // byte pattern the layout tests pin.
    inklinglooptest::WriteCheckpoint(in, /*sane=*/true, /*varloop=*/true);
    auto st = lsxcommon::inkling::ConvertInklingVarloopModel(in.string(),
                                                            out.string());
    if (!st.ok()) FAIL(st.ToString());
    return out;
}

ModelInitConfig Cfg(const fs::path &root)
{
    ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = std::string(vf::kModelType);
    return cfg;
}

// Sets/clears the operator knob for one scope.
struct PassEnv {
    explicit PassEnv(const char *value)
    {
        if (value != nullptr) setenv("LSX_INKLING_VARLOOP_PASSES", value, 1);
        else unsetenv("LSX_INKLING_VARLOOP_PASSES");
    }
    ~PassEnv() { unsetenv("LSX_INKLING_VARLOOP_PASSES"); }
};

bool LoadTokenizer(ArrowTokenizerDb &db, fs::path &pinned)
{
    pinned = fs::temp_directory_path() / "lsx_inkling_varloop_model_tok";
    fs::remove_all(pinned);
    fs::create_directories(pinned);
    if (!BuildTokenizerArrowFiles(kTokFixture, pinned).ok()) return false;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) return false;
    return load_tokenizer_metadata(*v, *m, db).ok();
}

}  // namespace

TEST_CASE("varloop model registers under its own kType", "[inkling][varloop][model]")
{
    CHECK(vf::kVarLoop);
    CHECK(std::string(InklingMoeVarloopModel::kType) == vf::kModelType);
    CHECK(std::string(InklingMoeVarloopModel::kType) == "tml-d24-varloop-research");
    // The generated registry (models/tml-d24-varloop-research.json) resolves it:
    // a class the factory cannot create "does not exist" at runtime.
    ModelInitConfig cfg;
    cfg.model_type = vf::kModelType;
    cfg.model_path = "/nonexistent";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "tml-d24-varloop-research");
}

TEST_CASE("varloop model runs the declared pass count by default",
          "[inkling][varloop][model]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path dir = ConvertVarloopFixture("default");
    PassEnv none(nullptr);

    auto model = LsxModelFactory::Create(Cfg(dir));
    REQUIRE(model != nullptr);
    auto *var = dynamic_cast<InklingMoeVarloopModel *>(model.get());
    REQUIRE(var != nullptr);
    INFO("load failed");
    REQUIRE(model->Load());
    // The fixture declares 2 passes; with no override that is what runs, and the
    // slab still covers the trained ceiling.
    CHECK(var->ActiveLoopPasses() == inklinglooptest::kLoopPasses);
    CHECK(var->CacheSlotCount() ==
          inklinglooptest::kLayers +
              (kVarPassesMax - 1) * inklinglooptest::kLoopLayers);
    CHECK(var->LayerApplications() ==
          inklinglooptest::kLayers +
              (inklinglooptest::kLoopPasses - 1) * inklinglooptest::kLoopLayers);
    model->Unload();
    fs::remove_all(dir);
}

TEST_CASE("varloop model clamps the override into the trained range",
          "[inkling][varloop][model]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path dir = ConvertVarloopFixture("knob");

    struct Case {
        const char *env;
        int want;
    };
    for (const Case &c : {Case{"1", 1}, Case{"5", 5}, Case{"9", 5},
                          Case{"0", inklinglooptest::kLoopPasses}}) {
        CAPTURE(c.env);
        PassEnv env(c.env);
        auto model = LsxModelFactory::Create(Cfg(dir));
        REQUIRE(model != nullptr);
        auto *var = dynamic_cast<InklingMoeVarloopModel *>(model.get());
        REQUIRE(var != nullptr);
        REQUIRE(model->Load());
        CHECK(var->ActiveLoopPasses() == c.want);
        // The slab never resizes with the count: it is the ceiling's.
        CHECK(var->CacheSlotCount() ==
              inklinglooptest::kLayers +
                  (kVarPassesMax - 1) * inklinglooptest::kLoopLayers);
        CHECK(var->LayerApplications() ==
              inklinglooptest::kLayers +
                  (c.want - 1) * inklinglooptest::kLoopLayers);
        model->Unload();
    }
    fs::remove_all(dir);
}

TEST_CASE("varloop model refuses the fixed-pass sibling's artifact",
          "[inkling][varloop][model]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    // A Mini tree at this row's path: the format gate must stop it, because the
    // pass knob this row offers does not exist for those weights.
    const fs::path in = fs::temp_directory_path() / "inkling_varloop_model_fixed_in";
    const fs::path out = fs::temp_directory_path() / "inkling_varloop_model_fixed_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    inklinglooptest::WriteCheckpoint(in, /*sane=*/true, /*varloop=*/false);
    REQUIRE(lsxcommon::inkling::ConvertInklingLoopModel(in.string(), out.string())
                .ok());

    auto model = LsxModelFactory::Create(Cfg(out));
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());
    fs::remove_all(in);
    fs::remove_all(out);
}

TEST_CASE("varloop model frames conversations and parses native tool calls",
          "[inkling][varloop][model][fast]")
{
    // Stage 4b: the checkpoint's template is byte-identical to Inkling-2-Mini's,
    // so the row must reach the SAME codec — not the generic fallback.
    ArrowTokenizerDb db;
    fs::path pinned;
    REQUIRE(LoadTokenizer(db, pinned));

    auto model = LsxModelFactory::Create(Cfg("/nonexistent"));
    REQUIRE(model != nullptr);

    ChatConversation conv;
    conv.messages.push_back({"user", "Weather in Paris?", "", {}});
    std::vector<int32_t> ids;
    REQUIRE(model->BuildConversationInputIds(db, conv, ids));
    REQUIRE_FALSE(ids.empty());
    // The template opens with the thinking-effort system turn (it is emitted
    // before the first NON-system message), then the native user marker — and
    // the tokenizer's auto-BOS never appears, because this checkpoint has none.
    CHECK(ids.front() == lt::MarkerId(db, "<|message_system|>"));
    CHECK(std::find(ids.begin(), ids.end(), lt::MarkerId(db, "<|message_user|>")) !=
          ids.end());
    for (int32_t id : ids) CHECK(id >= 0);

    // The single-prompt path frames the same way (a chat checkpoint must never
    // see an unframed prompt).
    std::vector<int32_t> prompt_ids;
    REQUIRE(model->BuildPromptInputIds(db, "Weather in Paris?", prompt_ids));
    CHECK(prompt_ids == ids);

    // Parse a native tool call out of generated ids: the invoke marker is
    // special:true, so the parser scans IDS, not decoded text.
    std::vector<int32_t> gen;
    gen.push_back(lt::MarkerId(db, "<|message_model|>"));
    gen.push_back(lt::MarkerId(db, "<|content_invoke_tool_json|>"));
    const std::string payload = R"({"name":"get_weather","args":{"city":"Paris"}})";
    {
        std::vector<int32_t> body(payload.size() + 16);
        std::size_t n = body.size();
        // Returns the TOKEN COUNT on success (negative on error).
        const int32_t count = arrow_tokenizer_encode(db, payload.data(),
                                                    payload.size(), body.data(),
                                                    &n);
        REQUIRE(count > 0);
        body.resize(static_cast<std::size_t>(count));
        REQUIRE_FALSE(body.empty());
        gen.insert(gen.end(), body.begin(), body.end());
    }
    gen.push_back(lt::MarkerId(db, "<|end_message|>"));
    std::vector<ToolCall> calls;
    REQUIRE(model->ParseToolCalls(db, gen, calls));
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris"})");

    fs::remove_all(pinned);
}
