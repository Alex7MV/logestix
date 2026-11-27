#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <shared_mutex>
#include <filesystem>
#include <string>

#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/gpu_perf.h"

namespace {

std::string TestModelPath()
{
    if (const char *p = std::getenv("LSX_TEST_MODEL_PATH"))
        return p;
    return R"(D:\Programs\lsx_model\DeepSeek-V4-Flash)";
}

// Real-model tests need a model dir with a ready logestix index. Unset or
// absent dirs (e.g. the Windows default path on a Linux runner) skip them
// instead of failing the whole suite.
bool TestModelPathUsable()
{
    const auto idx =
        std::filesystem::path(TestModelPath()) / "logestix" / "model.index.json";
    std::error_code ec;
    return std::filesystem::exists(idx, ec) && !ec;
}

} // namespace

TEST_CASE("DetectType returns normalized type", "[factory]")
{
    if (!TestModelPathUsable())
        SKIP("model dir lacks logestix/model.index.json (set LSX_TEST_MODEL_PATH)");
    std::string t = lsxcommon::LsxModelFactory::DetectType(TestModelPath());
    REQUIRE(t == "deepseek-v4-flash");
}

TEST_CASE("DetectType returns empty for invalid path", "[factory]")
{
    std::string t = lsxcommon::LsxModelFactory::DetectType(
        R"(D:\nonexistent\model\path)");
    REQUIRE(t.empty());
}

TEST_CASE("Create returns nullptr for unknown type", "[factory]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = TestModelPath();
    cfg.model_type = "no-such-model";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m == nullptr);
}

TEST_CASE("Registry round-trip with mock model", "[factory]")
{
    bool created = false;
    lsxcommon::RegisterModel("mock-model",
        [&](const lsxcommon::ModelInitConfig &) {
            created = true;
            return std::unique_ptr<lsxcommon::IModel>(nullptr);
        });

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = TestModelPath();
    cfg.model_type = "mock-model";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(created);
    REQUIRE(m == nullptr);  // mock returns nullptr by design
}

TEST_CASE("Create + Load + Inspect on real model dir", "[factory]")
{
    if (!TestModelPathUsable())
        SKIP("model dir lacks logestix/model.index.json (set LSX_TEST_MODEL_PATH)");
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = TestModelPath();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "deepseek-v4-flash");
    REQUIRE(m->Load());
    auto info = m->Inspect();
    REQUIRE(!info.shards.empty());
    m->Unload();
}
namespace {

class TypedProbe final : public lsxcommon::IModel {
public:
    static const std::string kType;
    explicit TypedProbe(const lsxcommon::ModelInitConfig &) {}
    bool Load() override { return false; }
    lsxcommon::gpu::InferenceResult Infer(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    void Unload() override {}
    lsxcommon::SafetensorsModel Inspect() override { return {}; }
    lsxcommon::gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    const std::string &Type() const override
    {
        static std::string t{"typed-probe"};
        return t;
    }
    lsxcommon::ArrowTokenizerDb &Tokenizer() override
    {
        static lsxcommon::ArrowTokenizerDb db{};
        return db;
    }
    std::shared_mutex &ResidencyMutex() override
    {
        static std::shared_mutex m;
        return m;
    }
};
const std::string TypedProbe::kType = "typed-probe";

} // namespace

TEST_CASE("Typed Register<T>() resolves kType and constructs", "[factory]")
{
    lsxcommon::LsxModelFactory::Register<TypedProbe>();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_type = "typed-probe";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "typed-probe");
}

TEST_CASE("Create applies the per-GPU perf policy without engine session",
          "[factory]")
{
    if (lsxcommon::perf::ProbeGpuPerfProfile().name.empty()) SKIP();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = R"(D:\nonexistent\model\path)";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);  // nullptr; policy on entry
    REQUIRE(m == nullptr);
    REQUIRE(lsxcommon::perf::CurrentKnobsSource() !=
            lsxcommon::perf::PerfSource::kLegacy);
}
