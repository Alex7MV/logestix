#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <shared_mutex>
#include <filesystem>
#include <fstream>
#include <string>

#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/gpu_perf.h"

#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"

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
        std::filesystem::path(TestModelPath()) / "model.index.json";
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

// ── The path a caller names vs the artifact it means ─────────────────────
//
// DetectType reads `<path>/model.index.json`, so it only answers for an ARTIFACT
// directory. Everything else reached Create() as "cannot autodetect model type",
// which named the wrong problem for the spelling the tree itself uses: the
// registry IDENTITY (`qwen3.5-9b`) — the CLI defaults, every gate and every
// document — and the store path `Qwen3.5-9B` that `ls` shows. Create now
// resolves anything DetectType could not read, and reports the resolver's own
// reason when the name really is unknown.
namespace {

// LSX_ARTIFACT_ROOT for one case, restored on scope exit: Resolve searches it.
class ScopedRoot {
public:
    explicit ScopedRoot(const std::filesystem::path &p) : name_("LSX_ARTIFACT_ROOT")
    {
        if (const char *prev = std::getenv(name_)) { previous_ = prev; had_ = true; }
#if defined(_WIN32)
        _putenv_s(name_, p.string().c_str());
#else
        setenv(name_, p.string().c_str(), 1);
#endif
    }
    ~ScopedRoot()
    {
#if defined(_WIN32)
        _putenv_s(name_, had_ ? previous_.c_str() : "");
#else
        if (had_) setenv(name_, previous_.c_str(), 1); else unsetenv(name_);
#endif
    }
    ScopedRoot(const ScopedRoot &) = delete;
    ScopedRoot &operator=(const ScopedRoot &) = delete;

private:
    const char *name_;
    std::string previous_;
    bool had_ = false;
};

} // namespace

TEST_CASE("Create resolves a registry identity to its artifact", "[factory]") {
    // A store-shaped artifact written by the real emitter, so the identity comes
    // from artifact.fb exactly as it would in production.
    const auto root = std::filesystem::temp_directory_path() / "lsx_factory_res";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const auto art = root / "Acme" / "Probe-Model" / "nvfp4";
    lsx::storage::ArtifactSpec spec;
    spec.model_type = TypedProbe::kType;
    spec.family = "probe";
    spec.profile = "nvfp4";
    spec.org = "Acme";
    spec.model_dir = "Probe-Model";
    spec.model_name = "Probe-Model";
    std::string err;
    lsx::storage::EmitArtifact(spec, art, &err);
    REQUIRE(lsx::storage::IsArtifactDir(art));
    {
        std::ofstream f(art / "model.index.json");
        f << R"({"model": "typed-probe"})";
    }
    ScopedRoot scoped(root);
    lsxcommon::LsxModelFactory::Register<TypedProbe>();

    SECTION("the artifact directory itself") {
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = art.string();
        REQUIRE(lsxcommon::LsxModelFactory::Create(cfg) != nullptr);
    }
    SECTION("the registry identity") {
        // The spelling every CLI default and every gate uses. Before Create
        // resolved, this was "cannot autodetect model type from typed-probe".
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = TypedProbe::kType;
        auto m = lsxcommon::LsxModelFactory::Create(cfg);
        REQUIRE(m != nullptr);
        CHECK(m->Type() == TypedProbe::kType);
    }
    SECTION("the store's directory name") {
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = "Probe-Model";
        REQUIRE(lsxcommon::LsxModelFactory::Create(cfg) != nullptr);
    }
    SECTION("a path one level inside the artifact") {
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = (art / "model.index.json").string();
        REQUIRE(lsxcommon::LsxModelFactory::Create(cfg) != nullptr);
    }
    SECTION("an unknown name is refused") {
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = "not-a-model";
        CHECK(lsxcommon::LsxModelFactory::Create(cfg) == nullptr);
    }
    std::filesystem::remove_all(root, ec);
}
