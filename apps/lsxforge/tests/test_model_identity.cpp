// Model identity from the manifest (host-only).
//
// The artifact path MIRRORS the source store:
//     <ARTIFACT_ROOT>/<org>/<model_dir>/<profile>
// Both identity fields come from models/<org>/<model>.json — not from the
// checkpoint's directory name, which is a download spelling, and not from the
// path the caller typed. "Qwen3.5-9B" is the store's directory; `qwen3.5-9b` is
// the model_type the manifest, the gates and the docs use.
#include <catch2/catch_test_macros.hpp>

#include "lsxstorage/roots.h"
#include "model_identity.h"

#include "model_manifest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

class ScopedEnv {
public:
    ScopedEnv(const char *name, const char *value) : name_(name)
    {
        if (const char *prev = std::getenv(name)) {
            previous_ = prev;
            had_ = true;
        }
        set(value);
    }
    ~ScopedEnv() { set(had_ ? previous_.c_str() : nullptr); }
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
    void set(const char *v)
    {
#if defined(_WIN32)
        _putenv_s(name_, v ? v : "");
#else
        if (v) setenv(name_, v, 1); else unsetenv(name_);
#endif
    }
    const char *name_;
    std::string previous_;
    bool had_ = false;
};

// The checkpoint of a registry row: `<source root>/<org>/<model_dir>`. Spelled
// here (not in a literal) so a store that moves disk cannot leave this test
// reading a path that does not exist.
std::filesystem::path CheckpointOf(std::string_view model_type)
{
    for (const auto &row : lsxmanifest::kModels)
        if (model_type == row.model_type)
            return lsx::storage::SourceRoot() / row.vault_org / row.vault_dir;
    return {};
}

} // namespace

TEST_CASE("a store path resolves to its manifest identity", "[model_identity]") {
    // The two Task 9a models, by the shape the store uses. The prefix is
    // illustrative: identity comes from the leaf + the registry, never from the
    // root, so a copy elsewhere resolves the same way (next case).
    const auto qwen = lsxforge::ModelIdentityFor("/some/store/Qwen/Qwen3.5-9B");
    REQUIRE(qwen.has_value());
    CHECK(qwen->model_type == "qwen3.5-9b");
    CHECK(qwen->family == "qwen");
    CHECK(qwen->org == "Qwen");
    CHECK(qwen->model_dir == "Qwen3.5-9B");

    const auto phi = lsxforge::ModelIdentityFor("/some/store/microsoft/phi-4");
    REQUIRE(phi.has_value());
    CHECK(phi->model_type == "phi-4");
    CHECK(phi->family == "phi");
    CHECK(phi->org == "microsoft");
    CHECK(phi->model_dir == "phi-4");
}

TEST_CASE("the org comes from the registry, not from the path it was typed with",
          "[model_identity]") {
    // A copy of a checkpoint on another disk, a path a script printed, an
    // operator's scratch directory: the parent's name says nothing about which
    // org owns the model. The registry does — and an artifact placed under a
    // directory named after wherever the input happened to sit is exactly the
    // bug the org-nested store exists to prevent.
    const auto id = lsxforge::ModelIdentityFor("/tmp/some-copy/Qwen3.5-9B");
    REQUIRE(id.has_value());
    CHECK(id->org == "Qwen");
    CHECK(id->model_dir == "Qwen3.5-9B");
    CHECK(id->model_type == "qwen3.5-9b");
}

TEST_CASE("a bare model name resolves too", "[model_identity]") {
    const auto id = lsxforge::ModelIdentityFor("phi-4");
    REQUIRE(id.has_value());
    CHECK(id->model_type == "phi-4");
}

TEST_CASE("an unknown name is refused rather than guessed", "[model_identity]") {
    // Guessing an identity would place the artifact where nothing looks for it.
    CHECK_FALSE(lsxforge::ModelIdentityFor("not-a-real-model").has_value());
    CHECK_FALSE(lsxforge::ModelIdentityFor("").has_value());
}

TEST_CASE("the profile dir mirrors the source store layout",
          "[model_identity]") {
    ScopedEnv e("LSX_ARTIFACT_ROOT", "/tmp/art");
    const auto id = lsxforge::ModelIdentityFor("Qwen3.5-9B");
    REQUIRE(id.has_value());

    const auto dir = lsxforge::ArtifactProfileDir(*id, "nvfp4");
    // Mirrors the source store: <root>/<org>/<model>/<profile>.
    CHECK(dir == "/tmp/art/Qwen/Qwen3.5-9B/nvfp4");
    // No intermediate `logestix` segment: the profile IS the artifact.
    CHECK(dir.find("logestix") == std::string::npos);
}

TEST_CASE("no artifact root means no profile dir, never an invented one",
          "[model_identity]") {
    ScopedEnv e("LSX_ARTIFACT_ROOT", nullptr);
    const auto id = lsxforge::ModelIdentityFor("phi-4");
    REQUIRE(id.has_value());
    CHECK(lsxforge::ArtifactProfileDir(*id, "nvfp4").empty());
}

// ── The chat template a checkpoint carries inside tokenizer_config.json ────
//
// Hugging Face allows the template in a chat_template.jinja beside the weights
// OR as a "chat_template" string inside tokenizer_config.json. microsoft/phi-4
// is the second shape, and the bootstrap conversion recorded chat_template = ""
// for it until this existed — an artifact that cannot reproduce the framing it
// was converted from, with nothing reporting the loss.
namespace {
class TempDir {
public:
    TempDir()
    {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("lsx-identity-" + std::to_string(counter++));
        std::filesystem::create_directories(path_);
    }
    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    const std::filesystem::path &path() const { return path_; }

private:
    std::filesystem::path path_;
};

void WriteText(const std::filesystem::path &p, const std::string &text)
{
    std::ofstream out(p, std::ios::binary);
    out << text;
}
} // namespace

TEST_CASE("the template is read out of tokenizer_config.json", "[model_identity]") {
    TempDir d;
    const auto cfg = d.path() / "tokenizer_config.json";
    // A JSON string needs its quotes escaped, which is exactly the content a
    // real template is full of.
    WriteText(cfg, R"({"chat_template": "{% for m in messages %}{{ '<|im_start|>' + m['role'] }}{% endfor %}", "model_max_length": 4096})");
    const auto tpl = lsxforge::ChatTemplateFromTokenizerConfig(cfg.string());
    CHECK(tpl.find("<|im_start|>") != std::string::npos);
    CHECK(tpl.find("m['role']") != std::string::npos);
}

TEST_CASE("a config with no template yields no template, never a guess",
          "[model_identity]") {
    TempDir d;
    const auto cfg = d.path() / "tokenizer_config.json";
    WriteText(cfg, R"({"model_max_length": 4096})");
    CHECK(lsxforge::ChatTemplateFromTokenizerConfig(cfg.string()).empty());
    // An absent file and a malformed one are the same answer: nothing to record.
    CHECK(lsxforge::ChatTemplateFromTokenizerConfig(
              (d.path() / "nope.json").string()).empty());
    WriteText(d.path() / "broken.json", "{not json");
    CHECK(lsxforge::ChatTemplateFromTokenizerConfig(
              (d.path() / "broken.json").string()).empty());
}

TEST_CASE("the checkpoint's own template is what phi-4 records",
          "[model_identity]") {
    // The real checkpoint, if it is on this box: this is the case the conversion
    // caught, and asserting it against the real file is what makes the fix
    // meaningful rather than a fixture agreeing with itself.
    const std::filesystem::path cfg =
        CheckpointOf("phi-4") / "tokenizer_config.json";
    if (!std::filesystem::exists(cfg))
        SKIP("microsoft/phi-4 is not in the source store");
    const auto tpl = lsxforge::ChatTemplateFromTokenizerConfig(cfg.string());
    REQUIRE_FALSE(tpl.empty());
    CHECK(tpl.find("<|im_start|>") != std::string::npos);
    CHECK(tpl.find("<|im_end|>") != std::string::npos);
}
