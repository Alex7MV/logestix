// Model identity from the manifest (host-only).
//
// The artifact path is <root>/<family>/<model_type>/<profile>, and both identity
// fields must come from models/<model_type>.json — not from the checkpoint's
// directory name, which is a download spelling. "Qwen3.5-9B" is the vault dir;
// `qwen3.5-9b` is the model_type that appears in the path.
#include <catch2/catch_test_macros.hpp>

#include "lsxstorage/roots.h"
#include "model_identity.h"

#include <cstdlib>
#include <string>

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

} // namespace

TEST_CASE("a vault directory resolves to its manifest identity",
          "[model_identity]") {
    // The two Task 9a models, by the paths they actually live at.
    const auto qwen = lsxforge::ModelIdentityFor("/mnt/fast_storage/lsx_model/Qwen/Qwen3.5-9B");
    REQUIRE(qwen.has_value());
    CHECK(qwen->model_type == "qwen3.5-9b");
    CHECK(qwen->family == "qwen");
    CHECK(qwen->org == "Qwen");
    CHECK(qwen->model_dir == "Qwen3.5-9B");

    const auto phi = lsxforge::ModelIdentityFor("/mnt/fast_storage/lsx_model/microsoft/phi-4");
    REQUIRE(phi.has_value());
    CHECK(phi->model_type == "phi-4");
    CHECK(phi->family == "phi");
    CHECK(phi->org == "microsoft");
    CHECK(phi->model_dir == "phi-4");
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
