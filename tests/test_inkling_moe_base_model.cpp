// Inkling (base) model-stage test (Model Ingestion Protocol stage 3/5).
//
// Runs the engine on the synthetic converted tree (LSX_TEST_ALLOW_SYNTHETIC_
// GEOMETRY=1). What this pins is structural, not numeric (a synthetic
// checkpoint's weights are a pattern, there is no source to compare against):
//   * the factory resolves "inkling" (and the vault alias);
//   * Load() succeeds and the frozen passport cross-check runs;
//   * Infer decodes tokens in range, deterministically (two identical requests
//     are bit-identical), and a different prompt changes the logits row;
//   * the streaming callback sees exactly the generated tokens, in order;
//   * a pre-set abort yields zero tokens and no error.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_base_convert.h>
#include "inkling_moe_base_model.h"

#include <lsxcommon/model_factory.h>
#include <lsxcommon/model_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "test_inkling_base_fixture.h"

namespace fs = std::filesystem;
using namespace inklingbasetest;
namespace base = lsxcommon::inkling::base;

namespace {

fs::path EnsureSyntheticModel()
{
    const fs::path in = fs::temp_directory_path() / "inkling_base_model_in";
    const fs::path out = fs::temp_directory_path() / "inkling_base_model_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);
    auto st = base::ConvertInklingBaseModel(in.string(), out.string());
    REQUIRE(st.ok());
    return out;
}

lsxcommon::ModelInitConfig Cfg(const fs::path &root)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = lsxcommon::InklingMoeBaseModel::kType;
    return cfg;
}

std::vector<int32_t> Prompt(int n, int seed)
{
    std::vector<int32_t> ids(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        ids[static_cast<size_t>(i)] = (seed * 37 + i * 11 + 5) % (kUnpadded - 1) + 1;
    return ids;
}

bool SameFloatRow(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size() || a.empty()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("inkling base factory resolves the type and the vault alias",
          "[inkling][model]")
{
    lsxcommon::EnsureModelRegistry();
    const fs::path root = fs::temp_directory_path();  // never Load()ed here
    auto by_type = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(by_type != nullptr);
    REQUIRE(by_type->Type() == "inkling");
    auto cfg = Cfg(root);
    cfg.model_type = "inkling";  // the lower-cased vault dir alias
    auto by_alias = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(by_alias != nullptr);
    REQUIRE(by_alias->Type() == "inkling");
}

TEST_CASE("inkling base model decodes deterministically on the synthetic tree",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    lsxcommon::EnsureModelRegistry();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "inkling");
    REQUIRE(model->Load());
    REQUIRE(model->ResidentWeightBytes() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = Prompt(6, 1);
    req.max_gen_tokens = 4;
    req.ignore_eos = true;  // synthetic ids: never stop early

    const auto first = model->Infer(req);
    INFO("error: " << first.error_msg);
    REQUIRE(first.ok);
    REQUIRE(first.num_tokens == 4);
    REQUIRE(first.first_gen_tok_id == first.gen_tokens.front());
    REQUIRE(first.num_layers == kLayers);
    REQUIRE(first.weight_resident_bytes > 0);
    REQUIRE(first.final_logits.size() == static_cast<size_t>(kUnpadded));
    for (int32_t id : first.gen_tokens)
        REQUIRE((id >= 0 && id < kUnpadded));
    for (float v : first.final_logits) REQUIRE(std::isfinite(v));

    // ── determinism: a second identical request is bit-identical ──────────
    const auto second = model->Infer(req);
    REQUIRE(second.ok);
    REQUIRE(second.gen_tokens == first.gen_tokens);
    REQUIRE(SameFloatRow(second.final_logits, first.final_logits));

    // ── not "one fixed answer for every prompt" ────────────────────────────
    lsxcommon::ModelRequest other = req;
    other.input_ids = Prompt(6, 9);
    const auto alt = model->Infer(other);
    REQUIRE(alt.ok);
    REQUIRE(alt.gen_tokens.size() == 4);
    CHECK_FALSE(SameFloatRow(alt.final_logits, first.final_logits));

    model->Unload();
}

TEST_CASE("inkling base model streams and honours a pre-set abort",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    lsxcommon::EnsureModelRegistry();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    SECTION("the callback sees exactly the generated tokens, in order")
    {
        lsxcommon::ModelRequest req;
        req.input_ids = Prompt(5, 2);
        req.max_gen_tokens = 3;
        req.ignore_eos = true;
        std::vector<int32_t> seen;
        std::vector<uint32_t> steps;
        req.token_callback = [&](const lsxcommon::TokenChunk &c) {
            seen.push_back(static_cast<int32_t>(c.token_id));
            steps.push_back(c.step_t);
        };
        const auto r = model->Infer(req);
        INFO("error: " << r.error_msg);
        REQUIRE(r.ok);
        REQUIRE(seen == r.gen_tokens);
        for (size_t i = 0; i < steps.size(); ++i)
            REQUIRE(steps[i] == static_cast<uint32_t>(i));
    }

    SECTION("an abort flag set before the first step yields no tokens")
    {
        lsxcommon::ModelRequest req;
        req.input_ids = Prompt(5, 2);
        req.max_gen_tokens = 8;
        req.ignore_eos = true;
        std::atomic<bool> abort{true};
        req.abort_flag = &abort;
        std::vector<int32_t> seen;
        req.token_callback = [&](const lsxcommon::TokenChunk &c) {
            seen.push_back(static_cast<int32_t>(c.token_id));
        };
        const auto r = model->Infer(req);
        // A cancellation is not an error: it is a short, successful generation.
        REQUIRE(r.ok);
        REQUIRE(r.gen_tokens.empty());
        REQUIRE(seen.empty());
        REQUIRE(r.ttft_ms == 0.0);
    }

    model->Unload();
}

TEST_CASE("inkling base model refuses a request longer than the engine context",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    lsxcommon::EnsureModelRegistry();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = Prompt(8, 3);
    req.max_gen_tokens = base::kMaxContext;  // prompt + gen > bound
    req.ignore_eos = true;
    const auto r = model->Infer(req);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.error_msg.empty());

    model->Unload();
}