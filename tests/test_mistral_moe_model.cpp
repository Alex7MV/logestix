// Mistral-Small-4-119B model-stage test (Model Ingestion Protocol stages 5/8).
// Runs the whole decode pipeline on a synthetic converted model and pins:
//   1. a decode step produces FINITE logits for the full vocab (catches a
//      missing kernel launch or a half-wired layer);
//   2. determinism: the same request twice gives bit-identical tokens;
//   3. streaming through the shared GenerationStream.

#include <catch2/catch_test_macros.hpp>

#include "test_mistral_moe_fixture.h"

#include "lsxcommon/log.h"
#include "lsxcommon/mistral_moe_convert.h"
#include "lsxcommon/model_factory.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace lsxcommon;
namespace fs = std::filesystem;

namespace {
bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

fs::path BuildConverted()
{
    auto in = fs::temp_directory_path() / "mistral_model_in";
    auto out = fs::temp_directory_path() / "mistral_model_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    mistraltest::WriteMistralParams(in);
    synthtest::WriteSafetensors(in, "consolidated-00001-of-00001.safetensors",
                                mistraltest::BuildSyntheticMistralModel());
    REQUIRE(mistral_moe::ConvertMistralModel(in.string(), out).ok());
    return out;
}

std::unique_ptr<IModel> MakeModel(const fs::path &root)
{
    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "mistral-small-4-119b";
    return LsxModelFactory::Create(cfg);
}
} // namespace

TEST_CASE("Mistral model runs a finite decode step", "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    const fs::path root = BuildConverted();

    auto model = MakeModel(root);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {5, 7, 9};
    req.max_gen_tokens = 2;
    const auto r = model->Infer(req);
    INFO("error: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(r.gen_tokens.size() == 2);
    REQUIRE(r.final_logits.size() ==
            static_cast<size_t>(mistraltest::MistralSynthGeom{}.vocab));
    bool any_nonzero = false;
    for (float v : r.final_logits)
    {
        REQUIRE(std::isfinite(v));
        if (v != 0.0f) any_nonzero = true;
    }
    REQUIRE(any_nonzero);
    model->Unload();
}

TEST_CASE("Mistral decode is deterministic", "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    const fs::path root = BuildConverted();

    auto run = [&]() {
        auto model = MakeModel(root);
        REQUIRE(model->Load());
        ModelRequest req;
        req.input_ids = {3, 4, 5};
        req.max_gen_tokens = 3;
        auto r = model->Infer(req);
        model->Unload();
        return r;
    };
    const auto a = run();
    const auto b = run();
    REQUIRE(a.ok);
    REQUIRE(b.ok);
    REQUIRE(a.gen_tokens == b.gen_tokens);
}

TEST_CASE("Mistral streaming reaches the token callback", "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    const fs::path root = BuildConverted();

    auto model = MakeModel(root);
    REQUIRE(model->Load());

    int emitted = 0;
    ModelRequest req;
    req.input_ids = {5};
    req.max_gen_tokens = 4;
    req.token_callback = [&](const TokenChunk &) { ++emitted; };
    const auto r = model->Infer(req);
    REQUIRE(r.ok);
    REQUIRE(emitted == static_cast<int>(r.gen_tokens.size()));
    model->Unload();
}
