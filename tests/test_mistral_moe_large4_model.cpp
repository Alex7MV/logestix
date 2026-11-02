// Mistral-Large-4 model-stage test (Model Ingestion Protocol stages 3 + 5/8).
// Runs the whole decode pipeline on a synthetic converted model and pins:
//   1. finite, soft-capped logits for the full vocab (a missing kernel launch
//      or a half-wired layer shows up here);
//   2. determinism across a fresh instance;
//   3. streaming through the shared GenerationStream, cooperative cancel, and
//      continuation-vs-fresh-prefill equivalence.
#include <catch2/catch_test_macros.hpp>

#include "mistral_moe_large4_constants.h"
#include "mistral_moe_large4_model.h"
#include "test_mistral_moe_large4_fixture.h"
#include "test_streaming_assert.h"

#include "lsxcommon/log.h"
#include "lsxcommon/mistral_moe_large4_convert.h"
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
    auto in = fs::temp_directory_path() / "mistral_large4_model_in";
    auto out = fs::temp_directory_path() / "mistral_large4_model_out";
    std::error_code ec;
    fs::remove_all(out, ec);
    mistrallarge4test::Large4SynthGeom g;
    mistrallarge4test::WriteCheckpoint(in, g);
    REQUIRE(mistral_large4::ConvertMistralLarge4Model(in.string(), out).ok());
    return out;
}

std::unique_ptr<IModel> MakeModel(const fs::path &root)
{
    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = mistral_large4::frozen::kType;
    return LsxModelFactory::Create(cfg);
}
} // namespace

TEST_CASE("Mistral-Large-4 runs a finite, soft-capped decode step",
          "[mistral_large4][inference]")
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
    req.ignore_eos = true;
    const auto r = model->Infer(req);
    INFO("error: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(r.gen_tokens.size() == 2);
    REQUIRE(r.final_logits.size() ==
            static_cast<size_t>(mistrallarge4test::Large4SynthGeom{}.vocab));
    bool any_nonzero = false;
    for (float v : r.final_logits)
    {
        REQUIRE(std::isfinite(v));
        REQUIRE(std::fabs(v) <= 30.0001f);
        if (v != 0.0f) any_nonzero = true;
    }
    REQUIRE(any_nonzero);
    model->Unload();
}

TEST_CASE("Mistral-Large-4 decode is deterministic", "[mistral_large4][inference]")
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

TEST_CASE("Mistral-Large-4 streaming reaches the token callback",
          "[mistral_large4][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    const fs::path root = BuildConverted();
    auto model = MakeModel(root);
    REQUIRE(model->Load());
    ModelRequest req;
    req.input_ids = {5};
    req.max_gen_tokens = 3;
    ExpectStreamMatchesTokens(*model, req);
    model->Unload();
}

TEST_CASE("Mistral-Large-4 honours cooperative cancel",
          "[mistral_large4][inference][lifecycle]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    const fs::path root = BuildConverted();
    auto model = MakeModel(root);
    REQUIRE(model->Load());
    ModelRequest req;
    req.input_ids = {5, 7, 9};
    ExpectPreSetAbortStops(*model, req);
    ExpectCooperativeCancelStops(*model, req);
    model->Unload();
}

TEST_CASE("Mistral-Large-4 continuation matches a fresh prefill",
          "[mistral_large4][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    const fs::path root = BuildConverted();
    auto model = MakeModel(root);
    REQUIRE(model->Load());

    ModelRequest first;
    first.input_ids = {5, 7, 9};
    first.max_gen_tokens = 2;
    first.ignore_eos = true;
    const auto r1 = model->Infer(first);
    REQUIRE(r1.ok);
    REQUIRE(r1.gen_tokens.size() == 2);

    ModelRequest cont;
    cont.input_ids = first.input_ids;
    cont.input_ids.insert(cont.input_ids.end(), r1.gen_tokens.begin(),
                          r1.gen_tokens.end());
    cont.max_gen_tokens = 2;
    cont.ignore_eos = true;
    const auto r2 = model->Infer(cont);
    REQUIRE(r2.ok);
    model->Unload();

    auto fresh = MakeModel(root);
    REQUIRE(fresh->Load());
    ModelRequest same = cont;
    const auto r3 = fresh->Infer(same);
    REQUIRE(r3.ok);
    fresh->Unload();

    CHECK(r2.gen_tokens == r3.gen_tokens);
    REQUIRE(r2.final_logits.size() == r3.final_logits.size());
    double max_abs = 0.0;
    for (size_t i = 0; i < r2.final_logits.size(); ++i)
        max_abs = std::max(max_abs,
                           (double)std::abs(r2.final_logits[i] -
                                            r3.final_logits[i]));
    INFO("max logit delta = " << max_abs);
    CHECK(max_abs < 1e-3);
}
