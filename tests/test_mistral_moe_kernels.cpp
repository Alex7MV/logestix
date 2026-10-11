// Mistral-Small-4-119B kernel-level test (stage 5): the MLA/RoPE/attention and
// MoE chain runs end to end on a synthetic converted model, produces FINITE
// logits, and repeated greedy SelfCheck runs are bit-identical (deterministic
// reductions, no atomics feeding the logits).

#include <catch2/catch_test_macros.hpp>

#include "test_mistral_moe_fixture.h"

#include "lsxcommon/mistral_moe_convert.h"
#include "lsxcommon/model_factory.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

namespace fs = std::filesystem;
using namespace lsxcommon;

namespace {
bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<IModel> MakeModel(const fs::path &root)
{
    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "mistral-small-4-119b";
    cfg.fast_math_mode = FastMathMode::kOff;
    return LsxModelFactory::Create(cfg);
}
} // namespace

TEST_CASE("Mistral MLA decode step is finite and bit-exact deterministic",
          "[mistral][inference]")
{
    if (!GpuAvailable()) std::exit(4);
    EnsureModelRegistry();
    auto in = fs::temp_directory_path() / "mistral_kern_in";
    auto out = fs::temp_directory_path() / "mistral_kern_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    mistraltest::WriteMistralParams(in);
    synthtest::WriteSafetensors(in, "consolidated-00001-of-00001.safetensors",
                                mistraltest::BuildSyntheticMistralModel());
    REQUIRE(mistral_moe::ConvertMistralModel(in.string(), out).ok());

    auto model = MakeModel(out);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {5, 7};
    req.max_gen_tokens = 2;

    const auto a = model->SelfCheck(req);
    const auto b = model->SelfCheck(req);
    REQUIRE(a.ok);
    REQUIRE(b.ok);
    bool any_nonzero = false;
    for (float v : a.final_logits)
    {
        REQUIRE(std::isfinite(v));
        if (v != 0.0f) any_nonzero = true;
    }
    REQUIRE(any_nonzero);
    REQUIRE(a.final_logits.size() == b.final_logits.size());
    REQUIRE(std::memcmp(a.final_logits.data(), b.final_logits.data(),
                        a.final_logits.size() * sizeof(float)) == 0);
    REQUIRE(a.first_gen_tok_id == b.first_gen_tok_id);
    model->Unload();
}
