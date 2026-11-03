// Kimi-K3 model-stage test: the full 93-layer pipeline on a synthetic converted
// tree (the fixture is a 4-layer pipe with 2 KDA + 2 MLA layers, AttnRes block
// 2, and 4 routed experts). Pins finite logits, determinism, streaming, cancel
// and continuation-vs-fresh-prefill equivalence.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "kimi_moe_k3_constants.h"
#include "kimi_moe_k3_model.h"
#include "lsxcommon/kimi_moe_k3_convert.h"
#include "lsxcommon/model_factory.h"
#include "test_kimi_k3_fixture.h"
#include "test_streaming_assert.h"

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
    auto in = fs::temp_directory_path() / "kimi_k3_model_in";
    auto out = fs::temp_directory_path() / "kimi_k3_model_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    kimik3test::WriteCheckpoint(in);
    const auto st = kimi_k3::ConvertKimiK3Model(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    return out;
}

std::unique_ptr<IModel> MakeModel(const fs::path &root)
{
    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kimi_k3::frozen::kModelType;
    cfg.fast_math_mode = FastMathMode::kOff;
    return LsxModelFactory::Create(cfg);
}

}  // namespace

TEST_CASE("kimi-k3 runs a finite decode step", "[kimi-k3][inference]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    EnsureModelRegistry();
    const fs::path root = BuildConverted();
    auto model = MakeModel(root);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("load failed");
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {3, 5, 7};
    req.max_gen_tokens = 2;
    req.ignore_eos = true;
    const auto r = model->Infer(req);
    if (!r.ok) FAIL(r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(r.gen_tokens.size() == 2);
    REQUIRE(r.final_logits.size() ==
            static_cast<size_t>(kimik3test::kVocab));
    bool any_nonzero = false;
    for (float v : r.final_logits) {
        REQUIRE(std::isfinite(v));
        if (v != 0.0f) any_nonzero = true;
    }
    REQUIRE(any_nonzero);
    CHECK(r.weight_resident_bytes > 0);
    model->Unload();
}

TEST_CASE("kimi-k3 decode is deterministic", "[kimi-k3][inference]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    EnsureModelRegistry();
    const fs::path root = BuildConverted();
    auto run = [&]() {
        auto model = MakeModel(root);
        REQUIRE(model->Load());
        ModelRequest req;
        req.input_ids = {2, 4, 6};
        req.max_gen_tokens = 3;
        req.ignore_eos = true;
        auto r = model->Infer(req);
        model->Unload();
        return r;
    };
    const auto a = run();
    const auto b = run();
    REQUIRE(a.ok);
    REQUIRE(b.ok);
    REQUIRE(a.gen_tokens == b.gen_tokens);
    REQUIRE(a.final_logits.size() == b.final_logits.size());
    for (size_t i = 0; i < a.final_logits.size(); ++i)
        REQUIRE(a.final_logits[i] == b.final_logits[i]);
}

TEST_CASE("kimi-k3 streams every token", "[kimi-k3][inference]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    EnsureModelRegistry();
    const fs::path root = BuildConverted();
    auto model = MakeModel(root);
    REQUIRE(model->Load());
    ModelRequest req;
    req.input_ids = {5};
    req.max_gen_tokens = 3;
    req.ignore_eos = true;
    ExpectStreamMatchesTokens(*model, req);
    model->Unload();
}

TEST_CASE("kimi-k3 honours cooperative cancel", "[kimi-k3][inference][lifecycle]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    EnsureModelRegistry();
    const fs::path root = BuildConverted();
    auto model = MakeModel(root);
    REQUIRE(model->Load());
    ModelRequest req;
    req.input_ids = {5, 7, 9};
    req.ignore_eos = true;
    ExpectPreSetAbortStops(*model, req);
    ExpectCallbackAbortStops(*model, req);
    model->Unload();
}

TEST_CASE("kimi-k3 continuation matches a fresh prefill",
          "[kimi-k3][inference]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
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

    ModelRequest cont = first;
    cont.input_ids.insert(cont.input_ids.end(), r1.gen_tokens.begin(),
                          r1.gen_tokens.end());
    const auto r2 = model->Infer(cont);
    REQUIRE(r2.ok);
    model->Unload();

    auto fresh = MakeModel(root);
    REQUIRE(fresh->Load());
    const auto r3 = fresh->Infer(cont);
    REQUIRE(r3.ok);
    fresh->Unload();

    CHECK(r2.gen_tokens == r3.gen_tokens);
    REQUIRE(r2.final_logits.size() == r3.final_logits.size());
    double max_abs = 0.0;
    for (size_t i = 0; i < r2.final_logits.size(); ++i)
        max_abs = std::max(max_abs,
                           static_cast<double>(std::abs(r2.final_logits[i] -
                                                        r3.final_logits[i])));
    INFO("max logit delta = " << max_abs);
    CHECK(max_abs < 1e-3);
}
