#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include "test_kimi_moe_fixture.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/model_factory.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "lsxcommon/log.h"
#include "lsxcommon/tokenizer.h"

using namespace lsxcommon;

namespace {

std::string BuildConvertedFixture()
{
    namespace fs = std::filesystem;
    auto in = fs::temp_directory_path() / "kimi_model_in";
    auto out = fs::temp_directory_path() / "kimi_model_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);

    kimitest::WriteKimiConfig(in);
    kimitest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               kimitest::BuildSyntheticKimiModel(3));
    kimitest::WriteKimiTokenizerFiles(in);
    REQUIRE(lsxcommon::kimi::ConvertKimiModel(in.string(), out).ok());
    return out.string();
}

} // namespace

TEST_CASE("Kimi model type is registered in the factory", "[kimi][model]")
{
    EnsureModelRegistry();
    ModelInitConfig empty_cfg;
    empty_cfg.model_type = "kimi-k2.7-code";
    auto model = LsxModelFactory::Create(empty_cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "kimi-k2.7-code");
}

TEST_CASE("Kimi model lifecycle on synthetic fixture", "[kimi][model]")
{
    EnsureModelRegistry();
    ModelInitConfig empty_cfg;
    empty_cfg.model_type = "kimi-k2.7-code";
    auto model = LsxModelFactory::Create(empty_cfg);
    REQUIRE(model != nullptr);

    // Fixture layout (not a real vault path): Load() must fail cleanly
    // because the pinned weights do not match the locked geometry.
    ModelInitConfig cfg;
    cfg.model_path = BuildConvertedFixture();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "kimi-k2.7-code";
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);

    // Geometry validation rejects the 3-layer/64-vocab fixture.
    CHECK_FALSE(m->Load());
    CHECK(m->ResidentWeightBytes() == 0);

    // Unload is safe even after failed Load.
    m->Unload();
}

// Perf baseline (F-11 Round 25): decode throughput on real weights. SKIPs
// without the converted vault model. Mirrors test_blackwell_perf_deepseek.
TEST_CASE("Kimi perf baseline: decode throughput", "[kimi][model][perf][real-model]")
{
    namespace fs = std::filesystem;
    const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    fs::path model = env ? fs::path(env)
                         : manifest_query::ConvertedDirFor("kimi-k2.7-code");
    if (!fs::exists(model / "pinned"))
        SKIP(std::string("converted Kimi model directory not present: ") + model.string());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    REQUIRE(lsxcommon::arrow_tokenizer_encode(m->Tokenizer(), prompt,
                                              std::strlen(prompt),
                                              ids.data(), &cap) > 0);
    ids.resize(cap);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    int gen = 16;
    if (const char *p = std::getenv("LSX_TEST_KIMI_GEN_TOKENS"))
        gen = std::atoi(p);
    if (gen < 1) gen = 1;
    req.max_gen_tokens = gen;

    auto r = m->Infer(req);
    if (!r.ok) FAIL("Infer failed: " + r.error_msg);
    REQUIRE(r.ok);

    lsxcommon::log::info(
        "[PERF][kimi-k2.7-code] tok/s=" +
        std::to_string(r.tokens_per_sec) + " ttft_ms=" +
        std::to_string(r.ttft_ms) + " total_ms=" +
        std::to_string(r.total_time_ms) + " gen_tokens=" +
        std::to_string(r.gen_tokens.size()));

    m->Unload();
}
