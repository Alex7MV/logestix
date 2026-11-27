// Muse-Glimmer-30B model test: Load + the frozen-geometry gate on a synthetic
// converted fixture. The factory/Infer/streaming cases land with the
// generation pipeline (the manifest row registers the factory key).
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_muse_dense_fixture.h"

#include "lsxcommon/muse_dense_convert.h"
#include "muse_dense_model.h"
#include "lsxcommon/token_stream.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeConvertedSynthetic()
{
    auto tmp = fs::temp_directory_path() / "muse_dense_model_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    synthtest::WriteSafetensors(tmp, "model.safetensors",
                                muse_dense_test::BuildSyntheticMuseDenseModel());
    muse_dense_test::WriteConfig(tmp);
    muse_dense_test::WriteTokenizerJson(tmp);
    auto out = tmp / "out";
    const auto st = lsxcommon::muse_dense::ConvertMuseDenseModel(
        tmp.string(), out.string());
    if (!st.ok()) FAIL("ConvertMuseDenseModel: " << st.ToString());
    return out;
}

} // namespace

TEST_CASE("muse dense model loads a synthetic converted tree",
          "[muse_dense][model]") {
    // Catch2 randomises case order: each case sets the env it needs instead
    // of inheriting it from a neighbour.
    ::setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    auto out = MakeConvertedSynthetic();

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "muse-glimmer-30b";
    lsxcommon::MuseGlimmer30BModel model(cfg);
    REQUIRE(model.Type() == "muse-glimmer-30b");
    REQUIRE(model.Load());
    CHECK(model.ResidentWeightBytes() > 0);

    // Idempotent re-load: Unload frees everything, a second Load works.
    model.Unload();
    CHECK(model.ResidentWeightBytes() == 0);
    REQUIRE(model.Load());
    model.Unload();
    fs::remove_all(out.parent_path());
}

TEST_CASE("muse dense model refuses a frozen-geometry mismatch",
          "[muse_dense][model]") {
    // The synthetic tree is 4 layers / hidden 64; with the synthetic bypass
    // OFF the frozen passport (52 / 6656) must refuse it.
    auto out = MakeConvertedSynthetic();
    ::unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "muse-glimmer-30b";
    lsxcommon::MuseGlimmer30BModel model(cfg);
    CHECK_FALSE(model.Load());
    CHECK(model.ResidentWeightBytes() == 0);
    fs::remove_all(out.parent_path());
}

TEST_CASE("muse dense model refuses a wrong layer pattern",
          "[muse_dense][model]") {
    ::setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    auto out = MakeConvertedSynthetic();
    // Flip layer 3 (frozen: full) to sliding in the index: the pattern gate
    // must refuse even though every scalar geometry field still matches.
    const auto idx = out / "model.index.json";
    std::ifstream f(idx);
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    f.close();
    const std::string from =
        "\"sliding_attention\",\"sliding_attention\",\"sliding_attention\","
        "\"full_attention\"";
    const std::string to =
        "\"sliding_attention\",\"sliding_attention\",\"sliding_attention\","
        "\"sliding_attention\"";
    REQUIRE(json.find(from) != std::string::npos);
    json.replace(json.find(from), from.size(), to);
    std::ofstream g(idx);
    g << json;
    g.close();

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "muse-glimmer-30b";
    lsxcommon::MuseGlimmer30BModel model(cfg);
    CHECK_FALSE(model.Load());
    fs::remove_all(out.parent_path());
}
TEST_CASE("muse dense model Infer generates deterministically + streams",
          "[muse_dense][model]") {
    ::setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    auto out = MakeConvertedSynthetic();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "muse-glimmer-30b";
    lsxcommon::MuseGlimmer30BModel model(cfg);
    REQUIRE(model.Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {259, 5, 10, 20};
    req.max_gen_tokens = 3;
    auto r = model.Infer(req);
    if (!r.ok) FAIL("Infer failed: " << r.error_msg);
    CHECK(r.gen_tokens.size() >= 1);
    CHECK(r.gen_tokens.size() <= 3);
    CHECK_FALSE(r.output_text.empty());
    CHECK(r.weight_resident_bytes > 0);

    auto r2 = model.Infer(req);
    REQUIRE(r2.ok);
    CHECK(r2.gen_tokens == r.gen_tokens);
    REQUIRE(r.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r.final_logits.data(), r2.final_logits.data(),
                      r.final_logits.size() * sizeof(float)) == 0);

    // Streaming: one Emit per decoded token, in order.
    std::vector<uint32_t> emitted;
    lsxcommon::TokenCallback cb = [&](const lsxcommon::TokenChunk &c) {
        emitted.push_back(c.token_id);
    };
    lsxcommon::ModelRequest sreq = req;
    sreq.token_callback = cb;
    auto sr = model.Infer(sreq);
    REQUIRE(sr.ok);
    REQUIRE(emitted.size() == sr.gen_tokens.size());
    for (size_t i = 0; i < emitted.size(); ++i)
        CHECK(emitted[i] == static_cast<uint32_t>(sr.gen_tokens[i]));

    // Cooperative cancel: a pre-set abort stops after the first token.
    std::atomic<bool> abort{true};
    lsxcommon::ModelRequest areq = req;
    areq.max_gen_tokens = 8;
    areq.abort_flag = &abort;
    auto ar = model.Infer(areq);
    CHECK(ar.gen_tokens.size() <= 1);

    model.Unload();
    fs::remove_all(out.parent_path());
}
