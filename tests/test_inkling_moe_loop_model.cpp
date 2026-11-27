// Inkling-2-Mini model-stage test (Model Ingestion Protocol stage 5).
//
// Runs the WHOLE engine — the three-pass Forward, the routed-expert store, the
// per-pass KV/conv session and the sampler — on the synthetic converted tree
// (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1), because the real 18 GiB tree cannot be
// exercised from a unit test.
//
// What this pins is not the numbers (a synthetic checkpoint's weights are a
// pattern, so there is no source to compare against) but the things that break
// a decode engine structurally:
//   * Load() succeeds, the frozen passport cross-check runs, and a geometry
//     outside the passport is REJECTED (proven on the topology check itself,
//     because the synthetic fixture deliberately has non-frozen geometry);
//   * the model produces tokens at all, in range, and the registration resolves
//     through LsxModelFactory;
//   * two identical requests are BIT-IDENTICAL (same token ids and the same
//     final-logits row) — the three-pass scheduler restarts every extra pass
//     from the same snapshots and owns per-pass state, so a leak shows up here;
//   * a different prompt generally produces different logits, so the determinism
//     above is not the trivial "always emits the same token" kind;
//   * aborting before the first step produces zero tokens and no error;
//   * the streaming callback sees exactly the generated tokens, in order;
//   * a request longer than the engine context bound is refused, not truncated.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_loop_convert.h>
#include "inkling_moe_loop_constants.h"

#include <simdjson.h>
#include "inkling_moe_loop_model.h"
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "test_inkling_moe_loop_fixture.h"
#include "test_streaming_assert.h"

namespace fs = std::filesystem;
using namespace inklinglooptest;

namespace {

// The manifest row (and with it the generated registration) cannot land before
// every lifecycle stage target exists, so this test registers the class itself.
void RegisterInklingLoopOnce()
{
    static bool once = [] {
        lsxcommon::LsxModelFactory::Register<lsxcommon::InklingMoeLoopModel>();
        return true;
    }();
    (void)once;
}

// The converted fixture is cached in the temp dir (converting costs seconds and
// every case wants it) and validated against THIS binary's compiled format tag,
// so a tree left behind by an older converter is rebuilt instead of surfacing
// as a bogus Load() failure.
bool CachedFixtureIsCurrent(const fs::path &out)
{
    const fs::path index = out / "model.index.json";
    std::error_code ec;
    if (!fs::exists(index, ec) || ec) return false;
    std::ifstream f(index);
    if (!f) return false;
    const std::string text((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(text).get(doc) != simdjson::SUCCESS) return false;
    std::string_view format;
    if (doc["format"].get_string().get(format) != simdjson::SUCCESS)
        return false;
    return format == lsxcommon::inkling::loop::frozen::kFormatTag;
}

fs::path EnsureSyntheticModel()
{
    const fs::path in = fs::temp_directory_path() / "inkling_loop_model_in_v1";
    const fs::path out = fs::temp_directory_path() / "inkling_loop_model_out_v1";
    if (CachedFixtureIsCurrent(out)) return out;
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in, /*sane=*/true);
    auto st = lsxcommon::inkling::ConvertInklingLoopModel(in.string(), out.string());
    REQUIRE(st.ok());
    return out;
}

lsxcommon::ModelInitConfig Cfg(const fs::path &root)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = lsxcommon::InklingMoeLoopModel::kType;
    return cfg;
}

// A prompt of well-spread ids inside the real-token band.
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

TEST_CASE("inkling loop model decodes deterministically on the synthetic tree",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    RegisterInklingLoopOnce();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "inkling-2-mini");
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
    REQUIRE(static_cast<int>(first.gen_tokens.size()) == 4);
    REQUIRE(first.first_gen_tok_id == first.gen_tokens.front());
    REQUIRE(first.num_layers == kLayers);
    REQUIRE(first.weight_resident_bytes > 0);
    REQUIRE(first.final_logits.size() == static_cast<size_t>(kUnpadded));
    for (int32_t id : first.gen_tokens) REQUIRE((id >= 0 && id < kUnpadded));
    for (float v : first.final_logits) REQUIRE(std::isfinite(v));
    // The logit softcap bounds every recorded logit.
    for (float v : first.final_logits) REQUIRE(std::fabs(v) <= 30.0f + 1e-3f);

    // ── determinism: a second identical request is bit-identical ──────────
    // Every extra pass restarts from the pre-loop snapshots and writes its own
    // KV/conv slots; a pass that leaked the previous request's state (or the
    // previous pass's) lands here.
    lsxcommon::ModelRequest req2 = req;
    const auto second = model->Infer(req2);
    REQUIRE(second.ok);
    REQUIRE(second.gen_tokens == first.gen_tokens);
    REQUIRE(SameFloatRow(second.final_logits, first.final_logits));

    // ── the determinism above is not "one fixed answer for every prompt" ──
    lsxcommon::ModelRequest other = req;
    other.input_ids = Prompt(6, 9);
    const auto alt = model->Infer(other);
    REQUIRE(alt.ok);
    REQUIRE(alt.gen_tokens.size() == 4);
    CHECK_FALSE(SameFloatRow(alt.final_logits, first.final_logits));

    model->Unload();
}

TEST_CASE("inkling loop model streams and honours a pre-set abort",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    RegisterInklingLoopOnce();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest base;
    base.input_ids = Prompt(5, 2);
    base.max_gen_tokens = 3;
    base.ignore_eos = true;

    SECTION("the callback sees exactly the generated tokens, in order")
    {
        ExpectStreamMatchesTokens(*model, base);
    }

    SECTION("an abort flag set before the first step yields no tokens")
    {
        lsxcommon::ModelRequest req = base;
        req.max_gen_tokens = 8;
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

TEST_CASE("inkling loop model refuses a request longer than the engine context",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    RegisterInklingLoopOnce();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = Prompt(8, 3);
    req.max_gen_tokens = lsxcommon::inkling::kLoopMaxContext;  // prompt + gen
    req.ignore_eos = true;
    const auto r = model->Infer(req);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.error_msg.empty());

    model->Unload();
}

TEST_CASE("inkling loop geometry outside the frozen passport is rejected",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    lsxcommon::inkling::InklingLoopTopology top;
    REQUIRE(lsxcommon::inkling::LoadInklingLoopTopology(root.string(), top).ok());
    // The fixture is deliberately NOT the frozen geometry.
    REQUIRE(top.num_layers == kLayers);
    REQUIRE(top.hidden == kHidden);
    REQUIRE(top.hidden != lsxcommon::inkling::loop::frozen::kHidden);

    // With the synthetic-geometry gate CLOSED, the passport check must reject
    // exactly what Load() relies on for the real vault. (The gate is an
    // env-var read at call time, so a test can close it for one case.)
    unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    const bool rejected =
        !lsxcommon::inkling::InklingLoopTopologyMatchesFrozen(top);
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    CHECK(rejected);

    // ... and the same tree under the gate is accepted (so the case above is
    // not vacuous: the check runs and says no, rather than always saying no).
    CHECK(lsxcommon::inkling::InklingLoopTopologyMatchesFrozen(top));
}
