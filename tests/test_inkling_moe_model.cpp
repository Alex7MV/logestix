// Inkling-2-Small model-stage test (Model Ingestion Protocol stage 5).
//
// Runs the WHOLE engine — Forward(), the routed-expert store, the KV session and
// the sampler — on the synthetic converted tree (LSX_TEST_ALLOW_SYNTHETIC_
// GEOMETRY=1), because the real 594B tree cannot be exercised from a unit test.
//
// What this pins is not the numbers (a synthetic checkpoint's weights are a byte
// pattern, so there is no source to compare against) but the things that break a
// decode engine structurally:
//   * Load() succeeds and the frozen passport cross-check runs;
//   * the model produces tokens at all, in range, and the registration resolves
//     through LsxModelFactory;
//   * two identical requests are BIT-IDENTICAL (same token ids and the same
//     final-logits row), which is the determinism contract every gate downstream
//     depends on;
//   * a different prompt generally produces different logits, so the determinism
//     above is not the trivial "always emits the same token" kind;
//   * aborting before the first step produces zero tokens and no error;
//   * the streaming callback sees exactly the generated tokens, in order.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/index_layout.h>
#include <lsxcommon/inkling_moe_convert.h>
#include "inkling_moe_constants.h"

#include <simdjson.h>

#include "lsxstorage/artifact.h"
#include "inkling_moe_model.h"
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

#include "test_inkling_fixture.h"

namespace fs = std::filesystem;
using namespace inklingtest;

namespace {

// There is no models/<org>/<model>.json row yet (the decode engine is the last stage), so
// the generated registration list does not carry this class. The test registers
// it itself; the factory row arrives with the manifest in the closing commit.
void RegisterInklingOnce()
{
    static bool once = [] {
        lsxcommon::LsxModelFactory::Register<lsxcommon::InklingMoeModel>();
        return true;
    }();
    (void)once;
}

// Builds the synthetic tree once per process. The directory names carry a LAYOUT
// VERSION (`_v2`) on purpose: an artifact left behind by an older converter is
// silently reused otherwise, and every later assertion then reports a defect
// that is really a stale directory. Both hazards have already cost a debugging
// session here (a byte-pattern tree whose values overflowed, and a tree whose
// embed/unembed were still NVFP4). Bump the version whenever the converter's
// output layout changes.
// The converted fixture is cached in the temp dir because converting it costs
// seconds and every case wants it. The cache is only reusable if it was written
// by THIS binary's format: the tag moved once already
// (`logestix-inkling-2-small-nvfp4` -> `logestix-inkling-2-small`) and a stale
// cache then failed as `Load() == false` inside whatever case ran first, which
// reads like a loader defect rather than a fixture from another build. So the
// cached index is validated against the compiled `frozen::kFormatTag` and
// rebuilt on any mismatch or unreadable file.
bool CachedFixtureIsCurrent(const fs::path &out)
{
    const fs::path index = out / "model.index.json";
    std::error_code ec;
    if (!fs::exists(index, ec) || ec) return false;
    std::ifstream f(index);
    if (!f) return false;
    const std::string text((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    // An EXACT value comparison, not a substring search: the tag that replaced
    // this one (`logestix-inkling-2-small-nvfp4`) has the current tag as its
    // prefix, so `find` would accept exactly the stale file this check exists to
    // reject.
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(text).get(doc) != simdjson::SUCCESS) return false;
    // "Is this tree this family's?" was answered by a per-model format tag.
    // That tag is retired (design §2.4.3): the index now declares its schema
    // GENERATION, and the model identity lives in artifact.fb's model_type,
    // checked once in ModelBase::IdentityAllows. What a fixture at this level can
    // still assert is the generation it was written in.
    int64_t layout = 0;
    if (doc["layout"].get_int64().get(layout) != simdjson::SUCCESS)
        return false;
    if (layout != lsxcommon::kIndexLayout) return false;
    // A cache of a GENERATED tree must also be a tree the current contract
    // accepts. The generation tag cannot see a schema change: an expert plane
    // that predates the declared-codec column (Task 9c) still says layout 2 and
    // is still refused by the verifier, so reusing it silently loaded a tree
    // whose per-layer codec could not be established.
    return lsx::storage::ScanArtifactTree(out).ok;
}

fs::path EnsureSyntheticModel()
{
    const fs::path in = fs::temp_directory_path() / "inkling_model_in_v2";
    const fs::path out = fs::temp_directory_path() / "inkling_model_out_v2";
    if (CachedFixtureIsCurrent(out)) return out;
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in, /*sane=*/true);
    auto st = lsxcommon::inkling::ConvertInklingModel(in.string(), out.string());
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
    cfg.model_type = lsxcommon::InklingMoeModel::kType;
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

TEST_CASE("inkling model decodes deterministically on the synthetic tree",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    RegisterInklingOnce();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "inkling-2-small");
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
    for (int32_t id : first.gen_tokens)
        REQUIRE((id >= 0 && id < kUnpadded));
    for (float v : first.final_logits) REQUIRE(std::isfinite(v));
    // The logit softcap bounds every recorded logit.
    for (float v : first.final_logits) REQUIRE(std::fabs(v) <= 30.0f + 1e-3f);

    // ── determinism: a second identical request is bit-identical ──────────
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

TEST_CASE("inkling model streams and honours a pre-set abort", "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    RegisterInklingOnce();
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
        REQUIRE(steps.size() == seen.size());
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

TEST_CASE("inkling model refuses a request longer than the engine context",
          "[inkling][model]")
{
    if (std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") == nullptr)
        SKIP("synthetic-geometry fixture not enabled");
    const fs::path root = EnsureSyntheticModel();

    RegisterInklingOnce();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = Prompt(8, 3);
    req.max_gen_tokens = lsxcommon::inkling::kMaxContext;  // prompt + gen > bound
    req.ignore_eos = true;
    const auto r = model->Infer(req);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.error_msg.empty());

    model->Unload();
}
