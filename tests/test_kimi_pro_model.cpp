// Kimi-K3.1-Pro model/pipeline stage: the shared Kimi-K3 generation driver on a
// synthetic Kimi-K3.1-Pro tree, with the Gated-DSA indexer branch.
//
// Written in two stages. The first (Task 5) drives `RunKimiK3Generation`
// directly, so the DSA branch is testable before the IModel exists: the branch
// must stay DORMANT (bit-identical to the dense Kimi-K3 path) while
// `n_pos <= index_topk`, and above it every full-attention layer must compute
// the indexer key of the new position, score every causal position, and hand
// the exact top-k to the attention. The second (Task 6) adds the
// IModel/registry/lifecycle cases on top of the same fixture.
//
// The fixture's synthetic weights make the converted model's LOGITS invariant
// to its input (a pre-existing property of the kimi-k3 fixture, not of this
// variant: the dense Kimi-K3 tree answers two different prompts with bit-equal
// logits). So "the sparse answer differs from the dense answer" is not a
// measurable claim here and is deliberately not asserted. What is asserted is
// the part that can be: the exact step/key counts, the returned score vector
// and selection checked against the test's own top-k over those scores, the
// dormant path's bit-exactness against the dense tree, and determinism.
//
// `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` is set by this target's ctest ENVIRONMENT
// property and enforced inside the converter; the test never reads it.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "kimi_moe_k3_constants.h"
#include "kimi_moe_pro_constants.h"
#include "kimi_moe_k3_model.h"
#include "kimi_moe_pro_model.h"
#include "lsxcommon/kimi_moe_k3_convert.h"
#include "lsxcommon/kimi_moe_k3_inference_pipeline.h"
#include "lsxcommon/kimi_moe_k3_load.h"
#include "lsxcommon/kimi_moe_pro_convert.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/sampling.h"
#include "test_kimi_pro_fixture.h"
#include "test_streaming_assert.h"

namespace fs = std::filesystem;
using namespace lsxcommon;

namespace {

bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Builds (and converts) a synthetic tree. `kPro` writes the DSA variant, `k3`
// the plain Kimi-K3 tree; both come from the SAME base tensor generator, so the
// dense half of the two converted trees is byte-identical.
fs::path BuildTree(const std::string &tag, kimiprotest::Variant v)
{
    const fs::path in = fs::temp_directory_path() / ("kimi_pro_model_in_" + tag);
    const fs::path out =
        fs::temp_directory_path() / ("kimi_pro_model_out_" + tag);
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);
    kimiprotest::WriteCheckpoint(in, v);
    const auto st = (v == kimiprotest::Variant::k3)
                        ? kimi_k3::ConvertKimiK3Model(in.string(), out.string())
                        : kimi_pro::ConvertKimiK31ProModel(in.string(),
                                                           out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    return out;
}

// One loaded session: the pinned graph plus the routed-expert store. Nothing
// here is model-owned, so the pipeline can be driven before the IModel exists.
struct Session {
    kimi_k3::KimiK3PinnedWeights pw;
    kimi_k3::KimiK3ExpertDb db;
    bool pw_loaded = false;
    bool db_loaded = false;

    ~Session()
    {
        if (db_loaded) kimi_k3::FreeKimiK3ExpertIndex(db);
        if (pw_loaded) kimi_k3::FreeKimiK3PinnedWeights(pw);
    }
};

bool OpenSession(const fs::path &root, Session &s)
{
    if (!kimi_k3::LoadKimiK3PinnedWeights(root.string(), s.pw, nullptr))
        return false;
    s.pw_loaded = true;
    s.db = kimi_k3::LoadKimiK3ExpertIndex(
        root.string(), s.pw.top.num_layers, s.pw.top.num_experts,
        s.pw.top.dense_layers, /*max_open=*/64);
    s.db_loaded = true;
    return true;
}

kimi_k3::KimiK3GenerationResult Generate(Session &s,
                                         const std::vector<int32_t> &prompt,
                                         int max_new, int max_context = 64)
{
    sample::GreedySampler sampler;
    kimi_k3::KimiK3GenerationResult g;
    kimi_k3::RunKimiK3Generation(nullptr, s.pw, s.db, prompt, max_new,
                                 /*eos_id=*/-1, max_context, sampler, g,
                                 /*gs=*/nullptr);
    return g;
}

// The forward count of one run: every prompt token, then one more per generated
// token except the last (the last one only samples).
int ForwardCount(size_t prompt_tokens, int max_new)
{
    return static_cast<int>(prompt_tokens) + std::max(0, max_new - 1);
}

}  // namespace

// ───────────────────────── Task 5: the DSA branch ─────────────────────────

TEST_CASE("kimi-k3.1-pro pipeline: the DSA branch is dormant at n_pos <= topk",
          "[kimi-pro][pipeline]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path pro = BuildTree("dormant_pro", kimiprotest::Variant::kPro);
    const fs::path k3 = BuildTree("dormant_k3", kimiprotest::Variant::k3);

    Session a;
    Session b;
    REQUIRE(OpenSession(pro, a));
    REQUIRE(OpenSession(k3, b));
    REQUIRE(a.pw.top.dsa);
    REQUIRE_FALSE(b.pw.top.dsa);
    REQUIRE(a.pw.top.index_topk == kimiprotest::kIndexTopK);

    // The fixture's topk is 4 and this run never presents more than 4 keys
    // (3 prompt tokens + 2 generated => forward positions 0..3), so every DSA
    // layer must take the dense path: the run must be the Kimi-K3 run, token
    // for token and bit for bit (same base tensors, same converter).
    const std::vector<int32_t> prompt = {3, 5, 7};
    REQUIRE(ForwardCount(prompt.size(), 2) == kimiprotest::kIndexTopK);
    const auto pro_out = Generate(a, prompt, /*max_new=*/2);
    const auto k3_out = Generate(b, prompt, /*max_new=*/2);
    if (!pro_out.ok) FAIL(pro_out.error_msg);
    REQUIRE(pro_out.ok);
    REQUIRE(k3_out.ok);
    CHECK(pro_out.dsa_steps == 0);
    CHECK(pro_out.dsa_keys_selected == 0);
    CHECK(pro_out.dsa_last_selection.empty());
    CHECK(pro_out.dsa_last_score.empty());
    REQUIRE(pro_out.tokens == k3_out.tokens);
    REQUIRE(pro_out.final_logits.size() == k3_out.final_logits.size());
    for (size_t i = 0; i < pro_out.final_logits.size(); ++i)
        REQUIRE(pro_out.final_logits[i] == k3_out.final_logits[i]);
}

TEST_CASE("kimi-k3.1-pro pipeline: the DSA branch selects above the top-k",
          "[kimi-pro][pipeline]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    const fs::path pro = BuildTree("engage_pro", kimiprotest::Variant::kPro);

    Session a;
    REQUIRE(OpenSession(pro, a));
    const int dsa_layers = static_cast<int>(a.pw.top.dsa_layers.size());
    REQUIRE(dsa_layers == 2);
    const int topk = a.pw.top.index_topk;

    // 8 prompt tokens + 2 generated => forward positions 0..8, so the positions
    // 4..8 exceed topk 4 on every DSA layer: (9 - topk) selections per layer.
    // The counts are exact, so a silently skipped branch cannot pass.
    const std::vector<int32_t> prompt = {3, 5, 7, 9, 11, 13, 15, 17};
    const int forwards = ForwardCount(prompt.size(), 2);
    const int engaged = forwards - topk;
    REQUIRE(engaged == 5);
    const auto out = Generate(a, prompt, /*max_new=*/2);
    if (!out.ok) FAIL(out.error_msg);
    REQUIRE(out.ok);
    CHECK(out.dsa_steps == static_cast<int64_t>(dsa_layers) * engaged);
    CHECK(out.dsa_keys_selected ==
          static_cast<int64_t>(dsa_layers) * engaged * topk);

    // The last engaged step's raw material: its score vector covers every
    // causal position of that forward, and its selection is exactly the top-k
    // of those scores (the pipeline's own data, checked against this test's
    // independent selection). The emission order is not value-sorted -- the
    // kernel emits the strictly-above group first -- so the sets are compared.
    REQUIRE(out.dsa_last_n_pos == forwards);
    REQUIRE(out.dsa_last_score.size() == static_cast<size_t>(forwards));
    REQUIRE(out.dsa_last_selection.size() == static_cast<size_t>(topk));
    std::vector<size_t> order(static_cast<size_t>(forwards));
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        if (out.dsa_last_score[x] != out.dsa_last_score[y])
            return out.dsa_last_score[x] > out.dsa_last_score[y];
        return x < y;  // boundary ties: the lowest position wins
    });
    std::vector<int32_t> want(order.begin(), order.begin() + topk);
    std::sort(want.begin(), want.end());
    std::vector<int32_t> got = out.dsa_last_selection;
    for (int32_t i : got) {
        CHECK(i >= 0);
        CHECK(i < forwards);
    }
    std::sort(got.begin(), got.end());
    CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());
    CHECK(got == want);

    // Deterministic: the score pass, the radix select and the access order are
    // all fixed, so a second run must be identical.
    const auto again = Generate(a, prompt, /*max_new=*/2);
    REQUIRE(again.ok);
    REQUIRE(again.tokens == out.tokens);
    REQUIRE(again.dsa_last_selection == out.dsa_last_selection);
    REQUIRE(again.final_logits.size() == out.final_logits.size());
    for (size_t i = 0; i < out.final_logits.size(); ++i)
        REQUIRE(out.final_logits[i] == again.final_logits[i]);
}

// ─────────────────────── Task 6: model, registry, lifecycle ───────────────

namespace {

std::unique_ptr<IModel> MakeModel(const fs::path &root)
{
    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kimi_pro::frozen::kModelType;
    cfg.fast_math_mode = FastMathMode::kOff;
    return LsxModelFactory::Create(cfg);
}

// Rewrites one string in the converted index in place, so a hand-broken tree
// can be handed to `Load` without touching the converter.
void PatchIndex(const fs::path &root, const std::string &from,
                const std::string &to)
{
    const fs::path idx = root / "logestix" / "model.index.json";
    std::ifstream in(idx, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    in.close();
    std::string s = ss.str();
    const auto at = s.find(from);
    if (at == std::string::npos) FAIL("index does not contain " << from);
    REQUIRE(at != std::string::npos);
    s.replace(at, from.size(), to);
    std::ofstream out(idx, std::ios::binary);
    out << s;
}

}  // namespace

TEST_CASE("kimi-k3.1-pro model: registry and variant type", "[kimi-pro][model]")
{
    EnsureModelRegistry();
    CHECK(KimiK31ProModel::kType == "kimi-k3.1-pro");
    CHECK(KimiK31ProModel::kType == kimi_pro::frozen::kModelType);
    // The variant must not shadow its parent's registration.
    CHECK(KimiK3Model::kType == "kimi-k3");
}

TEST_CASE("kimi-k3.1-pro model: a Kimi-K3 tree is rejected",
          "[kimi-pro][model][lifecycle]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    EnsureModelRegistry();
    const fs::path k3 = BuildTree("model_k3", kimiprotest::Variant::k3);
    auto model = MakeModel(k3);
    REQUIRE(model != nullptr);
    // Asked to load as the DSA variant, a Kimi-K3 tree must fail: the format
    // tag, the topology's `dsa` flag and the passport all disagree.
    CHECK_FALSE(model->Load());
    model->Unload();
}

TEST_CASE("kimi-k3.1-pro model: an off-passport indexer geometry fails Load",
          "[kimi-pro][model][lifecycle]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    EnsureModelRegistry();
    const fs::path root = BuildTree("model_heads", kimiprotest::Variant::kPro);
    // The indexer's own shape check (the frozen top-k alone does not cover it):
    // a tree claiming a different head count is rejected at load, before any
    // wrong-shaped tensor can reach a kernel.
    PatchIndex(root, "\"index_num_heads\": 2", "\"index_num_heads\": 3");
    auto model = MakeModel(root);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());
    model->Unload();
}

TEST_CASE("kimi-k3.1-pro model: lifecycle", "[kimi-pro][model][lifecycle]")
{
    if (!GpuAvailable()) SKIP("no CUDA device");
    EnsureModelRegistry();

    // An Infer before Load is a clean error, not a crash.
    {
        ModelInitConfig cfg;
        cfg.model_type = kimi_pro::frozen::kModelType;
        cfg.fast_math_mode = FastMathMode::kOff;
        auto unloaded = LsxModelFactory::Create(cfg);
        REQUIRE(unloaded != nullptr);
        ModelRequest req;
        req.input_ids = {3};
        req.max_gen_tokens = 1;
        const auto r = unloaded->Infer(req);
        CHECK_FALSE(r.ok);
        CHECK_FALSE(r.error_msg.empty());
    }

    const fs::path root = BuildTree("model_life", kimiprotest::Variant::kPro);
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
    REQUIRE(r.final_logits.size() == static_cast<size_t>(kimik3test::kVocab));
    bool any_nonzero = false;
    for (float v : r.final_logits) {
        REQUIRE(std::isfinite(v));
        if (v != 0.0f) any_nonzero = true;
    }
    CHECK(any_nonzero);
    CHECK(r.weight_resident_bytes > 0);
    CHECK(r.num_layers == kimik3test::kLayers);

    // Streaming and cancel, the shared contract every family wires.
    ModelRequest stream_req;
    stream_req.input_ids = {5};
    stream_req.max_gen_tokens = 3;
    stream_req.ignore_eos = true;
    ExpectStreamMatchesTokens(*model, stream_req);
    ModelRequest cancel_req;
    cancel_req.input_ids = {5, 7, 9};
    cancel_req.ignore_eos = true;
    ExpectPreSetAbortStops(*model, cancel_req);
    ExpectCallbackAbortStops(*model, cancel_req);
    model->Unload();
}
