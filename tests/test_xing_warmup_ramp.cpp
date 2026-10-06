// Warm-up ramp: does decode throughput RISE during a session, and if so how much?
//
// Why ask. Every piece of context-scaling work in this repo so far assumed decode
// gets SLOWER as the KV cache grows - the KV-layout change, the tiling target, the
// kAttnSplits curve all came from that model. Measured end to end on Xing-4.0 that
// is true (127.1 t/s at max-gen 2048, 77.3 at 8191), but an operator's separate
// observation was that throughput sometimes GROWS. Those two cannot both describe
// the same run, and the difference decides what is worth optimising:
//
//   - if the growth is real, it is a warm-up ramp (lazy module load, CUDA graph
//     warm, GPU clock ramp) and the lever is a FIXED cost, not context scaling;
//   - if it is not real, the context-scaling model stands and this test is the
//     record that says so.
//
// So it measures per-token wall time from the token callback and reports the shape:
// the mean t/s of the first and last quarters, and the same split at a longer
// horizon. The callback is timestamped on the HOST, which is the right clock here -
// the question is when the engine hands a token back, not how long a kernel took.
//
// This is a MEASUREMENT, not a gate: nothing asserts a throughput claim, because a
// warm-up shape depends on the box (clock policy, driver, power state) and a budget
// taken from one of them would flake on another. What it does assert is that the
// timings are plausible and monotonic in aggregate - a callback that never fires, or
// a session that produced nothing, would otherwise read as "no ramp found".
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"

namespace
{
using Clock = std::chrono::steady_clock;
using Ms = double;

std::string Root()
{
    if (const char *env = std::getenv("LSX_TEST_XING_ROOT")) return env;
    return "/mnt/llm_models/models/Xing4.0-29B-A4B";
}

struct Ramp
{
    std::vector<Ms> dt;          // per-token wall time, index 0 unused
    Ms first_quarter_tps = 0.0;  // 1/token, from the first quarter
    Ms last_quarter_tps = 0.0;
    Ms first_tenth_tps = 0.0;    // where a clock ramp would be most visible
    Ms median_tps = 0.0;
};

// Mean tokens/sec over [lo, hi) of the generated stream.
Ms Tps(const std::vector<Ms> &dt, int lo, int hi)
{
    double sum = 0.0;
    int n = 0;
    for (int i = std::max(1, lo); i < hi && i < static_cast<int>(dt.size()); ++i)
    {
        if (dt[static_cast<size_t>(i)] > 0.0)
        {
            sum += dt[static_cast<size_t>(i)];
            ++n;
        }
    }
    return n > 0 ? 1000.0 * n / sum : 0.0;
}

Ramp Measure(int max_gen)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = Root();
    cfg.model_type = "xing4.0-29b-a4b";
    // NO numeric-class pin: the ramp being asked about is the one a user sees, which
    // is the default per-GPU policy (~127 t/s here). Pinning kOff measured the
    // legacy bit-exact path instead - 20 t/s - and would have described a ramp that
    // never happens in production.
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(
        tok, "The capital of France is", ids));
    REQUIRE(!ids.empty());

    Ramp r;
    r.dt.assign(static_cast<size_t>(max_gen) + 1, 0.0);
    Clock::time_point prev = Clock::now();
    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = max_gen;
    // Exactly max_gen steps, so the horizon is the one asked for regardless of
    // where EOS would otherwise land.
    req.ignore_eos = true;
    req.token_callback = [&r, &prev](const lsxcommon::TokenChunk &c) {
        const Clock::time_point now = Clock::now();
        const int step = static_cast<int>(c.step_t);
        if (step >= 0 && step < static_cast<int>(r.dt.size()))
            r.dt[static_cast<size_t>(step)] =
                std::chrono::duration<Ms, std::milli>(now - prev).count();
        prev = now;
    };
    auto res = model->Infer(req);
    INFO("infer error: " << res.error_msg);
    REQUIRE(res.ok);

    const int q = max_gen / 4;
    r.first_quarter_tps = Tps(r.dt, 1, 1 + q);
    r.last_quarter_tps = Tps(r.dt, max_gen - q, max_gen);
    r.first_tenth_tps = Tps(r.dt, 1, 1 + std::max(1, max_gen / 10));
    std::vector<Ms> all;
    for (int i = 1; i <= max_gen; ++i)
        if (r.dt[static_cast<size_t>(i)] > 0.0) all.push_back(r.dt[static_cast<size_t>(i)]);
    if (!all.empty())
    {
        std::sort(all.begin(), all.end());
        r.median_tps = 1000.0 / all[all.size() / 2];
    }
    return r;
}

std::string Row(const char *label, int max_gen, const Ramp &r)
{
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "%-22s max_gen=%-5d first10%%=%6.1f  Q1=%6.1f  median=%6.1f  "
                  "Q4=%6.1f  t/s   (Q4/Q1 = %.2fx)",
                  label, max_gen, r.first_tenth_tps, r.first_quarter_tps,
                  r.median_tps, r.last_quarter_tps,
                  r.first_quarter_tps > 0.0 ? r.last_quarter_tps / r.first_quarter_tps
                                           : 0.0);
    return buf;
}
}  // namespace

TEST_CASE("Xing decode throughput ramp over a session", "[perf][warmup]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("Xing4.0-29B-A4B converted model not present");
    lsxcommon::EnsureModelRegistry();

    // ONE horizon per PROCESS, selected by LSX_RAMP_MAX_GEN (default 2048: the
    // canonical protocol). Two resident Xing instances need ~30 GiB between them
    // and the second expert-slab allocation dies of OOM on a 32 GiB card - and not
    // only across SECTIONs: destroying the first model does NOT return the memory
    // within the process, so this has to be a separate invocation. Run it twice:
    //   LSX_RAMP_MAX_GEN=2048 ./test_xing_warmup_ramp
    //   LSX_RAMP_MAX_GEN=4096 ./test_xing_warmup_ramp
    int max_gen = 2048;
    if (const char *env = std::getenv("LSX_RAMP_MAX_GEN"))
        max_gen = std::atoi(env);
    REQUIRE(max_gen >= 64);

    const Ramp r = Measure(max_gen);
    int timed = 0;
    for (size_t i = 1; i < r.dt.size(); ++i)
        if (r.dt[i] > 0.0) ++timed;
    // The callback must have fired for essentially every step, or "no ramp found"
    // would just mean "no measurement".
    REQUIRE(timed > max_gen * 9 / 10);
    REQUIRE(r.last_quarter_tps > 0.0);
    WARN("Xing decode throughput ramp (host-timestamped token callback):\n"
         << Row("xing", max_gen, r));
}