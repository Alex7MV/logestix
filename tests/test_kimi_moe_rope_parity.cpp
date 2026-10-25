// Pins the MLA RoPE and attention temperature to the checkpoint's own
// reference (modeling_deepseek.py). The fixture is captured by
// tools/kimi_rope_capture.py: the vendor de-interleaves with
// view(d/2, 2).transpose(4, 3).reshape(...) before rotate_half, so the
// effective rotation is on ADJACENT pairs (2p, 2p+1) and the attention
// temperature carries yarn_get_mscale(64, 1)**2. The fixture is NOT produced
// by these helpers, so a regression cannot be self-consistent.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "lsxcommon/kimi_moe_load.h"

namespace {

struct Fixture {
    int rope_dim = 0;
    int positions = 0;
    std::vector<float> inv_freq;
    std::vector<float> k_raw, k_roped;
    std::map<int, std::vector<float>> q_raw, q_roped;
    std::map<int, float> dot;
};

float NextWord(FILE *f)
{
    char buf[64];
    REQUIRE(std::fscanf(f, "%63s", buf) == 1);
    return std::strtof(buf, nullptr);
}

Fixture LoadFixture()
{
    FILE *f = std::fopen(LSX_KIMI_FIXTURE_DIR "/rope_attn.txt", "rb");
    REQUIRE(f != nullptr);
    Fixture x;
    char key[64];
    while (std::fscanf(f, "%63s", key) == 1) {
        const std::string k(key);
        if (k == "rope_dim") {
            x.rope_dim = static_cast<int>(NextWord(f));
        } else if (k == "positions") {
            x.positions = static_cast<int>(NextWord(f));
        } else if (k == "inv_freq") {
            x.inv_freq.resize(x.rope_dim / 2);
            for (auto &v : x.inv_freq) v = NextWord(f);
        } else if (k == "k_raw" || k == "k_roped") {
            // k is captured at position 0 (rotated by zero -> identity) and
            // feeds the cross-position score dots below.
            std::vector<float> v(x.rope_dim);
            for (auto &e : v) e = NextWord(f);
            if (k == "k_raw") x.k_raw = std::move(v);
            else x.k_roped = std::move(v);
        } else if (k == "q_raw" || k == "q_roped") {
            const int p = static_cast<int>(NextWord(f));
            std::vector<float> v(x.rope_dim);
            for (auto &e : v) e = NextWord(f);
            if (k == "q_raw") x.q_raw[p] = std::move(v);
            else x.q_roped[p] = std::move(v);
        } else if (k == "dot") {
            const int p = static_cast<int>(NextWord(f));
            x.dot[p] = NextWord(f);
        } else {
            FAIL("unexpected fixture key " << k);
        }
    }
    std::fclose(f);
    REQUIRE(x.rope_dim == 64);
    REQUIRE(x.positions == 9);
    REQUIRE(x.inv_freq.size() == 32);
    REQUIRE(x.k_raw.size() == 64);
    REQUIRE(x.k_roped.size() == 64);
    return x;
}

} // namespace

TEST_CASE("kimi MLA rope matches the vendored interleaved implementation",
          "[kimi][rope][fast]")
{
    const Fixture fx = LoadFixture();
    const std::vector<float> inv = lsxcommon::kimi::KimiYarnInvFreq();
    REQUIRE(inv.size() == fx.inv_freq.size());
    for (size_t i = 0; i < inv.size(); ++i) {
        INFO("inv_freq[" << i << "] engine=" << inv[i]
                         << " ref=" << fx.inv_freq[i]);
        CHECK(std::fabs(inv[i] - fx.inv_freq[i]) <=
              1e-6f * std::fabs(fx.inv_freq[i]) + 1e-12f);
    }

    // k: position 0 -> the reference rotation is the identity.
    std::vector<float> k = fx.k_raw;
    for (int pair = 0; pair < fx.rope_dim / 2; ++pair)
        lsxcommon::kimi::KimiRopePair(pair, 0.0f, k.data());
    for (int i = 0; i < fx.rope_dim; ++i) {
        INFO("k dim " << i << " engine=" << k[i]
                      << " ref=" << fx.k_roped[i]);
        CHECK(std::fabs(k[i] - fx.k_roped[i]) <= 1e-3f);
    }

    for (int p = 0; p < fx.positions; ++p) {
        std::vector<float> q = fx.q_raw.at(p);
        for (int pair = 0; pair < fx.rope_dim / 2; ++pair)
            lsxcommon::kimi::KimiRopePair(
                pair, inv[pair] * static_cast<float>(p), q.data());
        double dot = 0.0;
        for (int i = 0; i < fx.rope_dim; ++i) {
            INFO("pos " << p << " dim " << i << " engine=" << q[i]
                        << " ref=" << fx.q_roped.at(p)[i]);
            CHECK(std::fabs(q[i] - fx.q_roped.at(p)[i]) <= 1e-3f);
            dot += static_cast<double>(q[i]) * k[i];
        }
        INFO("pos " << p << " dot engine=" << dot << " ref=" << fx.dot.at(p));
        CHECK(std::fabs(dot - fx.dot.at(p)) <= 1e-3);
    }
}

TEST_CASE("kimi attention temperature includes the YaRN mscale squared",
          "[kimi][rope][fast]")
{
    const double mscale = 0.1 * std::log(64.0) + 1.0;
    const double ref = std::pow(192.0, -0.5) * mscale * mscale;
    INFO("engine=" << lsxcommon::kimi::KimiAttnScale() << " ref=" << ref);
    CHECK(std::fabs(lsxcommon::kimi::KimiAttnScale() - ref) <= 1e-7);
    CHECK(lsxcommon::kimi::KimiAttnScale() > 1.9 * std::pow(192.0, -0.5));
}
