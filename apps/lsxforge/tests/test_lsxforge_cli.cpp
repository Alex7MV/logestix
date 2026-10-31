// lsxforge pipeline dispatch (host-only; no CUDA, no model, no conversion).
//
// main() of lsxforge is a 490-line chain of `if (flag) { convert; return; }`
// branches — one per ingested family plus the tokenizer / repack / vram-recompute
// / generation side paths. Before this suite only Core::Initialize/Shutdown was
// covered, so a reordered or dropped flag silently changed which converter ran.
//
// ResolvePipeline() is the extracted decision: flags in, one Pipeline out. This
// suite pins the precedence and the ambiguity contract; main() keeps only the
// execution of the resolved pipeline.
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "cli.h"

using lsxforge::Flags;
using lsxforge::Pipeline;
using lsxforge::ResolvePipeline;

namespace {

Flags Base()
{
    Flags f;
    f.input = "/src/model";
    f.output = "/dst/model";
    return f;
}

} // namespace

TEST_CASE("forge cli: an empty input selects the version banner",
          "[lsxforge][cli]")
{
    Flags f;  // no --input
    CHECK(ResolvePipeline(f).pipeline == Pipeline::kVersion);
}

TEST_CASE("forge cli: --run selects generation before any converter",
          "[lsxforge][cli]")
{
    Flags f = Base();
    f.run = true;
    f.glm = true;  // generation must win: it bypasses the conversion dispatch
    CHECK(ResolvePipeline(f).pipeline == Pipeline::kGenerate);
}

TEST_CASE("forge cli: --tokenizer selects the tokenizer bridge", "[lsxforge][cli]")
{
    Flags f;
    f.tokenizer = "/src/tok";
    CHECK(ResolvePipeline(f).pipeline == Pipeline::kTokenizer);
}

TEST_CASE("forge cli: every family flag maps to its own pipeline",
          "[lsxforge][cli]")
{
    const std::pair<const char *, Pipeline> kCases[] = {
        {"glm", Pipeline::kGlm},
        {"kimi", Pipeline::kKimi},
        {"qwen", Pipeline::kQwenDense},
        {"qwen35moe", Pipeline::kQwen35Moe},
        {"qwen36moe", Pipeline::kQwen36Moe},
        {"ornith", Pipeline::kOrnith},
        {"qwen38flashnext", Pipeline::kQwen38FlashNext},
        {"gemma4", Pipeline::kGemma4},
        {"gemma4dense", Pipeline::kGemma4Dense},
        {"nemotron", Pipeline::kNemotron},
        {"nemotron_nvfp4", Pipeline::kNemotronNvfp4},
        {"gigachat", Pipeline::kGigachat},
        {"glmflash", Pipeline::kGlmFlash},
        {"granite", Pipeline::kGranite},
        {"phi", Pipeline::kPhi},
        {"mimo", Pipeline::kMimo},
        {"alice", Pipeline::kAlice},
        {"mistral", Pipeline::kMistral},
        {"minimax", Pipeline::kMinimax},
        {"minimax_flash", Pipeline::kMinimaxFlash},
        {"hy", Pipeline::kHy},
        {"hy_simulmt", Pipeline::kHySimulMT},
    };
    for (const auto &[name, expected] : kCases) {
        CAPTURE(name);
        Flags f = Base();
        if (std::string(name) == "glm") f.glm = true;
        else if (std::string(name) == "kimi") f.kimi = true;
        else if (std::string(name) == "qwen") f.qwen = true;
        else if (std::string(name) == "qwen35moe") f.qwen35moe = true;
        else if (std::string(name) == "qwen36moe") f.qwen36moe = true;
        else if (std::string(name) == "ornith") f.ornith = true;
        else if (std::string(name) == "qwen38flashnext") f.qwen38flashnext = true;
        else if (std::string(name) == "gemma4") f.gemma4 = true;
        else if (std::string(name) == "gemma4dense") f.gemma4dense = true;
        else if (std::string(name) == "nemotron") f.nemotron = true;
        else if (std::string(name) == "nemotron_nvfp4") f.nemotron_nvfp4 = true;
        else if (std::string(name) == "gigachat") f.gigachat = true;
        else if (std::string(name) == "glmflash") f.glmflash = true;
        else if (std::string(name) == "granite") f.granite = true;
        else if (std::string(name) == "phi") f.phi = true;
        else if (std::string(name) == "mimo") f.mimo = true;
        else if (std::string(name) == "alice") f.alice = true;
        else if (std::string(name) == "mistral") f.mistral = true;
        else if (std::string(name) == "minimax") f.minimax = true;
        else if (std::string(name) == "minimax_flash") f.minimax_flash = true;
        else if (std::string(name) == "hy") f.hy = true;
        else if (std::string(name) == "hy_simulmt") f.hy_simulmt = true;
        const auto r = ResolvePipeline(f);
        CHECK(r.error.empty());
        CHECK(r.pipeline == expected);
    }
}

TEST_CASE("forge cli: no family flag falls back to the generic converter",
          "[lsxforge][cli]")
{
    CHECK(ResolvePipeline(Base()).pipeline == Pipeline::kGeneric);
}

TEST_CASE("forge cli: repack and vram-recompute side paths", "[lsxforge][cli]")
{
    Flags f = Base();
    f.repack_shared = true;
    CHECK(ResolvePipeline(f).pipeline == Pipeline::kRepackShared);

    Flags g = Base();
    g.repack_experts = true;
    CHECK(ResolvePipeline(g).pipeline == Pipeline::kRepackRouted);

    Flags v = Base();
    v.vram_recompute = true;
    CHECK(ResolvePipeline(v).pipeline == Pipeline::kVramRecompute);

    Flags d = Base();
    d.deepseek_nvfp4 = true;
    CHECK(ResolvePipeline(d).pipeline == Pipeline::kDeepseekNvfp4);

    Flags e = Base();
    e.deepseek_v41 = true;
    CHECK(ResolvePipeline(e).pipeline == Pipeline::kDeepseekV41);

    Flags p = Base();
    p.deepseek_pro = true;
    CHECK(ResolvePipeline(p).pipeline == Pipeline::kDeepseekPro);
}

TEST_CASE("forge cli: two family flags are rejected as ambiguous",
          "[lsxforge][cli]")
{
    // Previously first-match-wins in main()'s order, so `--glm --alice` ran
    // the alice pipeline and the typo in the other flag went unnoticed while
    // the wrong converter ran. Selecting two different converters is now a
    // usage error (exit 2).
    for (const auto &[a, b] : {std::pair{"glm", "alice"},
                                std::pair{"kimi", "granite"},
                                {"phi", "glm"}, {"gemma4", "mimo"}}) {
        CAPTURE(a);
        CAPTURE(b);
        Flags f = Base();
        const std::string first(a), second(b);
        if (first == "glm") f.glm = true;
        else if (first == "alice") f.alice = true;
        else if (first == "kimi") f.kimi = true;
        else if (first == "granite") f.granite = true;
        else if (first == "phi") f.phi = true;
        else if (first == "gemma4") f.gemma4 = true;
        else if (first == "mimo") f.mimo = true;
        if (second == "glm") f.glm = true;
        else if (second == "alice") f.alice = true;
        else if (second == "kimi") f.kimi = true;
        else if (second == "granite") f.granite = true;
        else if (second == "phi") f.phi = true;
        else if (second == "gemma4") f.gemma4 = true;
        else if (second == "mimo") f.mimo = true;

        const auto r = ResolvePipeline(f);
        CHECK(r.pipeline == Pipeline::kError);
        CHECK_FALSE(r.error.empty());
        // The message must name both flags so the user can find the typo.
        CHECK(r.error.find("--") != std::string::npos);
    }
}

TEST_CASE("forge cli: a family flag without --input is a usage error",
          "[lsxforge][cli]")
{
    Flags f;  // no --input
    f.alice = true;
    const auto r = ResolvePipeline(f);
    CHECK(r.pipeline == Pipeline::kError);
    CHECK_FALSE(r.error.empty());
}

TEST_CASE("forge cli: the precedence order is version < generate < side paths "
          "< tokenizer < families",
          "[lsxforge][cli]")
{
    // Everything on at once: the earliest rule in the dispatch order must win.
    Flags f = Base();
    f.run = true;
    f.tokenizer = "/src/tok";
    f.vram_recompute = true;
    f.repack_shared = true;
    f.deepseek_v41 = true;
    f.alice = true;
    CHECK(ResolvePipeline(f).pipeline == Pipeline::kGenerate);

    Flags g = Base();
    g.vram_recompute = true;
    g.repack_shared = true;
    g.tokenizer = "/src/tok";
    g.alice = true;
    CHECK(ResolvePipeline(g).pipeline == Pipeline::kVramRecompute);

    Flags h = Base();
    h.repack_shared = true;
    h.tokenizer = "/src/tok";
    h.alice = true;
    CHECK(ResolvePipeline(h).pipeline == Pipeline::kRepackShared);

    Flags i = Base();
    i.tokenizer = "/src/tok";
    i.alice = true;
    CHECK(ResolvePipeline(i).pipeline == Pipeline::kTokenizer);
}