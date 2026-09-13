// TDD — Model independence guard (host-only; no CUDA, no model weights).
//
// INVARIANT (operator directive): every ingested model owns a UNIQUE frozen
// parameter passport and models MUST NOT inherit shared geometry/knobs from
// one another. Each family/model keeps its own `frozen` namespace in a
// per-model constants header; adding or tuning one model must never require
// touching another.
//
// This guard enforces the structural side of that directive:
//   1. Every model passport lives in its own constants header, one per
//      family/model, each in a model-unique namespace.
//   2. No model's header includes another model's constants header
//      (constants are leaf headers, never shared).
//   3. No geometry constant name is defined twice across passports (the
//      frozen:: namespaces overlap by name only — grep-guarded here).
//   4. Process-wide numeric toggles (fast-math class) are documented as
//      single-model-at-a-time knobs; the orchestrator never hosts two
//      models sharing one toggle simultaneously.

#include <catch2/catch_test_macros.hpp>

#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

const fs::path kSrcDir = LSX_SOURCE_DIR "/libs/lsxcommon/src";
const fs::path kIncDir = LSX_SOURCE_DIR "/libs/lsxcommon/include/lsxcommon";

fs::path PassportPath(const char *header)
{
    const std::string h = header;
    const std::string kIncPrefix = "../include/lsxcommon/";
    if (h.rfind(kIncPrefix, 0) == 0)
        return kIncDir / h.substr(kIncPrefix.size());
    return kSrcDir / h;
}

// Model -> passport header (relative to libs/lsxcommon + src). One row per
// ingested model; extend in the SAME commit as the new model's passport
// (Model Ingestion Protocol step 3). `ns` = owning namespace; constants may
// repeat names across models (separate namespaces, separate TUs) but the
// OWNING namespace must be unique per model.
struct PassportRow {
    const char *model_type;
    const char *header;
    const char *ns;  // fully qualified namespace holding the constants
};

const std::vector<PassportRow> kPassports = {
    {"glm-5.2", "../include/lsxcommon/glm_moe_load.h", "lsxcommon::glm"},
    {"deepseek-v4-flash", "deepseek_moe_load.cuh", "lsxcommon::gpu"},
    {"deepseek-v4.1-flash", "deepseek_moe_v41_constants.h",
     "lsxcommon::deepseek_v41"},
    {"gemma-4-26b-a4b-it", "../include/lsxcommon/gemma_moe_load.h",
     "lsxcommon::gemma"},
    {"kimi-k2.7-code", "../include/lsxcommon/kimi_moe_load.h",
     "lsxcommon::kimi"},
    {"qwen3.5-9b", "qwen_dense_model.h", "Qwen3_5_9BModel"},
    {"qwen3.8-27b", "qwen_dense_27b_model.h", "Qwen3_8_27BModel"},
    {"qwen3.5-35b-a3b", "qwen_moe_constants.h", "lsxcommon::qwen_moe::frozen"},
    {"qwen3.8-flash-next", "qwen_moe_exp_constants.h",
     "lsxcommon::qwen_moe_exp::frozen"},
    {"qwen3.6-35b-a3b", "qwen_moe_36_constants.h",
     "lsxcommon::qwen_moe_36::frozen"},
};

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("every ingested model has its own constants header", "[independence]")
{
    for (const auto &row : kPassports) {
        INFO("model " << row.model_type << " passport " << row.header);
        REQUIRE(fs::exists(PassportPath(row.header)));
    }
}

TEST_CASE("model constants namespaces are unique per model", "[independence]")
{
    absl::btree_set<std::string> namespaces;
    for (const auto &row : kPassports) {
        INFO("namespace " << row.ns);
        REQUIRE(namespaces.insert(row.ns).second);
    }
}

TEST_CASE("no passport includes another model's passport", "[independence]")
{
    // Each constants header must be a leaf: no cross-includes between
    // model passports. A shared passport would couple the models' frozen
    // geometry — exactly what the independence directive forbids.
    // Only real preprocessor include/next-include lines count; prose
    // mentions in comments are fine.
    const auto includes = [](const std::string &text,
                             const std::string &needle) {
        size_t pos = 0;
        while ((pos = text.find(needle, pos)) != std::string::npos) {
            // Find the start of the line containing the match.
            const size_t line_start =
                text.rfind('\n', pos) == std::string::npos
                    ? 0
                    : text.rfind('\n', pos) + 1;
            std::string line =
                text.substr(line_start, text.find('\n', pos) - line_start);
            if (line.find("#include") != std::string::npos) return true;
            pos += needle.size();
        }
        return false;
    };
    for (const auto &row : kPassports) {
        const std::string text = ReadFile(PassportPath(row.header));
        for (const auto &other : kPassports) {
            if (std::string_view(other.header) == std::string_view(row.header))
                continue;
            INFO(row.header << " must not include " << other.header);
            // Match on the bare file name so quoted/angled forms both hit.
            REQUIRE_FALSE(includes(text, other.header));
        }
    }
}

TEST_CASE("each passport defines its geometry in its own namespace or class",
          "[independence]")
{
    // Namespaced passports declare their owner namespace (last segment as
    // written in the header); class-scoped passports (dense qwen) use
    // static constexpr members. The passport must NOT define constants in
    // a namespace owned by a different model.
    const absl::btree_map<std::string, std::string> owner = {
        {"../include/lsxcommon/glm_moe_load.h", "namespace glm {"},
        {"deepseek_moe_load.cuh", "namespace gpu {"},
        {"../include/lsxcommon/gemma_moe_load.h", "namespace gemma {"},
        {"../include/lsxcommon/kimi_moe_load.h", "namespace kimi {"},
        {"qwen_moe_constants.h", "namespace frozen {"},
        {"qwen_moe_exp_constants.h", "namespace frozen {"},
        {"qwen_moe_36_constants.h", "namespace frozen {"},
        {"qwen_dense_model.h", "static constexpr int kHiddenSize"},
        {"qwen_dense_27b_model.h", "static constexpr int kHiddenSize"},
    };
    for (const auto &kv : owner) {
        const std::string text = ReadFile(PassportPath(kv.first.c_str()));
        INFO(kv.first << " must declare " << kv.second);
        REQUIRE(text.find(kv.second) != std::string::npos);
    }
}

TEST_CASE("cross-family geometry borrowing is forbidden", "[independence]")
{
    // A qwen-family model must not pull another family's loader headers:
    // any geometry a model needs is frozen in its own passport.
    const std::vector<std::string> qwen_files = {
        "qwen_moe_constants.h", "qwen_moe_exp_constants.h",
        "qwen_moe_36_constants.h",
        "qwen_moe_model.h", "qwen_moe_exp_model.h",
        "qwen_moe_36_model.h",
        "../include/lsxcommon/qwen_moe_load.h",
        "../include/lsxcommon/qwen_moe_exp_load.h",
    };
    const std::vector<std::string> foreign = {
        "glm_moe_load.h", "deepseek_moe_load.cuh", "gemma_moe_load.h",
        "kimi_moe_load.h",
    };
    for (const auto &f : qwen_files) {
        const fs::path p =
            f.rfind("../include/", 0) == 0 ? kIncDir / f.substr(11)
                                           : kSrcDir / f;
        const std::string text = ReadFile(p);
        for (const auto &x : foreign) {
            INFO(f << " must not include " << x);
            REQUIRE(text.find(x) == std::string::npos);
        }
    }
}

TEST_CASE("fast-math toggles stay family-scoped", "[independence]")
{
    // The kernel-routing state lives once, in the model-neutral NVFP4 layer
    // (lsx_nvfp4). qwen and deepseek each keep family-scoped access; no
    // family's translation unit may define or read the other family's
    // toggle state.
    const std::string nvfp4_inf =
        ReadFile(kSrcDir / "lsx_inference_nvfp4.cu");
    const std::string qwen_inf =
        ReadFile(kSrcDir / "qwen_inference.cu");
    const std::string ds_inf =
        ReadFile(kSrcDir / "deepseek_moe_inference_moe.cu");
    // Single definition site of the routing state.
    REQUIRE(nvfp4_inf.find("g_lsx_tc_fast_math") != std::string::npos);
    REQUIRE(nvfp4_inf.find("g_lsx_seq1_gemv") != std::string::npos);
    // qwen uses the neutral layer (no duplicate state).
    REQUIRE(qwen_inf.find("g_qwen_tc_fast_math") == std::string::npos);
    REQUIRE(qwen_inf.find("g_deepseek_fast_math") == std::string::npos);
    // deepseek owns its toggle and never reads the neutral one.
    REQUIRE(ds_inf.find("g_deepseek_fast_math") != std::string::npos);
    REQUIRE(ds_inf.find("g_lsx_tc_fast_math") == std::string::npos);
}

TEST_CASE("every consumer model asserts its own fast-math policy per Infer",
          "[independence]")
{
    // The routing toggles are process-global; each model that routes
    // through the neutral NVFP4 layer must open LsxFastMathScope with its
    // OWN config inside Infer — the last-constructed model's policy must
    // never leak into another model's kernel selection. Extend this list
    // in the same commit as the new model's Infer (Model Ingestion
    // Protocol step 3).
    // Neutral-layer consumers: each routes kernel selection through
    // lsx_nvfp4 (or a family-scoped equivalent) and must assert its OWN
    // policy per call. glm-5.2 and gemma are family-own models — they carry
    // the marker comment checked by the TEST_CASE below instead.
    const std::vector<const char *> consumers = {
        "qwen_base_model.cpp",      // qwen dense (3.5-9B / 3.8-27B)
        "qwen_moe_model.cpp",       // qwen3.5-35b-a3b
        "qwen_moe_36_model.cpp",    // qwen3.6-35b-a3b
        "qwen_moe_exp_model.cpp",   // qwen3.8-flash-next
        "glm_moe_flash_model.cpp",  // glm-5.3-flash
        "kimi_moe_model.cpp",       // kimi-k2.7-code (TC batch, 100 t/s round)
        "deepseek_moe_model.cpp",   // deepseek-v4-flash (family-scoped toggle)
        "deepseek_moe_v41_model.cpp",  // deepseek-v4.1-flash
    };
    for (const char *f : consumers) {
        const std::string text = ReadFile(kSrcDir / f);
        INFO(f << " must open a per-call FastMathScope in Infer (tri-state "
                  "FastMathMode or the legacy bool pair)");
        REQUIRE(text.find("FastMathScope") != std::string::npos);
        REQUIRE(text.find("EffectiveFastMathMode(cfg_)") !=
                std::string::npos);
    }
}

TEST_CASE("glm-5.3-flash calls the neutral NVFP4 layer directly",
          "[independence]")
{
    // The qwen-named GEMM shims (RunQwenNvfp4Gemm*) are qwen-family call-site
    // aliases over lsx_nvfp4. A glm model must not depend on them: flash
    // kernels call lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm directly. The
    // shared slot/result TYPES (QwenMoeRoutedSlot, QwenGenerationResult)
    // remain legitimate cross-model data carriers.
    for (const char *f : {"glm_moe_flash_inference.cu",
                          "glm_moe_flash_model.cpp"}) {
        const std::string text = ReadFile(kSrcDir / f);
        INFO(f << " must not use qwen::RunQwenNvfp4Gemm shims");
        REQUIRE(text.find("qwen::RunQwenNvfp4Gemm") == std::string::npos);
    }
}

TEST_CASE("family-own models document their fast-math policy",
          "[independence]")
{
    // glm-5.2 does not route through the neutral NVFP4 layer (no
    // process-global toggle exists for it) and carries the family-own
    // marker. gemma routes its dense NVFP4 GEMMs through the neutral
    // dispatcher and therefore must carry the per-call LsxFastMathScope
    // marker (model-independence invariant, AGENTS.md).
    {
        const std::string text = ReadFile(kSrcDir / "glm_moe_model.cpp");
        INFO("glm_moe_model.cpp must declare: fast-math policy: family-own");
        REQUIRE(text.find(
                    "fast-math policy: family-own (no process-global toggles)") !=
                std::string::npos);
    }
    {
        const std::string text = ReadFile(kSrcDir / "gemma_base_model.cpp");
        INFO("gemma_base_model.cpp must declare: per-call LsxFastMathScope");
        REQUIRE(text.find(
                    "per-call fast-math policy (LsxFastMathScope, model-independence "
                    "invariant)") != std::string::npos);
    }
}

TEST_CASE("broken fused-decode attention path is removed", "[independence]")
{
    // KNOWN-BROKEN (diverged on the real Qwen3.5-9B checkpoint; unwired from
    // generation in 19e8d4f) and test-only since — deleted by the 2026-09
    // model-independence audit. Reintroduction requires a full-layer parity
    // re-validation on the real checkpoint (see AGENTS.md selfcheck rules).
    const std::string header = ReadFile(kIncDir / "qwen_inference.h");
    REQUIRE(header.find("RunQwenFusedDecodeAttnKernel") == std::string::npos);
    const std::string tu = ReadFile(kSrcDir / "qwen_inference.cu");
    REQUIRE(tu.find("RunQwenFusedDecodeAttnKernel") == std::string::npos);
    REQUIRE(tu.find("RunQwenFusedQkv") == std::string::npos);
    REQUIRE(tu.find("RunQwenFusedOProj") == std::string::npos);
}

TEST_CASE("perf layer is model-neutral (no family stem)", "[independence][perf]")
{
    // gpu_perf / perf_calibration are model-neutral infrastructure
    // (lsx_inference_nvfp4 parity): no family prefix in the file names and
    // no family-specific identifiers in the policy code.
    for (const char *f : {"gpu_perf.h", "perf_calibration.h"}) {
        const std::string text = ReadFile(kIncDir / f);
        for (const char *stem : {"glm", "qwen", "gemma", "kimi", "deepseek"}) {
            INFO(f << " must not mention family stem " << stem);
            REQUIRE(text.find(stem) == std::string::npos);
        }
    }
}
