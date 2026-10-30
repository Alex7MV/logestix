// TDD — Model test-suite guard (host-only static scan of tests/).
//
// INVARIANT (operator directive): model test suites must be AUTONOMOUS
// relative to each other — mirroring the model-independence invariant of the
// production code (test_model_independence_guard.cpp) — and every suite must
// cover the FULL model lifecycle (Model Ingestion Protocol):
//
//     convert -> load -> model -> kernels -> selfcheck -> fsi -> code -> det
//
// This guard enforces the structural side of that directive:
//   1. Every model suite declares every lifecycle stage in its registry row
//      and each stage target is actually registered in tests/CMakeLists.txt.
//      Stages shared between sibling variants (e.g. the dense converter reused
//      by qwen3.8-27b) point at the SAME target on purpose — the sharing is
//      explicit, never silent.
//   2. Every selfcheck / fsi / code / det-stage target declares ctest
//      SKIP_RETURN_CODE 4 (real weights and golden are optional; absence must
//      skip, not fail).
//   3. No test file includes another model FAMILY's fixture/header. Family is
//      derived from the first path segment after the test_ prefix (glm, qwen,
//      deepseek, gemma, kimi); family-neutral headers (no family stem, e.g.
//      test_synth_checkpoint.h) are allowed everywhere. This is the test-side
//      mirror of "cross-family geometry borrowing is forbidden".

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

const std::set<std::string> &FamilyStems()
{
    static const std::set<std::string> v = {"glm", "qwen", "deepseek", "gemma",
                                            "kimi", "nemotron", "gigachat",
                                            "ornith", "granite", "mimo", "phi",
                                            "alice", "flint"};
    return v;
}

// "test_glm_moe_fixture.h" -> "glm"; "test_synth_checkpoint.h" -> nullopt
// (family-neutral infrastructure, allowed for every suite).
std::optional<std::string> FamilyStem(const std::string &file_name)
{
    const std::string kPrefix = "test_";
    if (file_name.rfind(kPrefix, 0) != 0) return std::nullopt;
    const std::string rest = file_name.substr(kPrefix.size());
    const auto pos = rest.find('_');
    if (pos == std::string::npos) return std::nullopt;
    const std::string stem = rest.substr(0, pos);
    if (!FamilyStems().count(stem)) return std::nullopt;
    return stem;
}

struct StageSpec
{
    const char *stage;
    std::vector<const char *> targets;
};

struct SuiteSpec
{
    const char *model;
    std::vector<StageSpec> stages;
};

// One row per ingested model suite; extend in the SAME commit as the new
// model's tests (Model Ingestion Protocol step 8). Stage order is the
// lifecycle order. A stage target may be shared with a sibling variant —
// that sharing is the registry row, in plain sight.
const std::vector<SuiteSpec> &Suites()
{
    static const std::vector<SuiteSpec> v = {
        {"glm-5.2",
         {{"convert", {"test_glm_moe_convert"}},
          {"load", {"test_glm_moe_loader", "test_glm_moe_load_fast",
                    "test_glm_moe_expert_cache"}},
          {"model",
           {"test_glm_moe_model", "test_glm_moe_model_lifecycle",
            "test_glm_moe_prompt_output", "test_glm_moe_prompt_framing"}},
          {"kernels",
           {"test_glm_moe_moe_compute", "test_glm_moe_attention",
            "test_glm_moe_experts", "test_glm_moe_dense_compute",
            "test_glm_moe_layer_compute", "test_glm_moe_moe_router",
            "test_glm_moe_full_compute", "test_glm_moe_inference",
            "test_glm_moe_prefill_seqp"}},
          {"selfcheck", {"test_glm_moe_selfcheck"}},
          {"fsi", {"test_glm_moe_fsi_selfcheck"}},
          {"code", {"test_glm_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_glm_5_2"}},
          {"det", {"test_glm_moe_det_selfcheck"}}}},
        {"glm-5.3-flash",
         {{"convert", {"test_glm_moe_flash_convert"}},
          {"load", {"test_glm_moe_flash_loader"}},
          {"model",
           {"test_glm_moe_flash_model", "test_glm_flash_tool_framing"}},
          {"kernels", {"test_glm_moe_flash_kernels"}},
          {"selfcheck", {"test_glm_moe_flash_selfcheck"}},
          {"fsi", {"test_glm_moe_flash_fsi_selfcheck"}},
          {"code", {"test_glm_moe_flash_code_selfcheck"}}, {"perf", {"test_perf_baseline_glm_5_3_flash"}},
          {"det", {"test_glm_moe_flash_det_selfcheck"}}}},
        // No convert stage by design: DeepSeek-V4-Flash ingests a
        // pre-converted logestix layout (no converter exists in the repo);
        // classification/topology of that layout is the model stage.
        {"deepseek-v4-flash",
         {{"convert",
           {"test_deepseek_moe_convert", "test_deepseek_nvfp4_repack",
            "test_deepseek_nvfp4_repack_tool"}},
          {"load", {"test_deepseek_moe_loader", "test_deepseek_moe_load_fast"}},
          {"model",
           {"test_deepseek_moe_model", "test_deepseek_moe_prompt_output",
            "test_deepseek_tool_framing"}},
          {"kernels",
           {"test_deepseek_moe_inference_kernels", "test_deepseek_moe_arena",
            "test_deepseek_moe_moe", "test_deepseek_moe_norm_kernels"}},
          {"selfcheck", {"test_deepseek_moe_selfcheck"}},
          {"fsi", {"test_deepseek_moe_fsi_selfcheck"}},
          {"code", {"test_deepseek_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_deepseek_v4_flash"}},
          {"det", {"test_deepseek_moe_det_selfcheck"}}}},
        {"deepseek-v4.1-flash",
         {{"convert", {"test_deepseek_moe_v41_convert"}},
          {"load",
           {"test_deepseek_moe_v41_loader", "test_deepseek_moe_v41_load_fast"}},
          {"model",
           {"test_deepseek_moe_v41_model", "test_deepseek_tool_framing"}},
          {"kernels",
           {"test_deepseek_moe_v41_dequant", "test_deepseek_moe_v41_kernels",
            "test_deepseek_moe_v41_actquant",
            "test_deepseek_moe_v41_pipeline"}},
          {"selfcheck", {"test_deepseek_moe_v41_selfcheck"}},
          {"fsi", {"test_deepseek_moe_v41_fsi_selfcheck"}},
          {"code", {"test_deepseek_moe_v41_code_selfcheck"}}, {"perf", {"test_perf_baseline_deepseek_v41_flash"}},
          {"det", {"test_deepseek_moe_v41_det_selfcheck"}}}},
        {"deepseek-v4-pro-0813",
         {{"convert", {"test_deepseek_moe_pro_convert"}},
          {"load",
           {"test_deepseek_moe_pro_loader", "test_deepseek_moe_pro_load_fast"}},
          {"model", {"test_deepseek_moe_pro_model", "test_deepseek_tool_framing"}},
          {"kernels", {"test_deepseek_moe_pro_kernels"}},
          {"selfcheck", {"test_deepseek_moe_pro_selfcheck"}},
          {"fsi", {"test_deepseek_moe_pro_fsi_selfcheck"}},
          {"code", {"test_deepseek_moe_pro_code_selfcheck"}},
          {"perf", {"test_perf_baseline_deepseek_v4_pro"}},
          {"det", {"test_deepseek_moe_pro_det_selfcheck"}}}},
        {"qwen3.5-9b",
         {{"convert",
           {"test_qwen_dense_convert", "test_qwen_dense_converter_integrity"}},
          {"load", {"test_qwen_dense_loader"}},
          {"model", {"test_qwen_dense_model", "test_qwen_tool_framing"}},
          {"kernels",
           {"test_qwen_dense_mlp", "test_qwen_dense_full_attn",
            "test_qwen_dense_nvfp4", "test_qwen_dense_norm",
            "test_qwen_dense_rotary", "test_qwen_dense_shuffle_reduce",
            "test_qwen_dense_full_attn_parallel",
            "test_qwen_dense_linear_attn", "test_qwen_dense_fused_mlp",
            "test_qwen_dense_fused_linear_attn", "test_qwen_dense_kv_cache",
            "test_qwen_dense_kv_parity"}},
          {"selfcheck", {"test_qwen_dense_selfcheck"}},
          {"fsi", {"test_qwen_dense_fsi_selfcheck"}},
          {"code", {"test_qwen_dense_code_selfcheck"}}, {"perf", {"test_perf_baseline_qwen3_5_9b"}},
          {"det", {"test_qwen_dense_det_selfcheck"}}}},
        // qwen3.8-27b shares the dense convert/load/kernels pipeline with
        // qwen3.5-9b (QwenBaseModel); its own stages are model+selfcheck.
        {"qwen3.8-27b",
         {{"convert", {"test_qwen_dense_convert"}},
          {"load", {"test_qwen_dense_loader"}},
          {"model", {"test_qwen_dense_27b_model", "test_qwen_tool_framing"}},
          {"kernels",
           {"test_qwen_dense_nvfp4", "test_qwen_nvfp4_gemv",
            "test_qwen_nvfp4_tile_gemm"}},
          {"selfcheck", {"test_qwen_dense_27b_selfcheck"}},
          {"fsi", {"test_qwen_dense_27b_fsi_selfcheck"}},
          {"code", {"test_qwen_dense_27b_code_selfcheck"}}, {"perf", {"test_perf_baseline_qwen3_8_27b"}},
          {"det", {"test_qwen_dense_27b_det_selfcheck"}}}},
        {"qwen3.5-35b-a3b",
         {{"convert", {"test_qwen_moe_convert"}},
          {"load", {"test_qwen_moe_loader"}},
          {"model", {"test_qwen_moe_model"}},
          {"kernels", {"test_qwen_moe_kernels"}},
          {"selfcheck", {"test_qwen_moe_selfcheck"}},
          {"fsi", {"test_qwen_moe_fsi_selfcheck"}},
          {"code", {"test_qwen_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_qwen3_5_35b_a3b"}},
          {"det", {"test_qwen_moe_det_selfcheck"}}}},
        // qwen3.6 shares loader + kernels with qwen3.5-35b (QwenMoeModelBase).
        {"qwen3.6-35b-a3b",
         {{"convert", {"test_qwen_moe_36_convert"}},
          {"load", {"test_qwen_moe_loader"}},
          {"model", {"test_qwen_moe_36_model"}},
          {"kernels", {"test_qwen_moe_kernels"}},
          {"selfcheck", {"test_qwen_moe_36_selfcheck"}},
          {"fsi", {"test_qwen_moe_36_fsi_selfcheck"}},
          {"code", {"test_qwen_moe_36_code_selfcheck"}}, {"perf", {"test_perf_baseline_qwen3_6_35b_a3b"}},
          {"det", {"test_qwen_moe_36_det_selfcheck"}}}},
        {"ornith-1.5-35b-a3b",
         {{"convert", {"test_ornith_moe_convert"}},
          {"load", {"test_ornith_moe_loader"}},
          {"model", {"test_ornith_moe_model"}},
          {"kernels", {"test_ornith_moe_kernels"}},
          {"selfcheck", {"test_ornith_moe_selfcheck"}},
          {"fsi", {"test_ornith_moe_fsi_selfcheck"}},
          {"code", {"test_ornith_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_ornith"}},
          {"det", {"test_ornith_moe_det_selfcheck"}}}},
        {"qwen3.8-flash-next",
         {{"convert", {"test_qwen_moe_exp_convert"}},
          {"load", {"test_qwen_moe_exp_loader"}},
          {"model", {"test_qwen_moe_exp_model"}},
          {"kernels", {"test_qwen_moe_exp_kernels"}},
          {"selfcheck", {"test_qwen_moe_exp_selfcheck",
                         "test_qwen_moe_exp_eos_selfcheck"}},
          {"fsi", {"test_qwen_moe_exp_fsi_selfcheck"}},
          {"code", {"test_qwen_moe_exp_code_selfcheck"}}, {"perf", {"test_perf_baseline_qwen3_8_flash_next"}},
          {"det", {"test_qwen_moe_exp_det_selfcheck"}}}},
        {"gemma-4-26b-a4b-it",
         {{"convert", {"test_gemma_moe_convert"}},
          {"load", {"test_gemma_moe_loader"}},
          {"model",
           {"test_gemma_moe_model", "test_gemma_prompt_framing",
            "test_gemma_tool_framing"}},
          {"kernels",
           {"test_gemma_moe_norm", "test_gemma_moe_rotary",
            "test_gemma_moe_attn", "test_gemma_moe_moe"}},
          {"selfcheck", {"test_gemma_moe_selfcheck"}},
          {"fsi", {"test_gemma_moe_fsi_selfcheck"}},
          {"code", {"test_gemma_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_gemma_4_26b_a4b_it"}},
          {"det", {"test_gemma_moe_det_selfcheck"}}}},
        {"gemma-4-31b-it",
         {{"convert", {"test_gemma_dense_convert"}},
          {"load", {"test_gemma_dense_loader"}},
          {"model", {"test_gemma_dense_model"}},
          {"kernels", {"test_gemma_dense_kernels"}},
          {"selfcheck", {"test_gemma_dense_selfcheck"}},
          {"fsi", {"test_gemma_dense_fsi_selfcheck"}},
          {"code", {"test_gemma_dense_code_selfcheck"}},
          {"perf", {"test_perf_baseline_gemma_4_31b_it"}},
          {"det", {"test_gemma_dense_det_selfcheck"}}}},
        {"kimi-k2.7-code",
         {{"convert", {"test_kimi_moe_convert"}},
          {"load",
           {"test_kimi_moe_loader", "test_kimi_moe_expert_cache",
            "test_kimi_moe_tokenizer", "test_kimi_moe_tokenizer_parity"}},
          {"model", {"test_kimi_moe_model", "test_kimi_tool_framing"}},
          {"kernels",
           {"test_kimi_moe_router", "test_kimi_moe_attention",
            "test_kimi_moe_rope_parity", "test_kimi_moe_experts",
            "test_kimi_moe_int4_dequant", "test_kimi_moe_inference",
            "test_kimi_moe_int4_real", "test_kimi_moe_mla_proj_real",
            "test_kimi_moe_tc_batch",
            "test_kimi_moe_tc_diag"}},
          {"selfcheck", {"test_kimi_moe_selfcheck"}},
          {"fsi", {"test_kimi_moe_fsi_selfcheck"}},
          {"code", {"test_kimi_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_kimi_k2_7_code"}},
          {"det", {"test_kimi_moe_det_selfcheck"}}}},
        {"nemotron-3.5-lightning-30b-a3b",
         {{"convert", {"test_nemotron_moe_convert"}},
          {"load",
           {"test_nemotron_moe_loader", "test_nemotron_moe_expert_cache"}},
          {"model", {"test_nemotron_moe_model"}},
          {"kernels",
           {"test_nemotron_moe_kernels", "test_nemotron_moe_math",
            "test_nemotron_session"}},
          {"selfcheck", {"test_nemotron_moe_selfcheck"}},
          {"fsi", {"test_nemotron_moe_fsi_selfcheck"}},
          {"code", {"test_nemotron_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_nemotron"}},
          {"det", {"test_nemotron_moe_det_selfcheck"}}}},
        // Pre-quantized ModelOpt sibling: same kernels/model contour as
        // nemotron-3.5-lightning-30b-a3b, its own converter + frozen passport.
        {"nemotron-3.5-lightning-30b-a3b-nvfp4",
         {{"convert", {"test_nemotron_moe_nvfp4_convert"}},
          {"load", {"test_nemotron_moe_nvfp4_loader"}},
          {"model", {"test_nemotron_moe_nvfp4_model"}},
          {"kernels",
           {"test_nemotron_moe_kernels", "test_nemotron_moe_math",
            "test_nemotron_session"}},
          {"selfcheck", {"test_nemotron_moe_nvfp4_selfcheck"}},
          {"fsi", {"test_nemotron_moe_nvfp4_fsi_selfcheck"}},
          {"code", {"test_nemotron_moe_nvfp4_code_selfcheck"}},
          {"perf", {"test_perf_baseline_nemotron_nvfp4"}},
          {"det", {"test_nemotron_moe_nvfp4_det_selfcheck"}}}},
        {"gigachat3.5-432b-a28b-reasoning",
         {{"convert", {"test_gigachat_moe_convert"}},
          {"load",
           {"test_gigachat_moe_loader", "test_gigachat_moe_expert_cache",
            "test_gigachat_moe_load_fast"}},
          {"model",
           {"test_gigachat_moe_model", "test_gigachat_tool_framing"}},
          {"kernels",
           {"test_gigachat_moe_kernels", "test_gigachat_moe_attn",
            "test_gigachat_moe_devkernels", "test_gigachat_moe_parity",
            "test_gigachat_moe_math"}},
          {"selfcheck", {"test_gigachat_moe_selfcheck"}},
          {"fsi", {"test_gigachat_moe_fsi_selfcheck"}},
          {"code", {"test_gigachat_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_gigachat"}},
          {"det", {"test_gigachat_moe_det_selfcheck"}}}},
        {"granite-4.2-30b",
         {{"convert", {"test_granite_dense_convert"}},
          {"load", {"test_granite_dense_loader"}},
          {"model", {"test_granite_dense_model",
                     "test_granite_tool_framing"}},
          {"kernels", {"test_granite_dense_kernels"}},
          {"selfcheck", {"test_granite_dense_selfcheck"}},
          {"fsi", {"test_granite_dense_fsi_selfcheck"}},
          {"code", {"test_granite_dense_code_selfcheck"}}, {"perf", {"test_perf_baseline_granite_4_2_30b"}},
          {"det", {"test_granite_dense_det_selfcheck"}}}},
        {"mimo-v2.6-flash-rl",
         {{"convert", {"test_mimo_moe_convert"}},
          {"load",
           {"test_mimo_moe_loader", "test_mimo_moe_expert_cache",
            "test_mimo_moe_geometry"}},
          {"model", {"test_mimo_moe_model", "test_mimo_tool_framing"}},
          {"kernels", {"test_mimo_moe_kernels", "test_mimo_moe_dequant"}},
          {"selfcheck", {"test_mimo_moe_selfcheck"}},
          {"fsi", {"test_mimo_moe_fsi_selfcheck"}},
          {"code", {"test_mimo_moe_code_selfcheck"}}, {"perf", {"test_perf_baseline_mimo_v2_6_flash_rl"}},
          {"det", {"test_mimo_moe_det_selfcheck"}}}},
        {"phi-4",
         {{"convert", {"test_phi_dense_convert"}},
          {"load", {"test_phi_dense_loader"}},
          {"model", {"test_phi_dense_model", "test_phi_tool_framing"}},
          {"kernels", {"test_phi_dense_kernels"}},
          {"selfcheck", {"test_phi_dense_selfcheck"}},
          {"fsi", {"test_phi_dense_fsi_selfcheck"}},
          {"code", {"test_phi_dense_code_selfcheck"}},
          {"perf", {"test_perf_baseline_phi_4"}},
          {"det", {"test_phi_dense_det_selfcheck"}}}},
        {"xing4.0-29b-a4b",
         {{"convert", {"test_xing_moe_convert"}},
          {"load", {"test_xing_moe_loader"}},
          {"model", {"test_xing_moe_model", "test_xing_tool_framing"}},
          {"kernels", {"test_xing_moe_kernels", "test_xing_kv_layout_parity"}},
          {"selfcheck", {"test_xing_moe_selfcheck"}},
          {"fsi", {"test_xing_moe_fsi_selfcheck"}},
          {"code", {"test_xing_moe_code_selfcheck"}},
          {"perf", {"test_perf_baseline_xing_4_0_29b_a4b"}},
          {"det", {"test_xing_moe_det_selfcheck"}}}},
        {"alice-foundation-80b-a3b",
         {{"convert", {"test_alice_moe_convert"}},
          {"load", {"test_alice_moe_loader", "test_alice_tokenizer"}},
          {"model", {"test_alice_moe_model", "test_alice_tool_framing"}},
          {"kernels", {"test_alice_moe_kernels"}},
          {"selfcheck", {"test_alice_moe_selfcheck"}},
          {"fsi", {"test_alice_moe_fsi_selfcheck"}},
          {"code", {"test_alice_moe_code_selfcheck"}},
          {"perf", {"test_perf_baseline_alice_foundation_80b_a3b"}},
          {"det", {"test_alice_moe_det_selfcheck"}}}},
        {"mistral-small-4-119b",
         {{"convert", {"test_mistral_moe_convert"}},
          {"load", {"test_mistral_moe_loader"}},
          {"model", {"test_mistral_moe_model", "test_mistral_moe_tool_framing"}},
          {"kernels", {"test_mistral_moe_kernels"}},
          {"selfcheck", {"test_mistral_moe_selfcheck"}},
          {"fsi", {"test_mistral_moe_fsi_selfcheck"}},
          {"code", {"test_mistral_moe_code_selfcheck"}},
          {"perf", {"test_perf_baseline_mistral_small_4"}},
          {"det", {"test_mistral_moe_det_selfcheck"}}}},
        {"minimax-m3",
         {{"convert", {"test_minimax_moe_convert"}},
          {"load", {"test_minimax_moe_loader"}},
          {"model", {"test_minimax_moe_model", "test_minimax_moe_tool_framing"}},
          {"kernels", {"test_minimax_moe_kernels"}},
          {"selfcheck", {"test_minimax_moe_selfcheck"}},
          {"fsi", {"test_minimax_moe_fsi_selfcheck"}},
          {"code", {"test_minimax_moe_code_selfcheck"}},
          {"perf", {"test_perf_baseline_minimax_m3"}},
          {"det", {"test_minimax_moe_det_selfcheck"}}}},
        {"hy-memory-30b-a3b",
         {{"convert", {"test_hy_moe_convert"}},
          {"load", {"test_hy_moe_loader"}},
          {"model", {"test_hy_moe_model", "test_hy_moe_tool_framing"}},
          {"kernels", {"test_hy_moe_kernels"}},
          {"selfcheck", {"test_hy_moe_selfcheck"}},
          {"fsi", {"test_hy_moe_fsi_selfcheck"}},
          {"code", {"test_hy_moe_code_selfcheck"}},
          {"det", {"test_hy_moe_det_selfcheck"}},
          {"perf", {"test_perf_baseline_hy_memory_30b_a3b"}}}},
        {"flint",
         {{"convert", {"test_flint_moe_convert"}},
          {"load", {"test_flint_moe_loader", "test_flint_moe_pretokenizer",
                    "test_flint_moe_tokenizer_parity"}},
          {"model", {"test_flint_moe_model", "test_flint_moe_tool_framing"}},
          {"kernels", {"test_flint_moe_kernels"}},
          {"selfcheck", {"test_flint_moe_selfcheck"}},
          {"fsi", {"test_flint_moe_fsi_selfcheck"}},
          {"code", {"test_flint_moe_code_selfcheck"}},
          {"det", {"test_flint_moe_det_selfcheck"}},
          {"perf", {"test_perf_baseline_flint"}}}},

    };
    return v;
}

std::string ReadFileOrThrow(const fs::path &p)
{
    std::ifstream in(p);
    if (!in.is_open())
        throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> RegisteredTestNames(const std::string &text)
{
    std::vector<std::string> names;
    static const std::regex add_re(
        R"(add_test\s*\(\s*NAME\s+([A-Za-z0-9_]+))");
    static const std::regex fe_re(
        R"(foreach\s*\(\s*\w+((?:\s+(?!IN\b|LISTS\b|RANGE\b)[A-Za-z0-9_]+)+)\s*\))");
    std::smatch m;
    auto begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, add_re))
    {
        names.push_back(m[1]);
        begin = m.suffix().first;
    }
    begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, fe_re))
    {
        std::istringstream is(m[1].str());
        std::string tok;
        while (is >> tok) names.push_back(tok);
        begin = m.suffix().first;
    }
    return names;
}

std::vector<std::string> CheckLifecycle(const std::vector<SuiteSpec> &suites,
                                        const std::set<std::string> &registered)
{
    std::vector<std::string> v;
    for (const auto &s : suites)
        for (const auto &st : s.stages)
            for (const auto *t : st.targets)
                if (!registered.count(t))
                    v.push_back(std::string("suite ") + s.model + " stage " +
                                st.stage + ": target not registered: " + t);
    return v;
}

bool HasSkipReturnCode4(const std::string &cmake_text, const std::string &name)
{
    static const std::regex re(
        R"(set_tests_properties\s*\(([^)]*?)SKIP_RETURN_CODE\s+4)");
    std::smatch m;
    auto begin = cmake_text.cbegin();
    while (std::regex_search(begin, cmake_text.cend(), m, re))
    {
        std::istringstream is(m[1].str());
        std::string tok;
        while (is >> tok)
            if (tok == name) return true;
        begin = m.suffix().first;
    }
    return false;
}

std::vector<std::string> CheckSelfcheckSkip(const std::string &cmake_text)
{
    std::vector<std::string> v;
    for (const auto &s : Suites())
        for (const auto &st : s.stages)
        {
            const std::string stage = st.stage;
            if (stage != "selfcheck" && stage != "fsi" &&
                stage != "code" && stage != "det") continue;
            for (const auto *t : st.targets)
                if (!HasSkipReturnCode4(cmake_text, t))
                    v.push_back(std::string("selfcheck target without "
                                            "SKIP_RETURN_CODE 4: ") +
                                t);
        }
    return v;
}

std::vector<std::string> CheckCrossFamilyIncludes(const fs::path &tests_dir)
{
    std::vector<std::string> v;
    static const std::regex inc_re(
        R"inc(^\s*#\s*include\s+"(test_[A-Za-z0-9_.]+\.h)")inc");
    static const std::regex ext_re(R"(\.(cpp|cu|h)$)");
    std::error_code ec;
    for (auto &e : fs::directory_iterator(tests_dir, ec))
    {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (!std::regex_search(name, ext_re)) continue;
        const auto self = FamilyStem(name);
        if (!self) continue;  // family-neutral file: nothing to check
        std::istringstream in(ReadFileOrThrow(e.path()));
        std::string line;
        while (std::getline(in, line))
        {
            std::smatch m;
            if (!std::regex_match(line, m, inc_re)) continue;
            const auto other = FamilyStem(m[1].str());
            if (!other || *other == *self) continue;
            v.push_back(name + " borrows another family's header " +
                        m[1].str() + " (cross-suite coupling)");
        }
    }
    return v;
}

// Loader-stage tests run on SYNTHETIC converted fixtures only. A loader test
// that opens the model vault turns the whole suite into a multi-GB download
// (exactly the regression the convert-suite discipline killed). Rules:
//   A. every test_*_loader.cpp must build its checkpoint from a fixture
//      header (its own family fixture or the family-neutral
//      test_synth_checkpoint.h);
//   B. no loader-named test (test_*_loader.*, test_*_load_fast.*) may touch
//      the model vault: no vault-root literals, no std::getenv model-root
//      lookups (real-weight coverage lives in selfcheck/perf suites).
//   C. family model-stage tests (test_<family>_model.cpp) must not hardcode
//      vault roots either — env-var opt-in + SKIP is the allowed real-weight
//      pattern (factory/selfcheck/perf stages own vault integration).
bool IsLoaderTest(const std::string &name)
{
    static const std::regex re(
        R"(test_[A-Za-z0-9_]+_(loader|load_fast)\.(cpp|cu)$)");
    return std::regex_match(name, re);
}

bool IsLoaderStageTest(const std::string &name)
{
    static const std::regex re(R"(test_[A-Za-z0-9_]+_loader\.cpp$)");
    return std::regex_match(name, re);
}

std::vector<std::string> CheckLoaderSyntheticDiscipline(
    const fs::path &tests_dir)
{
    std::vector<std::string> v;
    static const std::regex ext_re(R"(\.(cpp|cu)$)");
    static const std::regex fixture_inc_re(
        R"inc(^\s*#\s*include\s+"(test_[A-Za-z0-9_]+_fixture\.h|test_synth_checkpoint\.h)")inc");
    static const std::regex model_stage_re(
        R"(test_(glm|qwen|deepseek|gemma|kimi|nemotron|gigachat|ornith|granite|mimo|phi)[A-Za-z0-9_]*_model\.cpp$)");
    const std::vector<std::string> vault_markers = {
        "mnt/llm_models/models", "huggingface.co"};
    std::error_code ec;
    for (auto &e : fs::directory_iterator(tests_dir, ec))
    {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (!std::regex_search(name, ext_re)) continue;
        const std::string text = ReadFileOrThrow(e.path());
        const bool loader = IsLoaderTest(name);
        const bool model_stage = std::regex_match(name, model_stage_re);
        if (!loader && !model_stage) continue;
        // Strip line comments so prose mentions of vault paths in docs
        // don't trip the literal scan.
        std::string stripped;
        {
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line))
            {
                const auto pos = line.find("//");
                stripped += line.substr(0, pos) + "\n";
            }
        }
        bool has_fixture_inc = false;
        {
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line))
                if (std::regex_match(line, fixture_inc_re))
                {
                    has_fixture_inc = true;
                    break;
                }
        }
        if (IsLoaderStageTest(name) && !has_fixture_inc)
            v.push_back(name +
                        ": loader-stage test without a synthetic fixture "
                        "include (must build its checkpoint, not load a "
                        "vault)");
        for (const auto &marker : vault_markers)
            if (stripped.find(marker) != std::string::npos)
                v.push_back(name + ": hardcodes vault path (" + marker +
                            ")");
        if (loader && stripped.find("getenv") != std::string::npos)
            v.push_back(name +
                        ": loader test resolves the model root via getenv "
                        "(synthetic fixtures only)");
    }
    return v;
}

} // namespace

TEST_CASE("suite guard: family stem classification", "[suite][guard][unit]")
{
    CHECK(FamilyStem("test_glm_moe_fixture.h") == std::optional<std::string>{"glm"});
    CHECK(FamilyStem("test_qwen_dense_fixture.h") == std::optional<std::string>{"qwen"});
    CHECK(FamilyStem("test_deepseek_moe_model.cpp") ==
          std::optional<std::string>{"deepseek"});
    CHECK(FamilyStem("test_kimi_moe_fixture.h") == std::optional<std::string>{"kimi"});
    CHECK(FamilyStem("test_gemma_moe_loader.cpp") == std::optional<std::string>{"gemma"});
    // Family-neutral infrastructure has no family stem.
    CHECK(FamilyStem("test_synth_checkpoint.h") == std::nullopt);
    CHECK(FamilyStem("test_core.cpp") == std::nullopt);
    CHECK(FamilyStem("CMakeLists.txt") == std::nullopt);
}

TEST_CASE("suite guard: lifecycle checker flags missing targets",
          "[suite][guard][unit]")
{
    const std::vector<SuiteSpec> suites = {
        {"m-x",
         {{"convert", {"t_convert"}}, {"model", {"t_model"}}}}};
    CHECK(CheckLifecycle(suites, {"t_convert"}).size() == 1);
    CHECK(CheckLifecycle(suites, {"t_convert", "t_model"}).empty());
}

TEST_CASE("suite guard: selfcheck skip checker", "[suite][guard][unit]")
{
    const std::string cm = R"(add_test(NAME t_self COMMAND t_self)
set_tests_properties(t_self PROPERTIES SKIP_RETURN_CODE 4)
add_test(NAME t_bare COMMAND t_bare))";
    CHECK(HasSkipReturnCode4(cm, "t_self"));
    CHECK_FALSE(HasSkipReturnCode4(cm, "t_bare"));
    const std::vector<SuiteSpec> suites = {
        {"m-x", {{"selfcheck", {"t_self", "t_bare"}}}}};
    // Re-run via the public checker by mirroring its loop.
    std::vector<std::string> v;
    for (const auto &s : suites)
        for (const auto &st : s.stages)
            for (const auto *t : st.targets)
                if (!HasSkipReturnCode4(cm, t))
                    v.push_back(t);
    REQUIRE(v.size() == 1);
    CHECK(v[0] == "t_bare");
}

TEST_CASE("suite guard: cross-family include checker",
          "[suite][guard][unit]")
{
    SECTION("cross-family borrow is a violation (stems differ)")
    {
        CHECK(FamilyStem("test_glm_moe_fixture.h") ==
              std::optional<std::string>{"glm"});
        CHECK(FamilyStem("test_kimi_moe_fixture.h") ==
              std::optional<std::string>{"kimi"});
        CHECK(*FamilyStem("test_glm_moe_fixture.h") !=
              *FamilyStem("test_kimi_moe_fixture.h"));
    }
    SECTION("neutral header has no family stem")
    {
        CHECK(FamilyStem("test_synth_checkpoint.h") == std::nullopt);
    }
}

TEST_CASE("suite guard: every model suite covers the full lifecycle",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto registered = [&] {
        std::set<std::string> s;
        for (const auto &n : RegisteredTestNames(cmake_text)) s.insert(n);
        return s;
    }();
    INFO("registered tests: " << registered.size());
    auto v = CheckLifecycle(Suites(), registered);
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("lifecycle violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("suite guard: selfcheck, fsi, code and det targets declare "
          "SKIP_RETURN_CODE 4",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    auto v = CheckSelfcheckSkip(cmake_text);
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("stage skip violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("suite guard: loader tests stay on synthetic fixtures",
          "[suite][guard][unit]")
{
    SECTION("loader-stage classification")
    {
        CHECK(IsLoaderStageTest("test_glm_moe_flash_loader.cpp"));
        CHECK(IsLoaderStageTest("test_deepseek_moe_loader.cpp"));
        CHECK_FALSE(IsLoaderStageTest("test_glm_moe_load_fast.cu"));
        CHECK(IsLoaderTest("test_glm_moe_load_fast.cu"));
        CHECK(IsLoaderTest("test_qwen_dense_loader.cpp"));
        CHECK_FALSE(IsLoaderTest("test_blackwell_perf_glm.cpp"));
    }
}

TEST_CASE("suite guard: no cross-family fixture borrowing", "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    auto v = CheckCrossFamilyIncludes(root / "tests");
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("cross-family violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("suite guard: loader tests never load the model vault",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    auto v = CheckLoaderSyntheticDiscipline(root / "tests");
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("loader discipline violations:\n" << joined);
    REQUIRE(v.empty());
}

// The lifecycle check above only proves that a registered target EXISTS. The
// reverse direction is what was missing: a family test target that runs on
// every CI sweep but sits in no Suites() row is invisible to the completeness
// report, so its breakage never surfaces as "incomplete suite". Every
// family-prefixed target must be claimed by exactly one model row.
TEST_CASE("suite guard: every family test target is claimed by a suite row",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto registered = [&] {
        std::set<std::string> s;
        for (const auto &n : RegisteredTestNames(cmake_text)) s.insert(n);
        return s;
    }();

    std::set<std::string> claimed;
    for (const auto &s : Suites())
        for (const auto &st : s.stages)
            for (const auto *t : st.targets)
                claimed.insert(t);

    std::vector<std::string> orphans;
    for (const auto &name : registered)
    {
        if (!FamilyStem(name)) continue;
        if (claimed.count(name)) continue;
        orphans.push_back(name);
    }
    std::sort(orphans.begin(), orphans.end());
    std::string joined;
    for (const auto &s : orphans) joined += s + "\n";
    INFO("family test targets absent from Suites():\n" << joined);
    CHECK(orphans.empty());
}

TEST_CASE("suite guard: every suite row declares its tool-calling verdict",
          "[suite][guard]")
{
    // Stage 4b is MANDATORY in the ingestion protocol, yet it lived only as a
    // target listed inside the "model" stage row: a new family could ship with
    // no tool test at all and every guard stayed green. A row must therefore
    // say, explicitly, either
    //   * the target(s) that prove the native render+parse round-trip, or
    //   * "generic" — the model keeps model.h's Hermes default and nothing more
    //     is claimed (parse_tool_calls must then be empty).
    struct ToolRow {
        const char *model;
        std::vector<const char *> targets;  // empty => generic fallback
    };
    static const std::vector<ToolRow> kTooling = {
        {"glm-5.2", {}},
        {"glm-5.3-flash", {"test_glm_flash_tool_framing"}},
        {"deepseek-v4-flash", {"test_deepseek_tool_framing"}},
        {"deepseek-v4.1-flash", {"test_deepseek_tool_framing"}},
        {"deepseek-v4-pro-0813", {"test_deepseek_tool_framing"}},
        {"qwen3.5-9b", {"test_qwen_tool_framing"}},
        {"qwen3.8-27b", {"test_qwen_tool_framing"}},
        {"qwen3.5-35b-a3b", {"test_qwen_tool_framing"}},
        {"qwen3.6-35b-a3b", {"test_qwen_tool_framing"}},
        {"qwen3.8-flash-next", {"test_qwen_tool_framing"}},
        {"gemma-4-26b-a4b-it", {"test_gemma_tool_framing"}},
        // The 31B ships a tokenizer.json / chat_template.jinja that are
        // BYTE-IDENTICAL to the 26B's, so the shared gemma codec target proves
        // the render+parse round-trip for both.
        {"gemma-4-31b-it", {"test_gemma_tool_framing"}},
        {"kimi-k2.7-code", {"test_kimi_tool_framing"}},
        {"ornith-1.5-35b-a3b", {"test_qwen_tool_framing"}},
        {"phi-4", {"test_phi_tool_framing"}},
        {"granite-4.2-30b", {}},
        // nemotron keeps its tool case inside the model target rather than in
        // a dedicated one; naming the target that actually carries it is what
        // the guard checks.
        {"nemotron-3.5-lightning-30b-a3b", {"test_nemotron_moe_model"}},
        // The NVFP4 sibling reuses the parent's Qwen-style XML render/parse
        // verbatim; its model target carries the same tool case.
        {"nemotron-3.5-lightning-30b-a3b-nvfp4",
         {"test_nemotron_moe_nvfp4_model"}},
        {"gigachat3.5-432b-a28b-reasoning", {"test_gigachat_tool_framing"}},
        {"mimo-v2.6-flash-rl", {"test_mimo_tool_framing"}},
        {"alice-foundation-80b-a3b", {"test_alice_tool_framing"}},
        // Xing has a native renderer verified against its chat_template.jinja
        // (not the generic Hermes fallback), so it is named here.
        {"xing4.0-29b-a4b", {"test_xing_tool_framing"}},
        // Mistral Small 4 ships a native [INST]/[TOOL_CALLS] template.
        {"mistral-small-4-119b", {"test_mistral_moe_tool_framing"}},
        {"minimax-m3", {"test_minimax_moe_tool_framing"}},
        {"hy-memory-30b-a3b", {"test_hy_moe_tool_framing"}},
        {"flint", {"test_flint_moe_tool_framing"}},
    };
    // Every ingested model must be in the table (a new family must state its
    // verdict), and every named target must be a registered ctest target.
    const std::set<std::string> registered = [] {
        std::set<std::string> names;
        for (const auto &n : RegisteredTestNames(
                 ReadFileOrThrow(fs::path(LSX_SOURCE_DIR) / "tests" / "CMakeLists.txt")))
            names.insert(n);
        return names;
    }();
    std::vector<std::string> problems;
    for (const auto &s : Suites())
    {
        const auto it = std::find_if(kTooling.begin(), kTooling.end(),
                                     [&](const ToolRow &r) {
                                         return std::string(r.model) == s.model;
                                     });
        if (it == kTooling.end()) {
            problems.push_back(std::string(s.model) +
                               ": no tool-calling verdict declared");
            continue;
        }
        for (const char *t : it->targets)
            if (!registered.count(t))
                problems.push_back(std::string(s.model) + ": tool target '" + t +
                                   "' is not a registered ctest target");
    }
    for (const auto &row : kTooling)
    {
        const bool known = std::any_of(Suites().begin(), Suites().end(),
                                       [&](const SuiteSpec &s) {
                                           return std::string(s.model) == row.model;
                                       });
        if (!known)
            problems.push_back(std::string(row.model) +
                               ": tool verdict row has no Suites() row");
    }
    std::string joined;
    for (const auto &x : problems) joined += x + "\n";
    INFO("tool-calling verdicts:\n" << joined);
    CHECK(problems.empty());
}

TEST_CASE("suite guard: every suite row declares the full stage set",
          "[suite][guard]")
{
    // The header invariant promises convert -> load -> model -> kernels ->
    // selfcheck -> fsi -> code -> det for EVERY row, but nothing enforced the
    // stage list — only that the named targets exist. A row that quietly
    // dropped "model" stayed green forever.
    static const std::vector<const char *> kRequired = {
        "convert", "load", "model", "kernels",
        "selfcheck", "fsi", "code", "det", "perf"};
    // DeepSeek-V4-Flash has no converter in the repo by design (it ingests a
    // pre-converted logestix layout), so "convert" is exempt for that row only.
    static const std::set<std::string> kConvertExempt = {"deepseek-v4-flash"};

    std::vector<std::string> missing;
    for (const auto &s : Suites())
    {
        std::set<std::string> have;
        for (const auto &st : s.stages) have.insert(st.stage);
        for (const auto *r : kRequired)
        {
            if (std::string(r) == "convert" && kConvertExempt.count(s.model))
                continue;
            if (!have.count(r))
                missing.push_back(std::string(s.model) + ": missing stage " + r);
        }
    }
    std::string joined;
    for (const auto &s : missing) joined += s + "\n";
    INFO("incomplete suite rows:\n" << joined);
    CHECK(missing.empty());
}

// ── Golden coverage: a selfcheck stage that SKIPs is not coverage ────────
//
// The fsi/code/det stages are three target families that SKIP whenever their golden
// file is absent (real weights or golden missing -> SKIP, exit 4). A SKIP is
// indistinguishable from a pass in the ctest summary, so a stage can be fully
// wired - target registered, SKIP_RETURN_CODE 4 set, suite-guard row present - and
// still never execute a single assertion. gemma-4-26b-a4b-it sat exactly there:
// its det golden existed and its fsi/code goldens did not, so two of four parity
// stages were reporting SUCCESS while doing nothing, and nothing in the tree
// noticed. Nine families are still in that state.
//
// Nothing cross-referenced the suite rows against tests/fixtures/{fsi,code,det},
// so this closes that. The debt below is declared rather than tolerated: a family
// with no golden FAILS unless it is listed here, which makes "add a model and forget
// the goldens" impossible to do quietly, and makes the remaining gaps enumerable
// from the test output instead of only from a human diffing two directories.
//
// Two directions, because either one alone leaves a hole:
//   - undeclared gap  -> FAIL. A new family must ship its goldens or declare debt.
//   - stale debt row  -> FAIL. A debt entry whose golden now exists must be deleted,
//     so the list cannot outlive the problem and quietly rot.
TEST_CASE("suite guard: every parity golden is present or declared as debt",
          "[guard]")
{
    static const std::vector<const char *> kGoldenStages = {"fsi", "code", "det"};
    // Declared debt, as "<model>:<stage>". Every entry is a real gap; keeping the
    // list here rather than inferring it means the test output names what is
    // missing instead of merely counting what is present.
    static const std::set<std::string> kGoldenDebt = {
        // Flint: captured from a CPU reference built on the source BF16 tree
        // (tools/flint_oracle.py); removed as each stage goes green.
        "flint:code",
        "flint:det",
        "flint:fsi",
        "deepseek-v4-flash:code",
        "deepseek-v4-flash:det",
        "deepseek-v4-flash:fsi",
        "gigachat3.5-432b-a28b-reasoning:code",
        "gigachat3.5-432b-a28b-reasoning:det",
        "gigachat3.5-432b-a28b-reasoning:fsi",
        "glm-5.2:code",
        "glm-5.2:det",
        "glm-5.2:fsi",
        "kimi-k2.7-code:code",
        "kimi-k2.7-code:det",
        "kimi-k2.7-code:fsi",
        "mimo-v2.6-flash-rl:code",
        "mimo-v2.6-flash-rl:det",
        "mimo-v2.6-flash-rl:fsi",
        "nemotron-3.5-lightning-30b-a3b:code",
        "nemotron-3.5-lightning-30b-a3b:det",
        "nemotron-3.5-lightning-30b-a3b:fsi",
        // FSI/CODE/DET capture is unresolved for the pre-quantized source (HF
        // transformers cannot load a ModelOpt checkpoint without
        // compressed_tensors; the dequantized 30B reference does not answer
        // these synthetic prompts); declared debt, see
        // docs/open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md.
        "nemotron-3.5-lightning-30b-a3b-nvfp4:code",
        "nemotron-3.5-lightning-30b-a3b-nvfp4:det",
        "nemotron-3.5-lightning-30b-a3b-nvfp4:fsi",
        // Mistral-Small-4 FSI/CODE/DET goldens are captured from the source
        // safetensors model, which is not yet converted here; declared debt.
        "mistral-small-4-119b:code",
        "mistral-small-4-119b:det",
        "mistral-small-4-119b:fsi",
    };

    const fs::path root(LSX_SOURCE_DIR);
    const std::vector<SuiteSpec> suites = Suites();

    // The rows a stage is claimed for. A stage absent from a row is a separate
    // guard's business (kRequired above); here we only ask about the goldens of
    // the rows that DO claim the stage, so this test cannot double-report.
    std::vector<std::string> undeclared_gap;
    std::vector<std::string> stale_debt;
    std::set<std::string> seen_debt;

    for (const auto &s : suites)
    {
        std::set<std::string> stages;
        for (const auto &st : s.stages) stages.insert(st.stage);
        for (const auto *g : kGoldenStages)
        {
            if (!stages.count(g)) continue;
            const std::string key = std::string(s.model) + ":" + g;
            const bool present = fs::exists(root / "tests" / "fixtures" / g /
                                            (std::string(s.model) + ".json"));
            if (present)
            {
                if (kGoldenDebt.count(key)) stale_debt.push_back(key);
                continue;
            }
            if (kGoldenDebt.count(key))
            {
                seen_debt.insert(key);
                continue;
            }
            undeclared_gap.push_back(key);
        }
    }

    std::string a, b;
    for (const auto &x : undeclared_gap)
        a += "golden missing and NOT declared as debt (add the golden, or add \"" +
             x + "\" to kGoldenDebt): " + x + "\n";
    for (const auto &x : stale_debt)
        b += "debt row whose golden now exists - delete \"" + x +
             "\" from kGoldenDebt: " + x + "\n";
    INFO("undeclared golden gaps:\n" << a);
    INFO("stale debt rows:\n" << b);
    CHECK(undeclared_gap.empty());
    CHECK(stale_debt.empty());

    // A debt row for a model or stage that no longer exists is dead weight that
    // hides a rename; catching it here keeps the list honest rather than letting
    // it accumulate.
    std::vector<std::string> unknown_debt;
    for (const auto &d : kGoldenDebt)
        if (!seen_debt.count(d)) unknown_debt.push_back(d);
    for (const auto &x : unknown_debt)
        INFO("debt row matching no live gap: " << x);
    CHECK(unknown_debt.empty());
}

// ── Golden provenance: never sell a template golden as model-defined ────────
//
// `tools/{fsi,code,det}_capture.py` has two capture modes, and the golden's
// `source` field records which one ran. A genuine capture runs the SOURCE
// safetensors and records the model's decoded output in each case's `raw`, and
// writes the source it ran as a BARE path (`/mnt/llm_models/models/…`). The
// `--template-golden` stopgap, for a source that cannot run locally, records
// the prompt's Expected Template as `expected`, leaves `raw` empty, and says so
// with a `prompt-template (…)` / `template-derived (…)` prefix.
//
// The parity gate is only evidence of anything if its provenance is honest. A
// template golden labelled as a model-defined one tells an auditor that the
// SOURCE model's own answer was captured when no model ever ran — the exact
// claim the FSI/CODE/DET stages exist to establish. The debt registry cannot
// catch it either: it only asks whether the golden FILE exists, so a committed
// golden satisfies "present or declared debt" whether or not it passes.
//
// The discriminator is `raw`, and it holds across the whole tree: every genuine
// capture carries non-empty `raw` for every case, and every template golden has
// `raw` empty throughout. (A real capture may legitimately AGREE with the
// Expected Template — granite's FSI golden does — so agreement with the prompt
// set is NOT evidence either way; only `raw` is.)
TEST_CASE("suite guard: a golden with no captured raw declares a template source",
          "[guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> mislabelled;

    for (const char *stage : {"fsi", "code", "det"})
    {
        const fs::path dir = root / "tests" / "fixtures" / stage;
        if (!fs::exists(dir)) continue;
        for (const auto &e : fs::directory_iterator(dir))
        {
            if (e.path().extension() != ".json") continue;
            std::ifstream in(e.path());
            REQUIRE(in.good());
            const std::string text((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

            std::smatch src;
            // Custom raw-string delimiter: the pattern contains `*)"` , which
            // would otherwise close a plain R"( ... )" literal early.
            const bool has_src = std::regex_search(
                text, src,
                std::regex(R"RX("source"\s*:\s*"((?:[^"\\]|\\.)*)")RX"));
            REQUIRE(has_src);
            const std::string source = src[1].str();

            // Every `"raw"` key present, and how many of them are empty.
            const std::regex raw_key(R"("raw"\s*:)");
            const std::regex raw_empty(R"("raw"\s*:\s*"")");
            const int total = std::distance(
                std::sregex_iterator(text.begin(), text.end(), raw_key),
                std::sregex_iterator());
            const int empty = std::distance(
                std::sregex_iterator(text.begin(), text.end(), raw_empty),
                std::sregex_iterator());
            if (total == 0 || empty != total) continue;  // a real capture

            const bool declared_template =
                source.rfind("prompt-template", 0) == 0 ||
                source.rfind("template-derived", 0) == 0;
            if (!declared_template)
                mislabelled.push_back(
                    std::string(stage) + "/" + e.path().filename().string() +
                    " has an empty raw in every case (a --template-golden "
                    "capture) but its source does not declare one: \"" +
                    source.substr(0, 80) + "\"");
        }
    }

    for (const auto &x : mislabelled) WARN("mislabelled golden: " << x);
    CHECK(mislabelled.empty());
}
