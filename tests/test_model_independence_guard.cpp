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
//   5. Every family model wires the SHARED lifecycle contract of ModelBase
//      (residency lock, streaming/abort handle, governor lease, context
//      teardown, published result) — see `kLifecycle` below.

#include <catch2/catch_test_macros.hpp>

#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <string_view>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
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
    {"glm-5.3-flash", "glm_moe_flash_constants.h",
     "lsxcommon::glm_moe_flash::frozen"},
    {"deepseek-v4-flash", "deepseek_moe_load.cuh", "lsxcommon::gpu"},
    {"deepseek-v4.1-flash", "deepseek_moe_v41_constants.h",
     "lsxcommon::deepseek_v41"},
    {"deepseek-v4-pro-0813", "deepseek_moe_pro_constants.h",
     "lsxcommon::deepseek_pro"},
    {"gemma-4-26b-a4b-it", "../include/lsxcommon/gemma_moe_load.h",
     "lsxcommon::gemma"},
    {"gemma-4-31b-it", "../include/lsxcommon/gemma_dense_constants.h",
     "lsxcommon::gemma_dense::frozen"},
    {"kimi-k2.7-code", "../include/lsxcommon/kimi_moe_load.h",
     "lsxcommon::kimi"},
    {"qwen3.5-9b", "qwen_dense_model.h", "Qwen3_5_9BModel"},
    {"qwen3.8-27b", "qwen_dense_27b_model.h", "Qwen3_8_27BModel"},
    {"qwen3.5-35b-a3b", "qwen_moe_constants.h", "lsxcommon::qwen_moe::frozen"},
    {"qwen3.8-flash-next", "qwen_moe_exp_constants.h",
     "lsxcommon::qwen_moe_exp::frozen"},
    {"qwen3.6-35b-a3b", "qwen_moe_36_constants.h",
     "lsxcommon::qwen_moe_36::frozen"},
    {"nemotron-3.5-lightning-30b-a3b", "nemotron_moe_constants.h",
     "lsxcommon::nemotron_moe::frozen"},
    {"nemotron-3.5-lightning-30b-a3b-nvfp4", "nemotron_moe_nvfp4_constants.h",
     "lsxcommon::nemotron_nvfp4::frozen"},
    {"gigachat3.5-432b-a28b-reasoning", "gigachat_moe_constants.h",
     "lsxcommon::gigachat::frozen"},
    {"ornith-1.5-35b-a3b", "ornith_moe_constants.h",
     "lsxcommon::ornith::frozen"},
    {"granite-4.2-30b", "../include/lsxcommon/granite_dense_constants.h",
     "lsxcommon::granite::frozen"},
    {"mimo-v2.6-flash-rl", "mimo_moe_constants.h", "lsxcommon::mimo::frozen"},
    {"phi-4", "../include/lsxcommon/phi_dense_constants.h",
     "lsxcommon::phi::frozen"},
    {"alice-foundation-80b-a3b", "alice_moe_constants.h",
     "lsxcommon::alice::frozen"},
    {"xing4.0-29b-a4b", "xing_moe_constants.h", "lsxcommon::xing::frozen"},
    {"mistral-small-4-119b", "mistral_moe_constants.h",
     "lsxcommon::mistral_moe::frozen"},
    {"minimax-m3", "minimax_moe_constants.h", "lsxcommon::minimax::frozen"},
    {"hy-memory-30b-a3b", "hy_moe_constants.h", "lsxcommon::hy::frozen"},
    {"flint", "flint_moe_constants.h", "lsxcommon::flint::frozen"},
    {"ling", "ling_moe_constants.h", "lsxcommon::ling::frozen"},
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
        {"glm_moe_flash_constants.h", "namespace frozen {"},
        {"deepseek_moe_load.cuh", "namespace gpu {"},
        {"deepseek_moe_v41_constants.h", "namespace deepseek_v41 {"},
        {"deepseek_moe_pro_constants.h", "namespace deepseek_pro {"},
        {"../include/lsxcommon/gemma_moe_load.h", "namespace gemma {"},
        {"../include/lsxcommon/gemma_dense_constants.h",
         "namespace frozen {"},
        {"../include/lsxcommon/kimi_moe_load.h", "namespace kimi {"},
        {"qwen_moe_constants.h", "namespace frozen {"},
        {"qwen_moe_exp_constants.h", "namespace frozen {"},
        {"qwen_moe_36_constants.h", "namespace frozen {"},
        {"nemotron_moe_constants.h", "namespace frozen {"},
        {"nemotron_moe_nvfp4_constants.h", "namespace frozen {"},
        {"gigachat_moe_constants.h", "namespace frozen {"},
        {"ornith_moe_constants.h", "namespace frozen {"},
        {"qwen_dense_model.h", "static constexpr int kHiddenSize"},
        {"qwen_dense_27b_model.h", "static constexpr int kHiddenSize"},
        {"../include/lsxcommon/granite_dense_constants.h", "namespace frozen {"},
        {"mimo_moe_constants.h", "namespace frozen {"},
        {"../include/lsxcommon/phi_dense_constants.h", "namespace frozen {"},
        {"alice_moe_constants.h", "namespace frozen {"},
        {"ling_moe_constants.h", "namespace frozen {"},
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

    // The ornith family is independent from every other family: its files must
    // not include qwen/dense model passports, except the generic kernel layer
    // (lsx_attn_inference.*) which the repo documents as shared.
    const std::vector<std::string> ornith_files = {
        "ornith_moe_constants.h", "ornith_moe_model.h",
        "ornith_moe_model.cpp", "ornith_moe_load.cpp",
        "../include/lsxcommon/ornith_moe_load.h",
        "../include/lsxcommon/ornith_moe_inference_pipeline.h",
    };
    const std::vector<std::string> ornith_foreign = {
        "qwen_moe_constants.h", "qwen_moe_36_constants.h",
        "qwen_moe_exp_constants.h", "glm_moe_load.h",
        "deepseek_moe_load.cuh", "gemma_moe_load.h", "kimi_moe_load.h",
    };
    for (const auto &f : ornith_files) {
        const fs::path p =
            f.rfind("../include/", 0) == 0 ? kIncDir / f.substr(11)
                                           : kSrcDir / f;
        const std::string text = ReadFile(p);
        for (const auto &x : ornith_foreign) {
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
        ReadFile(kSrcDir / "lsx_attn_inference.cu");
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
        "deepseek_moe_pro_model.cpp",  // deepseek-v4-pro-0813
        "deepseek_moe_v41_model.cpp",  // deepseek-v4.1-flash
        "nemotron_moe_model.cpp",      // nemotron-3.5-lightning-30b-a3b
        "nemotron_moe_nvfp4_model.cpp",  // nemotron-3.5-lightning-30b-a3b-nvfp4
        "gigachat_moe_model.cpp",      // gigachat3.5-432b-a28b-reasoning
        "ornith_moe_model.cpp",        // ornith-1.5-35b-a3b
        "granite_dense_model.cpp",     // granite-4.2-30b
        "gemma_dense_model.cpp",       // gemma-4-31b-it (dense)
        "phi_dense_model.cpp",         // phi-4
        "mimo_moe_model.cpp",          // mimo-v2.6-flash-rl
        "alice_moe_model.cpp",         // alice-foundation-80b-a3b
        "xing_moe_model.cpp",          // xing4.0-29b-a4b (mHC four-stream)
        "mistral_moe_model.cpp",       // mistral-small-4-119b (MLA + MoE)
        "minimax_moe_model.cpp",       // minimax-m3 (GQA + MSA indexer + MoE)
        "hy_moe_model.cpp",            // hy-memory-30b-a3b (GQA + sigmoid router MoE)
        "flint_moe_model.cpp",         // flint (hybrid sink/SWA/no-RoPE MoE)
        "ling_moe_model.cpp",          // ling (KDA + gated MLA + 512-expert MoE)
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
    // glm's kernels must reach the NVFP4 GEMM through the neutral
    // lsxcommon::lsx_nvfp4 entry point, never through a qwen-family shim.
    // This used to read "no lsx_nvfp4::RunLsxNvfp4Gemm", because the GEMM was
    // then only reachable as the qwen alias RunQwenNvfp4Gemm; that alias is
    // gone (it was a pure passthrough, the same shape as qwen_reduce.cuh), so
    // the rule is now stated directly: no lsxcommon::qwen GEMM entry point.
    for (const char *f : {"glm_moe_flash_inference.cu",
                          "glm_moe_flash_model.cpp"}) {
        const std::string text = ReadFile(kSrcDir / f);
        INFO(f << " must not call a qwen-namespaced GEMM entry point");
        REQUIRE(text.find("qwen::RunLsxNvfp4Gemm") == std::string::npos);
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

TEST_CASE("every family model wires the shared lifecycle contract",
          "[independence][lifecycle]")
{
    // The canonical lifecycle lives in ModelBase (residency lock, streaming
    // handle, governor leases, context teardown, result fill). Each family
    // re-implements its own Load/Infer, so the contract is enforced HERE, per
    // model, on the translation units that make up its generation path: the
    // TU owning the Infer body (a thin leaf that only overrides
    // ValidateGeometry names the shared base instead) plus, when the model
    // delegates the whole run to a device pipeline TU, that pipeline.
    //
    // Naming the pipeline is not optional bookkeeping. Twelve rows listed only
    // the Infer TU while their Emit/abort polls lived in an unlisted .cu, so
    // the marker check ran against code that makes no such decision: it passed
    // families whose generation loop was never read at all. A row may omit a
    // pipeline only when the loop really is host-side - which is the case for
    // nemotron (x2), gigachat, mimo, xing and mistral, whose Infer bodies step
    // the session themselves and therefore poll gs.Aborted() in the same TU
    // that wires gs.
    //
    // Extend this registry in the SAME commit as a new model's Infer.
    struct LifecycleRow {
        const char *model_type;
        std::vector<const char *> path;  // TUs forming the generation path
    };
    const std::vector<LifecycleRow> kLifecycle = {
        {"glm-5.2", {"glm_moe_model.cpp", "glm_moe_inference.cu"}},
        {"glm-5.3-flash",
         {"glm_moe_flash_model.cpp", "glm_moe_flash_inference.cu"}},
        {"deepseek-v4-flash",
         {"deepseek_moe_model.cpp", "deepseek_moe_inference_pipeline.cu"}},
        {"deepseek-v4.1-flash",
         {"deepseek_moe_v41_model.cpp", "deepseek_moe_v41_inference_pipeline.cu"}},
        {"deepseek-v4-pro-0813",
         {"deepseek_moe_pro_model.cpp",
          "deepseek_moe_pro_inference_pipeline.cu"}},
        {"qwen3.5-9b", {"qwen_base_model.cpp",
                        "qwen_dense_inference_pipeline.cu"}},
        {"qwen3.8-27b", {"qwen_base_model.cpp",
                         "qwen_dense_inference_pipeline.cu"}},
        {"qwen3.5-35b-a3b", {"qwen_moe_model.cpp",
                             "qwen_moe_inference_pipeline.cu"}},
        {"qwen3.6-35b-a3b", {"qwen_moe_model.cpp",
                             "qwen_moe_inference_pipeline.cu"}},
        {"qwen3.8-flash-next", {"qwen_moe_exp_model.cpp",
                                "qwen_moe_exp_inference.cu"}},
        {"gemma-4-26b-a4b-it",
         {"gemma_base_model.cpp", "gemma_moe_inference.cu"}},
        {"gemma-4-31b-it",
         {"gemma_dense_model.cpp", "gemma_dense_inference.cu"}},
        {"kimi-k2.7-code", {"kimi_moe_model.cpp", "kimi_moe_inference.cu"}},
        {"ornith-1.5-35b-a3b",
         {"ornith_moe_model.cpp", "ornith_moe_inference_pipeline.cu"}},
        {"granite-4.2-30b",
         {"granite_dense_model.cpp", "granite_dense_inference.cu"}},
        {"phi-4", {"phi_dense_model.cpp", "phi_dense_inference.cu"}},
        {"nemotron-3.5-lightning-30b-a3b", {"nemotron_moe_model.cpp"}},
        {"nemotron-3.5-lightning-30b-a3b-nvfp4",
         {"nemotron_moe_nvfp4_model.cpp", "nemotron_moe_model.cpp"}},
        {"gigachat3.5-432b-a28b-reasoning", {"gigachat_moe_model.cpp"}},
        {"mimo-v2.6-flash-rl", {"mimo_moe_model.cpp"}},
        {"alice-foundation-80b-a3b", {"alice_moe_model.cpp",
                                      "alice_moe_inference_pipeline.cu"}},
        {"xing4.0-29b-a4b", {"xing_moe_model.cpp"}},
        {"mistral-small-4-119b",
         {"mistral_moe_model.cpp", "mistral_moe_inference.cu"}},
        {"minimax-m3",
         {"minimax_moe_model.cpp", "minimax_moe_inference_pipeline.cu"}},
        {"hy-memory-30b-a3b",
         {"hy_moe_model.cpp", "hy_moe_inference_pipeline.cu"}},
        {"flint",
         {"flint_moe_model.cpp", "flint_moe_inference_pipeline.cu"}},
        {"ling", {"ling_moe_model.cpp", "ling_moe_inference.cu"}},
    };
    // marker group -> what it guarantees. EVERY spelling in a group must be
    // absent for the contract to be considered unwired (any-of semantics:
    // a family may publish the result field directly or through the shared
    // runtime::FillInferenceResult).
    struct MarkerGroup {
        const char *id;
        std::vector<const char *> any_of;
    };
    const std::vector<MarkerGroup> kMarkers = {
        {"residency-lock", {"RunUnderResidencyLock"}},
        {"streaming",
         // An actual INVOCATION — a GenerationStream that is only accepted
         // and then discarded (`(void)gs;`) does not satisfy the contract.
         {"gs->Emit", "GenerationStream gs", "token_callback("}},
        // An actual POLL, not the field assignment. `gs.abort_flag =
        // req.abort_flag;` only proves the flag was handed to the handle; the
        // contract is that the generation loop READS it, so that an abort
        // request stops the request instead of running to max_gen_tokens. That
        // is why `abort_flag` (the previous spelling) was not enough: every
        // family satisfies it by construction, including one that never
        // polled - which is exactly how qwen3.8-flash-next shipped with a
        // dead abort path under a green guard.
        {"abort", {"gs.Aborted()", "gs->Aborted()"}},
        {"context-teardown", {"TeardownContext"}},
        {"governor-lease", {"AcquirePermanentWeights", "AcquireExpertCacheLease"}},
        {"output-text", {"DecodeGeneratedText", "output_text"}},
        {"resident-bytes", {"weight_resident_bytes", "FillInferenceResult"}},
    };
    // Accumulate every violation so one run reports the whole gap set.
    std::vector<std::string> violations;
    absl::btree_set<std::string> claimed;
    for (const auto &row : kLifecycle) {
        std::string path;  // every TU of the generation path, concatenated
        for (const char *tu : row.path) {
            const std::string text = ReadFile(kSrcDir / tu);
            if (text.empty()) {
                violations.push_back(
                    absl::StrCat(row.model_type, ": cannot read ", tu));
                continue;
            }
            path += text;
        }
        if (path.empty()) continue;
        claimed.insert(row.model_type);
        for (const auto &g : kMarkers) {
            bool found = false;
            for (const char *needle : g.any_of) {
                if (path.find(needle) != std::string::npos) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                violations.push_back(
                    absl::StrCat(row.model_type, " (",
                                 absl::StrJoin(row.path, "+"),
                                 "): unwired lifecycle contract '", g.id,
                                 "' — expected one of {",
                                 absl::StrJoin(g.any_of, ", "), "}"));
            }
        }
    }
    // Every ingested model must be claimed by the registry itself.
    for (const auto &p : kPassports) {
        if (claimed.count(p.model_type) != 1) {
            violations.push_back(
                absl::StrCat(p.model_type, ": no kLifecycle registry row"));
        }
    }
    // Incremental-session families keep persistent decode state (the KV cache
    // plus the generated tail). Creating, resetting or publishing that state
    // must happen INSIDE the lock scope that owns the generation, so the state
    // is created, decoded through and published together.
    //
    // What this check does NOT claim: that the lock makes concurrent Infer
    // calls safe. RunUnderResidencyLock takes a SHARED lock, so two Infer calls
    // on one instance both hold it and neither excludes the other (E1b). This
    // is a structural check on ordering and scope only: the session mutations
    // must not sit outside the scope that decodes through them. It is scoped to
    // the Infer body - the lock must be opened before the first session
    // mutation there (a Load-time lock elsewhere in the file does not protect
    // Infer state).
    for (const auto &row : kLifecycle) {
        std::string text;
        for (const char *tu : row.path) text += ReadFile(kSrcDir / tu);
        std::size_t body = text.find("::Infer(const ModelRequest");
        const std::size_t shared_body =
            text.find("::RunInferShared(const ModelRequest");
        if (shared_body != std::string::npos &&
            (body == std::string::npos || shared_body < body))
            body = shared_body;  // shared base carries the real generation body
        if (body == std::string::npos) continue;  // not an Infer owner
        const std::size_t lock_at = text.find("RunUnderResidencyLock", body);
        if (lock_at == std::string::npos) continue;  // reported above
        for (const char *stateful : {"DecideSession", "kv_.reset()"}) {
            const std::size_t at = text.find(stateful, body);
            if (at == std::string::npos) continue;
            if (at < lock_at) {
                violations.push_back(absl::StrCat(
                    row.model_type, " (", absl::StrJoin(row.path, "+"),
                    "): session state '", stateful,
                    "' is mutated before the residency lock is taken — the "
                    "session state it protects must be created, decoded through "
                    "and published inside the one lock scope (note: the shared "
                    "lock orders this against Unload(), not against a second "
                    "Infer — see OPEN_TASKS E1b)"));
            }
        }
    }
    INFO("lifecycle contract violations:\n  " << absl::StrJoin(violations, "\n  "));
    REQUIRE(violations.empty());
}

TEST_CASE("broken fused-decode attention path is removed", "[independence]")
{
    // KNOWN-BROKEN (diverged on the real Qwen3.5-9B checkpoint; unwired from
    // generation in 19e8d4f) and test-only since — deleted by the 2026-09
    // model-independence audit. Reintroduction requires a full-layer parity
    // re-validation on the real checkpoint (see AGENTS.md selfcheck rules).
    const std::string header = ReadFile(kIncDir / "lsx_attn_inference.h");
    REQUIRE(header.find("RunQwenFusedDecodeAttnKernel") == std::string::npos);
    const std::string tu = ReadFile(kSrcDir / "lsx_attn_inference.cu");
    REQUIRE(tu.find("RunQwenFusedDecodeAttnKernel") == std::string::npos);
    REQUIRE(tu.find("RunQwenFusedQkv") == std::string::npos);
    REQUIRE(tu.find("RunQwenFusedOProj") == std::string::npos);
}

// The KV cache layout is a MEASURED per-family decision, not an inherited default,
// so this guard does not mandate a layout. What it mandates is that the layout is
// expressed through the shared geometry - a hand-rolled (t*heads + h)*dim is how
// pos-major spread to nine files in the first place, and how a family ends up
// unable to try the other layout without rewriting seven call sites.
//
// Measured on Xing-4.0 (docs/OPEN_TASKS.md): head-major reads K/V 6-10% faster in
// isolation at ctx 4096/8191 and is 2-6% SLOWER end to end at every context. So
// pos-major ships. This test therefore checks MECHANISM, never layout.
TEST_CASE("KV cache layout goes through the shared geometry", "[guard][fast]")
{
    // Families exempt from the shared geometry, each with the reason. Both are
    // "not a choice we can make here", not "we forgot":
    //   nemotron - head-major by construction, its attention indexes
    //               (hk * cache_stride + t) * head_dim already
    //   glm      - one THREAD per head, not one block: a warp-of-heads has a
    //               different coalescing trade-off and needs its own analysis
    // Exemptions, each with its reason - a bare filename list would rot into a
    // place where "we forgot" looks the same as "we decided". Two kinds:
    //   structural  - the layout is not a choice this family can make
    //   pending     - still hand-rolled; migrate and delete the row
    static const std::vector<std::pair<std::string, std::string>> kKvLayoutExempt = {
        {"nemotron_moe_inference_attn.cu",
         "structural: head-major by construction, indexes (hk*cache_stride + t)*head_dim"},
        {"glm_moe_inference.cu",
         "structural: one THREAD per head, not one block - a warp-of-heads has a different coalescing trade-off and needs its own analysis"},
        {"glm_moe_flash_inference.cu",
         "structural: same one-thread-per-head blocking as glm"},
        {"lsx_attn_inference.cu",
         "pending: migrates to the shared geometry when its decode attention is next touched"},
        {"gemma_moe_inference.cu",
         "pending: migrates to the shared geometry when its decode attention is next touched"},
        {"mimo_moe_inference_attn.cu",
         "pending: migrates to the shared geometry when its decode attention is next touched"},
        {"gigachat_moe_inference_attn.cu",
         "pending: migrates to the shared geometry when its decode attention is next touched"},
        {"deepseek_moe_v41_inference.cu",
         "pending: migrates to the shared geometry when its decode attention is next touched"},
    };
    // The pos-major idiom: `* heads + h` in an index. Written with character
    // classes, not `\*` and `\b`: std::regex uses the ECMAScript grammar, where
    // these escapes are spelled differently from PCRE, and a pattern that matches
    // in a Python sanity check can silently match NOTHING here - which is how this
    // test first "passed" with 26 files scanned and zero findings.
    static const std::regex kPosMajorIdiom(
        R"([*][ \t]*(kv_)?heads[ \t]*[+][ \t]*(h|head|hh)([^a-zA-Z0-9_]|$))");
    std::error_code ec;
    std::vector<std::string> violations;
    std::size_t scanned = 0;
    for (const auto &entry : fs::directory_iterator(kSrcDir, ec))
    {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.find("_inference") == std::string::npos) continue;
        if (name.size() < 3 || name.substr(name.size() - 3) != ".cu") continue;
        const std::string text = ReadFile(entry.path());
        if (text.empty()) continue;
        if (text.find("KvCacheGeometry") != std::string::npos) continue;
        ++scanned;
        auto ex = std::find_if(kKvLayoutExempt.begin(), kKvLayoutExempt.end(),
                               [&](const auto &e) { return e.first == name; });
        const bool exempt = ex != kKvLayoutExempt.end();
        if (exempt && ex->second.empty())
            violations.push_back(name + ": exempt with no reason recorded");
        for (std::smatch m; std::regex_search(text, m, kPosMajorIdiom);)
        {
            if (!exempt)
            {
                violations.push_back(name + ": hand-rolled KV stride '" +
                                     m.str() +
                                     "' - use KvCacheGeometry from "
                                     "lsx_kv_cache.h");
                break;
            }
            break;
        }
    }
    INFO("scanned " << scanned << " inference .cu files without KvCacheGeometry");
    INFO("violations:\n" << [&] {
        std::string j;
        for (const auto &v : violations) j += v + "\n";
        return j;
    }());
    CHECK(violations.empty());
}

TEST_CASE("every family threads the request sampling policy",
          "[independence]")
{
    // E4 was "temperature/top_p accepted and silently ignored". The fix is
    // only as good as its coverage, so this scans every model TU that defines
    // Infer(const ModelRequest&) and requires it to derive its sampler from
    // req.sampling -- directly, or through the one shared RunInferShared path
    // (qwen MoE / nemotron), which is checked to do it.
    int checked = 0;
    for (const auto &entry : fs::directory_iterator(kSrcDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".cpp")
            continue;
        const std::string text = ReadFile(entry.path());
        if (text.find("::Infer(const ModelRequest &req)") == std::string::npos)
            continue;
        const bool direct =
            text.find("SamplerFor(req.sampling)") != std::string::npos;
        const bool delegated =
            text.find("RunInferShared(req)") != std::string::npos;
        INFO(entry.path().filename()
             << " must sample from req.sampling (directly or via the shared "
                "RunInferShared)");
        REQUIRE((direct || delegated));
        ++checked;
    }
    INFO("scanned " << checked << " Infer definitions");
    REQUIRE(checked >= 20);

    for (const char *base : {"qwen_moe_model.cpp", "nemotron_moe_model.cpp"}) {
        const std::string text = ReadFile(kSrcDir / base);
        INFO(base << " is a RunInferShared base and must thread req.sampling");
        REQUIRE(text.find("SamplerFor(req.sampling)") != std::string::npos);
    }
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

TEST_CASE("the routed-MoE inference layer is model-neutral in name and deps",
          "[independence]")
{
    // The routed-MoE kernels are geometry-parameterised (hidden size, top-k
    // and the expert count are runtime parameters) and are consumed by the
    // Gemma, GLM-flash and Qwen-MoE-exp families, so they carry the neutral
    // lsx_ name.
    //
    // "Neutral" is scoped to MODEL dependencies. Two qwen-named things are
    // deliberately still allowed, both of them repository-wide shared layers
    // rather than qwen state:
    //   - lsx_attn_inference.h / namespace qwen  -- the designated generic kernel
    //     layer (17 dependents; AGENTS.md has families reuse it);
    //   - qwen_moe_expert_cache.h -- a thin alias over the neutral
    //     gpu::ExpertDeviceCache.
    // What it must NOT do is reach into a model's own weights or passports.
    const fs::path hdr = kIncDir / "lsx_moe_inference.h";
    const fs::path cu = kSrcDir / "lsx_moe_inference.cu";
    REQUIRE(fs::exists(hdr));
    REQUIRE(fs::exists(cu));

    for (const fs::path &p : {hdr, cu}) {
        const std::string text = ReadFile(p);
        REQUIRE_FALSE(text.empty());

        // No family-named symbol may survive in a neutral file.
        INFO(p.filename() << " must not define any QwenMoe* symbol");
        REQUIRE(text.find("QwenMoe") == std::string::npos);

        // No model-owned header may be pulled in.
        static const char *kModelOwned[] = {
            "qwen_moe_load.h",         "qwen_moe_model.h",
            "qwen_moe_convert.h",      "qwen_moe_constants.h",
            "qwen_moe_inference_pipeline.h", "qwen_inference_load.h"};
        for (const char *h : kModelOwned) {
            INFO(p.filename() << " must not include the model header " << h);
            REQUIRE(text.find(h) == std::string::npos);
        }
    }
}

TEST_CASE("the attention kernel layer is model-neutral in name and deps",
          "[independence]")
{
    // Geometry-neutral in content (every dimension is a runtime parameter) and
    // consumed by Qwen MoE, Qwen MoE Exp, Ornith and Alice, so it carries the
    // neutral lsx_ name. Scoped the same way as the routed-MoE guard: no model
    // dependency, no family-named symbol of its own. The shared fast-math
    // aliases and the KV cache / result structs keep their qwen names on
    // purpose -- they are held by three families, so renaming them would make
    // a non-qwen family depend on a qwen-owned header.
    const fs::path hdr = kIncDir / "lsx_attn_inference.h";
    const fs::path cu = kSrcDir / "lsx_attn_inference.cu";
    REQUIRE(fs::exists(hdr));
    REQUIRE(fs::exists(cu));

    static const char *kAllowed[] = {
        "QwenTcFastMath",  "SetQwenTcFastMath", "QwenSeq1Gemv",
        "SetQwenSeq1Gemv", "QwenGemmLarge",     "QwenGemmLargeActive",
            "QwenFastMathScope"};

    for (const fs::path &p : {hdr, cu}) {
        const std::string text = ReadFile(p);
        REQUIRE_FALSE(text.empty());

        // Comments are prose, not symbols: citing the upstream model whose
        // numerics a kernel reproduces is exactly the provenance we want.
        std::string stripped;
        for (const std::string_view line :
             absl::StrSplit(text, absl::ByChar('\n'))) {
            const size_t c = line.find("//");
            stripped.append(c == std::string::npos ? line : line.substr(0, c));
            stripped.push_back('\n');
        }
        for (const char *a : kAllowed) {
            const size_t n = std::strlen(a);
            for (size_t at = stripped.find(a); at != std::string::npos;
                 at = stripped.find(a, at + n))
                stripped.replace(at, n, std::string(n, '_'));
        }
        INFO(p.filename() << " must not define a family-named symbol");
        REQUIRE(stripped.find("Qwen") == std::string::npos);

        static const char *kModelOwned[] = {
            "qwen_dense_convert.h", "qwen_inference_load.h",
            "qwen_dense_inference_pipeline.h",    "qwen_moe_load.h"};
        for (const char *h : kModelOwned) {
            INFO(p.filename() << " must not include the model header " << h);
            REQUIRE(text.find(h) == std::string::npos);
        }
    }
}
