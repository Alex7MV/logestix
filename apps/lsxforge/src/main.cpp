#include "lsxcommon/converter.h"
#include "lsxcommon/arrow_tree_verify.h"
#include "lsxcommon/glm_moe_flash_convert.h"
#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/generation_cli.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/gemma_dense_convert.h"
#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/alice_moe_convert.h"
#include "lsxcommon/kimi_moe_k3_convert.h"
#include "lsxcommon/kimi_moe_pro_convert.h"
#include "lsxcommon/inkling_moe_convert.h"
#include "lsxcommon/inkling_moe_base_convert.h"
#include "lsxcommon/inkling_moe_loop_convert.h"
#include "lsxcommon/muse_dense_convert.h"
#include "lsxcommon/xing_moe_convert.h"
#include "lsxcommon/mistral_moe_convert.h"
#include "lsxcommon/mistral_moe_large4_convert.h"
#include "lsxcommon/hy_moe_convert.h"
#include "lsxcommon/hy_moe_simulmt_convert.h"
#include "lsxcommon/step_moe_convert.h"
#include "lsxcommon/beam_moe_convert.h"
#include "lsxcommon/flint_moe_convert.h"
#include "lsxcommon/ling_moe_convert.h"
#include "lsxcommon/minimax_moe_convert.h"
#include "lsxcommon/minimax_moe_flash_convert.h"

#include <simdjson.h>
#include "lsxcommon/mimo_moe_convert.h"
#include "lsxcommon/nemotron_moe_convert.h"
#include "lsxcommon/nemotron_moe_nvfp4_convert.h"
#include "lsxcommon/nemotron_moe_ultra_convert.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/qwen_moe_36_convert.h"
#include "lsxcommon/ornith_moe_convert.h"
#include "lsxcommon/qwen_moe_exp_convert.h"
#include "lsxcommon/glm_moe_convert.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/qwen_dense_convert.h"
#include "lsxcommon/granite_dense_convert.h"
#include "lsxcommon/granite_dense_switch_convert.h"
#include "lsxcommon/phi_dense_convert.h"
#include "lsxcommon/dory_dense_convert.h"
#include "lsxcommon/deepseek_moe_repack.h"
#include "lsxcommon/deepseek_moe_v41_convert.h"
#include "lsxcommon/deepseek_moe_pro_convert.h"
#include "lsxcommon/log.h"
#include "cli.h"
#include <arrow/api.h>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

ABSL_FLAG(std::string, input, "", "input model directory");
ABSL_FLAG(std::string, output, "converted_model", "output directory");
ABSL_FLAG(bool, verify, false,
          "verify the Arrow artifacts of --input (or --output) and exit; "
          "post-conversion step (LSX_ARROW_AUDIT_{FULL,MAX_BYTES} apply)");
ABSL_FLAG(bool, repack_shared, false, "re-pack shared experts into shared_experts.arrow");
ABSL_FLAG(bool, repack_experts, false, "re-pack routed experts (fp4 passthrough)");
ABSL_FLAG(std::string, tokenizer, "", "build tokenizer Arrow files from tokenizer.json");
ABSL_FLAG(bool, glm, false, "GLM-5.2 conversion pipeline");
ABSL_FLAG(bool, kimi, false, "Kimi-K2.7-Code conversion pipeline");
ABSL_FLAG(bool, qwen, false, "Qwen3.5 conversion pipeline");
ABSL_FLAG(bool, qwen35moe, false, "Qwen3.5-35B-A3B MoE conversion pipeline");
ABSL_FLAG(bool, qwen36moe, false, "Qwen3.6-35B-A3B MoE conversion pipeline");
ABSL_FLAG(bool, ornith, false,
          "Ornith-1.5-35B-A3B conversion pipeline (MLX or transformers "
          "source layout)");
ABSL_FLAG(bool, qwen38flashnext, false,
          "Qwen3.8-Flash-Next (qwen4_exp) conversion pipeline");
ABSL_FLAG(bool, gemma4, false, "Gemma-4-26B-A4B (MoE) conversion pipeline");
ABSL_FLAG(bool, gemma4dense, false,
          "Gemma-4-31B-it (dense) conversion pipeline");
ABSL_FLAG(bool, nemotron, false,
          "Nemotron-3.5-Lightning-30B-A3B (NemotronH hybrid) conversion pipeline");
ABSL_FLAG(std::string, layers, "",
          "Layer window for resumable conversion, \"first:last\" (inclusive). "
          "Only the --nemotron_ultra pipeline uses it.");
ABSL_FLAG(bool, nemotron_ultra, false,
          "NVIDIA-Nemotron-3-Ultra-550B-A55B pipeline (BF16 source -> unified "
          "logestix layout, logestix-nemotron-ultra-nvfp4)");
ABSL_FLAG(bool, nemotron_nvfp4, false,
          "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (pre-quantized "
          "ModelOpt W4A16_NVFP4 source) conversion pipeline");
ABSL_FLAG(bool, gigachat, false,
          "GigaChat3.5-432B-A28B-Reasoning (hybrid MLA+GDN+MoE) conversion pipeline");
ABSL_FLAG(bool, glmflash, false,
          "GLM-5.3-Flash (glm5_next) conversion pipeline");
ABSL_FLAG(bool, granite, false,
          "Granite-4.2-30B (dense) conversion pipeline");
ABSL_FLAG(bool, granite_switch, false,
          "Granite-Switch 4.1 (dense + aLoRA adapters) conversion pipeline");
ABSL_FLAG(bool, phi, false, "phi-4 (dense) conversion pipeline");
ABSL_FLAG(bool, dory, false,
          "NVIDIA-Nemotron-Labs-Dory1.0-4B (looped nGPT) conversion "
          "pipeline");
ABSL_FLAG(bool, mimo, false,
          "MiMo-V2.6-Flash-RL (hybrid SWA/full sparse MoE) conversion pipeline");
ABSL_FLAG(bool, kimi_k3, false,
          "Kimi-K3 (KDA + gated NoPE MLA + AttnRes + Stable LatentMoE) "
          "conversion pipeline");
ABSL_FLAG(bool, kimi_k31_pro, false,
          "Kimi-K3.1-Pro (Kimi-K3 tower + Gated DSA Lightning Indexer) "
          "conversion pipeline");
ABSL_FLAG(bool, inkling, false,
          "Inkling-2-Small (hybrid sliding/full + AttnRes + relative-bias MoE, "
          "mixed NVFP4/MXFP8 expert source) conversion pipeline");
ABSL_FLAG(bool, inkling_base, false,
          "Inkling base (hybrid sliding/full + relative-bias MoE with 2 "
          "shared-sink experts, pure-BF16 source) conversion pipeline");
ABSL_FLAG(bool, inkling_loop, false,
          "Inkling-2-Mini (looped inkling2_loop: 24 layers, middle 12 x 3 "
          "passes, MessageGate controllers, mixed NVFP4/MXFP8 expert source) "
          "conversion pipeline");
ABSL_FLAG(bool, muse, false,
          "Muse-Glimmer-30B (dense gated attention, NoPE full layers, "
          "centered norms, pure-BF16 source) conversion pipeline");
ABSL_FLAG(bool, alice, false,
          "AliceAI-Foundation-80B-A3B-Base (KDA + gated attention sparse MoE) "
          "conversion pipeline");
ABSL_FLAG(bool, xing, false,
          "Xing4.0-29B-A4B (mHC + MLA + MTP sparse MoE) conversion pipeline");
ABSL_FLAG(bool, minimax, false,
          "convert MiniMax-M3 (vision tower stripped) to logestix-minimax-nvfp4");
ABSL_FLAG(bool, minimax_flash, false,
          "convert MiniMax-M3.1-Flash (MXFP8 dense + NVFP4 experts, MSA "
          "throughout) to logestix-minimax-nvfp4");
ABSL_FLAG(bool, flint, false,
          "Recursive Flint (hybrid sink/SWA/no-RoPE MoE) conversion pipeline");
ABSL_FLAG(bool, ling, false,
          "Ling-3.1-flash (63 KDA + 9 gated MLA + 512-expert MoE) conversion "
          "pipeline");
ABSL_FLAG(bool, beam, false,
          "Beam-1.0-501B-A23B (gated GQA SWA/NoPE + 256-expert MoE) conversion "
          "pipeline");
ABSL_FLAG(bool, hy, false,
          "convert Hy-Memory-30B-A3B (GQA + sigmoid-router MoE) to logestix-hy-nvfp4");
ABSL_FLAG(bool, hy_simulmt, false,
          "convert Hy-SimulMT-30B-A3B (same text backbone, audio tower "
          "stripped) to logestix-hy-simulmt-nvfp4");
ABSL_FLAG(bool, step, false,
          "convert Step-5-Preview (GQA + DSA + 352-expert MoE) to "
          "logestix-step5-nvfp4");
ABSL_FLAG(bool, mistral, false,
          "Mistral-Small-4-119B-2603 (MLA + 128-expert MoE) conversion pipeline");
ABSL_FLAG(bool, mistral_large4, false,
          "Mistral-Large-4-1T-A52B (MLA + DSA indexer + KV sharing + "
          "128-expert MoE) conversion pipeline");
ABSL_FLAG(bool, vram_recompute, false,
          "recompute the vram block of a converted model.index.json "
          "(--input = source safetensors dir, --output = converted root)");
// DeepSeek routed-expert NVFP4 repack: logestix-v1 (E8M0 per-32 expert
// scales) -> logestix-v1-nvfp4 (pow2-E4M3 per-16 + folded global_scale).
// --input = source vault, --output = destination vault (same layout).
ABSL_FLAG(bool, deepseek_nvfp4, false,
          "repack DeepSeek routed experts to NVFP4 (logestix-v1 -> v1-nvfp4)");
// DeepSeek-V4.1-Flash ingestion: HF safetensors -> logestix-v41-nvfp4
// (dense FP8 passthrough, routed experts E8M0/32 -> NVFP4, vision/engram/MTP
// excluded with a counted record).
ABSL_FLAG(bool, deepseek_v41, false,
          "convert DeepSeek-V4.1-Flash to the logestix-v41-nvfp4 layout");
// DeepSeek-V4-Pro-0813 ingestion: HF safetensors -> logestix-v1-nvfp4 with the
// frozen 61/7168/129280/384 passport, DSpark/MTP ballast stripped and counted.
ABSL_FLAG(bool, deepseek_pro, false,
          "convert DeepSeek-V4-Pro-0813 (DSpark/MTP stripped) to "
          "logestix-v1-nvfp4");
// Generation path execution option: routes the resolved prompt straight into
// the tokenizer encode channel and the single end-to-end cudaStream_t
// Prefill->Decode autogeneration sequence loop (IModel::Infer), bypassing the
// standalone --selfcheck truncation logic and the --benchmark dummy-token
// layers. A bare --run/-r (no value) falls back to the default blueprint
// prompt via lsxcommon::cli.
ABSL_FLAG(std::string, run, "", "generation prompt (--run \"PROMPT\" / -r \"PROMPT\")");

int main(int argc, char* argv[]) {
    // Canonicalize the generation-path flag (-r/--run, bare -> default prompt)
    // before absl::ParseCommandLine consumes argv.
    auto normalized = lsxcommon::cli::NormalizeGenerationArgv(argc, argv);
    std::vector<char*> argv_ptrs;
    argv_ptrs.reserve(normalized.size());
    for (auto& arg : normalized) argv_ptrs.push_back(arg.data());
    absl::SetProgramUsageMessage(
        "lsxforge — LogestiX data preprocessing (NVFP4 conversion, tokenizer, generation)");
    absl::ParseCommandLine(static_cast<int>(argv_ptrs.size()), argv_ptrs.data());

    std::string input_path  = absl::GetFlag(FLAGS_input);
    std::string output_path = absl::GetFlag(FLAGS_output);
    bool repack_shared = absl::GetFlag(FLAGS_repack_shared);
    bool repack_experts = absl::GetFlag(FLAGS_repack_experts);
    if (!absl::GetFlag(FLAGS_tokenizer).empty())
        input_path = absl::GetFlag(FLAGS_tokenizer);
    bool glm = absl::GetFlag(FLAGS_glm);
    bool kimi = absl::GetFlag(FLAGS_kimi);
    bool qwen = absl::GetFlag(FLAGS_qwen);
    bool qwen35moe = absl::GetFlag(FLAGS_qwen35moe);
    bool qwen36moe = absl::GetFlag(FLAGS_qwen36moe);
    bool ornith = absl::GetFlag(FLAGS_ornith);
    bool qwen38flashnext = absl::GetFlag(FLAGS_qwen38flashnext);
    bool gemma4 = absl::GetFlag(FLAGS_gemma4);
    bool gemma4dense = absl::GetFlag(FLAGS_gemma4dense);
    bool nemotron = absl::GetFlag(FLAGS_nemotron);
    bool nemotron_nvfp4 = absl::GetFlag(FLAGS_nemotron_nvfp4);
    bool nemotron_ultra = absl::GetFlag(FLAGS_nemotron_ultra);
    const std::string layers = absl::GetFlag(FLAGS_layers);
    bool gigachat = absl::GetFlag(FLAGS_gigachat);
    bool glmflash = absl::GetFlag(FLAGS_glmflash);
    bool granite = absl::GetFlag(FLAGS_granite);
    bool granite_switch = absl::GetFlag(FLAGS_granite_switch);
    bool phi = absl::GetFlag(FLAGS_phi);
    bool dory = absl::GetFlag(FLAGS_dory);
    bool mimo = absl::GetFlag(FLAGS_mimo);
    bool kimi_k3 = absl::GetFlag(FLAGS_kimi_k3);
    bool kimi_k31_pro = absl::GetFlag(FLAGS_kimi_k31_pro);
    bool inkling = absl::GetFlag(FLAGS_inkling);
    bool inkling_base = absl::GetFlag(FLAGS_inkling_base);
    bool inkling_loop = absl::GetFlag(FLAGS_inkling_loop);
    bool muse = absl::GetFlag(FLAGS_muse);
    bool alice = absl::GetFlag(FLAGS_alice);
    bool xing = absl::GetFlag(FLAGS_xing);
    bool mistral = absl::GetFlag(FLAGS_mistral);
    bool mistral_large4 = absl::GetFlag(FLAGS_mistral_large4);
    bool minimax = absl::GetFlag(FLAGS_minimax);
    bool minimax_flash = absl::GetFlag(FLAGS_minimax_flash);
    bool hy = absl::GetFlag(FLAGS_hy);
    bool hy_simulmt = absl::GetFlag(FLAGS_hy_simulmt);
    bool flint = absl::GetFlag(FLAGS_flint);
    bool ling = absl::GetFlag(FLAGS_ling);
    bool beam = absl::GetFlag(FLAGS_beam);
    bool step = absl::GetFlag(FLAGS_step);
    const std::string run_prompt = lsxcommon::cli::ResolveGenerationPrompt(absl::GetFlag(FLAGS_run));

    // Post-conversion verification: the same shared verifier the audit test
    // uses. Runs before pipeline resolution so it is a standalone mode:
    //   lsxforge --verify --input <converted model root>
    if (absl::GetFlag(FLAGS_verify)) {
        lsxcommon::Core::Initialize("lsxforge_runtime.log");
        const std::string target = !input_path.empty() ? input_path : output_path;
        std::filesystem::path tree(target);
        if (tree.empty() || !std::filesystem::exists(tree)) {
            lsxcommon::log::error("lsxforge --verify: no such path: " + target);
            lsxcommon::Core::Shutdown();
            return 2;
        }
        if (std::filesystem::exists(tree / "logestix")) tree /= "logestix";
        lsxcommon::arrow_check::TreeVerifyOptions opts;
        if (const char *f = std::getenv("LSX_ARROW_AUDIT_FULL"))
            opts.full = f[0] == '1';
        if (const char *b = std::getenv("LSX_ARROW_AUDIT_MAX_BYTES")) {
            const long long v = std::atoll(b);
            if (v > 0) opts.budget_bytes = v;
        }
        const auto st = lsxcommon::arrow_check::VerifyArrowTree(tree, opts);
        lsxcommon::log::info(absl::StrCat(
            "arrow verify: ", tree.string(), " files=", st.files,
            " footer_ok=", st.footer_ok, " deep=", st.deep, " deep_MiB=",
            st.deep_bytes >> 20, " problems=", st.problems.size()));
        for (const auto &msg : st.problems)
            lsxcommon::log::error("arrow verify: " + msg);
        lsxcommon::Core::Shutdown();
        return st.problems.empty() ? 0 : 1;
    }

    // The flag-to-pipeline decision lives in cli.h (ResolvePipeline) so it is
    // unit-testable; this function only executes the resolved pipeline. The
    // precedence this switch relies on is pinned by tests/test_lsxforge_cli.cpp.
    lsxforge::Flags flags;
    flags.input = input_path;
    flags.output = output_path;
    flags.tokenizer = absl::GetFlag(FLAGS_tokenizer);
    flags.repack_shared = repack_shared;
    flags.repack_experts = repack_experts;
    flags.vram_recompute = absl::GetFlag(FLAGS_vram_recompute);
    flags.deepseek_nvfp4 = absl::GetFlag(FLAGS_deepseek_nvfp4);
    flags.deepseek_v41 = absl::GetFlag(FLAGS_deepseek_v41);
    flags.deepseek_pro = absl::GetFlag(FLAGS_deepseek_pro);
    flags.run = !absl::GetFlag(FLAGS_run).empty();
    flags.glm = glm;
    flags.kimi = kimi;
    flags.qwen = qwen;
    flags.qwen35moe = qwen35moe;
    flags.qwen36moe = qwen36moe;
    flags.ornith = ornith;
    flags.qwen38flashnext = qwen38flashnext;
    flags.gemma4 = gemma4;
    flags.gemma4dense = gemma4dense;
    flags.nemotron = nemotron;
    flags.nemotron_nvfp4 = nemotron_nvfp4;
    flags.nemotron_ultra = nemotron_ultra;
    if (!layers.empty() && !nemotron_ultra) {
        lsxcommon::log::error(
            "--layers is only meaningful with --nemotron_ultra (a partial tree "
            "from another converter would be unloadable)");
        return 2;
    }
    flags.gigachat = gigachat;
    flags.glmflash = glmflash;
    flags.granite = granite;
    flags.granite_switch = granite_switch;
    flags.phi = phi;
    flags.dory = dory;
    flags.mimo = mimo;
    flags.kimi_k3 = kimi_k3;
    flags.kimi_k31_pro = kimi_k31_pro;
    flags.inkling = inkling;
    flags.inkling_base = inkling_base;
    flags.inkling_loop = inkling_loop;
    flags.muse = muse;
    flags.alice = alice;
    flags.xing = xing;
    flags.mistral = mistral;
    flags.mistral_large4 = mistral_large4;
    flags.minimax = minimax;
    flags.minimax_flash = minimax_flash;
    flags.hy = hy;
    flags.hy_simulmt = hy_simulmt;
    flags.flint = flint;
    flags.ling = ling;
    flags.beam = beam;
    flags.step = step;
    const lsxforge::Resolution resolution = lsxforge::ResolvePipeline(flags);
    if (resolution.pipeline == lsxforge::Pipeline::kError) {
        lsxcommon::Core::Initialize("lsxforge_runtime.log");
        lsxcommon::log::error("lsxforge: " + resolution.error);
        lsxcommon::Core::Shutdown();
        return 2;
    }

    if (resolution.pipeline == lsxforge::Pipeline::kVersion) {
        lsxcommon::Core::Initialize("lsxforge_runtime.log");
        lsxcommon::version();
        lsxcommon::log_gpu_info();
        lsxcommon::Core::Shutdown();
        return 0;
    }

    lsxcommon::Core::Initialize("lsxforge_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();

    // Generation path execution option. Resolved prompt (or the default
    // blueprint prompt) is encoded through the model tokenizer and handed to
    // the single end-to-end cudaStream_t Prefill->Decode autogeneration loop.
    if (resolution.pipeline == lsxforge::Pipeline::kGenerate) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = input_path;
        args.prompt = run_prompt;
        args.max_gen_tokens = 64;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }

    lsxcommon::log::info(absl::StrFormat("Converting model: %s -> %s", input_path, output_path));
    switch (resolution.pipeline) {
    case lsxforge::Pipeline::kVramRecompute: {
        // Detect the converted format from the index and dispatch to the
        // matching family recompute.
        namespace fs = std::filesystem;
        auto idx = fs::path(output_path) / "logestix" / "model.index.json";
        std::ifstream f(idx);
        if (!f) {
            lsxcommon::log::error(absl::StrFormat("model.index.json not found under %s", output_path));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        arrow::Status st = arrow::Status::Invalid("unknown index format");
        // GLM-5.3-Flash shares the "logestix-glm" prefix with GLM-5.2 but not
        // its geometry (hybrid KDA + DSA MLA): the 5.2 estimate reads
        // kv_a_proj_with_mqa as the KV width and under-reports by ~28x, so the
        // flash tag has to win the dispatch.
        if (json.find("logestix-glm5next") != std::string::npos)
            st = lsxcommon::glm_moe_flash::RecomputeGlmFlashVram(input_path,
                                                                 output_path);
        else if (json.find("logestix-glm") != std::string::npos)
            st = lsxcommon::glm::RecomputeGlmVram(input_path, output_path);
        else if (json.find("logestix-qwen") != std::string::npos)
            st = lsxcommon::qwen::RecomputeQwenVram(input_path, output_path);
        else if (json.find("logestix-phi") != std::string::npos)
            st = lsxcommon::phi::RecomputePhiVram(input_path, output_path);
        else if (json.find("logestix-dory") != std::string::npos)
            st = lsxcommon::dory::RecomputeDoryVram(input_path, output_path);
        else
            st = lsxcommon::RecomputeVramBlock(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("VRAM recompute failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info(absl::StrFormat("VRAM recompute complete -> %s", idx.string()));
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kDeepseekNvfp4: {
        lsxcommon::log::info(absl::StrFormat(
            "DeepSeek NVFP4 repack (logestix-v1 -> v1-nvfp4): %s -> %s",
            input_path, output_path));
        auto st = lsxcommon::deepseek::RepackDeepseekExpertsNvfp4(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("DeepSeek NVFP4 repack failed: %s",
                                st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("DeepSeek NVFP4 repack complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kDeepseekV41: {
        lsxcommon::log::info(absl::StrFormat(
            "DeepSeek-V4.1-Flash conversion (%s -> %s)", input_path,
            output_path));
        auto st = lsxcommon::deepseek_v41::ConvertDeepSeekV41Model(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "DeepSeek-V4.1-Flash conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("DeepSeek-V4.1-Flash conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kDeepseekPro: {
        lsxcommon::log::info(absl::StrFormat(
            "DeepSeek-V4-Pro-0813 conversion (%s -> %s)", input_path,
            output_path));
        auto st = lsxcommon::deepseek_pro::ConvertDeepSeekProModel(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "DeepSeek-V4-Pro-0813 conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("DeepSeek-V4-Pro-0813 conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kRepackShared: {
        lsxcommon::log::info(absl::StrFormat("Re-packing shared experts from %s ...", input_path));
        auto st = lsxcommon::RepackSharedExperts(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Shared experts re-pack failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Shared experts re-pack complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kRepackRouted: {
        lsxcommon::log::info(absl::StrFormat("Re-packing routed experts (fp4 passthrough) from %s ...",
                     input_path));
        auto st = lsxcommon::RepackRoutedExperts(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Routed experts re-pack failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Routed experts re-pack complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kTokenizer: {
        namespace fs = std::filesystem;
        auto tokenizer_json = fs::path(input_path) / "tokenizer.json";
        auto pinned_dir = fs::path(input_path) / "logestix" / "pinned";
        fs::create_directories(pinned_dir);
        lsxcommon::log::info(absl::StrFormat("Building tokenizer Arrow files from %s", tokenizer_json.string()));
        // `--tokenizer <dir> --minimax` rebuilds MiniMax-M3's tokenizer with
        // ITS policy (no auto-BOS, Llama-3/o200k pre-tokenizer, literal
        // markers); the plain form keeps the legacy defaults for every other
        // family.
        lsxcommon::TokenizerSpecialPolicy tok_policy;
        if (flags.step) {
            // Step-5: the template spells BOS out as text; the pre-tokenizer
            // is the DeepSeek three-stage Sequence (digits + CJK isolated,
            // then the six-alternative main pattern) -> WordStream / kind 0.
            tok_policy.bos_id = -1;
            tok_policy.pretokenizer = 0;
        }
        if (flags.hy || flags.hy_simulmt) {
            // Hy-Memory: no auto-BOS (the template spells it out), the
            // digit-run/CJK/6-alternative pre-tokenizer, and a vocabulary that
            // deliberately lacks the CR and UTF-8-illegal-lead-byte tokens.
            tok_policy.bos_id = -1;
            tok_policy.pretokenizer = 0;
            tok_policy.drop_unmapped_bytes = 1;
            const auto cfgp = fs::path(input_path) / "config.json";
            if (fs::exists(cfgp)) {
                std::ifstream cf(cfgp);
                std::stringstream cs;
                cs << cf.rdbuf();
                simdjson::ondemand::parser p2;
                simdjson::padded_string pad(cs.str());
                simdjson::ondemand::document doc;
                if (p2.iterate(pad).get(doc) == simdjson::SUCCESS) {
                    simdjson::ondemand::object root;
                    if (doc.get_object().get(root) == simdjson::SUCCESS) {
                        int64_t v = -1;
                        if (root["eos_token_id"].get_int64().get(v) ==
                            simdjson::SUCCESS)
                            tok_policy.eos_id = static_cast<int32_t>(v);
                        if (root["pad_token_id"].get_int64().get(v) ==
                            simdjson::SUCCESS)
                            tok_policy.pad_id = static_cast<int32_t>(v);
                    }
                }
            }
        }
        if (flags.beam) {
            // Beam: no auto-BOS (Harmony spells <|start|> out) and the
            // Qwen2/3 single-run pre-tokenizer (kind 6).
            tok_policy.bos_id = -1;
            tok_policy.pretokenizer = 4;
        }
        if (flags.flint) {
            // Flint: no auto-BOS (its template spells the marker out) and the
            // 3-split digit/CJK + o200k case-run pre-tokenizer (kind 4).
            tok_policy.bos_id = -1;
            tok_policy.pretokenizer = 2;
            const auto cfgp = fs::path(input_path) / "config.json";
            if (fs::exists(cfgp)) {
                std::ifstream cf(cfgp);
                std::stringstream cs;
                cs << cf.rdbuf();
                simdjson::ondemand::parser p3;
                simdjson::padded_string pad3(cs.str());
                simdjson::ondemand::document doc3;
                if (p3.iterate(pad3).get(doc3) == simdjson::SUCCESS) {
                    simdjson::ondemand::object root;
                    if (doc3.get_object().get(root) == simdjson::SUCCESS) {
                        int64_t v = -1;
                        if (root["eos_token_id"].get_int64().get(v) ==
                            simdjson::SUCCESS)
                            tok_policy.eos_id = static_cast<int32_t>(v);
                        else if (auto arr = root["eos_token_id"].get_array();
                                 arr.error() == simdjson::SUCCESS)
                            for (auto e : arr.value())
                                if (e.get_int64().get(v) == simdjson::SUCCESS) {
                                    tok_policy.eos_id = static_cast<int32_t>(v);
                                    break;
                                }
                        if (root["pad_token_id"].get_int64().get(v) ==
                            simdjson::SUCCESS)
                            tok_policy.pad_id = static_cast<int32_t>(v);
                    }
                }
            }
        }
        if (flags.qwen || flags.qwen35moe || flags.qwen36moe ||
            flags.qwen38flashnext || flags.ornith) {
            // Qwen3.5/3.6/3.8 + Ornith: ONE Split stage, \p{L}\p{M} letter
            // runs (Han is not isolated), one digit per number word.
            // tokenizer_kind 7.
            tok_policy.pretokenizer = 5;
        }
        if (flags.glm || flags.glmflash || flags.phi || flags.granite_switch) {
            // GLM / phi-4 / Granite-Switch: one stage, digit groups of <= 3,
            // standalone contraction. tokenizer_kind 8.
            tok_policy.pretokenizer = 6;
        }
        if (flags.mistral || flags.nemotron || flags.nemotron_nvfp4 ||
            flags.dory) {
            // Mistral-Small-4 / Nemotron: o200k case runs WITHOUT a
            // contraction suffix, one digit, [\r\n/]* after punctuation.
            // tokenizer_kind 9.
            tok_policy.pretokenizer = 7;
        }
        if (flags.gemma4 || flags.gemma4dense) {
            // Gemma-4: raw-text BPE from tokenizer.json (kind 11).
            tok_policy.pretokenizer = 9;
        }
        if (flags.ling) {
            // Ling: one letter run with an optional prefix, ONE digit per
            // number word, apostrophe contractions and a punctuation run.
            // tokenizer_kind 5.
            tok_policy.pretokenizer = 3;
        }
        if (flags.gigachat) {
            // GigaChat3.5: the Llama-3/o200k 7-alternative case-run pattern
            // (contraction suffix, \p{N}{1,3}, CJK inside the letter runs).
            // tokenizer_kind 3; `<s>`/`</s>`/pad resolve from
            // tokenizer_config.json. The chat framing prepends the single BOS.
            tok_policy.pretokenizer = 1;
        }
        if (flags.granite) {
            // Granite-4.2: original GPT-2 ByteLevel pattern (case-sensitive
            // contractions, unbounded runs). tokenizer_kind 10.
            tok_policy.pretokenizer = 8;
        }
        if (flags.minimax || flags.minimax_flash) {
            tok_policy.bos_id = -1;
            tok_policy.pretokenizer = 1;
            const auto gen = fs::path(input_path) / "generation_config.json";
            if (fs::exists(gen)) {
                std::ifstream gf(gen);
                std::stringstream gs;
                gs << gf.rdbuf();
                simdjson::ondemand::parser p;
                simdjson::padded_string pad(gs.str());
                simdjson::ondemand::document doc;
                if (p.iterate(pad).get(doc) == simdjson::SUCCESS) {
                    simdjson::ondemand::object root;
                    if (doc.get_object().get(root) == simdjson::SUCCESS) {
                        int64_t v = -1;
                        if (root["eos_token_id"].get_int64().get(v) ==
                            simdjson::SUCCESS)
                            tok_policy.eos_id = static_cast<int32_t>(v);
                        if (root["pad_token_id"].get_int64().get(v) ==
                            simdjson::SUCCESS)
                            tok_policy.pad_id = static_cast<int32_t>(v);
                    }
                }
            }
        }
        auto st = lsxcommon::BuildTokenizerArrowFiles(tokenizer_json.string(),
                                                      pinned_dir, tok_policy);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Tokenizer build failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info(absl::StrFormat("Tokenizer Arrow files complete -> %s", pinned_dir.string()));
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kGlmFlash: {
        lsxcommon::log::info("GLM-5.3-Flash pipeline (unified logestix layout)");
        auto st = lsxcommon::glm_moe_flash::ConvertGlm5NextModel(input_path,
                                                                 output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                "GLM-5.3-Flash conversion failed: " + st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("GLM-5.3-Flash conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kGranite: {
        lsxcommon::log::info("Granite-4.2-30B pipeline (logestix-granite layout)");
        auto st = lsxcommon::granite::ConvertGraniteModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Granite conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Granite conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kGraniteSwitch: {
        lsxcommon::log::info(
            "Granite-Switch 4.1 pipeline (logestix-granite-switch layout)");
        auto st = lsxcommon::granite_switch::ConvertGraniteSwitchModel(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Granite-Switch conversion failed: %s",
                                st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Granite-Switch conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kPhi: {
        lsxcommon::log::info("phi-4 pipeline (logestix-phi layout)");
        auto st = lsxcommon::phi::ConvertPhiModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("phi conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("phi conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kDory: {
        lsxcommon::log::info(
            "NVIDIA-Nemotron-Labs-Dory1.0-4B pipeline (logestix-dory-nvfp4 "
            "layout)");
        auto st = lsxcommon::dory::ConvertDoryModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("dory conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("dory conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kInkling: {
        lsxcommon::log::info(
            "Inkling-2-Small pipeline (logestix-inkling-2-small-nvfp4 layout)");
        auto st = lsxcommon::inkling::ConvertInklingModel(input_path,
                                                         output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Inkling-2-Small conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Inkling-2-Small conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kInklingBase: {
        lsxcommon::log::info(
            "Inkling base pipeline (logestix-inkling-nvfp4 layout)");
        auto st = lsxcommon::inkling::base::ConvertInklingBaseModel(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Inkling base conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Inkling base conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kInklingLoop: {
        lsxcommon::log::info(
            "Inkling-2-Mini pipeline (logestix-inkling-2-mini layout)");
        auto st = lsxcommon::inkling::ConvertInklingLoopModel(input_path,
                                                             output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Inkling-2-Mini conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Inkling-2-Mini conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kMuse: {
        lsxcommon::log::info(
            "Muse-Glimmer-30B pipeline (logestix-muse-nvfp4 layout)");
        auto st = lsxcommon::muse_dense::ConvertMuseDenseModel(input_path,
                                                              output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Muse-Glimmer conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Muse-Glimmer conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kKimiK3: {
        lsxcommon::log::info(
            "Kimi-K3 pipeline (logestix-kimi-k3-nvfp4 layout)");
        auto st = lsxcommon::kimi_k3::ConvertKimiK3Model(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Kimi-K3 conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Kimi-K3 conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kKimiK31Pro: {
        lsxcommon::log::info(
            "Kimi-K3.1-Pro pipeline (logestix-kimi-k31-pro-nvfp4 layout, "
            "Gated DSA indexer on every full-attention layer)");
        auto st = lsxcommon::kimi_pro::ConvertKimiK31ProModel(input_path,
                                                             output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Kimi-K3.1-Pro conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Kimi-K3.1-Pro conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kAlice: {
        lsxcommon::log::info(
            "AliceAI-Foundation pipeline (logestix-alice-nvfp4 layout)");
        auto st = lsxcommon::alice::ConvertAliceModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Alice conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Alice conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kXing: {
        lsxcommon::log::info(
            "Xing4.0-29B-A4B pipeline (logestix-xing-nvfp4 layout)");
        auto st = lsxcommon::xing::ConvertXingModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Xing conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Xing conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kMistral: {
        lsxcommon::log::info(
            "Mistral-Small-4-119B-2603 pipeline (logestix-mistral-nvfp4 layout)");
        auto st = lsxcommon::mistral_moe::ConvertMistralModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Mistral conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Mistral conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kMinimaxFlash: {
        lsxcommon::log::info(
            "MiniMax-M3.1-Flash pipeline (logestix-minimax-nvfp4 layout, "
            "MXFP8 dense + NVFP4 experts, MSA throughout)");
        auto st = lsxcommon::minimax::ConvertMinimaxFlashModel(input_path,
                                                               output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "MiniMax-M3.1-Flash conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("MiniMax-M3.1-Flash conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kMinimax: {
        lsxcommon::log::info(
            "MiniMax-M3 pipeline (logestix-minimax-nvfp4 layout)");
        auto st = lsxcommon::minimax::ConvertMinimaxMoeModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("MiniMax conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("MiniMax conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kFlint: {
        lsxcommon::log::info("Recursive Flint pipeline (logestix-flint-nvfp4 layout)");
        auto st = lsxcommon::flint::ConvertFlintModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("flint conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("flint conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kBeam: {
        lsxcommon::log::info(
            "Beam-1.0-501B-A23B pipeline (logestix-beam-nvfp4 layout)");
        auto st = lsxcommon::beam::ConvertBeamModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("beam conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("beam conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kLing: {
        lsxcommon::log::info(
            "Ling-3.1-flash pipeline (logestix-ling-nvfp4 layout)");
        auto st = lsxcommon::ling::ConvertLingModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("ling conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Ling conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kMistralLarge4: {
        lsxcommon::log::info(
            "Mistral-Large-4-1T-A52B pipeline "
            "(logestix-mistral-large4-nvfp4 layout)");
        auto st = lsxcommon::mistral_large4::ConvertMistralLarge4Model(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Mistral-Large-4 conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Mistral-Large-4 conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kStep: {
        lsxcommon::log::info(
            "Step-5-Preview pipeline (logestix-step5-nvfp4 layout)");
        auto st = lsxcommon::step::ConvertStepMoeModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Step-5 conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Step-5 conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kHySimulMT: {
        lsxcommon::log::info(
            "Hy-SimulMT-30B-A3B pipeline (logestix-hy-simulmt-nvfp4 layout)");
        auto st = lsxcommon::hy_simulmt::ConvertHySimulMTModel(input_path,
                                                               output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Hy-SimulMT conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Hy-SimulMT conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kHy: {
        lsxcommon::log::info(
            "Hy-Memory-30B-A3B pipeline (logestix-hy-nvfp4 layout)");
        auto st = lsxcommon::hy::ConvertHyMoeModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Hy-Memory conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Hy-Memory conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kGemma4Dense: {
        lsxcommon::log::info(
            "Gemma-4 dense pipeline (logestix-gemma-dense layout)");
        auto st = lsxcommon::gemma_dense::ConvertGemmaDenseModel(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Gemma-4 dense conversion failed: %s",
                                st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Gemma-4 dense conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kGemma4: {
        lsxcommon::log::info("Gemma-4 pipeline (logestix-gemma layout)");
        auto st = lsxcommon::gemma::ConvertGemmaModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Gemma conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Gemma conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kNemotron: {
        lsxcommon::log::info(
            "Nemotron-3.5-Lightning pipeline (unified logestix layout)");
        auto st = lsxcommon::nemotron_moe::ConvertNemotronModel(input_path,
                                                                output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("Nemotron conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Nemotron conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kNemotronUltra: {
        lsxcommon::log::info(
            "NVIDIA-Nemotron-3-Ultra-550B-A55B pipeline (unified logestix "
            "layout; grouped router, latent MoE)");
        // --layers a:b converts a slice and marks the tree partial, so a killed
        // 275 GB run resumes at the next layer instead of from zero.
        lsxcommon::nemotron_ultra::ConversionWindow window;
        {
            std::string range_err;
            if (!lsxforge::ParseLayerWindow(layers, window.first_layer,
                                            window.last_layer, range_err)) {
                lsxcommon::log::error(range_err);
                lsxcommon::Core::Shutdown();
                return 1;
            }
        }
        auto st = lsxcommon::nemotron_ultra::ConvertNemotronUltraModel(
            input_path, output_path, window);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Nemotron-Ultra conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Nemotron-Ultra conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kNemotronNvfp4: {
        lsxcommon::log::info(
            "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 pipeline "
            "(pre-quantized ModelOpt source -> unified logestix layout)");
        auto st = lsxcommon::nemotron_nvfp4::ConvertNemotronNvfp4Model(
            input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat(
                "Nemotron-NVFP4 conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Nemotron-NVFP4 conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kGigachat: {
        lsxcommon::log::info(
            "GigaChat3.5-432B-A28B-Reasoning pipeline (unified logestix layout)");
        auto st = lsxcommon::gigachat::ConvertGigaChatModel(input_path,
                                                            output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("GigaChat conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("GigaChat conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kMimo: {
        lsxcommon::log::info(
            "MiMo-V2.6-Flash-RL pipeline (logestix-mimo-nvfp4 layout)");
        auto st = lsxcommon::mimo::ConvertMiMoModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(
                absl::StrFormat("MiMo conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("MiMo conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kQwen38FlashNext: {
        lsxcommon::log::info("Qwen3.8-Flash-Next pipeline (unified logestix layout)");
        auto st = lsxcommon::qwen_moe_exp::ConvertQwen38FlashNextModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Qwen4-exp conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Qwen4-exp conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kQwen36Moe: {
        lsxcommon::log::info("Qwen3.6-MoE pipeline (unified logestix layout)");
        auto st = lsxcommon::qwen_moe_36::ConvertQwen36MoeModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Qwen3.6-MoE conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Qwen3.6-MoE conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kOrnith: {
        lsxcommon::log::info(
            "Ornith-1.5-35B-A3B pipeline (MLX / transformers -> unified "
            "logestix layout)");
        auto st = lsxcommon::ornith::ConvertOrnithMoeModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Ornith conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Ornith conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kQwen35Moe: {
        lsxcommon::log::info("Qwen3.5-MoE pipeline (unified logestix layout)");
        auto st = lsxcommon::qwen_moe::ConvertQwen35MoeModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Qwen-MoE conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Qwen-MoE conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kQwenDense: {
        lsxcommon::log::info("Qwen3.5 pipeline (logestix-qwen layout)");
        auto st = lsxcommon::qwen::ConvertQwenModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Qwen conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Qwen conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kGlm: {
        lsxcommon::log::info("GLM-5.2 pipeline (logestix-glm layout)");
        auto st = lsxcommon::glm::ConvertGlmModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("GLM conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("GLM conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    case lsxforge::Pipeline::kKimi: {
        lsxcommon::log::info("Kimi-K2.7-Code pipeline (logestix-kimi layout)");
        auto st = lsxcommon::kimi::ConvertKimiModel(input_path, output_path);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Kimi conversion failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info("Kimi conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    default: {
        // Generic safetensors converter (the historical lsxforge path).
        lsxcommon::Converter converter;
        converter.SetInputPath(input_path)
                 .SetOutputPath(output_path)
                 .SetVerbose(true);
        auto status = converter.Run();
        if (!status.ok()) {
            lsxcommon::log::error(absl::StrFormat("Conversion failed: %s", status.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }

        lsxcommon::log::info("Conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }
    }
}
