#include "lsxcommon/converter.h"
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
#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/mimo_moe_convert.h"
#include "lsxcommon/nemotron_moe_convert.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/qwen_moe_36_convert.h"
#include "lsxcommon/ornith_moe_convert.h"
#include "lsxcommon/qwen_moe_exp_convert.h"
#include "lsxcommon/glm_moe_convert.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/qwen_dense_convert.h"
#include "lsxcommon/granite_dense_convert.h"
#include "lsxcommon/deepseek_moe_repack.h"
#include "lsxcommon/deepseek_moe_v41_convert.h"
#include "lsxcommon/log.h"
#include <arrow/api.h>
#include <fstream>
#include <iterator>
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
ABSL_FLAG(bool, repack_shared, false, "re-pack shared experts into shared_experts.arrow");
ABSL_FLAG(bool, repack_experts, false, "re-pack routed experts (fp4 passthrough)");
ABSL_FLAG(std::string, tokenizer, "", "build tokenizer Arrow files from tokenizer.json");
ABSL_FLAG(bool, glm, false, "GLM-5.2 conversion pipeline");
ABSL_FLAG(bool, kimi, false, "Kimi-K2.7-Code conversion pipeline");
ABSL_FLAG(bool, qwen, false, "Qwen3.5 conversion pipeline");
ABSL_FLAG(bool, qwen35moe, false, "Qwen3.5-35B-A3B MoE conversion pipeline");
ABSL_FLAG(bool, qwen36moe, false, "Qwen3.6-35B-A3B MoE conversion pipeline");
ABSL_FLAG(bool, ornith, false, "Ornith-1.5-35B-A3B (MLX) conversion pipeline");
ABSL_FLAG(bool, qwen38flashnext, false,
          "Qwen3.8-Flash-Next (qwen4_exp) conversion pipeline");
ABSL_FLAG(bool, gemma4, false, "Gemma-4-26B-A4B conversion pipeline");
ABSL_FLAG(bool, nemotron, false,
          "Nemotron-3.5-Lightning-30B-A3B (NemotronH hybrid) conversion pipeline");
ABSL_FLAG(bool, gigachat, false,
          "GigaChat3.5-432B-A28B-Reasoning (hybrid MLA+GDN+MoE) conversion pipeline");
ABSL_FLAG(bool, glmflash, false,
          "GLM-5.3-Flash (glm5_next) conversion pipeline");
ABSL_FLAG(bool, granite, false,
          "Granite-4.2-30B (dense) conversion pipeline");
ABSL_FLAG(bool, mimo, false,
          "MiMo-V2.6-Flash-RL (hybrid SWA/full sparse MoE) conversion pipeline");
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
    bool tokenizer_only = false;
    if (!absl::GetFlag(FLAGS_tokenizer).empty()) {
        tokenizer_only = true;
        input_path = absl::GetFlag(FLAGS_tokenizer);
    }
    bool glm = absl::GetFlag(FLAGS_glm);
    bool kimi = absl::GetFlag(FLAGS_kimi);
    bool qwen = absl::GetFlag(FLAGS_qwen);
    bool qwen35moe = absl::GetFlag(FLAGS_qwen35moe);
    bool qwen36moe = absl::GetFlag(FLAGS_qwen36moe);
    bool ornith = absl::GetFlag(FLAGS_ornith);
    bool qwen38flashnext = absl::GetFlag(FLAGS_qwen38flashnext);
    bool gemma4 = absl::GetFlag(FLAGS_gemma4);
    bool nemotron = absl::GetFlag(FLAGS_nemotron);
    bool gigachat = absl::GetFlag(FLAGS_gigachat);
    bool glmflash = absl::GetFlag(FLAGS_glmflash);
    bool granite = absl::GetFlag(FLAGS_granite);
    bool mimo = absl::GetFlag(FLAGS_mimo);
    const std::string run_prompt = lsxcommon::cli::ResolveGenerationPrompt(absl::GetFlag(FLAGS_run));

    if (input_path.empty()) {
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
    if (!absl::GetFlag(FLAGS_run).empty()) {
        lsxcommon::runtime::GenerationCliArgs args;
        args.model_path = input_path;
        args.prompt = run_prompt;
        args.max_gen_tokens = 64;
        int rc = lsxcommon::runtime::RunCliGeneration(args);
        lsxcommon::Core::Shutdown();
        return rc;
    }

    if (absl::GetFlag(FLAGS_vram_recompute)) {
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
        if (json.find("logestix-glm") != std::string::npos)
            st = lsxcommon::glm::RecomputeGlmVram(input_path, output_path);
        else if (json.find("logestix-qwen") != std::string::npos)
            st = lsxcommon::qwen::RecomputeQwenVram(input_path, output_path);
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

    if (absl::GetFlag(FLAGS_deepseek_nvfp4)) {
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

    if (absl::GetFlag(FLAGS_deepseek_v41)) {
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

    if (repack_shared) {
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

    if (repack_experts) {
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

    if (tokenizer_only) {
        namespace fs = std::filesystem;
        auto tokenizer_json = fs::path(input_path) / "tokenizer.json";
        auto pinned_dir = fs::path(input_path) / "logestix" / "pinned";
        fs::create_directories(pinned_dir);
        lsxcommon::log::info(absl::StrFormat("Building tokenizer Arrow files from %s", tokenizer_json.string()));
        auto st = lsxcommon::BuildTokenizerArrowFiles(tokenizer_json.string(), pinned_dir);
        if (!st.ok()) {
            lsxcommon::log::error(absl::StrFormat("Tokenizer build failed: %s", st.ToString()));
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::log::info(absl::StrFormat("Tokenizer Arrow files complete -> %s", pinned_dir.string()));
        lsxcommon::Core::Shutdown();
        return 0;
    }

    lsxcommon::log::info(absl::StrFormat("Converting model: %s -> %s", input_path, output_path));
    if (glmflash) {
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

    if (granite) {
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

    if (gemma4) {
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

    if (nemotron) {
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

    if (gigachat) {
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

    if (mimo) {
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

    if (qwen38flashnext) {
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

    if (qwen36moe) {
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

    if (ornith) {
        lsxcommon::log::info("Ornith-1.5-35B-A3B pipeline (MLX -> unified logestix layout)");
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

    if (qwen35moe) {
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

    if (qwen) {
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

    if (glm) {
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

    if (kimi) {
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
