#include "lsxcommon/converter.h"
#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/generation_cli.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/glm_convert.h"
#include "lsxcommon/qwen_convert.h"
#include <arrow/api.h>
#include <cstring>
#include <filesystem>
#include <optional>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <absl/container/flat_hash_map.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>

ABSL_FLAG(std::string, input, "", "input model directory");
ABSL_FLAG(std::string, output, "converted_model", "output directory");
ABSL_FLAG(bool, verify, false, "verify a converted model");
ABSL_FLAG(bool, shared_only, false, "extract shared experts from pinned layers");
ABSL_FLAG(bool, repack_shared, false, "re-pack shared experts into shared_experts.arrow");
ABSL_FLAG(bool, repack_experts, false, "re-pack routed experts (fp4 passthrough)");
ABSL_FLAG(std::string, tokenizer, "", "build tokenizer Arrow files from tokenizer.json");
ABSL_FLAG(bool, glm, false, "GLM-5.2 conversion pipeline");
ABSL_FLAG(bool, qwen, false, "Qwen3.5 conversion pipeline");
// Generation path execution option: routes the resolved prompt straight into
// the tokenizer encode channel and the single end-to-end cudaStream_t
// Prefill->Decode autogeneration sequence loop (IModel::Infer), bypassing the
// standalone --selfcheck truncation logic and the --benchmark dummy-token
// layers. A bare --run/-r (no value) falls back to the default blueprint
// prompt via lsxcommon::cli.
ABSL_FLAG(std::string, run, "", "generation prompt (--run \"PROMPT\" / -r \"PROMPT\")");

namespace {

// layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale} -> layer index
std::optional<int> parse_shared_expert_layer(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.shared_experts\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    int layer = 0;
    if (!absl::SimpleAtoi(m[1].str(), &layer)) return std::nullopt;
    return layer;
}

std::optional<int> parse_shared_expert_weight_id(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.shared_experts\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    int wid = 0;
    if (!absl::SimpleAtoi(m[2].str(), &wid)) return std::nullopt;
    return wid;
}

bool parse_shared_expert_is_scale(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.shared_experts\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return false;
    return m[3] == "scale";
}

// Extracts the NVFP4-packed shared expert weights from the existing per-layer
// pinned files and writes a single pinned/shared_experts.arrow (one row per
// layer, same 9-column layout as ExpertSchema).
arrow::Status RunSharedOnly(const std::string &model_path)
{
    namespace fs = std::filesystem;
    auto pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir))
        return arrow::Status::Invalid("no pinned dir found: ", pinned_dir.string());

    absl::flat_hash_map<int, lsxcommon::ExpertData> shared_by_layer;
    for (int l = 0; l < 128; ++l) {
        auto fn = absl::StrCat("layer_", l, ".arrow");
        auto path = pinned_dir / fn;
        if (!fs::exists(path)) {
            if (l > 0) break;
            continue;
        }

        auto batch_res = lsxcommon::ReadRecordBatchFromFile(path);
        if (!batch_res.ok()) {
            spdlog::warn("  {} read failed: {}", fn, batch_res.status().ToString());
            continue;
        }
        auto tensors_res = lsxcommon::PinnedSchema::UnpackBatch(*batch_res);
        if (!tensors_res.ok()) {
            spdlog::warn("  {} unpack failed: {}", fn, tensors_res.status().ToString());
            continue;
        }

        int found = 0;
        for (auto &t : *tensors_res) {
            auto layer = parse_shared_expert_layer(t.name);
            if (!layer) continue;
            auto wid = parse_shared_expert_weight_id(t.name);
            if (!wid) continue;
            auto &sd = shared_by_layer[*layer];
            if (parse_shared_expert_is_scale(t.name)) {
                switch (*wid) {
                case 1:
                    sd.w1_scale_raw   = std::move(t.data);
                    sd.w1_scale_shape = t.shape;
                    break;
                case 2:
                    sd.w2_scale_raw   = std::move(t.data);
                    sd.w2_scale_shape = t.shape;
                    break;
                case 3:
                    sd.w3_scale_raw   = std::move(t.data);
                    sd.w3_scale_shape = t.shape;
                    break;
                }
            } else {
                switch (*wid) {
                case 1:
                    sd.w1_raw   = std::move(t.data);
                    sd.w1_shape = t.shape;
                    sd.w1_dtype = t.dtype;
                    break;
                case 2:
                    sd.w2_raw   = std::move(t.data);
                    sd.w2_shape = t.shape;
                    sd.w2_dtype = t.dtype;
                    break;
                case 3:
                    sd.w3_raw   = std::move(t.data);
                    sd.w3_shape = t.shape;
                    sd.w3_dtype = t.dtype;
                    break;
                }
            }
            ++found;
        }
        spdlog::info("  {} — {} shared-expert tensors", fn, found);
    }

    if (shared_by_layer.empty())
        return arrow::Status::Invalid("no shared expert tensors found in pinned/");

    int max_layer = -1;
    for (auto &[layer, sd] : shared_by_layer)
        max_layer = std::max(max_layer, layer);
    std::vector<lsxcommon::PackedExpert> shared_layers(
        static_cast<size_t>(max_layer) + 1);

    int packed_ok = 0;
    for (auto &[layer, sd] : shared_by_layer) {
        if (sd.w1_raw.empty() || sd.w2_raw.empty() || sd.w3_raw.empty()) {
            spdlog::warn("  incomplete shared expert layer {} (w1={} w2={} w3={})",
                layer, sd.w1_raw.size(), sd.w2_raw.size(), sd.w3_raw.size());
            continue;
        }
        auto packed_res = lsxcommon::PackExpertNVFP4(std::move(sd));
        if (!packed_res.ok()) {
            spdlog::error("  pack failed layer {}: {}",
                layer, packed_res.status().ToString());
            continue;
        }
        shared_layers[static_cast<size_t>(layer)] = *packed_res;
        ++packed_ok;
    }

    while (!shared_layers.empty() && shared_layers.back().w1_packed.empty())
        shared_layers.pop_back();
    if (shared_layers.empty())
        return arrow::Status::Invalid("no shared expert layer was packed");

    auto batch_res = lsxcommon::SharedExpertSchema::MakeBatch(shared_layers);
    if (!batch_res.ok())
        return batch_res.status();
    auto out = pinned_dir / "shared_experts.arrow";
    ARROW_RETURN_NOT_OK(lsxcommon::WriteRecordBatchToFile(*batch_res, out));
    spdlog::info("packed {} shared-expert layers -> {} ({} layers total)",
        packed_ok, out.string(), shared_layers.size());
    return arrow::Status::OK();
}

} // anonymous namespace

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
    bool shared_only = absl::GetFlag(FLAGS_shared_only);
    bool repack_shared = absl::GetFlag(FLAGS_repack_shared);
    bool repack_experts = absl::GetFlag(FLAGS_repack_experts);
    bool tokenizer_only = false;
    if (!absl::GetFlag(FLAGS_tokenizer).empty()) {
        tokenizer_only = true;
        input_path = absl::GetFlag(FLAGS_tokenizer);
    }
    bool verify_only = absl::GetFlag(FLAGS_verify);
    bool glm = absl::GetFlag(FLAGS_glm);
    bool qwen = absl::GetFlag(FLAGS_qwen);
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
        spdlog::info("Running generation path on prompt: {}", run_prompt);
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = input_path;
        cfg.max_gen_tokens = 64;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        if (!model || !model->Load()) {
            spdlog::error("Failed to create/load model: {}", input_path);
            lsxcommon::Core::Shutdown();
            return 1;
        }
        std::vector<int32_t> input_ids;
        if (!lsxcommon::prompt::BuildPromptInputIds(
                model->Type(), model->Tokenizer(), run_prompt, input_ids)) {
            spdlog::error("Tokenizer encode failed for prompt: {}",
                          run_prompt);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
        lsxcommon::ModelRequest req{std::move(input_ids), 64};
        auto result = model->Infer(req);
        if (!result.ok) {
            spdlog::error("Inference failed: {}", result.error_msg);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("=== --run Results ===");
        spdlog::info("prompt   : {}", run_prompt);
        spdlog::info("output   : {}", result.output_text);
        spdlog::info("tokens   : {} at {:.1f} t/s",
                     result.num_tokens, result.tokens_per_sec);
        model->Unload();
        lsxcommon::Core::Shutdown();
        return 0;
    }

    if (verify_only) {
        spdlog::info("Verifying converted model: {}", input_path);
        auto vres = lsxcommon::integrity::VerifyConvertedModel(input_path);
        if (!vres.ok()) {
            spdlog::error("Verify failed: {}", vres.status().ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("Verify OK — {} model tensors across {} arrow files",
                     vres->pinned_tensor_count, vres->arrow_files_checked);
        lsxcommon::Core::Shutdown();
        return 0;
    }

    if (shared_only) {
        spdlog::info("Extracting shared experts from {} ...", input_path);
        auto st = RunSharedOnly(input_path);
        if (!st.ok()) {
            spdlog::error("Shared experts extraction failed: {}", st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("Shared experts extraction complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    if (repack_shared) {
        spdlog::info("Re-packing shared experts from {} ...", input_path);
        auto st = lsxcommon::RepackSharedExperts(input_path, output_path);
        if (!st.ok()) {
            spdlog::error("Shared experts re-pack failed: {}", st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("Shared experts re-pack complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    if (repack_experts) {
        spdlog::info("Re-packing routed experts (fp4 passthrough) from {} ...",
                     input_path);
        auto st = lsxcommon::RepackRoutedExperts(input_path, output_path);
        if (!st.ok()) {
            spdlog::error("Routed experts re-pack failed: {}", st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("Routed experts re-pack complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    if (tokenizer_only) {
        namespace fs = std::filesystem;
        auto tokenizer_json = fs::path(input_path) / "tokenizer.json";
        auto pinned_dir = fs::path(input_path) / "logestix" / "pinned";
        fs::create_directories(pinned_dir);
        spdlog::info("Building tokenizer Arrow files from {}", tokenizer_json.string());
        auto st = lsxcommon::BuildTokenizerArrowFiles(tokenizer_json.string(), pinned_dir);
        if (!st.ok()) {
            spdlog::error("Tokenizer build failed: {}", st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("Tokenizer Arrow files complete -> {}", pinned_dir.string());
        lsxcommon::Core::Shutdown();
        return 0;
    }

    spdlog::info("Converting model: {} -> {}", input_path, output_path);
    if (qwen) {
        spdlog::info("Qwen3.5 pipeline (logestix-qwen layout)");
        auto st = lsxcommon::qwen::ConvertQwenModel(input_path, output_path);
        if (!st.ok()) {
            spdlog::error("Qwen conversion failed: {}", st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("Qwen conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    if (glm) {
        spdlog::info("GLM-5.2 pipeline (logestix-glm layout)");
        auto st = lsxcommon::glm::ConvertGlmModel(input_path, output_path);
        if (!st.ok()) {
            spdlog::error("GLM conversion failed: {}", st.ToString());
            lsxcommon::Core::Shutdown();
            return 1;
        }
        spdlog::info("GLM conversion complete");
        lsxcommon::Core::Shutdown();
        return 0;
    }

    lsxcommon::Converter converter;
    converter.SetInputPath(input_path)
             .SetOutputPath(output_path)
             .SetVerbose(true);
    auto status = converter.Run();
    if (!status.ok()) {
        spdlog::error("Conversion failed: {}", status.ToString());
        lsxcommon::Core::Shutdown();
        return 1;
    }

    spdlog::info("Conversion complete");
    lsxcommon::Core::Shutdown();
    return 0;
}
