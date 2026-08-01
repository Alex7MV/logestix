#include "lsxcommon/converter.h"
#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/schema.h"
#include <arrow/api.h>
#include <cstring>
#include <filesystem>
#include <optional>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>

namespace {

// layers.N.ffn.shared_experts.w{1,2,3}.weight -> layer index
std::optional<int> parse_shared_expert_layer(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.shared_experts\.w(\d)\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    return std::stoi(m[1]);
}

std::optional<int> parse_shared_expert_weight_id(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.shared_experts\.w(\d)\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    return std::stoi(m[2]);
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

    std::unordered_map<int, lsxcommon::ExpertData> shared_by_layer;
    for (int l = 0; l < 128; ++l) {
        char fn[64];
        std::snprintf(fn, sizeof(fn), "layer_%d.arrow", l);
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
    std::string input_path;
    std::string output_path = "converted_model";
    bool shared_only = false;
    bool tokenizer_only = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (std::strcmp(argv[i], "--shared-only") == 0)
            shared_only = true;
        else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_only = true;
            input_path = argv[++i];
        }
    }

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
