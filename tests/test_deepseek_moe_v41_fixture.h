#pragma once

// Synthetic DeepSeek-V4.1-Flash index fixture (host-only). P1 needs only the
// index JSON; P2/P3 extend this header with pinned/ and expert Arrow writers.
// Family-neutral helpers (FakeTensor/WriteSafetensors/MakeBf16*) come from
// test_synth_checkpoint.h when P2 lands — never from another family's fixture.
#include <filesystem>
#include <fstream>
#include <string>

#include <arrow/status.h>

namespace v41fixtest {

inline void WriteIndex(const std::filesystem::path &model_dir,
                       const std::string &json)
{
    std::filesystem::create_directories(model_dir / "logestix");
    std::ofstream f(model_dir / "logestix" / "model.index.json");
    f << json;
}

inline std::string DefaultIndexJson(int num_layers, int experts,
                                    const std::string &format)
{
    return "{\n"
           "  \"format\": \"" + format + "\",\n"
           "  \"model\": \"DeepSeek-V4.1-Flash\",\n"
           "  \"num_layers\": " + std::to_string(num_layers) + ",\n"
           "  \"experts_per_layer\": " + std::to_string(experts) + ",\n"
           "  \"total_experts\": " +
           std::to_string(static_cast<long long>(num_layers) * experts) +
           ",\n"
           "  \"vram\": {\n"
           "    \"min_bytes\": 1024,\n"
           "    \"optimal_bytes\": 2048,\n"
           "    \"max_bytes\": 4096,\n"
           "    \"context_tokens\": 1048576\n"
           "  }\n"
           "}\n";
}

inline arrow::Status BuildSyntheticIndex(
    const std::filesystem::path &model_dir, int num_layers = 40,
    int experts = 384, const std::string &format = "logestix-v41-nvfp4")
{
    WriteIndex(model_dir, DefaultIndexJson(num_layers, experts, format));
    return arrow::Status::OK();
}

inline arrow::Status BuildSyntheticIndexRaw(
    const std::filesystem::path &model_dir, const std::string &json)
{
    WriteIndex(model_dir, json);
    return arrow::Status::OK();
}

} // namespace v41fixtest
