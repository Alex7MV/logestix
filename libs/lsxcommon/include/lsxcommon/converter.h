#pragma once

#include "lsxcommon/schema.h"
#include "lsxcommon/vram_estimate.h"
#include <arrow/api.h>
#include <filesystem>
#include <memory>
#include <string>

namespace lsxcommon {

arrow::Result<PackedExpert> PackExpertNVFP4(const ExpertData &raw);

arrow::Status SaveExpertToArrow(
    const PackedExpert &expert,
    int layer_idx,
    int expert_idx,
    const std::filesystem::path &output_root);

arrow::Status SavePinnedTensors(
    const std::vector<TensorRecord> &tensors,
    const std::filesystem::path &output_root);

// Reads only the shared-expert tensors (w{1,2,3}.weight + w{1,2,3}.scale)
// from the source safetensors shards and re-packs them into
// <output_root>/pinned/shared_experts.arrow. Used to regenerate the shared
// experts after a converter change without re-converting the full model.
arrow::Status RepackSharedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root);

// Reads only the routed-expert tensors (experts.N.w{1,2,3}.weight + .scale)
// from the source safetensors shards and re-writes
// <output_root>/experts/layer_XXX/exp_MMM.arrow for every expert. The source
// weights are already NVFP4 fp4-packed with F8_E8M0 scales, so this is a pure
// CPU passthrough — no re-quantization. Used to regenerate the routed experts
// after a converter change without re-converting the full model.
arrow::Status RepackRoutedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root);

arrow::Status WriteModelIndex(
    const std::string &model_name,
    int num_layers,
    int experts_per_layer,
    int64_t total_pinned,
    int64_t total_experts,
    const VramProfile &vram,
    const std::filesystem::path &output_root);

class Converter {
public:
    Converter();
    ~Converter();

    Converter(Converter &&) noexcept;
    Converter &operator=(Converter &&) noexcept;

    Converter &SetInputPath(const std::string &path);
    Converter &SetOutputPath(const std::string &path);
    Converter &SetVerbose(bool v);

    arrow::Status Run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

arrow::Status BuildTokenizerArrowFiles(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir);

} // namespace lsxcommon
