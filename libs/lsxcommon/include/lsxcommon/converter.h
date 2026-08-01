#pragma once

#include "lsxcommon/schema.h"
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

arrow::Status WriteModelIndex(
    const std::string &model_name,
    int num_layers,
    int experts_per_layer,
    int64_t total_pinned,
    int64_t total_experts,
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
