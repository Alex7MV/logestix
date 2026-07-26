#pragma once

#include "lsxcommon/schema.h"
#include <arrow/api.h>
#include <filesystem>
#include <memory>
#include <string>

namespace lsxcommon {

// NVFP4 packing stub — real implementation would call a CUDA kernel
arrow::Result<PackedExpert> PackExpertNVFP4(const ExpertData &raw);

// Write a single expert to an Arrow IPC file
arrow::Status SaveExpertToArrow(
    const PackedExpert &expert,
    int layer_idx,
    int expert_idx,
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

} // namespace lsxcommon
