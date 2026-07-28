#pragma once

#include <string>
#include <cstdint>

namespace lsxfabric {

struct ModelInfo {
    std::string format;
    std::string model_name;
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_pinned = 0;
    int64_t total_experts = 0;
};

struct ValidationResult {
    bool ok = true;
    int errors = 0;
    int warnings = 0;

    void add_error() { ok = false; ++errors; }
    void add_warning() { ++warnings; }
};

ModelInfo LoadModelIndex(const std::string &index_path);
ValidationResult ValidatePinnedTensors(const std::string &pinned_dir, int64_t expected_count);
ValidationResult ValidateExpert(const std::string &arrow_path);
bool ValidateNVFP4Data(const uint8_t *data, int64_t nbytes);
bool ValidateE4M3Scales(const uint8_t *scales, int64_t count);
bool ValidateGlobalScale(float scale);
ValidationResult RunValidator(const std::string &model_path);

} // namespace lsxfabric
