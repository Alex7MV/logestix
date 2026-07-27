#pragma once

#include "lsxcommon/schema.h"
#include <arrow/api.h>
#include <memory>
#include <string>

namespace lsxcommon {

struct LoadedExpert {
    std::shared_ptr<arrow::RecordBatch> batch;
    std::shared_ptr<arrow::io::MemoryMappedFile> mmap;

    const uint8_t *w1_weight = nullptr;
    const uint8_t *w2_weight = nullptr;
    const uint8_t *w3_weight = nullptr;
    const uint8_t *w1_scale  = nullptr;
    const uint8_t *w2_scale  = nullptr;
    const uint8_t *w3_scale  = nullptr;
    int64_t w1_weight_size = 0;
    int64_t w2_weight_size = 0;
    int64_t w3_weight_size = 0;
    int64_t w1_scale_count = 0;
    int64_t w2_scale_count = 0;
    int64_t w3_scale_count = 0;
    float global_scale = 1.0f;
    bool cuda_registered = false;
};

class Loader {
public:
    Loader() = default;
    ~Loader();

    Loader(const Loader &) = delete;
    Loader &operator=(const Loader &) = delete;
    Loader(Loader &&) = default;
    Loader &operator=(Loader &&) = default;

    arrow::Result<LoadedExpert>
    LoadExpertFromArrow(const std::string &arrow_path);

    arrow::Status RegisterCudaPinned(LoadedExpert &expert);

    static void UnregisterCudaPinned(LoadedExpert &expert);
};

} // namespace lsxcommon
