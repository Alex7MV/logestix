#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"   // gpu::InferenceResult
#include "lsxcommon/safetensors.h"     // SafetensorsModel
#include "lsxcommon/tokenizer.h"       // ArrowTokenizerDb

namespace lsxcommon {

struct ModelInitConfig {
    std::string model_path;   // required
    std::string model_type;   // empty = autodetect from logestix/model.index.json
    int max_gen_tokens = 64;
};

struct ModelRequest {
    std::vector<int32_t> input_ids;   // pre-tokenized prompt
    int max_gen_tokens = 64;
};

class IModel {
public:
    virtual ~IModel() = default;

    virtual bool Load() = 0;
    virtual gpu::InferenceResult Infer(const ModelRequest &req) = 0;
    virtual void Unload() = 0;
    virtual SafetensorsModel Inspect() = 0;
    virtual gpu::InferenceResult SelfCheck(const ModelRequest &req) = 0;

    virtual const std::string &Type() const = 0;
    virtual ArrowTokenizerDb &Tokenizer() = 0;
};

} // namespace lsxcommon