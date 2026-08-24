#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lsxcommon/gemma4_inference.h"
#include "lsxcommon/gemma4_inference_load.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {

// Shared base for Gemma-4 family models: GLM-style pinned loader, on-demand
// routed experts, single-stream generation loop, incremental session KV.
class GemmaBaseModel : public ModelBase {
public:
    explicit GemmaBaseModel(ModelInitConfig cfg);
    ~GemmaBaseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return type_; }
    void SetType(std::string t) { type_ = std::move(t); }

    // Gemma-4 chat framing — model-owned token bridge (BOS-led template).
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override
    {
        return gemma::BuildPromptInputIds(tok, raw_prompt, out_ids);
    }

protected:
    // Hard geometry gate (protocol step 3): false when the loaded topology
    // does not match the frozen compile-time constants.
    virtual bool ValidateGeometry(const gemma::GemmaTopology &top) const;

    static bool SyntheticGeometryAllowed();

    std::string type_;
    gemma::GemmaPinnedWeights pw_;
    bool weights_loaded_ = false;

    // Incremental session state (Qwen policy): a request whose input_ids equal
    // the previous generated tail continues the cache; anything else resets.
    std::unique_ptr<gemma::GemmaKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
