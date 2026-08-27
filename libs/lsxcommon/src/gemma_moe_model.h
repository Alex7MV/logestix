#pragma once

#include "gemma_base_model.h"

namespace lsxcommon {

// Gemma-4-26B-A4B-it: frozen compile-time geometry (protocol step 3).
class Gemma4_26BA4BModel final : public GemmaBaseModel {
public:
    static const std::string kType;

    explicit Gemma4_26BA4BModel(ModelInitConfig cfg)
        : GemmaBaseModel(std::move(cfg))
    {
        type_ = kType;
    }

protected:
    bool ValidateGeometry(const gemma::GemmaTopology &top) const override;
};

} // namespace lsxcommon
