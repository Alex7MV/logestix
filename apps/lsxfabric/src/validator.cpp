#include "validator.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_engine.h"

#include <absl/strings/str_format.h>

#include <memory>

namespace lsxfabric {

bool RunModelValidation(const std::string &model_path) {
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_path;

    std::unique_ptr<lsxcommon::IModel> model =
        lsxcommon::InferenceEngine::Instance().CreateModel(cfg);
    if (!model) {
        lsxcommon::log::error(absl::StrFormat(
            "Validation failed: cannot create a model for %s", model_path));
        return false;
    }

    const bool ok = model->Load();
    model->Unload();
    if (!ok)
        lsxcommon::log::error(
            absl::StrFormat("Validation failed: load error for %s", model_path));
    return ok;
}

} // namespace lsxfabric
