#include <catch2/catch_test_macros.hpp>
#include "validator.h"
#include "lsxcommon/core.h"
#include "lsxcommon/model_engine.h"

#include <string>

TEST_CASE("RunModelValidation rejects an undetectable model path",
          "[lsxfabric][validate]") {
    const std::string log_path = "test_validator.log";
    lsxcommon::Core::Initialize(log_path);
    {
        lsxcommon::InferenceEngine::Session session;
        REQUIRE_FALSE(
            lsxfabric::RunModelValidation("/nonexistent/logestix-audit-model"));
    }
    lsxcommon::Core::Shutdown();
    std::remove(log_path.c_str());
}
