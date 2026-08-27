// TDD — ModelBase shared geometry-field validator + synthetic-geometry gate
// (host-only; no CUDA, no model weights).
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "lsxcommon/model_base.h"

namespace {

class GeoProbe final : public lsxcommon::ModelBase {
public:
    using ModelBase::ModelBase;
    using ModelBase::CheckGeometryFields;
    using ModelBase::SyntheticGeometryAllowed;

    bool Load() override { return false; }
    lsxcommon::gpu::InferenceResult Infer(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    lsxcommon::gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    const std::string &Type() const override
    {
        static std::string t{"geo-probe"};
        return t;
    }
};

} // namespace

TEST_CASE("CheckGeometryFields passes a fully matching table", "[geometry]")
{
    GeoProbe p{lsxcommon::ModelInitConfig{}};
    std::vector<lsxcommon::ModelBase::GeoFieldCheck> fields = {
        {"hidden_size", 2048, 2048},
        {"num_layers", 40, 40},
        {"num_experts_per_tok", 8, 8},
    };
    REQUIRE(p.CheckGeometryFields("probe-model", fields.data(), fields.size()));
}

TEST_CASE("CheckGeometryFields fails on first mismatch with uniform text",
          "[geometry]")
{
    GeoProbe p{lsxcommon::ModelInitConfig{}};
    lsxcommon::ModelBase::GeoFieldCheck fields[] = {
        {"hidden_size", 2048, 2048},
        {"num_layers", 40, 39},
    };
    CHECK_FALSE(p.CheckGeometryFields("probe-model", fields, 2));
}

TEST_CASE("SyntheticGeometryAllowed reads the env gate once", "[geometry]")
{
    // Contract: callable from any model without re-implementing the getenv
    // probe; value depends on LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
    GeoProbe p{lsxcommon::ModelInitConfig{}};
    (void)p.SyntheticGeometryAllowed();  // must not throw / crash
    SUCCEED();
}
