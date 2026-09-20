// TDD — GigaChat3.5 model contour. Synthetic converted fixture only; the
// frozen geometry is bypassed via LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 (set by
// the ctest environment). Inference kernels arrive in Plan 2, so Infer returns
// a not-implemented sentinel here.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_moe_fixture.h"

#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/gigachat_moe_model.h"
#include "lsxcommon/model_factory.h"

using namespace lsxcommon;

TEST_CASE("gigachat model loads a synthetic converted fixture",
          "[gigachat][model]")
{
    const auto root = gigachattest::MakeTempDir("gc_model");
    gigachattest::WriteGigaChatCheckpoint(root);
    auto st = gigachat::ConvertGigaChatModel(root.string(), root.string());
    REQUIRE(st.ok());

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat35Model m(cfg);
    REQUIRE(m.Load());
    CHECK(m.IndexForTest().num_layers == 2);
    CHECK(m.Inspect().shards.size() == 1);

    ModelRequest req;
    req.input_ids = {1};
    auto r = m.Infer(req);
    CHECK_FALSE(r.ok);
    m.Unload();
}

TEST_CASE("gigachat factory resolves the kType", "[gigachat][model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = GigaChat35Model::kType;
    cfg.model_path = "/nonexistent";
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "gigachat3.5-432b-a28b-reasoning");
}
