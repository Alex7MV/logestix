#include "gemma_moe_model.h"

namespace lsxcommon {

const std::string Gemma4_26BA4BModel::kType = "gemma-4-26b-a4b-it";

// Frozen geometry (compile-time lock, Model Ingestion Protocol step 3).
namespace {
constexpr int kHidden = 2816;
constexpr int kLayers = 30;
constexpr int kHeads = 16;
constexpr int kKVHeads = 8;
constexpr int kGlobalKVHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kGlobalHeadDim = 512;
constexpr int kExperts = 128;
constexpr int kTopK = 8;

static_assert(kHeadDim * kHeads > 0, "gemma geometry must be positive");
static_assert(kGlobalKVHeads > 0 && kGlobalHeadDim > 0,
              "gemma global kv geometry must be positive");
static_assert(kTopK > 0 && kTopK <= kExperts,
              "top_k_experts must be within (0, num_experts]");
static_assert(kLayers > 0, "num layers must be positive");
} // namespace

// Hard geometry gate: loaded topology must match the frozen constants unless
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY is set (synthetic test fixtures only).
bool Gemma4_26BA4BModel::ValidateGeometry(
    const gemma::GemmaTopology &top) const
{
    if (!top.error_msg.empty()) return false;
    if (SyntheticGeometryAllowed()) return true;
    const GeoFieldCheck fields[] = {
        {"hidden_size", kHidden, top.hidden_size},
        {"num_layers", kLayers, top.num_layers},
        {"num_attention_heads", kHeads, top.num_attention_heads},
        {"num_key_value_heads", kKVHeads, top.num_key_value_heads},
        {"num_global_key_value_heads", kGlobalKVHeads,
         top.num_global_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"global_head_dim", kGlobalHeadDim, top.global_head_dim},
        {"num_experts", kExperts, top.num_experts},
        {"top_k_experts", kTopK, top.top_k_experts},
    };
    return CheckGeometryFields("gemma4-26b-a4b", fields, 9);
}

} // namespace lsxcommon
