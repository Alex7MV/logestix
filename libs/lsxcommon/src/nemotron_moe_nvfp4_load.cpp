// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 loader
// (Model Ingestion Protocol stage 3). See nemotron_moe_nvfp4_load.h.
#include "lsxcommon/nemotron_moe_nvfp4_load.h"

#include "nemotron_moe_constants.h"   // sibling geometry, cross-checked below
#include "nemotron_moe_nvfp4_constants.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace lsxcommon {
namespace nemotron_nvfp4 {

bool LoadNemotronNvfp4Index(const std::string &model_path, NemotronIndex &out,
                            std::string &error)
{
    if (!nemotron_moe::LoadNemotronIndex(model_path, out, error)) return false;

    // Both distributions emit `logestix-nemotron-nvfp4`, so the format tag alone
    // cannot tell them apart. The `model` field can - and it has to, because the
    // two models carry separate FSI/CODE/DET goldens and a mixed-up tree would
    // silently verify against the wrong one.
    const std::string want = frozen::kModelName;
    if (out.model != want) {
        error = absl::StrCat("model.index.json declares model '", out.model,
                             "' but this loader serves '", want,
                             "' - the BF16 sibling's tree is a different model");
        return false;
    }
    return true;
}

bool ValidateNemotronNvfp4Geometry(const NemotronIndex &idx, std::string &error)
{
    using namespace frozen;
    auto bad = [&error](const char *field, long long got, long long want) {
        error = absl::StrFormat("%s: got %lld, want %lld", field, got, want);
        return false;
    };

    // First: the sibling's gate, so a shared regression is reported once. It
    // compares against nemotron_moe::frozen, which this model's passport
    // re-declares value-for-value (pinned by
    // test_nemotron_moe_nvfp4_loader::"nvfp4 passport is self-consistent").
    if (!nemotron_moe::ValidateNemotronGeometry(idx, error)) return false;
    if (SyntheticGeometryAllowed()) return true;

    // Then this model's OWN passport, so the check is not merely a second view
    // of the sibling's numbers.
    if (idx.hidden_size != kHidden) return bad("hidden_size", idx.hidden_size, kHidden);
    if (idx.vocab_size != kVocab) return bad("vocab_size", idx.vocab_size, kVocab);
    if (idx.num_layers != kLayers) return bad("num_layers", idx.num_layers, kLayers);
    if (idx.num_experts != kNumRouted) return bad("num_experts", idx.num_experts, kNumRouted);
    if (idx.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", idx.num_experts_per_tok, kTopK);
    if (idx.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", idx.moe_intermediate_size,
                   kMoeIntermediate);
    if (idx.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("shared_expert_intermediate_size",
                   idx.shared_expert_intermediate_size, kSharedIntermediate);
    if (idx.bos_token_id != kBosTokenId)
        return bad("bos_token_id", idx.bos_token_id, kBosTokenId);
    if (idx.pad_token_id != kPadTokenId)
        return bad("pad_token_id", idx.pad_token_id, kPadTokenId);

    // The stop SET, not just the primary stop. A tree that omits id 11 runs past
    // the model's own stop; one that claims a stop the vault does not declare is
    // a different checkpoint wearing this model's name.
    const std::vector<int> want_eos(
        kEosTokenIds, kEosTokenIds + kNumEosTokenIds);
    if (idx.eos_token_ids != want_eos) {
        error = absl::StrCat("eos_token_ids: got [",
                             absl::StrJoin(idx.eos_token_ids, ", "), "], want [",
                             absl::StrJoin(want_eos, ", "), "]");
        return false;
    }
    return true;
}

} // namespace nemotron_nvfp4
} // namespace lsxcommon