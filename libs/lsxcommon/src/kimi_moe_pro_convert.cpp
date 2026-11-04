// Kimi-K3.1-Pro converter entry: a thin variant discriminator over the shared
// Kimi-K3 converter (the `minimax-m3.1-flash` precedent).
//
// Everything that differs between the two checkpoints -- the accepted
// `model_type`, the mandatory `dsa_config`, the five indexer tensors per DSA
// layer, the format tag and the index fields -- lives in that shared entry, so
// there is exactly one place where a Pro tree can become a K3 tree by accident
// and it fails loudly there instead.
#include "lsxcommon/kimi_moe_pro_convert.h"

#include "lsxcommon/kimi_moe_k3_convert.h"

namespace lsxcommon::kimi_pro {

arrow::Status ConvertKimiK31ProModel(const std::string &input_dir,
                                     const std::string &output_root)
{
    return kimi_k3::ConvertKimiK3Variant(input_dir, output_root,
                                         kimi_k3::KimiVariant::k31_pro);
}

}  // namespace lsxcommon::kimi_pro
