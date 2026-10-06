#pragma once

// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 — converter contract
// (Model Ingestion Protocol stages 1-2).
//
// Same architecture and same frozen geometry as the BF16-distribution sibling
// `nemotron-3.5-lightning-30b-a3b`; the difference is the SOURCE ENCODING. The
// vault checkpoint is already quantized by ModelOpt 0.44.0rc5:
//
//   routed/shared experts + lm_head : W4A16_NVFP4
//       U8 packed E2M1 + F8_E4M3 block scale per 16 [R, C/16]
//       + F32 per-tensor weight_scale_2
//   mamba in_proj / out_proj        : FP8 per-tensor
//       F8_E4M3 + SCALAR F32 weight_scale + F32 input_scale
//   attention q/k/v/o_proj          : BF16 (excluded from quantization),
//       plus F32[1] k_scale / v_scale — the FP8 KV-cache scales, which are NOT
//       weight dequant factors and are dropped
//   embeddings / norms / conv / A_log / D / dt_bias : BF16
//   gate.weight                     : F32 (left high-precision by ModelOpt)
//
// The load-bearing fact, measured on the real vault tensor
// backbone.layers.1.mixer.experts.0.down_proj: the source block scales use the
// FULL E4M3 (all 8 mantissa values present), while the repo's runtime decoder
// (nvfp4_codec.h::E4M3ScaleToFloat / nvfp4_scale.h) reads the e4 exponent field
// ONLY. A byte passthrough therefore decodes a median 0.73x / max 1.00x
// per-block scale error - ~30% relative L2 on the expert weights. So the
// converter DEQUANTIZES with the complete E4M3 decode and REQUANTIZES through
// the repo's own neutral `quantize_tensor_gpu`. Round-trip cost is ~11% relative
// L2, which is the price of the repo's documented pow2-only scale deviation -
// the same deviation every already-ingested family carries.

#include <arrow/api.h>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/nemotron_moe_convert.h"
#include "lsxcommon/nemotron_moe_load.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace nemotron_nvfp4 {

using nemotron_moe::NemotronIndex;
using nemotron_moe::NemotronTopology;
using nemotron_moe::ParseNemotronConfig;

// Tensor kinds of the PRE-QUANTIZED source. The weight kinds mirror the BF16
// sibling one-for-one; the *Scale kinds are the companions this source adds and
// the sibling never had.
enum class NemotronNvfp4Kind {
    Unknown,
    Embed,             // backbone.embeddings.weight                 BF16 raw
    LmHead,            // lm_head.weight                             NVFP4 source
    FinalNorm,         // backbone.norm_f.weight                     BF16 raw
    LayerNorm,         // backbone.layers.N.norm.weight              BF16 raw
    MambaInProj,       // layers.N.mixer.in_proj.weight              FP8 source
    MambaOutProj,      // layers.N.mixer.out_proj.weight             FP8 source
    MambaConv,         // layers.N.mixer.conv1d.weight               BF16 raw
    MambaConvBias,     // layers.N.mixer.conv1d.bias                 BF16 raw
    MambaALog,         // layers.N.mixer.A_log                       BF16 raw
    MambaD,            // layers.N.mixer.D                           BF16 raw
    MambaDtBias,       // layers.N.mixer.dt_bias                     BF16 raw
    MambaNorm,         // layers.N.mixer.norm.weight                 BF16 raw
    AttnQ,             // layers.N.mixer.q_proj.weight               BF16 -> NVFP4
    AttnK,             // layers.N.mixer.k_proj.weight               BF16 -> NVFP4
    AttnV,             // layers.N.mixer.v_proj.weight               BF16 -> NVFP4
    AttnO,             // layers.N.mixer.o_proj.weight               BF16 -> NVFP4
    RouterGate,        // layers.N.mixer.gate.weight                 F32 raw
    RouterBias,        // layers.N.mixer.gate.e_score_correction_bias F32 raw
    SharedUpProj,      // layers.N.mixer.shared_experts.up_proj.weight   NVFP4
    SharedDownProj,    // layers.N.mixer.shared_experts.down_proj.weight NVFP4
    ExpertUpProj,      // layers.N.mixer.experts.E.up_proj.weight       NVFP4
    ExpertDownProj,    // layers.N.mixer.experts.E.down_proj.weight     NVFP4

    // Companion scales of the pre-quantized source.
    Nvfp4BlockScale,   // <w>.weight_scale    F8_E4M3 [R, C/16]
    Nvfp4GlobalScale,  // <w>.weight_scale_2  F32 scalar (per tensor)
    Fp8WeightScale,    // <w>.weight_scale     F32 SCALAR (per tensor)
    Fp8InputScale,     // <w>.input_scale      F32 scalar (activations)
    Fp8KvScale,        // k_proj.k_scale / v_proj.v_scale  F32[1], DROPPED

    Mtp,               // mtp.*   cut + counted
};

struct ParsedNemotronNvfp4Tensor {
    NemotronNvfp4Kind kind = NemotronNvfp4Kind::Unknown;
    int layer = -1;
    int expert = -1;  // only for ExpertUpProj / ExpertDownProj
    // For a *_Scale kind: the base weight tensor name this scale belongs to
    // ("" for Fp8KvScale, which has no weight of its own).
    std::string base_weight;
};

// Classifies one tensor name of the pre-quantized checkpoint. Malformed layer /
// expert indices yield Unknown and NEVER throw (the sibling's contract).
ParsedNemotronNvfp4Tensor classify_nemotron_nvfp4_tensor(const std::string &name);

// True for the weight kinds that leave the converter in the NVFP4 path (the
// converted layout is identical to the BF16 sibling's).
bool nemotron_nvfp4_weight_is_nvfp4(NemotronNvfp4Kind kind);

// Reads <input_dir>/generation_config.json and returns the eos ids. The vault
// declares eos_token_id [2, 11] - a LIST - while config.json carries only the
// scalar 2, so a converter that read config.json alone would silently drop the
// second stop token.
std::vector<int32_t> ParseNemotronEosIds(const std::string &input_dir);

// Converts a pre-quantized Nemotron-3.5-Lightning-30B-A3B-NVFP4 directory into
// the SAME `logestix-nemotron-nvfp4` layout the BF16 sibling produces, so the
// sibling's loader, expert cache, kernels, model pipeline and tool codec are
// reused verbatim.
arrow::Status ConvertNemotronNvfp4Model(const std::string &input_dir,
                                        const std::string &output_root);

// Reads back one converted expert arrow file (test/diagnostic helper): the
// single record it carries.
arrow::Result<PackedExpert> ReadDenseNvfp4Records(const std::string &path);

} // namespace nemotron_nvfp4
} // namespace lsxcommon