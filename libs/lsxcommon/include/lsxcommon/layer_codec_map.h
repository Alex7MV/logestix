#pragma once

// Projecting a per-layer codec mask onto a family's topology flags.
//
// artifact.fb::layer_codec is the single source of truth for "which codec is
// layer N" (design §2.4.3 rule 5). The readers that needed a bool per layer —
// inkling's NVFP4-vs-MXFP8 expert planes are the case that made this necessary —
// must therefore derive their flags from that map instead of from a copy of it
// in model.index.json, which is how the two could disagree.
//
// This header is intentionally tiny and dependency-free (no lsxstorage type, no
// absl): the topology structs live in headers that .cu TUs include, so it may
// only use <cstdint>/<vector>.

#include <cstdint>
#include <vector>

namespace lsxcommon {

// Bit values match lsxcommon::hw::PlaneCodec (hardware_target.h) and the
// artifact.fb schema. Spelled here rather than included because this header is
// reachable from device TUs and must stay free of anything heavier.
inline constexpr uint64_t kLayerCodecNvFp4 = 1ull << 5;
inline constexpr uint64_t kLayerCodecMxFp8 = 1ull << 4;

// Fills `nvfp4[l]` / `fp8[l]` for every layer the mask vector covers.
//
// A layer with NEITHER bit set (no per-layer file named it) stays false in both,
// which is the honest answer: the caller then has no evidence that the layer is
// quantized at all, and a caller that needs one must fail rather than guess.
inline void ProjectLayerCodecs(const std::vector<uint64_t> &layer_codec,
                               std::vector<bool> *nvfp4, std::vector<bool> *fp8)
{
    if (nvfp4 == nullptr || fp8 == nullptr) return;
    nvfp4->assign(layer_codec.size(), false);
    fp8->assign(layer_codec.size(), false);
    for (size_t l = 0; l < layer_codec.size(); ++l) {
        (*nvfp4)[l] = (layer_codec[l] & kLayerCodecNvFp4) != 0;
        (*fp8)[l] = (layer_codec[l] & kLayerCodecMxFp8) != 0;
    }
}

} // namespace lsxcommon
