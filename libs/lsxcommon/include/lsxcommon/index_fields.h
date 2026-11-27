#pragma once

// ── Registry of model.index.json fields a loader may REQUIRE ───────────────
//
// Why this exists, in one paragraph of history. The index used to carry a
// per-model `format` tag. An audit found 18 spellings for one schema family, 11
// of them written by a converter and read by nothing. Nobody noticed for months,
// because the only way to find out was to grep, and grep is unreliable here: keys
// are read through local lambdas (`get_i(root, "num_layers", ...)`) and written
// through five different helpers (`scalar`, `bool_array`, `write_array`,
// streamed `<<`, and plain literals), so every pattern-based audit reports a
// different answer. Four successive scans during that audit produced four
// different "dead key" lists, and three of the four contained false positives
// that had to be refuted by hand.
//
// So the authority is this table, not a scan. A key is listed here when its
// ABSENCE is a hard load failure (not a defaulted value); the guard
// (tests/test_index_fields.cpp) proves that each one is emitted by the converter
// that owns it. A key that is merely defaulted when missing does not need a row:
// the loader already has an answer for its absence.
//
// Nothing in libs/ consumes this table at runtime — it is an instrument, read by
// the guard and available to future tooling (a `--verify` could check these on a
// real tree). Keeping it a header means a new required read is added in the same
// commit that introduces it.

#include <cstdint>

namespace lsxcommon {

struct IndexField {
    const char *key;         // the JSON key
    // Converter that must emit it. "" means "every converter that writes an
    // index" (only `layout` is that universal).
    const char *owner_file;
    const char *why;         // what breaks when it is missing
};

// Keys a loader cannot do without. Order is alphabetical for reviewability.
inline constexpr IndexField kRequiredIndexFields[] = {
    {"dense_layers", "mistral_moe_large4_convert.cpp",
     "the per-layer kind map; without it the loader cannot tell a dense MLP "
     "layer from a MoE one"},
    {"experts_per_layer", "converter.cpp",
     "the routing counts; the deepseek loaders fail the whole index without them"},
    {"indexer_layers", "mistral_moe_large4_convert.cpp",
     "which layers carry the DSA indexer; absent, every layer would be treated "
     "as non-indexed"},
    {"kv_source_layers", "mistral_moe_large4_convert.cpp",
     "which layers reuse another layer's KV; absent, the loader has no source "
     "for them"},
    {"layout", "",
     "the schema generation; absent or different is a refusal, never a guess"},
    {"moe_layers", "mistral_moe_large4_convert.cpp",
     "the routed-expert layer set; absent, no expert is ever streamed"},
    {"total_experts", "converter.cpp",
     "the routed-expert inventory size, used to size the streamed store"},
    {"rope_layers", "mistral_moe_large4_convert.cpp",
     "which layers rotate; absent, RoPE is applied to every layer or none"},
};

inline constexpr std::size_t kRequiredIndexFieldCount =
    sizeof(kRequiredIndexFields) / sizeof(kRequiredIndexFields[0]);

} // namespace lsxcommon
