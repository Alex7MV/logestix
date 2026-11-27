#pragma once

// ── model.index.json layout generation ────────────────────────────────────
//
// A model index declares ONE integer: the schema LAYOUT generation its files are
// written in. It does not declare a per-model format name.
//
// History, because the field it replaced caused real problems:
//
//   * The index used to carry `"format": "logestix-<model>[-nvfp4]"` — 18
//     spellings for one schema family. That string duplicated the model identity
//     (which `artifact.fb` model_type and the index's own `"model"` already
//     carry) and the plane codec (which `artifact.fb` used_codecs carries).
//   * Ten loaders GATED on it, and six of the seven gated literals embedded the
//     codec ("…-nvfp4"). A bf16 artifact of the same model would therefore pass
//     the codec gate in ModelBase::BeginLoad and then be refused by its own
//     loader — a direct conflict with multi-profile support (invariant I4).
//   * Eleven of the eighteen tags were dead: written, never read.
//
// `layout` says the one thing that is genuinely per-generation and that a loader
// must know: which file/subtree schema it is about to read. Layout 2 is "one
// directory per codec profile; content under pinned/ experts/ router/ shared/
// ple/; PinnedSchema / DenseNvfp4Schema / ExpertSchema; codec declared in
// artifact.fb". Layout 1 was the `format`-tag generation.
//
// There is deliberately NO fallback. Layout-1 trees were deleted, and a loader
// that guessed would be reading a schema it does not know. A missing or
// different layout is a refusal that names both numbers.
//
// This header is reachable from `.cu` (deepseek loaders include it): POD and
// <string> only, no absl containers, no Arrow, no logging.

#include <cstdint>
#include <string>

namespace lsxcommon {

// The layout generation this build reads and writes.
inline constexpr int64_t kIndexLayout = 2;

// True only for exactly the layout this build understands. A range check would
// silently accept a future schema whose files may not be readable.
inline bool IndexLayoutSupported(int64_t layout)
{
    return layout == kIndexLayout;
}

// The shared refusal text, so all loaders say the same thing and name both
// numbers. `layout` 0 means the field is absent.
std::string IndexLayoutError(int64_t layout);

} // namespace lsxcommon
