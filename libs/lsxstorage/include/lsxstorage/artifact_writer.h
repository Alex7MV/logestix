#pragma once

// ── artifact.fb writer (libs/lsxstorage) ───────────────────────────────────
//
// The emitter takes a DECLARATION of the planes a converter produced plus the
// real files, and writes artifact.fb. What it checks itself: every declared
// plane exists on disk, the mask it writes is the OR of the declared bits, the
// write is atomic, and the file reads back with the same mask.
//
// What it deliberately does NOT check: that the declared codec matches the
// bytes. Reading every plane's tensors during a conversion would double the I/O
// on a 30 GB model, and the check belongs to `lsxforge --verify`
// (VerifyArtifact), which infers the codec from content and is an independent
// input rather than a restatement. Run --verify after a conversion.
//
// Deliberate omission: v1 ships no ballast. A plane the load path cannot reach
// (an MTP head, say) is not written at all, so `used_codecs` has exactly one
// meaning — the codecs of the planes in this artifact — and there is no
// stored-vs-executed distinction anywhere in the gate.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lsx::storage {

namespace fs = std::filesystem;

// One plane the converter wrote. `codec_bit` is exactly one hw::PlaneCodec bit
// as an integer (this header must not depend on lsxcommon).
struct PlaneDecl {
    std::string role;      // "dense.mlp", "expert.routed", "embed", "meta", ...
    uint64_t codec_bit = 0;
    std::string path;      // relative to the artifact dir; "" for a meta plane
    uint64_t bytes = 0;
};

struct ArtifactSpec {
    std::string model_type;  // manifest id: "phi-4"
    std::string family;      // "phi"
    std::string profile;     // "bf16", "nvfp4", "int8-w8a16"
    std::string model_name;  // display name, "Phi-4"
    std::string source_dir;  // "phi-4", relative to LSX_SOURCE_ROOT

    // Advisory provenance only: the target this artifact was BUILT for. The
    // load gate never reads it (it compares used_codecs against the ACTIVE
    // device), so copying an artifact elsewhere cannot smuggle it past I1.
    std::string target_id;   // "nvidia:sm_120"

    std::vector<PlaneDecl> planes;

    // Per-layer codec mask, index = physical layer, each entry an OR of codec
    // bits. This is the record six load/inference sites read to pick a kernel
    // per layer; model.index.json must carry no codec key at all.
    std::vector<uint64_t> layer_codec;

    // Absolute source paths of the files to copy INTO the artifact so it is
    // self-sufficient. Empty means the checkpoint has none.
    std::string chat_template_src;
    std::string tokenizer_json_src;

    std::string converter_git_rev;
    int layer_count = 0;
    int pinned_tensors = 0;
    int expert_files = 0;
};

// Writes <artifact_dir>/artifact.fb atomically (tmp + fsync + rename), copies
// the tokenizer/template in, digests every plane, then re-reads and re-verifies
// its own output. Returns the mask that was written, or 0 on failure with
// *error set.
uint64_t EmitArtifact(const ArtifactSpec &spec, const fs::path &artifact_dir,
                      std::string *error);

// True iff `dir` (or its nearest existing ancestor) has at least
// `bytes_needed` free. A guard against a half-written tree — a partial artifact
// has no artifact.fb and is refused by the gate for a reason that reads like a
// codec bug, which is the worst failure mode to debug.
bool PreflightDisk(const fs::path &dir, uint64_t bytes_needed, std::string *error);

} // namespace lsx::storage
