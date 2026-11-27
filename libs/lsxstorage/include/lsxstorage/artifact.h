#pragma once

// ── artifact.fb: reader and verifier (libs/lsxstorage) ─────────────────────
//
// artifact.fb is the single source of truth for what a converted artifact
// CONTAINS: the codec mask, the planes, the per-layer codec map, the tokenizer
// and provenance. model.index.json keeps geometry only (design §2.4.3).
//
// This header exposes PLAIN STRUCTS. FlatBuffers types stay in artifact.cpp, so
// a consumer — including any .cu TU that reaches lsxcommon's ModelBase — never
// pulls the FlatBuffers headers (or their compile time) in.
//
// Nothing here logs, throws or exits: the caller decides how a refusal is
// reported. Arrow reads elsewhere in the tree are mmap-only for the same reason
// the big weight files are: the artifact reader mmaps the buffer and verifies it
// in place.

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lsx::storage {

namespace fs = std::filesystem;

struct PlaneInfo {
    // Exactly one hw::PlaneCodec bit, as an integer so this header does not
    // depend on lsxcommon (the dependency runs the other way).
    uint64_t codec_bit = 0;
    std::string role;      // "dense.mlp", "expert.routed", "embed", "meta", ...
    std::string path;      // relative to the artifact dir; "" = inside model.index.json
    uint64_t bytes = 0;
    std::array<uint8_t, 32> sha256{};
    bool has_digest = false;
};

struct ArtifactInfo {
    uint32_t format_version = 0;
    std::string artifact_id;
    std::string model_type;   // factory registry id, "phi-4"
    std::string family;       // "phi"
    std::string profile;      // "bf16", "nvfp4", "int8-w8a16"
    std::string target_id;    // ADVISORY: "nvidia:sm_86". The gate never reads it.
    uint64_t used_codecs = 0; // the mask the gate consumes
    std::vector<PlaneInfo> planes;
    std::vector<uint64_t> layer_codec; // index = physical layer
    std::string tokenizer_vocab, tokenizer_merges, chat_template;
    std::string model_name, source_dir;
    int layer_count = 0;
    int pinned_tensors = 0;
    int expert_files = 0;
};

enum class ReadStatus {
    kOk,
    kMissing,     // no artifact.fb at all: an unconverted or pre-contract tree
    kUnreadable,  // exists but cannot be mapped
    kBadVersion,  // format_version this build does not understand
    kMalformed,   // present and mapped, but the buffer does not verify
};

const char *ReadStatusName(ReadStatus s);

struct ReadResult {
    ReadStatus status = ReadStatus::kMissing;
    ArtifactInfo info;
    std::string detail; // human-readable; never empty unless status == kOk
};

// Reads <artifact_dir>/artifact.fb. kMissing is a distinct status on purpose:
// the load gate turns it into "this is not a LogestiX artifact; convert it with
// lsxforge", which is a different message from "the file is corrupt".
ReadResult ReadArtifact(const fs::path &artifact_dir);
ReadResult ReadArtifactFile(const fs::path &artifact_fb);

// True when the directory holds an artifact.fb. Presence only: it says nothing
// about validity, so callers that need a loadable artifact still call
// ReadArtifact. This is what replaces "is there a directory named logestix?".
bool IsArtifactDir(const fs::path &dir);

} // namespace lsx::storage
