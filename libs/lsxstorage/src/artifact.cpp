// artifact.fb reader (design §2.4.4).
//
// mmap + flatbuffers::Verifier over the mapped range, then a copy into plain
// structs. The verifier runs over the buffer BEFORE any accessor is called: a
// truncated or hostile file must not be able to make us read past it.
#include "lsxstorage/artifact.h"

#include "artifact_generated.h"

#include <flatbuffers/verifier.h>

#include <cstring>
#include <fstream>

namespace lsx::storage {

namespace {

constexpr uint32_t kSupportedFormatVersion = 1;

// FlatBuffers declares a file_identifier; a buffer written by something else
// (or by a much older schema) has different bytes there.
constexpr size_t kIdentifierOffset = 4;  // root offset(4) then the identifier
constexpr size_t kIdentifierLength = 4;

// Reads the whole file. artifact.fb is small (tens of KB: identity, plane list,
// digests), so a plain read is simpler than mmap here — the mmap discipline in
// AGENTS.md is about the multi-gigabyte Arrow weight planes, which this is not.
// Returns an empty vector and fills *error on failure.
std::vector<uint8_t> ReadWholeFile(const fs::path &p, std::string *error)
{
    std::ifstream in(p, std::ios::binary);
    if (!in.is_open()) {
        if (error != nullptr) *error = "cannot open " + p.string();
        return {};
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size <= 0) {
        if (error != nullptr) *error = "empty file " + p.string();
        return {};
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    in.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!in) {
        if (error != nullptr) *error = "short read on " + p.string();
        return {};
    }
    return bytes;
}

PlaneInfo CopyPlane(const fb::Plane *p)
{
    PlaneInfo out;
    if (p == nullptr) return out;
    out.codec_bit = static_cast<uint64_t>(p->codec());
    if (p->role() != nullptr) out.role = p->role()->str();
    if (p->path() != nullptr) out.path = p->path()->str();
    out.bytes = p->bytes();
    if (const auto *digest = p->sha256(); digest != nullptr && digest->size() == 32) {
        for (size_t i = 0; i < 32; ++i) out.sha256[i] = digest->Get(i);
        out.has_digest = true;
    }
    return out;
}

std::string Str(const flatbuffers::String *s)
{
    return s != nullptr ? s->str() : std::string{};
}

} // namespace

const char *ReadStatusName(ReadStatus s)
{
    switch (s) {
    case ReadStatus::kOk: return "ok";
    case ReadStatus::kMissing: return "missing";
    case ReadStatus::kUnreadable: return "unreadable";
    case ReadStatus::kBadVersion: return "bad-version";
    case ReadStatus::kMalformed: return "malformed";
    }
    return "unknown";
}

bool IsArtifactDir(const fs::path &dir)
{
    if (dir.empty()) return false;
    std::error_code ec;
    return fs::is_regular_file(dir / "artifact.fb", ec);
}

ReadResult ReadArtifactFile(const fs::path &artifact_fb)
{
    ReadResult r;
    std::error_code ec;
    if (!fs::exists(artifact_fb, ec)) {
        r.status = ReadStatus::kMissing;
        r.detail = "no artifact.fb at " + artifact_fb.string();
        return r;
    }

    std::string read_error;
    const auto bytes = ReadWholeFile(artifact_fb, &read_error);
    if (bytes.empty()) {
        r.status = ReadStatus::kUnreadable;
        r.detail = read_error;
        return r;
    }

    // The identifier check is cheap and turns "some other FlatBuffer" into a
    // clear message instead of a generic verify failure.
    if (bytes.size() < kIdentifierOffset + kIdentifierLength) {
        r.status = ReadStatus::kUnreadable;
        r.detail = "artifact.fb is too small to be a FlatBuffer (" +
                   std::to_string(bytes.size()) + " bytes)";
        return r;
    }
    if (std::memcmp(bytes.data() + kIdentifierOffset, fb::ArtifactIdentifier(),
                    kIdentifierLength) != 0) {
        r.status = ReadStatus::kMalformed;
        r.detail = "artifact.fb is not an LSXA buffer (bad file identifier)";
        return r;
    }

    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!fb::VerifyArtifactBuffer(verifier)) {
        r.status = ReadStatus::kMalformed;
        r.detail = "artifact.fb failed FlatBuffers verification (truncated or "
                   "written by an incompatible schema)";
        return r;
    }

    const auto *a = fb::GetArtifact(bytes.data());
    if (a == nullptr || a->format_version() != kSupportedFormatVersion) {
        r.status = ReadStatus::kBadVersion;
        r.detail = "artifact.fb format_version " +
                   std::to_string(a != nullptr ? a->format_version() : 0) +
                   " is not supported (this build reads " +
                   std::to_string(kSupportedFormatVersion) + ")";
        return r;
    }

    ArtifactInfo &info = r.info;
    info.format_version = a->format_version();
    info.artifact_id = Str(a->artifact_id());
    info.model_type = Str(a->model_type());
    info.family = Str(a->family());
    info.profile = Str(a->profile());
    info.org = Str(a->org());
    info.model_dir = Str(a->model_dir());
    if (const auto *t = a->target(); t != nullptr)
        info.target_id = std::string(fb::EnumNameVendor(t->vendor())) + ":" +
                         Str(t->arch());
    info.used_codecs = a->used_codecs();

    if (const auto *planes = a->planes(); planes != nullptr) {
        info.planes.reserve(planes->size());
        for (const auto *p : *planes) info.planes.push_back(CopyPlane(p));
    }
    if (const auto *layers = a->layer_codec(); layers != nullptr) {
        info.layer_codec.reserve(layers->size());
        for (const auto v : *layers) info.layer_codec.push_back(v);
    }
    if (const auto *tok = a->tokenizer(); tok != nullptr) {
        info.tokenizer_vocab = Str(tok->vocab());
        info.tokenizer_merges = Str(tok->merges());
        info.chat_template = Str(tok->chat_template());
    }
    if (const auto *src = a->source(); src != nullptr) {
        info.model_name = Str(src->model_name());
        info.source_dir = Str(src->source_dir());
    }
    info.layer_count = a->layer_count();
    info.pinned_tensors = a->pinned_tensors();
    info.expert_files = a->expert_files();

    // A verified buffer that carries no identity is not usable: the gate would
    // have nothing to check and the loader nothing to size itself by.
    if (info.model_type.empty() || info.family.empty() || info.profile.empty()) {
        r.status = ReadStatus::kMalformed;
        r.detail = "artifact.fb verifies but is missing model_type/family/profile";
        return r;
    }

    r.status = ReadStatus::kOk;
    return r;
}

ReadResult ReadArtifact(const fs::path &artifact_dir)
{
    if (artifact_dir.empty()) {
        ReadResult r;
        r.status = ReadStatus::kMissing;
        r.detail = "empty artifact directory";
        return r;
    }
    return ReadArtifactFile(artifact_dir / "artifact.fb");
}

} // namespace lsx::storage
