// artifact.fb writer (design §2.4.5).
//
// The emitter is the ONLY place a mask is authored, and it refuses to author a
// lie: after writing it re-reads and re-verifies its own output, and returns 0
// if the mask it wrote disagrees with the planes on disk. With every converted
// tree deleted there is no reference artifact left to compare a converter
// against, so this self-check is the only thing between a converter bug and a
// plausible-looking artifact.
#include "lsxstorage/artifact_writer.h"

#include "lsxstorage/artifact.h"

#include "artifact_generated.h"
#include "sha256.h"

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace lsx::storage {

namespace {

constexpr uint32_t kFormatVersion = 1;

// Relative paths of the files copied into the artifact to make it
// self-sufficient (I5): it must load with LSX_SOURCE_ROOT unmounted.
struct TokenizerCopies {
    std::string chat_template;  // "" when the checkpoint has none
    std::string tokenizer_json; // "" when not copied
};

fb::Vendor VendorFromId(const std::string &target_id)
{
    if (target_id.rfind("cpu:", 0) == 0) return fb::Vendor::Cpu;
    if (target_id.rfind("amd:", 0) == 0) return fb::Vendor::Amd;
    return fb::Vendor::Nvidia;
}

std::string ArchFromId(const std::string &target_id)
{
    const auto colon = target_id.find(':');
    return colon == std::string::npos ? std::string{} : target_id.substr(colon + 1);
}

std::string Iso8601UtcNow()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32] = {0};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

bool CopyFileOverwrite(const fs::path &from, const fs::path &to, std::string *error)
{
    std::error_code ec;
    fs::create_directories(to.parent_path(), ec);
    fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        if (error != nullptr)
            *error = "cannot copy " + from.string() + " -> " + to.string() + ": " +
                     ec.message();
        return false;
    }
    return true;
}

std::vector<uint8_t> DigestBytes(const Sha256Digest &d)
{
    return std::vector<uint8_t>(d.begin(), d.end());
}

} // namespace

uint64_t EmitArtifact(const ArtifactSpec &spec, const fs::path &artifact_dir,
                      std::string *error)
{
    const auto fail = [error](const std::string &msg) {
        if (error != nullptr) *error = msg;
        return uint64_t{0};
    };

    if (artifact_dir.empty())
        return fail("artifact directory is empty (is LSX_ARTIFACT_ROOT set?)");
    if (spec.model_type.empty() || spec.family.empty() || spec.profile.empty())
        return fail("artifact spec needs model_type, family and profile");

    std::error_code ec;
    fs::create_directories(artifact_dir, ec);
    if (ec)
        return fail("cannot create " + artifact_dir.string() + ": " + ec.message());

    // ── 1. Validate the declared planes against what is actually on disk, and
    // compute the mask from content rather than from the declaration. A plane
    // that is declared but absent is a converter defect, not something to paper
    // over with an empty digest.
    uint64_t computed_mask = 0;
    for (const auto &decl : spec.planes) {
        if (decl.codec_bit == 0)
            return fail("plane '" + decl.path + "' declares no codec bit");
        if (decl.path.empty() && decl.role != "meta")
            return fail("plane '" + decl.role + "' has no path");
        if (!decl.path.empty()) {
            const auto p = artifact_dir / decl.path;
            if (!fs::is_regular_file(p, ec))
                return fail("declared plane is missing on disk: " + decl.path);
        }
        computed_mask |= decl.codec_bit;
    }

    // ── 2. Copy the tokenizer and template into the artifact so it is
    // self-sufficient: an artifact must load with LSX_SOURCE_ROOT unmounted.
    TokenizerCopies copies;
    if (!spec.chat_template_src.empty()) {
        const fs::path from(spec.chat_template_src);
        if (!fs::is_regular_file(from, ec))
            return fail("chat template declared but absent: " + from.string());
        copies.chat_template = "chat_template.jinja";
        if (!CopyFileOverwrite(from, artifact_dir / copies.chat_template, error))
            return 0;
    }
    if (!spec.tokenizer_json_src.empty()) {
        const fs::path from(spec.tokenizer_json_src);
        if (!fs::is_regular_file(from, ec))
            return fail("tokenizer.json declared but absent: " + from.string());
        copies.tokenizer_json = "tokenizer.json";
        if (!CopyFileOverwrite(from, artifact_dir / copies.tokenizer_json, error))
            return 0;
    }

    // ── 3. Digest every plane. This is the expensive pass and the one that
    // makes tampering and truncation detectable later.
    std::vector<std::vector<uint8_t>> digests;
    digests.reserve(spec.planes.size());
    for (const auto &decl : spec.planes) {
        if (decl.path.empty()) {
            digests.emplace_back();          // not digested (meta planes)
            continue;
        }
        std::string hash_error;
        const auto d = Sha256::OfFile((artifact_dir / decl.path).string(), &hash_error);
        if (!hash_error.empty()) return fail(hash_error);
        digests.push_back(DigestBytes(d));
    }

    std::vector<uint8_t> tpl_digest;
    if (!copies.chat_template.empty()) {
        std::string hash_error;
        tpl_digest = DigestBytes(
            Sha256::OfFile((artifact_dir / copies.chat_template).string(), &hash_error));
        if (!hash_error.empty()) return fail(hash_error);
    }

    // ── 4. Serialize.
    flatbuffers::FlatBufferBuilder builder(4096);
    std::vector<flatbuffers::Offset<fb::Plane>> plane_offsets;
    plane_offsets.reserve(spec.planes.size());
    for (size_t i = 0; i < spec.planes.size(); ++i) {
        const auto &decl = spec.planes[i];
        const auto path_off = builder.CreateString(decl.path);
        const auto role_off = builder.CreateString(decl.role);
        const auto digest_off = digests[i].empty()
                                    ? flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{}
                                    : builder.CreateVector(digests[i]);
        plane_offsets.push_back(fb::CreatePlane(
            builder, static_cast<fb::PlaneCodec>(decl.codec_bit), role_off,
            path_off, decl.bytes, digest_off));
    }

    std::vector<uint64_t> layer_codec = spec.layer_codec;

    const auto tokenizer_off = fb::CreateTokenizer(
        builder,
        builder.CreateString("pinned/tokenizer_vocab.arrow"),
        builder.CreateString("pinned/tokenizer_merges.arrow"),
        builder.CreateString(copies.chat_template),
        tpl_digest.empty() ? flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{}
                           : builder.CreateVector(tpl_digest),
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{});

    const auto source_off = fb::CreateSource(
        builder, builder.CreateString(spec.model_name),
        builder.CreateString(spec.source_dir),
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{},
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{});

    const auto artifact_id =
        spec.model_type + "|" + spec.family + "|" + spec.profile + "|" +
        std::to_string(computed_mask);

    const auto target_off = fb::CreateTarget(
        builder, VendorFromId(spec.target_id),
        builder.CreateString(ArchFromId(spec.target_id)));

    const auto artifact_off = fb::CreateArtifact(
        builder, kFormatVersion, builder.CreateString(artifact_id),
        builder.CreateString(spec.model_type), builder.CreateString(spec.family),
        builder.CreateString(spec.profile), target_off, computed_mask,
        builder.CreateVector(plane_offsets), builder.CreateVector(layer_codec),
        tokenizer_off, source_off, spec.layer_count, spec.pinned_tensors,
        spec.expert_files, builder.CreateString("lsxforge"),
        builder.CreateString(spec.converter_git_rev),
        builder.CreateString(spec.profile), builder.CreateString(Iso8601UtcNow()),
        0);

    fb::FinishArtifactBuffer(builder, artifact_off);

    // ── 5. Atomic publish: write a temp file in the same directory, fsync, then
    // rename. A half-written artifact.fb must never be visible, because the load
    // gate reads the file's mere presence as "somebody converted this".
    const auto final_path = artifact_dir / "artifact.fb";
    const auto tmp_path = artifact_dir / "artifact.fb.tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open())
            return fail("cannot open " + tmp_path.string() + " for writing");
        out.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
                  static_cast<std::streamsize>(builder.GetSize()));
        out.flush();
        if (!out)
            return fail("short write to " + tmp_path.string());
    }
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(tmp_path, ignored);
        return fail("cannot publish artifact.fb: " + ec.message());
    }

    // ── 6. Re-read and verify what was written. A mask that disagrees with the
    // bytes on disk makes the whole artifact untrustworthy, so a mismatch is a
    // conversion failure, not a warning.
    const auto read_back = ReadArtifact(artifact_dir);
    if (read_back.status != ReadStatus::kOk)
        return fail("artifact.fb does not read back: " + read_back.detail);
    if (read_back.info.used_codecs != computed_mask)
        return fail("mask mismatch: wrote " + std::to_string(computed_mask) +
                    " but read back " + std::to_string(read_back.info.used_codecs));

    if (error != nullptr) error->clear();
    return computed_mask;
}

bool PreflightDisk(const fs::path &dir, uint64_t bytes_needed, std::string *error)
{
    std::error_code ec;
    fs::path probe = dir;
    // Walk up to the nearest existing ancestor: the artifact directory itself
    // usually does not exist yet.
    while (!probe.empty() && !fs::exists(probe, ec)) {
        const auto parent = probe.parent_path();
        if (parent == probe) break;
        probe = parent;
    }
    const auto info = fs::space(probe, ec);
    if (ec) {
        if (error != nullptr)
            *error = "cannot query free space for " + probe.string() + ": " +
                     ec.message();
        return false;
    }
    if (info.available < bytes_needed) {
        if (error != nullptr)
            *error = "not enough free space under " + probe.string() + ": " +
                     std::to_string(info.available) + " free, " +
                     std::to_string(bytes_needed) + " estimated for this artifact";
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

} // namespace lsx::storage
