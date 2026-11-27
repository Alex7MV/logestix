#pragma once

// Minimal SHA-256, host-only, used by the artifact emitter and verifier to
// digest planes and copied tokenizer files.
//
// Why not a library: the only other digest users in the tree are build scripts;
// FlatBuffers/Arrow/absl ship no SHA-256, and pulling OpenSSL in for one hash
// would be a dependency the project does not otherwise need. This is a
// straightforward FIPS 180-4 implementation, verified against the standard test
// vectors in tests/test_storage_artifact_writer.cpp.
//
// NOT for cryptography: artifact.fb digests detect corruption and accidental
// substitution, not a hostile forger (who could re-emit the file).

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace lsx::storage {

using Sha256Digest = std::array<uint8_t, 32>;

class Sha256 {
public:
    Sha256();

    void Update(const void *data, size_t len);
    // Finishes the digest. The object must not be updated afterwards.
    Sha256Digest Final();

    static Sha256Digest OfFile(const std::string &path, std::string *error);
    static Sha256Digest OfBytes(const void *data, size_t len);

private:
    void Block(const uint8_t *p);

    uint32_t h_[8];
    uint8_t buffer_[64];
    size_t buffer_len_ = 0;
    uint64_t total_len_ = 0;
};

// Lowercase hex, 64 chars.
std::string Sha256Hex(const Sha256Digest &d);

} // namespace lsx::storage
