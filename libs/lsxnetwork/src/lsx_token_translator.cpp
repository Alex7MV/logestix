#include "lsxnetwork/lsx_token_translator.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <absl/container/flat_hash_map.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lsxnetwork {

namespace {

constexpr std::size_t kLineBytes = 64;

constexpr std::size_t RoundUp64(std::size_t n) noexcept {
    return (n + (kLineBytes - 1)) & ~(kLineBytes - 1);
}

/// Length of the longest common substring of [a, a+na) and [b, b+nb).
/// Classic suffix-length DP over a rolling 1-D row (iterated downward so
/// dp[j-1] still holds the previous row's value).
std::size_t LongestCommonSubstring(const char* a, std::size_t na,
                                   const char* b, std::size_t nb) {
    if (na == 0 || nb == 0) return 0;
    std::vector<std::size_t> dp(nb + 1, 0);
    std::size_t best = 0;
    for (std::size_t i = 0; i < na; ++i) {
        for (std::size_t j = nb; j > 0; --j) {
            if (a[i] == b[j - 1]) {
                dp[j] = dp[j - 1] + 1;
                best = std::max(best, dp[j]);
            } else {
                dp[j] = 0;
            }
        }
    }
    return best;
}

}  // namespace

std::vector<std::byte> BuildPassport(const BuildPassportOptions& opts) {
    const std::uint32_t n_glm = static_cast<std::uint32_t>(opts.glm_tokens.size());
    const std::uint32_t n_ds = static_cast<std::uint32_t>(opts.deepseek_tokens.size());

    // Reverse index of DeepSeek token string -> id (exact match wins).
    absl::flat_hash_map<std::string, std::uint32_t> ds_exact;
    ds_exact.reserve(static_cast<std::size_t>(n_ds) * 2);
    // First-byte buckets so substring overlap scanning only touches a bounded,
    // deterministic candidate set (the two vocabularies commonly share token
    // stems under the same leading byte).
    absl::flat_hash_map<std::uint32_t, std::vector<std::uint32_t>> buckets;
    for (std::uint32_t j = 0; j < n_ds; ++j) {
        const std::string& t = opts.deepseek_tokens[j];
        ds_exact.emplace(t, j);  // lowest id wins on duplicate strings
        if (!t.empty()) buckets[static_cast<unsigned char>(t[0])].push_back(j);
    }

    std::vector<std::uint32_t> glm_to_ds(n_glm, opts.deepseek_unk_id);
    for (std::uint32_t i = 0; i < n_glm; ++i) {
        const std::string& g = opts.glm_tokens[i];
        const auto exact = ds_exact.find(g);
        if (exact != ds_exact.end()) {
            glm_to_ds[i] = exact->second;
            continue;
        }
        if (g.empty()) continue;  // stays unk

        std::size_t best = 0;
        std::uint32_t best_id = opts.deepseek_unk_id;
        const auto it = buckets.find(static_cast<unsigned char>(g[0]));
        if (it != buckets.end()) {
            for (const std::uint32_t j : it->second) {
                const std::string& t = opts.deepseek_tokens[j];
                const std::size_t len =
                    LongestCommonSubstring(g.data(), g.size(), t.data(), t.size());
                if (len > best) {  // ascending bucket order keeps ties on low id
                    best = len;
                    best_id = j;
                }
            }
        }
        if (best >= opts.lmcs_min_overlap) glm_to_ds[i] = best_id;
    }

    // Reverse inversion: for each DeepSeek id keep the first (lowest) GLM id
    // that maps to it; unmapped DeepSeek ids fall back to the GLM unk id.
    std::vector<std::uint32_t> ds_to_glm(n_ds, opts.glm_unk_id);
    for (std::uint32_t i = 0; i < n_glm; ++i) {
        const std::uint32_t d = glm_to_ds[i];
        if (d < n_ds && ds_to_glm[d] == opts.glm_unk_id) ds_to_glm[d] = i;
    }

    token_passport::PassportHeader hdr{};
    hdr.glm_vocab_size = n_glm;
    hdr.deepseek_vocab_size = n_ds;
    hdr.glm_unk_id = opts.glm_unk_id;
    hdr.deepseek_unk_id = opts.deepseek_unk_id;

    // Every index array is padded to a full cache line so both arrays start on
    // 64-byte boundaries within the mapped file.
    const std::size_t glm_bytes =
        RoundUp64(static_cast<std::size_t>(n_glm) * sizeof(std::uint32_t));
    const std::size_t ds_bytes =
        RoundUp64(static_cast<std::size_t>(n_ds) * sizeof(std::uint32_t));
    const std::size_t total = sizeof(token_passport::PassportHeader) + glm_bytes + ds_bytes;

    std::vector<std::byte> out(total, std::byte{0});
    std::memcpy(out.data(), &hdr, sizeof(hdr));
    std::memcpy(out.data() + sizeof(hdr), glm_to_ds.data(),
                sizeof(std::uint32_t) * n_glm);
    std::memcpy(out.data() + sizeof(hdr) + glm_bytes, ds_to_glm.data(),
                sizeof(std::uint32_t) * n_ds);
    return out;
}

int WritePassport(const char* path, const BuildPassportOptions& opts) {
    const std::vector<std::byte> bytes = BuildPassport(opts);
    // The destination may live inside a metadata vault that does not exist
    // yet (first conversion on a machine) — create the parent chain.
    if (auto parent = std::filesystem::path(path).parent_path();
        !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        SPDLOG_ERROR("lsx_token_packer: cannot open '{}' for writing", path);
        return -1;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    if (!out) {
        SPDLOG_ERROR("lsx_token_packer: short write on '{}'", path);
        return -1;
    }
    out.close();
    return 0;
}

LsxTokenTranslator::LsxTokenTranslator(LsxTokenTranslator&& o) noexcept
    : glm_to_ds_(o.glm_to_ds_),
      ds_to_glm_(o.ds_to_glm_),
      glm_vocab_(o.glm_vocab_),
      ds_vocab_(o.ds_vocab_),
      glm_unk_(o.glm_unk_),
      ds_unk_(o.ds_unk_),
      handle_(o.handle_),
      base_(o.base_),
      len_(o.len_) {
    o.handle_ = nullptr;
    o.base_ = nullptr;
    o.len_ = 0;
    o.glm_to_ds_ = nullptr;
    o.ds_to_glm_ = nullptr;
    o.glm_vocab_ = o.ds_vocab_ = o.glm_unk_ = o.ds_unk_ = 0;
}

LsxTokenTranslator& LsxTokenTranslator::operator=(LsxTokenTranslator&& o) noexcept {
    if (this == &o) return *this;
    Unmap();
    glm_to_ds_ = o.glm_to_ds_;
    ds_to_glm_ = o.ds_to_glm_;
    glm_vocab_ = o.glm_vocab_;
    ds_vocab_ = o.ds_vocab_;
    glm_unk_ = o.glm_unk_;
    ds_unk_ = o.ds_unk_;
    handle_ = o.handle_;
    base_ = o.base_;
    len_ = o.len_;
    o.handle_ = nullptr;
    o.base_ = nullptr;
    o.len_ = 0;
    o.glm_to_ds_ = nullptr;
    o.ds_to_glm_ = nullptr;
    o.glm_vocab_ = o.ds_vocab_ = o.glm_unk_ = o.ds_unk_ = 0;
    return *this;
}

void LsxTokenTranslator::Unmap() noexcept {
    if (base_) {
#ifdef _WIN32
        ::UnmapViewOfFile(base_);
#else
        ::munmap(base_, len_);
#endif
        base_ = nullptr;
    }
#ifdef _WIN32
    if (handle_) {
        ::CloseHandle(handle_);
        handle_ = nullptr;
    }
#endif
    handle_ = nullptr;
    len_ = 0;
    glm_to_ds_ = nullptr;
    ds_to_glm_ = nullptr;
    glm_vocab_ = ds_vocab_ = glm_unk_ = ds_unk_ = 0;
}

int LsxTokenTranslator::Load(const char* path) noexcept {
    Unmap();
#ifdef _WIN32
    HANDLE hFile = ::CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        SPDLOG_ERROR("lsx_token_translator: open '{}' failed", path);
        return -1;
    }
    LARGE_INTEGER fsz{};
    if (!::GetFileSizeEx(hFile, &fsz) || fsz.QuadPart <= 0) {
        ::CloseHandle(hFile);
        return -1;
    }
    handle_ = ::CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    ::CloseHandle(hFile);
    if (!handle_) return -1;
    base_ = ::MapViewOfFile(handle_, FILE_MAP_READ, 0, 0, 0);
    if (!base_) {
        ::CloseHandle(handle_);
        handle_ = nullptr;
        return -1;
    }
    len_ = static_cast<std::size_t>(fsz.QuadPart);
#else
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        SPDLOG_ERROR("lsx_token_translator: open '{}': {}", path, std::strerror(errno));
        return -1;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return -1;
    }
    len_ = static_cast<std::size_t>(st.st_size);
    base_ = ::mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        return -1;
    }
    handle_ = nullptr;
#endif
    if (!base_) return -1;

    const auto* hdr = reinterpret_cast<const token_passport::PassportHeader*>(base_);
    if (hdr->magic != token_passport::kMagic || hdr->version != token_passport::kVersion) {
        SPDLOG_ERROR("lsx_token_translator: bad magic/version in '{}'", path);
        Unmap();
        return -1;
    }
    const std::size_t glm_bytes =
        RoundUp64(static_cast<std::size_t>(hdr->glm_vocab_size) * sizeof(std::uint32_t));
    const std::size_t ds_bytes =
        RoundUp64(static_cast<std::size_t>(hdr->deepseek_vocab_size) * sizeof(std::uint32_t));
    const std::size_t need = sizeof(token_passport::PassportHeader) + glm_bytes + ds_bytes;
    if (need > len_) {
        SPDLOG_ERROR("lsx_token_translator: '{}' truncated ({}/{})", path, len_, need);
        Unmap();
        return -1;
    }

    glm_vocab_ = hdr->glm_vocab_size;
    ds_vocab_ = hdr->deepseek_vocab_size;
    glm_unk_ = hdr->glm_unk_id;
    ds_unk_ = hdr->deepseek_unk_id;

    const auto* raw = reinterpret_cast<const std::byte*>(base_);
    glm_to_ds_ = reinterpret_cast<const std::uint32_t*>(
        raw + sizeof(token_passport::PassportHeader));
    ds_to_glm_ = reinterpret_cast<const std::uint32_t*>(
        raw + sizeof(token_passport::PassportHeader) + glm_bytes);
    return 0;
}

}  // namespace lsxnetwork