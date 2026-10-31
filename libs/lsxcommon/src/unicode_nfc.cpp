// Canonical NFC normalization (UAX #15), backed by generated Unicode 16.0.0
// tables (unicode_nfc_tables.inc). Host-only: used by the tokenizer encoder
// before the pre-tokenizer, matching the reference pipeline's normalizer.
#include "lsxcommon/unicode.h"
#include "unicode_nfc_tables.inc"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lsxcommon {
namespace unicode {
namespace {

constexpr uint32_t kHangulSBase = 0xAC00;
constexpr uint32_t kHangulLBase = 0x1100;
constexpr uint32_t kHangulVBase = 0x1161;
constexpr uint32_t kHangulTBase = 0x11A7;
constexpr uint32_t kHangulLCount = 19;
constexpr uint32_t kHangulVCount = 21;
constexpr uint32_t kHangulTCount = 28;
constexpr uint32_t kHangulNCount = kHangulVCount * kHangulTCount;
constexpr uint32_t kHangulSCount = kHangulLCount * kHangulNCount;

bool DecodeUtf8(const char *p, size_t n, size_t &len, uint32_t &cp)
{
    const auto *s = reinterpret_cast<const uint8_t *>(p);
    if (n == 0) return false;
    const uint8_t b0 = s[0];
    if (b0 < 0x80) { cp = b0; len = 1; return true; }
    if ((b0 & 0xE0) == 0xC0) {
        if (n < 2 || (s[1] & 0xC0) != 0x80 || b0 < 0xC2) return false;
        cp = (uint32_t(b0 & 0x1F) << 6) | (s[1] & 0x3F);
        len = 2;
        return true;
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (n < 3 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80)
            return false;
        cp = (uint32_t(b0 & 0x0F) << 12) |
             (uint32_t(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        len = 3;
        return true;
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (n < 4 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 ||
            (s[3] & 0xC0) != 0x80)
            return false;
        cp = (uint32_t(b0 & 0x07) << 18) |
             (uint32_t(s[1] & 0x3F) << 12) |
             (uint32_t(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) return false;
        len = 4;
        return true;
    }
    return false;
}

void AppendUtf8(std::string &out, uint32_t cp)
{
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

const NfcDecompEntry *FindDecomp(uint32_t cp)
{
    size_t lo = 0, hi = kNfcDecompsCount;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (kNfcDecomps[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < kNfcDecompsCount && kNfcDecomps[lo].cp == cp)
        return &kNfcDecomps[lo];
    return nullptr;
}

uint8_t CombiningClass(uint32_t cp)
{
    size_t lo = 0, hi = kNfcCombiningCount;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (kNfcCombining[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < kNfcCombiningCount && kNfcCombining[lo].cp == cp)
        return kNfcCombining[lo].ccc;
    return 0;
}

const NfcComposeEntry *FindCompose(uint32_t first, uint32_t second)
{
    size_t lo = 0, hi = kNfcComposesCount;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const auto &e = kNfcComposes[mid];
        if (e.first < first || (e.first == first && e.second < second))
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < kNfcComposesCount && kNfcComposes[lo].first == first &&
        kNfcComposes[lo].second == second)
        return &kNfcComposes[lo];
    return nullptr;
}

bool HangulCompose(uint32_t starter, uint32_t cp, uint32_t &out)
{
    if (starter >= kHangulLBase && starter < kHangulLBase + kHangulLCount &&
        cp >= kHangulVBase && cp < kHangulVBase + kHangulVCount) {
        out = kHangulSBase + (starter - kHangulLBase) * kHangulNCount +
              (cp - kHangulVBase) * kHangulTCount;
        return true;
    }
    if (starter >= kHangulSBase && starter < kHangulSBase + kHangulSCount &&
        (starter - kHangulSBase) % kHangulTCount == 0 &&
        cp > kHangulTBase && cp < kHangulTBase + kHangulTCount) {
        out = starter + (cp - kHangulTBase);
        return true;
    }
    return false;
}

void Decompose(uint32_t cp, std::vector<uint32_t> &out)
{
    if (cp >= kHangulSBase && cp < kHangulSBase + kHangulSCount) {
        const uint32_t s = cp - kHangulSBase;
        out.push_back(kHangulLBase + s / kHangulNCount);
        out.push_back(kHangulVBase + (s % kHangulNCount) / kHangulTCount);
        const uint32_t t = s % kHangulTCount;
        if (t != 0) out.push_back(kHangulTBase + t);
        return;
    }
    const NfcDecompEntry *d = FindDecomp(cp);
    if (d == nullptr) {
        out.push_back(cp);
        return;
    }
    for (uint32_t i = 0; i < d->len; ++i)
        out.push_back(kNfcDecompData[d->offset + i]);
}

} // namespace

std::string NfcNormalize(std::string_view input)
{
    std::vector<uint32_t> cps;
    cps.reserve(input.size());
    for (size_t i = 0; i < input.size();) {
        size_t len = 0;
        uint32_t cp = 0;
        if (!DecodeUtf8(input.data() + i, input.size() - i, len, cp))
            return std::string(input);   // malformed: leave untouched
        cps.push_back(cp);
        i += len;
    }

    std::vector<uint32_t> decomposed;
    decomposed.reserve(cps.size() + 8);
    for (uint32_t cp : cps) Decompose(cp, decomposed);

    // Canonical ordering: stable sort inside each run of non-zero ccc.
    for (size_t i = 1; i < decomposed.size(); ++i) {
        const uint8_t cc = CombiningClass(decomposed[i]);
        if (cc == 0) continue;
        size_t j = i;
        while (j > 0) {
            const uint8_t prev = CombiningClass(decomposed[j - 1]);
            if (prev == 0 || prev <= cc) break;
            std::swap(decomposed[j - 1], decomposed[j]);
            --j;
        }
    }

    // Canonical composition.
    std::vector<uint32_t> composed;
    composed.reserve(decomposed.size());
    constexpr size_t kNoStarter = static_cast<size_t>(-1);
    size_t starter_pos = kNoStarter;
    uint32_t starter = 0;
    uint8_t last_cc = 0;
    for (uint32_t cp : decomposed) {
        const uint8_t cc = CombiningClass(cp);
        if (starter_pos != kNoStarter && (last_cc == 0 || last_cc < cc)) {
            uint32_t merged = 0;
            bool ok = HangulCompose(starter, cp, merged);
            if (!ok) {
                const NfcComposeEntry *e = FindCompose(starter, cp);
                if (e != nullptr) {
                    merged = e->composed;
                    ok = true;
                }
            }
            if (ok) {
                composed[starter_pos] = merged;
                starter = merged;
                // last_cc stays: the composed starter is a starter (cc=0).
                continue;
            }
        }
        composed.push_back(cp);
        if (cc == 0) {
            starter_pos = composed.size() - 1;
            starter = cp;
        }
        last_cc = cc;
    }

    std::string out;
    out.reserve(input.size());
    for (uint32_t cp : composed) AppendUtf8(out, cp);
    return out;
}

} // namespace unicode
} // namespace lsxcommon
