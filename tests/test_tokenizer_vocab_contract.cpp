// Per-family vocabulary contract for the decode -> frame -> encode round-trip.
//
// A chat transcript must survive being fed back to the model: answer -> decode
// -> history -> arrow_tokenizer_encode. The engine guarantees that only if the
// decoder never emits text the encoder rejects, and the encoder does reject any
// byte that cannot start or complete a well-formed UTF-8 sequence
// (arrow_tokenizer_encode -> -1, BuildConversationInputIds -> false, the next
// question is silently dropped).
//
// The hazard is vocabulary-shaped, not family-shaped: a byte-level vocab maps
// each raw byte to a token whose value IS that byte, so the model can emit a
// token that decodes to a lone lead or continuation byte. Measured on
// gemma-4-26B-A4B-it: 125 single-byte entries >= 0x80, with
// byte_to_token_id[0xE9] pointing at the one-byte 0xE9 token. arrow_tokenizer_decode
// now substitutes U+FFFD for those, which is what keeps the round-trip whole.
//
// This guard checks the invariant against every converted model actually present
// on the machine, so a newly ingested family cannot ship a vocabulary that
// breaks chat framing without the suite saying so. It SKIPs when no converted
// model is available.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/byte_unicode.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

namespace {

// How a family stores non-printable bytes.
//
// kNoLiftedBytes: the space is an ordinary 0x20 token and bytes are stored
// verbatim, so the decoder needs no bytes_to_unicode reversal. Confirmed on
// gemma-4-26B-A4B-it (0 lifted entries).
//
// kCallerReversesLiftedBytes: the vocab lifts bytes into U+0100..U+0143 GPT-2
// style. The engine has NO reversal — it hands back the lifted glyph, so the
// calling side must undo bytes_to_unicode. Declaring this keeps the assumption
// explicit; such a family renders "Ġ" instead of a space until that contract is
// honoured.
enum class LiftContract { kNoLiftedBytes, kCallerReversesLiftedBytes };

struct KnownFamily {
    const char *model_dir;
    LiftContract lift;
};

// Every family that may appear in the models directory. A model that is present
// but missing here fails the test on purpose: adding a family has to state its
// byte-lifting contract rather than inherit an unchecked assumption.
constexpr KnownFamily kKnownFamilies[] = {
    {"AliceAI-Foundation-80B-A3B-Base", LiftContract::kNoLiftedBytes},
    {"Beam-1.0-501B-A23B", LiftContract::kNoLiftedBytes},
    {"DeepSeek-V4-Pro-0813", LiftContract::kNoLiftedBytes},
    {"flint", LiftContract::kNoLiftedBytes},
    {"gemma-4-26B-A4B-it", LiftContract::kNoLiftedBytes},
    {"gemma-4-31B-it", LiftContract::kNoLiftedBytes},
    {"GLM-5.3-Flash", LiftContract::kNoLiftedBytes},
    {"granite-4.2-30b", LiftContract::kNoLiftedBytes},
    {"granite-switch-4.1-30b", LiftContract::kNoLiftedBytes},
    {"Hy-Memory-30B-A3B", LiftContract::kNoLiftedBytes},
    {"Hy-SimulMT-30B-A3B", LiftContract::kNoLiftedBytes},
    {"Kimi-K2.7-Code", LiftContract::kNoLiftedBytes},
    {"Ling-3.1-flash", LiftContract::kNoLiftedBytes},
    {"MiniMax-M3", LiftContract::kNoLiftedBytes},
    {"MiniMax-M3.1-Flash", LiftContract::kNoLiftedBytes},
    {"Mistral-Small-4-119B-2603-NVFP4", LiftContract::kNoLiftedBytes},
    {"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", LiftContract::kNoLiftedBytes},
    {"Ornith-1.5-35B-A3B", LiftContract::kNoLiftedBytes},
    {"phi-4", LiftContract::kNoLiftedBytes},
    {"Qwen3.5-9B", LiftContract::kNoLiftedBytes},
    {"Qwen3.6-35B-A3B", LiftContract::kNoLiftedBytes},
    {"Qwen3.8-27B", LiftContract::kNoLiftedBytes},
    {"Qwen3.8-Flash-Next", LiftContract::kNoLiftedBytes},
    {"Step-5-Preview", LiftContract::kNoLiftedBytes},
    {"Xing4.0-29B-A4B", LiftContract::kNoLiftedBytes},
};

const KnownFamily *FindKnown(const std::string &name)
{
    for (const auto &k : kKnownFamilies)
        if (name == k.model_dir) return &k;
    return nullptr;
}

std::string DecodeOne(const lsxcommon::ArrowTokenizerDb &db, std::int32_t id)
{
    std::size_t cap = 256;
    std::string out;
    for (int attempt = 0; attempt < 4; ++attempt) {
        out.assign(cap, '\0');
        std::size_t ocap = out.size();
        const std::int32_t rc =
            lsxcommon::arrow_tokenizer_decode(db, &id, 1, out.data(), &ocap);
        if (rc == 0) { out.resize(ocap); return out; }
        if (rc == -2 && ocap > out.size() && ocap <= (1u << 20)) {
            cap = ocap;
            continue;
        }
        return std::string();
    }
    return std::string();
}

std::int32_t Encode(const lsxcommon::ArrowTokenizerDb &db, const std::string &text)
{
    std::vector<std::int32_t> ids(1024);
    std::size_t cap = ids.size();
    return lsxcommon::arrow_tokenizer_encode(db, text.data(), text.size(),
                                             ids.data(), &cap);
}

// How many bytes the vocabulary stores under their GPT-2 bytes_to_unicode glyph
// instead of the byte itself.
//
// Counting two-byte entries by their lead byte would be wrong: real accented
// letters (U+0100..U+017F) share that encoding with lifted bytes, and
// gemma-4-26B-A4B-it carries 58 such genuine letters. The only unambiguous test
// is to compare the token that byte_to_token_id names against the canonical key
// for that byte — which is also exactly the decision the decoder has to get
// right, since it never reverses the mapping.
std::int32_t CountLiftedBytes(const lsxcommon::ArrowTokenizerDb &db)
{
    std::int32_t lifted = 0;
    for (int b = 0; b < 256; ++b) {
        const std::int32_t id = db.byte_to_token_id[b];
        if (id < 0 || id >= db.vocab_size || db.is_special[id]) continue;
        const std::int64_t off = db.token_offsets[id];
        const std::int64_t len = db.token_offsets[id + 1] - off;
        const auto raw = static_cast<uint8_t>(b);
        // Bytes that GPT-2 maps to themselves are never "lifted" — their key is
        // the byte, so comparing against it would count every printable ASCII
        // character as lifted. Only the 68 remapped bytes can be.
        if (lsxcommon::byte_unicode::byte_to_unicode(b) == static_cast<char32_t>(b))
            continue;
        const std::string key = lsxcommon::byte_unicode::bytes_to_key(&raw, 1);
        if (static_cast<std::size_t>(len) != key.size()) continue;
        if (std::memcmp(db.token_bytes + off, key.data(), key.size()) == 0)
            ++lifted;
    }
    return lifted;
}

} // namespace

TEST_CASE("every converted vocabulary keeps the decode->encode round-trip",
          "[tokenizer][vocab][real-model]")
{
    const fs::path models_root = lsxcommon::lsx_model_models_dir();
    if (!fs::exists(models_root)) {
        SKIP("models directory not present: " + models_root.string());
    }

    std::vector<fs::path> found;
    for (const auto &entry : fs::directory_iterator(models_root)) {
        if (!entry.is_directory()) continue;
        if (fs::exists(entry.path() / "pinned" /
                       "tokenizer_vocab.arrow"))
            found.push_back(entry.path());
    }
    if (found.empty()) {
        SKIP("no converted model with a pinned vocabulary under " +
             models_root.string());
    }

    for (const auto &root : found) {
        const std::string name = root.filename().string();
        CAPTURE(name);
        const KnownFamily *known = FindKnown(name);
        REQUIRE(known != nullptr);   // a new family must declare its contract

        auto vocab = arrow::io::MemoryMappedFile::Open(
            (root / "pinned" / "tokenizer_vocab.arrow").string(),
            arrow::io::FileMode::READ);
        REQUIRE(vocab.ok());
        auto merges = arrow::io::MemoryMappedFile::Open(
            (root / "pinned" / "tokenizer_merges.arrow").string(),
            arrow::io::FileMode::READ);
        REQUIRE(merges.ok());

        lsxcommon::ArrowTokenizerDb db;
        const auto loaded = lsxcommon::load_tokenizer_metadata(*vocab, *merges, db);
        REQUIRE(loaded.ok());
        REQUIRE(db.valid());

        // The contract itself: for every byte the model can emit, what the
        // decoder produces must be something the encoder accepts again.
        for (int b = 0; b < 256; ++b) {
            const std::int32_t id = db.byte_to_token_id[b];
            INFO("byte 0x" << std::hex << b << std::dec << " -> token " << id);
            if (id < 0) continue;          // not representable: nothing to emit
            const std::string text = DecodeOne(db, id);
            INFO("decoded " << text.size() << " byte(s)");
            CHECK(Encode(db, text) >= 0);
        }

        // And the declared lifting contract must match the vocabulary.
        const std::int32_t lifted = CountLiftedBytes(db);
        INFO("bytes stored under their GPT-2 glyph: " << lifted);
        if (known->lift == LiftContract::kNoLiftedBytes) {
            CHECK(lifted == 0);
        } else {
            CHECK(lifted > 0);
        }
    }
}