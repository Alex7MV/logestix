#include "lsxcommon/converter.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/byte_unicode.h"
#include "lsxcommon/log.h"
#include <arrow/api.h>
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <vector>

namespace lsxcommon {

arrow::Status BuildTokenizerArrowFiles(const std::string &tokenizer_json_path,
                                       const std::filesystem::path &pinned_dir,
                                       const TokenizerSpecialPolicy &policy)
{
    std::ifstream f(tokenizer_json_path, std::ios::binary | std::ios::ate);
    if (!f) return arrow::Status::IOError("cannot open ", tokenizer_json_path);
    const auto sz = f.tellg();
    f.seekg(0);
    std::string json(static_cast<size_t>(sz), '\0');
    f.read(json.data(), sz);

    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
        return arrow::Status::Invalid("tokenizer: JSON parse failed");
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("tokenizer: root is not an object");

    absl::flat_hash_map<std::string, int32_t> key_to_id;
    absl::flat_hash_map<int32_t, std::string> id_to_key;
    absl::flat_hash_set<int32_t> special_ids;
    absl::flat_hash_map<int32_t, std::string> added_content;
    bool has_added = false;

    // SentencePiece-style vocabularies (Gemma-4) mark word-leading spaces
    // with U+2581 '▁'. The engine operates in GPT-2 byte-glyph space where
    // byte 0x20 maps to U+0120 'Ġ'. Normalize SP markers into engine keys so
    // vocab entries and merges resolve against the byte-level BPE.
    // kind 11 (Gemma-4) reads the vocabulary VERBATIM: its keys are raw text
    // pieces ("▁the"), not GPT-2 byte glyphs, and its merges concatenate raw
    // keys. Every other kind normalizes SP markers into byte-glyph space.
    const bool raw_text = (policy.pretokenizer == 9);
    auto EngineKey = [raw_text](std::string_view k) {
        if (raw_text) return std::string(k);
        std::string out;
        out.reserve(k.size());
        for (size_t i = 0; i < k.size();) {
            if (i + 2 < k.size() &&
                static_cast<uint8_t>(k[i]) == 0xE2 &&
                static_cast<uint8_t>(k[i + 1]) == 0x96 &&
                static_cast<uint8_t>(k[i + 2]) == 0x81) {
                out += "\xC4\xA0"; // U+0120 'Ġ' == byte-glyph(0x20)
                i += 3;
            } else {
                out += k[i];
                ++i;
            }
        }
        return out;
    };

    // added_tokens precede "model" in the JSON. Read them first so the shared
    // simdjson token stream never rewinds past the model object.
    {
        simdjson::ondemand::array added_arr;
        if (root["added_tokens"].get_array().get(added_arr) == simdjson::SUCCESS) {
            has_added = true;
            for (auto at : added_arr) {
                simdjson::ondemand::object at_obj;
                if (at.get_object().get(at_obj) != simdjson::SUCCESS) continue;
                std::string_view content;
                int64_t id = -1;
                bool special = false;
                if (at_obj["id"].get_int64().get(id) != simdjson::SUCCESS)
                    continue;
                if (at_obj["content"].get_string().get(content) !=
                    simdjson::SUCCESS)
                    continue;
                if (at_obj["special"].get_bool().get(special) != simdjson::SUCCESS) special = false;
                if (id < 0) continue;
                auto i = static_cast<int32_t>(id);
                // Added/special tokens are stored VERBATIM: the SP '▁' ->
                // 'Ġ' normalization is a byte-level-BPE vocabulary concern
                // only, and rewriting control-token content (e.g.
                // "<|fim▁hole|>") would corrupt decode. (EngineKey is applied
                // to model.vocab entries below.)
                added_content[i] = content;
                key_to_id[std::string(content)] = i;
                id_to_key[i] = content;
                if (special) special_ids.insert(i);
            }
        }
    }

    // Fallback specials for byte-level models without added_tokens.
    if (!has_added) {
        special_ids.insert(0);
        special_ids.insert(1);
        special_ids.insert(2);
    }

    // The normalizer runs BEFORE the pre-tokenizer in the reference pipeline.
    // Only NFC is declared by the current checkpoints; record it so the
    // encoder can normalize (an empty/absent normalizer, or Gemma's Replace
    // space -> U+2581, leaves this false). Must be read before "model": the
    // simdjson on-demand stream is forward-only.
    bool normalize_nfc = false;
    {
        simdjson::ondemand::value nv;
        if (root["normalizer"].get(nv) == simdjson::SUCCESS) {
            std::string_view raw;
            if (nv.raw_json().get(raw) == simdjson::SUCCESS)
                normalize_nfc =
                    raw.find("\"NFC\"") != std::string_view::npos;
        }
    }

    simdjson::ondemand::object model_obj;
    if (root["model"].get_object().get(model_obj) != simdjson::SUCCESS)
        return arrow::Status::Invalid("tokenizer: no model object");

    std::string model_unk;
    bool model_byte_fallback = false;
    {
        // Field order in every HF tokenizer.json: unk_token ... byte_fallback
        // ... vocab ... merges; the on-demand stream is forward-only.
        std::string_view u;
        if (model_obj["unk_token"].get_string().get(u) == simdjson::SUCCESS)
            model_unk.assign(u);
        const auto byte_fallback_err =
            model_obj["byte_fallback"].get_bool().get(model_byte_fallback);
        (void)byte_fallback_err;
    }

    int32_t model_vocab_count = 0;
    {
        simdjson::ondemand::object vocab_obj;
        if (model_obj["vocab"].get_object().get(vocab_obj) != simdjson::SUCCESS)
            return arrow::Status::Invalid("tokenizer: no model.vocab object");
        for (auto entry : vocab_obj) {
            ++model_vocab_count;
            std::string_view key;
            int64_t id;
            if (entry.unescaped_key().get(key) != simdjson::SUCCESS) continue;
            if (entry.value().get_int64().get(id) != simdjson::SUCCESS) continue;
            auto k = EngineKey(key);
            auto i = static_cast<int32_t>(id);
            key_to_id[k] = i;
            if (id_to_key.find(i) == id_to_key.end())
                id_to_key[i] = std::move(k);
        }
    }

    // Authoritative special-token names come from <dir>/tokenizer_config.json
    // when present (bos_token / eos_token / pad_token, string or {content:...}).
    // GLM-5.2 omits bos_token entirely → no BOS prepend; DeepSeek names all
    // three explicitly.
    std::string cfg_bos, cfg_eos, cfg_pad;
    {
        auto cfg_path = std::filesystem::path(tokenizer_json_path).parent_path() /
                        "tokenizer_config.json";
        std::ifstream cf(cfg_path, std::ios::binary | std::ios::ate);
        if (cf) {
            const auto csz = cf.tellg();
            cf.seekg(0);
            std::string cjson(static_cast<size_t>(csz), '\0');
            cf.read(cjson.data(), csz);
            simdjson::padded_string cpad(cjson.data(), cjson.size());
            simdjson::ondemand::parser cparser;
            simdjson::ondemand::document cdoc;
            if (cparser.iterate(cpad).get(cdoc) == simdjson::SUCCESS) {
                simdjson::ondemand::object croot;
                if (cdoc.get_object().get(croot) == simdjson::SUCCESS) {
                    auto get_name = [&](const char *key,
                                        std::string &out) -> void {
                        simdjson::ondemand::value v;
                        if (croot[key].get(v) != simdjson::SUCCESS) return;
                        std::string_view sv;
                        if (v.get_string().get(sv) == simdjson::SUCCESS) {
                            out.assign(sv.data(), sv.size());
                        } else {
                            simdjson::ondemand::object obj;
                            if (v.get_object().get(obj) != simdjson::SUCCESS)
                                return;
                            if (obj["content"].get_string().get(sv) !=
                                simdjson::SUCCESS)
                                return;
                            out.assign(sv.data(), sv.size());
                        }
                    };
                    get_name("bos_token", cfg_bos);
                    get_name("eos_token", cfg_eos);
                    get_name("pad_token", cfg_pad);
                }
            }
        }
    }

    int32_t bos_id = -1, eos_id = -1, pad_id = -1;
    {
        // Only added tokens explicitly flagged special (e.g. DeepSeek's
        // "<|begin▁of▁sentence|>") may fill bos/eos/pad. Unflagged control
        // tokens ("<|fim▁begin|>") must never be picked.
        auto find_exact = [&](const std::string &name) -> int32_t {
            for (auto &[id, k] : added_content)
                if (k == name) return id;
            return -1;
        };
        auto find_special = [&](const char *needle) -> int32_t {
            for (auto &[id, k] : added_content)
                if (k.find(needle) != std::string::npos &&
                    special_ids.count(id) > 0)
                    return id;
            return -1;
        };

        // 1) Authoritative tokenizer_config.json names win when present.
        //    A name that is absent from added_tokens is treated as "no BOS"
        //    (stays -1), matching GLM-5.2, which must NOT auto-prepend.
        if (!cfg_bos.empty()) bos_id = find_exact(cfg_bos);
        if (!cfg_eos.empty()) eos_id = find_exact(cfg_eos);
        if (!cfg_pad.empty()) pad_id = find_exact(cfg_pad);

        // 2) Heuristic fallback for models without tokenizer_config.json.
        //    Never let generic "begin"/"end" substrings misclassify GLM's
        //    multimodal <|begin_of_*|> / <|end_of_*|> controls as sentence
        //    markers — match explicit BOS/EOS spellings only.
        if (bos_id < 0) {
            int32_t b = find_exact("<|bos|>");
            if (b < 0) b = find_exact("<s>");
            if (b < 0) b = find_exact("<|begin▁of▁sentence|>");
            if (b < 0) b = find_exact("<|begin_of_sentence|>");
            if (b < 0) b = find_special("sentence");
            bos_id = b;
        }
        if (eos_id < 0) {
            int32_t e = find_exact("<|eos|>");
            if (e < 0) e = find_exact("<|endoftext|>");
            if (e < 0) e = find_exact("</s>");
            if (e < 0) e = find_exact("<|eot_id|>");
            eos_id = e;
        }
        if (pad_id < 0) {
            int32_t p = find_exact("<|pad|>");
            if (p < 0) p = find_exact("<pad>");
            if (p < 0) p = find_special("pad");
            pad_id = p;
        }
    }

    // Family policy override (a checkpoint whose reference tokenizer emits no
    // BOS passes -1; see TokenizerSpecialPolicy).
    if (policy.bos_id != INT32_MIN) bos_id = policy.bos_id;
    if (policy.eos_id != INT32_MIN) eos_id = policy.eos_id;
    if (policy.pad_id != INT32_MIN) pad_id = policy.pad_id;

    // byte_to_token_id via vocab lookup of the GPT-2 byte table. Tokenizers
    // without full GPT-2 glyph coverage (e.g. Gemma-4: SentencePiece-style
    // <0xXX> byte-fallback tokens) fall back to the "<0xXX>" hex form.
    std::array<int32_t, 256> byte_to_token_id{};
    for (int b = 0; b < 256; ++b) {
        uint8_t b8 = static_cast<uint8_t>(b);
        std::string key = byte_unicode::bytes_to_key(&b8, 1);
        // Raw-text (kind 11) vocabularies are NOT written in GPT-2 glyph
        // space: a raw "Ø" piece is the letter, not byte 0xD8. Byte fallback
        // must therefore resolve through the "<0xHH>" pieces only (using the
        // glyph table here would silently map the byte to an ordinary letter
        // piece and corrupt every non-Latin script).
        auto it = raw_text ? key_to_id.end() : key_to_id.find(key);
        if (it == key_to_id.end()) {
            char hex[8];
            std::snprintf(hex, sizeof(hex), "<0x%02X>", b);
            it = key_to_id.find(hex);
            if (it == key_to_id.end()) {
                // Lowercase hex variant used by some tokenizers.
                char lox[8];
                std::snprintf(lox, sizeof(lox), "<0x%02x>", b);
                it = key_to_id.find(lox);
            }
            if (it == key_to_id.end()) {
                if (policy.drop_unmapped_bytes) {
                    // Reference-faithful: the checkpoint's vocabulary has no
                    // token for this byte and its encoder never emits it (CR is
                    // dropped, the rest are UTF-8-illegal lead bytes). The
                    // encoder SKIPS -1 entries.
                    byte_to_token_id[static_cast<size_t>(b)] = -1;
                    continue;
                }
                return arrow::Status::Invalid(
                    "tokenizer: byte token missing in vocab: '", key, "'");
            }
        }
        byte_to_token_id[static_cast<size_t>(b)] = it->second;
    }

    // Merge table via vocab lookup only.
    std::vector<MergeRow> merges;
    {
        simdjson::ondemand::array merges_arr;
        if (model_obj["merges"].get_array().get(merges_arr) == simdjson::SUCCESS) {
            int64_t rank = 0;
            for (auto m : merges_arr) {
                // Two merge encodings exist in the wild:
                //   DeepSeek:  "h e"            (single string, space-separated)
                //   GLM-5.2+:  ["h","e"]        (array of two strings, new HF)
                std::string A, B;
                if (m.is_string()) {
                    std::string_view sv;
                    if (m.get_string().get(sv) != simdjson::SUCCESS) continue;
                    auto s = std::string(sv);
                    auto sp = s.find(' ');
                    if (sp == std::string::npos) continue;
                    A = s.substr(0, sp);
                    B = s.substr(sp + 1);
                } else {
                    simdjson::ondemand::array pair;
                    if (m.get_array().get(pair) != simdjson::SUCCESS) continue;
                    simdjson::ondemand::array_iterator it = pair.begin();
                    if (it == pair.end()) continue;
                    std::string_view a_sv, b_sv;
                    if ((*it).get_string().get(a_sv) != simdjson::SUCCESS) continue;
                    ++it;
                    if (it == pair.end()) continue;
                    if ((*it).get_string().get(b_sv) != simdjson::SUCCESS) continue;
                    A = std::string(a_sv);
                    B = std::string(b_sv);
                }

                auto itL = key_to_id.find(EngineKey(A));
                auto itR = key_to_id.find(EngineKey(B));
                // Gemma-4-style tokenizers carry raw-text (non byte-glyph)
                // merges and occasional empty operands. Such merges cannot be
                // represented in the byte-level engine; skip them (ASCII text
                // still encodes exactly; other scripts fall back to byte
                // tokens) instead of failing the whole conversion.
                if (itL == key_to_id.end() || itR == key_to_id.end() ||
                    A.empty() || B.empty())
                    continue;

                std::string merged_key;
                if (raw_text) {
                    // Raw-text BPE: the merged piece is the literal
                    // concatenation of the two operands.
                    merged_key = A + B;
                } else {
                    std::vector<uint8_t> bytesA, bytesB;
                    if (!byte_unicode::key_to_bytes(EngineKey(A), bytesA) ||
                        !byte_unicode::key_to_bytes(EngineKey(B), bytesB))
                        continue;
                    std::vector<uint8_t> merged_bytes = bytesA;
                    merged_bytes.insert(merged_bytes.end(), bytesB.begin(),
                                        bytesB.end());
                    merged_key = byte_unicode::bytes_to_key(
                        merged_bytes.data(), merged_bytes.size());
                }
                auto itM = key_to_id.find(merged_key);
                if (itM == key_to_id.end()) continue;

                MergeRow r;
                r.merge_left = itL->second;
                r.merge_right = itR->second;
                r.merge_rank = static_cast<int32_t>(rank++);
                r.merge_token_id = itM->second;
                merges.push_back(r);
            }
        }
    }
    std::sort(merges.begin(), merges.end(), [](const MergeRow &a, const MergeRow &b) {
        if (a.merge_left != b.merge_left) return a.merge_left < b.merge_left;
        return a.merge_right < b.merge_right;
    });

    // Vocab rows indexed by token_id.
    int32_t vocab_size = 0;
    for (auto &[id, k] : id_to_key) vocab_size = std::max(vocab_size, id + 1);

    std::vector<VocabRow> vocab_rows;
    vocab_rows.resize(static_cast<size_t>(vocab_size));
    for (int32_t id = 0; id < vocab_size; ++id) {
        auto it = id_to_key.find(id);
        if (it == id_to_key.end())
            return arrow::Status::Invalid(absl::StrCat("tokenizer: missing vocab id ", id));
        auto &row = vocab_rows[static_cast<size_t>(id)];
        row.token_id = id;

        std::vector<uint8_t> bytes;
        bool special = special_ids.count(id) > 0;
        if (raw_text || !byte_unicode::key_to_bytes(it->second, bytes)) {
            // Control tokens (e.g. DeepSeek's "<|fim▁hole|>") are shipped in
            // added_tokens with "special": false but contain chars outside the
            // GPT-2 byte table. They can never be produced by the byte-level
            // encoder, so keep them as ordinary tokens whose token_bytes carry
            // the raw UTF-8 text; decode must emit them verbatim (no stripping).
            if (!special && !raw_text)
                lsxcommon::log::debug(absl::StrCat(
                    "tokenizer: non-byte-level token id=", id,
                    " kept as text: '", it->second, "'"));
            row.is_special = special ? 1 : 0;
            row.token_bytes.assign(
                reinterpret_cast<const uint8_t *>(it->second.data()),
                reinterpret_cast<const uint8_t *>(it->second.data()) + it->second.size());
        } else {
            row.is_special = special ? 1 : 0;
            row.token_bytes = std::move(bytes);
        }
    }

    TokenizerVocabMeta meta;
    meta.bos_id = bos_id;
    meta.eos_id = eos_id;
    meta.pad_id = pad_id;
    meta.vocab_size = vocab_size;
    meta.byte_to_token_id = byte_to_token_id;
    meta.drop_unmapped_bytes = policy.drop_unmapped_bytes;
    meta.normalize_nfc = normalize_nfc ? 1 : 0;
    // kind 0 with a known base-vocabulary size: ids at or above it are ADDED
    // tokens, which the reference encoder matches literally before BPE. Without
    // the size the classic path stays pure-BPE (every existing family).
    if (policy.pretokenizer == 0 && model_vocab_count > 0 &&
        model_vocab_count < vocab_size)
        meta.base_vocab_size = model_vocab_count;
    if (policy.pretokenizer == 1) {
        // Llama-3/o200k split over a separate BPE table. `base_vocab_size` is
        // the MODEL vocab's rank space: ids at or above it are added tokens the
        // encoder matches literally (the chat markers must never be BPE'd as
        // raw bytes), and merging is table-driven over the rank space.
        meta.tokenizer_kind = 3;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 2) {
        // Flint: the same table-driven BPE with a 3-split pre-tokenizer
        // (\p{N}{1,3} Isolated, CJK Isolated, then the o200k case-run pattern
        // with Flint's [\p{P}\p{S}]+ tail). Written as tokenizer_kind = 4.
        meta.tokenizer_kind = 4;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 3) {
        // Ling: one letter run with an optional prefix, one digit per number
        // word, apostrophe contractions and a punctuation run.
        // Written as tokenizer_kind = 5.
        meta.tokenizer_kind = 5;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 4) {
        // Beam: the Qwen2/3 single-run pattern (one letter run, one digit per
        // number word, Qwen contraction list, \s*[\r\n]+). Written as
        // tokenizer_kind = 6. The Harmony control markers are added tokens
        // (special:false) and are matched literally by the encoder.
        meta.tokenizer_kind = 6;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 5) {
        // Qwen3.5/3.6/3.8/Flash-Next and Ornith: ONE split stage with
        // [:L:][:M:] letter runs (Han is NOT isolated), ONE digit per number
        // word and a standalone contraction alternative. Written as
        // tokenizer_kind = 7.
        meta.tokenizer_kind = 7;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 6) {
        // GLM-5.3-Flash / phi-4 / Granite-Switch: ONE split stage, GPT-2-style
        // letter run (no [:M:]), digit groups of up to three, standalone
        // contraction alternative. Written as tokenizer_kind = 8.
        meta.tokenizer_kind = 8;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 7) {
        // Mistral-Small-4 / Nemotron-3.5: o200k case-run alternatives WITHOUT
        // the contraction suffix, ONE digit per number word, [\r\n/]* after
        // the punctuation run. Written as tokenizer_kind = 9.
        meta.tokenizer_kind = 9;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 8) {
        // Granite-4.2: the original GPT-2 ByteLevel pattern (case-SENSITIVE
        // contractions, unbounded letter/digit runs, a literal space prefix,
        // no CRLF alternative). Written as tokenizer_kind = 10.
        meta.tokenizer_kind = 10;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 10) {
        // Mistral-Large-4: the Small-4 case-run pattern with o200k's
        // RIGHT-grouped number alternative (\p{N}{1,3} aligned to the run's
        // right end). Written as tokenizer_kind = 12.
        meta.tokenizer_kind = 12;
        meta.base_vocab_size = model_vocab_count;
    }
    if (policy.pretokenizer == 9) {
        // Gemma-4: raw-text BPE (vocabulary + merges verbatim), normalizer
        // Replace(" " -> U+2581), ByteFallback, no dummy prefix. Written as
        // tokenizer_kind = 11.
        meta.tokenizer_kind = 11;
        meta.base_vocab_size = model_vocab_count;
        meta.byte_fallback = model_byte_fallback ? 1 : 0;
        meta.add_dummy_prefix = 0;
        meta.escape_whitespaces = 1;
        if (!model_unk.empty()) {
            auto it = key_to_id.find(model_unk);
            if (it != key_to_id.end()) meta.unk_id = it->second;
        }
    }

    auto vbatch = TokenizerVocabSchema::MakeBatch(vocab_rows, meta);
    if (!vbatch.ok()) return vbatch.status();
    auto mbatch = TokenizerMergesSchema::MakeBatch(merges,
        static_cast<int32_t>(merges.size()));
    if (!mbatch.ok()) return mbatch.status();

    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*vbatch, pinned_dir / "tokenizer_vocab.arrow"));
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*mbatch, pinned_dir / "tokenizer_merges.arrow"));
    return arrow::Status::OK();
}


// ── TikToken bridge (Kimi-K2.7-Code) ───────────────────────────────────────

namespace {

// RFC 4648 base64 decode (standard alphabet, '=' padding).
bool Base64Decode(const std::string &in, std::vector<uint8_t> &out)
{
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear();
    int buf = 0, bits = 0;
    for (char c : in) {
        if (c == '=') break;
        int v = val(c);
        if (v < 0) return false;
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return true;
}

} // namespace

arrow::Status BuildTokenizerArrowFilesTikToken(
    const std::string &tiktoken_model_path,
    const std::string &tokenizer_config_path,
    const std::filesystem::path &pinned_dir,
    const TikTokenSpecialPolicy &policy)
{
    absl::flat_hash_map<std::string, int32_t> key_to_id;
    absl::flat_hash_map<int32_t, std::string> id_to_key;
    absl::flat_hash_set<int32_t> special_ids;

    // (1) Base ranks: "base64 id" lines; ids may be sparse but normally are
    // dense 0..N-1.
    {
        std::ifstream f(tiktoken_model_path);
        if (!f)
            return arrow::Status::IOError("cannot open ", tiktoken_model_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::vector<uint8_t> bytes;
            if (!Base64Decode(line.substr(0, sp), bytes))
                return arrow::Status::Invalid(
                    "tokenizer: bad base64 in ", tiktoken_model_path);
            int64_t id = -1;
            if (!absl::SimpleAtoi(line.substr(sp + 1), &id) || id < 0)
                return arrow::Status::Invalid("tokenizer: bad rank line");
            auto key = byte_unicode::bytes_to_key(bytes.data(), bytes.size());
            auto i = static_cast<int32_t>(id);
            key_to_id[key] = i;
            id_to_key[i] = std::move(key);
        }
    }

    const int32_t base_rank_count = static_cast<int32_t>(key_to_id.size());

    // (2) Special tokens from tokenizer_config.json added_tokens_decoder +
    // authoritative bos/eos/pad names.
    int32_t bos_id = -1, eos_id = -1, pad_id = -1;
    {
        std::ifstream cf(tokenizer_config_path);
        if (!cf)
            return arrow::Status::IOError("cannot open ",
                                          tokenizer_config_path);
        std::string cj((std::istreambuf_iterator<char>(cf)),
                       std::istreambuf_iterator<char>());
        simdjson::padded_string cpad(cj.data(), cj.size());
        simdjson::ondemand::parser cparser;
        simdjson::ondemand::document cdoc;
        if (cparser.iterate(cpad).get(cdoc) != simdjson::SUCCESS)
            return arrow::Status::Invalid(
                "tokenizer_config.json parse failed");
        simdjson::ondemand::object croot;
        if (cdoc.get_object().get(croot) != simdjson::SUCCESS)
            return arrow::Status::Invalid(
                "tokenizer_config.json root not an object");

        simdjson::ondemand::object atd;
        if (croot["added_tokens_decoder"].get_object().get(atd) ==
            simdjson::SUCCESS) {
            for (simdjson::ondemand::field field : atd) {
                std::string_view key_sv;
                int64_t id = -1;
                if (field.unescaped_key().get(key_sv) == simdjson::SUCCESS)
                    (void)absl::SimpleAtoi(std::string(key_sv), &id);
                simdjson::ondemand::object obj;
                if (field.value().get_object().get(obj) != simdjson::SUCCESS)
                    continue;
                std::string_view content;
                if (obj["content"].get_string().get(content) !=
                    simdjson::SUCCESS)
                    continue;
                bool special = false;
                const auto special_err =
                    obj["special"].get_bool().get(special);
                (void)special_err;
                auto i = static_cast<int32_t>(id);
                key_to_id[std::string(content)] = i;
                id_to_key[i] = std::string(content);
                if (special) special_ids.insert(i);
            }
        }

        auto find_exact = [&](const std::string &name) -> int32_t {
            auto it = key_to_id.find(name);
            return it == key_to_id.end() ? -1 : it->second;
        };
        std::string_view name_sv;
        if (croot["bos_token"].get_string().get(name_sv) == simdjson::SUCCESS)
            bos_id = find_exact(std::string(name_sv));
        if (croot["eos_token"].get_string().get(name_sv) == simdjson::SUCCESS)
            eos_id = find_exact(std::string(name_sv));
        if (croot["pad_token"].get_string().get(name_sv) == simdjson::SUCCESS)
            pad_id = find_exact(std::string(name_sv));
    }

    // (3) Merge reconstruction: walk base tokens by ascending id; every
    // multi-byte token decomposes into the pair of lower-id pieces that
    // minimizes the max piece id (BPE priority order). Specials never merge.
    std::vector<MergeRow> merges;
    {
        std::vector<int32_t> ids;
        ids.reserve(id_to_key.size());
        for (auto &[id, k] : id_to_key)
            if (special_ids.count(id) == 0) ids.push_back(id);
        std::sort(ids.begin(), ids.end());

        int64_t rank = 0;
        for (int32_t id : ids) {
            const std::string &key = id_to_key[id];
            std::vector<uint8_t> bytes;
            if (!byte_unicode::key_to_bytes(key, bytes) || bytes.size() < 2)
                continue;
            int32_t best_a = -1, best_b = -1;
            int32_t best_max = INT32_MAX;
            for (size_t split = 1; split < bytes.size(); ++split) {
                std::string ka = byte_unicode::bytes_to_key(bytes.data(),
                                                            split);
                std::string kb =
                    byte_unicode::bytes_to_key(bytes.data() + split,
                                               bytes.size() - split);
                auto ia = key_to_id.find(ka);
                auto ib = key_to_id.find(kb);
                if (ia == key_to_id.end() || ib == key_to_id.end()) continue;
                if (ia->second >= id || ib->second >= id) continue;
                int32_t mx = std::max(ia->second, ib->second);
                if (mx < best_max ||
                    (mx == best_max &&
                     std::min(ia->second, ib->second) <
                         std::min(best_a, best_b))) {
                    best_max = mx;
                    best_a = ia->second;
                    best_b = ib->second;
                }
            }
            if (best_a < 0) {
                // Unreachable-from-bytes token (no binary split whose both
                // halves are lower-id vocab members). It can never be
                // produced by pairwise BPE, so skip it from the merges table;
                // encoding of ordinary text is unaffected.
                lsxcommon::log::debug(absl::StrCat(
                    "tokenizer: no valid BPE split for id ", id,
                    " (unreachable; skipped)"));
                continue;
            }
            MergeRow r;
            r.merge_left = best_a;
            r.merge_right = best_b;
            r.merge_rank = static_cast<int32_t>(rank++);
            r.merge_token_id = id;
            merges.push_back(r);
        }
        std::sort(merges.begin(), merges.end(),
                  [](const MergeRow &a, const MergeRow &b) {
                      if (a.merge_left != b.merge_left)
                          return a.merge_left < b.merge_left;
                      return a.merge_right < b.merge_right;
                  });
    }

    // (4) byte_to_token_id via single-byte vocab lookup.
    std::array<int32_t, 256> byte_to_token_id{};
    for (int b = 0; b < 256; ++b) {
        uint8_t b8 = static_cast<uint8_t>(b);
        std::string key = byte_unicode::bytes_to_key(&b8, 1);
        auto it = key_to_id.find(key);
        if (it == key_to_id.end())
            return arrow::Status::Invalid(
                "tokenizer: byte token missing in vocab: '", key, "'");
        byte_to_token_id[static_cast<size_t>(b)] = it->second;
    }

    // (5) Vocab rows indexed by token id; sparse gaps get placeholder rows.
    int32_t vocab_size = 0;
    for (auto &[id, k] : id_to_key) vocab_size = std::max(vocab_size, id + 1);

    std::vector<VocabRow> vocab_rows;
    vocab_rows.resize(static_cast<size_t>(vocab_size));
    for (int32_t id = 0; id < vocab_size; ++id) {
        auto it = id_to_key.find(id);
        auto &row = vocab_rows[static_cast<size_t>(id)];
        row.token_id = id;
        if (it == id_to_key.end()) {
            row.is_special = 0;   // gap placeholder, never referenced
            continue;
        }
        row.is_special = special_ids.count(id) > 0 ? 1 : 0;
        std::vector<uint8_t> bytes;
        if (!byte_unicode::key_to_bytes(it->second, bytes)) {
            row.token_bytes.assign(
                reinterpret_cast<const uint8_t *>(it->second.data()),
                reinterpret_cast<const uint8_t *>(it->second.data()) +
                    it->second.size());
        } else {
            row.token_bytes = std::move(bytes);
        }
    }

    // Family policy overrides (a checkpoint whose reference tokenizer does not
    // add BOS, or whose stop id differs from its tokenizer_config entry).
    if (policy.bos_id != INT32_MIN) bos_id = policy.bos_id;
    if (policy.eos_id != INT32_MIN) eos_id = policy.eos_id;
    if (policy.pad_id != INT32_MIN) pad_id = policy.pad_id;

    TokenizerVocabMeta meta;
    meta.bos_id = bos_id;
    meta.eos_id = eos_id;
    meta.pad_id = pad_id;
    meta.vocab_size = vocab_size;
    meta.byte_to_token_id = byte_to_token_id;
    // tiktoken: the vocabulary is the merge table and ids >= base_vocab_size
    // are added tokens matched literally before BPE.
    meta.tokenizer_kind = 2;
    meta.base_vocab_size = base_rank_count;


    auto vbatch = TokenizerVocabSchema::MakeBatch(vocab_rows, meta);
    if (!vbatch.ok()) return vbatch.status();
    auto mbatch = TokenizerMergesSchema::MakeBatch(
        merges, static_cast<int32_t>(merges.size()));
    if (!mbatch.ok()) return mbatch.status();

    ARROW_RETURN_NOT_OK(
        WriteRecordBatchToFile(*vbatch, pinned_dir / "tokenizer_vocab.arrow"));
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(
        *mbatch, pinned_dir / "tokenizer_merges.arrow"));
    return arrow::Status::OK();
}

// ── SentencePiece-BPE bridge (AliceAI) ──────────────────────────────────────

namespace {

struct SpPiece {
    std::string bytes;
    float score = 0.0f;
    uint32_t type = 1;  // 1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte
};

struct PbCursor {
    const uint8_t *p = nullptr;
    const uint8_t *end = nullptr;
};

bool PbVarint(PbCursor &c, uint64_t &v)
{
    v = 0;
    int shift = 0;
    while (c.p < c.end && shift < 64) {
        uint8_t b = *c.p++;
        v |= static_cast<uint64_t>(b & 0x7f) << shift;
        shift += 7;
        if (!(b & 0x80)) return true;
    }
    return false;
}

bool PbLen(PbCursor &c, const uint8_t *&data, size_t &len)
{
    uint64_t n;
    if (!PbVarint(c, n)) return false;
    if (c.p + n > c.end) return false;
    data = c.p;
    len = static_cast<size_t>(n);
    c.p += n;
    return true;
}

void PbSkip(PbCursor &c, uint32_t wire)
{
    switch (wire) {
    case 0: { uint64_t v; (void)PbVarint(c, v); break; }
    case 1: c.p += 8; break;
    case 2: { const uint8_t *d; size_t l; (void)PbLen(c, d, l); break; }
    case 5: c.p += 4; break;
    default: c.p = c.end; break;
    }
    if (c.p > c.end) c.p = c.end;
}

bool ParseSpPiece(const uint8_t *d, size_t n, SpPiece &out)
{
    PbCursor c{d, d + n};
    while (c.p < c.end) {
        uint64_t tag;
        if (!PbVarint(c, tag)) return false;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        if (field == 1 && wire == 2) {
            const uint8_t *p;
            size_t l;
            if (!PbLen(c, p, l)) return false;
            out.bytes.assign(reinterpret_cast<const char *>(p), l);
        } else if (field == 2 && wire == 5) {
            if (c.p + 4 > c.end) return false;
            uint32_t bits = static_cast<uint32_t>(c.p[0]) |
                            (static_cast<uint32_t>(c.p[1]) << 8) |
                            (static_cast<uint32_t>(c.p[2]) << 16) |
                            (static_cast<uint32_t>(c.p[3]) << 24);
            std::memcpy(&out.score, &bits, 4);
            c.p += 4;
        } else if (field == 3 && wire == 0) {
            uint64_t v;
            if (!PbVarint(c, v)) return false;
            out.type = static_cast<uint32_t>(v);
        } else {
            PbSkip(c, wire);
        }
    }
    return true;
}

int SpUtf8Count(const std::string &s)
{
    int n = 0;
    for (size_t i = 0; i < s.size();) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        size_t l = 1;
        if ((c >> 5) == 0x6) l = 2;
        else if ((c >> 4) == 0xE) l = 3;
        else if ((c >> 3) == 0x1E) l = 4;
        i += l;
        ++n;
    }
    return n;
}

} // namespace

arrow::Status BuildTokenizerArrowFilesSentencePiece(
    const std::string &tokenizer_model_path,
    const std::filesystem::path &pinned_dir,
    const SpSpecialTokens &specials)
{
    std::ifstream f(tokenizer_model_path, std::ios::binary | std::ios::ate);
    if (!f)
        return arrow::Status::IOError("tokenizer: cannot open ",
                                      tokenizer_model_path);
    const auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char *>(data.data()), sz);

    std::vector<SpPiece> pieces;
    uint32_t model_type = 0;
    bool byte_fallback = false;
    bool add_dummy_prefix = true;
    bool escape_whitespaces = true;
    bool have_byte_fallback = false, have_dummy = false, have_escape = false;
    std::string norm_name;

    PbCursor top{data.data(), data.data() + data.size()};
    while (top.p < top.end) {
        uint64_t tag;
        if (!PbVarint(top, tag)) break;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        if (field == 1 && wire == 2) {
            const uint8_t *d;
            size_t l;
            if (!PbLen(top, d, l)) break;
            SpPiece p;
            if (!ParseSpPiece(d, l, p))
                return arrow::Status::Invalid("tokenizer: bad sentencepiece piece");
            pieces.push_back(std::move(p));
        } else if (field == 2 && wire == 2) {
            const uint8_t *d;
            size_t l;
            if (!PbLen(top, d, l)) break;
            PbCursor t{d, d + l};
            while (t.p < t.end) {
                uint64_t tg;
                if (!PbVarint(t, tg)) break;
                uint32_t tf = static_cast<uint32_t>(tg >> 3);
                uint32_t tw = static_cast<uint32_t>(tg & 7);
                if (tf == 3 && tw == 0) {
                    uint64_t v;
                    if (!PbVarint(t, v)) break;
                    model_type = static_cast<uint32_t>(v);
                } else if (tf == 35 && tw == 0) {
                    uint64_t v;
                    if (!PbVarint(t, v)) break;
                    byte_fallback = (v != 0);
                    have_byte_fallback = true;
                } else {
                    PbSkip(t, tw);
                }
            }
        } else if (field == 3 && wire == 2) {
            const uint8_t *d;
            size_t l;
            if (!PbLen(top, d, l)) break;
            PbCursor t{d, d + l};
            while (t.p < t.end) {
                uint64_t tg;
                if (!PbVarint(t, tg)) break;
                uint32_t tf = static_cast<uint32_t>(tg >> 3);
                uint32_t tw = static_cast<uint32_t>(tg & 7);
                if (tf == 1 && tw == 2) {
                    const uint8_t *p;
                    size_t pl;
                    if (!PbLen(t, p, pl)) break;
                    norm_name.assign(reinterpret_cast<const char *>(p), pl);
                } else if (tf == 3 && tw == 0) {
                    uint64_t v;
                    if (!PbVarint(t, v)) break;
                    add_dummy_prefix = (v != 0);
                    have_dummy = true;
                } else if (tf == 5 && tw == 0) {
                    uint64_t v;
                    if (!PbVarint(t, v)) break;
                    escape_whitespaces = (v != 0);
                    have_escape = true;
                } else {
                    PbSkip(t, tw);
                }
            }
        } else {
            PbSkip(top, wire);
        }
    }
    (void)have_byte_fallback;
    (void)have_dummy;
    (void)have_escape;

    if (pieces.empty())
        return arrow::Status::Invalid("tokenizer: no sentencepiece pieces parsed");
    if (model_type != 2)
        return arrow::Status::Invalid(
            "tokenizer: only SentencePiece BPE (model_type=2) is supported");
    if (!norm_name.empty() && norm_name != "identity")
        return arrow::Status::Invalid(
            "tokenizer: only the identity normalizer is supported, got ",
            norm_name);

    const int32_t vocab_size = static_cast<int32_t>(pieces.size());
    absl::flat_hash_map<std::string, int32_t> piece_to_id;
    piece_to_id.reserve(static_cast<size_t>(vocab_size) * 2);
    for (int32_t id = 0; id < vocab_size; ++id) {
        auto [it, inserted] = piece_to_id.emplace(pieces[id].bytes, id);
        (void)it;
        (void)inserted;
    }

    auto find_piece = [&](const std::string &s) -> int32_t {
        auto it = piece_to_id.find(s);
        return it == piece_to_id.end() ? -1 : it->second;
    };

    std::array<int32_t, 256> byte_to_token_id{};
    for (int b = 0; b < 256; ++b) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "<0x%02X>", b);
        int32_t id = find_piece(hex);
        if (id < 0) {
            std::snprintf(hex, sizeof(hex), "<0x%02x>", b);
            id = find_piece(hex);
        }
        if (id < 0)
            return arrow::Status::Invalid(
                "tokenizer: byte-fallback piece missing for byte ", b);
        byte_to_token_id[static_cast<size_t>(b)] = id;
    }

    // Derive BPE merge rules exactly as HF's generate_merges: for every piece
    // emit every split (a,b) into two existing pieces with the piece's own
    // score, order splits within a piece by (id(a),id(b)), then stable-sort all
    // candidates by (score desc, len(a) desc, len(b) desc); the resulting index
    // is the merge rank.
    struct Candidate {
        float score;
        int32_t len_l;
        int32_t len_r;
        int32_t left;
        int32_t right;
        int32_t token;
    };
    std::vector<Candidate> candidates;
    for (int32_t id = 0; id < vocab_size; ++id) {
        const SpPiece &p = pieces[id];
        if (p.bytes.size() < 2) continue;
        std::vector<Candidate> local;
        for (size_t k = 1; k < p.bytes.size(); ++k) {
            std::string a = p.bytes.substr(0, k);
            std::string b = p.bytes.substr(k);
            int32_t ia = find_piece(a);
            int32_t ib = find_piece(b);
            if (ia < 0 || ib < 0) continue;
            local.push_back({p.score, SpUtf8Count(a), SpUtf8Count(b), ia, ib, id});
        }
        std::sort(local.begin(), local.end(),
                  [](const Candidate &x, const Candidate &y) {
                      if (x.left != y.left) return x.left < y.left;
                      return x.right < y.right;
                  });
        candidates.insert(candidates.end(), local.begin(), local.end());
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate &a, const Candidate &b) {
                         if (a.score != b.score) return a.score > b.score;
                         if (a.len_l != b.len_l) return a.len_l > b.len_l;
                         return a.len_r > b.len_r;
                     });

    std::vector<MergeRow> merges;
    merges.reserve(candidates.size());
    for (size_t rank = 0; rank < candidates.size(); ++rank) {
        const Candidate &c = candidates[rank];
        MergeRow r;
        r.merge_left = c.left;
        r.merge_right = c.right;
        r.merge_rank = static_cast<int32_t>(rank);
        r.merge_token_id = c.token;
        merges.push_back(r);
    }
    std::sort(merges.begin(), merges.end(),
              [](const MergeRow &a, const MergeRow &b) {
                  if (a.merge_left != b.merge_left)
                      return a.merge_left < b.merge_left;
                  return a.merge_right < b.merge_right;
              });

    std::vector<VocabRow> vocab_rows;
    vocab_rows.resize(static_cast<size_t>(vocab_size));
    for (int32_t id = 0; id < vocab_size; ++id) {
        auto &row = vocab_rows[static_cast<size_t>(id)];
        row.token_id = id;
        const SpPiece &p = pieces[id];
        row.is_special = (p.type == 2 || p.type == 3) ? 1 : 0;
        row.token_bytes.assign(
            reinterpret_cast<const uint8_t *>(p.bytes.data()),
            reinterpret_cast<const uint8_t *>(p.bytes.data()) + p.bytes.size());
    }

    TokenizerVocabMeta meta;
    meta.bos_id = find_piece(specials.bos);
    meta.eos_id = find_piece(specials.eos);
    meta.pad_id = find_piece(specials.pad);
    meta.unk_id = find_piece(specials.unk);
    if (meta.unk_id < 0) meta.unk_id = 0;
    meta.vocab_size = vocab_size;
    meta.byte_to_token_id = byte_to_token_id;
    meta.tokenizer_kind = 1;
    meta.byte_fallback = byte_fallback ? 1 : 0;
    meta.add_dummy_prefix = add_dummy_prefix ? 1 : 0;
    meta.escape_whitespaces = escape_whitespaces ? 1 : 0;

    auto vbatch = TokenizerVocabSchema::MakeBatch(vocab_rows, meta);
    if (!vbatch.ok()) return vbatch.status();
    auto mbatch = TokenizerMergesSchema::MakeBatch(
        merges, static_cast<int32_t>(merges.size()));
    if (!mbatch.ok()) return mbatch.status();

    ARROW_RETURN_NOT_OK(
        WriteRecordBatchToFile(*vbatch, pinned_dir / "tokenizer_vocab.arrow"));
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(
        *mbatch, pinned_dir / "tokenizer_merges.arrow"));
    return arrow::Status::OK();
}
} // namespace lsxcommon
