#include "lsxcommon/converter.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/byte_unicode.h"
#include "lsxcommon/log.h"
#include <arrow/api.h>
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <climits>
#include <filesystem>
#include <fstream>
#include <string>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <vector>

namespace lsxcommon {

arrow::Status BuildTokenizerArrowFiles(const std::string &tokenizer_json_path,
                                       const std::filesystem::path &pinned_dir)
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
                at_obj["id"].get_int64().get(id);
                at_obj["content"].get_string().get(content);
                if (at_obj["special"].get_bool().get(special) != simdjson::SUCCESS) special = false;
                if (id < 0) continue;
                auto i = static_cast<int32_t>(id);
                auto k = std::string(content);
                added_content[i] = k;
                key_to_id[k] = i;
                id_to_key[i] = k;
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

    simdjson::ondemand::object model_obj;
    if (root["model"].get_object().get(model_obj) != simdjson::SUCCESS)
        return arrow::Status::Invalid("tokenizer: no model object");

    {
        simdjson::ondemand::object vocab_obj;
        if (model_obj["vocab"].get_object().get(vocab_obj) != simdjson::SUCCESS)
            return arrow::Status::Invalid("tokenizer: no model.vocab object");
        for (auto entry : vocab_obj) {
            std::string_view key;
            int64_t id;
            if (entry.unescaped_key().get(key) != simdjson::SUCCESS) continue;
            if (entry.value().get_int64().get(id) != simdjson::SUCCESS) continue;
            auto k = std::string(key);
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

    // byte_to_token_id via vocab lookup of the GPT-2 byte table.
    std::array<int32_t, 256> byte_to_token_id{};
    for (int b = 0; b < 256; ++b) {
        uint8_t b8 = static_cast<uint8_t>(b);
        std::string key = byte_unicode::bytes_to_key(&b8, 1);
        auto it = key_to_id.find(key);
        if (it == key_to_id.end())
            return arrow::Status::Invalid("tokenizer: byte token missing in vocab: '", key, "'");
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

                auto itL = key_to_id.find(A);
                auto itR = key_to_id.find(B);
                if (itL == key_to_id.end() || itR == key_to_id.end())
                    return arrow::Status::Invalid("tokenizer: merge operand missing in vocab: '", A, "'/'", B, "'");

                std::vector<uint8_t> bytesA, bytesB;
                if (!byte_unicode::key_to_bytes(A, bytesA) ||
                    !byte_unicode::key_to_bytes(B, bytesB))
                    return arrow::Status::Invalid("tokenizer: merge operand not byte-level: '", A, "'/'", B, "'");

                std::vector<uint8_t> merged_bytes = bytesA;
                merged_bytes.insert(merged_bytes.end(), bytesB.begin(), bytesB.end());
                std::string merged_key =
                    byte_unicode::bytes_to_key(merged_bytes.data(), merged_bytes.size());
                auto itM = key_to_id.find(merged_key);
                if (itM == key_to_id.end())
                    return arrow::Status::Invalid("tokenizer: merge product missing in vocab: '",
                                                  merged_key, "'");

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
        if (!byte_unicode::key_to_bytes(it->second, bytes)) {
            // Control tokens (e.g. DeepSeek's "<|fim▁hole|>") are shipped in
            // added_tokens with "special": false but contain chars outside the
            // GPT-2 byte table. They can never be produced by the byte-level
            // encoder, so keep them as ordinary tokens whose token_bytes carry
            // the raw UTF-8 text; decode must emit them verbatim (no stripping).
            if (!special)
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
    const std::filesystem::path &pinned_dir)
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
                (void)obj["special"].get_bool().get(special);
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
            if (best_a < 0)
                return arrow::Status::Invalid(
                    "tokenizer: no valid BPE split for id ", id);
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

    TokenizerVocabMeta meta;
    meta.bos_id = bos_id;
    meta.eos_id = eos_id;
    meta.pad_id = pad_id;
    meta.vocab_size = vocab_size;
    meta.byte_to_token_id = byte_to_token_id;

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
