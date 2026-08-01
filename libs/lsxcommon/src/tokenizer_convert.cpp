#include "lsxcommon/converter.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/byte_unicode.h"
#include <arrow/api.h>
#include <simdjson.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

    std::unordered_map<std::string, int32_t> key_to_id;
    std::unordered_map<int32_t, std::string> id_to_key;
    std::unordered_set<int32_t> special_ids;
    std::unordered_map<int32_t, std::string> added_content;
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

    int32_t bos_id = 0, eos_id = 1, pad_id = 2;
    {
        // Only added tokens explicitly flagged special (e.g. DeepSeek's
        // "<|begin▁of▁sentence|>") may fill bos/eos/pad. Unflagged control
        // tokens ("<|fim▁begin|>") must never be picked; missing → 0/1/2.
        auto find_special = [&](const char *needle) -> int32_t {
            for (auto &[id, k] : added_content)
                if (k.find(needle) != std::string::npos &&
                    special_ids.count(id) > 0)
                    return id;
            return -1;
        };
        int32_t b = find_special("begin");
        if (b < 0) b = find_special("sentence");
        if (b >= 0) bos_id = b;
        int32_t e = find_special("end");
        if (e >= 0) eos_id = e;
        int32_t p = find_special("pad");
        if (p >= 0) pad_id = p;
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
                std::string_view sv;
                if (m.get_string().get(sv) != simdjson::SUCCESS) continue;
                auto s = std::string(sv);
                auto sp = s.find(' ');
                if (sp == std::string::npos) continue;
                std::string A = s.substr(0, sp);
                std::string B = s.substr(sp + 1);

                auto itL = key_to_id.find(A);
                auto itR = key_to_id.find(B);
                if (itL == key_to_id.end() || itR == key_to_id.end())
                    return arrow::Status::Invalid("tokenizer: merge operand missing in vocab: '", s, "'");

                std::vector<uint8_t> bytesA, bytesB;
                if (!byte_unicode::key_to_bytes(A, bytesA) ||
                    !byte_unicode::key_to_bytes(B, bytesB))
                    return arrow::Status::Invalid("tokenizer: merge operand not byte-level: '", s, "'");

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
            return arrow::Status::Invalid("tokenizer: missing vocab id ", std::to_string(id));
        auto &row = vocab_rows[static_cast<size_t>(id)];
        row.token_id = id;

        std::vector<uint8_t> bytes;
        bool special = special_ids.count(id) > 0;
        if (!byte_unicode::key_to_bytes(it->second, bytes)) {
            // Control tokens (e.g. DeepSeek's "<|fim▁hole|>") are shipped in
            // added_tokens with "special": false but contain chars outside the
            // GPT-2 byte table. They can never be produced by the byte-level
            // encoder, so treat them as special (decode skips them).
            if (!special)
                spdlog::warn("tokenizer: treating non-byte-level token id={} as special: '{}'",
                             id, it->second);
            row.is_special = 1;
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

} // namespace lsxcommon
