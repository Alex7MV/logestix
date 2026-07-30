#include "lsxcommon/tokenizer.h"
#include <simdjson.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <utility>

namespace lsxcommon {

struct BpeTokenizer::Impl {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> id_to_token;
    std::vector<std::pair<std::string, std::string>> merges;
};

BpeTokenizer::BpeTokenizer() : impl_(std::make_unique<Impl>()) {}
BpeTokenizer::~BpeTokenizer() = default;
BpeTokenizer::BpeTokenizer(BpeTokenizer &&) noexcept = default;
BpeTokenizer &BpeTokenizer::operator=(BpeTokenizer &&) noexcept = default;

bool BpeTokenizer::Load(const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto sz = f.tellg();
    if (sz <= 0) return false;
    f.seekg(0);
    std::string json_str(static_cast<size_t>(sz), '\0');
    f.read(json_str.data(), sz);
    if (!f) return false;

    simdjson::padded_string pad(json_str.data(), json_str.size());
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return false;

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return false;

    // Parse model.vocab and model.merges
    simdjson::ondemand::object model_obj;
    if (root["model"].get_object().get(model_obj) != simdjson::SUCCESS) return false;

    simdjson::ondemand::object vocab_obj;
    if (model_obj["vocab"].get_object().get(vocab_obj) != simdjson::SUCCESS) return false;

    for (auto entry : vocab_obj) {
        std::string_view key;
        int64_t id;
        if (entry.unescaped_key().get(key) != simdjson::SUCCESS) continue;
        if (entry.value().get_int64().get(id) != simdjson::SUCCESS) continue;
        auto k = std::string(key);
        auto i = static_cast<int32_t>(id);
        impl_->vocab[k] = i;
        impl_->id_to_token[i] = std::move(k);
    }
    vocab_size_ = static_cast<int>(impl_->vocab.size());

    simdjson::ondemand::array merges_arr;
    if (model_obj["merges"].get_array().get(merges_arr) == simdjson::SUCCESS) {
        for (auto m : merges_arr) {
            std::string_view sv;
            if (m.get_string().get(sv) == simdjson::SUCCESS) {
                auto s = std::string(sv);
                auto space = s.find(' ');
                if (space != std::string::npos) {
                    impl_->merges.emplace_back(s.substr(0, space), s.substr(space + 1));
                }
            }
        }
    }

    // Parse added_tokens for BOS/EOS
    simdjson::ondemand::array added_arr;
    if (root["added_tokens"].get_array().get(added_arr) == simdjson::SUCCESS) {
        for (auto at : added_arr) {
            simdjson::ondemand::object at_obj;
            if (at.get_object().get(at_obj) != simdjson::SUCCESS) continue;
            std::string_view content;
            int64_t id = 0;
            at_obj["content"].get_string().get(content);
            at_obj["id"].get_int64().get(id);
            auto c = std::string(content);
            auto i = static_cast<int32_t>(id);
            if (c.find("begin") != std::string::npos || c.find("sentence") != std::string::npos)
                bos_id_ = i;
            if (c.find("end") != std::string::npos)
                eos_id_ = i;
            impl_->vocab[c] = i;
            impl_->id_to_token[i] = std::move(c);
        }
    }

    return vocab_size_ > 0;
}

namespace {

std::vector<std::string> pre_tokenize(const std::string &text)
{
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isspace(c)) {
            if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
            continue;
        }
        if (std::ispunct(c)) {
            if (!current.empty()) {
                words.push_back(std::move(current));
                current.clear();
            }
            words.emplace_back(1, static_cast<char>(c));
            continue;
        }
        current += static_cast<char>(c);
    }
    if (!current.empty()) words.push_back(std::move(current));
    return words;
}

std::vector<std::string> bpe_encode_word(
    const std::string &word,
    const std::unordered_map<std::string, int32_t> &vocab,
    const std::vector<std::pair<std::string, std::string>> &merges)
{
    if (word.empty()) return {};

    // Check if word is already in vocab
    if (vocab.find(word) != vocab.end()) return {word};

    std::vector<std::string> symbols;
    for (size_t i = 0; i < word.size(); ++i)
        symbols.emplace_back(1, word[i]);

    // Build merge rank map for O(1) lookup
    std::unordered_map<std::string, int> merge_rank;
    for (size_t i = 0; i < merges.size(); ++i) {
        merge_rank[merges[i].first + " " + merges[i].second] = static_cast<int>(i);
    }

    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = 0;
        bool found = false;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto pair = symbols[i] + symbols[i + 1];
            auto it = merge_rank.find(pair);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
                found = true;
            }
        }
        if (!found) break;

        symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
        symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_idx) + 1);
    }

    return symbols;
}

} // anonymous namespace

std::vector<int32_t> BpeTokenizer::Encode(const std::string &text) const
{
    std::vector<int32_t> ids;
    if (bos_id_ >= 0) ids.push_back(bos_id_);

    auto words = pre_tokenize(text);
    for (const auto &word : words) {
        auto it = impl_->vocab.find(word);
        if (it != impl_->vocab.end()) {
            ids.push_back(it->second);
        } else {
            auto subwords = bpe_encode_word(word, impl_->vocab, impl_->merges);
            for (const auto &sw : subwords) {
                auto it2 = impl_->vocab.find(sw);
                if (it2 != impl_->vocab.end())
                    ids.push_back(it2->second);
            }
        }
    }

    return ids;
}

std::string BpeTokenizer::Decode(const std::vector<int32_t> &ids) const
{
    std::string result;
    for (auto id : ids) {
        auto it = impl_->id_to_token.find(id);
        if (it != impl_->id_to_token.end()) {
            auto &s = it->second;
            if (s.empty()) continue;
            if (s[0] == '<') continue; // skip special tokens
            if (s.size() == 6 && s[0] == '[' && s[5] == ']') continue;
            result += s;
        }
    }
    return result;
}

} // namespace lsxcommon
