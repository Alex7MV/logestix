#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lsxcommon {

class BpeTokenizer {
public:
    BpeTokenizer();
    ~BpeTokenizer();

    BpeTokenizer(const BpeTokenizer &) = delete;
    BpeTokenizer &operator=(const BpeTokenizer &) = delete;
    BpeTokenizer(BpeTokenizer &&) noexcept;
    BpeTokenizer &operator=(BpeTokenizer &&) noexcept;

    bool Load(const std::string &tokenizer_json_path);

    std::vector<int32_t> Encode(const std::string &text) const;
    std::string Decode(const std::vector<int32_t> &ids) const;

    int vocab_size() const { return vocab_size_; }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int vocab_size_ = 0;
    int bos_id_ = -1;
    int eos_id_ = -1;
};

} // namespace lsxcommon
