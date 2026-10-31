// Generic tokenizer parity probe: compares the engine's encoder
// (arrow_tokenizer_encode) against reference ids captured from the checkpoint's
// own tokenizer, for ANY converted model tree.
//
//     tokenizer_parity <model-root> <cases.jsonl> [--verbose]
//
// <model-root> is the directory that contains logestix/ (the sweep runs it over
// the whole vault). <cases.jsonl> is the fixture format already used by the
// per-family parity tests: one object per line
//
//     {"text": "...", "ids": [1, 2, 3]}
//
// Exit status is 0 only when every case matches (a leading BOS is reported as
// its own class, because some families' references add it in the post-processor
// and others do not).
#include "lsxcommon/tokenizer.h"

#include <arrow/io/file.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <simdjson.h>

namespace {

struct Case {
    std::string text;
    std::vector<int32_t> ids;
};

std::string Join(const std::vector<int32_t> &v, size_t limit = 64)
{
    std::string s;
    for (size_t i = 0; i < v.size() && i < limit; ++i) {
        if (i) s += ' ';
        s += std::to_string(v[i]);
    }
    if (v.size() > limit) s += " ...";
    return s;
}

std::string Escape(std::string_view text, size_t limit = 100)
{
    static const char *kHex = "0123456789ABCDEF";
    std::string out;
    size_t i = 0;
    for (const char ch : text) {
        if (i++ >= limit) { out += "..."; break; }
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c >= 0x20 && c < 0x7F) {
            out += static_cast<char>(c);
        } else {
            out += "\\x";
            out += kHex[c >> 4];
            out += kHex[c & 0x0F];
        }
    }
    return out;
}

std::string DecodeIds(const lsxcommon::ArrowTokenizerDb &db,
                      const std::vector<int32_t> &ids, size_t limit)
{
    std::string out;
    for (size_t i = 0; i < ids.size() && i < limit; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= db.vocab_size) {
            out += "<out-of-range:" + std::to_string(id) + ">";
            continue;
        }
        if (db.is_special[id]) {
            out += "<special:" + std::to_string(id) + ">";
            continue;
        }
        const int64_t off = db.token_offsets[id];
        const int64_t len = db.token_offsets[id + 1] - off;
        std::string bytes(reinterpret_cast<const char *>(db.token_bytes + off),
                          static_cast<size_t>(len));
        out += '[' + Escape(bytes) + ']';
    }
    if (ids.size() > limit) out += " ...";
    return out;
}

std::vector<Case> LoadCases(const std::string &path, std::string &error)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<Case> cases;
    if (!f.good()) {
        error = "cannot open " + path;
        return cases;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        if (parser.parse(line).get(doc) != simdjson::SUCCESS) {
            error = "bad JSONL line: " + line.substr(0, 200);
            return {};
        }
        Case c;
        std::string_view text;
        if (doc["text"].get_string().get(text) != simdjson::SUCCESS) {
            error = "case without text";
            return {};
        }
        c.text.assign(text);
        simdjson::dom::array ids;
        if (doc["ids"].get_array().get(ids) == simdjson::SUCCESS) {
            for (auto e : ids) {
                int64_t v = 0;
                (void)e.get_int64().get(v);
                c.ids.push_back(static_cast<int32_t>(v));
            }
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

bool LoadDb(const std::string &root, lsxcommon::ArrowTokenizerDb &db,
            std::string &error)
{
    const std::string v = root + "/logestix/pinned/tokenizer_vocab.arrow";
    const std::string m = root + "/logestix/pinned/tokenizer_merges.arrow";
    auto vf = arrow::io::MemoryMappedFile::Open(v, arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(m, arrow::io::FileMode::READ);
    if (!vf.ok() || !mf.ok()) {
        error = "missing tokenizer arrows under " + root;
        return false;
    }
    const auto st = lsxcommon::load_tokenizer_metadata(*vf, *mf, db);
    if (!st.ok() || !db.valid()) {
        error = "load_tokenizer_metadata: " + st.ToString();
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::printf("usage: tokenizer_parity <model-root> <cases.jsonl> "
                    "[--verbose]\n");
        return 2;
    }
    const std::string root = argv[1];
    const std::string cases_path = argv[2];
    const bool verbose = argc > 3 && std::strcmp(argv[3], "--verbose") == 0;

    lsxcommon::ArrowTokenizerDb db;
    std::string error;
    if (!LoadDb(root, db, error)) {
        std::printf("load failed: %s\n", error.c_str());
        return 2;
    }
    const std::vector<Case> cases = LoadCases(cases_path, error);
    if (cases.empty()) {
        std::printf("cases failed: %s\n", error.c_str());
        return 2;
    }

    std::printf("root=%s vocab=%d kind=%d base=%d merges=%d "
                "bos=%d eos=%d pad=%d drop=%d nfc=%d\n",
                root.c_str(), db.vocab_size, db.tokenizer_kind,
                db.base_vocab_size, db.merge_count, db.bos_id, db.eos_id,
                db.pad_id, db.drop_unmapped_bytes, db.normalize_nfc);

    size_t exact = 0, bos_only = 0, mismatch = 0, encode_error = 0;
    for (const auto &c : cases) {
        std::vector<int32_t> out(c.text.size() * 4 + 64, 0);
        size_t cap = out.size();
        const int32_t rc = lsxcommon::arrow_tokenizer_encode(
            db, c.text.data(), c.text.size(), out.data(), &cap);
        if (rc < 0) {
            ++encode_error;
            std::printf("[ENCODE-ERR] rc=%d text=[%s]\n", rc,
                        Escape(c.text).c_str());
            continue;
        }
        out.resize(static_cast<size_t>(rc));
        if (out == c.ids) {
            ++exact;
            if (verbose) std::printf("[ok] text=[%s]\n", Escape(c.text).c_str());
            continue;
        }
        // Some references encode add_special_tokens=False while the engine
        // prepends the checkpoint's BOS (kind-0 legacy path). Accept that one
        // case explicitly and report it as its own class instead of hiding it.
        if (!out.empty() && db.bos_id >= 0 && out.front() == db.bos_id &&
            std::vector<int32_t>(out.begin() + 1, out.end()) == c.ids) {
            ++bos_only;
            if (verbose)
                std::printf("[bos] text=[%s] (reference has no BOS)\n",
                            Escape(c.text).c_str());
            continue;
        }
        ++mismatch;
        size_t diff = 0;
        while (diff < out.size() && diff < c.ids.size() &&
               out[diff] == c.ids[diff])
            ++diff;
        std::printf("[FAIL] text=[%s]\n", Escape(c.text).c_str());
        std::printf("  want: %s\n", Join(c.ids).c_str());
        std::printf("  got : %s\n", Join(out).c_str());
        std::printf("  first diff at %zu (want %s, got %s)\n", diff,
                    diff < c.ids.size() ? std::to_string(c.ids[diff]).c_str()
                                        : "<end>",
                    diff < out.size() ? std::to_string(out[diff]).c_str()
                                      : "<end>");
        std::printf("  want pieces: %s\n",
                    DecodeIds(db, c.ids, 24).c_str());
        std::printf("  got  pieces: %s\n", DecodeIds(db, out, 24).c_str());
    }

    std::printf("summary: cases=%zu exact=%zu bos_only=%zu mismatch=%zu "
                "encode_error=%zu\n",
                cases.size(), exact, bos_only, mismatch, encode_error);
    return (mismatch == 0 && encode_error == 0) ? 0 : 1;
}
