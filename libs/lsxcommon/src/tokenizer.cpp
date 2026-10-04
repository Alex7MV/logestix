#include "lsxcommon/tokenizer.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/pretokenizer.h"
#include <arrow/ipc/reader.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {

bool ArrowTokenizerDb::valid() const
{
    return vocab_mmap && merges_mmap && token_offsets && token_bytes && is_special &&
           vocab_size > 0 && merge_left && merge_right && merge_rank &&
           merge_token_id && merge_count > 0;
}

namespace {

const ArrowTokenizerDb *g_active_db = nullptr;

} // namespace

void set_active_tokenizer_db(const ArrowTokenizerDb *db)
{
    g_active_db = db;
}

arrow::Status load_tokenizer_metadata(
    const std::shared_ptr<arrow::io::MemoryMappedFile> &vocab_mmap,
    const std::shared_ptr<arrow::io::MemoryMappedFile> &merges_mmap,
    ArrowTokenizerDb &db)
{
    auto impl = [&]() -> arrow::Status {
        if (!vocab_mmap || !merges_mmap)
            return arrow::Status::Invalid("tokenizer: null MemoryMappedFile");
        db = ArrowTokenizerDb{};
        db.vocab_mmap = vocab_mmap;
        db.merges_mmap = merges_mmap;

        ARROW_ASSIGN_OR_RAISE(int64_t v_size, vocab_mmap->GetSize());
        db.vocab_file_size = v_size;
        ARROW_ASSIGN_OR_RAISE(auto base_buf, vocab_mmap->ReadAt(0, 1));
        db.vocab_file_base = base_buf->data();

        auto v_reader_res = arrow::ipc::RecordBatchFileReader::Open(vocab_mmap.get());
        if (!v_reader_res.ok()) return v_reader_res.status();
        if ((*v_reader_res)->num_record_batches() == 0)
            return arrow::Status::Invalid("tokenizer: vocab file has no batches");
        auto v_batch_res = (*v_reader_res)->ReadRecordBatch(0);
        if (!v_batch_res.ok()) return v_batch_res.status();
        auto v_batch = *v_batch_res;

        auto tids = std::dynamic_pointer_cast<arrow::Int32Array>(
            v_batch->column(TokenizerVocabSchema::kIdxTokenId));
        auto tbytes = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
            v_batch->column(TokenizerVocabSchema::kIdxTokenBytes));
        auto tspec = std::dynamic_pointer_cast<arrow::Int8Array>(
            v_batch->column(TokenizerVocabSchema::kIdxIsSpecial));
        if (!tids || !tbytes || !tspec)
            return arrow::Status::TypeError("tokenizer: vocab column type mismatch");

        const int64_t n = v_batch->num_rows();
        if (n <= 0 || n > INT32_MAX)
            return arrow::Status::Invalid("tokenizer: bad vocab row count");

        auto in_range = [&](const uint8_t *p, int64_t len) {
            return p >= db.vocab_file_base &&
                   p + len <= db.vocab_file_base + db.vocab_file_size;
        };

        auto offsets_buf = tbytes->data()->buffers[1];
        if (!offsets_buf || !in_range(offsets_buf->data(), offsets_buf->size()))
            return arrow::Status::Invalid("tokenizer: offsets buffer not backed by mmap");
        if (offsets_buf->size() != (n + 1) * static_cast<int64_t>(sizeof(int64_t)))
            return arrow::Status::Invalid("tokenizer: offsets length != vocab_size+1");
        auto values_buf = tbytes->value_data();
        if (!values_buf || !in_range(values_buf->data(), values_buf->size()))
            return arrow::Status::Invalid("tokenizer: values buffer not backed by mmap");
        auto spec_buf = tspec->data()->buffers[1];
        if (!spec_buf || !in_range(spec_buf->data(), spec_buf->size()))
            return arrow::Status::Invalid("tokenizer: is_special buffer not backed by mmap");

        db.vocab_size = static_cast<int32_t>(n);
        db.token_offsets = tbytes->raw_value_offsets();
        db.token_bytes = values_buf->data();
        db.is_special = reinterpret_cast<const uint8_t *>(tspec->raw_values());

        auto meta = v_batch->schema()->metadata();
        if (!meta) return arrow::Status::Invalid("tokenizer: vocab schema has no metadata");
        auto get_i32 = [&](const char *key, int32_t &out) -> arrow::Status {
            ARROW_ASSIGN_OR_RAISE(auto s, meta->Get(key));
            int32_t v = 0;
            if (!absl::SimpleAtoi(s, &v))
                return arrow::Status::Invalid("tokenizer: bad metadata ", key);
            out = v;
            return arrow::Status::OK();
        };
        ARROW_RETURN_NOT_OK(get_i32("bos_id", db.bos_id));
        ARROW_RETURN_NOT_OK(get_i32("eos_id", db.eos_id));
        ARROW_RETURN_NOT_OK(get_i32("pad_id", db.pad_id));
        ARROW_RETURN_NOT_OK(get_i32("vocab_size", db.vocab_size));
        if (db.vocab_size != static_cast<int32_t>(n))
            return arrow::Status::Invalid("tokenizer: vocab_size metadata != row count");

        // byte_to_token_id (256 comma-separated ints)
        {
            ARROW_ASSIGN_OR_RAISE(auto btt, meta->Get("byte_to_token_id"));
            size_t i = 0;
            for (absl::string_view part : absl::StrSplit(btt, ',')) {
                if (i >= 256) break;
                int32_t v = 0;
                if (!absl::SimpleAtoi(part, &v))
                    return arrow::Status::Invalid("tokenizer: bad byte_to_token_id metadata");
                db.byte_to_token_id[i] = v;
                ++i;
            }
            if (i != 256)
                return arrow::Status::Invalid("tokenizer: bad byte_to_token_id metadata");
        }

        // Optional encoding-family metadata (absent in pre-SP-BPE artifacts →
        // defaults keep the GPT-2 byte-level behaviour).
        {
            auto get_opt_i32 = [&](const char *key, int32_t &out) -> void {
                auto s = meta->Get(key);
                if (!s.ok()) return;
                int32_t v = 0;
                if (absl::SimpleAtoi(*s, &v)) out = v;
            };
            get_opt_i32("tokenizer_kind", db.tokenizer_kind);
            get_opt_i32("unk_id", db.unk_id);
            get_opt_i32("byte_fallback", db.byte_fallback);
            get_opt_i32("add_dummy_prefix", db.add_dummy_prefix);
            get_opt_i32("escape_whitespaces", db.escape_whitespaces);
        }

        // ---- merges file ----
        ARROW_ASSIGN_OR_RAISE(int64_t m_size, merges_mmap->GetSize());
        ARROW_ASSIGN_OR_RAISE(auto m_base_buf, merges_mmap->ReadAt(0, 1));
        const uint8_t *m_file_base = m_base_buf->data();
        const int64_t m_file_size = m_size;

        auto m_reader_res = arrow::ipc::RecordBatchFileReader::Open(merges_mmap.get());
        if (!m_reader_res.ok()) return m_reader_res.status();
        if ((*m_reader_res)->num_record_batches() == 0)
            return arrow::Status::Invalid("tokenizer: merges file has no batches");
        auto m_batch_res = (*m_reader_res)->ReadRecordBatch(0);
        if (!m_batch_res.ok()) return m_batch_res.status();
        auto m_batch = *m_batch_res;

        auto mleft = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeLeft));
        auto mright = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeRight));
        auto mrank = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeRank));
        auto mtok = std::dynamic_pointer_cast<arrow::Int32Array>(
            m_batch->column(TokenizerMergesSchema::kIdxMergeTokenId));
        if (!mleft || !mright || !mrank || !mtok)
            return arrow::Status::TypeError("tokenizer: merges column type mismatch");

        const int64_t mn = m_batch->num_rows();
        if (mn <= 0 || mn > INT32_MAX)
            return arrow::Status::Invalid("tokenizer: bad merges row count");

        auto m_in_range = [&](const uint8_t *p, int64_t len) {
            return p >= m_file_base && p + len <= m_file_base + m_file_size;
        };
        for (int c = 0; c < 4; ++c) {
            auto buf = m_batch->column(c)->data()->buffers[1];
            if (!buf || !m_in_range(buf->data(), buf->size()))
                return arrow::Status::Invalid("tokenizer: merges column buffer not backed by mmap");
        }

        auto m_meta = m_batch->schema()->metadata();
        if (!m_meta) return arrow::Status::Invalid("tokenizer: merges schema has no metadata");
        {
            ARROW_ASSIGN_OR_RAISE(auto s, m_meta->Get("merge_count"));
            int32_t mc = 0;
            if (!absl::SimpleAtoi(s, &mc) || mc != mn)
                return arrow::Status::Invalid("tokenizer: merge_count metadata != row count");
        }

        db.merge_count = static_cast<int32_t>(mn);
        db.merge_left = mleft->raw_values();
        db.merge_right = mright->raw_values();
        db.merge_rank = mrank->raw_values();
        db.merge_token_id = mtok->raw_values();

        return arrow::Status::OK();
    };
    auto st = impl();
    if (!st.ok()) db.error_msg = st.ToString();
    return st;
}

namespace {

// Greedy lowest-rank adjacent-pair merge over an arbitrary symbol-id sequence.
// Shared by the GPT-2 byte-level BPE seed (bytes -> single-byte ids) and the
// SentencePiece-BPE seed (codepoints -> piece ids / byte-fallback ids).
// Returns the number of surviving symbols.
size_t merge_symbols_in_place(const ArrowTokenizerDb &db, int32_t *toks,
                              size_t n)
{
    while (n > 1) {
        int best_i = -1;
        int32_t best_rank = INT32_MAX;
        int32_t best_tok = -1;
        for (size_t i = 0; i + 1 < n; ++i) {
            int32_t left = toks[i], right = toks[i + 1];
            // binary search over merges sorted by (merge_left, merge_right)
            int lo = 0, hi = db.merge_count;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (db.merge_left[mid] < left ||
                    (db.merge_left[mid] == left && db.merge_right[mid] < right))
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo < db.merge_count && db.merge_left[lo] == left &&
                db.merge_right[lo] == right && db.merge_rank[lo] < best_rank) {
                best_i = static_cast<int>(i);
                best_rank = db.merge_rank[lo];
                best_tok = db.merge_token_id[lo];
            }
        }
        if (best_i < 0) break;
        toks[best_i] = best_tok;
        for (size_t i = static_cast<size_t>(best_i) + 1; i + 1 < n; ++i)
            toks[i] = toks[i + 1];
        --n;
    }
    return n;
}

// Returns the number of merged tokens for one word, or:
//   -1  a byte has no token mapping (db malformed)
//   -3  word longer than kMaxWordBytes
// Fills toks[0..n) with the merged token ids.
int32_t bpe_encode_word(const ArrowTokenizerDb &db, const char *w, size_t wlen,
                        int32_t *toks)
{
    if (wlen > kMaxWordBytes) return -3;

    for (size_t i = 0; i < wlen; ++i) {
        int32_t t = db.byte_to_token_id[static_cast<uint8_t>(w[i])];
        if (t < 0) return -1;
        toks[i] = t;
    }
    return static_cast<int32_t>(merge_symbols_in_place(db, toks, wlen));
}

// ── SentencePiece-BPE (tokenizer_kind == 1) ─────────────────────────────────
//
// Identity normalizer + U+2581 '▁' word marker. Encoding: prepend '▁' when
// add_dummy_prefix (non-empty input only), replace each ASCII space with '▁'
// when escape_whitespaces, seed the symbol sequence with the single-codepoint
// piece id (byte-fallback ids for uncovered codepoints), then merge greedily by
// derived rank. Decode: skip specials, map <0xXX> pieces to raw bytes and '▁'
// to ' '.

constexpr char kSpSpace[] = "\xE2\x96\x81";  // U+2581 ▁ (3 bytes)

size_t Utf8SeqLen(uint8_t c)
{
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

// True when piece bytes are exactly the 6-byte "<0xHH>" byte-fallback form;
// writes the decoded byte to *out.
bool SpBytePiece(const uint8_t *p, int64_t len, uint8_t *out)
{
    if (len != 6) return false;
    if (p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>') return false;
    auto hex = [](uint8_t c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    int hi = hex(p[3]), lo = hex(p[4]);
    if (hi < 0 || lo < 0) return false;
    *out = static_cast<uint8_t>((hi << 4) | lo);
    return true;
}

int32_t sp_bpe_encode(const ArrowTokenizerDb &db, const char *text,
                      size_t text_len, int32_t *out_tokens, size_t *max_out_len)
{
    std::vector<int32_t> ids;
    if (text_len > 0) {
        // 1. normalized byte string.
        std::string norm;
        norm.reserve(text_len + 3);
        if (db.add_dummy_prefix) norm.append(kSpSpace, 3);
        for (size_t i = 0; i < text_len; ++i) {
            if (text[i] == ' ' && db.escape_whitespaces)
                norm.append(kSpSpace, 3);
            else
                norm += text[i];
        }

        // 2. single-token byte lookup (skip control/unknown specials).
        absl::flat_hash_map<std::string_view, int32_t> by_bytes;
        by_bytes.reserve(static_cast<size_t>(db.vocab_size) * 2);
        for (int32_t id = 0; id < db.vocab_size; ++id) {
            if (db.is_special[id]) continue;
            const int64_t off = db.token_offsets[id];
            const int64_t len = db.token_offsets[id + 1] - off;
            by_bytes.emplace(
                std::string_view(reinterpret_cast<const char *>(db.token_bytes + off),
                                 static_cast<size_t>(len)),
                id);
        }

        // 3. seed symbols codepoint-by-codepoint.
        ids.reserve(norm.size());
        size_t i = 0;
        while (i < norm.size()) {
            size_t clen = Utf8SeqLen(static_cast<uint8_t>(norm[i]));
            if (i + clen > norm.size()) clen = 1;
            std::string_view cp(norm.data() + i, clen);
            auto it = by_bytes.find(cp);
            if (it != by_bytes.end()) {
                ids.push_back(it->second);
            } else if (db.byte_fallback) {
                for (size_t b = 0; b < clen; ++b)
                    ids.push_back(
                        db.byte_to_token_id[static_cast<uint8_t>(norm[i + b])]);
            } else {
                ids.push_back(db.unk_id);
            }
            i += clen;
        }

        // 4. merge.
        const size_t n = merge_symbols_in_place(db, ids.data(), ids.size());
        ids.resize(n);
    }

    // 5. prepend BOS.
    const size_t bos = (db.bos_id >= 0) ? 1 : 0;
    const size_t total = bos + ids.size();
    if (total > *max_out_len) {
        *max_out_len = total;
        return -2;
    }
    size_t o = 0;
    if (bos) out_tokens[o++] = db.bos_id;
    for (int32_t id : ids) out_tokens[o++] = id;
    *max_out_len = o;
    return static_cast<int32_t>(o);
}

// ── UTF-8 output guarantee, shared by both decode paths ──────────────────────
//
// Decode concatenates the raw vocabulary bytes of every non-special token.
// That stream is NOT guaranteed to be valid UTF-8: a byte-level vocabulary maps
// each raw byte to a token whose value IS that byte (gemma-4-26B-A4B-it ships
// 125 such single-byte entries, and byte_to_token_id[0xE9] points at the
// one-byte 0xE9 token), so a model can emit a lone lead or continuation byte.
// The encoder rejects such text (arrow_tokenizer_encode -> -1), which makes
// BuildConversationInputIds fail and silently drops every follow-up turn of a
// chat. Decoding therefore substitutes U+FFFD for any byte that cannot begin or
// complete a well-formed sequence, so the decode -> frame -> encode round-trip
// always holds.
//
// Byte-fallback pieces legitimately assemble multi-byte codepoints across token
// boundaries, so this is a streaming sequence-aware filter, not a per-byte one.

// U+FFFD REPLACEMENT CHARACTER.
constexpr char kReplacement[] = "\xEF\xBF\xBD";
constexpr size_t kReplacementLen = 3;

// Incremental UTF-8 filter: Feed() one byte at a time, then Finish(). Bytes are
// buffered only while a multi-byte sequence is still incomplete, so a codepoint
// split across tokens survives intact.
class Utf8Sanitizer {
public:
    // Appends this byte's contribution to `out`; returns the byte count added.
    size_t Feed(uint8_t b, char *out) {
        if (n_pending_ == 0) {
            // Not mid-sequence: ASCII goes straight through, a lead byte starts
            // a new one, anything else is undecodable on its own.
            if (b < 0x80u) { out[0] = static_cast<char>(b); return 1; }
            need_ = utf8_expected_len(b);
            if (need_ == 0) return EmitReplacement(out);
            pending_[n_pending_++] = b;
            return 0;
        }
        // A byte that is not a continuation byte does not belong to this
        // sequence: flush the maximal invalid subpart as one replacement and
        // reprocess the offending byte, which may itself start a valid
        // sequence (the Unicode "maximal subpart" rule).
        if ((b & 0xC0u) != 0x80u) {
            n_pending_ = 0;
            const size_t n = EmitReplacement(out);
            return n + Feed(b, out + n);
        }
        pending_[n_pending_++] = b;
        if (n_pending_ < need_) return 0;      // still incomplete
        size_t len = 0;
        const uint32_t cp = decode_utf8(reinterpret_cast<const char *>(pending_), n_pending_, &len);
        n_pending_ = 0;
        if (cp == kInvalidUtf8) return EmitReplacement(out);
        std::memcpy(out, pending_, len);
        return len;
    }

    // Flushes a trailing incomplete sequence as one replacement character.
    size_t Finish(char *out) {
        if (n_pending_ == 0) return 0;
        n_pending_ = 0;
        return EmitReplacement(out);
    }

private:
    static size_t EmitReplacement(char *out) {
        std::memcpy(out, kReplacement, kReplacementLen);
        return kReplacementLen;
    }

    uint8_t pending_[4] = {};
    size_t n_pending_ = 0;
    size_t need_ = 0;
};

// Feeds one SentencePiece token's decoded bytes to `sink`: a <0xXX> piece
// expands to the raw byte it stands for, the U+2581 word marker becomes a
// space, everything else is verbatim. Shared by both decode passes so the
// capacity count and the emitted bytes can never disagree.
template <typename Sink>
void FeedSpToken(const uint8_t *p, int64_t len, Sink &&sink)
{
    uint8_t byte;
    if (SpBytePiece(p, len, &byte)) { sink(byte); return; }
    for (int64_t k = 0; k < len;) {
        if (k + 2 < len && p[k] == 0xE2 && p[k + 1] == 0x96 && p[k + 2] == 0x81) {
            sink(static_cast<uint8_t>(' '));
            k += 3;
        } else {
            sink(p[k]);
            ++k;
        }
    }
}

int32_t sp_bpe_decode(const ArrowTokenizerDb &db, const int32_t *ids,
                      size_t n_ids, char *out_text, size_t *max_out_len)
{
    // Pass 1: sanitized length (a bad byte expands 1 -> 3, so the raw byte
    // count cannot be reused).
    size_t total = 0;
    {
        Utf8Sanitizer san;
        char scratch[kReplacementLen];
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t id = ids[i];
            if (id < 0 || id >= db.vocab_size) return -1;
            if (db.is_special[id]) continue;
            FeedSpToken(db.token_bytes + db.token_offsets[id],
                        db.token_offsets[id + 1] - db.token_offsets[id],
                        [&](uint8_t b) { total += san.Feed(b, scratch); });
        }
        total += san.Finish(scratch);
    }
    if (total > *max_out_len) {
        *max_out_len = total;
        return -2;
    }

    // Pass 2: emit through the same filter, so a <0xXX> piece that expands to a
    // byte no UTF-8 sequence can start becomes U+FFFD instead of raw bytes the
    // encoder would then reject.
    size_t o = 0;
    {
        Utf8Sanitizer san;
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t id = ids[i];
            if (db.is_special[id]) continue;
            FeedSpToken(db.token_bytes + db.token_offsets[id],
                        db.token_offsets[id + 1] - db.token_offsets[id],
                        [&](uint8_t b) { o += san.Feed(b, out_text + o); });
        }
        o += san.Finish(out_text + o);
    }
    // SentencePiece drops the leading whitespace produced by the dummy-prefix
    // '▁' (e.g. decode('▁The') == "The", decode('▁▁') == " ").
    if (o > 0 && out_text[0] == ' ') {
        std::memmove(out_text, out_text + 1, o - 1);
        --o;
    }
    *max_out_len = o;
    return 0;
}

} // anonymous namespace

// ── gpt2 byte-level BPE (tokenizer_kind == 0) ────────────────────────────────

int32_t arrow_tokenizer_encode(const ArrowTokenizerDb &db, const char *text,
                               size_t text_len, int32_t *out_tokens,
                               size_t *max_out_len)
{
    if (!db.valid()) return -1;
    if (text_len == 0) text = "";
    if (db.tokenizer_kind == 1)
        return sp_bpe_encode(db, text, text_len, out_tokens, max_out_len);

    // Pass 1: count.
    size_t total = (db.bos_id >= 0) ? 1 : 0;
    {
        WordStream ws(text, text_len);
        WordSpan w;
        int32_t stack[kMaxWordBytes];
        while (ws.next(w)) {
            int32_t n = bpe_encode_word(db, w.begin, w.length, stack);
            if (n < 0) return n;                    // -1 or -3
            total += static_cast<size_t>(n);
        }
        if (ws.error()) return -1;                  // malformed UTF-8
    }
    if (total > *max_out_len) { *max_out_len = total; return -2; }

    // Pass 2: emit bos + each word.
    size_t o = 0;
    if (db.bos_id >= 0) out_tokens[o++] = db.bos_id;
    {
        WordStream ws(text, text_len);
        WordSpan w;
        int32_t stack[kMaxWordBytes];
        while (ws.next(w)) {
            int32_t n = bpe_encode_word(db, w.begin, w.length, stack);
            for (int32_t i = 0; i < n; ++i) out_tokens[o++] = stack[i];
        }
    }
    *max_out_len = o;
    return static_cast<int32_t>(o);
}

int32_t arrow_tokenizer_encode(const char *text, size_t text_len,
                               int32_t *out_tokens, size_t *max_out_len)
{
    if (!g_active_db) return -1;
    return arrow_tokenizer_encode(*g_active_db, text, text_len, out_tokens, max_out_len);
}

// Valid gpt2 byte-level BPE decode: specials skipped, raw vocabulary bytes
// appended, every undecodable byte replaced by U+FFFD (see Utf8Sanitizer).
// Returns 0 on success, -1 invalid db or id, -2 output capacity too small.
// *max_out_len is in/out.
int32_t arrow_tokenizer_decode(const ArrowTokenizerDb &db, const int32_t *ids,
                               size_t n_ids, char *out_text, size_t *max_out_len)
{
    if (!db.valid()) return -1;
    if (db.tokenizer_kind == 1)
        return sp_bpe_decode(db, ids, n_ids, out_text, max_out_len);

    // Pass 1: sanitized length (a bad byte expands 1 -> 3, so this cannot reuse
    // the raw byte count).
    size_t total = 0;
    {
        Utf8Sanitizer san;
        for (size_t i = 0; i < n_ids; ++i) {
            int32_t id = ids[i];
            if (id < 0 || id >= db.vocab_size) return -1;
            if (db.is_special[id]) continue;
            const uint8_t *p = db.token_bytes + db.token_offsets[id];
            const size_t len = static_cast<size_t>(db.token_offsets[id + 1] -
                                                   db.token_offsets[id]);
            for (size_t k = 0; k < len; ++k) {
                // Feed into a scratch sized for the widest single contribution.
                char scratch[kReplacementLen];
                total += san.Feed(p[k], scratch);
            }
        }
        char scratch[kReplacementLen];
        total += san.Finish(scratch);
    }
    if (total > *max_out_len) { *max_out_len = total; return -2; }

    // Pass 2: emit sanitized bytes.
    size_t o = 0;
    {
        Utf8Sanitizer san;
        for (size_t i = 0; i < n_ids; ++i) {
            int32_t id = ids[i];
            if (db.is_special[id]) continue;
            const uint8_t *p = db.token_bytes + db.token_offsets[id];
            const size_t len = static_cast<size_t>(db.token_offsets[id + 1] -
                                                   db.token_offsets[id]);
            for (size_t k = 0; k < len; ++k)
                o += san.Feed(p[k], out_text + o);
        }
        o += san.Finish(out_text + o);
    }
    *max_out_len = o;
    return 0;
}

int32_t arrow_tokenizer_lookup(const ArrowTokenizerDb &db, const char *content,
                               size_t len)
{
    if (!db.valid()) return -1;
    for (int32_t id = 0; id < db.vocab_size; ++id) {
        const int64_t off = db.token_offsets[id];
        const int64_t tlen = db.token_offsets[id + 1] - off;
        if (tlen == static_cast<int64_t>(len) &&
            std::memcmp(db.token_bytes + off, content, len) == 0)
            return id;
    }
    return -1;
}

} // namespace lsxcommon
