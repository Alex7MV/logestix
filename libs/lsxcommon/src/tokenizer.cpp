#include "lsxcommon/tokenizer.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/pretokenizer.h"
#include <arrow/ipc/reader.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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
    size_t n = wlen;

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
    return static_cast<int32_t>(n);
}

} // anonymous namespace

int32_t arrow_tokenizer_encode(const ArrowTokenizerDb &db, const char *text,
                               size_t text_len, int32_t *out_tokens,
                               size_t *max_out_len)
{
    if (!db.valid()) return -1;
    if (text_len == 0) text = "";

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

// Stub — real implementation lands in Task 7.
int32_t arrow_tokenizer_decode(const ArrowTokenizerDb &db, const int32_t *ids,
                               size_t n_ids, char *out_text, size_t *max_out_len)
{
    if (!db.valid()) return -1;

    // Pass 1: count raw bytes, skipping specials.
    size_t total = 0;
    for (size_t i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        if (id < 0 || id >= db.vocab_size) return -1;
        if (db.is_special[id]) continue;
        total += static_cast<size_t>(db.token_offsets[id + 1] - db.token_offsets[id]);
    }
    if (total > *max_out_len) { *max_out_len = total; return -2; }

    // Pass 2: copy raw bytes.
    size_t o = 0;
    for (size_t i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        if (db.is_special[id]) continue;
        int64_t off = db.token_offsets[id];
        int64_t len = db.token_offsets[id + 1] - off;
        std::memcpy(out_text + o, db.token_bytes + off, static_cast<size_t>(len));
        o += static_cast<size_t>(len);
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
