#include "lsxcommon/converter.h"
#include "lsxcommon/safetensors.h"
#include <arrow/ipc/writer.h>
#include <arrow/io/file.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace lsxcommon {

namespace {

struct ExpertKey {
    int layer;
    int expert;
    bool operator==(const ExpertKey &o) const noexcept {
        return layer == o.layer && expert == o.expert;
    }
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey &k) const noexcept {
        return (static_cast<std::size_t>(k.layer) << 32)
             | static_cast<std::size_t>(k.expert);
    }
};

// Parse "model.layers.<layer>.ffn.experts.<exp>.w<id>.weight"
struct ParsedName {
    int layer = -1;
    int expert = -1;
    int weight_id = -1; // 1, 2, or 3
};

std::optional<ParsedName> parse_expert_name(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.w(\d)\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedName p;
    p.layer     = std::stoi(m[1]);
    p.expert    = std::stoi(m[2]);
    p.weight_id = std::stoi(m[3]);
    return p;
}

} // anonymous namespace

// ── NVFP4 Packing stub ──────────────────────────────────────────────────────

arrow::Result<PackedExpert> PackExpertNVFP4(const ExpertData &raw)
{
    PackedExpert pe;
    pe.w1_shape = raw.w1_shape;
    pe.w2_shape = raw.w2_shape;
    pe.w3_shape = raw.w3_shape;

    auto pack_one = [](const std::vector<uint8_t> &src,
                       int64_t numel,
                       std::vector<uint8_t> &packed,
                       std::vector<float> &scales)
    {
        int64_t npacked = (numel + 1) / 2;
        packed.assign(npacked, 0);

        int64_t nblocks = (numel + kNVFP4BlockSize - 1) / kNVFP4BlockSize;
        scales.assign(nblocks, 1.0f);

        for (int64_t i = 0; i < numel; ++i) {
            float v = 0.0f;
            if (i < static_cast<int64_t>(src.size())) {
                v = static_cast<float>(static_cast<int8_t>(src[i]))
                    / 127.0f;
            }
            uint8_t q = static_cast<uint8_t>(
                std::min(15, std::max(0, static_cast<int>(std::round(v * 7.0f)))));
            if (i % 2 == 0)
                packed[i / 2] = (packed[i / 2] & 0xF0) | q;
            else
                packed[i / 2] = (packed[i / 2] & 0x0F) | (q << 4);
        }
    };

    auto n1 = std::accumulate(raw.w1_shape.begin(), raw.w1_shape.end(),
                               int64_t(1), std::multiplies<>());
    auto n2 = std::accumulate(raw.w2_shape.begin(), raw.w2_shape.end(),
                               int64_t(1), std::multiplies<>());
    auto n3 = std::accumulate(raw.w3_shape.begin(), raw.w3_shape.end(),
                               int64_t(1), std::multiplies<>());

    pack_one(raw.w1_raw, n1, pe.w1_packed, pe.w1_scales);
    pack_one(raw.w2_raw, n2, pe.w2_packed, pe.w2_scales);
    pack_one(raw.w3_raw, n3, pe.w3_packed, pe.w3_scales);

    return pe;
}

// ── Save single expert to Arrow IPC file ────────────────────────────────────

arrow::Status SaveExpertToArrow(
    const PackedExpert &expert,
    int layer_idx,
    int expert_idx,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    auto layer_dir = output_root /
        ("layer_" + std::to_string(layer_idx)) / "experts";
    fs::create_directories(layer_dir);

    auto out_path = layer_dir /
        ("exp_" + std::to_string(expert_idx) + ".arrow");

    ARROW_ASSIGN_OR_RAISE(auto batch, ExpertSchema::MakeBatch(expert));

    ARROW_ASSIGN_OR_RAISE(auto outfile,
        arrow::io::FileOutputStream::Open(out_path.string()));

    ARROW_ASSIGN_OR_RAISE(auto writer,
        arrow::ipc::MakeFileWriter(outfile.get(), batch->schema()));

    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    ARROW_RETURN_NOT_OK(writer->Close());
    ARROW_RETURN_NOT_OK(outfile->Close());

    spdlog::info("  wrote layer={:03d} expert={:03d}  {} -> {}",
        layer_idx, expert_idx,
        batch->column(0)->length(),
        out_path.string());

    return arrow::Status::OK();
}

// ── Converter implementation ────────────────────────────────────────────────

struct Converter::Impl {
    std::string input_path;
    std::string output_path;
    bool verbose = false;
};

Converter::Converter()
    : impl_(std::make_unique<Impl>()) {}

Converter::~Converter() = default;
Converter::Converter(Converter &&) noexcept = default;
Converter &Converter::operator=(Converter &&) noexcept = default;

Converter &Converter::SetInputPath(const std::string &path)
{
    impl_->input_path = path;
    return *this;
}

Converter &Converter::SetOutputPath(const std::string &path)
{
    impl_->output_path = path;
    return *this;
}

Converter &Converter::SetVerbose(bool v)
{
    impl_->verbose = v;
    return *this;
}

arrow::Status Converter::Run()
{
    namespace fs = std::filesystem;

    if (impl_->input_path.empty())
        return arrow::Status::Invalid("input path not set");
    if (impl_->output_path.empty())
        return arrow::Status::Invalid("output path not set");

    fs::create_directories(impl_->output_path);

    // 1. Scan the safetensors model directory
    auto model = lsxcommon::scan_directory(impl_->input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       impl_->input_path);

    auto total_shards = model.shards.size();
    spdlog::info("Found {} shards, {} tensors across {} shards",
        total_shards,
        [&]{
            std::size_t n = 0;
            for (auto &s : model.shards) n += s.tensors.size();
            return n;
        }(),
        total_shards);

    // 2. Scan shards and process experts incrementally
    std::unordered_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    int processed = 0;
    int shard_idx = 0;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        auto t0 = std::chrono::steady_clock::now();

        spdlog::info("[{}/{}] Scanning {} (partial={}, processed={})",
            shard_idx, total_shards, shard.filename,
            partial.size(), processed);

        std::ifstream sf(
            (fs::path(impl_->input_path) / shard.filename).string(),
            std::ios::binary | std::ios::ate);
        if (!sf) { spdlog::warn("  Cannot open, skipping"); continue; }
        int64_t sz = sf.tellg();
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(sz);
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf) { spdlog::warn("  Cannot read, skipping"); continue; }

        spdlog::info("  read {} MB", sz / (1024 * 1024));

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        int experts_in_shard = 0;
        for (const auto &tensor : shard.tensors) {
            auto parsed = parse_expert_name(tensor.name);
            if (!parsed) continue;

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz) {
                spdlog::warn("  Skipping {} (offset {}+{} exceeds file size {})",
                    tensor.name, data_off, tensor.data_begin, sz);
                continue;
            }

            ExpertKey key{parsed->layer, parsed->expert};
            auto &exp = partial[key];
            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            switch (parsed->weight_id) {
            case 1:
                exp.w1_raw   = std::move(raw);
                exp.w1_shape = tensor.shape;
                exp.w1_dtype = tensor.dtype;
                break;
            case 2:
                exp.w2_raw   = std::move(raw);
                exp.w2_shape = tensor.shape;
                exp.w2_dtype = tensor.dtype;
                break;
            case 3:
                exp.w3_raw   = std::move(raw);
                exp.w3_shape = tensor.shape;
                exp.w3_dtype = tensor.dtype;
                break;
            }

            if (!exp.w1_raw.empty() && !exp.w2_raw.empty() && !exp.w3_raw.empty()) {
                ++experts_in_shard;
                auto status = [&]() -> arrow::Status {
                    ARROW_ASSIGN_OR_RAISE(auto packed, PackExpertNVFP4(std::move(exp)));
                    ARROW_RETURN_NOT_OK(SaveExpertToArrow(
                        packed, key.layer, key.expert,
                        fs::path(impl_->output_path)));
                    return arrow::Status::OK();
                }();
                if (!status.ok())
                    spdlog::error("  Failed to pack expert layer={} expert={}: {}",
                        key.layer, key.expert, status.ToString());
                else
                    ++processed;
                partial.erase(key);
            }
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        spdlog::info("  done in {:0.1f}s — {} experts in this shard",
            elapsed / 1000.0, experts_in_shard);
    }

    for (auto &[key, raw] : partial) {
        spdlog::warn("  Incomplete expert layer={} expert={} (w1={} w2={} w3={})",
            key.layer, key.expert,
            raw.w1_raw.size(), raw.w2_raw.size(), raw.w3_raw.size());
    }

    spdlog::info("Done — processed {} experts, incomplete {}",
        processed, partial.size());
    return arrow::Status::OK();
}

} // namespace lsxcommon
