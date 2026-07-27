#include "lsxcommon/converter.h"
#include "lsxcommon/safetensors.h"
#include <arrow/ipc/writer.h>
#include <arrow/io/file.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iomanip>
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

struct ParsedName {
    int layer = -1;
    int expert = -1;
    int weight_id = -1;
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

// ── NVFP4 Packing ───────────────────────────────────────────────────────────

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
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, out_path));

    spdlog::info("  wrote layer={:03d} expert={:03d}  {} -> {}",
        layer_idx, expert_idx,
        batch->column(0)->length(),
        out_path.string());

    return arrow::Status::OK();
}

// ── Save all pinned tensors ────────────────────────────────────────────────

arrow::Status SavePinnedTensors(
    const std::vector<TensorRecord> &tensors,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    fs::create_directories(output_root / "pinned");

    std::unordered_map<std::string, std::vector<TensorRecord>> by_layer;
    std::vector<TensorRecord> global;

    for (auto &t : tensors) {
        static const std::regex layer_re(R"(layers\.(\d+)\..+)");
        std::smatch m;
        if (std::regex_match(t.name, m, layer_re)) {
            by_layer[std::string("layer_") + m[1].str()].push_back(t);
        } else {
            global.push_back(t);
        }
    }

    if (!global.empty()) {
        auto path = output_root / "pinned" / "global.arrow";
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(global));
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        spdlog::info("  wrote {} pinned tensors -> {}", global.size(), path.string());
    }

    for (auto &[layer, tens] : by_layer) {
        auto path = output_root / "pinned" / (layer + ".arrow");
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(tens));
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        spdlog::info("  wrote {} pinned tensors for {} -> {}",
            tens.size(), layer, path.string());
    }

    return arrow::Status::OK();
}

// ── Model index ─────────────────────────────────────────────────────────────

arrow::Status WriteModelIndex(
    const std::string &model_name,
    int num_layers,
    int experts_per_layer,
    int64_t total_pinned,
    int64_t total_experts,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    std::ofstream f(output_root / "model.index.json");
    if (!f)
        return arrow::Status::IOError("Cannot create model.index.json");

    f << "{\n";
    f << "  \"format\": \"logestix-v1\",\n";
    f << "  \"model\": \"" << model_name << "\",\n";
    f << "  \"num_layers\": " << num_layers << ",\n";
    f << "  \"experts_per_layer\": " << experts_per_layer << ",\n";
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << "  \"files\": {\n";
    f << "    \"pinned\": {\n";
    f << "      \"global\": \"pinned/global.arrow\"\n";
    for (int i = 0; i < num_layers; ++i) {
        f << "      ,\"layer_" << std::setw(3) << std::setfill('0') << i
          << "\": \"pinned/layer_" << std::setw(3) << std::setfill('0') << i << ".arrow\"\n";
    }
    f << "    },\n";
    f << "    \"experts\": {\n";
    for (int i = 0; i < num_layers; ++i) {
        f << "      \"layer_" << std::setw(3) << std::setfill('0') << i
          << "\": \"experts/layer_" << std::setw(3) << std::setfill('0') << i << "/\"\n";
    }
    f << "    }\n";
    f << "  }\n";
    f << "}\n";
    f.close();

    spdlog::info("Wrote model.index.json");
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

    std::unordered_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    std::vector<TensorRecord> pinned_tensors;
    int experts_processed = 0;
    int pinned_processed = 0;
    int shard_idx = 0;
    int max_layer = 0;
    int max_experts_per_layer = 0;
    std::unordered_map<int, int> experts_per_layer;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        auto t0 = std::chrono::steady_clock::now();

        spdlog::info("[{}/{}] Scanning {} (pinned={}, experts={})",
            shard_idx, total_shards, shard.filename,
            pinned_processed, partial.size());

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

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz) {
                spdlog::warn("  Skipping {} (offset {}+{} exceeds file size {})",
                    tensor.name, data_off, tensor.data_begin, sz);
                continue;
            }

            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            if (parsed) {
                ExpertKey key{parsed->layer, parsed->expert};
                auto &exp = partial[key];
                max_layer = std::max(max_layer, parsed->layer + 1);
                experts_per_layer[parsed->layer]++;

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
                        ++experts_processed;
                    partial.erase(key);
                }
            } else {
                TensorRecord t;
                t.name  = tensor.name;
                t.dtype = tensor.dtype;
                t.shape = tensor.shape;
                t.data  = std::move(raw);
                pinned_tensors.push_back(std::move(t));
                ++pinned_processed;
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

    for (auto &[layer, cnt] : experts_per_layer)
        max_experts_per_layer = std::max(max_experts_per_layer, cnt);

    spdlog::info("Saving {} pinned tensors...", pinned_tensors.size());
    ARROW_RETURN_NOT_OK(SavePinnedTensors(pinned_tensors, impl_->output_path));

    ARROW_RETURN_NOT_OK(WriteModelIndex(
        fs::path(impl_->input_path).filename().string(),
        max_layer, max_experts_per_layer,
        static_cast<int64_t>(pinned_tensors.size()),
        static_cast<int64_t>(experts_processed),
        impl_->output_path));

    spdlog::info("Done — {} pinned tensors, {} experts processed, incomplete {}",
        pinned_processed, experts_processed, partial.size());
    return arrow::Status::OK();
}

} // namespace lsxcommon