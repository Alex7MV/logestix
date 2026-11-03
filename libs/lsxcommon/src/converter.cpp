#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/synthetic_geometry.h"
#include "lsxcommon/diagnostics.h"
#include "nvfp4_scale.h"
#include <arrow/ipc/writer.h>
#include <arrow/io/file.h>
#include <simdjson.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <regex>
#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <filesystem>

namespace lsxcommon {

namespace {

// ── Safetensors helpers ──────────────────────────────────────────────────────

std::string format_size(int64_t bytes)
{
    if (bytes >= 1024LL * 1024 * 1024)
        return absl::StrCat(bytes / (1024 * 1024 * 1024), " GiB");
    if (bytes >= 1024 * 1024)
        return absl::StrCat(bytes / (1024 * 1024), " MiB");
    if (bytes >= 1024)
        return absl::StrCat(bytes / 1024, " KiB");
    return absl::StrCat(bytes, " B");
}

// ── Expert name parsing ──────────────────────────────────────────────────────

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
    bool is_scale = false;
};

std::optional<ParsedName> parse_expert_name(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedName p;
    if (!absl::SimpleAtoi(m[1].str(), &p.layer) ||
        !absl::SimpleAtoi(m[2].str(), &p.expert) ||
        !absl::SimpleAtoi(m[3].str(), &p.weight_id))
        return std::nullopt;
    p.is_scale  = (m[4] == "scale");
    return p;
}

// Shared expert tensors: layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}
std::optional<ParsedName> parse_shared_expert_name(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.shared_experts\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedName p;
    if (!absl::SimpleAtoi(m[1].str(), &p.layer) ||
        !absl::SimpleAtoi(m[2].str(), &p.weight_id))
        return std::nullopt;
    p.expert    = 0;
    p.is_scale  = (m[3] == "scale");
    return p;
}

bool is_staged_tensor(const std::string &name)
{
    return name.find(".ffn.experts.") != std::string::npos;
}

// ── VRAM profile classification (spec 2026-08-21) ───────────────────────────

bool DsIsNvfp4Tensor(const std::string &name)
{
    // Only the packed nibble weights are NVFP4; their E8M0 scale tensors
    // stay byte-sized (counting them through the packed formula would halve
    // real scale bytes).
    const bool is_expert = parse_expert_name(name).has_value()
        || parse_shared_expert_name(name).has_value();
    return is_expert && name.size() >= 7 &&
           name.compare(name.size() - 7, 7, ".weight") == 0;
}

VramTensorClass DsMemClass(const std::string &name)
{
    return parse_expert_name(name).has_value()
        ? VramTensorClass::RoutedExpert
        : VramTensorClass::Resident;
}

} // anonymous namespace

// ── Safetensors public API ───────────────────────────────────────────────────

static std::string pinned_category_name(const std::string &name)
{
    if (name == "embed.weight")
        return "Embedding";
    if (name == "head.weight")
        return "LM Head";
    if (name.rfind("hc_head_", 0) == 0)
        return "LM Head";
    if (name.find(".attn.") != std::string::npos || name.find(".attn_norm.") != std::string::npos)
        return "Attention";
    if (name.find(".hc_attn") != std::string::npos)
        return "Attention";
    if (name.find(".hc_ffn") != std::string::npos)
        return "HC-FFN";
    if (name.find(".ffn.gate.") != std::string::npos)
        return "MoE Router";
    if (name.find(".ffn.shared_experts.") != std::string::npos)
        return "Shared Experts";
    if (name.find(".ffn_norm.") != std::string::npos)
        return "Layer Norms";
    if (name.rfind("model.norm.", 0) == 0 || name == "norm.weight")
        return "Final Norm";
    if (name.find(".enorm.") != std::string::npos || name.find(".hnorm.") != std::string::npos)
        return "MTP Norms";
    if (name.find(".e_proj.") != std::string::npos || name.find(".h_proj.") != std::string::npos)
        return "MTP Projections";
    if (name.rfind("mtp.", 0) == 0)
        return "MTP Norms";
    return "Other";
}

void print_shard(const SafetensorsModel::Shard &shard)
{
    lsxcommon::log::info(absl::StrFormat("--- %s (%s) ---", shard.filename,
                                         format_size(shard.file_size)));
    for (auto &t : shard.tensors) {
        std::string shape_str = absl::StrCat("[", absl::StrJoin(t.shape, ", "), "]");

        lsxcommon::log::info(absl::StrFormat(
            "  %s %-50s %-8s %-24s %s",
            is_staged_tensor(t.name) ? "[S]" : "[P]",
            t.name, t.dtype, shape_str, format_size(t.byte_size())));
    }
}

void print_model(const SafetensorsModel &model)
{
    int64_t total_tensors = 0;
    int64_t total_bytes = 0;
    int64_t pinned_tensors = 0;
    int64_t pinned_bytes = 0;
    int64_t staged_tensors = 0;
    int64_t staged_bytes = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            total_tensors++;
            total_bytes += t.byte_size();
            if (is_staged_tensor(t.name)) {
                staged_tensors++;
                staged_bytes += t.byte_size();
            } else {
                pinned_tensors++;
                pinned_bytes += t.byte_size();
            }
        }
    }

    lsxcommon::log::info("=== Model Summary ===");
    lsxcommon::log::info(absl::StrFormat("Shards: %u", model.shards.size()));
    lsxcommon::log::info(absl::StrFormat("Tensors: %d", total_tensors));
    lsxcommon::log::info(absl::StrFormat("Total tensor data: %s", format_size(total_bytes)));
    lsxcommon::log::info("--- Memory classification ---");
    lsxcommon::log::info(absl::StrFormat("Pinned (always in VRAM): %d tensors, %s",
        pinned_tensors, format_size(pinned_bytes)));
    lsxcommon::log::info(absl::StrFormat("Staged (MoE experts):   %d tensors, %s",
        staged_tensors, format_size(staged_bytes)));
    lsxcommon::log::info("--- Pinned breakdown ---");
    absl::btree_map<std::string, int64_t> cat_tensors, cat_bytes;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            if (is_staged_tensor(t.name)) continue;
            auto cat = pinned_category_name(t.name);
            cat_tensors[cat]++;
            cat_bytes[cat] += t.byte_size();
        }
    }
    for (auto &[cat, cnt] : cat_tensors) {
        lsxcommon::log::info(absl::StrFormat("  %-22s %6d tensors, %s",
            cat, cnt, format_size(cat_bytes[cat])));
    }
    // Runtime VRAM estimates (min/optimal/max) are no longer guessed here:
    // they are computed at conversion time and reported via the "vram" block
    // of model.index.json / IModel::VramRequirements() (see vram_estimate).
}

// ── Safetensors file parser ──────────────────────────────────────────────────

SafetensorsModel::Shard read_safetensors_file(const std::string &path)
{
    SafetensorsModel::Shard shard;
    shard.filename = std::filesystem::path(path).filename().string();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        lsxcommon::log::error(absl::StrCat("Cannot open file: ", path));
        return shard;
    }

    file.seekg(0, std::ios::end);
    shard.file_size = static_cast<int64_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 8);
    // file_size < 8 must be rejected before the subtraction: casting a small
    // size to uint64_t and subtracting 8 underflows into "any header fits".
    if (shard.file_size < 8 || !file || header_len == 0 ||
        header_len > static_cast<uint64_t>(shard.file_size) - 8) {
        lsxcommon::log::error(absl::StrCat("Invalid safetensors header in ", path));
        return shard;
    }

    std::string json_str(header_len, '\0');
    file.read(json_str.data(), header_len);
    if (!file) {
        lsxcommon::log::error(absl::StrCat("Failed to read safetensors header from ", path));
        return shard;
    }

    auto report = ParseTensorInfos(json_str);
    if (report.rejected > 0) {
        lsxcommon::log::warn(absl::StrFormat(
            "%s: %d malformed tensor description(s) dropped", path,
            report.rejected));
    }
    shard.tensors = std::move(report.tensors);
    return shard;
}

// ── TensorInfo implementation ─────────────────────────────────────────────────

int64_t TensorInfo::numel() const
{
    // An EMPTY shape list is a RANK-0 scalar, not an absent tensor: the
    // safetensors spec stores `[]` for a single value, and ModelOpt writes its
    // per-tensor `weight_scale` / `weight_scale_2` exactly that way (verified in
    // the NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 vault). Returning 0 here
    // made byte_size() 0, so every rank-0 scale read back as an empty buffer and
    // the converter rejected the tensor as "inconsistent companion sizes".
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

int64_t TensorInfo::byte_size() const
{
    return numel() * TensorDtypeBytes(dtype);
}

const char *TensorInfoErrorMessage(TensorInfoError err)
{
    switch (err) {
    case TensorInfoError::kOk: return "ok";
    case TensorInfoError::kEmptyName: return "empty tensor name";
    case TensorInfoError::kUnknownDtype: return "unknown dtype";
    case TensorInfoError::kNegativeDimension: return "negative dimension";
    case TensorInfoError::kNumelOverflow: return "element count overflows int64";
    case TensorInfoError::kByteSizeOverflow: return "byte size overflows int64";
    case TensorInfoError::kNegativeOffset: return "negative data offset";
    case TensorInfoError::kReversedOffsets: return "data_offsets are reversed";
    case TensorInfoError::kSizeMismatch:
        return "data_offsets span != numel * dtype width";
    }
    return "unknown tensor info error";
}

int64_t TensorDtypeBytes(std::string_view dtype)
{
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" ||
        dtype == "F8_E5M2" || dtype == "F8_E4M3" || dtype == "F8_E8M0")
        return 1;
    if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16")
        return 2;
    if (dtype == "I32" || dtype == "U32" || dtype == "F32")
        return 4;
    if (dtype == "I64" || dtype == "U64" || dtype == "F64")
        return 8;
    return 0;
}

TensorInfoError ValidateTensorInfo(const TensorInfo &info)
{
    if (info.name.empty())
        return TensorInfoError::kEmptyName;

    const int64_t width = TensorDtypeBytes(info.dtype);
    if (width == 0)
        return TensorInfoError::kUnknownDtype;

    // Cheap scalar checks first so the reported reason does not depend on the
    // shape when both are invalid (documented precedence, pinned by tests).
    if (info.data_begin < 0 || info.data_end < 0)
        return TensorInfoError::kNegativeOffset;
    if (info.data_end < info.data_begin)
        return TensorInfoError::kReversedOffsets;

    // numel() multiplies without checking; a hostile header (2^62 x 4) would
    // wrap to a small positive number and the reader would allocate a buffer
    // far smaller than the tensor it then "reads".
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    int64_t numel = 1;
    for (int64_t dim : info.shape) {
        if (dim < 0)
            return TensorInfoError::kNegativeDimension;
        if (dim > 0 && numel > kMax / dim)
            return TensorInfoError::kNumelOverflow;
        numel *= dim;
    }
    if (numel > kMax / width)
        return TensorInfoError::kByteSizeOverflow;

    const int64_t bytes = numel * width;
    if (info.data_end - info.data_begin != bytes)
        return TensorInfoError::kSizeMismatch;
    return TensorInfoError::kOk;
}

TensorInfoParseReport ParseTensorInfos(std::string_view header_json)
{
    TensorInfoParseReport report;

    // DOM (not ondemand) on purpose: it validates the whole document, so a
    // header truncated after a complete value is rejected instead of being
    // half-read. A shard header is parsed once, so the extra pass is free.
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(simdjson::padded_string(header_json.data(),
                                             header_json.size()))
            .get(root) != simdjson::SUCCESS)
        return report;

    simdjson::dom::object obj;
    if (root.get_object().get(obj) != simdjson::SUCCESS)
        return report;

    for (auto [key, value] : obj) {
        if (key == "__metadata__")
            continue;

        simdjson::dom::object tensor_obj;
        if (value.get_object().get(tensor_obj) != simdjson::SUCCESS) {
            ++report.rejected;
            continue;
        }

        // Field order inside a tensor object is not part of the spec; DOM
        // lookups are order-independent.
        TensorInfo info;
        info.name.assign(key);

        std::string_view dtype;
        if (tensor_obj["dtype"].get_string().get(dtype) != simdjson::SUCCESS) {
            ++report.rejected;
            continue;
        }
        info.dtype.assign(dtype);

        simdjson::dom::array shape_arr;
        if (tensor_obj["shape"].get_array().get(shape_arr) != simdjson::SUCCESS) {
            ++report.rejected;
            continue;
        }
        bool malformed = false;
        for (auto dim_val : shape_arr) {
            int64_t dim = 0;
            if (dim_val.get_int64().get(dim) != simdjson::SUCCESS) {
                malformed = true;
                break;
            }
            info.shape.push_back(dim);
        }
        if (malformed) {
            ++report.rejected;
            continue;
        }

        simdjson::dom::array offsets_arr;
        if (tensor_obj["data_offsets"].get_array().get(offsets_arr) !=
            simdjson::SUCCESS) {
            ++report.rejected;
            continue;
        }
        int64_t offsets[2] = {0, 0};
        int64_t offset_count = 0;
        for (auto off_val : offsets_arr) {
            int64_t off = 0;
            if (offset_count >= 2 ||
                off_val.get_int64().get(off) != simdjson::SUCCESS) {
                malformed = true;
                break;
            }
            offsets[offset_count++] = off;
        }
        if (malformed || offset_count != 2) {
            ++report.rejected;
            continue;
        }
        info.data_begin = offsets[0];
        info.data_end = offsets[1];

        if (ValidateTensorInfo(info) != TensorInfoError::kOk) {
            ++report.rejected;
            continue;
        }
        report.tensors.push_back(std::move(info));
    }
    return report;
}

// ── Validated tensor payload reader ──────────────────────────────────────────

const char *TensorReadErrorMessage(TensorReadError err)
{
    switch (err) {
    case TensorReadError::kOk: return "ok";
    case TensorReadError::kInvalidDescriptor: return "invalid tensor descriptor";
    case TensorReadError::kOpenFailed: return "cannot open shard";
    case TensorReadError::kShortHeader: return "shard shorter than its header";
    case TensorReadError::kHeaderTooLarge: return "header length exceeds the file";
    case TensorReadError::kOutOfBounds: return "tensor range exceeds the payload";
    case TensorReadError::kShortRead: return "short read";
    }
    return "unknown tensor read error";
}

TensorReadResult ReadTensorDataChecked(const std::filesystem::path &input_dir,
                                       const SafetensorsModel::Shard &shard,
                                       const TensorInfo &info)
{
    TensorReadResult result;

    const TensorInfoError descriptor = ValidateTensorInfo(info);
    if (descriptor != TensorInfoError::kOk) {
        result.error = TensorReadError::kInvalidDescriptor;
        result.detail = absl::StrCat("tensor '", info.name, "': ",
                                     TensorInfoErrorMessage(descriptor));
        return result;
    }

    const std::filesystem::path path = input_dir / shard.filename;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        result.error = TensorReadError::kOpenFailed;
        result.detail = absl::StrCat(path.string(), ": cannot open");
        return result;
    }

    // The reader is self-contained: it re-derives the file size and the header
    // length instead of trusting shard.file_size.
    file.seekg(0, std::ios::end);
    const std::streamoff end = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file || end < 8) {
        result.error = TensorReadError::kShortHeader;
        result.detail = absl::StrCat(path.string(), ": shorter than 8 bytes");
        return result;
    }
    const int64_t file_size = static_cast<int64_t>(end);

    uint64_t header_len = 0;
    file.read(reinterpret_cast<char *>(&header_len), 8);
    if (!file || file.gcount() != 8) {
        result.error = TensorReadError::kShortHeader;
        result.detail = absl::StrCat(path.string(), ": cannot read header length");
        return result;
    }
    if (header_len > static_cast<uint64_t>(file_size) - 8) {
        result.error = TensorReadError::kHeaderTooLarge;
        result.detail = absl::StrFormat(
            "%s: header length %d exceeds file size %d", path.string(),
            static_cast<int64_t>(header_len), file_size);
        return result;
    }

    const int64_t data_off = 8 + static_cast<int64_t>(header_len);
    const int64_t payload_size = file_size - data_off;
    if (info.data_end > payload_size) {
        result.error = TensorReadError::kOutOfBounds;
        result.detail = absl::StrFormat(
            "%s: tensor '%s' exceeds the %d-byte payload (ends at %d)",
            path.string(), info.name, payload_size, info.data_end);
        return result;
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(info.byte_size()));
    if (!bytes.empty()) {
        file.seekg(data_off + info.data_begin, std::ios::beg);
        file.read(reinterpret_cast<char *>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (file.gcount() != static_cast<std::streamsize>(bytes.size())) {
            result.error = TensorReadError::kShortRead;
            result.detail = absl::StrFormat(
                "%s: tensor '%s': expected %d bytes, got %d", path.string(),
                info.name, info.byte_size(), static_cast<int64_t>(file.gcount()));
            return result;
        }
    }
    result.bytes = std::move(bytes);
    return result;
}

std::vector<uint8_t> ReadTensorData(const std::filesystem::path &input_dir,
                                    const SafetensorsModel::Shard &shard,
                                    const TensorInfo &info)
{
    TensorReadResult result = ReadTensorDataChecked(input_dir, shard, info);
    if (!result.ok()) {
        lsxcommon::log::error(absl::StrCat("ReadTensorData: ", result.detail));
        return {};
    }
    return std::move(result.bytes);
}

// ── Safetensors scanning ─────────────────────────────────────────────────────

SafetensorsModel scan_directory(const std::string &path)
{
    SafetensorsModel model;

    namespace fs = std::filesystem;
    fs::path dir(path);

    if (!fs::exists(dir)) {
        lsxcommon::log::error(absl::StrCat("Path does not exist: ", path));
        return model;
    }

    if (fs::is_regular_file(dir)) {
        auto shard = read_safetensors_file(dir.string());
        if (!shard.tensors.empty())
            model.shards.push_back(std::move(shard));
        return model;
    }

    std::vector<std::string> ordered_files;
    fs::path index_path = dir / "model.safetensors.index.json";

    if (fs::exists(index_path)) {
        std::ifstream index_file(index_path);
        if (index_file) {
            std::string index_json((std::istreambuf_iterator<char>(index_file)),
                                    std::istreambuf_iterator<char>());

            simdjson::ondemand::parser parser;
            simdjson::padded_string padded(index_json.data(), index_json.size());
            simdjson::ondemand::document doc;
            if (parser.iterate(padded).get(doc) == simdjson::SUCCESS) {
                simdjson::ondemand::object root;
                if (doc.get_object().get(root) == simdjson::SUCCESS) {
                    simdjson::ondemand::object weight_map;
                    if (root["weight_map"].get_object().get(weight_map) == simdjson::SUCCESS) {
                        for (auto entry : weight_map) {
                            std::string_view shard_name;
                            if (entry.value().get_string().get(shard_name) == simdjson::SUCCESS) {
                                std::string fn(shard_name);
                                if (std::find(ordered_files.begin(), ordered_files.end(), fn) == ordered_files.end())
                                    ordered_files.push_back(fn);
                            }
                        }
                    }
                }
            }
        }
    }

    if (ordered_files.empty()) {
        for (auto &entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".safetensors")
                ordered_files.push_back(entry.path().filename().string());
        }
        std::sort(ordered_files.begin(), ordered_files.end());
    }

    for (auto &fn : ordered_files) {
        auto full_path = (dir / fn).string();
        lsxcommon::log::info(absl::StrCat("Reading: ", fn));
        auto shard = read_safetensors_file(full_path);
        if (!shard.tensors.empty())
            model.shards.push_back(std::move(shard));
    }

    return model;
}

// ── NVFP4 Packing (GPU) ─────────────────────────────────────────────────────

arrow::Result<PackedExpert> PackExpertNVFP4(const ExpertData &raw)
{
    PackedExpert pe;
    pe.w1_shape = raw.w1_shape;
    pe.w2_shape = raw.w2_shape;
    pe.w3_shape = raw.w3_shape;

    auto product = [](const std::vector<int64_t> &v) -> int64_t {
        int64_t n = 1;
        for (auto d : v) n *= d;
        return n;
    };

    auto process = [&](const std::vector<uint8_t> &src,
                        const std::string &dtype,
                        const std::vector<int64_t> &shape,
                        const std::vector<uint8_t> &src_scale,
                        const std::vector<int64_t> &scale_shape,
                        std::vector<uint8_t> &packed,
                        std::vector<uint8_t> &scales,
                        float &global_scale) -> bool
    {
        int64_t numel = product(shape);
        if (numel == 0 || src.empty()) return false;

        // Routed experts on disk are already E2M1-packed 4-bit (I8/U8
        // storage) with paired F8_E8M0 per-32 exponent scales. The NVFP4
        // contour (lsx_nvfp4) decodes pow2-E4M3 per-16 scales multiplied by
        // a per-tensor global_scale, so the repack splits every 32-block
        // into two identical per-16 halves and folds an exact power-of-two
        // shift into the stored global_scale (nvfp4_scale.h; pinned by
        // tests/test_deepseek_moe_convert.cpp). F8_E4M3 payloads (shared
        // experts) keep the verbatim passthrough.
        if ((dtype == "I8" || dtype == "U8") && !src_scale.empty()) {
            packed = src;
            uint8_t b_min = 0xFF, b_max = 0x00;
            bool any_nonzero = false;
            for (const uint8_t b : src_scale) {
                if (b == 0) continue;
                any_nonzero = true;
                b_min = std::min(b_min, b);
                b_max = std::max(b_max, b);
            }
            int shift = 0;
            if (any_nonzero) {
                if (!nvfp4::ChooseE8m0ScaleShift(b_min, b_max, shift))
                    return false;   // spread too wide for exact repack
            }
            scales.resize(src_scale.size() * 2);
            for (size_t i = 0; i < src_scale.size(); ++i) {
                uint8_t pair[2];
                nvfp4::E8m0ToE4m3Pow2Pair(src_scale[i], shift, pair);
                scales[2 * i] = pair[0];
                scales[2 * i + 1] = pair[1];
            }
            global_scale = std::ldexp(1.0f, shift);
            return true;
        }

        // Shared experts (F8_E4M3 + per-128 tile scales): verbatim.
        if (dtype == "F8_E4M3" && !src_scale.empty()) {
            packed       = src;
            scales       = src_scale;
            global_scale = 1.0f;
            return true;
        }

        const uint8_t *e8m0 = src_scale.empty() ? nullptr : src_scale.data();
        return gpu::quantize_tensor_gpu(src.data(), numel, dtype,
                                         e8m0, shape, scale_shape,
                                         packed, scales, global_scale);
    };

    if (!process(raw.w1_raw, raw.w1_dtype, raw.w1_shape,
                  raw.w1_scale_raw, raw.w1_scale_shape,
                  pe.w1_packed, pe.w1_scales, pe.w1_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w1");

    if (!process(raw.w2_raw, raw.w2_dtype, raw.w2_shape,
                  raw.w2_scale_raw, raw.w2_scale_shape,
                  pe.w2_packed, pe.w2_scales, pe.w2_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w2");

    if (!process(raw.w3_raw, raw.w3_dtype, raw.w3_shape,
                  raw.w3_scale_raw, raw.w3_scale_shape,
                  pe.w3_packed, pe.w3_scales, pe.w3_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w3");

    return pe;
}

// ── Shared converter write helpers ──────────────────────────────────────────

arrow::Status WriteBatch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

std::filesystem::path ExpertArrowPath(const std::filesystem::path &model_root,
                                      int layer, int expert)
{
    return model_root / "experts" / absl::StrCat("layer_", layer) /
           absl::StrCat("exp_", expert, ".arrow");
}

arrow::Status WriteExpertArrow(const std::filesystem::path &path,
                               const PackedExpert &expert)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    auto batch = ExpertSchema::MakeBatchView(expert);
    if (!batch.ok()) return batch.status();
    diag::ValidateBatchBuffers(**batch, "WriteExpertArrow pre-write");
    return WriteRecordBatchToFile(*batch, path);
}

bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   const uint8_t *e8m0, const std::vector<int64_t> &shape,
                   const std::vector<int64_t> &e8m0_shape,
                   DenseNvfp4Record &out)
{
    return gpu::quantize_tensor_gpu(data, numel, dtype, e8m0, shape, e8m0_shape,
                                    out.packed, out.scales, out.global_scale);
}

bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   DenseNvfp4Record &out)
{
    return QuantizeSlice(data, numel, dtype, nullptr, out.shape, {}, out);
}

bool QuantizeSliceBf16(const uint8_t *data, int64_t numel,
                       DenseNvfp4Record &out)
{
    return QuantizeSlice(data, numel, "BF16", out);
}

// ── Save single expert to Arrow IPC file ────────────────────────────────────

arrow::Status SaveExpertToArrow(
    const PackedExpert &expert,
    int layer_idx,
    int expert_idx,
    const std::filesystem::path &output_root)
{
    const auto path = ExpertArrowPath(output_root, layer_idx, expert_idx);
    ARROW_RETURN_NOT_OK(WriteExpertArrow(path, expert));
    lsxcommon::log::info(absl::StrFormat("  wrote layer=%03d expert=%03d  -> %s",
                                         layer_idx, expert_idx, path.string()));
    return arrow::Status::OK();
}

// ── Save all pinned tensors ─────────────────────────────────────────────────

static arrow::Status SavePinnedTensors(
    const std::vector<TensorRecord> &tensors,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    fs::create_directories(output_root / "pinned");

    absl::flat_hash_map<std::string, std::vector<TensorRecord>> by_layer;
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
        diag::ValidateBatchBuffers(*batch, "SavePinnedTensors global");
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        lsxcommon::log::info(absl::StrFormat("  wrote %u pinned tensors -> %s",
                                             global.size(), path.string()));
    }

    for (auto &[layer, tens] : by_layer) {
        auto path = output_root / "pinned" / (layer + ".arrow");
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(tens));
        diag::ValidateBatchBuffers(*batch, ("SavePinnedTensors " + layer).c_str());
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        lsxcommon::log::info(absl::StrFormat("  wrote %u pinned tensors for %s -> %s",
            tens.size(), layer, path.string()));
    }

    return arrow::Status::OK();
}

// ── Save all shared expert layers ────────────────────────────────────────────

// Packs the NVFP4-compressed shared expert weights (w1/w2/w3) for every layer
// and writes a single pinned/shared_experts.arrow with one record per layer.
arrow::Status SaveSharedExpertsToArrow(
    const absl::flat_hash_map<int, ExpertData> &shared_by_layer,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    if (shared_by_layer.empty()) {
        lsxcommon::log::info("  no shared experts found, skipping shared_experts.arrow");
        return arrow::Status::OK();
    }

    int max_layer = -1;
    for (auto &[layer, sd] : shared_by_layer)
        max_layer = std::max(max_layer, layer);
    if (max_layer < 0)
        return arrow::Status::OK();

    std::vector<PackedExpert> shared_layers;
    shared_layers.resize(static_cast<size_t>(max_layer) + 1);

    int packed_ok = 0;
    for (auto &[layer, sd] : shared_by_layer) {
        if (sd.w1_raw.empty() || sd.w2_raw.empty() || sd.w3_raw.empty()) {
            lsxcommon::log::warn(absl::StrFormat(
                "  incomplete shared expert layer %d (w1=%u w2=%u w3=%u)",
                layer, sd.w1_raw.size(), sd.w2_raw.size(), sd.w3_raw.size()));
            continue;
        }
        auto packed_res = PackExpertNVFP4(std::move(sd));
        if (!packed_res.ok()) {
            lsxcommon::log::error(absl::StrFormat("  failed to pack shared expert layer %d: %s",
                layer, packed_res.status().ToString()));
            continue;
        }
        shared_layers[static_cast<size_t>(layer)] = *packed_res;
        ++packed_ok;
    }

    while (!shared_layers.empty() && shared_layers.back().w1_packed.empty())
        shared_layers.pop_back();
    if (shared_layers.empty())
        return arrow::Status::OK();

    fs::create_directories(output_root / "pinned");
    ARROW_ASSIGN_OR_RAISE(auto batch, SharedExpertSchema::MakeBatch(shared_layers));
    diag::ValidateBatchBuffers(*batch, "SaveSharedExpertsToArrow result");
    auto path = output_root / "pinned" / "shared_experts.arrow";
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
    lsxcommon::log::info(absl::StrFormat("  packed %d shared-expert layers -> %s",
                                         packed_ok, path.string()));

    return arrow::Status::OK();
}

// ── Targeted shared-expert repack ───────────────────────────────────────────

// Re-packs only the shared-expert weights from the source safetensors shards
// into pinned/shared_experts.arrow. Reads tensors lazily per shard; skips all
// routed experts and pinned tensors. Preserves the paired E8M0 scales so the
// already-quantized NVFP4 values are dequantized with their true magnitude.
arrow::Status RepackSharedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       input_path);

    absl::flat_hash_map<int, ExpertData> shared_by_layer;
    int shard_idx = 0;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        lsxcommon::log::info(absl::StrFormat("[%d/%u] scanning %s ...",
                                             shard_idx, model.shards.size(),
                                             shard.filename));

        std::ifstream sf((fs::path(input_path) / shard.filename).string(),
                         std::ios::binary | std::ios::ate);
        if (!sf) { lsxcommon::log::warn("  Cannot open, skipping"); continue; }
        int64_t sz = sf.tellg();
        if (sz <= 0) { lsxcommon::log::warn("  Zero/negative size, skipping"); continue; }
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf) { lsxcommon::log::warn("  Cannot read, skipping"); continue; }

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        for (const auto &tensor : shard.tensors) {
            auto sparsed = parse_shared_expert_name(tensor.name);
            if (!sparsed) continue;

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz) {
                lsxcommon::log::warn(absl::StrFormat("  Skipping %s (offset out of range)",
                                                     tensor.name));
                continue;
            }
            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            auto &sexp = shared_by_layer[sparsed->layer];
            if (sparsed->is_scale) {
                switch (sparsed->weight_id) {
                case 1:
                    sexp.w1_scale_raw   = std::move(raw);
                    sexp.w1_scale_shape = tensor.shape;
                    break;
                case 2:
                    sexp.w2_scale_raw   = std::move(raw);
                    sexp.w2_scale_shape = tensor.shape;
                    break;
                case 3:
                    sexp.w3_scale_raw   = std::move(raw);
                    sexp.w3_scale_shape = tensor.shape;
                    break;
                }
            } else {
                switch (sparsed->weight_id) {
                case 1:
                    sexp.w1_raw   = std::move(raw);
                    sexp.w1_shape = tensor.shape;
                    sexp.w1_dtype = tensor.dtype;
                    break;
                case 2:
                    sexp.w2_raw   = std::move(raw);
                    sexp.w2_shape = tensor.shape;
                    sexp.w2_dtype = tensor.dtype;
                    break;
                case 3:
                    sexp.w3_raw   = std::move(raw);
                    sexp.w3_shape = tensor.shape;
                    sexp.w3_dtype = tensor.dtype;
                    break;
                }
            }
        }
    }

    lsxcommon::log::info(absl::StrFormat("Packing %u shared-expert layers (NVFP4)...",
                                         shared_by_layer.size()));
    return SaveSharedExpertsToArrow(shared_by_layer, output_root);
}

// ── Targeted routed-expert repack (fp4 passthrough) ─────────────────────────

// Re-writes only the routed expert arrows from the source safetensors shards.
// The source weights are already fp4 E2M1 packed (2 elems/byte) with paired
// F8_E8M0 per-32-element scales, so the bytes are copied through unchanged.
// CPU-only: no CUDA quantization is involved.
arrow::Status RepackRoutedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       input_path);

    absl::flat_hash_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    int shard_idx = 0;
    int total_written = 0;
    int max_layer = 0;
    int max_expert_per_layer = 0;
    absl::flat_hash_map<int, int> experts_per_layer;

    auto try_flush = [&](ExpertKey key, ExpertData &exp) -> arrow::Status {
        if (exp.w1_raw.empty() || exp.w2_raw.empty() || exp.w3_raw.empty())
            return arrow::Status::OK();
        if (exp.w1_scale_raw.empty() || exp.w2_scale_raw.empty() ||
            exp.w3_scale_raw.empty()) {
            lsxcommon::log::warn(absl::StrFormat(
                "  expert layer=%d expert=%d missing E8M0 scales "
                "(w1=%u w2=%u w3=%u), skipping",
                key.layer, key.expert,
                exp.w1_scale_raw.size(), exp.w2_scale_raw.size(),
                exp.w3_scale_raw.size()));
            return arrow::Status::OK();
        }
        ARROW_ASSIGN_OR_RAISE(auto packed, PackExpertNVFP4(std::move(exp)));
        ARROW_RETURN_NOT_OK(SaveExpertToArrow(
            packed, key.layer, key.expert, output_root));
        ++total_written;
        ++experts_per_layer[key.layer];
        max_expert_per_layer = std::max(max_expert_per_layer, key.expert + 1);
        max_layer = std::max(max_layer, key.layer + 1);
        return arrow::Status::OK();
    };

    for (const auto &shard : model.shards) {
        ++shard_idx;
        lsxcommon::log::info(absl::StrFormat("[%d/%u] scanning %s ...",
                                             shard_idx, model.shards.size(),
                                             shard.filename));

        std::ifstream sf((fs::path(input_path) / shard.filename).string(),
                         std::ios::binary | std::ios::ate);
        if (!sf) { lsxcommon::log::warn("  Cannot open, skipping"); continue; }
        int64_t sz = sf.tellg();
        if (sz <= 0) { lsxcommon::log::warn("  Zero/negative size, skipping"); continue; }
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf) { lsxcommon::log::warn("  Cannot read, skipping"); continue; }

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        for (const auto &tensor : shard.tensors) {
            auto parsed = parse_expert_name(tensor.name);
            if (!parsed) continue;

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz) {
                lsxcommon::log::warn(absl::StrFormat("  Skipping %s (offset out of range)",
                                                     tensor.name));
                continue;
            }
            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            ExpertKey key{parsed->layer, parsed->expert};
            auto &exp = partial[key];

            if (parsed->is_scale) {
                switch (parsed->weight_id) {
                case 1:
                    exp.w1_scale_raw   = std::move(raw);
                    exp.w1_scale_shape = tensor.shape;
                    break;
                case 2:
                    exp.w2_scale_raw   = std::move(raw);
                    exp.w2_scale_shape = tensor.shape;
                    break;
                case 3:
                    exp.w3_scale_raw   = std::move(raw);
                    exp.w3_scale_shape = tensor.shape;
                    break;
                }
            } else {
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
            }

            if (exp.w1_raw.size() && exp.w2_raw.size() && exp.w3_raw.size() &&
                exp.w1_scale_raw.size() && exp.w2_scale_raw.size() &&
                exp.w3_scale_raw.size()) {
                auto st = try_flush(key, exp);
                if (!st.ok())
                    lsxcommon::log::error(absl::StrFormat(
                        "  Failed to pack expert layer=%d expert=%d: %s",
                        key.layer, key.expert, st.ToString()));
                partial.erase(key);
            }
        }
    }

    for (auto &[key, exp] : partial) {
        lsxcommon::log::warn(absl::StrFormat(
            "  incomplete expert layer=%d expert=%d (w1=%u w2=%u w3=%u)",
            key.layer, key.expert,
            exp.w1_raw.size(), exp.w2_raw.size(), exp.w3_raw.size()));
    }

    lsxcommon::log::info(absl::StrFormat(
        "Repacked %d routed experts (%d layers, up to %d experts/layer)",
        total_written, max_layer, max_expert_per_layer));
    return arrow::Status::OK();
}

// ── VRAM profile recompute (migration of pre-existing index files) ─────────

arrow::Status RecomputeVramBlock(
    const std::string &input_path,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    auto index_path = fs::path(output_root) / "logestix" / "model.index.json";
    std::ifstream f(index_path);
    if (!f) return arrow::Status::IOError("cannot open ", index_path.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    // Layer grid from routed-expert names.
    int max_layer = 0;
    absl::btree_map<int, absl::btree_set<int>> experts_by_layer;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            if (auto p = parse_expert_name(t.name)) {
                experts_by_layer[p->layer].insert(p->expert);
                max_layer = std::max(max_layer, p->layer + 1);
            }
        }
    }
    int experts_per_layer = 0;
    for (auto &[_, s] : experts_by_layer)
        experts_per_layer = std::max(experts_per_layer,
                                     static_cast<int>(s.size()));

    VramGeometry vg;
    vg.num_attn_layers = max_layer;
    vg.num_moe_layers = static_cast<int>(experts_by_layer.size());
    vg.experts_per_layer = experts_per_layer;
    vg.nvfp4_scale_block = 32;   // DeepSeek sources ship fp4, 32 elems/scale
    vg.context_length = ReadContextLength(
        (fs::path(input_path) / "config.json").string());
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            if (vg.kv_dim == 0 && t.shape.size() >= 2 &&
                t.name.find("wkv.weight") != std::string::npos) {
                vg.kv_dim = static_cast<int>(t.shape[0]);
                vg.hidden_dim = static_cast<int>(t.shape[1]);
            }
        }
    }

    // Per-layer KV compression from config (DeepSeek CSA).
    {
        std::ifstream cf((fs::path(input_path) / "config.json"));
        if (cf) {
            std::string cj((std::istreambuf_iterator<char>(cf)),
                           std::istreambuf_iterator<char>());
            simdjson::ondemand::parser cp;
            simdjson::padded_string cpj(cj.data(), cj.size());
            simdjson::ondemand::document cd;
            if (cp.iterate(cpj).get(cd) == simdjson::SUCCESS) {
                auto root = cd.get_object();
                if (root.error() == simdjson::SUCCESS) {
                    if (auto arr = root.value()["compress_ratios"].get_array();
                        arr.error() == simdjson::SUCCESS) {
                        for (auto v : arr.value()) {
                            int64_t x;
                            if (v.get_int64().get(x) == simdjson::SUCCESS)
                                vg.kv_compress_ratios.push_back(x);
                        }
                    }
                }
            }
        }
    }

    std::vector<VramTensorInfo> inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors)
            inventory.push_back({t.name, t.numel(), t.byte_size()});

    VramProfile profile = EstimateVram(
        inventory, {&DsIsNvfp4Tensor, &DsMemClass}, vg);
    lsxcommon::log::info(absl::StrFormat(
        "VRAM profile recomputed: min=%s optimal=%s max=%s (ctx=%d)",
        FormatBytesHuman(profile.min_bytes),
        FormatBytesHuman(profile.optimal_bytes),
        FormatBytesHuman(profile.max_bytes),
        profile.context_tokens));

    std::ofstream out(index_path, std::ios::trunc);
    if (!out) return arrow::Status::IOError("cannot write ", index_path.string());
    out << ReplaceVramBlock(json, profile);
    return arrow::Status::OK();
}

// ── Model index ─────────────────────────────────────────────────────────────

static arrow::Status WriteModelIndex(
    const std::string &model_name,
    int num_layers,
    int experts_per_layer,
    int64_t total_pinned,
    int64_t total_experts,
    const VramProfile &vram,
    const absl::btree_map<std::string, int> &deferred,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    std::ofstream f(output_root / "model.index.json");
    if (!f)
        return arrow::Status::IOError("Cannot create model.index.json");

    f << "{\n";
    f << "  \"format\": \"logestix-v1-nvfp4\",\n";
    f << "  \"model\": \"" << model_name << "\",\n";
    f << "  \"num_layers\": " << num_layers << ",\n";
    f << "  \"experts_per_layer\": " << experts_per_layer << ",\n";
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << FormatVramJson(vram) << ",\n";
    if (!deferred.empty()) {
        f << "  \"deferred\": {";
        bool first = true;
        for (const auto &[cat, n] : deferred) {
            if (!first) f << ", ";
            first = false;
            f << "\"" << cat << "\": " << n;
        }
        f << "},\n";
    }
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
          << "\": \"experts/layer_" << std::setw(3) << std::setfill('0') << i << "/\"";
        if (i < num_layers - 1) f << ",";
        f << "\n";
    }
    f << "    }\n";
    f << "  }\n";
    f << "}\n";
    f.close();

    lsxcommon::log::info("Wrote model.index.json");
    return arrow::Status::OK();
}

// ── Converter implementation ────────────────────────────────────────────────

struct Converter::Impl {
    std::string input_path;
    std::string output_path;
    bool verbose = false;
    std::vector<std::string> skip_prefixes;
    std::string model_name;       // empty -> input directory name
    bool expected_geometry_set = false;
    int expected_layers = 0;
    int expected_hidden = 0;
    int expected_vocab = 0;
    int expected_experts = 0;
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

Converter &Converter::SetSkipPrefixes(std::vector<std::string> prefixes)
{
    impl_->skip_prefixes = std::move(prefixes);
    return *this;
}

Converter &Converter::SetModelName(std::string name)
{
    impl_->model_name = std::move(name);
    return *this;
}

Converter &Converter::SetExpectedGeometry(int layers, int hidden, int vocab,
                                          int experts)
{
    impl_->expected_geometry_set = true;
    impl_->expected_layers = layers;
    impl_->expected_hidden = hidden;
    impl_->expected_vocab = vocab;
    impl_->expected_experts = experts;
    return *this;
}

arrow::Status Converter::Run()
{
    namespace fs = std::filesystem;

    if (impl_->input_path.empty())
        return arrow::Status::Invalid("input path not set");
    if (impl_->output_path.empty())
        return arrow::Status::Invalid("output path not set");

    // Mandatory layout: all converted artifacts live under <root>/logestix
    // (model.index.json, pinned/, experts/), mirroring the Qwen/GLM pipelines.
    fs::path log_root = fs::path(impl_->output_path) / "logestix";
    fs::create_directories(log_root);

    auto model = lsxcommon::scan_directory(impl_->input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       impl_->input_path);

    // (2) config.json is mandatory; parsed for geometry cross-checks.
    auto cfg_path = fs::path(impl_->input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ",
                                      impl_->input_path);

    int cfg_num_layers = 0;
    int cfg_hidden_size = 0;
    int cfg_vocab_size = 0;
    int cfg_experts_per_layer = 0;
    {
        std::ifstream cf(cfg_path);
        if (!cf)
            return arrow::Status::IOError("cannot open ", cfg_path.string());
        std::string cj((std::istreambuf_iterator<char>(cf)),
                       std::istreambuf_iterator<char>());
        simdjson::padded_string cpj(cj.data(), cj.size());
        simdjson::ondemand::parser cp;
        simdjson::ondemand::document cd;
        if (cp.iterate(cpj).get(cd) != simdjson::SUCCESS)
            return arrow::Status::Invalid("config.json parse failed");
        auto root = cd.get_object();
        if (root.error() != simdjson::SUCCESS)
            return arrow::Status::Invalid("config.json root not an object");
        auto get_i = [&](simdjson::ondemand::object &o, const char *k, int dflt) {
            int64_t v;
            if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
            return static_cast<int>(v);
        };
        cfg_num_layers = get_i(root.value(), "num_hidden_layers", 0);
        if (cfg_num_layers <= 0)
            cfg_num_layers = get_i(root.value(), "num_layers", 0);
        cfg_hidden_size = get_i(root.value(), "hidden_size", 0);
        cfg_vocab_size = get_i(root.value(), "vocab_size", 0);
        cfg_experts_per_layer = get_i(root.value(), "num_local_experts", 0);
        if (cfg_experts_per_layer <= 0)
            cfg_experts_per_layer = get_i(root.value(), "n_routed_experts", 0);
    }

    // (3) Verify every tensor read is complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(
        impl_->input_path, model));

    auto total_shards = model.shards.size();
    lsxcommon::log::info(absl::StrFormat(
        "Found %u shards, %u tensors across %u shards",
        total_shards,
        [&]{
            std::size_t n = 0;
            for (auto &s : model.shards) n += s.tensors.size();
            return n;
        }(),
        total_shards));

    absl::flat_hash_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    std::vector<TensorRecord> pinned_tensors;
    absl::flat_hash_map<int, ExpertData> shared_by_layer;
    int experts_processed = 0;
    int pinned_processed = 0;
    int deferred_tensors = 0;
    int shard_idx = 0;
    int max_layer = 0;
    int max_experts_per_layer = 0;
    absl::btree_map<std::string, int> deferred;
    absl::flat_hash_map<int, int> experts_per_layer;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        auto t0 = std::chrono::steady_clock::now();

        lsxcommon::log::info(absl::StrFormat(
            "[%d/%u] Scanning %s (pinned=%d, experts=%u)",
            shard_idx, total_shards, shard.filename,
            pinned_processed, partial.size()));

        std::ifstream sf(
            (fs::path(impl_->input_path) / shard.filename).string(),
            std::ios::binary | std::ios::ate);
        if (!sf)
            return arrow::Status::IOError("cannot open ", shard.filename);
        int64_t sz = sf.tellg();
        if (sz <= 0)
            return arrow::Status::IOError("zero/negative size: ",
                                          shard.filename);
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf)
            return arrow::Status::IOError("cannot read ", shard.filename);

        lsxcommon::log::info(absl::StrFormat("  read %d MB", sz / (1024 * 1024)));

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        int experts_in_shard = 0;
        for (const auto &tensor : shard.tensors) {
            // Ballast cut by prefix (e.g. DeepSeek-V4-Pro's DSpark `mtp.*`):
            // counted in the index's "deferred" block and never read,
            // classified or written. A silent drop is not a strip.
            bool cut = false;
            for (const auto &pfx : impl_->skip_prefixes) {
                if (tensor.name.rfind(pfx, 0) == 0) {
                    std::string key = pfx;
                    while (!key.empty() && key.back() == '.') key.pop_back();
                    ++deferred[key];
                    ++deferred_tensors;
                    cut = true;
                    break;
                }
            }
            if (cut) continue;

            auto parsed = parse_expert_name(tensor.name);

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz)
                return arrow::Status::Invalid(
                    shard.filename, " tensor ", tensor.name,
                    " offset ", data_off, "+", tensor.data_begin,
                    " exceeds file size ", sz);

            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            if (parsed) {
                ExpertKey key{parsed->layer, parsed->expert};
                auto &exp = partial[key];
                max_layer = std::max(max_layer, parsed->layer + 1);

                if (parsed->is_scale) {
                    switch (parsed->weight_id) {
                    case 1:
                        exp.w1_scale_raw   = std::move(raw);
                        exp.w1_scale_shape = tensor.shape;
                        break;
                    case 2:
                        exp.w2_scale_raw   = std::move(raw);
                        exp.w2_scale_shape = tensor.shape;
                        break;
                    case 3:
                        exp.w3_scale_raw   = std::move(raw);
                        exp.w3_scale_shape = tensor.shape;
                        break;
                    }
                } else {
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
                }

                // Flush only when weights AND their paired E8M0 scales
                // are all present — shard tensor order is arbitrary, and a
                // weights-only trigger strands scales that follow the last
                // weight as a forever-incomplete partial key.
                if (!exp.w1_raw.empty() && !exp.w2_raw.empty() && !exp.w3_raw.empty() &&
                    !exp.w1_scale_raw.empty() && !exp.w2_scale_raw.empty() &&
                    !exp.w3_scale_raw.empty()) {
                    ++experts_in_shard;
                    static bool first_expert = true;
                    if (first_expert) {
                        lsxcommon::log::debug("[diag] Converter: processing first expert "
                                              "(CUDA will be initialized here)");
                        diag::LogArrowSystemState("converter-before-first-expert");
                        first_expert = false;
                    }
                    auto status = [&]() -> arrow::Status {
                        ARROW_ASSIGN_OR_RAISE(auto packed, PackExpertNVFP4(std::move(exp)));
                        ARROW_RETURN_NOT_OK(SaveExpertToArrow(
                            packed, key.layer, key.expert, log_root));
                        return arrow::Status::OK();
                    }();
                    if (!status.ok())
                        lsxcommon::log::error(absl::StrFormat(
                            "  Failed to pack expert layer=%d expert=%d: %s",
                            key.layer, key.expert, status.ToString()));
                    else {
                        ++experts_processed;
                        ++experts_per_layer[parsed->layer];
                    }
                    partial.erase(key);
                }
            } else {
                auto sparsed = parse_shared_expert_name(tensor.name);
                if (sparsed) {
                    auto &sexp = shared_by_layer[sparsed->layer];
                    max_layer = std::max(max_layer, sparsed->layer + 1);
                    if (sparsed->is_scale) {
                        switch (sparsed->weight_id) {
                        case 1:
                            sexp.w1_scale_raw   = std::move(raw);
                            sexp.w1_scale_shape = tensor.shape;
                            break;
                        case 2:
                            sexp.w2_scale_raw   = std::move(raw);
                            sexp.w2_scale_shape = tensor.shape;
                            break;
                        case 3:
                            sexp.w3_scale_raw   = std::move(raw);
                            sexp.w3_scale_shape = tensor.shape;
                            break;
                        }
                    } else {
                        switch (sparsed->weight_id) {
                        case 1:
                            sexp.w1_raw   = std::move(raw);
                            sexp.w1_shape = tensor.shape;
                            sexp.w1_dtype = tensor.dtype;
                            break;
                        case 2:
                            sexp.w2_raw   = std::move(raw);
                            sexp.w2_shape = tensor.shape;
                            sexp.w2_dtype = tensor.dtype;
                            break;
                        case 3:
                            sexp.w3_raw   = std::move(raw);
                            sexp.w3_shape = tensor.shape;
                            sexp.w3_dtype = tensor.dtype;
                            break;
                        }
                    }
                    continue;   // packed separately into shared_experts.arrow
                }

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
        lsxcommon::log::info(absl::StrFormat(
            "  done in %.1fs — %d experts in this shard",
            elapsed / 1000.0, experts_in_shard));
    }

    for (auto &[key, raw] : partial)
        return arrow::Status::Invalid(
            "incomplete expert layer=", key.layer, " expert=", key.expert,
            " (w1=", raw.w1_raw.size(), " w2=", raw.w2_raw.size(),
            " w3=", raw.w3_raw.size(), ")");

    for (auto &[layer, cnt] : experts_per_layer)
        max_experts_per_layer = std::max(max_experts_per_layer, cnt);

    // (6) Geometry cross-check: config-declared vs tensor-derived.
    {
        int embed_vocab = 0, embed_hidden = 0;
        for (auto &shard : model.shards) {
            for (auto &t : shard.tensors) {
                // DeepSeek sources name the embedding "embed.weight"
                // (no "model." prefix); accept both spellings.
                if ((t.name == "model.embed_tokens.weight" ||
                     t.name == "embed.weight") && t.shape.size() >= 2) {
                    embed_vocab = static_cast<int>(t.shape[0]);
                    embed_hidden = static_cast<int>(t.shape[1]);
                }
            }
        }
        std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> geom;
        if (cfg_num_layers > 0)
            geom.push_back({"num_layers", {cfg_num_layers, max_layer}});
        if (cfg_hidden_size > 0)
            geom.push_back({"hidden_size", {cfg_hidden_size, embed_hidden}});
        if (cfg_vocab_size > 0)
            geom.push_back({"vocab_size", {cfg_vocab_size, embed_vocab}});
        if (cfg_experts_per_layer > 0)
            geom.push_back({"experts_per_layer",
                            {cfg_experts_per_layer, max_experts_per_layer}});
        if (!geom.empty())
            ARROW_RETURN_NOT_OK(integrity::CheckGeometryFields(geom));
    }

    // Frozen-passport gate (a family pipeline sets it; the generic converter
    // leaves it unset). Synthetic fixtures opt out through the shared runtime
    // env var — never a compile define.
    if (impl_->expected_geometry_set && !SyntheticGeometryAllowed()) {
        if (cfg_num_layers != impl_->expected_layers ||
            cfg_hidden_size != impl_->expected_hidden ||
            cfg_vocab_size != impl_->expected_vocab ||
            max_experts_per_layer != impl_->expected_experts) {
            return arrow::Status::Invalid(absl::StrFormat(
                "source geometry (layers=%d hidden=%d vocab=%d experts=%d) "
                "does not match the frozen passport (layers=%d hidden=%d "
                "vocab=%d experts=%d)",
                cfg_num_layers, cfg_hidden_size, cfg_vocab_size,
                max_experts_per_layer, impl_->expected_layers,
                impl_->expected_hidden, impl_->expected_vocab,
                impl_->expected_experts));
        }
    }

    lsxcommon::log::info(absl::StrFormat("Saving %u pinned tensors...",
                                         pinned_tensors.size()));
    ARROW_RETURN_NOT_OK(SavePinnedTensors(pinned_tensors, log_root));

    lsxcommon::log::info(absl::StrFormat("Packing %u shared-expert layers (NVFP4)...",
                                         shared_by_layer.size()));
    ARROW_RETURN_NOT_OK(SaveSharedExpertsToArrow(shared_by_layer, log_root));

    // Conversion-time VRAM profile (spec 2026-08-21-vram-profile-design.md).
    VramGeometry vg;
    vg.num_attn_layers = max_layer;
    vg.num_moe_layers = max_layer;
    vg.experts_per_layer = max_experts_per_layer;
    vg.context_length = ReadContextLength(
        (fs::path(impl_->input_path) / "config.json").string());
    vg.nvfp4_scale_block = 32;   // DeepSeek sources ship fp4, 32 elems/scale
    // Per-layer KV compression (DeepSeek-V4 CSA): stored tokens = ctx/ratio.
    {
        std::ifstream cf((fs::path(impl_->input_path) / "config.json"));
        if (cf) {
            std::string cj((std::istreambuf_iterator<char>(cf)),
                           std::istreambuf_iterator<char>());
            simdjson::ondemand::parser cp;
            simdjson::padded_string cpj(cj.data(), cj.size());
            simdjson::ondemand::document cd;
            if (cp.iterate(cpj).get(cd) == simdjson::SUCCESS) {
                auto root = cd.get_object();
                if (root.error() == simdjson::SUCCESS) {
                    if (auto arr = root.value()["compress_ratios"].get_array();
                        arr.error() == simdjson::SUCCESS) {
                        for (auto v : arr.value()) {
                            int64_t x;
                            if (v.get_int64().get(x) == simdjson::SUCCESS)
                                vg.kv_compress_ratios.push_back(x);
                        }
                    }
                }
            }
        }
    }
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            if (vg.kv_dim == 0 && t.shape.size() >= 2 &&
                t.name.find("wkv.weight") != std::string::npos) {
                vg.kv_dim = static_cast<int>(t.shape[0]);
                vg.hidden_dim = static_cast<int>(t.shape[1]);
            }
        }
    }

    std::vector<VramTensorInfo> vram_inventory;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            bool cut = false;
            for (const auto &pfx : impl_->skip_prefixes)
                if (t.name.rfind(pfx, 0) == 0) { cut = true; break; }
            if (cut) continue;
            vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
        }
    }

    VramProfile vram_profile = EstimateVram(
        vram_inventory, {&DsIsNvfp4Tensor, &DsMemClass}, vg);
    lsxcommon::log::info(absl::StrFormat(
        "VRAM profile: min=%s optimal=%s max=%s (ctx=%d)",
        FormatBytesHuman(vram_profile.min_bytes),
        FormatBytesHuman(vram_profile.optimal_bytes),
        FormatBytesHuman(vram_profile.max_bytes),
        vram_profile.context_tokens));

    const std::string model_name = impl_->model_name.empty()
        ? fs::path(impl_->input_path).filename().string()
        : impl_->model_name;
    ARROW_RETURN_NOT_OK(WriteModelIndex(
        model_name,
        max_layer, max_experts_per_layer,
        static_cast<int64_t>(pinned_tensors.size()),
        static_cast<int64_t>(experts_processed),
        vram_profile,
        deferred,
        log_root));

    // (8) Build and verify the binary tokenizer (mandatory).
    auto tokenizer_json = fs::path(impl_->input_path) / "tokenizer.json";
    if (!fs::exists(tokenizer_json))
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      impl_->input_path);
    ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
        tokenizer_json.string(),
        log_root / "pinned",
        integrity::TokenizerSpecialIds{.bos_required = true,
                                       .eos_required = true}));

    lsxcommon::log::info(absl::StrFormat(
        "Done — %d pinned tensors, %d experts processed, %d deferred "
        "ballast tensors, incomplete %u",
        pinned_processed, experts_processed, deferred_tensors,
        partial.size()));
    return arrow::Status::OK();
}

} // namespace lsxcommon
