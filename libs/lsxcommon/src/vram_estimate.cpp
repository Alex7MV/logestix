#include "lsxcommon/vram_estimate.h"

#include <spdlog/spdlog.h>
#include <simdjson.h>

#include <fstream>
#include <iterator>

namespace lsxcommon {

const VramProfile kEmptyVramProfile{};

namespace {

constexpr std::int64_t kDefaultContextTokens = 2048;

// NVFP4 converted size: packed E2M1 nibbles + E4M3 scales (block of 16).
std::int64_t Nvfp4ConvertedBytes(std::int64_t numel)
{
    return numel / 2 + numel / 16;
}

} // namespace

VramProfile EstimateVram(const std::vector<VramTensorInfo> &tensors,
                         const VramTensorPolicy &policy,
                         const VramGeometry &geom)
{
    VramProfile p;
    p.context_tokens = geom.context_length > 0
        ? geom.context_length : kDefaultContextTokens;

    if (tensors.empty())
        return p;   // zeroed weights: nothing to place in VRAM

    std::uint64_t resident = 0;
    std::uint64_t expert_pool = 0;

    for (const auto &t : tensors) {
        const bool quantized =
            policy.is_nvfp4 && policy.is_nvfp4(t.name);
        const std::uint64_t bytes = quantized
            ? static_cast<std::uint64_t>(Nvfp4ConvertedBytes(t.numel))
            : static_cast<std::uint64_t>(t.raw_bytes);
        const bool routed = policy.mem_class &&
            policy.mem_class(t.name) == VramTensorClass::RoutedExpert;
        if (routed)
            expert_pool += bytes;
        else
            resident += bytes;
    }

    const std::int64_t ctx = p.context_tokens;
    const std::int64_t attn =
        static_cast<std::int64_t>(geom.num_attn_layers > 0 ? geom.num_attn_layers : 0);
    const std::int64_t kv_dim =
        geom.kv_dim > 0 ? geom.kv_dim : 512;
    const std::int64_t hidden =
        geom.hidden_dim > 0 ? geom.hidden_dim : 4096;

    // Same formulas as print_model() (converter.cpp) but with the
    // config-declared context length instead of hardcoded 2048.
    const std::int64_t kv_cache = ctx * attn * kv_dim * 2 * 2;      // K+V, FP16
    const std::int64_t workspace = ctx * hidden * attn * 2 * 2;     // FP16

    p.min_bytes = resident;
    p.optimal_bytes =
        resident + static_cast<std::uint64_t>(kv_cache + workspace);
    p.max_bytes = p.optimal_bytes + expert_pool;
    return p;
}

std::int64_t ReadContextLength(const std::string &config_json_path)
{
    namespace sj = simdjson;
    std::ifstream f(config_json_path);
    if (!f) {
        spdlog::warn("[vram] cannot open {}: defaulting context to {}",
                     config_json_path, kDefaultContextTokens);
        return kDefaultContextTokens;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    sj::ondemand::parser parser;
    sj::padded_string pad(json.data(), json.size());
    sj::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != sj::SUCCESS) {
        spdlog::warn("[vram] {} parse failed: defaulting context to {}",
                     config_json_path, kDefaultContextTokens);
        return kDefaultContextTokens;
    }
    sj::ondemand::object root;
    if (doc.get_object().get(root) != sj::SUCCESS)
        return kDefaultContextTokens;

    auto find_ctx = [](sj::ondemand::object &o) -> std::int64_t {
        for (const char *k : {"max_position_embeddings", "context_length",
                              "seq_length"}) {
            int64_t v;
            if (o[k].get_int64().get(v) == sj::SUCCESS && v > 0) return v;
        }
        return 0;
    };

    if (int64_t v = find_ctx(root); v > 0) return v;
    // Qwen3.5 nests the language-model config under "text_config".
    if (auto tc = root["text_config"].get_object();
        tc.error() == sj::SUCCESS) {
        sj::ondemand::object src = tc.value();
        if (int64_t v = find_ctx(src); v > 0) return v;
    }
    spdlog::warn("[vram] no context field in {}: defaulting to {}",
                 config_json_path, kDefaultContextTokens);
    return kDefaultContextTokens;
}

std::string FormatVramJson(const VramProfile &p)
{
    std::string s = "  \"vram\": {\n";
    s += "    \"min_bytes\": " + std::to_string(p.min_bytes) + ",\n";
    s += "    \"optimal_bytes\": " + std::to_string(p.optimal_bytes) + ",\n";
    s += "    \"max_bytes\": " + std::to_string(p.max_bytes) + ",\n";
    s += "    \"context_tokens\": " + std::to_string(p.context_tokens) + "\n";
    s += "  }";
    return s;
}

bool ParseVramBlock(const std::string &index_json_text,
                    VramProfile &out, std::string &err)
{
    namespace sj = simdjson;
    out = {};
    sj::ondemand::parser parser;
    sj::padded_string pad(index_json_text.data(), index_json_text.size());
    sj::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != sj::SUCCESS) {
        err = "JSON parse error";
        return false;
    }
    sj::ondemand::object root;
    if (doc.get_object().get(root) != sj::SUCCESS) {
        err = "expected JSON object";
        return false;
    }
    sj::ondemand::object vram;
    if (root["vram"].get_object().get(vram) != sj::SUCCESS) {
        err = "missing 'vram' block";
        return false;
    }
    int64_t mn = 0, op = 0, mx = 0, ct = 0;
    if (vram["min_bytes"].get_int64().get(mn) != sj::SUCCESS ||
        vram["optimal_bytes"].get_int64().get(op) != sj::SUCCESS ||
        vram["max_bytes"].get_int64().get(mx) != sj::SUCCESS ||
        vram["context_tokens"].get_int64().get(ct) != sj::SUCCESS) {
        err = "malformed 'vram' block";
        return false;
    }
    if (mn < 0 || op < mn || mx < op || ct <= 0) {
        err = "inconsistent 'vram' values";
        return false;
    }
    out.min_bytes = static_cast<std::uint64_t>(mn);
    out.optimal_bytes = static_cast<std::uint64_t>(op);
    out.max_bytes = static_cast<std::uint64_t>(mx);
    out.context_tokens = ct;
    return true;
}

std::string FormatBytesHuman(std::uint64_t bytes)
{
    if (bytes >= 1024ull * 1024 * 1024)
        return std::to_string(bytes / (1024ull * 1024 * 1024)) + " GiB";
    if (bytes >= 1024ull * 1024)
        return std::to_string(bytes / (1024ull * 1024)) + " MiB";
    if (bytes >= 1024ull)
        return std::to_string(bytes / 1024ull) + " KiB";
    return std::to_string(bytes) + " B";
}

} // namespace lsxcommon
