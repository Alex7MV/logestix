#include "lsxcommon/vram_estimate.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <fstream>
#include <iterator>
#include <regex>

namespace lsxcommon {

const VramProfile kEmptyVramProfile{};

namespace {

constexpr std::int64_t kDefaultContextTokens = 2048;

// Transient activations never span the whole context: decode is one token,
// prefill runs in bounded chunks. Workspace is estimated from this chunk,
// while the KV cache honestly scales with the full context length.
constexpr std::int64_t kPrefillChunkTokens = 2048;

// NVFP4 converted size: packed E2M1 nibbles + E4M3 scales.
std::int64_t Nvfp4ConvertedBytes(std::int64_t numel, std::int64_t scale_block)
{
    const std::int64_t blk = scale_block > 0 ? scale_block : 16;
    return numel / 2 + (numel + blk - 1) / blk;
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
            ? static_cast<std::uint64_t>(
                  Nvfp4ConvertedBytes(t.numel, geom.nvfp4_scale_block))
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

    // KV cache scales with the config-declared context length; per-layer
    // compress_ratios shrink the stored token count where the architecture
    // uses compressed attention (DeepSeek-V4 CSA). The transient workspace is
    // computed from a bounded prefill chunk (full-context workspace would
    // scale absurdly: ~739 GiB at a 1M-token context).
    std::int64_t total_kv_tokens = 0;
    if (!geom.kv_compress_ratios.empty()) {
        for (const std::int64_t r : geom.kv_compress_ratios) {
            const std::int64_t div = r > 1 ? r : 1;
            total_kv_tokens += (ctx + div - 1) / div;
        }
    } else {
        total_kv_tokens = ctx * attn;
    }
    const std::int64_t kv_cache = total_kv_tokens * kv_dim * 2 * 2;  // K+V, FP16
    const std::int64_t workspace =
        kPrefillChunkTokens * hidden * attn * 2 * 2;                // FP16

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
        lsxcommon::log::warn(absl::StrFormat(
            "[vram] cannot open %s: defaulting context to %d",
            config_json_path, kDefaultContextTokens));
        return kDefaultContextTokens;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    sj::ondemand::parser parser;
    sj::padded_string pad(json.data(), json.size());
    sj::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != sj::SUCCESS) {
        lsxcommon::log::warn(absl::StrFormat(
            "[vram] %s parse failed: defaulting context to %d",
            config_json_path, kDefaultContextTokens));
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
    lsxcommon::log::warn(absl::StrFormat("[vram] no context field in %s: defaulting to %d",
                                         config_json_path, kDefaultContextTokens));
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

std::string ReplaceVramBlock(const std::string &index_json_text,
                             const VramProfile &p)
{
    // The block is machine-rendered by FormatVramJson (no nested braces),
    // so a non-greedy brace scan is safe.
    static const std::regex re(R"("vram"\s*:\s*\{[^{}]*\})");
    const std::string block = FormatVramJson(p);

    std::string out = std::regex_replace(index_json_text, re, block);
    if (out == index_json_text) {
        // No existing block: insert before the final closing brace.
        const auto pos = index_json_text.rfind('}');
        if (pos == std::string::npos) return index_json_text;
        std::string head = index_json_text.substr(0, pos);
        // Ensure the preceding content line ends with a comma.
        while (!head.empty() && (head.back() == ' ' || head.back() == '\t' ||
                                 head.back() == '\r' || head.back() == '\n'))
            head.pop_back();
        if (!head.empty() && head.back() != ',') head += ',';
        out = head + "\n" + block + "\n}" +
              index_json_text.substr(pos + 1);
    }
    return out;
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
