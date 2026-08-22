#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {

// One scanned safetensors tensor (pre-conversion inventory entry).
struct VramTensorInfo {
    std::string name;           // safetensors tensor name
    std::int64_t numel = 0;     // element count
    std::int64_t raw_bytes = 0; // original size: dtype_size * numel
};

// Memory residency class of a tensor in the converted model.
enum class VramTensorClass { Resident, RoutedExpert };

// Family-specific classification rules (supplied by each converter).
// Function pointers (not std::function) to keep the header NVCC-safe.
struct VramTensorPolicy {
    bool (*is_nvfp4)(const std::string &name) = nullptr;
    VramTensorClass (*mem_class)(const std::string &name) = nullptr;
};

// Model geometry needed by the estimate (derived from config.json + shapes).
struct VramGeometry {
    int hidden_dim = 0;
    int kv_dim = 0;             // KV latent dim (attention value width)
    int num_attn_layers = 0;
    int num_moe_layers = 0;
    int experts_per_layer = 0;
    std::int64_t context_length = 0; // from source config.json
    // Per-layer KV token-count divisors (e.g. DeepSeek-V4 "compress_ratios":
    // a layer storing ctx/ratio cached tokens; 0 = full-context storage).
    // Empty -> every attention layer stores the full context.
    std::vector<std::int64_t> kv_compress_ratios;
    // Elements per E4M3 scale byte in the NVFP4 layout. 16 = canonical NVFP4
    // (GLM/Qwen packers); DeepSeek sources ship fp4 with 32-element blocks.
    std::int64_t nvfp4_scale_block = 16;
};

// Three VRAM parameters computed at conversion time.
struct VramProfile {
    std::uint64_t min_bytes = 0;     // device-resident weights only
    std::uint64_t optimal_bytes = 0; // min + KV cache + workspace
    std::uint64_t max_bytes = 0;     // optimal + full routed-expert pool
    std::int64_t context_tokens = 0; // seq_len used in the estimate
};

extern const VramProfile kEmptyVramProfile;

// Computes the conversion-time VRAM profile. NVFP4 tensor size:
// numel/2 (packed E2M1 nibbles) + numel/16 (E4M3 per-block scales);
// all other tensors keep raw_bytes (norms/biases precision invariant).
VramProfile EstimateVram(const std::vector<VramTensorInfo> &tensors,
                         const VramTensorPolicy &policy,
                         const VramGeometry &geom);

// Reads the model context length from config.json. Fallback chain:
// max_position_embeddings -> context_length -> seq_length -> 2048
// (root first, then nested "text_config"). Missing fields: warning + 2048.
std::int64_t ReadContextLength(const std::string &config_json_path);

// Renders the mandatory "vram" JSON block (no trailing comma, 2-space base
// indent) for insertion into model.index.json by any index writer.
std::string FormatVramJson(const VramProfile &p);

// Strictly parses the "vram" object out of a model.index.json document.
// Returns false and fills err when the block is missing/malformed/inconsistent.
bool ParseVramBlock(const std::string &index_json_text,
                    VramProfile &out, std::string &err);

// Returns index_json_text with the "vram" block replaced by the rendered
// profile. When the document has no block yet, it is inserted before the
// final closing brace (with the preceding line terminated by a comma).
std::string ReplaceVramBlock(const std::string &index_json_text,
                             const VramProfile &p);

// Human-readable byte size ("123 GiB" / "45 MiB" / ...).
std::string FormatBytesHuman(std::uint64_t bytes);

} // namespace lsxcommon
