// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 converter
// (Model Ingestion Protocol stages 1-2).
//
// Same architecture and same frozen geometry as the BF16-distribution sibling
// `nemotron-3.5-lightning-30b-a3b`; only the SOURCE ENCODING differs. The vault
// checkpoint is already quantized by ModelOpt 0.44.0rc5 — W4A16_NVFP4 routed /
// shared experts and lm_head, per-tensor FP8 Mamba in/out_proj, BF16 attention
// projections carrying FP8 KV-cache scales, F32 router — so this converter
// exists to undo that encoding, hand BF16 to the repo's NEUTRAL
// `quantize_tensor_gpu`, and emit a layout byte-compatible with the sibling's.
// The sibling's loader, expert cache, kernels, model pipeline and tool codec are
// then reused verbatim.
//
// Why a round trip and not a passthrough: the source block scales use the
// COMPLETE OCP E4M3 (measured on the real vault: all 8 mantissa values present)
// while the repo's runtime decoder reads the e4 exponent field only. Passing the
// bytes through decodes a median 0.73x / max 1.00x per-block scale error, i.e.
// ~30% relative L2 on the expert weights. See
// docs/superpowers/specs/2026-10-06-nemotron-3.5-lightning-nvfp4-ingestion-design.md
// section 0.6 for the measurements.
#include "lsxcommon/nemotron_moe_nvfp4_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/synthetic_geometry.h"
#include "nemotron_moe_constants.h"
#include "nemotron_moe_nvfp4_dequant.h"

#include <absl/container/btree_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace lsxcommon {
namespace nemotron_nvfp4 {
namespace {

using nemotron_moe::NemotronTensorKind;

bool check_frozen_geometry(const NemotronTopology &top, std::string &why)
{
    using namespace nemotron_moe::frozen;
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.num_layers != kLayers)
        return bad("num_hidden_layers", top.num_layers, kLayers);
    if (top.hidden_size != kHidden)
        return bad("hidden_size", top.hidden_size, kHidden);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.mamba_num_heads != kMambaHeads)
        return bad("mamba_num_heads", top.mamba_num_heads, kMambaHeads);
    if (top.mamba_head_dim != kMambaHeadDim)
        return bad("mamba_head_dim", top.mamba_head_dim, kMambaHeadDim);
    if (top.ssm_state_size != kSsmStateSize)
        return bad("ssm_state_size", top.ssm_state_size, kSsmStateSize);
    if (top.n_groups != kSsmGroups) return bad("n_groups", top.n_groups, kSsmGroups);
    if (top.conv_kernel != kMambaConvKernel)
        return bad("conv_kernel", top.conv_kernel, kMambaConvKernel);
    if (top.num_attention_heads != kHeads)
        return bad("num_attention_heads", top.num_attention_heads, kHeads);
    if (top.num_key_value_heads != kKVHeads)
        return bad("num_key_value_heads", top.num_key_value_heads, kKVHeads);
    if (top.head_dim != kHeadDim) return bad("head_dim", top.head_dim, kHeadDim);
    if (top.num_experts != kNumRouted)
        return bad("n_routed_experts", top.num_experts, kNumRouted);
    if (top.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size,
                   kMoeIntermediate);
    if (top.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("moe_shared_expert_intermediate_size",
                   top.shared_expert_intermediate_size, kSharedIntermediate);

    int mamba = 0, moe = 0, attn = 0;
    for (const auto &t : top.layer_types) {
        if (t == "mamba") ++mamba;
        else if (t == "moe") ++moe;
        else if (t == "attention") ++attn;
        else { why = "unknown layer type: " + t; return false; }
    }
    if (mamba != kMambaLayers || moe != kMoeLayers || attn != kAttnLayers) {
        why = absl::StrFormat(
            "layer mix got mamba=%d moe=%d attn=%d, want %d/%d/%d", mamba, moe,
            attn, kMambaLayers, kMoeLayers, kAttnLayers);
        return false;
    }
    return true;
}

// Source kinds whose weight arrives as ModelOpt W4A16_NVFP4.
bool base_is_nvfp4_source(NemotronTensorKind k)
{
    switch (k) {
    case NemotronTensorKind::LmHead:
    case NemotronTensorKind::AttnQ:
    case NemotronTensorKind::AttnK:
    case NemotronTensorKind::AttnV:
    case NemotronTensorKind::AttnO:
    case NemotronTensorKind::SharedUpProj:
    case NemotronTensorKind::SharedDownProj:
    case NemotronTensorKind::ExpertUpProj:
    case NemotronTensorKind::ExpertDownProj:
        return true;
    default:
        return false;
    }
}

// ... and the two whose weight arrives as per-tensor FP8.
bool base_is_fp8_source(NemotronTensorKind k)
{
    return k == NemotronTensorKind::MambaInProj ||
           k == NemotronTensorKind::MambaOutProj;
}

// How one source weight arrives, resolved from the OBSERVED dtype rather than
// from the tensor name. The name only says which module a tensor belongs to;
// whether ModelOpt quantized it is a property of the file. Concretely, in the
// real vault the attention q/k/v/o projections are BF16 (excluded from
// quantization) while the routed experts are U8-packed NVFP4 - so a
// name-driven dispatch would try to read block scales off BF16 weights.
enum class SourceLayout {
    kBf16,     // unquantized: hand straight to the repo quantizer
    kNvfp4,    // U8 packed E2M1 + full-E4M3 block scale + F32 weight_scale_2
    kFp8,      // F8_E4M3 + scalar F32 weight_scale (per-tensor)
};

SourceLayout layout_of(const TensorInfo &t, std::string &error)
{
    if (t.dtype == "U8") return SourceLayout::kNvfp4;
    if (t.dtype == "F8_E4M3") return SourceLayout::kFp8;
    if (t.dtype == "BF16" || t.dtype == "F16") return SourceLayout::kBf16;
    error = absl::StrFormat("unsupported source dtype '", t.dtype,
                            "' for tensor ", t.name);
    return SourceLayout::kBf16;
}

// The LOGICAL [rows, cols] shape: an NVFP4 source stores its columns packed
// two-per-byte, so the logical width is twice the stored one.
std::vector<int64_t> logical_shape_of(const TensorInfo &t, SourceLayout layout,
                                      std::string &error)
{
    if (layout != SourceLayout::kNvfp4) return t.shape;
    if (t.shape.size() != 2) {
        error = absl::StrCat("expected a 2-D NVFP4 weight: ", t.name);
        return {};
    }
    return {t.shape[0], t.shape[1] * 2};
}

NemotronNvfp4Kind mirror_kind(NemotronTensorKind k)
{
    using K = NemotronNvfp4Kind;
    switch (k) {
    case NemotronTensorKind::Embed: return K::Embed;
    case NemotronTensorKind::LmHead: return K::LmHead;
    case NemotronTensorKind::FinalNorm: return K::FinalNorm;
    case NemotronTensorKind::LayerNorm: return K::LayerNorm;
    case NemotronTensorKind::MambaInProj: return K::MambaInProj;
    case NemotronTensorKind::MambaOutProj: return K::MambaOutProj;
    case NemotronTensorKind::MambaConv: return K::MambaConv;
    case NemotronTensorKind::MambaConvBias: return K::MambaConvBias;
    case NemotronTensorKind::MambaALog: return K::MambaALog;
    case NemotronTensorKind::MambaD: return K::MambaD;
    case NemotronTensorKind::MambaDtBias: return K::MambaDtBias;
    case NemotronTensorKind::MambaNorm: return K::MambaNorm;
    case NemotronTensorKind::AttnQ: return K::AttnQ;
    case NemotronTensorKind::AttnK: return K::AttnK;
    case NemotronTensorKind::AttnV: return K::AttnV;
    case NemotronTensorKind::AttnO: return K::AttnO;
    case NemotronTensorKind::RouterGate: return K::RouterGate;
    case NemotronTensorKind::RouterBias: return K::RouterBias;
    case NemotronTensorKind::SharedUpProj: return K::SharedUpProj;
    case NemotronTensorKind::SharedDownProj: return K::SharedDownProj;
    case NemotronTensorKind::ExpertUpProj: return K::ExpertUpProj;
    case NemotronTensorKind::ExpertDownProj: return K::ExpertDownProj;
    case NemotronTensorKind::Mtp: return K::Mtp;
    default: return K::Unknown;
    }
}

constexpr std::string_view kSfxBlockScale = ".weight_scale";
constexpr std::string_view kSfxGlobalScale = ".weight_scale_2";
constexpr std::string_view kSfxInputScale = ".input_scale";
constexpr std::string_view kSfxKScale = ".k_scale";
constexpr std::string_view kSfxVScale = ".v_scale";

bool strip_suffix(std::string_view name, std::string_view sfx,
                  std::string_view &base)
{
    if (name.size() > sfx.size() &&
        name.compare(name.size() - sfx.size(), sfx.size(), sfx) == 0) {
        base = name.substr(0, name.size() - sfx.size());
        return true;
    }
    return false;
}

// A weight plus its resolved ModelOpt scale companions. A scale whose base
// weight is absent is a hard error (a leftover tensor must fail the
// conversion, never pass).
struct SourceTensor {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
    const TensorInfo *nvfp4_block = nullptr;  // F8_E4M3 [R, C/16]
    const TensorInfo *nvfp4_global = nullptr;  // F32 scalar
    const TensorInfo *fp8_weight = nullptr;    // F32 scalar
    bool has_fp8_input = false;                 // F32 scalar, IGNORED
};

// One reusable device BF16 staging buffer: the walk touches ~5900 tensors and
// must never hold more than one on the device at a time.
class Bf16Stager {
public:
    Bf16Stager() = default;
    Bf16Stager(const Bf16Stager &) = delete;
    Bf16Stager &operator=(const Bf16Stager &) = delete;
    ~Bf16Stager()
    {
        if (d_buf_) cudaFree(d_buf_);
    }

    bool Ensure(size_t need, std::string &error)
    {
        if (need <= bytes_) return true;
        if (d_buf_) cudaFree(d_buf_);
        d_buf_ = nullptr;
        bytes_ = 0;
        if (cudaMalloc(&d_buf_, need) != cudaSuccess) {
            error = "cudaMalloc failed for the BF16 staging buffer";
            return false;
        }
        bytes_ = need;
        return true;
    }

    void *device() const { return d_buf_; }
    std::vector<uint8_t> &host() { return host_; }

private:
    void *d_buf_ = nullptr;
    size_t bytes_ = 0;
    std::vector<uint8_t> host_;
};

// Dequantize one pre-quantized source weight into `st` as BF16.
// `logical` is the LOGICAL [rows, cols] shape (the source stores the columns
// packed two-per-byte).
arrow::Status dequant_source(const SourceTensor &src,
                             const std::filesystem::path &input_dir,
                             const std::vector<int64_t> &logical,
                             SourceLayout layout, Bf16Stager &st,
                             std::string &error)
{
    const int64_t rows = logical.at(0);
    const int64_t cols = logical.at(1);
    const int64_t numel = rows * cols;
    if (!st.Ensure(static_cast<size_t>(numel) * 2, error))
        return arrow::Status::IOError(error);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return arrow::Status::IOError("cannot create the dequant stream");
    struct StreamGuard {
        cudaStream_t s;
        ~StreamGuard()
        {
            if (s) cudaStreamDestroy(s);
        }
    } guard{stream};

    if (layout == SourceLayout::kNvfp4) {
        if (!src.nvfp4_block || !src.nvfp4_global) {
            error = absl::StrFormat(
                "NVFP4 weight %s is missing weight_scale (block=%p) / "
                "weight_scale_2 (global=%p)",
                src.info->name.c_str(), static_cast<const void *>(src.nvfp4_block),
                static_cast<const void *>(src.nvfp4_global));
            return arrow::Status::Invalid(error);
        }
        auto packed = ReadTensorData(input_dir, *src.shard, *src.info);
        auto bscale = ReadTensorData(input_dir, *src.shard, *src.nvfp4_block);
        auto graw = ReadTensorData(input_dir, *src.shard, *src.nvfp4_global);
        if (packed.size() != static_cast<size_t>((numel + 1) / 2) ||
            bscale.size() != static_cast<size_t>((numel + 15) / 16) ||
            graw.size() < sizeof(float)) {
            error = absl::StrFormat(
                "NVFP4 source tensor %s has inconsistent companion sizes",
                src.info->name);
            return arrow::Status::Invalid(error);
        }
        float global = 1.0f;
        std::memcpy(&global, graw.data(), sizeof(float));
        Nvfp4SourceView view{packed.data(), bscale.data(), global, rows, cols};
        if (!Nvfp4SourceToBf16Gpu(view, st.device(), stream, error))
            return arrow::Status::IOError(error);
    } else {
        if (!src.fp8_weight) {
            error = absl::StrFormat(
                "FP8 weight %s is missing its per-tensor weight_scale",
                src.info->name);
            return arrow::Status::Invalid(error);
        }
        auto w = ReadTensorData(input_dir, *src.shard, *src.info);
        auto sraw = ReadTensorData(input_dir, *src.shard, *src.fp8_weight);
        if (w.size() != static_cast<size_t>(numel) || sraw.size() < sizeof(float)) {
            error = absl::StrFormat(
                "FP8 source tensor %s has inconsistent companion sizes "
                "(w=%zu want=%lld s=%zu shape=[%lld,%lld] dtype=%s)",
                src.info->name, w.size(), static_cast<long long>(numel),
                sraw.size(), static_cast<long long>(rows),
                static_cast<long long>(cols), src.info->dtype.c_str());
            return arrow::Status::Invalid(error);
        }
        float scale = 1.0f;
        std::memcpy(&scale, sraw.data(), sizeof(float));
        // The kernel reads the scalar through a device pointer, so stage it.
        Fp8SourceView view{w.data(), &scale, rows, cols};
        std::vector<uint8_t> bf16;
        std::string host_err;
        Fp8SourceToBf16Host(view, bf16, host_err);
        if (!host_err.empty()) return arrow::Status::Invalid(host_err);
        st.host() = std::move(bf16);
        if (cudaMemcpy(st.device(), st.host().data(), st.host().size(),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return arrow::Status::IOError("FP8 staging H2D copy failed");
        return arrow::Status::OK();
    }

    st.host().resize(static_cast<size_t>(numel) * 2);
    if (cudaMemcpy(st.host().data(), st.device(), st.host().size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return arrow::Status::IOError("dequant D2H copy failed");
    return arrow::Status::OK();
}

bool quantize_bf16(const std::vector<uint8_t> &bf16, int64_t numel,
                   const std::vector<int64_t> &logical, DenseNvfp4Record &r,
                   std::string &error)
{
    if (!gpu::quantize_tensor_gpu(bf16.data(), numel, "BF16", nullptr, logical,
                                  {}, r.packed, r.scales, r.global_scale)) {
        error = "GPU NVFP4 quantization failed for " + r.name;
        return false;
    }
    return true;
}

arrow::Status write_expert_file(const std::filesystem::path &base_dir, int layer,
                                int expert, PackedExpert &&pe)
{
    namespace fs = std::filesystem;
    const fs::path dir = base_dir / "experts" / absl::StrCat("layer_", layer);
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto batch = ExpertSchema::MakeBatch(pe);
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(
        *batch, dir / absl::StrCat("exp_", expert, ".arrow"));
}

arrow::Status write_batch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

} // namespace

ParsedNemotronNvfp4Tensor classify_nemotron_nvfp4_tensor(
    const std::string &name)
{
    using K = NemotronNvfp4Kind;
    ParsedNemotronNvfp4Tensor r;
    const std::string_view n(name);

    if (nemotron_moe::classify_nemotron_tensor(name).kind ==
        NemotronTensorKind::Mtp) {
        r.kind = K::Mtp;
        return r;
    }

    std::string_view base;
    // ModelOpt names every companion scale after the MODULE, not after the
    // weight tensor: `lm_head.weight_scale`, `...down_proj.weight_scale`,
    // `...in_proj.weight_scale`, `...in_proj.input_scale`, `...k_proj.k_scale`.
    // So the stripped base is a module path and ".weight" has to be re-attached
    // before the sibling classifier (which owns the weight-name grammar) can
    // be consulted. Verified against the real vault inventory.
    std::string_view module;
    if (strip_suffix(n, kSfxKScale, module) ||
        strip_suffix(n, kSfxVScale, module)) {
        // The FP8 KV-cache scale. NOT a weight dequant factor: q/k/v/o are BF16
        // in this checkpoint and the converted layout keeps an FP16 KV cache,
        // so these are counted and dropped.
        const auto p = nemotron_moe::classify_nemotron_tensor(
            absl::StrCat(module, ".weight"));
        if (p.kind == NemotronTensorKind::AttnK || p.kind == NemotronTensorKind::AttnV) {
            r.kind = K::Fp8KvScale;
            r.layer = p.layer;
            r.base_weight = absl::StrCat(module, ".weight");
        }
        return r;
    }
    if (strip_suffix(n, kSfxGlobalScale, module)) {
        const auto p = nemotron_moe::classify_nemotron_tensor(
            absl::StrCat(module, ".weight"));
        if (base_is_nvfp4_source(p.kind)) {
            r.kind = K::Nvfp4GlobalScale;
            r.layer = p.layer;
            r.expert = p.expert;
            r.base_weight = absl::StrCat(module, ".weight");
        }
        return r;
    }
    if (strip_suffix(n, kSfxInputScale, module)) {
        const auto p = nemotron_moe::classify_nemotron_tensor(
            absl::StrCat(module, ".weight"));
        if (base_is_fp8_source(p.kind)) {
            r.kind = K::Fp8InputScale;
            r.layer = p.layer;
            r.base_weight = absl::StrCat(module, ".weight");
        }
        return r;
    }
    if (strip_suffix(n, kSfxBlockScale, module)) {
        const auto p = nemotron_moe::classify_nemotron_tensor(
            absl::StrCat(module, ".weight"));
        if (base_is_nvfp4_source(p.kind)) {
            r.kind = K::Nvfp4BlockScale;
            r.layer = p.layer;
            r.expert = p.expert;
            r.base_weight = absl::StrCat(module, ".weight");
        } else if (base_is_fp8_source(p.kind)) {
            r.kind = K::Fp8WeightScale;
            r.layer = p.layer;
            r.base_weight = absl::StrCat(module, ".weight");
        }
        return r;
    }

    // Plain weight: the sibling classifier owns the name grammar, so the two
    // families cannot drift apart.
    const auto p = nemotron_moe::classify_nemotron_tensor(name);
    r.kind = mirror_kind(p.kind);
    r.layer = p.layer;
    r.expert = p.expert;
    return r;
}

bool nemotron_nvfp4_weight_is_nvfp4(NemotronNvfp4Kind kind)
{
    switch (kind) {
    case NemotronNvfp4Kind::MambaInProj:
    case NemotronNvfp4Kind::MambaOutProj:
    case NemotronNvfp4Kind::AttnQ:
    case NemotronNvfp4Kind::AttnK:
    case NemotronNvfp4Kind::AttnV:
    case NemotronNvfp4Kind::AttnO:
    case NemotronNvfp4Kind::SharedUpProj:
    case NemotronNvfp4Kind::SharedDownProj:
    case NemotronNvfp4Kind::ExpertUpProj:
    case NemotronNvfp4Kind::ExpertDownProj:
        return true;
    default:
        return false;
    }
}

std::vector<int32_t> ParseNemotronEosIds(const std::string &input_dir)
{
    std::vector<int32_t> out;
    namespace fs = std::filesystem;
    std::ifstream f(fs::path(input_dir) / "generation_config.json",
                    std::ios::binary);
    if (!f) return out;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return out;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return out;
    simdjson::ondemand::value v;
    if (root["eos_token_id"].get(v) != simdjson::SUCCESS) return out;
    if (auto arr = v.get_array(); arr.error() == simdjson::SUCCESS) {
        for (auto e : arr.value()) {
            int64_t x;
            if (e.get_int64().get(x) == simdjson::SUCCESS)
                out.push_back(static_cast<int32_t>(x));
        }
        return out;
    }
    int64_t x;
    if (v.get_int64().get(x) == simdjson::SUCCESS)
        out.push_back(static_cast<int32_t>(x));
    return out;
}

arrow::Result<PackedExpert> ReadDenseNvfp4Records(const std::string &path)
{
    auto batch = ReadRecordBatchFromFile(path);
    if (!batch.ok()) return batch.status();
    return ExpertSchema::UnpackBatch(*batch, kNVFP4BlockSize);
}

arrow::Status ConvertNemotronNvfp4Model(const std::string &input_dir,
                                        const std::string &output_root)
{
    namespace fs = std::filesystem;

    NemotronTopology top = ParseNemotronConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Nemotron config error: ", top.error_msg);
    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > n_routed_experts");
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!check_frozen_geometry(top, why))
            return arrow::Status::Invalid("Nemotron frozen geometry mismatch: ",
                                          why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // ── Inventory + scale pairing ─────────────────────────────────────────
    // Every scale tensor must find its base weight; an orphan is fatal.
    std::map<std::string, SourceTensor> weights;
    struct ScaleEntry {
        ParsedNemotronNvfp4Tensor parsed;
        const SafetensorsModel::Shard *shard = nullptr;
        const TensorInfo *info = nullptr;
    };
    std::vector<ScaleEntry> scales;
    int mtp_cut = 0;
    int fp8_kv_dropped = 0;
    absl::btree_map<int, int> layer_tensor_count;

    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_nemotron_nvfp4_tensor(t.name);
            switch (p.kind) {
            case NemotronNvfp4Kind::Mtp:
                ++mtp_cut;
                continue;
            case NemotronNvfp4Kind::Fp8KvScale:
                ++fp8_kv_dropped;
                continue;
            case NemotronNvfp4Kind::Nvfp4BlockScale:
            case NemotronNvfp4Kind::Nvfp4GlobalScale:
            case NemotronNvfp4Kind::Fp8WeightScale:
            case NemotronNvfp4Kind::Fp8InputScale:
                scales.push_back(ScaleEntry{p, &shard, &t});
                continue;
            case NemotronNvfp4Kind::Unknown:
                return arrow::Status::Invalid(
                    "unrecognized tensor in the pre-quantized nemotron_h "
                    "checkpoint: ",
                    t.name);
            default:
                break;
            }
            if ((p.kind == NemotronNvfp4Kind::ExpertUpProj ||
                 p.kind == NemotronNvfp4Kind::ExpertDownProj) &&
                (p.expert < 0 || p.expert >= top.num_experts))
                return arrow::Status::Invalid(
                    absl::StrCat("routed expert index out of range: ", t.name,
                                 " (n_routed_experts=", top.num_experts, ")"));
            SourceTensor st;
            st.shard = &shard;
            st.info = &t;
            weights[t.name] = std::move(st);
            if (p.layer >= 0) ++layer_tensor_count[p.layer];
        }
    }
    for (const auto &s : scales) {
        auto it = weights.find(s.parsed.base_weight);
        if (it == weights.end())
            return arrow::Status::Invalid(
                absl::StrCat("quant scale tensor '", s.info->name,
                             "' has no matching weight '",
                             s.parsed.base_weight, "'"));
        switch (s.parsed.kind) {
        case NemotronNvfp4Kind::Nvfp4BlockScale:
            it->second.nvfp4_block = s.info;
            break;
        case NemotronNvfp4Kind::Nvfp4GlobalScale:
            it->second.nvfp4_global = s.info;
            break;
        case NemotronNvfp4Kind::Fp8WeightScale:
            it->second.fp8_weight = s.info;
            break;
        case NemotronNvfp4Kind::Fp8InputScale:
            // The activation scale belongs to the FP8 input quantizer, which
            // this repo does not run (activations stay FP16). Recorded so the
            // tensor is accounted for, never applied to the weights.
            it->second.has_fp8_input = true;
            break;
        default:
            break;
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_tensor_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    for (const fs::path &d : {base, pinned_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    // ── Weight → output records, in source-tensor order ──────────────────
    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    std::vector<TensorRecord> global_raw;
    int64_t total_pinned = 0;
    Bf16Stager stager;

    // Raw (never-quantized) kinds stay raw: the router, the norms, the conv
    // block and every bias. The F32 router is the source's own precision
    // choice (ModelOpt excluded it from quantization) and is preserved.
    auto is_raw_kind = [](NemotronNvfp4Kind k) {
        switch (k) {
        case NemotronNvfp4Kind::Embed:
        case NemotronNvfp4Kind::FinalNorm:
        case NemotronNvfp4Kind::LayerNorm:
        case NemotronNvfp4Kind::MambaConv:
        case NemotronNvfp4Kind::MambaConvBias:
        case NemotronNvfp4Kind::MambaALog:
        case NemotronNvfp4Kind::MambaD:
        case NemotronNvfp4Kind::MambaDtBias:
        case NemotronNvfp4Kind::MambaNorm:
        case NemotronNvfp4Kind::RouterGate:
        case NemotronNvfp4Kind::RouterBias:
            return true;
        default:
            return false;
        }
    };

    // lm_head is special: the SIBLING loader uploads it from global.arrow as
    // BF16 -> FP16 (not through the NVFP4 path), so the pre-quantized source's
    // W4A16_NVFP4 head is dequantized to BF16 and stored RAW. That keeps the
    // converted layout identical to the sibling's, and the dequant is exact
    // with respect to the source (the source values ARE the NVFP4 grid).
    const bool lm_head_is_nvfp4_source = true;

    for (const auto &[name, st] : weights) {
        if (st.info == nullptr) continue;
        const auto p = classify_nemotron_nvfp4_tensor(name);
        // Routed experts are handled by the dedicated per-expert pass below
        // (they get their own PackedExpert arrow file, not a layer batch).
        if (p.kind == NemotronNvfp4Kind::ExpertUpProj ||
            p.kind == NemotronNvfp4Kind::ExpertDownProj)
            continue;
        auto data = ReadTensorData(input_dir, *st.shard, *st.info);
        if (data.empty() ||
            static_cast<int64_t>(data.size()) != st.info->byte_size())
            return arrow::Status::IOError("failed to read tensor data: ", name);

        if (is_raw_kind(p.kind)) {
            TensorRecord r;
            r.name = name;
            r.dtype = st.info->dtype;
            r.shape = st.info->shape;
            r.data = std::move(data);
            if (p.kind == NemotronNvfp4Kind::Embed ||
                p.kind == NemotronNvfp4Kind::FinalNorm)
                global_raw.push_back(std::move(r));
            else
                raw_by_layer[p.layer].push_back(std::move(r));
            ++total_pinned;
            continue;
        }

        // Quantized weight: resolve the source layout from the observed dtype,
        // derive the LOGICAL shape, dequantize, and either store it raw
        // (lm_head, sibling-compatible) or requantize through the repo's
        // neutral quantizer.
        std::string err;
        const SourceLayout layout = layout_of(*st.info, err);
        if (!err.empty()) return arrow::Status::Invalid(err);
        std::vector<int64_t> logical = logical_shape_of(*st.info, layout, err);
        if (!err.empty()) return arrow::Status::Invalid(err);
        if (layout == SourceLayout::kBf16)
            stager.host() = std::move(data);
        else
            ARROW_RETURN_NOT_OK(
                dequant_source(st, input_dir, logical, layout, stager, err));

        if (p.kind == NemotronNvfp4Kind::LmHead && lm_head_is_nvfp4_source) {
            TensorRecord r;
            r.name = name;
            r.dtype = "BF16";
            r.shape = logical;
            r.data = std::move(stager.host());
            global_raw.push_back(std::move(r));
            ++total_pinned;
            continue;
        }

        DenseNvfp4Record r;
        r.name = name;
        r.shape = logical;
        r.dtype = "NVFP4";
        if (!quantize_bf16(stager.host(), logical.at(0) * logical.at(1), logical,
                           r, err))
            return arrow::Status::IOError(err);
        if (p.layer >= 0) nvfp4_by_layer[p.layer].push_back(std::move(r));
        ++total_pinned;
    }

    // ── Routed experts: per-expert up/down pair (no gate projection) ──────
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int H = top.hidden_size;
    using ExpertKey = std::pair<int, int>;
    absl::btree_map<ExpertKey, DenseNvfp4Record> up_map, down_map;
    for (const auto &[name, st] : weights) {
        if (st.info == nullptr) continue;
        const auto p = classify_nemotron_nvfp4_tensor(name);
        if (p.kind != NemotronNvfp4Kind::ExpertUpProj &&
            p.kind != NemotronNvfp4Kind::ExpertDownProj)
            continue;
        std::string err;
        const SourceLayout layout = layout_of(*st.info, err);
        if (!err.empty()) return arrow::Status::Invalid(err);
        std::vector<int64_t> logical = logical_shape_of(*st.info, layout, err);
        if (!err.empty()) return arrow::Status::Invalid(err);
        if (p.kind == NemotronNvfp4Kind::ExpertUpProj) {
            if (logical.at(0) != I || logical.at(1) != H)
                return arrow::Status::Invalid("bad expert up shape: ", name);
        } else {
            if (logical.at(0) != H || logical.at(1) != I)
                return arrow::Status::Invalid("bad expert down shape: ", name);
        }
        if (layout == SourceLayout::kBf16)
            stager.host() = ReadTensorData(input_dir, *st.shard, *st.info);
        else
            ARROW_RETURN_NOT_OK(
                dequant_source(st, input_dir, logical, layout, stager, err));
        DenseNvfp4Record r;
        r.name = name;
        r.shape = logical;
        r.dtype = "NVFP4";
        if (!quantize_bf16(stager.host(), logical.at(0) * logical.at(1), logical,
                           r, err))
            return arrow::Status::IOError(err);
        if (p.kind == NemotronNvfp4Kind::ExpertUpProj)
            up_map.insert_or_assign({p.layer, p.expert}, std::move(r));
        else
            down_map.insert_or_assign({p.layer, p.expert}, std::move(r));
    }
    for (int l = 0; l < top.num_layers; ++l) {
        if (top.layer_types[l] != "moe") continue;
        for (int e = 0; e < E; ++e) {
            auto up = up_map.find({l, e});
            auto dn = down_map.find({l, e});
            if (up == up_map.end() || dn == down_map.end())
                return arrow::Status::Invalid(
                    "missing routed expert: layer ", l, " expert ", e);
            PackedExpert pe;
            // NemotronH experts are non-gated MLPs: w1 is unused, w3 = up,
            // w2 = down (repo PackedExpert slot convention).
            pe.w3_packed = std::move(up->second.packed);
            pe.w3_scales = std::move(up->second.scales);
            pe.w3_shape = up->second.shape;
            pe.w3_global_scale = up->second.global_scale;
            pe.w2_packed = std::move(dn->second.packed);
            pe.w2_scales = std::move(dn->second.scales);
            pe.w2_shape = dn->second.shape;
            pe.w2_global_scale = dn->second.global_scale;
            pe.w1_shape = {0, 0};
            ARROW_RETURN_NOT_OK(write_expert_file(base, l, e, std::move(pe)));
        }
    }
    int total_experts = 0;
    for (int l = 0; l < top.num_layers; ++l)
        if (top.layer_types[l] == "moe") total_experts += E;

    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(write_batch(PinnedSchema::MakeBatch(global_raw),
                                        pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(write_batch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(write_batch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // Tokenizer: mandatory for real checkpoints; synthetic fixtures may omit.
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.pretokenizer = 7;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true}, tok_policy));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ", input_dir);
    }

    // Conversion-time VRAM profile (routed experts are the max term).
    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim * top.num_key_value_heads;
        vg.num_attn_layers = nemotron_moe::frozen::kAttnLayers;
        vg.num_moe_layers = nemotron_moe::frozen::kMoeLayers;
        vg.experts_per_layer = top.num_experts;
        vg.context_length = ReadContextLength(
            (fs::path(input_dir) / "config.json").string());
        std::vector<VramTensorInfo> inv;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                inv.push_back({t.name, t.numel(), t.byte_size()});
        top.vram = EstimateVram(
            inv,
            {+[](const std::string &) {
                 return true;  // every quantized source leaves as NVFP4
             },
             +[](const std::string &n) -> VramTensorClass {
                 auto k = classify_nemotron_nvfp4_tensor(n).kind;
                 return (k == NemotronNvfp4Kind::ExpertUpProj ||
                         k == NemotronNvfp4Kind::ExpertDownProj)
                            ? VramTensorClass::RoutedExpert
                            : VramTensorClass::Resident;
             }},
            vg);
    }

    // Stop ids: generation_config.json carries the LIST [2, 11]; config.json
    // only the scalar 2. Losing the second stop is a silent behavioural change,
    // so the list is what lands in the index.
    std::vector<int32_t> eos = ParseNemotronEosIds(input_dir);
    if (eos.empty() && top.eos_token_id >= 0)
        eos.push_back(top.eos_token_id);

    // Structural manifest (deterministic).
    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"logestix-nemotron-nvfp4\",\n"
          << "  \"model\": \"Nemotron-3.5-Lightning-30B-A3B-NVFP4\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n"
          << "  \"layer_types\": [";
        for (size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "],\n  \"attn_layer_idx\": [";
        bool first = true;
        for (int i = 0; i < top.num_layers; ++i) {
            if (top.layer_types[i] != "attention") continue;
            if (!first) j << ",";
            j << i;
            first = false;
        }
        j << "],\n  \"hidden_size\": " << top.hidden_size
          << ",\n  \"vocab_size\": " << top.vocab_size
          << ",\n  \"mamba_num_heads\": " << top.mamba_num_heads
          << ",\n  \"mamba_head_dim\": " << top.mamba_head_dim
          << ",\n  \"ssm_state_size\": " << top.ssm_state_size
          << ",\n  \"n_groups\": " << top.n_groups
          << ",\n  \"conv_kernel\": " << top.conv_kernel
          << ",\n  \"num_attention_heads\": " << top.num_attention_heads
          << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"rope_theta\": " << top.rope_theta
          << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.num_experts_per_tok
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n  \"shared_expert_intermediate_size\": "
          << top.shared_expert_intermediate_size
          << ",\n  \"router_scaling_factor\": "
          << nemotron_moe::frozen::kRoutedScale
          << ",\n  \"experts_no_gate\": true"
          << ",\n  \"bos_token_id\": " << top.bos_token_id
          << ",\n  \"eos_token_id\": [";
        for (size_t i = 0; i < eos.size(); ++i) {
            if (i) j << ", ";
            j << eos[i];
        }
        j << "]"
          << ",\n  \"pad_token_id\": " << top.pad_token_id
          << ",\n  \"vram\": {\n"
          << "    \"min_bytes\": " << top.vram.min_bytes
          << ",\n    \"optimal_bytes\": " << top.vram.optimal_bytes
          << ",\n    \"max_bytes\": " << top.vram.max_bytes
          << ",\n    \"context_tokens\": " << top.vram.context_tokens << "\n  }"
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"deferred\": {\"mtp\": \"cut\", \"mtp_tensors\": " << mtp_cut
          << "}"
          << ",\n  \"fp8_kv_scales_dropped\": " << fp8_kv_dropped
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n  \"total_experts\": " << total_experts
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // Output verification: every produced arrow file re-opens cleanly.
    std::error_code ec;
    for (fs::recursive_directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        if (it->path().extension() != ".arrow") continue;
        auto res = ReadRecordBatchFromFile(it->path());
        if (!res.ok())
            return arrow::Status::IOError("unreadable output arrow: ",
                                          it->path().string());
    }
    if (ec)
        return arrow::Status::IOError("cannot list ", base.string());

    lsxcommon::log::info(absl::StrFormat(
        "Nemotron NVFP4 conversion: %d layers, %d experts, %d MTP tensors cut, "
        "%d FP8 KV scales dropped, %d pinned tensors",
        top.num_layers, top.num_experts, mtp_cut, fp8_kv_dropped,
        top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace nemotron_nvfp4
} // namespace lsxcommon