// deepseek_moe_repack.cpp — offline DeepSeek routed-expert NVFP4 repack
// (logestix-v1 -> logestix-v1-nvfp4). See deepseek_moe_repack.h for the
// format contract. Host-only; strict error handling.
#include "lsxcommon/deepseek_moe_repack.h"

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"
#include "nvfp4_scale.h"

#include <arrow/api.h>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace lsxcommon {
namespace deepseek {

namespace {

// Accepts either the model root (contains logestix/) or the logestix
// directory itself — mirrors expert_store.h path resolution.
fs::path ResolveLogestixRoot(const fs::path &p)
{
    return fs::exists(p / "logestix") ? p / "logestix" : p;
}

// Remap one E8M0 per-32 scale array onto pow2-E4M3 per-16 pairs with a
// per-tensor shift. @p src and @p dst may alias (in-place update); the
// remap runs through a local buffer. Zero bytes pass through as zero
// pairs; the chosen shift is folded into @p gs_out (2^m, exact float).
bool RemapScaleArray(const std::vector<uint8_t> &src,
                     std::vector<uint8_t> &dst, float &gs_out,
                     const std::string &ctx)
{
    uint8_t b_min = 0xFF, b_max = 0x00;
    bool any_nonzero = false;
    for (const uint8_t b : src) {
        if (b == 0) continue;
        any_nonzero = true;
        b_min = std::min(b_min, b);
        b_max = std::max(b_max, b);
    }
    int m = 0;
    if (any_nonzero) {
        if (!nvfp4::ChooseE8m0ScaleShift(b_min, b_max, m)) {
            log::error(absl::StrFormat(
                "[deepseek-repack] %s: E8M0 scale byte spread [%u..%u] "
                "exceeds the 13-step pow2-E4M3 window — refusing lossy "
                "repack", ctx.c_str(), static_cast<int>(b_min),
                static_cast<int>(b_max)));
            return false;
        }
    }
    std::vector<uint8_t> out(src.size() * 2);
    for (size_t i = 0; i < src.size(); ++i) {
        uint8_t pair[2];
        nvfp4::E8m0ToE4m3Pow2Pair(src[i], m, pair);
        out[2 * i] = pair[0];
        out[2 * i + 1] = pair[1];
    }
    dst = std::move(out);
    gs_out = std::ldexp(1.0f, m);
    return true;
}

} // namespace

arrow::Status RepackDeepseekExpertsNvfp4(const std::string &src,
                                         const std::string &dst)
{
    // Output mirrors the INPUT layout: a model-root input produces a
    // model-root output (dst/logestix), a logestix-dir input produces a
    // logestix-dir output — so the loader can consume either verbatim.
    const fs::path src_path(src);
    const fs::path dst_path(dst);
    const bool src_is_root = fs::exists(src_path / "logestix");
    const fs::path sroot = src_is_root ? src_path / "logestix" : src_path;
    const fs::path droot = src_is_root ? dst_path / "logestix" : dst_path;
    const fs::path s_experts = sroot / "experts";
    if (!fs::exists(s_experts))
        return arrow::Status::Invalid("no experts/ directory under ",
                                      sroot.string());

    std::error_code ec;
    fs::create_directories(droot, ec);
    if (ec)
        return arrow::Status::IOError("create dst failed: ", ec.message());

    // ── 1. model.index.json: swap the format tag, carry the rest verbatim ──
    {
        const fs::path s_index = sroot / "model.index.json";
        std::ifstream f(s_index);
        if (!f)
            return arrow::Status::Invalid("model.index.json missing under ",
                                          sroot.string());
        std::string json((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        static const std::regex fmt_re("(\"format\"\\s*:\\s*)\"[^\"]*\"");
        const std::string swapped =
            std::regex_replace(json, fmt_re, "$1\"logestix-v1-nvfp4\"");
        if (swapped == json)
            return arrow::Status::Invalid(
                "index format tag not found/unchanged in ", s_index.string());
        fs::create_directories(droot, ec);
        std::ofstream out(droot / "model.index.json");
        out << swapped;
        if (!out)
            return arrow::Status::IOError("write index failed under ",
                                          droot.string());
    }

    // ── 2. routed experts: per-tensor E8M0 -> pow2-E4M3/16 scale repack ──
    int64_t experts_done = 0;
    for (auto &layer_dir : fs::directory_iterator(s_experts)) {
        if (!layer_dir.is_directory()) continue;
        const fs::path d_layer =
            droot / "experts" / layer_dir.path().filename();
        fs::create_directories(d_layer);
        for (auto &entry : fs::directory_iterator(layer_dir.path())) {
            if (!entry.is_regular_file()) continue;
            const std::string fname = entry.path().filename().string();
            if (fname.find("exp_") != 0 || fname.find(".arrow") ==
                                              std::string::npos)
                continue;

            ARROW_ASSIGN_OR_RAISE(
                auto batch, lsxcommon::ReadRecordBatchFromFile(entry.path()));
            ARROW_ASSIGN_OR_RAISE(auto pe,
                                  lsxcommon::ExpertSchema::UnpackBatch(batch));

            const std::string ctx = fname;
            if (!RemapScaleArray(pe.w1_scales, pe.w1_scales,
                                 pe.w1_global_scale, ctx) ||
                !RemapScaleArray(pe.w2_scales, pe.w2_scales,
                                 pe.w2_global_scale, ctx) ||
                !RemapScaleArray(pe.w3_scales, pe.w3_scales,
                                 pe.w3_global_scale, ctx))
                return arrow::Status::Invalid("scale spread too wide: ",
                                              entry.path().string());

            ARROW_ASSIGN_OR_RAISE(auto out_batch,
                                  lsxcommon::ExpertSchema::MakeBatch(pe));
            ARROW_RETURN_NOT_OK(lsxcommon::WriteRecordBatchToFile(
                out_batch, d_layer / fname));
            if (++experts_done % 1000 == 0)
                log::info(absl::StrFormat("[deepseek-repack] %d experts",
                                          static_cast<int>(experts_done)));
        }
    }
    log::info(absl::StrFormat(
        "[deepseek-repack] repacked %d routed experts (e8m0/32 -> e4m3/16)",
        static_cast<int>(experts_done)));

    // ── 3. everything else (pinned/, tokenizer, extra files) verbatim ──
    for (auto &entry : fs::directory_iterator(sroot)) {
        const std::string name = entry.path().filename().string();
        if (name == "experts" || name == "model.index.json") continue;
        const fs::path target = droot / name;
        if (entry.is_directory()) {
            fs::copy(entry.path(), target,
                     fs::copy_options::recursive |
                         fs::copy_options::overwrite_existing,
                     ec);
            if (ec)
                return arrow::Status::IOError("copy dir failed: ", name);
        } else if (entry.is_regular_file()) {
            fs::copy_file(entry.path(), target,
                          fs::copy_options::overwrite_existing, ec);
            if (ec)
                return arrow::Status::IOError("copy file failed: ", name);
        }
    }

    return arrow::Status::OK();
}

} // namespace deepseek
} // namespace lsxcommon