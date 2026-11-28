// Inkling-2-Mini synthetic end-to-end converter test. Runs the full converter
// on a tiny legal fixture (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1) and asserts the
// logestix-inkling-2-mini layout: the sliding/full split, the loop fields, the
// AttnRes query renaming (source `attn_queries.L` is layer `L+1`), the merged
// cross-pass message queries ([q,k,v,mlp] per loop layer), the per-loop-layer
// MessageGate controllers, the two expert codec paths (NVFP4 verbatim with a
// per-expert global scale amax/2688, MXFP8 kept at 8 bits with duplicated
// ue8m0 scales), the row-interleaved w13 de-interleave for BOTH codecs, and the
// stripped+counted vision ballast.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"
#include <lsxcommon/inkling_moe_loop_convert.h>
#include <lsxcommon/schema.h>
#include <simdjson.h>

#include "lsxstorage/artifact.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_inkling_moe_loop_fixture.h"

namespace fs = std::filesystem;
using namespace inklinglooptest;

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

bool PinnedHasName(const fs::path &path, const std::string &name)
{
    if (!fs::exists(path)) return false;
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return false;
    auto rows = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!rows.ok()) return false;
    for (const auto &r : *rows)
        if (r.name == name) return true;
    return false;
}

// Returns the raw record with the given name (empty name when absent).
lsxcommon::TensorRecord PinnedRecord(const fs::path &path, const std::string &name)
{
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return {};
    auto rows = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!rows.ok()) return {};
    for (const auto &r : *rows)
        if (r.name == name) return r;
    return {};
}

}  // namespace

TEST_CASE("inkling loop synthetic convert emits logestix-inkling-2-mini",
          "[inkling][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_loop_synth_in";
    const fs::path out = fs::temp_directory_path() / "inkling_loop_synth_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);

    auto st = lsxcommon::inkling::ConvertInklingLoopModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const fs::path idx = base / "model.index.json";
    REQUIRE(fs::exists(idx));

    const std::string json = ReadFile(idx);
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    auto doc = parser.iterate(pad);

    // `layout` is an INTEGER generation, not a per-model tag: the assertion used
    // to read doc["format"].get_string(...) and compare it against
    // std::to_string(kIndexLayout), which cannot pass under either schema — the
    // index stopped writing `format` (design §2.4.3) and a string read of an
    // integer never succeeds. Half-migrated by the layout change and never run,
    // because this suite is label `convert`, which the fast sweep does not cover.
    int64_t layout = 0;
    REQUIRE(doc["layout"].get_int64().get(layout) == simdjson::SUCCESS);
    REQUIRE(layout == lsxcommon::kIndexLayout);

    // `model` is what path auto-detection resolves; normalized it must be
    // exactly the factory key "inkling-2-mini".
    std::string_view card;
    REQUIRE(doc["model"].get_string().get(card) == simdjson::SUCCESS);
    REQUIRE(card == "Inkling-2-Mini");
    {
        std::string normalized(card);
        for (char &c : normalized)
            if (c == ' ') c = '-';
            else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        REQUIRE(normalized == "inkling-2-mini");
    }

    int64_t layers = 0, experts = 0, inter = 0;
    REQUIRE(doc["num_layers"].get_int64().get(layers) == simdjson::SUCCESS);
    REQUIRE(doc["num_experts"].get_int64().get(experts) == simdjson::SUCCESS);
    REQUIRE(doc["moe_intermediate_size"].get_int64().get(inter) ==
            simdjson::SUCCESS);
    REQUIRE(layers == kLayers);
    REQUIRE(experts == kExperts);
    REQUIRE(inter == kMoeInter);

    // ── loop fields ───────────────────────────────────────────────────────
    int64_t loop_start = 0, loop_end = 0, loop_passes = 0, gate_rank = 0;
    REQUIRE(doc["loop_start"].get_int64().get(loop_start) == simdjson::SUCCESS);
    REQUIRE(doc["loop_end"].get_int64().get(loop_end) == simdjson::SUCCESS);
    REQUIRE(doc["loop_passes"].get_int64().get(loop_passes) == simdjson::SUCCESS);
    REQUIRE(doc["loop_gate_rank"].get_int64().get(gate_rank) == simdjson::SUCCESS);
    REQUIRE(loop_start == kLoopStart);
    REQUIRE(loop_end == kLoopEnd);
    REQUIRE(loop_passes == kLoopPasses);
    REQUIRE(gate_rank == kLoopGateRank);

    std::vector<int64_t> full_layers, sliding_layers;
    for (auto v : doc["full_layers"].get_array())
        full_layers.push_back(v.get_int64().value());
    for (auto v : doc["sliding_layers"].get_array())
        sliding_layers.push_back(v.get_int64().value());
    REQUIRE(full_layers == std::vector<int64_t>{kFullLayer});
    REQUIRE(sliding_layers == std::vector<int64_t>{0, 1, 2, 3, 4, 6, 7});

    std::vector<bool> codecs;
    // Rule 5: the index carries no per-layer codec key. The split is read back
    // from the bytes the converter just wrote — the same derivation artifact.fb
    // records in layer_codec — so this still proves the mixed NVFP4/MXFP8 planes
    // landed where the source put them.
    {
        const auto scan = lsx::storage::ScanArtifactTree(out);
        std::string joined;
        for (const auto &p : scan.problems) joined += "\n  " + p;
        CAPTURE(joined);
        REQUIRE(scan.ok);
        REQUIRE(scan.layer_codec.size() == static_cast<size_t>(kLayers));
        for (uint64_t layer_mask : scan.layer_codec)
            codecs.push_back((layer_mask & (1ull << 5)) != 0);
    }
    REQUIRE(codecs.size() == static_cast<size_t>(kLayers));
    for (int l = 0; l < kLayers; ++l)
        REQUIRE(codecs[static_cast<size_t>(l)] == (l < kNovfp4Layers));

    int64_t skipped_vis = -1;
    REQUIRE(doc["skipped_visual_tensors"].get_int64().get(skipped_vis) ==
            simdjson::SUCCESS);
    REQUIRE(skipped_vis == 2);

    // ── dense plane stays BF16 (the release's own precision) ─────────────
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.wq_du.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.gate_attn.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "mlp.gate.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "mlp.gate.bias")));
    REQUIRE_FALSE(fs::exists(base / "pinned" / "layer_0_nvfp4.arrow"));
    REQUIRE_FALSE(fs::exists(base / "pinned" / "global_nvfp4.arrow"));

    // Full layer 5: three input norms + dynamic conv pair.
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "q_input_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "attn.k_sconv.static_w")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "attn.k_sconv.weight_proj")));
    REQUIRE_FALSE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                                Layer(5, "attn_norm.weight")));

    // ── globals + AttnRes query renaming ──────────────────────────────────
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow", Llm("embed.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow", Llm("unembed.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.output_query")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.mlp_queries.0")));
    // Source attn_queries.0 belongs to layer 1; attn_queries.4 to full layer 5.
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.attn_queries.1")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.attn_queries.5")));
    REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow",
                                Llm("attn_res.attn_queries.0")));

    // ── cross-pass message queries: merged [loop_layers, 4, hidden] ───────
    // Order per loop layer i: [q, k, v, mlp] with q=queries[2i],
    // mlp=queries[2i+1], k=kv_queries[2i], v=kv_queries[2i+1] (verified
    // against the release's Inkling2LoopMergeMessageQueries on real tensors).
    {
        auto r = PinnedRecord(base / "pinned" / "global.arrow",
                              Llm("message_queries"));
        REQUIRE_FALSE(r.name.empty());
        REQUIRE(r.dtype == "F32");
        REQUIRE(r.shape == std::vector<int64_t>{kLoopLayers, 4, kHidden});
        REQUIRE(r.data.size() ==
                static_cast<size_t>(kLoopLayers * 4 * kHidden) * 4);
        auto src_q = synthtest::MakeF32Values(2 * kLoopLayers * kHidden,
                                              [](int64_t i) {
                                                  return 0.001f * static_cast<float>(i + 1);
                                              });
        auto src_kv = synthtest::MakeF32Values(2 * kLoopLayers * kHidden,
                                               [](int64_t i) {
                                                   return -0.001f * static_cast<float>(i + 1);
                                               });
        const float *merged = reinterpret_cast<const float *>(r.data.data());
        for (int i = 0; i < kLoopLayers; ++i) {
            for (int h = 0; h < kHidden; ++h) {
                const size_t row = static_cast<size_t>((i * 4) * kHidden + h);
                const float q = *reinterpret_cast<const float *>(
                    src_q.data() + static_cast<size_t>((2 * i) * kHidden + h) * 4);
                const float mlp = *reinterpret_cast<const float *>(
                    src_q.data() + static_cast<size_t>((2 * i + 1) * kHidden + h) * 4);
                const float k = *reinterpret_cast<const float *>(
                    src_kv.data() + static_cast<size_t>((2 * i) * kHidden + h) * 4);
                const float v = *reinterpret_cast<const float *>(
                    src_kv.data() + static_cast<size_t>((2 * i + 1) * kHidden + h) * 4);
                INFO("loop layer " << i << " hidden " << h);
                REQUIRE(merged[row] == q);
                REQUIRE(merged[row + static_cast<size_t>(kHidden)] == k);
                REQUIRE(merged[row + static_cast<size_t>(2 * kHidden)] == v);
                REQUIRE(merged[row + static_cast<size_t>(3 * kHidden)] == mlp);
            }
        }
        // The two source tensors themselves are not kept under their old names.
        REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow",
                                    Llm("middle_cycle_cross_pass_queries")));
        REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow",
                                    Llm("middle_cycle_cross_pass_kv_queries")));
    }

    // ── loop controllers: present on loop layers only ─────────────────────
    for (int l = 0; l < kLayers; ++l) {
        for (const char *m : {"attn", "k", "v", "mlp"}) {
            const std::string down =
                Layer(l, absl::StrCat("controllers.", m, ".down_weight"));
            const std::string forget =
                Layer(l, absl::StrCat("controllers.", m, ".forget_weight"));
            const std::string gain =
                Layer(l, absl::StrCat("controllers.", m, ".gain_weight"));
            INFO("layer " << l << " controller " << m);
            if (IsLoop(l)) {
                REQUIRE(PinnedHasName(base / "pinned" / ("layer_" + std::to_string(l) + ".arrow"), down));
                REQUIRE(PinnedHasName(base / "pinned" / ("layer_" + std::to_string(l) + ".arrow"), forget));
                REQUIRE(PinnedHasName(base / "pinned" / ("layer_" + std::to_string(l) + ".arrow"), gain));
            } else {
                REQUIRE_FALSE(PinnedHasName(base / "pinned" / ("layer_" + std::to_string(l) + ".arrow"), down));
            }
        }
    }

    // ── ballast stripped ─────────────────────────────────────────────────
    for (const auto &name :
         {"model.visual.backbone.ln_pre.weight", "model.visual.proj_out.weight"})
        REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow", name));

    // ── expert files: one per (layer, expert) ────────────────────────────
    for (int l = 0; l < kLayers; ++l)
        for (int e = 0; e < kExperts; ++e)
            REQUIRE(fs::exists(base / "experts" / ("layer_" + std::to_string(l)) /
                               ("exp_" + std::to_string(e) + ".arrow")));

    // ── NVFP4 source: byte-verbatim, de-interleaved, amax/2688 ────────────
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "experts" / "layer_1" / "exp_2.arrow");
        REQUIRE(batch.ok());
        auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch, 16);
        if (!pe_res.ok()) FAIL(pe_res.status().ToString());
        const lsxcommon::PackedExpert &pe = *pe_res;

        auto src = MakeExpertPlane(1);
        const auto &w13 = src.tensors[0].data;   // [E, 2I, H/2]
        const auto &s13 = src.tensors[1].data;   // [E, 2I, H/16]
        const auto &w2 = src.tensors[2].data;    // [E, H, I/2]
        const size_t w13_expert = static_cast<size_t>(2 * kMoeInter) * (kHidden / 2);
        const size_t w13_row = static_cast<size_t>(kHidden / 2);
        const size_t s13_expert = static_cast<size_t>(2 * kMoeInter) * (kHidden / 16);
        const size_t s13_row = static_cast<size_t>(kHidden / 16);
        const uint8_t *base13 = w13.data() + w13_expert * 2;
        const uint8_t *sbase13 = s13.data() + s13_expert * 2;

        REQUIRE(pe.w1_packed.size() ==
                static_cast<size_t>(kMoeInter) * w13_row);
        // The slab is ROW-INTERLEAVED: gate row r == source row 2r, up row r
        // == source row 2r+1, weights AND scales (the Small defect, pinned).
        for (int r = 0; r < kMoeInter; ++r) {
            INFO("gate row " << r);
            REQUIRE(std::memcmp(pe.w1_packed.data() + static_cast<std::size_t>(r) * w13_row,
                                base13 + static_cast<std::size_t>(2 * r) * w13_row,
                                w13_row) == 0);
            REQUIRE(std::memcmp(pe.w1_scales.data() + static_cast<std::size_t>(r) * s13_row,
                                sbase13 + static_cast<std::size_t>(2 * r) * s13_row,
                                s13_row) == 0);
            INFO("up row " << r);
            REQUIRE(std::memcmp(pe.w3_packed.data() + static_cast<std::size_t>(r) * w13_row,
                                base13 + static_cast<std::size_t>(2 * r + 1) * w13_row,
                                w13_row) == 0);
            REQUIRE(std::memcmp(pe.w3_scales.data() + static_cast<std::size_t>(r) * s13_row,
                                sbase13 + static_cast<std::size_t>(2 * r + 1) * s13_row,
                                s13_row) == 0);
        }
        const float expect_g13 = (0.08f + 0.001f * 2.0f) / 2688.0f;
        const float expect_g2 = (0.04f + 0.001f * 2.0f) / 2688.0f;
        REQUIRE(std::fabs(pe.w1_global_scale - expect_g13) < 1e-9f);
        REQUIRE(std::fabs(pe.w3_global_scale - expect_g13) < 1e-9f);
        REQUIRE(std::fabs(pe.w2_global_scale - expect_g2) < 1e-9f);
        REQUIRE(std::memcmp(pe.w2_packed.data(),
                            w2.data() + static_cast<size_t>(kHidden) *
                                            (kMoeInter / 2) * 2,
                            pe.w2_packed.size()) == 0);
    }

    // ── MXFP8 source: kept at 8 bits, ue8m0 duplicated per-32 -> per-16 ───
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "experts" / "layer_6" / "exp_1.arrow");
        REQUIRE(batch.ok());
        auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch, 0);
        if (!pe_res.ok()) FAIL(pe_res.status().ToString());
        const lsxcommon::PackedExpert &pe = *pe_res;
        REQUIRE(pe.w1_packed.size() ==
                static_cast<size_t>(kMoeInter) * kHidden);        // 1 byte/elem
        REQUIRE(pe.w1_scales.size() ==
                static_cast<size_t>(kMoeInter) * (kHidden / 16));  // per-16 count
        REQUIRE(pe.w1_global_scale == 1.0f);
        REQUIRE(pe.w2_global_scale == 1.0f);
        for (std::size_t i = 0; i + 1 < pe.w1_scales.size(); i += 2)
            REQUIRE(pe.w1_scales[i] == pe.w1_scales[i + 1]);
        for (uint8_t b : pe.w1_scales) REQUIRE(b < 255);
        // Same interleaved slab: gate row r is source row 2r.
        auto src8 = MakeExpertPlane(6);
        const auto &w13_8 = src8.tensors[0].data;    // [E, 2I, H]
        const auto &s13 = src8.tensors[1].data;      // [E, 2I, H/32]
        const std::size_t w13_row8 = static_cast<std::size_t>(kHidden);
        const std::size_t s13_row = static_cast<std::size_t>(kHidden / 32);
        const std::size_t w13_expert8 = static_cast<std::size_t>(2 * kMoeInter) * w13_row8;
        const std::size_t s13_expert = static_cast<std::size_t>(2 * kMoeInter) * s13_row;
        const uint8_t *b8 = w13_8.data() + w13_expert8 * 1;
        const uint8_t *p8 = s13.data() + s13_expert * 1;
        for (int r = 0; r < kMoeInter; ++r) {
            INFO("mxfp8 gate row " << r);
            REQUIRE(std::memcmp(pe.w1_packed.data() + static_cast<std::size_t>(r) * kHidden,
                                b8 + static_cast<std::size_t>(2 * r) * w13_row8,
                                w13_row8) == 0);
            REQUIRE(std::memcmp(pe.w3_packed.data() + static_cast<std::size_t>(r) * kHidden,
                                b8 + static_cast<std::size_t>(2 * r + 1) * w13_row8,
                                w13_row8) == 0);
            REQUIRE(pe.w1_scales[static_cast<std::size_t>(r) * (kHidden / 16)] ==
                    p8[static_cast<std::size_t>(2 * r) * s13_row]);
            REQUIRE(pe.w3_scales[static_cast<std::size_t>(r) * (kHidden / 16)] ==
                    p8[static_cast<std::size_t>(2 * r + 1) * s13_row]);
        }
    }
}

TEST_CASE("inkling loop convert rejects an unknown tensor", "[inkling][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_loop_synth_bad";
    const fs::path out = fs::temp_directory_path() / "inkling_loop_synth_bad_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);
    // Add a tensor the classifier does not know: the converter must error,
    // never silently skip it.
    {
        std::vector<synthtest::FakeTensor> extra;
        Push(extra, Llm("mystery_tensor"), "BF16", {kHidden},
             synthtest::MakeBf16(kHidden));
        synthtest::WriteSafetensors(in, "extra-00001-of-00001.safetensors", extra);
    }
    auto st = lsxcommon::inkling::ConvertInklingLoopModel(in.string(), out.string());
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.ToString().find("mystery_tensor") != std::string::npos);
}

TEST_CASE("inkling loop convert rejects a codec mismatch", "[inkling][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_loop_synth_codec";
    const fs::path out = fs::temp_directory_path() / "inkling_loop_synth_codec_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);
    // Rewrite hf_quant_config.json claiming layer 0 is MXFP8 while its tensors
    // are NVFP4: the dtype census must catch the disagreement.
    {
        std::string nvfp4, mxfp8;
        for (int l = 0; l < kLayers; ++l) {
            const bool nv = (l < kNovfp4Layers) && l != 0;
            for (const char *kind : {"w13", "w2"}) {
                std::string &dst = nv ? nvfp4 : mxfp8;
                if (!dst.empty()) dst += ",";
                dst += absl::StrCat("\"model.llm.layers.", l, ".mlp.experts.",
                                    kind, "_weight\"");
            }
        }
        std::ofstream f(in / "hf_quant_config.json");
        f << absl::StrCat("{\"model\": {\"parameter_groups\": [",
                          "{\"compute_dtype\": \"nvfp4\", \"tensors\": [", nvfp4,
                          "]}, {\"compute_dtype\": \"mxfp8\", \"tensors\": [", mxfp8,
                          "]}]}}\n");
    }
    auto st = lsxcommon::inkling::ConvertInklingLoopModel(in.string(), out.string());
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.ToString().find("codec mismatch") != std::string::npos);
}