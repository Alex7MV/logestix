// Inkling-2-Small synthetic end-to-end converter test. Runs the full converter
// on a tiny legal fixture (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1) and asserts the
// logestix-inkling-2-small layout: the sliding/full split, the AttnRes
// query renaming (source `attn_queries.L` is layer `L+1`), the two expert codec
// paths (NVFP4 verbatim with a per-expert global scale amax/2688, MXFP8
// requantized to NVFP4), the stripped+counted vision ballast, and the raw
// (never quantized) norm/bias/rel/sconv plane.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_convert.h>
#include "inkling_moe_model.h"
#include <lsxcommon/schema.h>
#include <simdjson.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../libs/lsxcommon/src/nvfp4_scale.h"
#include "test_inkling_fixture.h"

namespace fs = std::filesystem;
using namespace inklingtest;

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

bool DenseHasName(const fs::path &path, const std::string &name)
{
    if (!fs::exists(path)) return false;
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return false;
    auto rows = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    if (!rows.ok()) return false;
    for (const auto &r : *rows)
        if (r.name == name) return true;
    return false;
}

// The canonical codec, never a local replica: tests/test_nvfp4_scale.cpp
// fails any TU that hand-copies the E4M3 exponent field.
float E4m3(uint8_t b) { return lsxcommon::nvfp4::E4m3ToFloat(b); }

}  // namespace

TEST_CASE("inkling synthetic convert emits logestix-inkling-2-small",
          "[inkling][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_synth_in";
    const fs::path out = fs::temp_directory_path() / "inkling_synth_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);

    auto st = lsxcommon::inkling::ConvertInklingModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out / "logestix";
    const fs::path idx = base / "model.index.json";
    REQUIRE(fs::exists(idx));

    const std::string json = ReadFile(idx);
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    auto doc = parser.iterate(pad);

    std::string_view fmt;
    REQUIRE(doc["format"].get_string().get(fmt) == simdjson::SUCCESS);
    REQUIRE(fmt == "logestix-inkling-2-small");

    // `model` is what path auto-detection resolves: `gpu::detect_model_type`
    // reads this key from `<vault>/logestix/model.index.json`, normalizes it
    // (spaces -> '-', lower case) and hands the result to `LsxModelFactory`.
    // Without it `lsxbenchmark --model <vault>` / `lsxfabric --localmodel`
    // fail with "cannot autodetect model type" even though the tree is
    // complete -- every other family in the vault writes it. The normalized
    // value must be EXACTLY `IModel::Type()`, or the factory cannot resolve it.
    std::string_view card;
    REQUIRE(doc["model"].get_string().get(card) == simdjson::SUCCESS);
    REQUIRE(card == "Inkling-2-Small");
    {
        std::string normalized(card);
        for (char &c : normalized)
            if (c == ' ') c = '-';
            else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        REQUIRE(normalized == lsxcommon::InklingMoeModel::kType);
    }

    int64_t layers = 0, experts = 0, inter = 0;
    REQUIRE(doc["num_layers"].get_int64().get(layers) == simdjson::SUCCESS);
    REQUIRE(doc["num_experts"].get_int64().get(experts) == simdjson::SUCCESS);
    REQUIRE(doc["moe_intermediate_size"].get_int64().get(inter) ==
            simdjson::SUCCESS);
    REQUIRE(layers == kLayers);
    REQUIRE(experts == kExperts);
    REQUIRE(inter == kMoeInter);

    std::vector<int64_t> full_layers, sliding_layers;
    for (auto v : doc["full_layers"].get_array())
        full_layers.push_back(v.get_int64().value());
    for (auto v : doc["sliding_layers"].get_array())
        sliding_layers.push_back(v.get_int64().value());
    REQUIRE(full_layers == std::vector<int64_t>{kFullLayer});
    REQUIRE(sliding_layers == std::vector<int64_t>{0, 1, 2, 3, 4});

    std::vector<bool> codecs;
    for (auto v : doc["expert_nvfp4"].get_array())
        codecs.push_back(v.get_bool().value());
    REQUIRE(codecs.size() == static_cast<size_t>(kLayers));
    for (int l = 0; l < kLayers; ++l)
        REQUIRE(codecs[static_cast<size_t>(l)] == (l < kNovfp4Layers));

    int64_t skipped_vis = -1;
    REQUIRE(doc["skipped_visual_tensors"].get_int64().get(skipped_vis) ==
            simdjson::SUCCESS);
    REQUIRE(skipped_vis == 2);

    // ── pin split ─────────────────────────────────────────────────────────
    // EVERY dense tensor stays BF16, as in the release: `hf_quant_config.json`
    // puts only the routed experts in a low-precision codec. Requantizing the
    // projections costs a measured 9.5% relative L2 on them and turns the model
    // into a repetition loop, so the artifact must not do it.
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.wq_du.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.gate_attn.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.wr_du.weight")));
    REQUIRE_FALSE(fs::exists(base / "pinned" / "layer_0_nvfp4.arrow"));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.q_l2_clip.scale")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.rel_logits_proj.proj")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.attn_sink_proj.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn.k_sconv.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn_sconv.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "attn_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "mlp.gate.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_0.arrow",
                          Layer(0, "mlp.gate.bias")));

    // Full layer 5: three input norms and the dynamic conv pair instead of a
    // single attn_norm; the conv projections are NVFP4.
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "q_input_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "k_input_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "v_input_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "attn.k_sconv.static_w")));
    REQUIRE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                          Layer(5, "attn.k_sconv.weight_proj")));
    REQUIRE_FALSE(PinnedHasName(base / "pinned" / "layer_5.arrow",
                                Layer(5, "attn_norm.weight")));

    // ── globals + AttnRes query renaming ──────────────────────────────────
    // embed/unembed stay BF16 (the checkpoint's own precision for them, and the
    // two tensors that touch the residual stream at full width). They must NOT
    // appear in the NVFP4 plane.
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("embed.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("unembed.weight")));
    // No global NVFP4 plane exists at all now.
    REQUIRE_FALSE(fs::exists(base / "pinned" / "global_nvfp4.arrow"));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("embed_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow", Llm("norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.output_query")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.mlp_queries.0")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.mlp_queries.5")));
    // Source `attn_queries.0` belongs to layer 1 and `attn_queries.4` to layer
    // 5; there is no layer-0 query in either spelling.
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.attn_queries.1")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("attn_res.attn_queries.5")));
    REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow",
                                Llm("attn_res.attn_queries.0")));

    // ── ballast stripped ─────────────────────────────────────────────────
    for (const auto &name :
         {"model.visual.backbone.ln_pre.weight", "model.visual.proj_out.weight"})
        REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow", name));

    // ── expert files: one per (layer, expert) ────────────────────────────
    for (int l = 0; l < kLayers; ++l)
        for (int e = 0; e < kExperts; ++e)
            REQUIRE(fs::exists(base / "experts" / ("layer_" + std::to_string(l)) /
                               ("exp_" + std::to_string(e) + ".arrow")));

    // ── NVFP4 source: byte-verbatim, per-expert global scale amax/2688 ────
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "experts" / "layer_1" / "exp_2.arrow");
        REQUIRE(batch.ok());
        auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch, 16);
        if (!pe_res.ok()) FAIL(pe_res.status().ToString());
        const lsxcommon::PackedExpert &pe = *pe_res;

        auto src = MakeExpertPlane(1);
        const auto &w13 = src.tensors[0].data;   // [E, 2I, H/2]
        const auto &w2 = src.tensors[2].data;    // [E, H, I/2]
        const size_t w13_expert = static_cast<size_t>(2 * kMoeInter) * (kHidden / 2);
        const size_t w13_row = static_cast<size_t>(kHidden / 2);
        const size_t gate_bytes = static_cast<size_t>(kMoeInter) * w13_row;
        const uint8_t *base13 = w13.data() + w13_expert * 2;

        REQUIRE(pe.w1_packed.size() == gate_bytes);
        REQUIRE(pe.w3_packed.size() == gate_bytes);
        REQUIRE(pe.w1_scales.size() ==
                static_cast<size_t>(kMoeInter) * (kHidden / 16));
        // The checkpoint stores the fused slab ROW-INTERLEAVED -- source rows
        // alternate gate, up, gate, up -- and the release's own conversion
        // de-interleaves it (`Interleave(dim=1)` in transformers'
        // conversion_mapping.py for "inkling2_text", applied to `w13_weight` and
        // `w13_weight_scale` alike) before splitting gate from up. Copying the
        // first half instead, which is what an earlier converter did, pairs the
        // wrong scale with most rows: the artifact still looks byte-verbatim and
        // every plane still has the right size and magnitude, but the model
        // loses its knowledge. Measured on the real weights: perplexity 12,881
        // on "The capital of France is Paris" and the answer is " the"; with the
        // de-interleave it is 30.4 and the answer is " Paris".
        for (int r = 0; r < kMoeInter; ++r) {
            INFO("gate row " << r);
            REQUIRE(std::memcmp(pe.w1_packed.data() + static_cast<std::size_t>(r) * w13_row,
                                base13 + static_cast<std::size_t>(2 * r) * w13_row,
                                w13_row) == 0);
            INFO("up row " << r);
            REQUIRE(std::memcmp(pe.w3_packed.data() + static_cast<std::size_t>(r) * w13_row,
                                base13 + static_cast<std::size_t>(2 * r + 1) * w13_row,
                                w13_row) == 0);
        }

        const float expect_g13 = (0.08f + 0.001f * 2.0f) / 2688.0f;
        const float expect_g2 = (0.04f + 0.001f * 2.0f) / 2688.0f;
        REQUIRE(std::fabs(pe.w1_global_scale - expect_g13) < 1e-9f);
        REQUIRE(std::fabs(pe.w3_global_scale - expect_g13) < 1e-9f);
        REQUIRE(std::fabs(pe.w2_global_scale - expect_g2) < 1e-9f);
        REQUIRE(pe.w2_packed.size() ==
                static_cast<size_t>(kHidden) * (kMoeInter / 2));
        REQUIRE(std::memcmp(pe.w2_packed.data(),
                            w2.data() + static_cast<size_t>(kHidden) *
                                            (kMoeInter / 2) * 2,
                            pe.w2_packed.size()) == 0);
    }

    // ── MXFP8 source: kept as the release's own 8-bit plane ─────────────
    //
    // Layers 40-47 are E4M3 with a ue8m0 exponent per 32 values. The artifact
    // keeps those bytes verbatim and duplicates each exponent (per-32 -> per-16)
    // so the record satisfies the neutral schema's per-16 count, so the plane is
    // ONE byte per element rather than packed nibbles. Requantizing it to 4 bits
    // costs a measured 16-17% relative L2 on the MoE output of every affected
    // layer, which is enough to turn the model into a repetition loop.
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "experts" / "layer_4" / "exp_1.arrow");
        REQUIRE(batch.ok());
        // scale_block = 0: the per-16 heuristic assumes packed nibbles (two
        // elements per byte) and would reject a byte-per-element FP8 plane by 2x.
        auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch, 0);
        if (!pe_res.ok()) FAIL(pe_res.status().ToString());
        const lsxcommon::PackedExpert &pe = *pe_res;
        REQUIRE(pe.w1_packed.size() ==
                static_cast<size_t>(kMoeInter) * kHidden);        // 1 byte/elem
        REQUIRE(pe.w1_scales.size() ==
                static_cast<size_t>(kMoeInter) * (kHidden / 16));  // per-16
        REQUIRE(pe.w2_packed.size() == static_cast<size_t>(kHidden) * kMoeInter);
        REQUIRE(pe.w2_scales.size() ==
                static_cast<size_t>(kHidden) * (kMoeInter / 16));
        // The source bytes are passed through: no global scale is folded in and
        // the exponent duplicated in pairs is the source's per-32 byte.
        REQUIRE(pe.w1_global_scale == 1.0f);
        REQUIRE(pe.w2_global_scale == 1.0f);
        for (std::size_t i = 0; i + 1 < pe.w1_scales.size(); i += 2)
            REQUIRE(pe.w1_scales[i] == pe.w1_scales[i + 1]);
        // ... and the row pairing follows the same interleaved slab: gate row r
        // is source row 2r, up row r is source row 2r+1.
        {
            auto src8 = MakeExpertPlane(4);
            const auto &s13 = src8.tensors[1].data;      // [E, 2I, H/32]
            const std::size_t s13_row = static_cast<std::size_t>(kHidden / 32);
            const auto &w13_8 = src8.tensors[0].data;    // [E, 2I, H]
            const std::size_t w13_row8 = static_cast<std::size_t>(kHidden);
            const std::size_t s13_expert = static_cast<std::size_t>(2 * kMoeInter) * s13_row;
            const std::size_t w13_expert8 = static_cast<std::size_t>(2 * kMoeInter) * w13_row8;
            const uint8_t *b8 = w13_8.data() + w13_expert8 * 1;
            const uint8_t *p8 = s13.data() + s13_expert * 1;
            for (int r = 0; r < kMoeInter; ++r) {
                INFO("mxfp8 gate row " << r);
                REQUIRE(std::memcmp(pe.w1_packed.data() + static_cast<std::size_t>(r) * kHidden,
                                    b8 + static_cast<std::size_t>(2 * r) * w13_row8,
                                    w13_row8) == 0);
                REQUIRE(pe.w1_scales[static_cast<std::size_t>(r) * (kHidden / 16)] ==
                        pe.w1_scales[static_cast<std::size_t>(r) * (kHidden / 16) + 1]);
                REQUIRE(pe.w1_scales[static_cast<std::size_t>(r) * (kHidden / 16)] ==
                        p8[static_cast<std::size_t>(2 * r) * s13_row]);
            }
        }
        // ue8m0 is 2^(b-127), so a legal byte is 0..254 (255 is the reserved
        // "no scale" marker in some MX formats and must not appear here).
        for (uint8_t b : pe.w1_scales) REQUIRE(b < 255);
    }
}
