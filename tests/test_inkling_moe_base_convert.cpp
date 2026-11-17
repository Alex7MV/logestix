// Inkling (base) synthetic end-to-end converter test. Runs the full converter
// on a tiny legal fixture (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1) and asserts the
// logestix-inkling-nvfp4 layout: the sliding/full split with BOTH kv widths,
// the dense-MLP prefix, the stripped+counted vision/audio/MTP ballast, the raw
// (never quantized) norm/bias/rel/sconv/router plane, and — the defect class
// that cost Inkling-2-Small a whole debugging session — the ROW PAIRING of the
// three fused gate/up slabs:
//   * routed w13 is HALVES (gate rows [0,I), up rows [I,2I));
//   * shared_w13 and dense w13_dn are ROW-INTERLEAVED (gate=even, up=odd).
// The fixture writes row-identifiable values, so the pins below decode the
// written NVFP4 records with the canonical codec and compare against the
// CORRECT and the SWAPPED source rows: a converter that confuses the two
// layouts reconstructs the wrong rows at O(1) relative L2.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/inkling_moe_base_convert.h>
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
#include "test_inkling_base_fixture.h"

namespace fs = std::filesystem;
using namespace inklingbasetest;

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

const lsxcommon::DenseNvfp4Record *FindDense(
    const std::vector<lsxcommon::DenseNvfp4Record> &recs, const std::string &name)
{
    for (const auto &r : recs)
        if (r.name == name) return &r;
    return nullptr;
}

std::vector<lsxcommon::DenseNvfp4Record> DenseRecords(const fs::path &path)
{
    REQUIRE(fs::exists(path));
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    REQUIRE(batch.ok());
    auto rows = lsxcommon::DenseNvfp4Schema::UnpackBatch(*batch);
    REQUIRE(rows.ok());
    return *rows;
}

// The canonical E2M1 levels (the device codec's table, mirrored here for the
// host reconstruction; tests/test_lsx_kernels.cu pins the device side).
float E2m1(int nib)
{
    static const float kLv[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    const float v = kLv[nib & 7];
    return (nib & 8) ? -v : v;
}

// Dequantize a written NVFP4 record to its logical [R, C] matrix.
std::vector<float> Dequant(const lsxcommon::DenseNvfp4Record &r)
{
    const int64_t R = r.shape[0], C = r.shape[1];
    std::vector<float> out(static_cast<size_t>(R * C));
    for (int64_t i = 0; i < R * C; ++i) {
        const uint8_t byte = r.packed[static_cast<size_t>(i / 2)];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        out[static_cast<size_t>(i)] =
            E2m1(nib) *
            lsxcommon::nvfp4::E4m3ToFloat(r.scales[static_cast<size_t>(i / 16)]) *
            r.global_scale;
    }
    return out;
}

float Bf16At(const std::vector<uint8_t> &data, int64_t i)
{
    const uint32_t u = static_cast<uint32_t>(
        data[static_cast<size_t>(2 * i)] |
        (data[static_cast<size_t>(2 * i) + 1] << 8)) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Relative L2 of a decoded matrix against a row-pattern expectation.
float RelL2Vs(const std::vector<float> &got, int64_t rows, int64_t cols,
              int64_t tag)
{
    double num = 0, den = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) {
            const float want = RowValue(r, c, tag);
            const double d = got[static_cast<size_t>(r * cols + c)] - want;
            num += d * d;
            den += double(want) * want;
        }
    return static_cast<float>(std::sqrt(num / den));
}

}  // namespace

TEST_CASE("inkling base synthetic convert emits logestix-inkling-nvfp4",
          "[inkling][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_base_synth_in";
    const fs::path out = fs::temp_directory_path() / "inkling_base_synth_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);

    auto st = lsxcommon::inkling::base::ConvertInklingBaseModel(in.string(),
                                                               out.string());
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
    REQUIRE(fmt == "logestix-inkling-nvfp4");

    // `model` normalizes (spaces -> '-', lower case) to exactly kType "inkling"
    // for gpu::detect_model_type / LsxModelFactory path auto-detection.
    std::string_view card;
    REQUIRE(doc["model"].get_string().get(card) == simdjson::SUCCESS);
    REQUIRE(card == "Inkling");

    int64_t layers = 0, experts = 0, inter = 0, shared = 0, dense_layers = 0;
    REQUIRE(doc["num_layers"].get_int64().get(layers) == simdjson::SUCCESS);
    REQUIRE(doc["num_experts"].get_int64().get(experts) == simdjson::SUCCESS);
    REQUIRE(doc["moe_intermediate_size"].get_int64().get(inter) ==
            simdjson::SUCCESS);
    REQUIRE(doc["num_shared_experts"].get_int64().get(shared) ==
            simdjson::SUCCESS);
    REQUIRE(doc["dense_mlp_layers"].get_int64().get(dense_layers) ==
            simdjson::SUCCESS);
    REQUIRE(layers == kLayers);
    REQUIRE(experts == kExperts);
    REQUIRE(inter == kMoeInter);
    REQUIRE(shared == kShared);
    REQUIRE(dense_layers == kDenseLayers);

    std::vector<int64_t> full_layers, sliding_layers;
    for (auto v : doc["full_layers"].get_array())
        full_layers.push_back(v.get_int64().value());
    for (auto v : doc["sliding_layers"].get_array())
        sliding_layers.push_back(v.get_int64().value());
    REQUIRE(full_layers == std::vector<int64_t>{kFullLayer});
    REQUIRE(sliding_layers == std::vector<int64_t>{0, 1, 2, 3, 4});

    int64_t kv_swa = 0;
    REQUIRE(doc["swa_num_key_value_heads"].get_int64().get(kv_swa) ==
            simdjson::SUCCESS);
    REQUIRE(kv_swa == kKvHeadsSliding);

    double mup = 0;
    REQUIRE(doc["logits_mup_div"].get_double().get(mup) == simdjson::SUCCESS);
    REQUIRE(mup == 24.0);

    int64_t skipped_vis = -1, skipped_audio = -1, skipped_mtp = -1;
    REQUIRE(doc["skipped_visual_tensors"].get_int64().get(skipped_vis) ==
            simdjson::SUCCESS);
    REQUIRE(doc["skipped_audio_tensors"].get_int64().get(skipped_audio) ==
            simdjson::SUCCESS);
    REQUIRE(doc["skipped_mtp_tensors"].get_int64().get(skipped_mtp) ==
            simdjson::SUCCESS);
    REQUIRE(skipped_vis == 2);
    REQUIRE(skipped_audio == 1);
    REQUIRE(skipped_mtp == 2);

    // ── pin split: dense projections NVFP4, side tensors raw ───────────────
    for (int l = 0; l < kLayers; ++l) {
        const fs::path raw = base / "pinned" / ("layer_" + std::to_string(l) + ".arrow");
        const fs::path nv = base / "pinned" / ("layer_" + std::to_string(l) + "_nvfp4.arrow");
        REQUIRE(PinnedHasName(raw, Layer(l, "attn_norm.weight")));
        REQUIRE(PinnedHasName(raw, Layer(l, "mlp_norm.weight")));
        REQUIRE(PinnedHasName(raw, Layer(l, "attn.q_norm.weight")));
        REQUIRE(PinnedHasName(raw, Layer(l, "attn.k_norm.weight")));
        REQUIRE(PinnedHasName(raw, Layer(l, "attn.rel_logits_proj.proj")));
        REQUIRE(PinnedHasName(raw, Layer(l, "attn.k_sconv.weight")));
        REQUIRE(PinnedHasName(raw, Layer(l, "attn.v_sconv.weight")));
        REQUIRE(PinnedHasName(raw, Layer(l, "attn_sconv.weight")));
        REQUIRE(PinnedHasName(raw, Layer(l, "mlp_sconv.weight")));
        const auto recs = DenseRecords(nv);
        REQUIRE(FindDense(recs, Layer(l, "attn.wq_du.weight")) != nullptr);
        REQUIRE(FindDense(recs, Layer(l, "attn.wk_dv.weight")) != nullptr);
        REQUIRE(FindDense(recs, Layer(l, "attn.wv_dv.weight")) != nullptr);
        REQUIRE(FindDense(recs, Layer(l, "attn.wo_ud.weight")) != nullptr);
        REQUIRE(FindDense(recs, Layer(l, "attn.wr_du.weight")) != nullptr);
        if (IsDense(l)) {
            REQUIRE(FindDense(recs, Layer(l, "mlp.gate_proj.weight")) != nullptr);
            REQUIRE(FindDense(recs, Layer(l, "mlp.up_proj.weight")) != nullptr);
            REQUIRE(FindDense(recs, Layer(l, "mlp.down_proj.weight")) != nullptr);
            REQUIRE(PinnedHasName(raw, Layer(l, "mlp.global_scale")));
        } else {
            REQUIRE(PinnedHasName(raw, Layer(l, "mlp.gate.weight")));
            REQUIRE(PinnedHasName(raw, Layer(l, "mlp.gate.bias")));
            REQUIRE(PinnedHasName(raw, Layer(l, "mlp.gate.global_scale")));
            for (int s = 0; s < kShared; ++s) {
                const std::string p = "mlp.shared_experts." + std::to_string(s) + ".";
                REQUIRE(FindDense(recs, Layer(l, p + "gate_proj.weight")) != nullptr);
                REQUIRE(FindDense(recs, Layer(l, p + "up_proj.weight")) != nullptr);
                REQUIRE(FindDense(recs, Layer(l, p + "down_proj.weight")) != nullptr);
            }
        }
    }

    // ── globals: embed/unembed stay raw BF16 (FP16 on device) ──────────────
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow", Llm("embed.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow", Llm("unembed.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow",
                          Llm("embed_norm.weight")));
    REQUIRE(PinnedHasName(base / "pinned" / "global.arrow", Llm("norm.weight")));
    REQUIRE_FALSE(fs::exists(base / "pinned" / "global_nvfp4.arrow"));

    // ── ballast stripped ────────────────────────────────────────────────────
    for (const auto &name : {"model.visual.final_norm.weight",
                             "model.audio.encoder.weight",
                             "model.mtp.layers.0.input_proj.weight"})
        REQUIRE_FALSE(PinnedHasName(base / "pinned" / "global.arrow", name));

    // ── expert files: MoE layers only ──────────────────────────────────────
    for (int l = 0; l < kLayers; ++l)
        for (int e = 0; e < kExperts; ++e) {
            const fs::path p = base / "experts" / ("layer_" + std::to_string(l)) /
                               ("exp_" + std::to_string(e) + ".arrow");
            if (IsDense(l))
                REQUIRE_FALSE(fs::exists(p));
            else
                REQUIRE(fs::exists(p));
        }

    // ── ROW PAIRING 1: dense w13_dn is ROW-INTERLEAVED ─────────────────────
    // gate row r must reconstruct source row 2r (tag 500+l), NOT the halves
    // split's row r and NOT the odd (up) rows.
    {
        const auto recs = DenseRecords(base / "pinned" / "layer_0_nvfp4.arrow");
        const auto *gate = FindDense(recs, Layer(0, "mlp.gate_proj.weight"));
        const auto *up = FindDense(recs, Layer(0, "mlp.up_proj.weight"));
        REQUIRE(gate != nullptr);
        REQUIRE(up != nullptr);
        REQUIRE(gate->shape == std::vector<int64_t>{kDenseInter, kHidden});
        const auto g = Dequant(*gate);
        const auto u = Dequant(*up);
        INFO("dense gate vs correct rows relL2=" << RelL2Vs(g, kDenseInter, kHidden, 500));
        REQUIRE(RelL2Vs(g, kDenseInter, kHidden, 500) < 0.2f);
        REQUIRE(RelL2Vs(g, kDenseInter, kHidden, 600) > 0.5f);
        REQUIRE(RelL2Vs(u, kDenseInter, kHidden, 600) < 0.2f);
        REQUIRE(RelL2Vs(u, kDenseInter, kHidden, 500) > 0.5f);
        // The halves-split defect: gate would carry source rows [0,I) of the
        // interleaved slab, i.e. alternating gate/up rows 0..I-1 — a mixture
        // that matches NEITHER plane.
        const auto slab = DenseW13(0);
        double num = 0, den = 0;
        for (int64_t r = 0; r < kDenseInter; ++r)
            for (int64_t c = 0; c < kHidden; ++c) {
                const float halves = Bf16At(slab, r * kHidden + c);
                const double d = g[static_cast<size_t>(r * kHidden + c)] - halves;
                num += d * d;
                den += double(halves) * halves;
            }
        REQUIRE(std::sqrt(num / den) > 0.5);
    }

    // ── ROW PAIRING 2: shared_w13 is ROW-INTERLEAVED ───────────────────────
    {
        const auto recs = DenseRecords(base / "pinned" / "layer_3_nvfp4.arrow");
        for (int s = 0; s < kShared; ++s) {
            const std::string p = "mlp.shared_experts." + std::to_string(s) + ".";
            const auto *gate = FindDense(recs, Layer(3, p + "gate_proj.weight"));
            const auto *up = FindDense(recs, Layer(3, p + "up_proj.weight"));
            REQUIRE(gate != nullptr);
            REQUIRE(up != nullptr);
            REQUIRE(gate->shape == std::vector<int64_t>{kMoeInter, kHidden});
            const auto g = Dequant(*gate);
            const auto u = Dequant(*up);
            INFO("shared " << s << " gate relL2=" << RelL2Vs(g, kMoeInter, kHidden, 300 + s));
            REQUIRE(RelL2Vs(g, kMoeInter, kHidden, 300 + s) < 0.2f);
            REQUIRE(RelL2Vs(g, kMoeInter, kHidden, 400 + s) > 0.5f);
            REQUIRE(RelL2Vs(u, kMoeInter, kHidden, 400 + s) < 0.2f);
            REQUIRE(RelL2Vs(u, kMoeInter, kHidden, 300 + s) > 0.5f);
        }
    }

    // ── ROW PAIRING 3: routed w13 is HALVES (no de-interleave!) ────────────
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
            base / "experts" / "layer_3" / "exp_1.arrow");
        REQUIRE(batch.ok());
        auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch, 16);
        if (!pe_res.ok()) FAIL(pe_res.status().ToString());
        const lsxcommon::PackedExpert &pe = *pe_res;
        // ExpertSchema carries no shape columns (9-field contract); the plane
        // SIZES are the geometry pin, and the loader re-derives [I,H]/[H,I].
        REQUIRE(pe.w1_packed.size() == size_t(kMoeInter) * kHidden / 2);
        REQUIRE(pe.w3_packed.size() == size_t(kMoeInter) * kHidden / 2);
        REQUIRE(pe.w2_packed.size() == size_t(kHidden) * kMoeInter / 2);
        REQUIRE(pe.w1_scales.size() == size_t(kMoeInter) * kHidden / 16);
        REQUIRE(pe.w3_scales.size() == size_t(kMoeInter) * kHidden / 16);
        REQUIRE(pe.w2_scales.size() == size_t(kHidden) * kMoeInter / 16);
        auto deq = [](const std::vector<uint8_t> &p, const std::vector<uint8_t> &s,
                      float gs, int64_t n) {
            std::vector<float> out(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t byte = p[static_cast<size_t>(i / 2)];
                const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
                out[static_cast<size_t>(i)] =
                    E2m1(nib) *
                    lsxcommon::nvfp4::E4m3ToFloat(s[static_cast<size_t>(i / 16)]) * gs;
            }
            return out;
        };
        const auto w1 = deq(pe.w1_packed, pe.w1_scales, pe.w1_global_scale,
                            int64_t(kMoeInter) * kHidden);
        const auto w3 = deq(pe.w3_packed, pe.w3_scales, pe.w3_global_scale,
                            int64_t(kMoeInter) * kHidden);
        // expert 1: gate tag 101, up tag 201 (HALVES layout).
        INFO("routed gate relL2=" << RelL2Vs(w1, kMoeInter, kHidden, 101));
        REQUIRE(RelL2Vs(w1, kMoeInter, kHidden, 101) < 0.2f);
        REQUIRE(RelL2Vs(w1, kMoeInter, kHidden, 201) > 0.5f);
        REQUIRE(RelL2Vs(w3, kMoeInter, kHidden, 201) < 0.2f);
        REQUIRE(RelL2Vs(w3, kMoeInter, kHidden, 101) > 0.5f);
        // The de-interleave defect (inkling-2's layout applied here): gate
        // would carry the EVEN source rows, which alternate gate/up values.
        const auto slab = RoutedW13();
        const int64_t H = kHidden;
        const int64_t expert_off = int64_t(1) * 2 * kMoeInter * H;
        double num = 0, den = 0;
        for (int64_t r = 0; r < kMoeInter; ++r)
            for (int64_t c = 0; c < H; ++c) {
                const float even = Bf16At(slab, expert_off + (2 * r) * H + c);
                const double d = w1[static_cast<size_t>(r * H + c)] - even;
                num += d * d;
                den += double(even) * even;
            }
        REQUIRE(std::sqrt(num / den) > 0.5);
        // down plane: expert 1 tag 701.
        const auto w2 = deq(pe.w2_packed, pe.w2_scales, pe.w2_global_scale,
                            int64_t(kHidden) * kMoeInter);
        REQUIRE(RelL2Vs(w2, kHidden, kMoeInter, 701) < 0.2f);
    }
}

TEST_CASE("inkling base converter rejects an unknown tensor",
          "[inkling][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "inkling_base_bad_in";
    const fs::path out = fs::temp_directory_path() / "inkling_base_bad_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    WriteCheckpoint(in);
    // Append a shard with a tensor the classifier does not know: the converter
    // must ERROR, never silently skip it.
    std::vector<synthtest::FakeTensor> extra;
    Push(extra, "model.llm.layers.0.mlp.mystery_tensor", "BF16", {4},
         synthtest::MakeBf16(4));
    synthtest::WriteSafetensors(in, "model-extra.safetensors", extra);
    auto st = lsxcommon::inkling::base::ConvertInklingBaseModel(in.string(),
                                                               out.string());
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("mystery_tensor") != std::string::npos);
}