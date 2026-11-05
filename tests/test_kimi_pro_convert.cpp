// Kimi-K3.1-Pro synthetic end-to-end converter test.
//
// The variant's ONE delta against Kimi-K3 is the Gated-DSA Lightning Indexer on
// the 24 full-attention layers, so the gates below are exactly the variant
// discriminator, both ways:
//   * a Pro checkpoint converts into `logestix-kimi-k31-pro-nvfp4` with the
//     `dsa_layers` table and the 120 (here: 10) indexer tensors stored BF16
//     verbatim -- never quantized, because they decide WHICH keys are visible;
//   * a Pro checkpoint without indexers, or with an indexer on a KDA layer, is
//     a hard error (never a silent drop);
//   * a Kimi-K3 checkpoint converted as Pro is a hard error.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/kimi_moe_pro_convert.h>
#include <lsxcommon/schema.h>
#include <simdjson.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_kimi_pro_fixture.h"

namespace fs = std::filesystem;
using namespace kimiprotest;

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

const lsxcommon::TensorRecord *FindPinned(const fs::path &path,
                                          const std::string &name)
{
    static lsxcommon::TensorRecord out;
    if (!fs::exists(path)) return nullptr;
    auto batch = lsxcommon::ReadRecordBatchFromFile(path);
    if (!batch.ok()) return nullptr;
    auto rows = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!rows.ok()) return nullptr;
    for (const auto &r : *rows)
        if (r.name == name) {
            out = r;
            return &out;
        }
    return nullptr;
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

fs::path ConvertOrFail(const fs::path &in, const fs::path &out)
{
    auto st = lsxcommon::kimi_pro::ConvertKimiK31ProModel(in.string(),
                                                          out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    return out / "logestix";
}

std::string IndexerName(int layer, const char *suffix)
{
    return "language_model.model.layers." + std::to_string(layer) +
           ".self_attn.indexer." + suffix;
}

}  // namespace

TEST_CASE("kimi-k3.1-pro synthetic convert emits the DSA format tag and table",
          "[kimi-pro][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "kimi_pro_synth_in";
    const fs::path out = fs::temp_directory_path() / "kimi_pro_synth_out";
    fs::remove_all(in);
    fs::remove_all(out);
    WriteCheckpoint(in, Variant::kPro);
    const fs::path base = ConvertOrFail(in, out);

    const fs::path idx = base / "model.index.json";
    REQUIRE(fs::exists(idx));
    const std::string json = ReadFile(idx);
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    auto doc = parser.iterate(pad);

    std::string_view fmt, model, model_type;
    REQUIRE(doc["format"].get_string().get(fmt) == simdjson::SUCCESS);
    REQUIRE(fmt == "logestix-kimi-k31-pro-nvfp4");
    // `model` is the human-readable card name (Kimi-K3 writes "Kimi-K3");
    // `model_type` is the registry id the loader gates on.
    REQUIRE(doc["model"].get_string().get(model) == simdjson::SUCCESS);
    REQUIRE(model == "Kimi-K3.1-Pro");
    REQUIRE(doc["model_type"].get_string().get(model_type) ==
            simdjson::SUCCESS);
    REQUIRE(model_type == "kimi-k3.1-pro");

    // The DSA table is the MLA table (0-based; config listed 1-based {3,4}).
    std::vector<int64_t> dsa;
    bool has_dsa = false;
    for (auto v : doc["dsa_layers"].get_array()) {
        has_dsa = true;
        dsa.push_back(v.get_int64().value());
    }
    REQUIRE(has_dsa);
    REQUIRE(dsa == std::vector<int64_t>{2, 3});
    std::vector<int64_t> mla;
    for (auto v : doc["mla_layers"].get_array())
        mla.push_back(v.get_int64().value());
    REQUIRE(dsa == mla);

    bool indexer = false;
    REQUIRE(doc["indexer"].get_bool().get(indexer) == simdjson::SUCCESS);
    REQUIRE(indexer);
    int64_t head_dim = -1, heads = -1, topk = -1, rope_dim = -1;
    REQUIRE(doc["index_head_dim"].get_int64().get(head_dim) ==
            simdjson::SUCCESS);
    REQUIRE(doc["index_num_heads"].get_int64().get(heads) ==
            simdjson::SUCCESS);
    REQUIRE(doc["index_topk"].get_int64().get(topk) == simdjson::SUCCESS);
    REQUIRE(doc["index_rope_dim"].get_int64().get(rope_dim) ==
            simdjson::SUCCESS);
    REQUIRE(head_dim == kIndexHeadDim);
    REQUIRE(heads == kIndexHeads);
    REQUIRE(topk == kIndexTopK);
    REQUIRE(rope_dim == kIndexRopeDim);

    // The rest of the tree is the Kimi-K3 layout: same hybrid tables.
    std::vector<int64_t> kda;
    for (auto v : doc["kda_layers"].get_array())
        kda.push_back(v.get_int64().value());
    REQUIRE(kda == std::vector<int64_t>{0, 1});
    REQUIRE(DenseHasName(base / "pinned" / "layer_2_nvfp4.arrow",
                         "language_model.model.layers.2.self_attn.q_b_proj.weight"));
}

TEST_CASE("kimi-k3.1-pro indexer tensors are stored BF16 verbatim",
          "[kimi-pro][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "kimi_pro_bf16_in";
    const fs::path out = fs::temp_directory_path() / "kimi_pro_bf16_out";
    fs::remove_all(in);
    fs::remove_all(out);
    WriteCheckpoint(in, Variant::kPro);
    const fs::path base = ConvertOrFail(in, out);

    for (int l : {2, 3}) {
        const fs::path raw = base / "pinned" /
                             ("layer_" + std::to_string(l) + ".arrow");
        const fs::path nvfp4 = base / "pinned" /
                               ("layer_" + std::to_string(l) + "_nvfp4.arrow");
        struct Expect {
            const char *suffix;
            int64_t rows;
            int64_t cols;  // 0 => a 1-D tensor of `rows` elements
        };
        const Expect expects[] = {
            {"wq_b.weight", kIndexWqBRows, kIndexQLora},
            {"wk.weight", kIndexHeadDim, kimik3test::kHidden},
            {"k_norm.weight", kIndexHeadDim, 0},
            {"k_norm.bias", kIndexHeadDim, 0},
            {"weights_proj.weight", kIndexHeads, kimik3test::kHidden},
        };
        for (const auto &e : expects) {
            const std::string name = IndexerName(l, e.suffix);
            INFO("layer " << l << " " << name);
            const auto *rec = FindPinned(raw, name);
            REQUIRE(rec != nullptr);
            // Never quantized: BF16 bytes in the raw pinned batch, and the name
            // must not appear in the NVFP4 batch.
            REQUIRE(rec->dtype == "BF16");
            if (e.cols == 0) {
                REQUIRE(rec->shape.size() == 1);
                REQUIRE(rec->shape[0] == e.rows);
                REQUIRE(rec->data.size() == static_cast<size_t>(e.rows * 2));
            } else {
                REQUIRE(rec->shape.size() == 2);
                REQUIRE(rec->shape[0] == e.rows);
                REQUIRE(rec->shape[1] == e.cols);
                REQUIRE(rec->data.size() ==
                        static_cast<size_t>(e.rows * e.cols * 2));
            }
            REQUIRE_FALSE(DenseHasName(nvfp4, name));
        }
    }

    // Content, not just shape: the fixture's deterministic BF16 pattern must
    // survive byte for byte (a requantize would round every value).
    const auto *wq_b = FindPinned(base / "pinned" / "layer_2.arrow",
                                  IndexerName(2, "wq_b.weight"));
    REQUIRE(wq_b != nullptr);
    const auto expect = synthtest::MakeBf16Values(
        static_cast<int64_t>(kIndexWqBRows) * kIndexQLora,
        [](int64_t i) { return 0.02f * (1.0f + static_cast<float>(i % 11)); });
    REQUIRE(wq_b->data == expect);
}

TEST_CASE("kimi-k3.1-pro convert rejects a checkpoint without indexers",
          "[kimi-pro][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "kimi_pro_noidx_in";
    const fs::path out = fs::temp_directory_path() / "kimi_pro_noidx_out";
    fs::remove_all(in);
    fs::remove_all(out);
    WriteCheckpoint(in, Variant::kProNoIndexer);
    auto st = lsxcommon::kimi_pro::ConvertKimiK31ProModel(in.string(),
                                                          out.string());
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.ToString().find("indexer") != std::string::npos);
}

TEST_CASE("kimi-k3.1-pro convert rejects an indexer on a KDA layer",
          "[kimi-pro][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "kimi_pro_kdaidx_in";
    const fs::path out = fs::temp_directory_path() / "kimi_pro_kdaidx_out";
    fs::remove_all(in);
    fs::remove_all(out);
    WriteCheckpoint(in, Variant::kProIndexerOnKda);
    auto st = lsxcommon::kimi_pro::ConvertKimiK31ProModel(in.string(),
                                                          out.string());
    REQUIRE_FALSE(st.ok());
    const std::string why = st.ToString();
    INFO(why);
    REQUIRE(why.find("indexer") != std::string::npos);
    REQUIRE(why.find("KDA") != std::string::npos);
}

TEST_CASE("kimi-k3.1-pro convert rejects a Kimi-K3 checkpoint",
          "[kimi-pro][convert]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path in = fs::temp_directory_path() / "kimi_pro_k3_in";
    const fs::path out = fs::temp_directory_path() / "kimi_pro_k3_out";
    fs::remove_all(in);
    fs::remove_all(out);
    WriteCheckpoint(in, Variant::k3);
    auto st = lsxcommon::kimi_pro::ConvertKimiK31ProModel(in.string(),
                                                          out.string());
    REQUIRE_FALSE(st.ok());
    const std::string why = st.ToString();
    INFO(why);
    REQUIRE(why.find("kimi_k31") != std::string::npos);
}
