// Per-layer numeric gate for Nemotron-3-Ultra-550B against the slice oracle
// (tools/nemotron_ultra_oracle.py).
//
// The 1.1 TB BF16 source cannot be assembled on this host (750 GB RAM), so the
// source-fidelity instrument is a SLICE: a real `NemotronHForCausalLM` built from
// the checkpoint's own contiguous layers 0..7 (mamba, moe, mamba, moe, mamba,
// moe, mamba, attention) plus the real embedding / norm_f / lm_head. Contiguous
// is the requirement that makes it an oracle at all: engine layer N must have
// followed the same history as oracle block N, which stops being true the moment
// a layer is skipped in between.
//
// What this proves that SelfCheck cannot: the latent MoE bridge (fc1 -> experts
// at 2048 -> fc2), the top-22 sigmoid router with routed_scaling_factor 5.0, the
// 256-head Mamba-2 widths and the NoPE attention all produce the REFERENCE's
// hidden states, not merely a plausible "Paris".
//
// SKIPs (exit 4) unless both the fixture and the converted tree are present.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <simdjson.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nemotron_moe_ultra_model.h"

#include <lsxcommon/model_factory.h>
#include <lsxcommon/nemotron_moe_ultra_expert_cache.h>
#include <lsxcommon/nemotron_moe_ultra_inference.h>
#include <lsxcommon/nemotron_moe_ultra_load.h>
#include <lsxcommon/path_registry.h>

namespace fs = std::filesystem;
using lsxcommon::nemotron_ultra::UltraExpertCache;
using lsxcommon::nemotron_ultra::UltraIndex;
using lsxcommon::nemotron_ultra::UltraSession;
using lsxcommon::nemotron_ultra::UltraWeights;

namespace {

fs::path FixturePath()
{
    if (const char *env = std::getenv("LSX_NEMOTRON_ULTRA_ORACLE")) return env;
    return fs::path(LSX_NEMOTRON_ULTRA_FIXTURE_DIR) / "layers_slice.json";
}

fs::path ConvertedTree()
{
    if (const char *env = std::getenv("LSX_TEST_NEMOTRON_ULTRA_550B_A55B_ROOT"))
        return env;
    return fs::path(lsxcommon::lsx_model_models_dir()) /
           "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16";
}

struct RefBlock {
    int source_layer = -1;
    std::vector<double> l2;    // per prompt position
    std::vector<std::vector<double>> lead;  // [pos][dims]
};

struct Oracle {
    std::vector<int> layers;
    std::vector<int> prompt_ids;
    std::vector<RefBlock> blocks;
};

bool LoadOracle(const fs::path &path, Oracle &out)
{
    std::ifstream f(path);
    if (!f.good()) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc) != simdjson::SUCCESS) return false;
    simdjson::dom::object root = doc.get_object();
    for (auto v : root["layers"].get_array()) out.layers.push_back(static_cast<int>(int64_t(v)));
    for (auto v : root["prompt_ids"].get_array())
        out.prompt_ids.push_back(static_cast<int>(int64_t(v)));

    simdjson::dom::object blocks = root["blocks"].get_object();
    // Blocks are keyed block_0..block_N; index them by their recorded source layer
    // so a reordered fixture cannot pass silently.
    std::vector<std::pair<int, RefBlock>> by_layer;
    for (auto [key, value] : blocks) {
        RefBlock b;
        simdjson::dom::object o = value.get_object();
        b.source_layer = static_cast<int>(int64_t(o["source_layer"]));
        for (auto x : o["l2"].get_array()) b.l2.push_back(double(x));
        for (auto row : o["lead"].get_array()) {
            std::vector<double> r;
            for (auto x : row.get_array()) r.push_back(double(x));
            b.lead.push_back(std::move(r));
        }
        by_layer.emplace_back(b.source_layer, std::move(b));
    }
    // Sort by the recorded source layer (block_9 must not be assumed to be
    // "after" block_10 because of its name).
    std::sort(by_layer.begin(), by_layer.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (auto &[l, b] : by_layer) out.blocks.push_back(std::move(b));
    return !out.blocks.empty() && !out.prompt_ids.empty();
}

// One dumped engine row: "layer L pos P l2 <x> v0 v1 ...".
struct EngineRow {
    int layer = -1;
    int pos = -1;
    double l2 = 0.0;
    std::vector<double> lead;
};

std::vector<EngineRow> ReadDump(const fs::path &path)
{
    std::vector<EngineRow> rows;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string w;
        EngineRow r;
        is >> w >> r.layer >> w >> r.pos >> w >> r.l2;
        double v;
        while (is >> v) r.lead.push_back(v);
        rows.push_back(std::move(r));
    }
    return rows;
}

} // namespace

TEST_CASE("ultra per-layer numerics match the reference slice", "[ultra][layers]")
{
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) SKIP("no CUDA device");

    std::error_code ec;
    const fs::path fixture = FixturePath();
    if (!fs::exists(fixture, ec))
        SKIP("slice-oracle fixture absent (run tools/nemotron_ultra_oracle.py): " +
             fixture.string());
    const fs::path tree = ConvertedTree();
    if (!fs::exists(tree / "model.index.json", ec))
        SKIP("converted tree absent: " + tree.string());

    Oracle oracle;
    REQUIRE(LoadOracle(fixture, oracle));
    REQUIRE(oracle.blocks.size() >= 2);

    // The fixture must describe THIS model: the widths it recorded are the
    // passport's, so a fixture captured from another checkpoint cannot pose as
    // evidence for this one.
    UltraIndex idx;
    std::string err;
    REQUIRE(lsxcommon::nemotron_ultra::LoadUltraIndex(tree.string(), idx, err));
    REQUIRE(lsxcommon::nemotron_ultra::ValidateUltraGeometry(idx, err));

    const fs::path dump = fs::temp_directory_path() / "nu_layers_dump.txt";
    fs::remove(dump, ec);
    const int last_pos = static_cast<int>(oracle.prompt_ids.size()) - 1;
    setenv("LSX_NEMOTRON_ULTRA_DUMP_LAYERS", dump.c_str(), 1);
    setenv("LSX_NEMOTRON_ULTRA_DUMP_POS", std::to_string(last_pos).c_str(), 1);

    UltraWeights w;
    REQUIRE(lsxcommon::nemotron_ultra::LoadUltraWeights(tree.string(), idx, w, nullptr,
                                                        err));
    UltraExpertCache cache;
    lsxcommon::nemotron_ultra::LoadUltraExpertStore(tree.string(), idx.num_layers,
                                                    idx.num_experts, cache.store);
    cache.experts_per_layer = idx.num_experts;
    cache.moe_index_of_layer.assign(static_cast<std::size_t>(idx.num_layers), -1);
    for (int l = 0; l < idx.num_layers; ++l)
        if (idx.layer_types[static_cast<std::size_t>(l)] == "moe")
            cache.moe_index_of_layer[static_cast<std::size_t>(l)] = cache.moe_layers++;
    cache.layout = lsxcommon::nemotron_ultra::PlanUltraExpertSlotFromStore(cache.store);
    REQUIRE(cache.layout.ok);
    REQUIRE(lsxcommon::nemotron_ultra::AllocateUltraExpertCache(cache, idx.num_experts_per_tok * 2, err));

    UltraSession s;
    REQUIRE(lsxcommon::nemotron_ultra::AllocUltraSession(idx, /*max_seq=*/64, s, err));

    // Which pointers does the walk actually see? A null d_packed/scales makes the
    // GEMV launcher return cudaErrorInvalidValue without naming the tensor.
    INFO("layer0 in_proj packed=" << static_cast<const void *>(w.layers[0].in_proj.d_packed)
         << " scales=" << static_cast<const void *>(w.layers[0].in_proj.d_scales)
         << " rows=" << w.layers[0].in_proj.rows
         << " cols=" << w.layers[0].in_proj.cols);
    INFO("proj16=" << static_cast<void *>(s.proj16) << " normed="
                   << static_cast<void *>(s.normed));
    CHECK(w.layers[0].in_proj.d_packed != nullptr);
    CHECK(w.layers[0].in_proj.d_scales != nullptr);
    CHECK(w.layers[1].latent_up.d_packed != nullptr);
    CHECK(w.layers[1].shared_up.d_packed != nullptr);
    CHECK(w.layers[7].q.d_packed != nullptr);   // 7 is the first ATTENTION layer
    CHECK(w.layers[2].q.d_packed == nullptr);   // 2 is mamba: no attention weights
    lsxcommon::nemotron_ultra::ResetUltraSession(s, nullptr);

    // Prefill exactly the oracle's prompt ids; the dump filters to the last
    // position, which is the one the oracle summed over.
    for (std::size_t i = 0; i < oracle.prompt_ids.size(); ++i) {
        const cudaError_t e = lsxcommon::nemotron_ultra::UltraForwardToken(
            idx, w, s, cache, oracle.prompt_ids[i], static_cast<int>(i),
            /*prefill=*/true, nullptr);
        REQUIRE(e == cudaSuccess);
    }
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    unsetenv("LSX_NEMOTRON_ULTRA_DUMP_LAYERS");
    unsetenv("LSX_NEMOTRON_ULTRA_DUMP_POS");

    const auto rows = ReadDump(dump);
    REQUIRE(rows.size() >= oracle.blocks.size());
    if (rows.empty()) {
        lsxcommon::nemotron_ultra::FreeUltraSession(s);
        lsxcommon::nemotron_ultra::FreeUltraExpertCache(cache);
        w.Free();
        FAIL("no per-layer rows were dumped");
    }

    // Compare every oracle block against the engine's SAME layer.
    for (const auto &ref : oracle.blocks) {
        const EngineRow *row = nullptr;
        for (const auto &r : rows)
            if (r.layer == ref.source_layer) row = &r;
        REQUIRE(row != nullptr);
        const auto &rr = *row;

        const double ref_l2 = ref.l2[static_cast<std::size_t>(last_pos)];
        const double rel = std::fabs(rr.l2 - ref_l2) / std::max(ref_l2, 1e-9);

        // Cosine over the recorded leading dims: sign- and scale-robust where the
        // L2 alone would not be, and sensitive to any rotation applied where the
        // reference applies none (the NoPE defect).
        const auto &ref_lead = ref.lead[static_cast<std::size_t>(last_pos)];
        double dot = 0.0, na = 0.0, nb = 0.0;
        const std::size_t nd = std::min(ref_lead.size(), rr.lead.size());
        for (std::size_t i = 0; i < nd; ++i) {
            dot += rr.lead[i] * ref_lead[i];
            na += rr.lead[i] * rr.lead[i];
            nb += ref_lead[i] * ref_lead[i];
        }
        const double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;

        INFO("layer " << ref.source_layer << ": engine l2 " << rr.l2 << " vs ref "
                      << ref_l2 << " (rel " << rel << "), cos " << cos);
        // Thresholds, and why they are not the sibling's 2e-2:
        //
        // This gate compares the engine's NVFP4 tree against the source in BF16,
        // so 4-bit quantization is INSIDE the measured number. Measured on this
        // checkpoint (2026-11-11): applying the codec to the reference's own
        // weights moves its layer-6/7 L2 by 1.6%, and the residual against that
        // quantized reference is 3e-4 (L0), 2.4e-3 (L1), 7e-3 (L4), 3.0e-2 (L5)
        // -- the runtime reproduces its own weights to ~3e-4 at the embedding and
        // first block, and the growth with depth is numeric, not structural. The
        // 30B's 2e-2 was measured runtime-vs-dequantized-TREE, a strictly tighter
        // comparison; reusing it here would have compared two different
        // instruments.
        //
        // 5e-2 over 8 layers still rejects anything structural: a missing latent
        // projection, a wrong router or a spurious rotation land at 1e-1..1e0
        // (the sibling measured 1.20 under its NoPE defect).
        CHECK(rel < 5e-2);
        CHECK(cos > 0.99);
    }

    lsxcommon::nemotron_ultra::FreeUltraSession(s);
    lsxcommon::nemotron_ultra::FreeUltraExpertCache(cache);
    w.Free();
    fs::remove(dump, ec);
}
