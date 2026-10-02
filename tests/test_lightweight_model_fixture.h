#pragma once

// Index-only ("lightweight") converted-model fixture — MODEL-NEUTRAL helper,
// deliberately without a family stem so the model suite guard treats it as
// shared infrastructure and lets every family (and the app suites) use it.
//
// What it builds, and why this is enough:
//
//   <root>/logestix/model.index.json            {"model":"<MODEL>"}
//   <root>/logestix/pinned/tokenizer_vocab.arrow
//   <root>/logestix/pinned/tokenizer_merges.arrow
//
// No weights, no conversion, no CUDA. It covers the whole host-side
// model-path surface because:
//   * gpu::detect_model_type (deepseek_moe_load.cu) reads only the "model"
//     field of the index, and
//   * LsxModelFactory::Create constructs the model without calling Load(), so
//     framing/parsing works on the tokenizer alone,
//   * perf::ProbeGpuPerfProfile degrades to an empty profile when no CUDA
//     device is present (gpu_perf.cpp), so nothing here needs a GPU.
//
// Verified by tests/test_spike_fixture.cpp (fast, CUDA-free).

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/tokenizer.h>
#include <lsxcommon/tokenizer_arrows.h>

namespace synthmodel {

namespace fs = std::filesystem;

// Writes {"model":"<model>"} into <root>/logestix/model.index.json.
inline void WriteIndex(const fs::path &root, std::string_view model)
{
    std::ofstream idx(root / "logestix" / "model.index.json");
    idx << R"({"model":")" << model << R"("})";
}

// A fixture directory plus its LOADED tokenizer (db is usable on
// construction). Movable, not copyable.
struct LightweightModel {
    fs::path root;
    lsxcommon::ArrowTokenizerDb db;
    std::string model_name;

    LightweightModel() = default;
    LightweightModel(LightweightModel &&o) noexcept
        : root(std::move(o.root)), db(std::move(o.db)),
          model_name(std::move(o.model_name))
    {
        o.root.clear();
    }
    LightweightModel &operator=(LightweightModel &&o) noexcept
    {
        if (this != &o) {
            db = lsxcommon::ArrowTokenizerDb{};
            std::error_code ec;
            fs::remove_all(root, ec);
            root = std::move(o.root);
            db = std::move(o.db);
            model_name = std::move(o.model_name);
            o.root.clear();
        }
        return *this;
    }
    LightweightModel(const LightweightModel &) = delete;
    LightweightModel &operator=(const LightweightModel &) = delete;
    ~LightweightModel()
    {
        // Release the mmap handles before the directory goes away, otherwise
        // the Windows file lock reproduces the test_deepseek_moe_loader issue.
        db = lsxcommon::ArrowTokenizerDb{};
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    std::string path() const { return root.string(); }
};

// Builds the fixture from a tokenizer.json (a tests/data synthetic). The
// tokenizer json must be readable by BuildTokenizerArrowFiles.
inline LightweightModel MakeLightweightModel(
    const std::string &tokenizer_json, std::string_view model_name,
    std::string_view dir_name)
{
    const fs::path root = fs::temp_directory_path() / dir_name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "logestix" / "pinned");
    WriteIndex(root, model_name);
    auto st = lsxcommon::BuildTokenizerArrowFiles(
        tokenizer_json, (root / "logestix" / "pinned").string());
    LightweightModel m;
    m.root = root;
    m.model_name = std::string(model_name);
    if (!st.ok()) {
        m.db.error_msg = st.ToString();
        return m;
    }
    // Load the tokenizer so callers can encode immediately; db.error_msg stays
    // empty on success and carries the loader's reason on failure.
    if (!lsxcommon::LoadTokenizerArrows(m.path(), m.db)) {
        fs::remove_all(root, ec);
        return m;
    }
    return m;
}

// Raw-encodes text with the fixture tokenizer (BOS included, as the framing
// bridges do).
inline std::vector<int32_t> RawEncode(const lsxcommon::ArrowTokenizerDb &db,
                                      std::string_view text)
{
    std::vector<int32_t> ids(1024, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        db, text.data(), text.size(), ids.data(), &cap);
    if (n <= 0) return {};
    ids.resize(static_cast<size_t>(n));
    return ids;
}

}  // namespace synthmodel