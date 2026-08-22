// TDD — zero-copy Arrow IPC view loading (fast-load root cause fix).
// The owning UnpackBatch paths copy every byte out of the mmap; the view API
// must hand out pointers INTO the mmap-backed buffers with the batch lifetime
// pinned by keepalive, so the fast loaders upload straight from page cache.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/schema.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;

std::vector<lsxcommon::TensorRecord> SampleTensors()
{
    std::vector<lsxcommon::TensorRecord> tensors;
    lsxcommon::TensorRecord big;
    big.name = "model.layers.0.self_attn.q_proj.weight";
    big.dtype = "BF16";
    big.shape = {64, 32};
    big.data.resize(64 * 32 * 2);
    for (size_t i = 0; i < big.data.size(); ++i)
        big.data[i] = static_cast<uint8_t>(i * 7 + 3);
    tensors.push_back(std::move(big));

    lsxcommon::TensorRecord norm;
    norm.name = "model.layers.0.input_layernorm.weight";
    norm.dtype = "F32";
    norm.shape = {16};
    norm.data.resize(16 * 4);
    for (size_t i = 0; i < norm.data.size(); ++i)
        norm.data[i] = static_cast<uint8_t>(i * 11 + 1);
    tensors.push_back(norm);
    return tensors;
}

std::vector<lsxcommon::DenseNvfp4Record> SampleNvfp4()
{
    std::vector<lsxcommon::DenseNvfp4Record> recs;
    lsxcommon::DenseNvfp4Record rec;
    rec.name = "model.layers.0.mlp.down_proj.weight";
    rec.dtype = "E2M1";
    rec.shape = {128, 256};
    rec.packed.resize(128 * 256 / 2);
    for (size_t i = 0; i < rec.packed.size(); ++i)
        rec.packed[i] = static_cast<uint8_t>(i * 13 + 5);
    rec.scales.resize((128 * 256) / 16);
    for (size_t i = 0; i < rec.scales.size(); ++i)
        rec.scales[i] = static_cast<uint8_t>(i * 29 + 17);
    rec.global_scale = 0.03125f;
    recs.push_back(std::move(rec));
    return recs;
}

std::vector<lsxcommon::PackedExpert> SampleSharedExperts()
{
    std::vector<lsxcommon::PackedExpert> layers(2);
    for (int l = 0; l < 2; ++l) {
        auto &e = layers[static_cast<size_t>(l)];
        e.w1_packed.assign({1, 2, 3, 4});
        e.w2_packed.assign({5, 6, 7, 8});
        e.w3_packed.assign({9, 10, 11, 12});
        e.w1_scales.assign({100});
        e.w2_scales.assign({101, 102});
        e.w3_scales.assign({103});
        e.w1_global_scale = 0.5f + static_cast<float>(l);
        e.w2_global_scale = 0.25f;
        e.w3_global_scale = 0.125f;
    }
    return layers;
}

fs::path WritePinnedFile(const fs::path &dir,
                         const std::vector<lsxcommon::TensorRecord> &tensors)
{
    auto batch = lsxcommon::PinnedSchema::MakeBatch(tensors);
    REQUIRE(batch.ok());
    auto path = dir / "sample_pinned.arrow";
    auto st = lsxcommon::WriteRecordBatchToFile(*batch, path);
    REQUIRE(st.ok());
    return path;
}

fs::path WriteNvfp4File(const fs::path &dir,
                        const std::vector<lsxcommon::DenseNvfp4Record> &recs)
{
    auto batch = lsxcommon::DenseNvfp4Schema::MakeBatch(recs);
    REQUIRE(batch.ok());
    auto path = dir / "sample_nvfp4.arrow";
    auto st = lsxcommon::WriteRecordBatchToFile(*batch, path);
    REQUIRE(st.ok());
    return path;
}

} // namespace

TEST_CASE("PinnedSchema zero-copy views round-trip and keepalive",
          "[schema][views][pinned]")
{
    auto tmp = fs::temp_directory_path() /
               ("lsx_schema_views_pinned_" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    const auto tensors = SampleTensors();
    const auto path = WritePinnedFile(tmp, tensors);

    auto views_res = lsxcommon::LoadTensorBatchViews(path);
    REQUIRE(views_res.ok());
    auto views = std::move(*views_res);

    REQUIRE(views.tensors.size() == tensors.size());
    // Keepalive must own the mmap-backed buffers: no other reference exists
    // at this point, so any readable byte proves the view lifetime contract.
    int touched = 0;
    for (size_t i = 0; i < views.tensors.size(); ++i) {
        const auto &v = views.tensors[i];
        const auto &ref = tensors[i];
        REQUIRE(v.name == ref.name);
        REQUIRE(v.dtype == ref.dtype);
        REQUIRE(v.rank == static_cast<int64_t>(ref.shape.size()));
        REQUIRE(std::memcmp(v.shape, ref.shape.data(),
                            ref.shape.size() * sizeof(int64_t)) == 0);
        REQUIRE(v.bytes == ref.data.size());
        REQUIRE(v.data != nullptr);
        REQUIRE(std::memcmp(v.data, ref.data.data(), ref.data.size()) == 0);
        touched += static_cast<int>(v.bytes);
    }
    REQUIRE(touched > 0);

    // Views stay valid while only the views struct itself is alive.
    auto bytes_copy = views.tensors[0].bytes;
    auto *data_ptr = views.tensors[0].data;
    auto moved = std::move(views);
    REQUIRE(moved.keepalive != nullptr);
    REQUIRE(std::memcmp(data_ptr, tensors[0].data.data(), bytes_copy) == 0);

    fs::remove_all(tmp);
}

TEST_CASE("DenseNvfp4Schema zero-copy views round-trip and keepalive",
          "[schema][views][nvfp4]")
{
    auto tmp = fs::temp_directory_path() /
               ("lsx_schema_views_nvfp4_" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    const auto recs = SampleNvfp4();
    const auto path = WriteNvfp4File(tmp, recs);

    auto views_res = lsxcommon::LoadDenseNvfp4BatchViews(path);
    REQUIRE(views_res.ok());
    auto views = std::move(*views_res);

    REQUIRE(views.records.size() == recs.size());
    for (size_t i = 0; i < views.records.size(); ++i) {
        const auto &v = views.records[i];
        const auto &ref = recs[i];
        REQUIRE(v.name == ref.name);
        REQUIRE(v.dtype == ref.dtype);
        REQUIRE(v.rank == static_cast<int64_t>(ref.shape.size()));
        REQUIRE(std::memcmp(v.shape, ref.shape.data(),
                            ref.shape.size() * sizeof(int64_t)) == 0);
        REQUIRE(v.packed_bytes == static_cast<int64_t>(ref.packed.size()));
        REQUIRE(v.scale_bytes == static_cast<int64_t>(ref.scales.size()));
        REQUIRE(v.global_scale == ref.global_scale);
        REQUIRE(std::memcmp(v.packed, ref.packed.data(), ref.packed.size()) == 0);
        REQUIRE(std::memcmp(v.scales, ref.scales.data(), ref.scales.size()) == 0);
    }

    auto *packed_ptr = views.records[0].packed;
    auto packed_bytes = views.records[0].packed_bytes;
    auto moved = std::move(views);
    REQUIRE(moved.keepalive != nullptr);
    REQUIRE(std::memcmp(packed_ptr, recs[0].packed.data(), packed_bytes) == 0);

    fs::remove_all(tmp);
}

TEST_CASE("SharedExpertSchema zero-copy views round-trip and keepalive",
          "[schema][views][shared-experts]")
{
    auto tmp = fs::temp_directory_path() /
               ("lsx_schema_views_shared_" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    const auto layers = SampleSharedExperts();
    auto batch = lsxcommon::SharedExpertSchema::MakeBatch(layers);
    REQUIRE(batch.ok());
    const auto path = tmp / "shared_experts.arrow";
    REQUIRE(lsxcommon::WriteRecordBatchToFile(*batch, path).ok());

    auto views_res = lsxcommon::LoadSharedExpertBatchViews(path);
    REQUIRE(views_res.ok());
    auto views = std::move(*views_res);

    REQUIRE(views.layers.size() == layers.size());
    for (size_t i = 0; i < views.layers.size(); ++i) {
        const auto &v = views.layers[i];
        const auto &ref = layers[i];
        REQUIRE(std::memcmp(v.w1_packed, ref.w1_packed.data(),
                            ref.w1_packed.size()) == 0);
        REQUIRE(v.w1_packed_bytes == static_cast<int64_t>(ref.w1_packed.size()));
        REQUIRE(std::memcmp(v.w2_packed, ref.w2_packed.data(),
                            ref.w2_packed.size()) == 0);
        REQUIRE(v.w2_packed_bytes == static_cast<int64_t>(ref.w2_packed.size()));
        REQUIRE(std::memcmp(v.w3_packed, ref.w3_packed.data(),
                            ref.w3_packed.size()) == 0);
        REQUIRE(v.w3_packed_bytes == static_cast<int64_t>(ref.w3_packed.size()));
        REQUIRE(std::memcmp(v.w1_scales, ref.w1_scales.data(),
                            ref.w1_scales.size()) == 0);
        REQUIRE(v.w1_scale_bytes == static_cast<int64_t>(ref.w1_scales.size()));
        REQUIRE(std::memcmp(v.w2_scales, ref.w2_scales.data(),
                            ref.w2_scales.size()) == 0);
        REQUIRE(v.w2_scale_bytes == static_cast<int64_t>(ref.w2_scales.size()));
        REQUIRE(std::memcmp(v.w3_scales, ref.w3_scales.data(),
                            ref.w3_scales.size()) == 0);
        REQUIRE(v.w3_scale_bytes == static_cast<int64_t>(ref.w3_scales.size()));
        REQUIRE(v.w1_global_scale == ref.w1_global_scale);
        REQUIRE(v.w2_global_scale == ref.w2_global_scale);
        REQUIRE(v.w3_global_scale == ref.w3_global_scale);
    }

    // Views stay valid while only the views struct itself is alive.
    auto *p1 = views.layers[0].w1_packed;
    auto n1 = views.layers[0].w1_packed_bytes;
    auto moved = std::move(views);
    REQUIRE(moved.keepalive != nullptr);
    REQUIRE(std::memcmp(p1, layers[0].w1_packed.data(), n1) == 0);

    fs::remove_all(tmp);
}

TEST_CASE("View loaders report missing files as errors",
          "[schema][views][errors]")
{
    auto missing = fs::temp_directory_path() / "lsx_no_such_file.arrow";
    auto r1 = lsxcommon::LoadTensorBatchViews(missing);
    REQUIRE_FALSE(r1.ok());
    auto r2 = lsxcommon::LoadDenseNvfp4BatchViews(missing);
    REQUIRE_FALSE(r2.ok());
    auto r3 = lsxcommon::LoadSharedExpertBatchViews(missing);
    REQUIRE_FALSE(r3.ok());
}
