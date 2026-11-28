#include <catch2/catch_test_macros.hpp>
#include <string_view>
#include <catch2/catch_version_macros.hpp>
#include <spdlog/version.h>
#include <absl/base/config.h>
#include <lsxcommon/common.h>
#include <lsxcommon/version.h>

// The version guard below reads the tree (`project(VERSION ...)` and the
// sources), the same way the other shape guards do.
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Apache Arrow Core Data Layers
#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/config.h>
#include <arrow/table.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

// Arrow CUDA — GPU-accelerated tensors and NVFP4 conversion
#include <arrow/gpu/cuda_api.h>

// Google FlatBuffers Zero-Copy Serialization
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/flexbuffers.h>
#include <flatbuffers/idl.h>
#include <flatbuffers/util.h>
#include <flatbuffers/verifier.h>

// ZeroMQ Core Messaging Library (C-API)
#include <zmq.h>

// simdjson Fast JSON Parser
#include <simdjson.h>
#include <httplib.h>

// NVIDIA CCCL High-Performance Primitives
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include <vector>
#include <memory>
#include <cstring>

TEST_CASE("Apache Arrow pinned version 26.0.1 matches loaded shared library", "[dependencies][arrow]") {
    // Compile-time headers and the loaded shared library must stay in lockstep
    STATIC_REQUIRE(ARROW_VERSION == 26 * 1000000 + 0 * 1000 + 1);
    REQUIRE(std::string(ARROW_VERSION_STRING) == "26.0.1");

    const auto& build_info = arrow::GetBuildInfo();
    REQUIRE(build_info.version == ARROW_VERSION);
    REQUIRE(build_info.version_major == 26);
    REQUIRE(build_info.version_minor == 0);
    REQUIRE(build_info.version_patch == 1);
    REQUIRE(build_info.version_string == "26.0.1");
}

TEST_CASE("Apache Arrow deep memory and serialization validation", "[dependencies][arrow]") {
    // 1. Verify Memory Alignment and Buffer Integrity via Primitive Builders
    arrow::FloatBuilder builder;
    arrow::Status status = builder.AppendValues({1.1f, 2.2f, 3.3f, 4.4f, 5.5f});
    REQUIRE(status.ok());

    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    REQUIRE(status.ok());

    auto float_array = std::static_pointer_cast<arrow::FloatArray>(array);
    
    // Ensure zero-copy memory access to raw underlying buffers is correct
    const float* raw_data_ptr = float_array->raw_values();
    REQUIRE(raw_data_ptr != nullptr);
    REQUIRE(raw_data_ptr[2] == 3.3f);

    // 2. Validate RecordBatch Schema Construction and Structural Integrity
    auto schema = arrow::schema({arrow::field("input_tensors", arrow::float32())});
    auto record_batch = arrow::RecordBatch::Make(schema, array->length(), {array});
    
    REQUIRE(record_batch != nullptr);
    REQUIRE(record_batch->num_columns() == 1);
    REQUIRE(record_batch->num_rows() == 5);
    
    // Full validation checks for memory leaks or overlapping internal buffers
    REQUIRE(record_batch->ValidateFull().ok());

    // 3. Verify IPC (Inter-Process Communication) Stream Serialization
    // This simulates how a B2B client streams high-throughput tensor payloads to our engine
    auto mock_sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(mock_sink.get(), schema).ValueOrDie();
    
    status = writer->WriteRecordBatch(*record_batch);
    REQUIRE(status.ok());
    status = writer->Close();
    REQUIRE(status.ok());

    auto serialized_buffer = mock_sink->Finish().ValueOrDie();
    REQUIRE(serialized_buffer->size() > 0);

    // 4. Verify IPC Deserialization Safety (Ensuring Zero Data Mutation)
    arrow::io::BufferReader mock_source(serialized_buffer);
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(&mock_source).ValueOrDie();
    
    std::shared_ptr<arrow::RecordBatch> deserialized_batch;
    status = reader->ReadNext(&deserialized_batch);
    REQUIRE(status.ok());
    REQUIRE(deserialized_batch != nullptr);
    REQUIRE(deserialized_batch->num_rows() == 5);
    
    auto deserialized_array = std::static_pointer_cast<arrow::FloatArray>(deserialized_batch->column(0));
    REQUIRE(deserialized_array->Value(4) == 5.5f);
}

TEST_CASE("NVIDIA CCCL template compilation and algorithmic sanity", "[dependencies][cccl]") {
    // 1. Verify Thrust Host Vector and Sequence Math
    const size_t element_count = 1000;
    thrust::host_vector<float> input_buffer(element_count);
    
    // Fill with sequential floating-point increments (0.0f, 1.0f, 2.0f ... 999.0f)
    thrust::sequence(input_buffer.begin(), input_buffer.end(), 0.0f, 1.0f);
    REQUIRE(input_buffer[500] == 500.0f);

    // 2. Test Parallel Reduction Core Execution Logic (Summing elements)
    // Formula: (n * (n - 1)) / 2 -> (1000 * 999) / 2 = 499500
    float target_reduction_sum = thrust::reduce(input_buffer.begin(), input_buffer.end(), 0.0f, thrust::plus<float>());
    REQUIRE(target_reduction_sum == 499500.0f);

    // 3. Test Lazy Execution Modifiers via Counting Iterators and Custom Transformations
    // This guarantees that our thread-safe scheduling layouts map properly
    thrust::host_vector<float> output_buffer(element_count);
    thrust::transform(input_buffer.begin(), input_buffer.end(), output_buffer.begin(), 
                      [] (float val) { return val * 2.0f; });
                      
    REQUIRE(output_buffer[500] == 1000.0f);

    // 4. Validate libcu++ Type Traits and Complex Numerical Structs in C++17
    // This layout proves that the MSVC/GCC compiler matches NVIDIA's alignment rules
    cuda::std::complex<double> complex_tensor_a(4.0, 3.0);
    cuda::std::complex<double> complex_tensor_b(2.0, -1.0);
    
    // (4 + 3i) * (2 - 1i) = 8 - 4i + 6i - 3i^2 = 11 + 2i
    auto complex_product = complex_tensor_a * complex_tensor_b;
    
    REQUIRE(complex_product.real() == 11.0);
    REQUIRE(complex_product.imag() == 2.0);
    
    // Compile-time check to ensure libcu++ types are recognized as safe trivially copyable types
    STATIC_REQUIRE(cuda::std::is_trivially_copyable<cuda::std::complex<double>>::value);
}

TEST_CASE("Arrow CUDA device manager and context API compilation", "[dependencies][arrow][cuda]") {
    auto maybe_mgr = arrow::cuda::CudaDeviceManager::Instance();
    REQUIRE(maybe_mgr.ok());

    auto mgr = *maybe_mgr;
    int num_devices = mgr->num_devices();

    // Context creation. If device exists, verify it; otherwise skip gracefully.
    if (num_devices > 0) {
        auto maybe_ctx = mgr->GetContext(0);
        REQUIRE(maybe_ctx.ok());

        auto ctx = *maybe_ctx;
        auto dev = ctx->device();
        REQUIRE(dev != nullptr);
        REQUIRE(dev->total_memory() > 0);

        // CUDA buffer round-trip (allocate → CopyFromHost → CopyToHost → compare)
        constexpr int64_t buf_size = 1024;
        auto maybe_dev_buf = ctx->Allocate(buf_size);
        REQUIRE(maybe_dev_buf.ok());

        std::unique_ptr<arrow::cuda::CudaBuffer> dev_buf = std::move(maybe_dev_buf).ValueOrDie();
        REQUIRE(dev_buf->size() == buf_size);

        // Fill host data and copy to device
        std::vector<uint8_t> host_src(buf_size, 0x42);
        auto st = dev_buf->CopyFromHost(0, host_src.data(), buf_size);
        REQUIRE(st.ok());

        // Read back from device
        std::vector<uint8_t> host_dst(buf_size, 0);
        st = dev_buf->CopyToHost(0, buf_size, host_dst.data());
        REQUIRE(st.ok());

        REQUIRE(host_dst[0] == 0x42);
        REQUIRE(host_dst[buf_size - 1] == 0x42);

        // bytes_allocated should have increased
        REQUIRE(ctx->bytes_allocated() >= buf_size);
    }
}

TEST_CASE("FlexBuffers dynamic map and vector round-trip", "[dependencies][flatbuffers]") {
    flexbuffers::Builder flex_builder(1024);

    // Build a map emulating tensor metadata
    flex_builder.Map([&]() {
        flex_builder.String("model", "logestix_transformer");
        flex_builder.Int("version", 3);
        flex_builder.Vector("shape", [&]() {
            flex_builder.UInt(1);
            flex_builder.UInt(128);
            flex_builder.UInt(768);
        });
        flex_builder.Vector("data", [&]() {
            flex_builder.Double(0.5);
            flex_builder.Double(1.5);
            flex_builder.Double(2.5);
            flex_builder.Double(3.5);
        });
    });
    flex_builder.Finish();

    auto buf = flex_builder.GetBuffer();
    auto root = flexbuffers::GetRoot(buf);
    REQUIRE(root.IsMap());

    auto map = root.AsMap();
    REQUIRE(map["model"].AsString().str() == "logestix_transformer");
    REQUIRE(map["version"].AsInt64() == 3);

    auto shape = map["shape"].AsVector();
    REQUIRE(shape.size() == 3);
    REQUIRE(shape[0].AsUInt64() == 1);
    REQUIRE(shape[1].AsUInt64() == 128);
    REQUIRE(shape[2].AsUInt64() == 768);

    auto data = map["data"].AsVector();
    REQUIRE(data.size() == 4);
    REQUIRE(data[0].AsDouble() == 0.5);
    REQUIRE(data[3].AsDouble() == 3.5);

    REQUIRE(flexbuffers::VerifyBuffer(buf.data(), buf.size()));
}

TEST_CASE("FlatBuffers IDL schema parsing and buffer generation", "[dependencies][flatbuffers]") {
    const char* schema = R"(
table InferenceConfig {
    model_name: string;
    batch_size: int;
    use_fp16: bool = true;
    priority: float = 1.0;
}
root_type InferenceConfig;
)";

    flatbuffers::Parser parser;
    REQUIRE(parser.Parse(schema));

    const char* json = R"({ model_name: "bert-base", batch_size: 8, use_fp16: false, priority: 2.5 })";
    REQUIRE(parser.Parse(json));

    auto* buf = parser.builder_.GetBufferPointer();
    auto size = parser.builder_.GetSize();
    REQUIRE(buf != nullptr);
    REQUIRE(size > 0);

    // Validate buffer alignment and minimum size
    REQUIRE(reinterpret_cast<uintptr_t>(buf) % alignof(flatbuffers::Table) == 0);
    REQUIRE(size >= sizeof(flatbuffers::uoffset_t));

    // The root table offset is stored at the start of the buffer (little-endian uoffset_t)
    auto root_offset = flatbuffers::EndianScalar(
        *reinterpret_cast<const flatbuffers::uoffset_t*>(buf));
    REQUIRE(root_offset > 0);
    REQUIRE(root_offset < size);

    // Re-parse default values
    flatbuffers::Parser default_parser;
    REQUIRE(default_parser.Parse(schema));
    REQUIRE(default_parser.Parse(R"({ model_name: "default", batch_size: 1 })"));

    auto* default_buf = default_parser.builder_.GetBufferPointer();
    auto default_size = default_parser.builder_.GetSize();
    REQUIRE(default_buf != nullptr);
    REQUIRE(default_size > 0);
    REQUIRE(default_size >= sizeof(flatbuffers::uoffset_t));
}

TEST_CASE("FlatBuffers type traits and manual buffer construction", "[dependencies][flatbuffers]") {
    STATIC_REQUIRE(flatbuffers::is_same<float, float>::value);
    STATIC_REQUIRE_FALSE(flatbuffers::is_same<float, int>::value);

    REQUIRE(flatbuffers::EndianScalar(42) == 42);
    REQUIRE(flatbuffers::EndianScalar(static_cast<int16_t>(0x0102)) == 0x0102);

    flatbuffers::FlatBufferBuilder builder(128);
    REQUIRE(builder.GetSize() == 0);

    auto str_off = builder.CreateString("flatbuffer string");
    builder.Finish(str_off);

    auto* buf = builder.GetBufferPointer();
    auto size = builder.GetSize();
    REQUIRE(buf != nullptr);
    REQUIRE(size > 0);
    REQUIRE(reinterpret_cast<uintptr_t>(buf) % alignof(flatbuffers::Table) == 0);

    // Verify the buffer contains our string via raw access
    auto root = flatbuffers::GetRoot<flatbuffers::String>(buf);
    REQUIRE(root != nullptr);
    REQUIRE(root->str() == "flatbuffer string");
    REQUIRE(root->size() == 17);

    // Verify buffer validity with the generic flatbuffer verifier
    flatbuffers::Verifier verifier(buf, size);
    REQUIRE(verifier.VerifyString(root));
}

TEST_CASE("ZeroMQ version and context lifecycle", "[dependencies][zeromq]") {
    int major, minor, patch;
    zmq_version(&major, &minor, &patch);
    REQUIRE(major == 4);
    REQUIRE(minor == 3);
    REQUIRE(patch == 5);

    void* context = zmq_ctx_new();
    REQUIRE(context != nullptr);

    int io_threads = zmq_ctx_get(context, ZMQ_IO_THREADS);
    REQUIRE(io_threads > 0);

    int rc = zmq_ctx_set(context, ZMQ_IO_THREADS, 2);
    REQUIRE(rc == 0);
    REQUIRE(zmq_ctx_get(context, ZMQ_IO_THREADS) == 2);

    rc = zmq_ctx_destroy(context);
    REQUIRE(rc == 0);
}

TEST_CASE("ZeroMQ in-process PAIR socket send and receive", "[dependencies][zeromq]") {
    void* context = zmq_ctx_new();
    REQUIRE(context != nullptr);

    void* push_socket = zmq_socket(context, ZMQ_PAIR);
    REQUIRE(push_socket != nullptr);

    void* pull_socket = zmq_socket(context, ZMQ_PAIR);
    REQUIRE(pull_socket != nullptr);

    int rc = zmq_bind(push_socket, "inproc://test_pair");
    REQUIRE(rc == 0);

    rc = zmq_connect(pull_socket, "inproc://test_pair");
    REQUIRE(rc == 0);

    const char* message = "ping_from_zmq";
    int send_rc = zmq_send(push_socket, message, 13, 0);
    REQUIRE(send_rc == 13);

    char buffer[32] = {};
    int recv_rc = zmq_recv(pull_socket, buffer, sizeof(buffer) - 1, 0);
    REQUIRE(recv_rc == 13);
    REQUIRE(std::strcmp(buffer, "ping_from_zmq") == 0);

    rc = zmq_close(push_socket);
    REQUIRE(rc == 0);

    rc = zmq_close(pull_socket);
    REQUIRE(rc == 0);

    rc = zmq_ctx_destroy(context);
    REQUIRE(rc == 0);
}

TEST_CASE("simdjson parse simple object and access fields", "[dependencies][simdjson]") {
    auto json = R"({"name": "inference_model", "version": 3, "precision": 1.5})"_padded;

    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    simdjson::error_code error = parser.iterate(json).get(doc);
    REQUIRE(error == simdjson::SUCCESS);

    std::string_view name;
    error = doc["name"].get_string().get(name);
    REQUIRE(error == simdjson::SUCCESS);
    REQUIRE(name == "inference_model");

    int64_t version;
    error = doc["version"].get_int64().get(version);
    REQUIRE(error == simdjson::SUCCESS);
    REQUIRE(version == 3);

    double precision;
    error = doc["precision"].get_double().get(precision);
    REQUIRE(error == simdjson::SUCCESS);
    REQUIRE(precision == 1.5);
}

TEST_CASE("simdjson parse nested array and error handling", "[dependencies][simdjson]") {
    auto json = R"({"layers": [{"type": "dense", "units": 128}, {"type": "relu"}]})"_padded;

    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    REQUIRE(parser.iterate(json).get(doc) == simdjson::SUCCESS);

    simdjson::ondemand::array layers;
    REQUIRE(doc["layers"].get_array().get(layers) == simdjson::SUCCESS);

    int64_t count = 0;
    for (auto elem : layers) {
        simdjson::ondemand::object obj;
        REQUIRE(elem.get_object().get(obj) == simdjson::SUCCESS);
        ++count;
    }
    REQUIRE(count == 2);

    // Missing key → expected error
    auto missing_json = R"({"a": 1})"_padded;
    simdjson::ondemand::document miss_doc;
    REQUIRE(parser.iterate(missing_json).get(miss_doc) == simdjson::SUCCESS);
    REQUIRE(miss_doc["nonexistent"].error() == simdjson::NO_SUCH_FIELD);
}

TEST_CASE("pinned dependency versions", "[dependencies][versions]") {
    // spdlog
    STATIC_REQUIRE(SPDLOG_VERSION == 1 * 10000 + 17 * 100 + 0);

    // simdjson
    STATIC_REQUIRE(simdjson::SIMDJSON_VERSION_MAJOR == 5);
    STATIC_REQUIRE(simdjson::SIMDJSON_VERSION_MINOR == 2);
    STATIC_REQUIRE(simdjson::SIMDJSON_VERSION_REVISION == 0);

    // Catch2
    STATIC_REQUIRE(CATCH_VERSION_MAJOR == 3);
    STATIC_REQUIRE(CATCH_VERSION_MINOR == 16);
    STATIC_REQUIRE(CATCH_VERSION_PATCH == 1);

    // cpp-httplib (FetchContent, host-only HTTP). Pin enforced here.
    STATIC_REQUIRE(std::string_view(CPPHTTPLIB_VERSION) == "0.66.0");

    // NVIDIA CCCL — no longer vendored: it comes from the CUDA toolkit bundle
    // (<toolkit-include>/cccl). The validated band is defined once in the root
    // CMakeLists.txt and shared here (and with the configure-time guard) as
    // LSX_CCCL_VERSION_{MIN,MAX}; a toolkit whose CCCL leaves the band fails at
    // configure, and this pins the compiled TU to the same window.
    STATIC_REQUIRE(CCCL_VERSION >= LSX_CCCL_VERSION_MIN);
    STATIC_REQUIRE(CCCL_VERSION < LSX_CCCL_VERSION_MAX);

    // abseil-cpp (FetchContent). The LTS macros exist only on LTS tags, so a
    // non-LTS pin (e.g. a master commit) is a hard error here rather than a
    // silently unguarded version.
#if defined(ABSL_LTS_RELEASE_VERSION)
    STATIC_REQUIRE(ABSL_LTS_RELEASE_VERSION == 20260817);
    STATIC_REQUIRE(ABSL_LTS_RELEASE_PATCH_LEVEL == 0);
#else
    STATIC_REQUIRE(false &&
                   "abseil LTS macros missing: the FetchContent pin is not an "
                   "LTS tag");
#endif
}

TEST_CASE("lsxcommon common module version and GPU info", "[dependencies][lsxcommon]") {
    lsxcommon::version();
    lsxcommon::log_gpu_info();
}

// ── The reported version is GENERATED, and both halves are checked ──────────
//
// `version()` used to print a hand-written "logestix v0.1.0" while
// `libs/lsxcommon/include/lsxcommon/version.h.in` was being configured from
// `project(VERSION ...)` and included by nobody. So the project shipped tags
// v1.0.0 / v1.0.1 and still announced itself as 0.1.0, and no test could see it
// because both the header and the literal were plausible on their own.
//
// Two assertions, because the failure has two halves:
//   * the generated header and `project(VERSION ...)` agree — a bump that edits
//     one side only is the mistake this release made;
//   * no source file spells a version literal again — that is how the drift
//     started, and a literal compiles fine.
TEST_CASE("the reported version comes from project(VERSION), not a literal",
          "[dependencies][lsxcommon][version]") {
    namespace fs = std::filesystem;

    const std::string generated = lsxcommon::version_string();

    // Semver shape: three numeric components, nothing else. A stale or empty
    // configure_file result fails here rather than printing nonsense.
    int components = 0;
    bool component_empty = true;
    for (const char c : generated) {
        if (c >= '0' && c <= '9') {
            component_empty = false;
        } else if (c == '.') {
            REQUIRE_FALSE(component_empty);
            component_empty = true;
            ++components;
        } else {
            FAIL("version string has a non-semver character: " << generated);
        }
    }
    CHECK_FALSE(component_empty);
    CHECK(components == 2);

    // The macros the header exposes and the string must be one value.
    CHECK(generated == std::to_string(LSXCOMMON_VERSION_MAJOR) + "." +
                            std::to_string(LSXCOMMON_VERSION_MINOR) + "." +
                            std::to_string(LSXCOMMON_VERSION_PATCH));

    // Half one: the root CMakeLists must declare the same version. Read as text
    // because the value is a configure-time constant, not a runtime one.
    const fs::path root{LSX_SOURCE_DIR};
    const fs::path cmake = root / "CMakeLists.txt";
    INFO("reading " << cmake.string());
    REQUIRE(fs::is_regular_file(cmake));

    std::ifstream in(cmake);
    REQUIRE(in.good());
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    // `project(logestix\n    VERSION <x.y.z>` — take the token after VERSION
    // inside the project() call, not the first "VERSION" in the file (the
    // include of CMakePresets-era helpers comes first).
    const std::size_t project_at = text.find("project(");
    REQUIRE(project_at != std::string::npos);
    const std::size_t version_at = text.find("VERSION", project_at);
    REQUIRE(version_at != std::string::npos);
    std::size_t value_at = version_at + std::strlen("VERSION");
    while (value_at < text.size() &&
           (text[value_at] == ' ' || text[value_at] == '\t'))
        ++value_at;
    const std::size_t value_end = text.find_first_of(" \t\r\n)", value_at);
    REQUIRE(value_end != std::string::npos);
    const std::string declared = text.substr(value_at, value_end - value_at);

    INFO("CMakeLists declares " << declared << ", the built header says "
                                 << generated);
    CHECK(declared == generated);

    // Half two: no hardcoded version literal in the sources. Matched as
    // "logestix v<digit>" so a comment or an unrelated 1.2.3 in a test is not
    // a violation, while the exact historical spelling is.
    std::vector<fs::path> offenders;
    for (const char *dir : {"libs", "apps"}) {
        const fs::path base = root / dir;
        if (!fs::is_directory(base))
            continue;
        for (const auto &entry :
             fs::recursive_directory_iterator(base)) {
            if (!entry.is_regular_file())
                continue;
            const auto ext = entry.path().extension().string();
            if (ext != ".cpp" && ext != ".cu" && ext != ".h" &&
                ext != ".cuh")
                continue;
            std::ifstream f(entry.path());
            std::stringstream sb;
            sb << f.rdbuf();
            const std::string src = sb.str();
            const std::size_t at = src.find("logestix v");
            if (at != std::string::npos && at + 10 < src.size() &&
                src[at + 10] >= '0' && src[at + 10] <= '9')
                offenders.push_back(entry.path().lexically_relative(root));
        }
    }
    for (const auto &o : offenders)
        FAIL_CHECK("a hardcoded version literal: " << o.string()
                    << " — use LSXCOMMON_VERSION_STRING");
    CHECK(offenders.empty());
}
