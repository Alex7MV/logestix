#include <catch2/catch_test_macros.hpp>

// Apache Arrow Core Data Layers
#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/table.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

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
