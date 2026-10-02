// lsxcommon::diag — the Arrow memory/device diagnostic helpers (host-only).
//
// diagnostics.h is header-only and every entry point except MemoryManagerDesc
// writes to the log, so what is worth pinning here is the descriptor's string
// contract plus the fact that the validators stay silent (and crash-free) on a
// well-formed batch. Before this suite the header had no test at all, which
// matters because the strings it produces are what an operator reads when an
// Arrow device/CPU buffer mismatch shows up in the log.
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>

#include <cstring>
#include <vector>
#include <string>

#include <lsxcommon/diagnostics.h>

namespace diag = lsxcommon::diag;

TEST_CASE("diag: a null memory manager is reported as nullptr", "[diag]")
{
    CHECK(diag::MemoryManagerDesc(nullptr) == "nullptr");
}

TEST_CASE("diag: the default CPU memory manager is described as a CPU device",
          "[diag]")
{
    const auto desc = diag::MemoryManagerDesc(arrow::default_cpu_memory_manager());
    CHECK(desc != "nullptr");
    CHECK(desc != "no-device");
    // DeviceAllocationTypeToCStr(CPU) is "CPU"; the parentheses carry the
    // pool's own description.
    CHECK(desc.find("CPU") != std::string::npos);
    CHECK(desc.find('(') != std::string::npos);
    CHECK(desc.find(')') != std::string::npos);
}

TEST_CASE("diag: validators accept a well-formed CPU batch without warning",
          "[diag]")
{
    // ValidateBatchBuffers walks every buffer of every column; on a CPU pool
    // nothing is a mismatch, so this must complete quietly.
    auto pool = arrow::default_memory_pool();
    auto buf = arrow::AllocateBuffer(64, pool).ValueOrDie();

    arrow::Int32Builder builder;
    REQUIRE(builder.AppendValues({1, 2, 3, 4}).ok());
    auto array = builder.Finish().ValueOrDie();
    std::vector<std::shared_ptr<arrow::Array>> columns{array};
    auto batch = arrow::RecordBatch::Make(
        arrow::schema({arrow::field("x", arrow::int32())}), 4, columns);

    REQUIRE(batch->num_columns() == 1);
    REQUIRE(batch->num_rows() == 4);

    // These only log; the contract under test is that they return normally.
    diag::ValidateCpuBuffer(*buf, "test");
    diag::ValidateBatchBuffers(*batch, "test");
    diag::LogBuffer("primary", *buf);
    diag::LogMemoryPoolState("test");
    diag::LogArrowSystemState("test");
    diag::LogToStderr("test");
    CHECK(true);
}

TEST_CASE("diag: an empty batch has no buffers to validate", "[diag]")
{
    auto schema = arrow::schema({});
    std::vector<std::shared_ptr<arrow::Array>> columns;
    auto batch = arrow::RecordBatch::Make(schema, 0, columns);
    REQUIRE(batch->num_columns() == 0);
    diag::ValidateBatchBuffers(*batch, "empty");
    CHECK(batch->num_rows() == 0);
}