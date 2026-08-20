#pragma once

#include <cstdint>
#include <string>

namespace lsxcommon {
namespace gpu {

struct BenchmarkResult {
    double total_time_ms = 0.0;
    double avg_bus_latency_ms = 0.0;
    double tokens_per_sec = 0.0;
    int num_layers = 0;
    int num_tokens = 0;
    int64_t total_bytes_transferred = 0;
    std::uint64_t resident_bytes = 0;   // peak resident weight set during the run
    bool ok = false;
    std::string error_msg;
};

BenchmarkResult RunBenchmark(const std::string &model_path, int num_tokens = 2048);

} // namespace gpu
} // namespace lsxcommon