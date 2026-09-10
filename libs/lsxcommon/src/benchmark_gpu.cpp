#include "lsxcommon/benchmark_gpu.h"

#include "lsxcommon/model_factory.h"
#include "lsxcommon/model.h"

namespace lsxcommon {
namespace gpu {

BenchmarkResult RunBenchmark(const std::string &model_path, int num_tokens) {
    BenchmarkResult r;
    r.num_tokens = num_tokens;
    ModelInitConfig cfg;
    cfg.model_path = model_path;
    cfg.max_gen_tokens = num_tokens;
    auto model = LsxModelFactory::Create(cfg);
    if (!model) {
        r.error_msg = "failed to create model for " + model_path;
        return r;
    }
    if (!model->Load()) {
        r.error_msg = "failed to load model for " + model_path;
        return r;
    }
    r = model->Benchmark(num_tokens);
    model->Unload();
    return r;
}

} // namespace gpu
} // namespace lsxcommon