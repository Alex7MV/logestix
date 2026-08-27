#include "lsxcommon/benchmark_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/converter.h"
#include "deepseek_moe_load.cuh"
#include "nvfp4_codec.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <arrow/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include "lsxcommon/log.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

namespace lsxcommon {
namespace gpu {

// The legacy ping-pong expert benchmark was superseded by IModel::Benchmark
// (see benchmark_gpu.cpp shim). Its NVFP4 dequant kernel and the dead extern
// declarations in deepseek_moe_inference_pipeline.cu / deepseek_moe_inference_moe.cu
// were removed in the Phase-A unification; elementwise NVFP4 dequant lives in
// lsx_kernels_quant.cuh (Nvfp4DequantKernel).

} // namespace gpu
} // namespace lsxcommon