// Model-neutral dense-decoder kernels. Bodies are the Granite-4.2-30B runtime
// kernels moved verbatim; see tools/dedup_provenance.py for the proof.

#include "lsxcommon/lsx_dense_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"
