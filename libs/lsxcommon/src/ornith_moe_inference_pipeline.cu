// MoE generation driver — the body is the neutral lsxcommon::moe template
// (lsx_moe_inference_pipeline.cu, dedup phase 6: the ornith pipeline was a
// 99.4% rename-copy of this one). The public family API lives in the header
// as inline delegates; the explicit instantiations are in the neutral TU.
#include "lsxcommon/ornith_moe_inference_pipeline.h"
