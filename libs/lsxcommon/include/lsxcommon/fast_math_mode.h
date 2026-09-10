#pragma once

// ── Tri-state fast-math routing mode ────────────────────────────────────────
// Tiny dependency-free header so both host headers (model.h) and .cu-reachable
// headers (lsx_inference_nvfp4.h) can share the enum without pulling absl or
// CUDA into device translation units.
//
// kAuto = the GPU performance policy decides (committed base / heuristic /
//         calibration — see gpu_perf.h); globals were applied at engine init.
// kOn   = force the TC mxf4nvf4 fast path (+ streaming GEMV) for the scope.
// kOff  = force the bit-exact scalar path for the scope (numeric-class tests).

namespace lsxcommon {

enum class FastMathMode { kAuto, kOn, kOff };

} // namespace lsxcommon
