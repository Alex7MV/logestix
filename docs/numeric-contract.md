# Numeric & Reproducibility Contract

Status: draft (operator review required)
Owner: operator (FSI product requirements); engineering enforces

This document defines, once, what numerical guarantees the engine makes,
which techniques each guarantee allows, and how each is verified. Any
kernel/feature decision is taken against this contract instead of being
re-litigated per task.

## 1. Reproducibility levels

| Level | Guarantee | Not guaranteed |
|---|---|---|
| **R3 — golden** | Same build + same input + fixed PRNG seed ⇒ bit-identical `final_logits` across ≥3 runs **and across the supported GPU-model set** (e.g. RTX 5090 and RTX 5060 Ti), and the generated token sequence equals the model's expected golden sequence. | Cross-toolchain bit-identity; bit-identity on hardware outside the supported set. |
| **R2 — deterministic** | Same build + any supported GPU model + same input ⇒ bit-identical output across runs. No golden-token requirement. | Equality with the R3 reference; cross-toolchain. |
| **R1 — tolerance** | Output deterministic per run; numeric deviation from the R3 reference bounded by a declared tolerance and quality-validated on an eval set. | Bit-identity; cheap regression detection. |

R3 is the FSI default. R2/R1 are opt-in and must never be a silent default.

### 1.1 R3 GPU-model independence — mandatory engineering requirements

R3 spans GPU models in the supported set (different SM counts, clocks, board
SKUs), so the numeric result must not depend on device properties. Mandatory for
any C-exact kernel:

1. **No heuristic/library algorithm selection.** cuBLAS/cuBLASLt heuristics may
   pick a different algorithm per device (SM count, clocks, workspace). C-exact
   paths must use our own kernels with fixed geometry, or a pinned deterministic
   algorithm verified bit-identical across every model in the supported set.
2. **Device-independent reduction trees.** Reduction/split-K geometry must not
   depend on `#SMs` / `multiProcessorCount` / clocks / occupancy APIs.
3. **No atomics in the reduce path** (or a fixed serialization order).
4. **No device-conditional numeric path.** TC-vs-scalar, arch-specific kernels,
   or capability probes must not select differently across the supported models
   (a probe may gate *perf*, never *numerics*).
5. **Deterministic launch shapes.** Grid/block sizes fixed by problem shape, not
   by device properties.

Any kernel that cannot meet these is C-fast by definition.

## 2. Numeric classes

Two implementation classes, named in code and in every kernel's header:

- **C-exact** — every kernel preserves the R3 token output and bit-identical
  logits. Allowed: bf16/fp16 activations, weight-only quantization, fused
  kernels with a strictly fixed and invariant accumulation order, exact-verify
  speculative decoding, CUDA graphs / PDL / multi-stream that do not reorder
  numerics, TMA staging. Explicitly required: deterministic GEMM execution
  paths (e.g. forced deterministic cuBLAS tiling, static workspace size
  allocation, and banned dynamic autotuning/heuristics across execution
  environments).
- **C-fast** — tolerance-class. Allowed (in addition): activation quantization
  (NVFP4/FP8/two-term), tensor-core W4A4/W8A8, arbitrary split-K / reordered
  accumulation, batch-shape-dependent kernels.

Forbidden in both unless the class explicitly changes: non-deterministic
atomics in the reduce path, unseeded stochastic ops.

## 3. Decision rule (the point of this document)

1. A proposed optimization is classified **before** implementation:
   - If it preserves the golden tokens and exact bit-wise logits ⇒ **C-exact**,
     may be default.
   - If it changes numerics in a way that can alter tokens **or drift bits due
     to accumulation reordering** ⇒ **C-fast**, must be opt-in and
     quality-validated.
2. "Preserves golden tokens" is not argued analytically — it is proven by the
   model's R3 selfcheck (golden set) and the regression guards.
3. **Toolchain clause:** any update to the compiler, CUDA Toolkit, or core build
   toolchain is treated as a code change and requires a full re-validation of
   the R3/C-exact status.
4. FSI-facing default paths are always **C-exact / R3** for modes that need it.
   `--bit-exact` (`FastMathMode::kOff`) is the explicit R3/C-exact switch; the
   default `kAuto` must resolve to C-exact for FSI models unless the operator
   opts in.
5. Enabling any C-fast path requires: (a) explicit operator opt-in, (b) a
   quality harness result, (c) a tolerance bound recorded.
6. **Workspace & Allocation Invariance:** to guarantee R3 across the supported
   GPU-model set, kernels must not alter execution paths based on runtime
   host/device memory availability. All GEMM operations must use explicit,
   fixed-size workspace bounds.

## 4. Verification obligations

- **C-exact**: the existing SelfCheck harness — golden token sequence and
  bit-identical `final_logits` across ≥3 cycles (per model, `SKIP_RETURN_CODE 4`
  without vault). Any C-exact kernel change must keep it green. **Cross-model:**
  the harness must run on every GPU model in the supported set, with a recorded
  cross-model diff; a kernel is C-exact only if it holds across the whole set.
  Exception: for speculative decoding, `final_logits` bit-identity is enforced
  only for the accepted (actually generated) tokens.
- **C-fast**: (a) device-parity vs the C-exact path within the declared
  tolerance; (b) the quality harness (A/B eval, e.g. GSM8K/MMLU subset) within
  the declared delta; (c) a perf baseline entry.
- A quality harness does not yet exist; building it is the precondition for
  promoting any C-fast path (see §6).

## 5. Worked example (why this document exists)

- Kimi W4A4/A8/two-term TC paths: **C-fast**. They are deterministic but change
  the greedy token sequence (golden flipped), so they cannot be R3/default.
  Correct classification: opt-in C-fast, blocked from default by the contract —
  not a bug to keep chasing.
- Spec-decode with exact verify: **C-exact** (accepted tokens equal greedy
  bit-for-bit). Allowed as default.
- Fused FP4→bf16 dequant with bf16 activations: **C-fast candidate** if the
  accumulation order changes (math drift breaks R3 bit-identity), or C-exact
  candidate only if the implementation strictly guarantees an invariant math
  accumulation order that passes the R3 selfcheck.

## 6. Open operator decisions

- Confirm the R3 supported GPU-model set. Current: {RTX 5090, RTX 5060 Ti} (both
  sm_120). Adding a model requires cross-model selfcheck evidence and a §1.1
  audit before the claim extends to it.
- Is a C-fast mode ever exposed to FSI customers, or is C-exact the only
  shippable class?
- Approve building the quality harness (precondition for any C-fast promotion).

## 7. Non-goals

- Bit-identity on hardware **outside** the supported GPU-model set.
- Batch-invariance (single-stream engine; no batch composition to be invariant
  to).
- Matching third-party engines' (vLLM/SGLang) numerics.
