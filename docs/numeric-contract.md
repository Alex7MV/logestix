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
| **R3 — golden, device-independent** (FSI default) | Same build + same input ⇒ bit-identical `final_logits` across ≥3 runs **and across any supported device instance of the same GPU model class** (e.g. any RTX 5090), and the generated token sequence equals the model's expected golden sequence. | Cross-**model** bit-identity (sm_120 vs sm_121 etc.) unless a separate golden baseline exists (see R4). |
| **R4 — golden, cross-model** | As R3 but also across different GPU models in the supported set. Requires per-model golden baselines and a device-independent reduction core; practically forces fixed-point/exact or algorithm-pinned paths. |
| **R2 — deterministic** | Same build + same GPU ⇒ bit-identical output across runs. No golden-token requirement. | Equality with the R3 reference; device independence. |
| **R1 — tolerance** | Output deterministic per run; numeric deviation from the R3 reference bounded by a declared tolerance and quality-validated. | Bit-identity; cheap regression detection. |

R3 is the FSI default. R2/R1 are opt-in and must never be a silent default.

### 1.1 R3 device independence — engineering requirements

Because R3 spans device instances, the following are **mandatory** for any
C-exact kernel:

1. **No heuristic/library algorithm selection.** cuBLAS/cuBLASLt may pick a
   different algorithm per device (SM count, clocks, workspace). C-exact paths
   must use our own kernels with fixed geometry, or a library path whose
   algorithm is pinned and verified identical across instances. The legacy
   `cublasGemmEx` paths (Kimi/GLM) are **not yet conformant** and must be
   audited/pinned or replaced before an R3 cross-instance claim is made.
2. **Device-independent reduction trees.** Reduction/split-K geometry must not
   depend on `#SMs`, `multiProcessorCount`, clock, or `cudaOccupancy*`.
3. **No atomics in the reduce path** (or a fixed serialization order).
4. **No device-conditional numeric path.** TC-vs-scalar, arch-specific kernels,
   or capability probes that change the numerics must not select differently
   across conforming instances (probe may gate perf, never numerics).
5. **Deterministic launch shapes.** Grid/block sizes fixed by problem shape, not
   by device properties.

Any kernel that cannot meet these is C-fast by definition, regardless of intent.

## 2. Numeric classes

Two implementation classes, named in code and in every kernel's header:

- **C-exact** — every kernel preserves the R3 token output **and is
  device-independent** (§1.1). Allowed: bf16/fp16 activations, **weight-only**
  quantization, fused kernels with a *deterministic, device-independent*
  accumulation order, exact-verify speculative decoding, CUDA graphs / PDL /
  multi-stream that do not reorder numerics, TMA staging.
- **C-fast** — tolerance-class. Allowed (in addition): activation quantization
  (NVFP4/FP8/two-term), tensor-core W4A4/W8A8, arbitrary split-K / reordered
  accumulation, batch-shape-dependent kernels.

Forbidden in **both** unless the class explicitly changes: non-deterministic
atomics in the reduce path, unseeded stochastic ops.

## 3. Decision rule (the point of this document)

1. A proposed optimization is classified **before** implementation:
   - If it preserves the golden tokens ⇒ **C-exact**, may be default.
   - If it changes numerics in a way that can alter tokens ⇒ **C-fast**, must be
     opt-in and quality-validated.
2. "Preserves golden tokens" is *not* argued analytically — it is **proven** by
   the model's R3 selfcheck (golden set) and the regression guards.
3. FSI-facing default paths are **always C-exact / R3** for modes that need it.
   `--bit-exact` (`FastMathMode::kOff`) is the explicit R3/C-exact switch; the
   default `kAuto` must resolve to C-exact for FSI models unless the operator
   opts in.
4. Enabling any C-fast path requires: (a) explicit operator opt-in, (b) a
   quality harness result, (c) a tolerance bound recorded.

## 4. Verification obligations

- **C-exact**: the existing SelfCheck harness — golden token sequence and
  bit-identical `final_logits` across ≥3 cycles (per model, `SKIP_RETURN_CODE 4`
  without vault). Any C-exact kernel change must keep it green. **R3
  cross-instance additionally requires running the selfcheck on ≥2 device
  instances** (or a recorded cross-instance diff), plus the §1.1 audit.
- **C-fast**: (a) device-parity vs the C-exact path within the declared
  tolerance; (b) the quality harness (A/B eval, e.g. GSM8K/MMLU subset) within
  the declared delta; (c) a perf baseline entry.
- A quality harness does **not** yet exist; building it is the precondition for
  promoting any C-fast path (see §6).

## 5. Worked example (why this document exists)

- Kimi W4A4/A8/two-term TC paths: **C-fast**. They are deterministic but change
  the greedy token sequence (golden flipped), so they cannot be R3/default.
  Correct classification: opt-in C-fast, blocked from default by the contract —
  not a bug to keep chasing.
- Spec-decode with **exact verify**: **C-exact** (accepted tokens equal greedy
  bit-for-bit). Allowed as default.
- Fused FP4→bf16 dequant with bf16 activations: **C-exact candidate** (weights
  already quantized in the legacy path; only the accumulation order changes).
  Must be proven by the R3 selfcheck, but is a legitimate default candidate
  unlike activation quantization.

## 6. Open operator decisions

- Is **R4 (cross-model)** required (sm_120 vs sm_121, or future GPUs), or is R3
  "device-independent within a stated GPU model class" sufficient? R4 is far
  more expensive and may force fixed-point/exact paths.
- Which GPU model classes must be in the supported set (and therefore need
  cross-instance selfcheck evidence)?
- Is a C-fast mode ever exposed to FSI customers, or is C-exact the only
  shippable class?
- Approve the conformance audit of the existing `cublasGemmEx` legacy paths
  (§1.1.1) and building the quality harness (precondition for any C-fast
  promotion).

## 7. Non-goals

- Bit-identity **across different GPU models** unless R4 is required.
- Batch-invariance (single-stream engine; no batch composition to be invariant
  to).
- Matching third-party engines' (vLLM/SGLang) numerics.
