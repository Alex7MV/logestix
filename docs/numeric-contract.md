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
| **R3 — golden** | Same build + same GPU SKU (e.g. H100 SXM5) + same input + fixed PRNG seed ⇒ bit-identical `final_logits` across ≥3 runs, and the generated token sequence equals the model's expected golden sequence. | Cross-GPU architecture, cross-toolchain, cross-hardware bit-identity. |
| **R2 — deterministic** | Same build + same GPU SKU + same input ⇒ bit-identical output across runs. No golden-token requirement. | Equality with the R3 reference; cross-machine. |
| **R1 — tolerance** | Output deterministic per run; numeric deviation from the R3 reference bounded by a declared tolerance and quality-validated on an eval set. | Bit-identity; cheap regression detection. |

R3 is the FSI default. R2/R1 are opt-in and must never be a silent default.

## 2. Numeric classes

Two implementation classes, named in code and in every kernel's header:

- **C-exact** — every kernel preserves the R3 token output and bit-identical
  logits. Allowed: bf16/fp16 activations, weight-only quantization, fused
  kernels with a strictly fixed and invariant accumulation order, exact-verify
  speculative decoding, CUDA graphs / PDL / multi-stream that do not reorder
  numerics, TMA staging.
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

## 4. Verification obligations

- **C-exact**: the existing SelfCheck harness — golden token sequence and
  bit-identical `final_logits` across ≥3 cycles (per model, `SKIP_RETURN_CODE 4`
  without vault). Any C-exact kernel change must keep it green. Exception: for
  speculative decoding, `final_logits` bit-identity is enforced only for the
  accepted (actually generated) tokens.
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

- Is cross-machine (different GPU model / toolchain) reproducibility required,
  or is R3 "same build + same GPU SKU" sufficient for FSI?
- Is a C-fast mode ever exposed to FSI customers, or is C-exact the only
  shippable class?
- Approve building the quality harness (precondition for any C-fast promotion).

## 7. Non-goals

- Cross-hardware bit-exactness.
- Batch-invariance (single-stream engine; no batch composition to be invariant
  to).
- Matching third-party engines' (vLLM/SGLang) numerics.
