# Numeric & Reproducibility Contract

Status: accepted (2026-09-15)
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

- **C-exact** — every kernel preserves the R3 token output, and its logits are
  **bit-reproducible for the declared reference path** (identical across ≥3 runs
  and across the supported GPU-model set) — not necessarily bit-equal to the
  optional `kOff` scalar path. Allowed: bf16/fp16 activations, weight-only
  quantization, fused kernels with a strictly fixed and invariant accumulation
  order, exact-verify speculative decoding, CUDA graphs / PDL / multi-stream that
  do not reorder numerics, TMA staging. Explicitly required: deterministic GEMM
  execution paths (e.g. forced deterministic cuBLAS tiling, static workspace size
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
4. FSI-facing default paths are **R3**: the production path (default `kAuto`,
   including its tensor-core fast-math kernels) must be deterministic,
   device-independent (§1.1) and golden-token-correct; its own logits are the R3
   reference. `--bit-exact` (`FastMathMode::kOff`) is an **optional strict
   scalar mode** for verification/audit, not a required default, and it is not
   bit-equal to the production path.
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
  promoting any C-fast path (see §7).

## 5. Worked example (why this document exists)

- Kimi W4A4/A8/two-term TC paths: **C-fast**. They are deterministic but change
  the greedy token sequence (golden flipped), so they cannot be R3/default.
  Correct classification: opt-in C-fast, blocked from default by the contract —
  not a bug to keep chasing.
- Spec-decode with exact verify: **C-exact** (accepted tokens equal greedy
  bit-for-bit). Allowed as default.
- Default `kAuto` (tensor-core fast-math) vs `--bit-exact` (`kOff`): **both
  C-exact** — each is deterministic, device-independent and golden-token-correct.
  They are distinct numeric paths (different bits); the production (`kAuto`) path
  is the R3 reference, and `kOff` is the optional strict scalar mode.
- Fused FP4→bf16 dequant with bf16 activations: **C-exact** if it is
  deterministic, device-independent and golden-token-correct (its own path then
  is the reference); **C-fast** otherwise.

## 6. Conversion parity vs the source artifact (offline quantization)

Runtime reproducibility (§1) is about *the same artifact* producing the same
output. A separate question is whether the converted artifact reproduces the
**source** model's output — the question the FSI/CODE/DET gates probe. This
section fixes how to classify the answer so it is not re-litigated per model.

**Two guarantees, not one.**

- **Artifact determinism** — same converted model + same prompt ⇒ byte-identical
  output. Holds by construction (fixed offline conversion, R3 runtime); DET
  verifies it. A divergence here is a bug.
- **Source parity** — converted model's greedy token equals the source model's.
  **Not guaranteed by quantization; bounded only by the format's error.** The
  gates go red where that bound is exceeded at the decision margin.

**Mechanism.** Converting a BF16/FP8 source to NVFP4 re-rounds every weight by up
to half an E2M1 ulp and re-encodes the block scales to E4M3. At a token where the
source's top-2 logits sit within the per-block error, the quantization noise
decides the argmax. This is not a codec defect: the E4M3FN codec is verified
bit-identical to `cuda_fp8` / `torch.float8_e4m3fn`.

**Correction taxonomy — by cause, not by symptom:**

1. **Codec / encoder bug** ⇒ fix the codec. (Granite's tool round-trip recovered
   exactly this way after the E4M3FN subnormal alignment.)
2. **Near-tie below the format noise floor** ⇒ no codec change recovers it. The
   only levers are *precision allocation*, not corrections: (a) take the
   responsible tensor out of NVFP4 (router / embed / lm_head / norms / small
   projections — the pattern already used by `alice` and `xing`), or (b) finer
   block-scale granularity. Both change the artifact and cost size/speed.
3. **Biasing the quantizer to reproduce the source's argmax** ⇒ overfitting to a
   test set; forbidden as a general policy.

**Decision rule.** Measure the source logit margin at the deciding token. If the
margin ≲ the block error ⇒ inherent, document the deviation (live cases: gemma
FSI/CODE §J, phi-4 DET §O, GLM-5.3-Flash FSI §P of `OPEN_TASKS.md`). If the
margin is large and a flip is present ⇒ treat it as a bug/precision defect and
fix it. Never tune per prompt.

**Non-monotonic caveat.** Lower noise does not monotonically restore parity — a
near-tie can land either way — so parity itself is never *guaranteed*; the only
guarantee is that more precision puts the noise floor below more margins.

**Consequence.** Once real encoder bugs are fixed and precision has not been
deliberately raised for a tensor, the residual flips are a stable, deterministic
property of *that specific converted artifact*. The resolution is either a
documented deviation or a re-conversion with raised precision for the
responsible tensor — decided by the margin rule above.

## 7. Open operator decisions

- Confirm the R3 supported GPU-model set. Current: {RTX 5090, RTX 5060 Ti} (both
  sm_120). Adding a model requires cross-model selfcheck evidence and a §1.1
  audit before the claim extends to it.
- Is a C-fast mode ever exposed to FSI customers, or is C-exact the only
  shippable class?
- Approve building the quality harness (precondition for any C-fast promotion).

## 8. Non-goals

- Bit-identity on hardware **outside** the supported GPU-model set.
- Batch-invariance (single-stream engine; no batch composition to be invariant
  to).
