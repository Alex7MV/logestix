# GigaChat3.5 Validation — Whole-Stack Parity (now) and Real-Weight Run (gated)

**Status:** approved (design), 2026-09-20
**Scope owner:** Plan 3 pre-work + Plan 3 of the GigaChat3.5 ingestion.
**Related:** `docs/superpowers/specs/2026-09-20-gigachat35-432b-a28b-reasoning-ingestion-design.md`,
`docs/superpowers/plans/2026-09-20-gigachat35-ingestion-plan2b-attention-pipeline.md`.

## Problem

Plan 2b validated each new kernel against a host reference (interleaved YaRN
RoPE bit-exact, MLA block parity 1e-3, GDN block parity 1e-3 across two
tokens) and left a synthetic 2-layer model test that only checks the forward
pass is finite and deterministic. The **integration** of those kernels is
therefore unverified: the `pre_post` residual ordering, the zero-centered
gated-norm wiring, MoE routing / shared-expert composition, and the `lm_head`
could all be wrong and still produce finite, deterministic logits.

The real-weight SelfCheck (`test_gigachat_moe_selfcheck.cpp`) is the ultimate
gate but is blocked on the 416 GB conversion, which is an operator step.

## Part A — whole-stack synthetic forward parity (build now)

### Goal

A second, independent implementation of the full synthetic model on the host
(F32), driven from the **converted** Arrow weights, compared against
`GcPipeline::Step` on the GPU. This catches integration/wiring bugs without
the real checkpoint.

### Acceptance criterion

- Target `test_gigachat_moe_parity`, label `inference`, `SKIP_RETURN_CODE 4`,
  `SKIP()` when no CUDA device.
- Sequence of two tokens (`{1, 2}`), comparing logits after each `Step`.
- Per-position: greedy `argmax` must be equal, and
  `max|Δ| <= 5e-2 * max(1, max|ref|)`.
- The error budget is attributed to F16 activation I/O at each GPU GEMM
  (FP32 accumulate, see `libs/lsxcommon/src/lsx_inference_nvfp4.cu:112`);
  the host reference computes the same dequantized weights in F32. The
  tolerance is documented in the test header with this rationale.

### Components

1. **`tests/gigachat_host_ref.h`** — pure-host F32 reference, no CUDA, `std::`
   only (it is included by `.cu` translation units; no abseil containers).
   - Extracted from `tests/test_gigachat_moe_attn.cu`: `GemvRef`, `NormRef`
     (zero-centered gated norm), `RopeRef`.
   - New: `SiluRef`, `SwigluLimitRef`, `RouterTopKRef`, `MoeRef`,
     `DenseMlpRef`, `MlaRef` (causal, cached K/V), `GdnRef` (recurrent state +
     conv state), `LmHeadRef`.
2. **`tests/test_gigachat_moe_parity.cu`** — reads the converted fixture:
   `logestix/pinned/global.arrow`, `logestix/pinned/layer_NNN{,_nvfp4}.arrow`
   and `logestix/experts/layer_N/exp_E.arrow` through
   `lsxcommon::LoadTensorBatchViews`; host-dequantizes NVFP4 (E2M1 +
   pow2-E4M3/16 per 16-element block + `global_scale`) and passes BF16/F32
   through unchanged; composes the reference over both tokens.
3. **`tests/test_gigachat_moe_attn.cu`** — refactored to include the shared
   header; the moved local helpers are deleted. No behaviour change to the
   existing assertions.
4. **`tests/CMakeLists.txt`** — new target; `inference` label in the same
   commit. `test_ctest_labels_guard` enforces this. No
   `test_model_suite_guard` stage change (parity is not an ingestion stage).

### Data flow

```
synthetic source -> ConvertGigaChatModel -> converted Arrow
    host : Arrow bytes -> F32 weights -> gigachat_host_ref -> ref_logits[pos]
    gpu  : LoadGigaChatWeights + InitGigaChatExpertCache + GcPipeline
           -> Step(pos=0), Step(pos=1) -> gpu_logits[pos]
compare per position
```

### Out of scope (Part A)

- No production-code change: `GcPipeline` and the kernels are not touched.
- No new model-ingestion stage and no guard-registry row.
- The real checkpoint is not read.

## Part B — real-weight run (gated on the operator conversion)

Executed only once the operator has produced
`<vault>/GigaChat3.5-432B-A28B-Reasoning/logestix/` (~230 GB) with `lsxforge`.

1. **Conversion** (operator, offline, hours): `lsxforge` FP8 block-128 ->
   neutral NVFP4; 9472 routed experts streamed to disk; MTP layers 40-42
   recorded as a counted cut.
2. **SelfCheck** — `test_gigachat_moe_selfcheck` (label `inference`):
   `"The capital of France is"` -> `"Paris"` with `max_gen_tokens = 2` and
   `>= 3` bit-exact cycles (`LSX_TEST_GIGACHAT_SELFCHECK_CYCLES`), model
   located via `LSX_TEST_GIGACHAT_ROOT` or the vault default.
3. **Optional spot check** — greedy continuation of a fixed prompt compared
   against the shipped `modeling_gigachat3_5.py` (read-only, not committed)
   or an already-validated runtime; recorded in the plan as operator-run.
4. **Perf baseline** — measured at the canonical 2048-token protocol after
   correctness, recorded in `tests/perf/perf_baselines.json` with a
   `test_perf_baseline_gigachat` target (label `perf`). t/s is recorded as
   measured on the single RTX 5090, not targeted.

### Known perf debt carried from Plan 2b

`GcPipeline::Step` copies router top-k indices device-to-host and
`cudaStreamSynchronize`s once per MoE layer. This is a correctness-first
choice; Part B measures its cost before any optimisation.

## Testing

- Strict red -> green: write the parity test first, watch it fail against a
  deliberately incorrect reference/wiring, then make it pass.
- The new target carries its label in the same commit.
- `gigachat_host_ref.h` and the parity test are GPU-gated the same way as the
  existing `test_gigachat_moe_attn` (`SKIP_RETURN_CODE 4`).

## Risks / decision points

- **Tolerance calibration.** If F16 intermediate rounding over two layers
  exceeds 5e-2 relative with synthetic weights, the reference (not the kernel)
  is the thing to re-check first; the per-layer bit-exact/1e-3 tests remain
  the narrow gate. The tolerance may be tightened once measured.
- **Reference duplication.** Extracting the attn-test helpers into a shared
  header must not change the existing tests' numerics; the existing
  assertions are the guard.
- **Part B blocking.** No real-weight number can be produced until the
  operator runs the conversion; Part A is the only work that can land
  unblocked.
