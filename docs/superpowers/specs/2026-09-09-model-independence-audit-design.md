# Model Independence Audit — Design

Date: 2026-09-09
Branch: `audit/model-independence`

## Goal

Audit all 10 registered models for mutual independence and conformance to the
inference patterns defined in AGENTS.md (model-neutral NVFP4 layer, per-call
fast-math policy, guard tests), remove legacy/dead functionality, using TDD,
in a separate branch.

## Findings (audit results)

Models: glm-5.2, glm-5.3-flash, deepseek-v4-flash, gemma-4-26b-a4b-it,
kimi-k2.7-code, qwen3.5-9b, qwen3.8-27b, qwen3.5-35b-a3b, qwen3.6-35b-a3b,
qwen3.8-flash-next.

1. **deepseek-v4-flash** violates the per-call fast-math policy: flag is set
   once in the constructor (`gpu::SetDeepseekFastMath(cfg_.fast_math)`,
   `deepseek_moe_model.cpp:33`) instead of a scope opened inside `Infer`.
   Missing from the guard test's consumer list.
2. **glm-5.3-flash** calls the neutral NVFP4 layer through qwen-named shims
   (`qwen::RunQwenNvfp4Gemm`, 17 call sites) — cross-family naming coupling
   instead of `lsx_nvfp4::` directly.
3. **Dead code**: `RunQwenFusedDecodeAttnKernel` is KNOWN-BROKEN (diverged on
   the real Qwen3.5-9B checkpoint, unwired from generation in 19e8d4f),
   test-only. Its helpers `RunQwenFusedQkv`/`RunQwenFusedOProj` are used only
   by it.
4. **glm-5.2** and **gemma-4-26b-a4b-it** have no fast-math policy and are
   absent from the guard consumer list (they use family-own numeric paths,
   which is allowed; the guard must still track them).

## Out of scope

- Moving `RunExpQsaAttention` reuse out of `glm_moe_flash_inference.cu` and
  neutralizing gemma's `nvfp4gemv` path — these change numeric paths and risk
  bit-exact parity regressions. Separate audit.

## Design

Branch `audit/model-independence`, incremental TDD steps, one green commit
per step (red test first where applicable).

### Step 1 — Guard test extensions (red first)

In `tests/test_model_independence_guard.cpp`:
- Add `deepseek_moe_model.cpp`, `glm_moe_model.cpp`, `gemma_base_model.cpp`
  to the consumer list (require a fast-math policy marker in the file).
- New test case: `glm_moe_flash_inference.cu` must not reference
  `qwen::RunQwenNvfp4Gemm` / `RunQwenNvfp4GemmSeq1GemmV`.

### Step 2 — Flash shims removal (makes step 1 green)

`glm_moe_flash_inference.cu`: mechanically replace
`qwen::RunQwenNvfp4Gemm` → `lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm` (signatures
identical; the shims are inline forwarders, so behavior is 1:1). Replace the
`lsxcommon/qwen_inference.h` include with `lsxcommon/lsx_inference_nvfp4.h`
where no other qwen symbol is used; `QwenMoeRoutedSlot` usage in
`glm_moe_flash_device_math.h` remains legitimate (shared slot type via
`qwen_moe_inference.h`).

### Step 3 — Deepseek per-call policy (makes step 1 green)

- Remove the constructor call `gpu::SetDeepseekFastMath(cfg_.fast_math)`.
- Add a family-scoped RAII scope in `deepseek_moe_load.cuh` (mirrors
  `LsxFastMathScope`: set flag on construct, restore on destruct).
- Open the scope in `DeepSeekV4FlashModel::Infer` (and SelfCheck path) with
  `cfg_.fast_math`.

### Step 4 — glm-5.2 / gemma guard entries (makes step 1 fully green)

Two-tier consumer check in the guard test:
- **Neutral-layer consumers** (qwen dense/moe/36/exp, glm-5.3-flash, kimi,
  deepseek): require `FastMathScope` + `cfg_.fast_math` strings (deepseek's
  family scope is named `DeepseekFastMathScope`, so the check matches).
- **Family-own models** (`glm_moe_model.cpp`, `gemma_base_model.cpp`): no
  process-global toggle exists; the guard requires the greppable marker
  comment `// fast-math policy: family-own (no process-global toggles)`.

### Step 5 — Dead code removal

- Delete `RunQwenFusedDecodeAttnKernel`, `RunQwenFusedQkv`,
  `RunQwenFusedOProj` and their private helpers from
  `qwen_inference.cu` / `qwen_inference.h`.
- Delete `tests/test_qwen_dense_fused_decode_attn.cu` and its CMake target.

## Verification

1. `cmake --build` (linux-release-znver4 or linux-debug) — green.
2. Host test suite: `ctest -R "independence_guard|absl|naming|tokenizer|core|mosaic"`.
3. Real selfchecks for all models with weights present in the vault
   (deepseek, glm-5.3-flash, gemma, kimi, qwen ×5); glm-5.2 SKIPs (no weights
   on this machine) — expected.
4. Every step committed separately (bisectable history), English commit
   messages with `feat:`/`fix:`/`refactor:`/`test:` prefixes.
