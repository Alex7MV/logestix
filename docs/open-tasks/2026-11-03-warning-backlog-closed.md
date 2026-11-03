# Warning backlog closed (2026-11-03)

Branch `audit/2026-11-03-quick-wins`. Snapshot of the warning-backlog item
that the build-noise work created and the follow-up session closed (it briefly
used the letter W, which the 2026-10-23 environment-oracle entry had already
retired).

## What the item was

Scoping `-Wall -Wextra`/`-Wpedantic` to our own directories
(`lsx_warning_policy()`) and making the dependency headers SYSTEM took a clean
`linux-debug` build from **51 408 warning lines to 484**. 246 of the remainder
come from the vendored Arrow/abseil targets' own sources; **238 were ours and
had been buried** — 170 sites in 83 files. No `-Werror` gate was possible until
they were cleared.

## Inventory as measured (before)

| kind | warnings / sites |
|---|---|
| `-Wunused-result` | 53 / 45 |
| `-Wint-in-bool-context` | 42 / 1 (one workspace macro expansion in `alice_moe_inference_pipeline.cu`) |
| `-Wrestrict` | 39 / 39 |
| `-Wunused-parameter` | 26 / 19 |
| `-Wmissing-field-initializers` | 24 / 15 |
| `-Wunused-function` | 19 / 19 |
| `-Wunused-variable` | 7 / 7 |
| `-Wsign-compare` | 5 / 5 |
| `-Wclass-memaccess` | 5 / 5 |
| `-Wdangling-reference` | 5 / 5 |
| `-Woverflow` | 3 / 2 |
| `-Wswitch` | 2 / 2 |
| `-Wmaybe-uninitialized` | 2 / 2 |
| `-Wunused-but-set-variable`, `-Wunused-value`, `-Wmisleading-indentation` | 4 / 4 |

## What the fixes were (not silencers)

- **`-Wrestrict` (the load-bearing finding).** All 39 sites are in-place
  elementwise calls: `SwigluKernel(gate, up, gate, …)`, `SwigluAlphaKernel(a,
  a, b)`, in-place RMSNorm (`xing_rmsnorm_kernel`, `flash_rmsnorm_weighted_kernel`,
  `gemma_head_norm_kernel`), in-place GeGLU/add/residual kernels, and
  `hc_gated_residual_f32_kernel` reusing the `up` buffer as `g_up` scratch.
  The callees declared both operands `__restrict__`, which makes the aliasing
  formally UB and licenses reordering; the qualifiers were removed from the
  aliased parameters (shared `lsx_kernels_act.cuh` / `lsx_kernels_math.cuh` and
  the per-family kernels). Elementwise kernels lose nothing measurable; the
  kernel suites were re-run green.
- **`-Wint-in-bool-context`.** The alice workspace macro's ternary arms were
  `cudaError_t` constants, so `A(…) || A(…)` looked like `0 || 0`. The macro
  now yields `bool` (true = success).
- **`-Wunused-result`.** `absl::SimpleAtoi` in the tensor-name parsers now fails
  the parse (`return std::nullopt`) instead of silently leaving 0; missing
  `num_layers`/`experts_per_layer` counts in the deepseek/glm/kimi index readers
  now set `error_msg` and fail the load; the tokenizer added-token loop skips an
  entry whose `id`/`content` did not parse; type-dispatched JSON probes keep the
  result in a named variable; the gigachat fixture asserts its artifact writes.
- **`-Wdangling-reference`.** `const Scan &s = CachedScan(...)` bound a
  temporary in five `test_expert_store_guard` cases — now a value.
- **`-Wclass-memaccess`.** `memcpy` into `__half` (protected member) replaced by
  `__ushort_as_half`; the ling test converts element-wise.
- **`-Wmaybe-uninitialized` / `-Woverflow` / `-Wswitch` / `-Wsign-compare`.**
  Initialised the probe variables, used `numeric_limits<int64_t>::max()` instead
  of `(1ll << 63) - 1` (UB), handled the simdjson `BIGINT` enum (was falling
  through to whatever the switch's default did not cover), and cast the
  mixed-signedness comparisons.
- **Dead code.** 19 unused `static`/anonymous-namespace functions deleted; the
  unused variables removed; a misplaced `log::info` between two `switch` cases
  moved before the switch (`-Wimplicit-fallthrough`).

## Result

Clean `linux-debug` build: **51 408 → 246 warning lines, 0 from our sources**
(the 246 are the vendored targets' own builds; they keep their flags by design).
`ctest -L fast` 134/134, `-L convert` 37/37; kernel/inference subset
(`test_lsx_kernels`, glm/gemma/xing/ling/flint/deepseek/qwen kernel suites,
`test_blackwell_micro_gemm`, `test_nvfp4_pack`) green. `test_glm_moe_flash_kernels`
failed once under the subset run and passed alone and on a rerun — the known
GPU latency-budget flake class (`§A1`), not a regression.

`-Werror` is still not enabled: it has to be validated on the MSVC and AOCC
presets, whose warning sets differ. The tree-side regressions are already held
by `test_infrastructure_guard` and `lsx_warning_policy()`.
