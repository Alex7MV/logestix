# Residual closures — 2026-10-23

Verbatim §U from `docs/OPEN_TASKS.md`, with the resolution found in the same
session.

## U. granite-4.2-30b tool round-trip RED again at `31998bf` (found 2026-10-13)

`test_granite_dense_selfcheck`'s second test case ("end-to-end tool
round-trip", real weights) fails at HEAD:

```
CHECK( calls[0].name == "get_weather" )   with expansion: "get_weather.x" == "get_weather"
```

The model emits `get_weather.x` — it puts the `.x` into its **own reasoning
text** first ("We have tool: get_weather.x, …"), then into
`<function>get_weather.x</function>` and into the bare
`{"name": "get_weather.x", …}` object. So the parse path is not at fault: the
divergence is in the greedy generation. The first test case is unaffected
(`Paris`, `top1_tok=12366`, `logits_hash=3d1604aa75c99a1f` stable over 3
cycles), and FSI/CODE/DET are green.

**Not an Arrow regression.** Found while bumping Arrow 25.0.1 -> 26.0.0, so it
was attributed with a control: a second tree at the *same* commit (`31998bf`)
built against **Arrow 25.0.1** produces the identical `get_weather.x` **and the
identical `logits_hash=3d1604aa75c99a1f`** — the Arrow bump is bit-identical
on this path.

**Where the suspicion points instead:** the granite tool round-trip was
"recovered" on 2026-10-07
(`open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md`, archive row
L/M/N), and 9 commits have landed in `libs/lsxcommon` since — including three
that touch the shared attention layer granite decodes through (neutral-layer
split `2b4b41c`/`65db1c7`, `00e22ec` TC mxf4nvf4 GEMM tile fix, `f99008b`
`__host__ __device__` KV-geometry helpers). Which of them moved this token is
**not yet established** — the missing instrument is a generation-level oracle
(the token ids of the 256-token round-trip, plus the logit margin at the flip
step), not another pass/fail on the same assertion.

## Resolution

The defect was **not** in generation arithmetic: the shared attention/GEMM
changes listed above had not moved the token — the `get_weather.x` spelling was
the generic Hermes prompt, not the checkpoint's ChatML template. The native
Granite codec (`granite_tool_format.{h,cpp}`, commit `de0b160`) renders the
checkpoint's `<|im_start|>…<tool_call><function=…>` form, including the
`<think></think>` prefill and the `enable_thinking=false` newline. The real-weight
`test_granite_dense_selfcheck` tool round-trip now returns `get_weather` and the
Paris `logits_hash=3d1604aa75c99a1f` is unchanged, so no golden was re-captured.

## qwen fused-linear attention test fixtures

`test_qwen_dense_fused_linear_attn` had two RED cases at synthetic
`nk=2 kd=8` (`nanf < …`). They were test-fixture defects, not kernel defects:

- `FillNvfp4` generated E4M3 block scales `120 + (i % 12)`, which includes
  `0x7F` — the reserved E4M3FN NaN. The scale cycle stops at `120 + (i % 6)`.
- The host z reference accumulated the same projection twice.
- After the NaN was removed, the tiny full-layer geometry (`gs=0.5`) overflowed
  FP16 at the output; it now uses the production-like `gs=0.02`.

All 43,675 assertions pass.

## §Y — r3/bit-exact performance restored (and qwen3.6 closed)

The X4/F-04 strict path had been made environment-independent by disabling the
large-N streaming GEMV; qwen3.5-9b r3 fell to **19.3 t/s** against the 111.06
reference. Disabling the route was the wrong side of the trade.

The strict scope now sets a process-global `LsxBitExactMode` flag. The
dispatcher reads it as: both the large-N and the size-bounded (<=32 MiB)
streaming GEMV are canonical strict routes, TC/seq1-gemv stay off, and
`LSX_GEMV_LARGE`/`LSX_GEMV_MAX_MIB` are ignored, so the strict hash cannot move
with the environment. qkv and z use the neutral `RunLsxNvfp4Gemm` route in this
mode for every H (a/b stay warpdot at N=32), which is what closes qwen3.6
(H=2048): with the earlier H>=4096 floor it measured 69.6, without the floor
73.4.

Canonical 2048-token measurements and re-seeded references:

| model | old r3 | new r3 |
|---|---|---|
| qwen3.5-9b | 111.06 | **147.0** |
| qwen3.8-27b | 39.03 | **52.27** |
| qwen3.6-35B-A3B | 13.32 | **73.39** |

The perf entries that the harness rewrote on the same runs were kept. Correctness:
qwen3.5-9b, qwen3.8-27b, qwen3.6-35B-A3B and Ornith SelfCheck/FSI/CODE/DET all
pass. `test_perf_baseline_qwen3_5_9b` and `..._qwen3_6_35b_a3b` pass both modes.

Rejected alternative: keep the streaming GEMV off and re-seed r3 at ~19 t/s
(correct but 5.7x slower); it makes the strict mode useless for the main use case
and does not improve determinism.

### Ornith perf-mode 167.06 -> 160.2 (closed as a documented cost)

`test_perf_baseline_ornith` read 160.0/160.1 t/s on two cooled runs against
167.06. Attribution: the reference is from `36eddff` (2026-10-07), before the
neutral routed-MoE migration (`c3ddc4b`) and the overflow-proof batched SwiGLU
quantizer (`8e34314`). That commit explicitly records the cost: "one extra
elementwise launch per layer (Qwen3.6 256-token: 166.7 -> 161.3 t/s)" — the
same ~4% seen here, so the gap is a deliberate correctness fix (massive
activations no longer overflow FP16 and poison the block scale), not a new
regression. The perf entry was re-seeded to **160.20 t/s** with the measured
TTFT/power evidence, and the harness then re-seeded r3 from 12.70 to
**72.02 t/s** on the new strict route.

Router investigation (negative): the decode router is ~10% of GPU time, but
changing its launch from 128 to 64 threads per block moved the isolated kernel
only 11.80 -> 11.73 us and left E2E unchanged (160.5 vs 160.0); reverted.
`ornith.router.decode` had no budget row (the test SKIPped); a row was added
with 11.84 us observed and a 14.0 us max, so the real-shape router is now
guarded instead of silently skipped.

