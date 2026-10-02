# MiMo-V2.6-Flash-RL: closed as a vendor-model defect

Session date: 2026-10-02. Moved out of `docs/OPEN_TASKS.md` — item A2 of the
2026-09-30 snapshot, now closed and not actionable.

## Decision

`MiMo-V2.6-Flash-RL` does not answer the standard prompts in this environment,
and the cause is the checkpoint, not logestix. The implementation is
**bit-exact with the vendor's own `modeling_mimo_v2.py`**: a manual 48-layer
forward built from `vendor.MiMoV2DecoderLayer` (weights dequantized from the
source, experts selected by the router) produces
`top5 = [2181, 110015, 113764, 11853, 1268]`, and the device produces
`top1 = 2181`. The vendor oracle also fails on a plain question ("What is the
capital of France?") with the same repetitive garbage.

Per the operator (2026-10-02): tasks on this model may be closed without further
work, except the important ones.

## What was done

- The correct implementation was merged; the converted model was removed from
  the vault (it is no longer present under `/mnt/llm_models/models`).
- `test_perf_baseline_mimo_v2_6_flash_rl` — registered earlier the same day as a
  future perf gate — was removed again: a performance target for a model that
  cannot run is not a gate, it is a placeholder.
- `tests/test_mimo_moe_selfcheck.cpp` now documents the permanent failure in its
  header, so that if a MiMo checkpoint ever reappears in the vault the red
  result is read as "the checkpoint", not "start debugging the kernels".

## The two hypotheses that were wrong

Kept because the trail is what makes the conclusion believable:

1. **"The routed expert weights are wrong."** A Python decode of the original
   checkpoint showed every per-layer component matching the device within NVFP4
   quantization error (router top-8 identical, expert FFN norms equal to four
   decimals for layer 1 / expert 137). The apparent explosion of the residual
   stream turned out to be inherent to the checkpoint — a NumPy full forward on
   the source weights produces the same large norms.
2. **"Decode attention diverges at positions >= 2."** This came from
   diagnostic dumps; a filename collision across runs. With clean single-run
   dumps, attention matches exactly at pos 0, 1 and 2 for both the GA and SWA
   paths.

A third pass fixed two genuine framing/model findings along the way: the
prompt frame was missing the `<think></think>` prefill (now matches
`apply_chat_template(add_generation_prompt=True)` exactly), and dense FP8→NVFP4
requant error (~1–3% per component) compounds across 48 layers — a property of
the checkpoint's dynamic range, documented but not "fixed".

## What still runs, and stays green

These cover our code on synthetic fixtures and are the real signal for the
family — they are the "important" part of this model and are not touched by the
closure: `test_mimo_moe_convert`, `test_mimo_moe_loader`,
`test_mimo_moe_expert_cache`, `test_mimo_moe_kernels`, `test_mimo_moe_dequant`,
`test_mimo_moe_geometry`, `test_mimo_moe_model`, `test_mimo_tool_framing`.

`test_mimo_moe_{selfcheck,fsi,code,det}_selfcheck` SKIP without the checkpoint
and stay registered, so they run if a usable checkpoint ever returns.

## If it is ever revisited

Verify the checkpoint shards and checksums, or obtain the exact
`transformers` build the checkpoint was produced with. Do not re-open the
kernel investigation: the vendor oracle already matches.

Full investigation trail: the 2026-09-30 snapshot, section A2 —
[`2026-09-30-cuda-review-windows-bringup.md`](2026-09-30-cuda-review-windows-bringup.md).
