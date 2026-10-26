# Kimi-K2.7-Code — audit verification (2026-10-26)

Branch `feat/kimi-k2.7-code-audit`. This file records what was measured, with
which instrument, on the real checkpoint. Numbers are reproduced by the
commands below; nothing here is copied from an engine log without an external
oracle behind it.

## 1. Conversion

Source: `/mnt/fast_storage/huggingface/Kimi-K2.7-Code` (64 shards, 581 GB,
INT4 group-32 routed experts, BF16 attention/shared/dense, multimodal wrapper).

```bash
LSX_MODEL_ROOT=/mnt/fast_storage/huggingface \
  ./build/linux-release-znver4-aocc/apps/lsxforge/lsxforge --kimi \
  --input /mnt/fast_storage/huggingface/Kimi-K2.7-Code \
  --output /mnt/fast_storage/huggingface/Kimi-K2.7-Code
```

Result (42 min wall): `675 pinned tensors, 23040 experts processed`, tree 542 GB.

`logestix/model.index.json`:

```
format logestix-kimi-r2, num_layers 61, experts_per_layer 384,
total_experts 23040, total_pinned_tensors 980,
skipped_visual_tensors 335, skipped_shape_tensors 69120,
mtp_tensors 0, mtp_layer null
moe_layers 60, dense_layers [0], mla_nvfp4 layers 61
```

65 pinned files (1 global + 61 layers + 3 tokenizer), 60 expert directories,
384 experts in each.

## 2. Instruments and verdicts

| instrument | oracle | verdict |
|---|---|---|
| `test_kimi_moe_rope_parity` | vendored `modeling_deepseek` rope + `yarn_get_mscale(64,1)^2` | PASS, 2040 assertions. The engine's half-split pairing differed by O(1) (cross-position dot 7.52 vs 1.24) and the temperature was 0.0722 vs 0.1447 |
| `test_kimi_moe_tokenizer_parity` | the checkpoint's own tokenizer (`tiktoken` + Kimi `pat_str`) | PASS, 172 assertions / 14 cases (ASCII, code, whitespace, Cyrillic, CJK, emoji, 72-char repeats, chat markers) |
| `test_kimi_moe_int4_real` | source INT4 vs converted NVFP4 payload | PASS |
| `test_kimi_moe_mla_proj_real` | host NVFP4 decoder on the converted pinned bytes | PASS — o_proj (K=8192,kp=8), kv_a/q_a (kp=4), q_b (kp=1) all within 1e-3; routed expert FFN relL2 ≤ 2%; router BF16 matvec ≤ 1e-2 |
| `test_kimi_moe_selfcheck` | in-engine determinism + Paris | PASS, 3 cycles, first token 17374 (`Paris`), identical logits hash `0bdd98687b069382` |
| `tools/kimi_layer_oracle.py` | the checkpoint's own math on the converted weights, per-layer hidden states | PASS — relL2 / cos: layer 0 **0.0034 / 0.999996**, layer 1 **0.0400 / 0.99921**, layer 30 **0.0554 / 0.99886**, layer 60 **0.1226 / 0.99253** |

### Layer oracle — how to reproduce

```bash
# engine side (bit-exact numeric class; fast-math TC prefill is tolerance-class)
LSX_PERF_AUTO=0 KIMI_DUMP_LAYERS=0,1,30,60 LSX_EXPERT_PIN=0 \
  LSX_MODEL_ROOT=/mnt/fast_storage/huggingface LSX_TEST_KIMI_SELFCHECK_CYCLES=1 \
  ./build/linux-debug/tests/test_kimi_moe_selfcheck
# reference side
python3 tools/kimi_layer_oracle.py --model /mnt/fast_storage/huggingface/Kimi-K2.7-Code \
  --ids "The capital of France is" --layers 0,1,30,60 --dump-dir /tmp --weights converted
```

Two traps found while building this instrument (both are *instrument* bugs, not
engine defects — recorded so the next audit does not repeat them):

1. **`kAuto` is the tolerance-class TC path.** The first comparison ran with the
   default policy and showed 26-30 % at layer 0. It was the tensor-core
   MXF4/A4 prefill (`RunLsxTcNvfp4Gemm`), not the weights: in the bit-exact
   class (`LSX_PERF_AUTO=0`) the same layer matches to 0.34 %. A layer oracle
   must run in the same numeric class it compares against.
2. **Read packed INT4 as int32.** `safetensors` defaults to bf16 in the oracle
   helper; converting the packed words to bf16 and back produced a plausible
   but wrong weight matrix (the MoE "exploded" at layer 1). With the dtype
   fixed the reference tracks the engine to 0.9992 cosine at layer 1.

The remaining drift (0.4 % at layer 0 → 12 % at layer 60) is the engine's bf16
residual stream against the oracle's float32 accumulation, plus 4-bit weight
noise; the per-layer *direction* cosine stays ≥ 0.9925.

## 3. Fixes landed in this branch (all with the gate above them)

- `fix: point the Linux vault default at the moved model store`
- `test: pin the MLA rope/attention scale to the vendored reference (red)` +
  `fix: kimi MLA uses the reference interleaved rope and YaRN mscale^2`
- `fix: kimi token policy — no BOS, stop on <|im_end|> (config eos)`
- `fix: exact tiktoken BPE for kimi (vocab is the merge table) + parity gate`
  (the merge-table heuristic was unsound; also fixed two shared-pretokenizer
  defects: Han vs the letter alternative, and the `'s/'t/'re…` contraction
  suffix)
- `feat: native kimi tool-calling bridge (stage 4b)`
- `fix: kimi converter counts ballast and fails on unknown tensors`
- `feat: kimi per-layer hidden dump + reference layer oracle`
- `fix: kimi loader charges the lazy kv_b planes to the resident lease`

## 4. Still open

- **Async-decode determinism (F7)**: the per-step D2H+event readback is still in
  `RunKimiGeneration`; removing it has not been re-tested since the numeric
  fixes. Needs the reproduce → bisect → fix-or-record cycle.
- **FSI/CODE/DET goldens**: not captured. The source is 581 GB in
  `compressed-tensors` INT4; `compressed-tensors` 0.19 and `tiktoken` 0.14 are
  installed, but a full `from_pretrained` run on CPU has not been attempted yet.
  Until it is, these three targets SKIP (vault tree now present, golden absent).
- **Perf row**: not seeded; the converted tree is in place and
  `test_perf_baseline_kimi_k2_7_code` is the harness path.
- **KV cap**: `kKimiMaxSlots = 1024` (config window 262 144). A long-context
  gate is out of reach here; the cap is unchanged.
