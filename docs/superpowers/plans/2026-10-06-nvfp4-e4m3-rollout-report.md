# NVFP4 full-E4M3 codec — rollout + model-lifecycle verification (2026-10-06)

Companion to the design
`docs/superpowers/specs/2026-10-06-nvfp4-full-e4m3-scale-codec-design.md` and the
implementation plan `docs/superpowers/plans/2026-10-06-nvfp4-full-e4m3-scale-codec.md`.

## 1. Codec change — done

The NVFP4 block scale is now **full unsigned E4M3 (UE4M3)** on encode and decode
(was power-of-two-only). Canonical codecs live in
`libs/lsxcommon/src/nvfp4_scale.h` (`FloatToE4m3` RNE encode, `E4m3ToFloat`
decode); `nvfp4_codec.h::E4M3ScaleToFloat` and `schema.{h,cpp}`'s
`float_to_e4m3`/`e4m3_to_float` delegate to them. The retired
`FloatToE4m3P2`/`E4m3P2ToFloat` were removed; all production callers migrated
(offline quantizer, TC act-quant, and the family runtime act-quant in
`qwen_inference.cu`, `qwen_moe_inference.cu`, `kimi_moe_inference.cu`,
`ornith_moe_inference.cu`, `glm_moe_flash_fp8.cpp`, `qwen_moe_exp_inference.cu`).
`E8m0ToE4m3Pow2Pair` stays (E8M0 is exact). The skill
`nvfp4-blackwell-conventions` + `logestix-repo-map` and the external mirror
`/home/alexmv2025/projects/cuda/docs/NVIDIA_DOCS_NVFP4_MEMORY.md` were updated.

**Evidence**
- New host test `tests/test_nvfp4_scale` (`fast`): exhaustive decode of all 128
  positive bytes vs an independent `ldexp` reference, RNE points, ties-to-even,
  pow2 bit-identity, field-zero contract. PASS.
- Device parity `test_lsx_kernels` ("E4M3ScaleToFloat device parity, all 128
  positive bytes"). PASS.
- All codec-relevant unit/kernel tests updated and green:
  `test_lsx_kernels`, `test_deepseek_moe_loader`,
  `test_deepseek_moe_inference_kernels`, `test_qwen_nvfp4_tile_gemm`,
  `test_qwen_nvfp4_gemv`, `test_lsx_nvfp4_dispatch`, `test_glm_moe_flash_convert`,
  `test_blackwell_tc_mma`, `test_gemma_moe_moe`, `test_qwen_moe_exp_kernels`,
  `test_qwen_moe_kernels`, `test_ornith_moe_kernels`, `test_deepseek_moe_moe`,
  `test_nvfp4_pack`, `test_deepseek_nvfp4_repack`, `test_gigachat_moe_kernels`.
- `fast` label: **88/91 green**. The 3 reds are pre-existing and unrelated to
  the codec — the paused Nemotron-NVFP4 ingestion has not registered its files:
  `test_code_naming_guard` (variant token `nvfp4` absent from `SchemeRegex`),
  `test_ctest_labels_guard` (`test_nemotron_moe_nvfp4_convert` unlabeled),
  `test_model_suite_guard` (nvfp4 targets absent from `Suites()`; the guard
  requires the full 9-stage row, so this closes only when that ingestion's
  tests exist).
- Backward compatibility: a `mantissa == 0` byte decodes bit-identically to the
  old codec; every existing pow2 tree is therefore unaffected. Re-conversion is
  needed only to *gain* the mantissa (and because the encoder output changed).

## 2. End-to-end validation — Qwen3.5-9B (re-converted)

The vault's converted `logestix/` trees were deleted, so the smallest family was
re-converted with the new codec and the full lifecycle exercised:

| stage | command | result |
|---|---|---|
| convert | `lsxforge --qwen --input=<dir> --output=<dir>` | PASS (2 m 9 s, 4.8 GiB) |
| load | `test_qwen_dense_selfcheck` (pinned 4804 MiB in 440 ms) | PASS |
| selfcheck | Paris, 3 bit-exact cycles, `logits_hash=10ebf916f68e0ad7` | PASS |
| fsi | `test_qwen_dense_fsi_selfcheck` | PASS (3/3) — golden captured from source |
| code | `test_qwen_dense_code_selfcheck` | PASS (4.7 s) |
| det | `test_qwen_dense_det_selfcheck` | PASS (10.0 s) |
| perf | `test_perf_baseline_qwen3_5_9b` | not re-run this session |

The selfcheck hash moved (the `kAuto` TC act-quant now uses RNE full E4M3);
`tests/perf/perf_baselines.json` stores t/s, not logits hashes, so there is no
recorded hash to update there.

## 3. Re-conversion matrix (vault sources, ascending size)

Every family must be re-converted for the E4M3 encoder to take effect. One GPU
job at a time (`running-a-model` §8). Flag from `apps/lsxforge/src/main.cpp`.

| source dir | size | forge flag | note |
|---|---|---|---|
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 21G | **none yet** | needs the `--nemotron-nvfp4` pipeline (paused ingestion) |
| Qwen3.5-9B | 23G | `--qwen` | **DONE** (validated above) |
| phi-4 | 28G | `--phi` | |
| gemma-4-26B-A4B-it | 49G | `--gemma4` | |
| Qwen3.8-27B | 52G | `--qwen` | |
| granite-4.2-30b | 55G | `--granite` | |
| Xing4.0-29B-A4B | 59G | `--xing` | |
| Ornith-1.5-35B-A3B-MLX | 65G | `--ornith` | |
| Qwen3.5-35B-A3B | 67G | `--qwen35moe` | |
| Qwen3.6-35B-A3B | 67G | `--qwen36moe` | |
| DeepSeek-V4-Flash | 149G | `--deepseek_nvfp4` | repack path (no converter by design) |
| AliceAI-Foundation-80B-A3B-Base | 152G | `--alice` | |
| GLM-5.3-Flash | 306G | `--glmflash` | |
| Qwen3.8-Flash-Next | 336G | `--qwen38flashnext` | |
| GigaChat3.5-432B-A28B-Reasoning | 416G | `--gigachat` | |
| DeepSeek-V4.1-Flash | 476G | `--deepseek_v41` | |

Total source ≈ 2.3 TB; converted footprint is much smaller (Qwen3.5-9B: 4.8 GiB
from 23 GiB). Disk: 393 GiB free on `/`.

## 4. Per-family lifecycle stages (authoritative registry)

`tests/test_model_suite_guard.cpp::Suites()` is the source of truth: each family
row must declare `convert → load → model → kernels → selfcheck → fsi → code →
det → perf`, and each stage maps to ctest targets. After a re-conversion run:

```
LSX_TEST_<FAMILY>_MODEL_PATH=<model dir> ctest -R "<family>.*(selfcheck|fsi|code|det)"
```

FSI/CODE/DET SKIP (exit 4) when the golden is absent. Current golden inventory:

| golden | present for |
|---|---|
| fsi | alice, gemma, glm-flash, granite, phi, qwen3.5-35b, qwen3.6-35b, qwen3.8-27b, qwen3.8-flash-next, xing, **qwen3.5-9b** (captured 2026-10-06) |
| code | same minus granite, plus qwen3.5-9b |
| det | all of the above, plus deepseek-v4.1 |

Open parity gaps (declared in `kGoldenDebt`): granite CODE; deepseek
families FSI/CODE; the 3.5/3.6/3.8 MoE and gigachat/mimo goldens; etc.
The qwen3.5-9b FSI debt was cleared (capture + `kGoldenDebt` row removed).

## 5. Open items

1. **Finish the Nemotron-NVFP4 ingestion** (its three guard reds block a fully
   green `fast`); it also supplies `--nemotron-nvfp4`, required to re-convert the
   one vault family that has no working pipeline.
2. **Re-convert the remaining families** (matrix §3) and re-run their lifecycle
   stages; expect CODE/DET/selfcheck to pass or improve, and SKIP where a golden
   is absent.
3. **Re-run perf baselines** for families whose `kAuto` act-quant numerics move;
   the baseline judge is tolerance-based (no hashes stored).
4. Do NOT byte-pass source NVFP4 blocks to the decoder for the Nemotron
   checkpoint under the old pow2 flavor; with the new full-E4M3 codec the source
   scales now decode correctly and the passthrough becomes possible (the ~11 %
   requantization loss in `docs/OPEN_TASKS.md` section L goes to zero).
