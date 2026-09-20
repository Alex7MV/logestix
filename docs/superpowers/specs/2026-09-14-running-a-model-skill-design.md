# Running-a-model skill + repo-map completion — Design

Date: 2026-09-14
Status: approved (design dialogue 2026-09-14)

## Problem

The `.opencode/skills/` set covers the **engineering** lifecycle of a model
(add → build → kernels → tests → perf tuning) but not the **operational** one:
how to actually start an inference, where the converted model lives, what the
runtime can be configured with, and how the engine plans memory. The audit
(2026-09-14) found:

1. **No skill covers running inference.** Nothing documents the launch surface:
   `lsxbenchmark --bench/--selfcheck/--bit-exact`, `lsxfabric --localmodel
   --testprompt/--validate`, `lsxforge` conversion flags, `lsxcli
   --info/--infofull`, `taskmosaic`. The vault-path rule ("pass the
   vault ROOT, not `.../logestix`") lives only in a session note
   (`docs/QWEN38_SESSION_SUMMARY.md`). The engine lifecycle
   (`InferenceEngine::Session`), chat framing (`prompt_framing.h`) and sampling
   (`sampling.h`) are mentioned nowhere.
2. **No runtime env-knob catalog.** ~90 `LSX_*`/`KIMI_*` variables are read by
   production and test code; the perf skill documents roughly three
   (`LSX_PERF_AUTO`, `LSX_TEST_FAST_MATH`, `LSX_PERF_VAULT_SUBDIR`).
3. **Resource layer undocumented.** `resource_governor`, `vram_estimate`,
   `lifecycle_strategy`, `runtime_tuner` — the machinery that sizes the expert
   LRU cache — appear in no skill.
4. **`logestix-repo-map` claims "per-file inventory" but lists 45 of 128
   `libs/lsxcommon/src` files.** Entire families/plugs are absent: `gemma_*`,
   `qwen_dense_*`/`qwen_base_model`/`qwen_inference*`, and the model-neutral
   engine/runtime group (`model_engine`, `model_base`, `path_registry`,
   `model_integrity`, `perf_calibration`, `lsx_tc_gemm`, `sampling`,
   `prompt_framing`, `generation_cli`, `bf16_convert`, `glm_moe_load.cu`,
   `nvfp4_codec.h`, `fast_load_internal.h`).

## Non-goals

- No new code, no CUDA, no inference runs. This is a documentation-only change.
- No change to `perf-tuning-a-model` beyond a cross-reference (its gpu_perf
  knobs stay where they are; the new skill points at it for profiling).
- No restructuring of the existing five skills beyond the repo-map fill and
  minimal cross-links.

## Decisions (design dialogue 2026-09-14)

1. A single self-contained skill `running-a-model` (option A), not a split.
2. The env-knob catalog and the resource-layer description live **in the new
   skill**; `perf-tuning-a-model` keeps only perf profiling/baseline discipline.
3. `logestix-repo-map` is completed to a genuine per-file inventory (or its
   claim is corrected), covering the missing families and the engine/runtime
   plug.

## 1. New skill: `running-a-model`

Path: `.opencode/skills/running-a-model/SKILL.md` (tracked in git, auto-
discovered). Style and length follow `ingesting-a-model` / `perf-tuning-a-model`
(frontmatter `name` + `description` with triggers, overview, numbered sections,
verification block, common mistakes).

Sections:

1. **Overview / operational cycle** — build → locate vault model → `--validate`
   → `--selfcheck` → generate → bench. Cross-references: `ingesting-a-model`
   (how the model got into the vault), `perf-tuning-a-model` (raising t/s),
   `logestix-repo-map` (where the code is), `nvfp4-blackwell-conventions`
   (kernel-level rules when a change is needed).
2. **Vault and paths** — `path_registry.h`: platform roots
   (`/mnt/llm_models/models`, Win `D:\Programs\huggingface.co`),
   `model-metadata/`, the `lsx_model_metadata(rel)` invariant, and the
   **gotcha: the CLI takes the vault root, not the `logestix/` subdir**.
3. **Applications and exact invocations** — a table with one canonical command
   per app and its flags (`lsxbenchmark`, `lsxfabric`, `lsxforge`, `lsxcli`,
   `taskmosaic`), plus which preset/target to build first.
4. **Engine lifecycle** — `InferenceEngine::Session`
   (`InitializeEngine → CreateModel/Run → ShutdownEngine`),
   `GenerationCliArgs`/`RunCliGeneration`, `FastMathMode` (apps run `kAuto`;
   `--bit-exact` exists only on lsxbenchmark).
5. **Prompt framing and sampling** — `prompt_framing.h` (`FramingDescriptor`
   owned by each model, no central registry, BOS policy, raw-BPE fallback);
   `sampling.h` (`ISampler`/`GreedySampler`, greedy tie → lowest index).
6. **Resource layer and VRAM planning** — `resource_governor` (HardwareProfile,
   `LeaseKind`, leases), `vram_estimate` (`VramProfile`), `lifecycle_strategy`
   (`Archetype`, `LifecyclePlan`, `TuningBounds`), `runtime_tuner`
   (sliding-window LRU resize). Shows how expert-cache slots are chosen.
7. **Env-knob catalog** — grouped tables marked prod/test/debug:
   - GPU perf: `LSX_PERF_AUTO`, `LSX_TEST_FAST_MATH`, `LSX_GEMV_*`;
   - expert cache per model: `LSX_*_EXPERT_CACHE_SLOTS`, `LSX_FLASH_SLOTS`,
     `LSX_V41_EXPERT_CACHE`, `LSX_EXPERT_PIN`;
   - model toggles: `LSX_KIMI_TC[_OFF]`, `LSX_TC_BATCH_OFF`, `LSX_GRAPH_FFN`,
     `LSX_*_BF16`, `LSX_FLASH_*`;
   - tracing/diagnostics: `KIMI_TRACE_FILE`, `KIMI_DIAG_*`,
     `LSX_TRACE_TOKENS`, `LSX_TRACE_FLASH`, `LSX_PERF_PHASES`,
     `LSX_AUDIT_STAGES`, `LSX_GEMV_TRACE`;
   - test/selfcheck: `LSX_TEST_*_SELFCHECK_CYCLES`, `LSX_TEST_*_MODEL_PATH`,
     `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY`, `LSX_PERF_VAULT_SUBDIR`,
     `LSX_TEST_PERF_GEN_TOKENS`;
   - build: `LSX_ARCH_ZNVER4`, `LSX_LTO`, `LSX_PRESET`, `LSX_LIST`.
8. **Runtime troubleshooting** — OOM / VRAM refusal, pinning a >300 GiB store,
   pageable H2D vs pinned, `ulimit -l unlimited`, TC-path diagnostics, the
   one-GPU-workload rule; pointers to `docs/GEMMA_DEBUG_RESOLVED.md`,
   `docs/KIMI_DEBUG_NEXT.md`, `docs/QWEN38_SESSION_SUMMARY.md`.
9. **Common mistakes**.

Every claim is verified against the tree (headers/parsers) while writing; the
skill must not invent flags. Machine-checked registries stay the source of
truth and the skill points at them.

## 2. `logestix-repo-map` completion

Add the missing entries grouped as the map already groups:

- **gemma family**: `gemma_moe_convert/load/inference/model`, `gemma_base_model`;
- **qwen dense**: `qwen_dense_convert`, `qwen_dense_model`,
  `qwen_dense_27b_model`, `qwen_base_model`, `qwen_inference.cu`,
  `qwen_inference_load`, `qwen_device_math.h`, `qwen_reduce.cuh`;
- **engine/runtime plug**: `model_engine`, `model_base`, `model_integrity`,
  `path_registry`, `resource_governor`, `runtime_tuner`, `lifecycle_strategy`,
  `vram_estimate`, `perf_calibration`, `perf_baseline`, `sampling`,
  `prompt_framing`, `generation_cli`, `lsx_tc_gemm`, `bf16_convert`,
  `glm_moe_load.cu`, `nvfp4_codec.h`, `nvfp4_scale.h`, `fast_load_internal.h`;
- **kernel headers**: `lsx_kernels_{act,common,math,norm,quant}.cuh`,
  `nvfp4_gemv_math.cuh`, `glm_moe_expert_kernels.cuh`;
- fill remaining detail in the grouped `deepseek_moe_v41_*` entry,
  `glm_moe_flash_fp8.*`, `kimi_moe_expert_cache`.

## 3. Anchors

- `AGENTS.md` gains a pointer to the new skill next to the existing skill
  references (Running/Structure), so the operational entry point is
  discoverable from the top-level instruction file.

## Verification

Documentation change only; no build or GPU work.

- Every documented flag/path/knob is cross-checked against the parser/header
  that owns it (`rg` against `apps/*/src/cli.*`, `libs/lsxcommon/include`).
- Skill frontmatter is valid and the `description` carries the trigger terms.
- `git status` shows only the intended files: the new `SKILL.md`, the
  `logestix-repo-map/SKILL.md` edit, this spec, and the `AGENTS.md` anchor.
- No empty placeholders (`TBD`/`TODO`) in either skill.

## Risks

- **Catalog drift.** Env knobs move; the catalog is a snapshot. Mitigation:
  mark it as such and state the grep that regenerates it.
- **Duplication with `perf-tuning-a-model`.** Mitigation: the new skill points
  to perf for profiling/baseline; perf keeps the `gpu_perf` resolution story.
- **Repo-map churn.** Filling it fully makes it long; accept the length since
  the document's stated purpose is the per-file inventory.
