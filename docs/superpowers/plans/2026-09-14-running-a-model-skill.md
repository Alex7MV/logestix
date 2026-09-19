# Running-a-model skill + repo-map completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE (repo policy):** `AGENTS.md` forbids parallel/subagents and requires an
> explicit operator request before any `git commit`. Execute inline, and treat
> every commit step as gated on operator approval.

**Goal:** Add an operational skill `running-a-model` (launch surface, vault paths, engine lifecycle, framing/sampling, resource layer, env-knob catalog) and make `logestix-repo-map` a genuine per-file inventory.

**Architecture:** Documentation-only. One new `SKILL.md` under `.opencode/skills/`, a completion pass over `logestix-repo-map/SKILL.md`, a cross-link in `perf-tuning-a-model/SKILL.md`, and one `AGENTS.md` anchor. Every claim is verified against the owning parser/header before it is written.

**Tech Stack:** Markdown skill files (frontmatter `name`/`description` + body), the logestix C++/CUDA tree as the source of truth.

## Global Constraints

- No production code, no CUDA, no inference run. Docs only.
- `AGENTS.md`: no subagents; do not commit unless the operator explicitly asks.
- Every documented flag/path/knob MUST exist in the tree (verify with `rg` against the owning file).
- No `TBD`/`TODO`/placeholder text in any skill.
- Each skill keeps YAML frontmatter `name:` + a `description:` that carries trigger terms.
- Cross-reference, do not duplicate: profiling/baseline stays in `perf-tuning-a-model`; ingestion protocol stays in `ingesting-a-model`.

---

### Task 1: Author `.opencode/skills/running-a-model/SKILL.md`

**Files:**
- Create: `.opencode/skills/running-a-model/SKILL.md`

**Interfaces:**
- Consumes: the verified facts below (from `apps/*/src/cli.*`, `libs/lsxcommon/include/lsxcommon/*.h`).
- Produces: a discoverable skill whose `name` is `running-a-model`; `logestix-repo-map` and `perf-tuning-a-model` will point at it in Task 2/3.

**Verified facts to embed (do not invent others):**

- Platform roots (`path_registry.h`): Linux `/mnt/llm_models/models`, Windows `D:\Programs\huggingface.co`; subdir `model-metadata/`; helpers `lsx_model_metadata_root()`, `lsx_model_metadata(rel)`, `lsx_model_models_dir()`. Gotcha: pass the model vault ROOT (`<root>/<Model>`), not the converted `logestix/` subdir.
- `lsxbenchmark` (`apps/lsxbenchmark/src/cli.h`): `--model <path>` (default `<models_dir>/DeepSeek-V4-Flash`), `--bench` (default), `--selfcheck`, `--bit-exact`, `--max-gen <n>` (default 2048), `--prompt <p>` (default "The capital of France is"). `--selfcheck` forces bench off. Unknown tokens silently ignored.
- `lsxfabric` (`apps/lsxfabric/src/cli.h`): `--localmodel <path>`, `--validate`, `--testprompt`, `--max-gen <n>` (default 64), `--prompt <p>` (default "The capital of France is"). Benchmark-era flags are ignored.
- `lsxforge` (`apps/lsxforge/src/main.cpp`): `--input`, `--output` (default `converted_model`), `--verify`, `--tokenizer <tokenizer.json>`, `--vram_recompute`, `--repack_shared`, `--repack_experts`, `--run/-r` (generation path, bare = default blueprint prompt), and family pipelines `--glm`, `--kimi`, `--qwen`, `--qwen35moe`, `--qwen36moe`, `--qwen38flashnext`, `--gemma4`, `--glmflash`, `--deepseek_nvfp4`, `--deepseek_v41`.
- `lsxcli` (`apps/lsxcli/src/main.cpp`): `--model <path>`, `--info`, `--infofull` (both route to `runtime::RunCliInspect`).
- `taskmosaic` (`apps/taskmosaic/src/main.cpp`): no CLI flags; init/version/GPU log only.
- `lsxhome`: Windows-only (`if(NOT WIN32) return()` in CMake).
- Engine (`model_engine.h`): `InferenceEngine::Instance()`, nested `Session` RAII (`InitializeEngine`/`ShutdownEngine`), `CreateModel(ModelInitConfig)`, `Run(IModel&, ModelRequest&)`.
- Runtime (`model_runtime.h`): `runtime::RunCliGeneration`, `RunCliBenchmark`, `RunCliInspect`, `DecodeGeneratedText`, `ParisSelfCheckOptions`, unified SelfCheck.
- Framing (`prompt_framing.h`): `prompt::FramingDescriptor` owned per model (e.g. `glm::kFraming`, `gemma::kFraming`), no central registry; raw-BPE fallback; `leading_bos` policy.
- Sampling (`sampling.h`): `sample::ISampler`, `GreedySampler`, `DefaultSampler()`, `LogitType{kF32,kF16,kF32Rounded16}`, tie → lowest index.
- Resource layer: `resource_governor.h` (`HardwareProfile`, `LeaseKind{kPermanentWeights,kExpertCache,kInferArena,kHostPinned,kHostRam}`), `vram_estimate.h` (`VramProfile`), `lifecycle_strategy.h` (`Archetype`, `LifecyclePlan`, `TuningBounds`), `runtime_tuner.h` (sliding-window LRU resize).
- FastMathMode (`fast_math_mode.h`): apps run `kAuto`; `--bit-exact` is lsxbenchmark-only.
- Env knobs (verified via `rg 'getenv\("(LSX_|KIMI_)' libs apps`), grouped prod/test/debug: perf `LSX_PERF_AUTO`, `LSX_TEST_FAST_MATH`, `LSX_GEMV_MAX_MIB`, `LSX_GEMV_HEAD`, `LSX_GEMV_LARGE`, `LSX_GEMV_EXPERTS`, `LSX_GEMV_TRACE`, `LSX_TC_BATCH_OFF`, `LSX_GRAPH_FFN`, `LSX_PROBE_FFN`; cache/residency `LSX_EXPERT_PIN`, `LSX_FLASH_SLOTS`, `LSX_V41_EXPERT_CACHE`, `LSX_V41_RLOC`, `LSX_GLM_EXPERT_CACHE_SLOTS`, `LSX_QWEN_MOE_EXPERT_CACHE_SLOTS`, `LSX_QWEN38_EXPERT_CACHE_SLOTS`, `LSX_GEMMA_EXPERT_CACHE_SLOTS`, `LSX_KIMI_EXPERT_CACHE_SLOTS`, `LSX_DEEPSEEK_EXPERT_CACHE_SLOTS`; model toggles `LSX_KIMI_TC`, `LSX_KIMI_CACHE_BYPASS`, `LSX_GEMMA_*`, `LSX_FLASH_*`, `LSX_DS_*`, `LSX_V41_DEBUG`, `LSX_DEC_ENABLE`/`LSX_DEC_DISABLE`; tracing `KIMI_TRACE_FILE`, `KIMI_DIAG_LAYER`, `KIMI_DIAG_IDS`, `KIMI_CACHE_MAX_LAYER`, `KIMI_NOEVICTION`, `LSX_TRACE_TOKENS`, `LSX_TRACE_FLASH`, `LSX_PERF_PHASES`, `LSX_AUDIT_STAGES`, `LSX_TEST_MOE_STAGE_DEBUG`; tests `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY`, `LSX_PERF_VAULT_SUBDIR`, `LSX_TEST_PERF_GEN_TOKENS`, `LSX_TEST_<MODEL>_SELFCHECK_CYCLES`, `LSX_TEST_<MODEL>_MODEL_PATH`; build `LSX_ARCH_ZNVER4`, `LSX_LTO`, `LSX_PRESET`, `LSX_LIST`.

- [ ] **Step 1: Verify every fact above still holds**

Run: `rg -n 'strcmp|ABSL_FLAG' apps/lsxbenchmark/src/cli.cpp apps/lsxfabric/src/cli.cpp apps/lsxforge/src/main.cpp apps/lsxcli/src/main.cpp`
Expected: the flag names above appear verbatim. If any differs, use the tree value, not this plan.

- [ ] **Step 2: Write the file with frontmatter + sections 1–3**

Frontmatter (verbatim):
```yaml
---
name: running-a-model
description: Use when starting or operating inference in logestix — building and launching lsxbenchmark/lsxfabric/lsxforge/lsxcli/taskmosaic/lsxhome, locating a converted model in the vault, the engine session lifecycle, prompt framing and sampling, the resource-governor/VRAM planner, or looking up a runtime LSX_*/KIMI_* env knob. Triggers - run inference, launch model, lsxbenchmark --bench, lsxfabric --localmodel, vault path, env knob.
---
```
Body sections 1–3: `# Running a model`, `## Overview` (operational cycle + cross-refs to `ingesting-a-model`/`perf-tuning-a-model`/`logestix-repo-map`/`nvfp4-blackwell-conventions`), `## Vault and paths` (roots, `model-metadata/`, `lsx_model_metadata`, the root-not-`logestix/` gotcha), `## Applications and exact invocations` (table with one canonical command per app from the verified facts).

- [ ] **Step 3: Write sections 4–6**

`## Engine lifecycle` (Session, CreateModel/Run, FastMathMode), `## Prompt framing and sampling` (FramingDescriptor ownership + fallback; GreedySampler/tie rule), `## Resource layer and VRAM planning` (Governor/HardwareProfile/LeaseKind, VramProfile, Archetype/LifecyclePlan/TuningBounds, RuntimeTuner sliding window → expert-cache slots).

- [ ] **Step 4: Write section 7 (env-knob catalog)**

One table per group from the verified list, each row `variable — meaning — prod/test/debug`. Include a header line: "snapshot 2026-09-14; regenerate with `rg -o 'getenv\\(\"(LSX_|KIMI_)[A-Z0-9_]+' libs apps`".

- [ ] **Step 5: Write sections 8–9**

`## Runtime troubleshooting` (OOM/VRAM refusal; pinning a >300 GiB store and `ulimit -l unlimited`; pageable-vs-pinned H2D; TC-path diagnostics via `test_kimi_moe_tc_diag`; one-GPU-workload rule; pointers to `docs/GEMMA_DEBUG_RESOLVED.md`, `docs/KIMI_DEBUG_NEXT.md`, `docs/QWEN38_SESSION_SUMMARY.md`), `## Common mistakes` (wrong vault subdir, benchmarking with a stale binary, quoting a 128-token run, running two GPU jobs, `{}`-style assumptions about flags).

- [ ] **Step 6: Verify the skill**

Run: `rg -n 'TBD|TODO|FIXME' .opencode/skills/running-a-model/SKILL.md; wc -l .opencode/skills/running-a-model/SKILL.md`
Expected: no placeholder hits; ~200–280 lines. Then `for t in lsxbenchmark lsxfabric lsxforge lsxcli taskmosaic lsxhome InferenceEngine FramingDescriptor GreedySampler HardwareProfile LeaseKind VramProfile Archetype LifecyclePlan TuningBounds RuntimeTuner; do rg -q "$t" .opencode/skills/running-a-model/SKILL.md || echo "MISSING $t"; done`
Expected: no `MISSING` lines.

- [ ] **Step 7: Commit (gated on operator approval)**

```bash
git add .opencode/skills/running-a-model/SKILL.md
git commit -m "docs: add running-a-model skill (launch surface, engine, knobs)"
```

---

### Task 2: Complete `logestix-repo-map` and add the `AGENTS.md` anchor

**Files:**
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md`
- Modify: `AGENTS.md` (the skills pointer area)

**Interfaces:**
- Consumes: the new skill name `running-a-model` from Task 1.
- Produces: a repo map that references every `libs/lsxcommon/src` file, plus a discoverable operational entry point.

- [ ] **Step 1: Add the gemma + qwen-dense family entries**

Add bullets next to the existing family groups: `gemma_moe_convert.cpp`, `gemma_moe_load.cpp`, `gemma_moe_inference.cu`, `gemma_moe_model.{h,cpp}`, `gemma_base_model.{h,cpp}`; `qwen_dense_convert.cpp`, `qwen_dense_model.{h,cpp}`, `qwen_dense_27b_model.{h,cpp}`, `qwen_base_model.{h,cpp}`, `qwen_inference.cu`, `qwen_inference_load.cpp`, `qwen_device_math.h`, `qwen_reduce.cuh`.

- [ ] **Step 2: Add the engine/runtime plug entries**

Add: `model_engine.{h,cpp}`, `model_base.{h,cpp}`, `model_integrity.{h,cpp}`, `path_registry.{h,cpp}`, `resource_governor.{h,cpp}`, `runtime_tuner.{h,cpp}`, `lifecycle_strategy.{h,cpp}`, `vram_estimate.{h,cpp}`, `perf_calibration.{h,cu}`, `perf_baseline.{h,cpp}`, `sampling.{h,cu}`, `prompt_framing.{h,cpp}`, `generation_cli.{h,cpp}`, `lsx_tc_gemm.{h,cu}`, `bf16_convert.cu`, `glm_moe_load.cu`, `nvfp4_codec.h`, `nvfp4_scale.h`, `fast_load_internal.h`.

- [ ] **Step 3: Add kernel-header and remaining entries**

Add: `lsx_kernels_{act,common,math,norm,quant}.cuh`, `nvfp4_gemv_math.cuh`, `glm_moe_expert_kernels.cuh`, `glm_moe_flash_fp8.{h,cpp}`, `kimi_moe_expert_cache` detail, `deepseek_moe_inference_{attn,csa,head,moe,pipeline}.cu`.

- [ ] **Step 4: Correct the document's claim and cross-link the new skill**

Change the intro claim from an implied complete inventory to either a complete one (preferred, after Steps 1–3) or an explicit "curated" caveat; add `running-a-model` to the cross-reference line (`Model ingestion ... is ingesting-a-model; ... throughput work is perf-tuning-a-model`).

- [ ] **Step 5: Add the `AGENTS.md` anchor**

In the Structure/skills area, add one line pointing to `running-a-model` for "running/operating inference" (vault path, apps, env knobs).

- [ ] **Step 6: Verify the map is now complete**

Run:
```bash
S=.opencode/skills/logestix-repo-map/SKILL.md
for p in libs/lsxcommon/src/*; do b=$(basename "$p"); s="${b%.*}"; rg -q --fixed-strings "$s" "$S" || echo "MISSING $s"; done
```
Expected: no `MISSING` lines (or only deliberately grouped ones documented as such). Then confirm `running-a-model` is referenced in both the map and `AGENTS.md`.

- [ ] **Step 7: Commit (gated on operator approval)**

```bash
git add .opencode/skills/logestix-repo-map/SKILL.md AGENTS.md
git commit -m "docs: complete repo map (gemma, qwen dense, engine layer) + running-a-model anchor"
```

---

### Task 3: Cross-link `perf-tuning-a-model` and run the spec-coverage check

**Files:**
- Modify: `.opencode/skills/perf-tuning-a-model/SKILL.md`

**Interfaces:**
- Consumes: `running-a-model` (Task 1).
- Produces: a one-line pointer so a reader profiling a model can find the launch surface.

- [ ] **Step 1: Add the pointer**

In the overview or Step 1, add: "Launching/operating the engine (apps, vault paths, env knobs) is the `running-a-model` skill; this skill owns profiling and baselines."

- [ ] **Step 2: Spec coverage self-review**

Read `docs/superpowers/specs/2026-09-14-running-a-model-skill-design.md` section by section; confirm each of the 9 sections and the repo-map additions is present in the delivered skills. Record the mapping in the task notes.

- [ ] **Step 3: Final verification**

Run: `rg -n 'TBD|TODO|FIXME' .opencode/skills/ ; git status --short`
Expected: no placeholders; only the intended files changed (new SKILL.md, two skill edits, `AGENTS.md`, the spec/plan docs). No build, no GPU run.

- [ ] **Step 4: Commit (gated on operator approval)**

```bash
git add .opencode/skills/perf-tuning-a-model/SKILL.md
git commit -m "docs: cross-link perf-tuning to running-a-model"
```

---

## Self-Review

- **Spec coverage:** spec sections 1–3 → Task 1 (sections 1–9) ; spec §2 (repo-map) → Task 2 ; spec §3 (anchor) → Task 2 Step 5 ; verification → Task 3. No gaps.
- **Placeholders:** none — every task lists exact files, commands, and the verified fact set.
- **Type/name consistency:** the skill `name: running-a-model` is used identically in Tasks 1–3 and in the spec.
