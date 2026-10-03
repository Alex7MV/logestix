---
name: running-a-model
description: Use when starting or operating inference in logestix — building and launching lsxbenchmark/lsxfabric/lsxforge/lsxcli/taskmosaic, locating a converted model in the vault, the engine session lifecycle, prompt framing and sampling, the resource-governor/VRAM planner, or looking up a runtime LSX_*/KIMI_* env knob. Triggers - run inference, launch model, lsxbenchmark --bench, lsxfabric --localmodel, vault path, env knob.
---

# Running a model

## Overview

The operational half of the model lifecycle: build the binary, find the
converted model, validate it, run the Paris self-check, generate, then measure.
The engineering half is elsewhere:

- `ingesting-a-model` — how a model got into the vault (protocol, naming, guards).
- `perf-tuning-a-model` — how to raise t/s (nsys attribution, budget tests,
  baselines). This skill only says *which* command to run, not how to profile.
- `logestix-repo-map` — which file owns what.
- `nvfp4-blackwell-conventions` — kernel rules, only if you have to change code.

Machine-checked registries (`test_model_suite_guard`, `kPassports`, …) stay the
source of truth for whether a model is complete; this skill is the operator's
entry point.

## 1. Vault and paths

The model root defaults to a compile-time platform constant in
`libs/lsxcommon/include/lsxcommon/path_registry.h`, overridable at runtime with
the `LSX_MODEL_ROOT` environment variable (an absolute path to the models
directory):

| platform | `kModelRootBase` (default) | converted models live in |
|---|---|---|
| Linux | `/mnt/llm_models/models` | `<base>/<Model>/logestix/` |
| Windows | `D:\Programs\huggingface.co` | `<base>\<Model>\logestix\` |

- Every vault read/write goes through `lsx_model_metadata_root()` /
  `lsx_model_metadata(rel)` / `lsx_model_models_dir()` — **no other code may
  reconstruct the root** (the `model-metadata/` block is appended by the
  helper). The metadata vault is `<base>/model-metadata/` (passport, perf cache).
- A converted model `lsxforge` output lands in `<output>/logestix/` and carries
  `logestix/model.index.json`, `logestix/pinned/`, and per-expert `.arrow` files.
- **Gotcha:** the CLI apps take the **model root** (`<base>/<Model>`), *not* the
  `logestix/` subdirectory. `--localmodel /mnt/llm_models/models/gemma-4-26B-A4B-it`
  is right; `.../gemma-4-26B-A4B-it/logestix` silently fails to load.

## 2. Applications and exact invocations

Build first (perf numbers require Release; see `perf-tuning-a-model`):

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug            # fast tests
cmake --build build/linux-release-znver4-aocc                               # perf reference
```

| app | purpose | canonical invocation |
|---|---|---|
| `lsxbenchmark` | measure t/s + bit-exactness (authoritative) | `lsxbenchmark --model <root> --bench --max-gen 2048 --prompt "<recorded>"` |
| `lsxbenchmark` | Paris self-check | `lsxbenchmark --model <root> --selfcheck` |
| `lsxbenchmark` | bit-exact measurement | `lsxbenchmark --model <root> --bench --bit-exact` |
| `lsxfabric` | generate exactly as the engine does | `lsxfabric --localmodel <root> --testprompt --prompt "<p>" --max-gen 64` |
| `lsxfabric` | load-validate a converted model through the engine | `lsxfabric --localmodel <root> --validate` |
| `lsxforge` | convert a source checkpoint | `lsxforge --input <src> --output <src> --<family>` —
  `--output` is the **model root**, not the destination: every converter appends
  `logestix/` itself, so `--output <src>/logestix` silently produces
  `<src>/logestix/logestix` |
| `lsxforge` | one-shot generation path | `lsxforge --input <root> --run "<prompt>"` (or `-r`) |
| `lsxcli` | inspect a converted model | `lsxcli --model <root> --info` / `--infofull` |
| `taskmosaic` | orchestrator stub (init + GPU log) | `taskmosaic` (no flags) |

Flag details (verified against `apps/*/src/cli.*`):

- **lsxbenchmark** — defaults: `--bench` on, `--max-gen 2048`,
  `--prompt "The capital of France is"`, model defaults to
  `<models_dir>/DeepSeek-V4-Flash`. `--selfcheck` turns bench off and selects
  verification mode. Unknown tokens are ignored.
- **lsxfabric** — defaults: `--prompt "The capital of France is"`,
  `--max-gen 64`. Benchmark-era flags (`--benchmark`, `--selfcheck`,
  `--bit-exact`, `--fast-math`) are **silently ignored** — speed measurement
  lives in lsxbenchmark.
- **lsxforge** — `--input` / `--output` (default `converted_model`), plus
  **one** pipeline flag; passing two family flags is a hard error (it used to
  silently run the first). All 23 flags are `ABSL_FLAG`s in
  `apps/lsxforge/src/main.cpp`:
  - family: `--glm`, `--glmflash`, `--qwen`, `--qwen35moe`, `--qwen36moe`,
    `--qwen38flashnext`, `--gemma4`, `--kimi`, `--ornith`, `--nemotron`,
    `--gigachat`, `--mimo`, `--granite`, `--phi`, `--alice`
  - deepseek: `--deepseek_nvfp4` (offline expert repack E8M0/32 → NVFP4
    E4M3/16), `--deepseek_v41` (V4.1 conversion)
  - utilities: `--tokenizer <tokenizer.json>`, `--vram_recompute`,
    `--repack_shared`, `--repack_experts`, `--run/-r <prompt>`
  - Dispatch order (`cli.cpp`): utilities → repack → deepseek → family.
- **lsxcli** — `--info` prints the model summary; `--infofull` also prints every
  shard. Both route to `runtime::RunCliInspect` and add the conversion-time VRAM
  profile (min / optimal / max).

## 2a. Fast launch on the built-in engine (the ordered path)

There is no speed flag — the engine resolves a per-GPU policy at
`InitializeEngine()`. "Fast" therefore means: the right build, a free GPU, the
canonical protocol, and a measurement you can compare. Run these in order; stop
at the first red step.

```bash
# 0. Pick the build: Debug for correctness, the AOCC/Release tree for speed.
cmake --build --preset linux-debug                 # correctness
ninja -C build/linux-release-znver4-aocc lsxbenchmark lsxfabric   # speed

# 1. Free the GPU. One workload at a time — contention skews t/s by up to 2x.
nvidia-smi --query-gpu=memory.used --format=csv,noheader

B=./build/linux-release-znver4-aocc/apps

# 2. Does the converted layout load through the engine?
$B/lsxfabric/lsxfabric --localmodel <vault-root> --validate

# 3. Does it answer correctly? (Paris; also prints logits_hash.)
$B/lsxbenchmark/lsxbenchmark --model <vault-root> --selfcheck

# 4. How fast, in the canonical protocol? Same prompt + --max-gen as the
#    baseline entry in tests/perf/perf_baselines.json, else it is not comparable.
$B/lsxbenchmark/lsxbenchmark --model <vault-root> --bench --max-gen 2048 \
    --prompt "The capital of France is"

# 5. Serve. ROUTER for requests, PULL for cancel; the client must see
#    incremental token frames (not one final DONE) or streaming is broken.
$B/lsxfabric/lsxfabric --localmodel <vault-root> --serve \
    --listen tcp://127.0.0.1:5555 --listen-cancel tcp://127.0.0.1:5556

# 6. Bridge it (OpenAI + Anthropic surfaces, tool calling). The config binds
#    a model id to the fabric endpoints from step 5 (apps/taskmosaic/src/config.h):
#    {"models":[{"id":"my-model","model_path":"/mnt/llm_models/models/<Model>",
#      "backend":{"endpoint":"tcp://127.0.0.1:5555",
#                 "cancel_endpoint":"tcp://127.0.0.1:5556"}}],
#     "default_model":"my-model"}
$B/taskmosaic/taskmosaic --config <config.json>
```

Reading the numbers:

- Step 4 also prints `resident:` — if it is 0 MiB or wildly below the
  conversion-time VRAM profile, the model is not publishing its residency.
- Throughput below the baseline is a perf problem, not a launch problem: go to
  `perf-tuning-a-model` (attribute the wall time first; do not tune blind).
- `LSX_PERF_AUTO=0` restores the legacy bit-exact defaults — an A/B switch, not
  a speed-up. `--bit-exact` is the verification mode (`FastMathMode::kOff`).
- Loading is per-process and can dominate wall time (Kimi re-pins 544 GiB:
  budget minutes and `ulimit -l unlimited`).

## 2b. Acceptance — when may a model be called "supported"?

Ingestion is not done until all of these are green for THAT family (the
developer-side checklist is `ingesting-a-model` § Definition of done):

| # | Gate | Command |
|---|---|---|
| 1 | the shared fast suite (guards included) | `ctest --preset linux-debug` |
| 2 | convert/load/model/kernels stages registered and green | `ctest --preset linux-debug-convert`, then the family's `inference` targets |
| 3 | SelfCheck: Paris + ≥3 bit-exact cycles | `LSX_TEST_<MODEL>_MODEL_PATH=<root> ctest -R <family>_selfcheck` |
| 4 | FSI / CODE / DET goldens recorded and green | `ctest -R "<family>_(fsi|code|det)_selfcheck"` with the model's path env |
| 5 | tool calling: native render + parse round-trip | `ctest -R <family>_tool_framing` |
| 6 | perf baseline entry + perf-labelled target green (Release only) | `ctest --preset linux-release-znver4-perf` |
| 7 | no HIGH item open for the family | `docs/OPEN_TASKS.md` |

Stages 3–4 need real weights and `SKIP` (exit 4) without them: a green sweep
that skipped every selfcheck is not acceptance, it is absence of evidence.
Steps 2–5 are enforced structurally by `test_model_suite_guard` (`Suites()`),
which fails if a stage target is missing, unclaimed, or if a loader/model stage
touches the vault.

## 3. Engine lifecycle

`libs/lsxcommon/include/lsxcommon/model_engine.h`:

- `InferenceEngine::Instance()` is a strict thread-safe singleton.
  Lifecycle: `InitializeEngine()` → `[CreateModel/Run]*` → `ShutdownEngine()`.
- The nested RAII guard `InferenceEngine::Session` drives that in a scope
  (a local `Session` at the top of `main` is what every app uses; it also
  resolves the per-GPU perf policy and logs during shutdown).
- `CreateModel(ModelInitConfig)` opens a model through `LsxModelFactory`;
  `Run(IModel&, ModelRequest&)` validates the lifecycle, checks the request's
  abort flag, calls `model.Infer`, and streams tokens via `req.token_callback`
  on the calling thread (no internal threads).
- The unified CLI helpers in `model_runtime.h` wrap Create → Load → frame/encode
  → run → print → Unload: `RunCliGeneration`, `RunCliSelfCheck`,
  `RunCliBenchmark`, `RunCliInspect`. `GenerationCliArgs.fast_math_mode` defaults
  to `FastMathMode::kAuto` (per-GPU maximum speed); `kOff` is the bit-exact
  scalar path and is reachable only through lsxbenchmark `--bit-exact`.

## 4. Prompt framing and sampling

- **Framing** (`libs/lsxcommon/include/lsxcommon/prompt_framing.h`): each framed
  model OWNS its `prompt::FramingDescriptor` next to its model code (`glm::kFraming`,
  `gemma::kFraming`) and exposes it through `IModel::BuildPromptInputIds`. There is
  deliberately **no central `model_type → framing` registry** — a new model cannot
  silently miss an entry, and BOS policy cannot leak across families. Sentinels are
  resolved via `arrow_tokenizer_lookup`; if any sentinel is missing the framing
  falls back to the raw BPE encode. `leading_bos` relocates the auto-prepended
  `bos_id` to position 0 for templates that start with `{{ bos_token }}`.
- **Sampling** (`libs/lsxcommon/include/lsxcommon/sampling.h`): `sample::ISampler`
  with `GreedySampler` as the default (`DefaultSampler()`), matching the engines'
  fused-argmax semantics — on ties the FIRST (lowest index) maximum wins. Rows may
  be FP32, FP16, or FP32-compared-as-FP16-rounded (`LogitType`).

## 5. Resource layer and VRAM planning

The engine, not the model, owns memory. Models ask for a plan and leases.

- `resource_governor.h` — `ResourceGovernor` is an engine-level singleton
  initialized by `InitializeEngine()`. It probes a `HardwareProfile` (CPU/RAM/GPU
  name/SM count/VRAM) and grants RAII `Lease`s whose `LeaseKind` is one of
  `kPermanentWeights`, `kExpertCache`, `kInferArena`, `kHostPinned`, `kHostRam`. A `kRequired` lease
  fails deterministically on shortage; `kAdaptive` is clamped to
  `[min_bytes, max_bytes]` and can be soft-resized. A `kDefaultVramKeepOut`
  (512 MiB) is never granted. **TEST-ONLY** `GovernorOptions.relax_memory_refusal`
  disables strict refusal — an in-process field, **not** an environment
  variable (there is no `LSX_*_RELAX_VRAM_REFUSAL` knob).
- `vram_estimate.h` — `VramProfile{min, optimal, max}` computed at conversion time
  (NVFP4 size = `numel/2` nibbles + `numel/16` E4M3 scales; norms/biases keep raw
  bytes). Stored in `model.index.json`'s `"vram"` block; `--vram_recompute`
  regenerates it. Context length is read from `config.json`
  (`max_position_embeddings → context_length → seq_length → 2048`).
- `lifecycle_strategy.h` — `Archetype{General,Dense,MoE}` from the manifest
  (non-empty `moe_layers[]` → MoE); `LifecyclePlan` fixes permanent residency,
  `expert_cache_slots` (start value inside `TuningBounds`), copy streams, staging
  ring, upload threads.
- `runtime_tuner.h` — `RuntimeTuner` grows/shrinks the expert-cache lease with a
  sliding window (grow on sustained misses, shrink on sustained hits, hysteresis
  between resizes). Reports arrive **between** inference requests, never in the
  hot loop; growth goes through the governor and can be soft-refused.

## 6. Env-knob catalog

Snapshot 2026-09-14. Regenerate the raw list with:

```bash
rg -o 'getenv\("(LSX_|KIMI_)[A-Z0-9_]+"' libs apps | sort -u
```

**Prod — vault**

| variable | meaning |
|---|---|
| `LSX_MODEL_ROOT` | absolute path to the models directory (overrides the compile-time `kModelRootBase`) |

**Prod — perf / dispatch**

| variable | meaning |
|---|---|
| `LSX_PERF_AUTO` | `=0` returns the legacy bit-exact `gpu_perf` defaults |
| `LSX_GEMV_MAX_MIB` | seq==1 weight-streaming GEMV size limit (MiB) |
| `LSX_GEMV_LARGE` | opt into the large-GEMV dispatch class |
| `LSX_GEMV_HEAD` | `=1` also routes the LM-head class through the GEMV |
| `LSX_GEMV_EXPERTS` | routed-expert GEMV dispatch override |
| `LSX_TC_BATCH_OFF` | disable the Qwen TC batch chain |
| `LSX_GRAPH_FFN` | route the Qwen MoE FFN through a CUDA graph |
| `LSX_DEC_ENABLE` | opt into decode-tile variant (Phase-3 A/B) in `lsx_tc_gemm` |

**Prod — residency / cache**

| variable | meaning |
|---|---|
| `LSX_EXPERT_PIN` | `=0` disables routed-expert host pinning (pageable, slow) |
| `LSX_FLASH_SLOTS` | GLM-5.3-Flash expert LRU slot override |
| `LSX_V41_EXPERT_CACHE` | DeepSeek-V4.1 expert-cache override |
| `LSX_*_EXPERT_CACHE_SLOTS` | per-model LRU slot override (`GLM`, `QWEN_MOE`, `QWEN38`, `GEMMA`, `KIMI`, `DEEPSEEK`, `NEMOTRON`, `GIGACHAT`, `ORNITH_MOE`, `MIMO`) |
| `LSX_KIMI_CACHE_BYPASS` | bypass the Kimi expert cache |

**Prod — numeric / model toggles** (A/B switches; leave unset in production)

`LSX_KIMI_TC`, `LSX_GEMMA_GEMV`, `LSX_GEMMA_MLP_BF16`, `LSX_GEMMA_ATTN_BF16`,
`LSX_GEMMA_INLINE_UPLOADS`, `LSX_GEMMA_MOE_BATCH`, `LSX_FLASH_GATE`,
`LSX_FLASH_QSCALE`, `LSX_FLASH_HEAD`, `LSX_FLASH_CONV_SILU`, `LSX_DS_PREFETCH`,
`LSX_DS_PREFETCH_SYNC`, `LSX_V41_DEBUG`, `LSX_DEC_ENABLE`, `LSX_EXP_FFN_NOSBATCH`.

Removed knobs (documented here in the past, no longer read by any TU):
`LSX_KIMI_TC_OFF`, `LSX_GEMMA_HEAD_BF16`, `LSX_GEMMA_KV_FP8`,
`LSX_GEMMA_DUAL_STREAM`. Note `LSX_DEC_ENABLE` (not `..._DISABLE`) is the
decode-tile switch — see the table above.

**Debug / tracing** (never in production)

`KIMI_TRACE_FILE=<path>` (per-layer top-8 trace), `KIMI_DIAG_LAYER`,
`KIMI_DIAG_IDS`, `KIMI_CACHE_MAX_LAYER`, `KIMI_NOEVICTION`, `KIMI_ALLOC`,
`LSX_KIMI_DUMP_L1` / `LSX_KIMI_DUMP_DONE` (L1 input capture for `test_kimi_moe_tc_diag`),
`LSX_TRACE_TOKENS`, `LSX_TRACE_FLASH`, `LSX_PERF_PHASES`, `LSX_AUDIT_STAGES`,
`LSX_FLASH_GEMV_DEBUG`, `LSX_FLASH_LOAD_DEBUG`,
`LSX_GEMV_TRACE`, `LSX_PROBE_FFN`, `LSX_DS_FFN_PROBE`,
`LSX_DS_PF_NOOP` / `LSX_DS_PF_NOPRE` (DeepSeek prefetch A/B),
`LSX_DS_VERIFY_COPY` / `LSX_DS_VERIFY_READ` (DeepSeek copy/read verify),
`LSX_V41_RLOC=<slots>` (DeepSeek-V4.1 routed-expert locality probe),
`LSX_FLASH_*_DEBUG`, `LSX_TEST_MOE_STAGE_DEBUG`.

**Tests / selfcheck** (read by test targets; see `ingesting-a-model`)

Almost all of these are read by test executables only, and a test whose weights
are absent `SKIP`s (exit 4) rather than failing. Four exceptions are read by
PRODUCTION code and must stay unset in a real run:
`LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` (the shared
`synthetic_geometry.h` gate every converter/loader/model checks),
`LSX_TEST_FAST_MATH` (`gpu_perf.h`), and the Gemma ablation set
`LSX_TEST_GEMMA_{DEBUG_SYNC,NO_ATTN,NO_DENSE,NO_MOE,HASH}`
(`gemma_moe_inference.cu`).

**Set by CTest — you never pass them:**

| variable | meaning |
|---|---|
| `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1` | relax the frozen-geometry gates for converter/loader/model fixtures (31 targets set it) |
| `LSX_PERF_VAULT_SUBDIR=<subdir>` | vault subdir per `test_perf_baseline_*` target (16 targets) |
| `LSX_TEST_NEMOTRON_ROOT=<dir>` | vault path for the nemotron selfcheck |

**Operator-supplied — the real-weight suites `SKIP` until you set these:**

| variable | meaning |
|---|---|
| `LSX_TEST_<MODEL>_SELFCHECK_CYCLES` | selfcheck stress-cycle count. One per family, and the stem is NOT the kType: `QWEN` (dense), `QWEN27`, `QWEN_MOE`, `QWEN36`, `QWEN38`, `GLM` (GLM-5.2; GLM-5.3-Flash reuses `LSX_TEST_GLM_SELFCHECK_CYCLES`), `GEMMA`, `KIMI`, `DEEPSEEK`, `DEEPSEEK_V41`, `GIGACHAT`, `GRANITE`, `PHI`, `ORNITH`, `NEMOTRON`, `MIMO` |
| `LSX_TEST_<MODEL>_MODEL_PATH` / `_ROOT` / `_MODEL_DIR` | converted-model directory. Same stems as above, suffix varies: `_MODEL_PATH` = `QWEN`, `QWEN27`, `QWEN_MOE`, `QWEN36`, `DEEPSEEK`, `GEMMA`, `GRANITE`, `KIMI`, `MIMO`, `ORNITH`, `PHI`; `_ROOT` = `ALICE`, `DEEPSEEK_V41`, `GIGACHAT`, `GLM_FLASH`, `NEMOTRON`; `_MODEL_DIR` = `QWEN38`; unprefixed `LSX_TEST_MODEL_PATH` = the `test_model_factory` real-model case |
| `LSX_TEST_<MODEL>_SOURCE_PATH` | source safetensors dir for the parity captures: `QWEN_MOE`, `QWEN36`, `ORNITH` |
| `LSX_TEST_PERF_GEN_TOKENS` | generated length for the `test_blackwell_perf_*` targets |
| `LSX_TEST_KIMI_GEN_TOKENS` | generated length for `test_kimi_moe_model` |
| `LSX_TEST_FAST_MATH` | force the TC path in tests |
| `LSX_TEST_GEMMA_DEBUG` | Gemma selfcheck debug print (read by the test) |
| `LSX_TEST_GEMMA_{DEBUG_SYNC,HASH,NO_ATTN,NO_DENSE,NO_MOE}` | Gemma ablation/debug aids read by the **production** TU `gemma_moe_inference.cu` — unset outside a debug session |
| `LSX_TEST_ALICE_TINY_{HF,OUT}` | HF checkpoint dir + converted output for the alice tiny-HF parity case; `_IDS` (comma list) and `_EXPECT` (argmax) parameterise it |

Regenerate this list instead of trusting it:
`grep -rhoE '"LSX_TEST_[A-Z0-9_]+"' tests/ | sort -u` (cross-check against
`grep -oE 'LSX_[A-Z0-9_]+' tests/CMakeLists.txt` for the CTest-set ones).

**Build / router**

`LSX_ARCH_ZNVER4`, `LSX_LTO` (CMake options), `LSX_PRESET`, `LSX_LIST`,
`LSX_DIFF` (test router; see AGENTS.md § Test).

## 7. Runtime troubleshooting

- **OOM / VRAM refusal** — the governor's `kRequired` leases fail deterministically;
  the log names the lease kind. Free VRAM first (the rule is one GPU workload at a
  time). Tests that need the continue-anyway semantics set
  `GovernorOptions::relax_memory_refusal` directly — there is no env knob for
  it (an earlier version of this skill named `LSX_TEST_RELAX_VRAM_REFUSAL`,
  which never existed).
- **Pinning a large routed-expert store** — a single `cudaHostAlloc` > 300 GiB is
  refused by the driver; the store is pinned in chunks
  (`ExpertStore::PinChunked`). Kimi re-pins 544 GiB and needs
  `ulimit -l unlimited`, otherwise pinning fails. `LSX_EXPERT_PIN=0` falls back
  to pageable (much slower).
- **Pageable H2D** — ~19 GB/s on the compute stream means the source is pageable
  (pin it or make it device-resident); ~57 GB/s is the Gen5 ceiling. Attribution
  recipe: `perf-tuning-a-model` Step 2.
- **TC-path correctness** — `test_kimi_moe_tc_diag` documents why Kimi's W4A4
  chain is numerically unusable (`LSX_KIMI_TC` stays off).
- **GPU exclusivity** — never run two GPU tasks concurrently; see §8.
- Session notes worth reading: `docs/GEMMA_DEBUG_RESOLVED.md`,
  `docs/KIMI_DEBUG_NEXT.md`, `docs/QWEN38_SESSION_SUMMARY.md`.

## 8. Host resource policy and GPU scheduling

Host CPU/RAM are never the constraint on the dev boxes: during builds and
tooling the CPU sits around ~10% and RAM around ~35%, so do not throttle
parallelism, threads/cores, or memory usage for ordinary tool work. Under Linux
there is nothing to economize at all.

The **exception** — both OS profiles — is inference and inference tests
(`lsxbenchmark --bench`/`--selfcheck`, `--testprompt`, infer tests, CUDA kernels
on real weights). There memory and GPU are critical:

- Run **one GPU workload at a time**. No parallel inference, CUDA test, or perf
  suite.
- `lsxbenchmark` is a **strict GPU-exclusive**: VRAM/SM contention skews t/s by
  up to ~2x, which makes the number worthless for the perf reference. Do not run
  it alongside anything else on the GPU.
- Watch VRAM/RAM and unload models when done; do not duplicate a running
  inference.
- **Perf comes last**, after every other GPU task — `test_perf_baseline_*` reads
  the committed `tests/perf/perf_baselines.json`, so a contaminated measurement
  would be recorded as the reference.

## Common mistakes

- Passing the `logestix/` subdirectory instead of the model root to `--model` /
  `--localmodel`.
- Measuring with `lsxfabric`: it has no numeric flag — use `lsxbenchmark --bench`.
- Quoting a 128-token run as the baseline (warmup-affected; canonical is 2048).
- A stale binary: rebuild the app target before measuring.
- Unsetting `LSX_EXPERT_PIN` on Kimi and then wondering why loading takes minutes.
- Leaving a debug/trace knob (e.g. `KIMI_TRACE_FILE`, `LSX_PERF_PHASES`) set and
  reading the resulting timings as production numbers.

## Verification

Binaries land at `build/<preset>/apps/<app>/<app>`.

```bash
cmake --build --preset linux-debug                       # build the app target
./build/linux-debug/apps/lsxcli/lsxcli --model <root> --info     # inspect
./build/linux-debug/apps/lsxfabric/lsxfabric --localmodel <root> --testprompt
./build/linux-debug/apps/lsxbenchmark/lsxbenchmark --model <root> --selfcheck
# perf (Release, GPU-exclusive, last):
./build/linux-release-znver4-aocc/apps/lsxbenchmark/lsxbenchmark \
    --model <root> --bench --max-gen 2048 --prompt "<recorded>"
```
