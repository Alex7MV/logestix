# logestix — AGENTS.md

## Project

Multi-app monorepo: fabric, forge, taskmosaic, lsxcli. C++20 / C23. CMake 3.25+ (CMakePresets).
Target architecture: NVIDIA Blackwell (CC 12.0 / 12.1). Active use of 5th-gen Tensor Cores and TMA.
Max warps per SM on Blackwell: 48 (factor into occupancy calculations).
Forbidden architectures: Hopper (90), Ada (89), Ampere (80).
No backward compatibility — any code targeting pre-Blackwell (sm_90, sm_80) is rejected.
CUDA Toolkit 13.4 (nvcc V13.4.92; minimum 13.2+), full native C++23 support in NVCC.

Open, triaged follow-up work (real defects, Windows platform gaps,
hardware-dependent budgets, missing coverage) — `docs/OPEN_TASKS.md`. It keeps
only genuinely open work; resolved items move to a dated snapshot in
`docs/open-tasks/` (index: `docs/open-tasks/README.md`). The
`fast` label is green (86/86); `inference` is not, and the reasons are
catalogued there rather than left to be rediscovered.

## Dependencies

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 25.0.1 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.17.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v4.6.11 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | 3.4.2 | submodule (`third_party/cccl`) | Header-only |
| abseil-cpp | 20260817.0 | FetchContent (`20260817.0`, LTS August 2026) | Header-only + static libs |
| Catch2 | v3.16.0 | FetchContent (test only) | Header-only |
| cpp-httplib | v0.58.0 | FetchContent (host-only HTTP) | Header-only |

Versions are enforced by code: `tests/test_core.cpp` → `TEST_CASE("pinned dependency
versions")` (`STATIC_REQUIRE` for Arrow, spdlog, simdjson, Catch2, cpp-httplib,
CCCL, abseil — the latter via `ABSL_LTS_RELEASE_VERSION`, so a non-LTS pin will
not build); there is no guard for `flatbuffers`/`libzmq`. The dependency bump
procedure
(pin the submodule/FetchContent `GIT_TAG` + guard + AGENTS/README tables
+ re-check after-fetch patches) — skill `cuda-cmake-build-gotchas`.

## Build

Presets, the AOCC build, the Windows toolchain (VS 2026) with exact paths, toolchain
minimums and binary locations — skill **`building-the-project`**
(`.opencode/skills/building-the-project/SKILL.md`).

Quick check:

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug                     # fast only
```

Non-obvious build pitfalls — skill **`cuda-cmake-build-gotchas`**
(`.opencode/skills/cuda-cmake-build-gotchas/SKILL.md`): CMake 4.x and
`enable_language(CUDA)` in a nested directory, host flags leaking into NVCC,
`/Zc:preprocessor` vs CCCL/thrust, `CMAKE_POLICY_VERSION_MINIMUM` for xsimd,
Arrow/Boost/abseil patches, `lsx_tune`/LTO gating, the `cc12x`-only arch set, host-only
CUDA surfaces and the `_putenv` guard, dependency bumps. It also has the
"symptom → cause" table.

## Resources (host)

Host-machine resource policy, the inference exemption and the
"one GPU load at a time" rule (including strict GPU exclusivity for `lsxbenchmark` and
running perf last) — the "Host resource policy and GPU scheduling" section of skill
**`running-a-model`**
(`.opencode/skills/running-a-model/SKILL.md`).

## Test

`BUILD_TESTING=ON` is enabled in all presets. Test categories (labels
`fast`/`convert`/`inference`/`perf`), presets, the path router
(`tools/test_router.cmake`, `LSX_PRESET`/`LSX_LIST`/`LSX_DIFF`), label discipline
in the same commit, the guard-test registry and the WINDOWS restriction
(`ctest -R "qwen|gemma"`) — skill **`test-lifecycle`**
(`.opencode/skills/test-lifecycle/SKILL.md`).

Converter test rules (label `convert`, bounded TIMEOUT, synthetic
checkpoint, `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY` as a runtime env) — the
"Converter test rules" section in `ingesting-a-model`; the measurement/speed-up
procedure — `perf-tuning-a-model`.

## Structure

- `libs/lsxcommon/` — shared STATIC library; namespace `lsxcommon`. Core:
  `core.cpp` (logging/init), `schema.cpp` (ExpertSchema/PackedExpert),
  `converter.cpp` (GPU NVFP4 quantization), `tokenizer.cpp` +
  `pretokenizer.cpp`/`unicode.cpp`, `model_factory.cpp`/`model_registry.cpp`,
  `model_runtime.cpp` (lsxbenchmark --bench), `gpu_perf.cpp`, `token_stream.h`
  (`GenerationStream`: CUDA-safe per-token callback + abort, threaded into every
  family generation loop; `Emit(token, step, logit_maxv)`. The shared lifecycle
  contract each family must wire — residency lock, streaming/abort, context
  teardown, governor lease, result fill, session state under the lock — is
  machine-enforced per model by the `kLifecycle` registry in
  `test_model_independence_guard.cpp`), `chat_conversation.h`/`tool_format.{h,cpp}` (tool
  calling render/parse), `incremental_detokenizer.{h,cpp}`, the model-neutral
  layer `lsx_inference_nvfp4.cu` + device-independent GEMV `lsx_gemv.cu`
  (`RunLxGemmVec`/`RunLxGemmVecWT`; the only GEMM path), `expert_store.h`,
  `lsx_device_math.h` (model-neutral glibc-replica `ExpfGlibc`/`RsqrtfGlibc`;
  `glm_moe_device_math.h` aliases it) and `lsx_reduce.cuh` (model-neutral
  `BlockMaxShfl`; `qwen_reduce.cuh` aliases it); `synthetic_geometry.h`
  (`SyntheticGeometryAllowed()`, the single `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY`
  gate shared by every converter, loader and `ModelBase`).
  FSISelfCheck (`fsi_selfcheck.{h,cpp}`, host-only, model-neutral; capture
  `tools/fsi_capture.py` + `tools/fsi_prompts.json`; per-family goldens
  `tests/fixtures/fsi/<type>.json`; separate from SelfCheck — compares the
  converted model's canonicalized answer at `kOff` to a golden captured from the
  source safetensors; `lsxbenchmark --fsi-selfcheck`; per-family
  `test_<family>_fsi_selfcheck.cpp`, label `inference`, SKIP 4; findings
  `docs/FSI_SELFCHECK_FINDINGS.md`). Capture/verify workflow, golden schema and
  per-model quirks — skill **`fsi-selfcheck`**
  (`.opencode/skills/fsi-selfcheck/SKILL.md`).
  CODESelfCheck (`code_selfcheck.{h,cpp}`, host-only, model-neutral — the
  compiler-syntactic-strictness analogue of FSI, comparing the converted model's
  *exactly* normalized answer at `kOff` to a golden captured from the source
  safetensors; capture `tools/code_capture.py` + `tools/code_prompts.json`;
  per-family goldens `tests/fixtures/code/<type>.json`; `lsxbenchmark
  --code-selfcheck`; per-family `test_<family>_code_selfcheck.cpp`, label
  `inference`, SKIP 4; suite-guard `code` stage; findings
  `docs/CODE_SELFCHECK_FINDINGS.md`; design
  `docs/superpowers/specs/2026-09-27-code-selfcheck-design.md`). Capture/verify
  workflow, golden schema (incl. the prompt-echo `body` cut), extraction
  (json/assignment/regex) and per-model quirks — skill **`code-selfcheck`**
  (`.opencode/skills/code-selfcheck/SKILL.md`).
  DETSelfCheck (`det_selfcheck.{h,cpp}`, host-only, model-neutral — the
  determinism + near-tie analogue of FSI/CODE: runs each prompt R times
  (`stress_runs`, default 3, floored at 3, `LSX_DET_STRESS_RUNS` override),
  requires byte-identical `gen_tokens` across runs and the extracted canonical
  token at `kOff` to match a golden captured from the source safetensors;
  capture `tools/det_capture.py` + `tools/det_prompts.json`; per-family goldens
  `tests/fixtures/det/<type>.json`; `lsxbenchmark --det-selfcheck`; per-family
  `test_<family>_det_selfcheck.cpp`, label `inference`, SKIP 4; suite-guard
  `det` stage; findings `docs/DET_SELFCHECK_FINDINGS.md`; design
  `docs/superpowers/specs/2026-09-27-det-selfcheck-design.md`). Capture/verify
  workflow, golden schema, extraction (`token`) and per-model quirks — skill
  **`det-selfcheck`** (`.opencode/skills/det-selfcheck/SKILL.md`).
  Models: glm / qwen / deepseek / gemma / kimi / nemotron / ornith (NemotronH hybrid
  Mamba-2 + MoE + full attention: `nemotron_moe_constants.h` (frozen passport),
  `nemotron_moe_convert.cpp`, `nemotron_moe_load.cpp` (dense-only resident graph),
  `nemotron_moe_expert_cache.{h,cpp}` (host mmap store of 2944 experts + device
  slot-LRU; dense always resident, experts streamed — the model fits in 16 GiB,
  knob `LSX_NEMOTRON_EXPERT_CACHE_SLOTS`), `nemotron_moe_model.*`,
  `nemotron_moe_kernels.{h,cu}` (conv/SSM/gated-RMSNorm/router/relu2),
  `nemotron_moe_inference_attn.cu` (RoPE+GQA), `nemotron_moe_inference_pipeline.cu`,
  `nemotron_moe_device_math.h`; format tag `logestix-nemotron-nvfp4`).
  GigaChat (GigaChat3.5-432B-A28B-Reasoning hybrid MLA + GatedDeltaNet + MoE:
  `gigachat_moe_constants.h` (frozen passport), `gigachat_moe_convert.{h,cpp}`
  (FP8 block-128 -> neutral NVFP4), `gigachat_moe_dequant.cu` (partial-tile
  FP8->BF16),   `gigachat_moe_load.{h,cpp}`, `gigachat_moe_model.{h,cpp}`,
  `gigachat_moe_device_math.h`, `gigachat_moe_norm_kernels.cu` /
  `gigachat_moe_kernels.{h,cu}` (ZeroCenteredGatedNorm, GDN output norm,
  sigmoid router, swiglu_limit, sigmoid gate),
  `gigachat_moe_expert_cache.{h,cpp}` (host mmap store of 9472 experts + device
  slot-LRU, knob `LSX_GIGACHAT_EXPERT_CACHE_SLOTS`);
  format tag `logestix-gigachat-nvfp4`; MLA/GatedDeltaNet attention + decode
  pipeline land in Plan 2b).
  Ornith-1.5-35B-A3B (independent family `ornith`: an MLX-exported
  Qwen3.5-MoE reasoning fine-tune, architecture-identical to Qwen3.5/3.6-35B;
  `ornith_moe_constants.h` (frozen passport), the standalone MLX converter
  `ornith_moe_convert.{h,cpp}` (prefix `language_model.model.*`,
  `mlp.switch_mlp.{gate,up,down}_proj` stacked experts, conv `[C,K,1]`),
  `ornith_moe_load.{h,cpp}` + `ornith_moe_inference.{h,cu}` (duplicated
  qwen_moe runtime layer; reuses the shared generic `qwen_inference.*` kernels
  and routes NVFP4 through the neutral `lsx_nvfp4`), `ornith_moe_model.{h,cpp}`;
  format tag `logestix-ornithmoe`; knob
  `LSX_ORNITH_MOE_EXPERT_CACHE_SLOTS`).
  Granite-4.2-30B (independent family `granite`: dense decoder-only,
  64L / 4096H / 32Q / 8KV / 128HD / 32768I, SwiGLU, full-dim RoPE,
  `attention_multiplier=1/128`; `granite_dense_constants.h` (frozen passport),
  `granite_dense_convert.{h,cpp}` (flat Granite config -> `logestix-granite`),
  `granite_dense_load.{h,cpp}` (NVFP4 pinned weights, fused `gate_up`),
  `granite_dense_inference.{h,cu}` (causal GQA + full RoPE + SwiGLU dense loop
  over the neutral `lsx_nvfp4` layer and the shared `lsx_kernels_*.cuh`
  primitives), `granite_dense_model.{h,cpp}`; converted in place with
  `lsxforge --granite`; full 2048-token pipeline is correct-first
  (KV-cache decode, non-incremental prefill)).
  phi-4 (independent family `phi`: dense decoder-only, 40L / 5120H / 40Q / 10KV
  / 128HD / 17920I, SwiGLU, full-dim RoPE, attention scale `1/sqrt(128)`, source
  projections pre-fused `qkv_proj`/`gate_up_proj`; `phi_dense_constants.h`
  (frozen passport), `phi_dense_convert.{h,cpp}` (flat phi3 config ->
  `logestix-phi`), `phi_dense_load.{h,cpp}` (NVFP4 pinned weights, fused
  `qkv`/`gate_up` row-slices), `phi_dense_inference.{h,cu}` (causal GQA + full
  RoPE + SwiGLU dense loop over the neutral `lsx_nvfp4` layer and the shared
  `lsx_kernels_*.cuh` primitives), `phi_dense_model.{h,cpp}` (native ChatML
  `<|im_start|>role<|im_sep|>content<|im_end|>` framing; generic tool parser);
  converted in place with `lsxforge --phi`).
  MiMo-V2.6-Flash-RL (independent family `mimo`: hybrid SWA + full-attention
  sparse MoE, 48L / 4096H / 64Q / 4KV(GA)·8KV(SWA) / 192QK·128V head, 256 routed
  experts / top-8, partial RoPE on 64 dims, learned sink bias on the 39 SWA
  layers, sigmoid `noaux_tc` router; `mimo_moe_constants.h` (frozen passport),
  `mimo_moe_convert.{h,cpp}` + `mimo_moe_dequant.cu` (FP8-block-128 dense ->
  NVFP4, MXFP4 experts via neutral `PackExpertNVFP4` with an E8M0-spread
  fallback, multimodal stripped, MTP stored+counted), `mimo_moe_load.{h,cpp}`
  (format/geometry gates), `mimo_moe_kernels.{h,cu}` (RMSNorm / router / SwiGLU),
  `mimo_moe_inference_attn.{h,cu}` (fused-QKV split, partial RoPE, GQA,
  causal/sliding-window + sink bias, KV cache), `mimo_moe_expert_cache.{h,cpp}`
  (host mmap store of 12032 experts + device slot-LRU, knob
  `LSX_MIMO_EXPERT_CACHE_SLOTS`), `mimo_moe_model.{h,cpp}`, `mimo_tool_format.*`
  (ChatML + `<tool_call>` codec); format tag `logestix-mimo-nvfp4`; converted
  with `lsxforge --mimo`; correctness-first token-by-token prefill/decode over
  the neutral `lsx_nvfp4` layer).
  AliceAI-Foundation-80B-A3B-Base (independent family `alice`: hybrid KDA
  (Kimi Delta Attention) + gated-attention sparse MoE with a **block
  attention-residual (AttnRes)** depth mixer, 48L / 12×(3×KDA→MoE … 1×gated
  attn→MoE) / 2048H / 16Q·2KV·256HD (partial RoPE 64, output gate) / KDA
  32K·32V·128HD conv-4 / 512 experts top-10 I=512 + shared I=512 / block=4;
  `alice_moe_constants.h` (frozen passport), `alice_moe_convert.{h,cpp}` (BF16 →
  NVFP4 **only the routed + shared MoE**; KDA/attention projections kept FP16
  because the exponential KDA gate amplifies their quantization), the
  model-neutral **SentencePiece-BPE** tokenizer bridge
  (`BuildTokenizerArrowFilesSentencePiece` + `tokenizer_kind="sp-bpe"`),
  `alice_moe_load.{h,cpp}` (pinned graph + `AliceExpertHost` lazy host store),
  `alice_moe_kernels.{h,cu}` (ZC-RMSNorm / AttnRes depth-softmax / sigmoid
  top-k router / SwiGLU / KDA conv+delta recurrence), `alice_moe_inference.{h,cu}`
  (gated full attention: per-head q/gate split + ZC q/k norm + partial RoPE +
  GQA + output gate), `alice_moe_inference_pipeline.{h,cu}` (AttnRes-blocked
  layer graph + generation), `alice_moe_model.{h,cpp}`; format tag
  `logestix-alice-nvfp4`; converted with `lsxforge --alice`; real-weight
  selfcheck passes).
  Xing4.0-29B-A4B (independent family `xing`: TeleChat, mHC (hyper-connection)
  depth mixer with **4 parallel hidden streams** + full MLA + stored MTP layer,
  40L resident + MTP layer 40 / 3584H / 32 MLA heads / `q_lora=768` /
  `kv_lora=512` / `qk_nope=128` + `qk_rope=64` / `v_head=128` / 64 routed
  experts top-4 (2 dense first layers) + 1 shared I=9216 / vocab 131072 /
  YaRN factor 64 with `mscale^2` folded into the softmax scale ONLY (the tables
  stay unscaled — upstream's rotary returns `attention_factor = 1.0`);
  `xing_moe_constants.h` (frozen passport), `xing_moe_convert.{h,cpp}` (BF16 →
  NVFP4 for every linear incl. embed/lm_head; **mHC tensors, all norms and the
  BF16 router stay high precision** — NVFP4 rounding in the router does not blur
  an output, it swaps the expert set), `xing_moe_load.{h,cpp}` (all 2432 routed
  experts **resident**, three NVFP4 views per expert in the repo's w1/w2/w3 order
  `[gate | down | up]`; no expert cache/LRU — the model fits in 16 GiB),
  `xing_moe_kernels.{h,cu}` (unweighted RMSNorm / mHC + 20-iteration Sinkhorn /
  four-stream mix / sigmoid noaux_tc router), `xing_moe_inference.{h,cu}`
  (MLA with interleaved YaRN RoPE, **per-layer** KV cache slices, dense+MoE
  decode, `LaunchXingRopeQ/K` export the per-head offset internally),
  `xing_moe_model.{h,cpp}`, `xing_moe_tool_format.{h,cpp}` (native tool calling;
  the markers at ids 11/12 are the **plain** `<tool_call>` spellings — the
  zero-width-space variants in `tokenizer_config.json` are not vocabulary pieces);
  format tag `logestix-xing-nvfp4`; converted with `lsxforge --xing`; real-weight
  SelfCheck + FSI + CODE + DET all pass. Decode is still correctness-first:
  108.0 t/s (perf) / 20.0 t/s (r3) at 2048 tokens against a 694 t/s memory
  roofline (`tests/perf/perf_baselines.json`); the mHC and router kernels are
  were single-block by design (all three since parallelised: mHC, router,
  attention, split-K attention, warp-Sinkhorn mHC, split collapse, split router)
  and the remaining costs are the per-MoE-layer routed-id
  D2H sync, the per-token host logits round trip and token-by-token prefill
  (`docs/OPEN_TASKS.md` D2).
- `libs/lsxnetwork/` — shared STATIC library; namespace `lsxnetwork`; zero-alloc
  async network layer for TaskMosaic CE (FrameQueue SPSC, ZMQ context, I/O worker,
  OpenAI endpoint, affinity).
- `apps/lsxfabric/` — B2B inference engine, always maximum speed
  (`--localmodel`, `--validate`, `--testprompt`); `--serve` runs the v2 streaming
  ZeroMQ bus (`--listen` ROUTER + `--listen-cancel` PULL, DEALER/ROUTER token
  frames, per-request cancel); speed measurement and bit-exactness
  moved to lsxbenchmark. `apps/lsxforge/` — data preprocessing utility
  (`--glm`/GLM-5.2 pipeline, `--tokenizer`, repack flags). `apps/taskmosaic/` —
  OpenAI- and Anthropic-compatible HTTP/SSE bridge to a running
  `lsxfabric --serve` over the v2 streaming fabric protocol (`--config`):
  `/v1/chat/completions` (+ `tools`/`tool_calls`, `stop`), `/v1/completions`,
  `/v1/models`, `/health`, `/ready`, and the Anthropic Messages API
  (`/v1/messages`, `/v1/messages/count_tokens`, `HEAD /api/hello`,
  `stop_sequences`). Tool calling has verified native render/parse for
  Qwen3.5/3.6 (ChatML XML), GLM-5.3-Flash (`<|system|>` + `<arg_key>`),
  DeepSeek-V4/V4.1 (DSML; `deepseek_moe_tool_format.{h,cpp}`), Gemma-4
  (`<|tool_call>call:…`; `gemma_tool_format.{h,cpp}`), GigaChat-3.5
  (chatml-v5 GCML; `gigachat_tool_format.{h,cpp}`), Nemotron-3.5
  (Qwen-style XML) and Xing-4.0 (`<tool_call>` + `<param_key>`; the markers
  are the control ids 11/12, so the parse splices them from the ids -
  `xing_moe_tool_format.{h,cpp}`, real-weight verified), with a generic Hermes
  fallback elsewhere. The fallback
  families emit a bare `{"name":…,"arguments":…}` object (Granite-4.2 is
  verified on real weights — `test_granite_dense_selfcheck`), so the generic
  parser tries `tools::ParseBareToolCallJson` after `tools::ParseToolCallsText`;
  do NOT add a "native" renderer for a fallback family without re-verifying that
  round-trip. Entry points: the per-family
  `BuildConversationInputIds`/`ParseToolCalls` virtuals. `apps/lsxcli/` — CLI client
  (`--model`, `--info`, `--infofull`).
- `apps/lsxbenchmark/` — generation-speed measurement + bit-exactness: `--bench`
  (real generation via `IModel::Infer`), `--selfcheck` (Paris), `--bit-exact`
  (`FastMathMode::kOff`); the only app with a numeric mode.
- `tests/` — Catch2 v3.16.0 suite; labels `fast`/`convert`/`inference`/`perf`
  (see § Test) + model/naming/absl/toolchain guard tests.
- `third_party/cccl/`, `third_party/flatbuffers/`, `third_party/libzmq/`,
  `third_party/simdjson/` — submodules.

The full per-file inventory (by family and subsystem: suite membership and
assertions, selfcheck env vars, model perf details) — skill
**`logestix-repo-map`**
(`.opencode/skills/logestix-repo-map/SKILL.md`). Its machine-checked
equivalents: `test_model_suite_guard.cpp::Suites()` (model → stages →
ctest targets), `test_model_independence_guard.cpp::kPassports` (model →
passport header → namespace) and `::kLifecycle` (model → the TU(s) that must
wire the shared lifecycle contract) — when they disagree, they are
authoritative.

### Skill map — the model lifecycle, phase by phase

| phase | skill | what it decides |
|---|---|---|
| 0. intake / triage | **`ingesting-a-model`** § Stage 0 | is this a new family or a sibling? geometry, tokenizer verdict, what to strip, does it fit this machine |
| 1. ingestion | **`ingesting-a-model`** stages 1–8 | converter → NVFP4 → registration → tokenizer bridge → 4b tool calling → kernels → sandbox → passport + tests |
| 2. build | **`building-the-project`**, `cuda-cmake-build-gotchas` | presets, toolchain, where the binaries land, why the build breaks |
| 3. correctness gates | **`test-lifecycle`**, `fsi-selfcheck`, `code-selfcheck`, `det-selfcheck` | labels, targets, and the three parity/determinism goldens |
| 4. throughput | **`perf-tuning-a-model`** | attribute the wall time, budget tests, baselines |
| 5. launch + serve | **`running-a-model`** §2a/§2b | the ordered fast-launch runbook and the acceptance checklist |
| 6. locating code / rules | **`logestix-repo-map`**, `nvfp4-blackwell-conventions`, `absl-in-cuda-cpp`, `coding-conventions` | which file owns what; kernel and host-C++ rules |
| host resources / GPU | **`running-a-model`** §8 | one GPU workload at a time, inference exemption |

Running and operating inference — skill **`running-a-model`**
(`.opencode/skills/running-a-model/SKILL.md`): apps and their flags
(`lsxbenchmark --bench/--selfcheck`, `lsxfabric --localmodel --testprompt/--validate`,
`lsxforge`, `lsxcli`), vault paths (`path_registry`; override the default root with
`LSX_MODEL_ROOT`, and the CLI takes the model ROOT, not `logestix/`), the engine
lifecycle (`InferenceEngine::Session`), prompt
framing/sampling, the resource layer (governor/VRAM/LRU planning) and the catalog of
runtime env knobs `LSX_*`/`KIMI_*`.

## Model Ingestion Protocol

Adding/adapting a new model — skill **`ingesting-a-model`**
(`.opencode/skills/ingesting-a-model/SKILL.md`): the 8-step MODEL INGEST INVARIANT
(lsxforge converter → NVFP4 quantization → registration in `LsxModelFactory` +
frozen geometry → tokenizer bridge → CUDA kernels → shared-mem sandbox → passport →
SelfCheck/tests), the model file naming convention
(`<family>_<type>[_<variant>]_<role>[_<subsystem>]`) and the guard-test registries
(`test_code_naming_guard`, `test_model_independence_guard`,
`test_model_suite_guard`), which must be extended in the same commit as the
model files. The single source of truth is SKILL.md; only an anchor is kept here
because the protocol is cited in code as "Model Ingestion Protocol step N".

## CMake Key Points

- `lsxcommon` — STATIC (`libs/lsxcommon/CMakeLists.txt`); PUBLIC links to spdlog,
  flatbuffers, libzmq-static, cudart, lsxnetwork; PRIVATE — arrow_shared,
  arrow_cuda_shared, simdjson. Arrow include paths are added PUBLIC directly to
  `lsxcommon` (`_deps/arrow-src/cpp/src`, `_deps/arrow-build/src`).
- CUDA `.cu` files are split into separate OBJECT libraries (`lsxcommon_cu`,
  `lsxcommon_glm_cu`, `lsxcommon_qwen_moe_cu`, ...), linked into `lsxcommon`
  PUBLIC — this isolates NVCC from MSVC flags. `cudart` is found via
  `find_package(CUDAToolkit)` (Linux and Windows), not hardcoded paths; cuBLAS is not
  linked and not used at all — the only GEMM path is device-independent `lsx_gemv`
  (see `numeric-conformance-audit.md`).
- `lsxnetwork` — STATIC; simdjson, libzmq-static and Threads PUBLIC
  (a STATIC library re-exports link dependencies), spdlog PRIVATE.
- Every app in `apps/*/` links `lsxcommon` PRIVATE; Catch2 v3.16.0
  via FetchContent when `BUILD_TESTING=ON`.
- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` = `${CMAKE_BINARY_DIR}/bin` **Win32 only**
  (`if(WIN32)` before `add_subdirectory`) — on Linux the binaries stay in
  `build/<preset>/apps/<app>/`; POST_BUILD copies `arrow.dll`/`arrow_cuda.dll`
  (Win32).
- Flags (MSVC/NVCC), `lsx_tune`/LTO gating, dependency patches and CUDA architectures
  — skill `cuda-cmake-build-gotchas`.

## Conventions

Language C++20/C23, logging only via spdlog (no `std::cout`/`printf`),
`#include` style, `target_include_directories` with `$<BUILD_INTERFACE:...>`, placement
of new `.cpp` files (app `add_executable` vs `libs/lsxcommon/`) and preservation of
Doxygen/JSDoc — skill **`coding-conventions`**
(`.opencode/skills/coding-conventions/SKILL.md`).

Kernel/Blackwell rules — skill **`nvfp4-blackwell-conventions`**
(`.opencode/skills/nvfp4-blackwell-conventions/SKILL.md`): one stream and minimal
`cudaDeviceSynchronize`, shared-mem sandbox (thread 0 + barriers), NVFP4 only via the
neutral `lsx_inference_nvfp4` + `std::mdspan`, CCCL priority (cub, thrust
fallback on MSVC), Tile C++, CompileIQ, alignment/48 warps, zero-copy Arrow+CUDA and
simdjson+libzmq, "if unsure about a signature — check the `.h`/`.cuh`".

## absl:: Priority (Container/String policy)

Host-only C++ (NOT compiled by NVCC) prefers `absl::`: `unordered_map/set` →
`flat_hash_map/set`, `map/set` → `btree_map/set`, simple concatenations →
`absl::StrCat`/`StrAppend`. The priority table, exceptions (what stays `std::`),
the NVCC container incompatibility with the reachability rule and recipe check,
the printf contract of `absl::StrFormat` and the CMake anchors (ABSL_PROPAGATE_CXX_STD,
crc32 and format-checker patches) — skill **`absl-in-cuda-cpp`**
(`.opencode/skills/absl-in-cuda-cpp/SKILL.md`). The inviolable parts stay here:

- Headers reachable from `.cu`/`.cuh` must NOT include `<absl/container/*>`
  (NVCC fails in raw_hash_map: `C2794`/`C2938`); in `.cu`/`.cuh` containers and
  streams stay `std::`. The headers that do include a container header are
  `lsx_orchestrator.h`, `model_integrity.h` and taskmosaic's `lsx_framer.h` (all
  host-only, not reachable from `.cu`); the transitive include closure is
  machine-checked by `test_absl_container_guard.cpp` (host-only static scan,
  label `fast`).
- `absl::StrFormat`/`StrAppendFormat`/`Format` — printf style (`%s/%d/%u/%f`);
  `{}` is printed literally (guard `test_absl_format_guard`).
- These stay `std::`: `string_view`/`optional`/`variant`/`function`/`unique_ptr`,
  stream builders with `setw`/`setfill`, custom hasher functors.

## NVIDIA Docs Reference (mandatory)

- NVIDIA documentation is NOT vendored: dumps from the `nvidia-cuda-docs` MCP
  have been removed from the repository to avoid distributing copyrighted material.
  The pointer and procedure — `docs/NVIDIA_DOCS.md`.
- Before editing NVFP4/CUDA kernels, load the current documentation from the
  `nvidia-cuda-docs` MCP (PTX ISA, CUDA C++ Programming Guide, `cuda_fp4`/`cuda_fp8`,
  CUTLASS, TMA, tcgen05) instead of relying on outdated excerpts. If the MCP is
  unavailable — stop and get access to the current docs.
- Discipline (verify semantics BEFORE editing, record derived
  invariants in code/regression tests) — skill `nvfp4-blackwell-conventions`.

## GPU Perf Auto-Tuning

All speed work — decode-token profiling (nsys), TDD budget tests,
`gpu_perf` knobs, the perf baseline — is described in skill **`perf-tuning-a-model`**
(`.opencode/skills/perf-tuning-a-model/SKILL.md`). The inviolable parts stay here:

- **No CLI speed/perf flags**: the engine always runs a per-GPU auto policy;
  the only numeric run flag is `--bit-exact` (lsxbenchmark), which maps
  to `FastMathMode::kOff`. `LSX_PERF_AUTO=0` restores the legacy bit-exact defaults,
  `LSX_TEST_FAST_MATH` forces TC. Tests that pin the numeric class set `kOff`.
- `gpu_perf` (`include/lsxcommon/gpu_perf.h`) resolves the knobs at `InitializeEngine`
  (TC fast-math, GEMV MiB limit, split-K threshold, % VRAM for the expert cache,
  dual-stream staging); the resolution priority and knob consumers are in the skill.
- `lsxbenchmark` — strict GPU exclusivity (see Resources).

## Git

English commit messages and the prefixes `build:`/`feat:`/`fix:`/`refactor:`/
`test:`/`docs:` (plus the "commit only on explicit request" rule) — skill
**`coding-conventions`** (`.opencode/skills/coding-conventions/SKILL.md`).

## Autonomous Mode

- The AI agent is the sole creator and editor of code files (`.cu`, `.cpp`, `.h`, `CMakeLists.txt`).
- The operator manages the process exclusively through high-level prompts and changes to instruction files.
- If manual code changes are detected outside AI sessions, perform a full diff audit and reconcile with the operator before proceeding.
- **No parallel/subagents**: using Task subagents (explore/general/parallel agents) is forbidden — they make many mistakes. All codebase research, audits and edits are done by the main session directly (Read/Grep/Edit/etc.).
