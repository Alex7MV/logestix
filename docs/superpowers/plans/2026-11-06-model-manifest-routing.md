# AGENTS.md Structure → routing table

Migration evidence for the model-manifest work (2026-11-06). Every paragraph of
the old `## Structure` section (104 paragraphs, 826 lines) is routed to exactly
one destination. The destinations:

- **A** — an invariant/rule: stays in `AGENTS.md`, in a numbered section.
- **M** — a per-model fact: generated `docs/MODELS.md`.
- **D** — a measured result with a canonical home: stays there, linked from
  `docs/MODELS.md` (`tests/perf/perf_baselines.json`, `docs/SELFCHECK_MATRIX.md`).
- **R** — rationale/history: `docs/model-reports/` or `docs/open-tasks/`, with a
  pointer in `docs/MODELS.md`.

| # | subject of the paragraph | destination | where |
|---|---|---|---|
| 1 | monorepo layout, C++20, CMake presets, Blackwell | A | Project |
| 2 | 48 warps/SM, forbidden archs, no pre-Blackwell | A | Project |
| 3 | CUDA 13.5 vs driver 13.4 skew, `test_cuda_toolkit_guard` | A | Project |
| 4 | `docs/OPEN_TASKS.md` + archive + `fast` green count | A | Project |
| 5 | `docs/SELFCHECK_MATRIX.md`, `docs/OPEN_ENDEDNESS.md` | A | Project |
| 6 | dependency table + the pins guard | A | Dependencies |
| 7 | CCCL band guard, dependency bump procedure | A | Dependencies |
| 8 | build presets → skill | A | Build |
| 9 | host resource policy → skill | A | Resources |
| 10 | test labels/presets/router → skill | A | Test |
| 11 | `lsxcommon` core: `core.cpp`, `schema.cpp`, `converter.cpp`, safetensors | A | Structure → lsxcommon |
| 12 | `tokenizer.cpp`/`pretokenizer`/`unicode`, `model_factory`/`model_registry` | A | same |
| 13 | `model_runtime.cpp`, `gpu_perf.cpp`, `sampling.{h,cu}`, `token_stream.h` | A | same |
| 14 | `chat_conversation.h`/`tool_format`, `incremental_detokenizer` | A | same |
| 15 | neutral layer `lsx_inference_nvfp4` + `lsx_gemm`, only GEMM path | A | same (invariant) |
| 16 | `expert_store.h` + pinned staging ring note, measured neutral | A + R | lsxcommon; `docs/OPEN_TASKS.md` §R17 P6 |
| 17 | `lsx_attn_inference`, `lsx_dense_inference`, `lsx_moe_inference` | A | same |
| 18 | `lsx_device_math.h`, `lsx_reduce.cuh`, `synthetic_geometry.h` | A | same |
| 19 | FSISelfCheck design/capture/goldens → skill | A | same |
| 20 | CODESelfCheck → skill | A | same |
| 21 | DETSelfCheck → skill | A | same |
| 22 | the model list (glm/qwen/deepseek/...) with geometry, gates, t/s | M + D | generated `docs/MODELS.md`; numbers stay in `perf_baselines.json` / matrix |
| 23 | NemotronH contour + NoPE rationale (0.09 → 1.20 relL2) | M + R | MODELS.md; model report |
| 24 | Qwen3.8-Flash-Next geometry + gates + t/s + arena-required | M + R | MODELS.md; `docs/open-tasks/2026-10-14-qwen38-arena-required-negative.md` |
| 25 | GigaChat geometry + template-derived goldens + t/s | M + R | MODELS.md; model report |
| 26 | Ornith geometry + both source layouts + format tag | M | MODELS.md |
| 27 | Granite-4.2 geometry + kOff rationale | M + R | MODELS.md; model report |
| 28 | Granite-Switch aLoRA + routing + kOff + DET debt | M + R | MODELS.md; `OPEN_TASKS` §R10 |
| 29 | phi-4 residual plane + FP16 edges + why all four gates green | M + R | MODELS.md; model report |
| 30 | MiMo geometry + FP8/MXFP4 + sink bias | M | MODELS.md |
| 31 | Alice geometry + AttnRes + FP16 projections rationale | M + R | MODELS.md; model report |
| 32 | Xing mHC + MLA + YaRN + decode history (146.65 t/s) | M + R + D | MODELS.md; `open-tasks/2026-10-23-residual-closures.md`; baseline |
| 33 | Gemma-4 MoE + residual plane + why 156.7 → 128.25 | M + R + D | MODELS.md; `open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md` |
| 34 | Gemma-4 dense 31B + the NVFP4-borderline FSI-01 class | M + D | MODELS.md; matrix |
| 35 | Mistral-Small-4 MLA + consolidated layout + goldens debt | M + R | MODELS.md; model report; `OPEN_TASKS` §T |
| 36 | Mistral-Large-4 five deltas (hybrid RoPE/NoPE, LeRoPE, ssmax, DSA, KV sharing) | M + R | MODELS.md; model report |
| 37 | DeepSeek-V4-Pro geometry + HC FP32 + Free() note | M + R | MODELS.md; `OPEN_TASKS` §R3 |
| 38 | MiniMax-M3 MSA + SwiGLU-OAI + rendering | M + R | MODELS.md; `OPEN_TASKS` §R4 |
| 39 | MiniMax-M3.1-Flash pre-quantized source + passport discriminator | M + R | MODELS.md; `open-tasks/2026-10-31-minimax-m31-flash-ingest.md` |
| 40 | Flint geometry + massive activations + FFN scaling | M + R | MODELS.md; `OPEN_TASKS` §R6 |
| 41 | Hy-Memory geometry + expert_bias + FP32 combine | M + R | MODELS.md; `OPEN_TASKS` §R5 |
| 42 | Hy-SimulMT direction-bound translator + stop ids | M + R | MODELS.md; `OPEN_TASKS` §R11 |
| 43 | Kimi-K2.7 MLB + INT4 → NVFP4 + tiktoken | M + R | MODELS.md; `open-tasks/2026-10-26-kimi-audit-verification.md` |
| 44 | Kimi-K3 KimiLinear + Stable LatentMoE + AttnRes | M + R | MODELS.md; `OPEN_TASKS` §R15 |
| 45 | Kimi-K3.1-Pro Gated DSA + top-2048 exactness + perf rows | M + R + D | MODELS.md; `OPEN_TASKS` §R17; baseline |
| 46 | Ling KDA + per-expert global scales + SwiGLU limits | M + R | MODELS.md; `OPEN_TASKS` §R7 |
| 47 | Beam geometry + Harmony channels | M + R | MODELS.md; `OPEN_TASKS` §R8 |
| 48 | Step-5 five deltas (ZC-RMSNorm, per-layer RoPE span, gate, FP8 stacks, FP32 residual) | M + R | MODELS.md; `OPEN_TASKS` §R9 |
| 49 | `lsxnetwork`, apps (`lsxfabric`, `lsxforge`, `taskmosaic`, `lsxcli`) | A | Structure → apps |
| 50 | taskmosaic tool-calling matrix + the fallback warning | A | same (invariant) |
| 51 | `tests/` labels + the Arrow guards | A | Structure → tests |
| 52 | repo-map skill + the machine-checked registries | A | Structure → guards |
| 53 | skill map table | A | unchanged |
| 54 | Model Ingestion Protocol anchor | A | unchanged |
| 55 | CMake key points (targets, `lsx_dep_includes`, arch flags, helpers) | A | CMake key points |
| 56 | conventions (logging, includes, naming) → skill | A | Conventions |
| 57 | absl priority + NVCC reachability rule | A | absl priority |
| 58 | NVIDIA docs discipline | A | NVIDIA Docs |
| 59 | perf auto-tuning + `--bit-exact` + `test_perf_baseline` | A | GPU Perf |
| 60 | git conventions | A | Git |
| 61 | autonomous mode (no subagents) | A | Autonomous Mode |

Rules applied:

- A per-model paragraph carrying a measured number is **M + D**, never M alone:
  the catalogue names the gate/perf row, it does not restate the number.
- A per-model paragraph carrying a *reason* (why kOff, why FP16 projections, why
  the residual plane) is **M + R**: the catalogue gets one line, the reasoning
  keeps its report.
- Nothing routed to R is deleted without the destination existing; the phase
  commit adds the pointer lines to `docs/MODELS.md` in the same commit.
