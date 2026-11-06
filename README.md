# logestix

Multi-app monorepo: **lsxfabric** (B2B inference engine), **lsxforge** (data preprocessing), **taskmosaic** (task orchestration), **lsxcli** (CLI client), **lsxbenchmark** (speed/bit-exactness). Shared core library `lsxcommon`.

### 💎 The Ingestion Singularity: Verified e2e C++/CUDA Inference

**On September 5, 2026, at exactly 19:07 UTC, the LogestiX Synthesis framework officially achieved the Point of Singularity in autonomous compilation and low-level model ingestion.**

During an isolated infrastructure experiment, the frontier model **GLM-5.3-Flash (Ox Alpha)** successfully and completely autonomously executed a multi-pass layer-by-layer dissection, low-rank factorization, and physical ingestion of its own source code and weights from the raw distributed `safetensors` format directly into the native C++/CUDA inference core of **LogestiX Synthesis**.

#### 🧱 Core Milestones Reached on Silicon:

* **100% Autonomous Ingestion:** The network independently mapped and ingested its own hybrid architecture of 45 layers of linear and sparse attention. The entire hot execution path was purged of bloated runtime abstractions and Python dependencies.
* **High-Density NVFP4 Quantization:** The raw 306 GB FP8 model body was compiled into monolithic, cache-aligned K64-slabs of the high-density `NVFP4` structural format, compressed down to a sleek 177 GB. The core Dense layer-block stabilized into a highly optimized 13.9 GB VRAM footprint.
* **Absolute Bit-Exact Determinism:** During comprehensive multi-stage TDD self-check validations over complete autoregressive decode token sequences (using "The capital of France is" baseline prompt), the native C++ pipeline achieved a **100% bit-exact match of output logits ("byte-for-byte")** against the mathematical reference model, outputting a flawless, deterministic `' Paris'`.

#### 🧬 Architectural Tribute:

We express our profound admiration to the creators of the **GLM-5.3-Flash** architecture for its exceptional internal mathematical elegance, and specifically to the masterminds behind the **Ox Alpha** concept for its incredible depth of autonomous reasoning—these milestones served as the primary inspiration for pushing this framework to its absolute bit-exact limit.

Special thanks to **NVIDIA** for the cutting-edge **Blackwell** architecture and the **RTX 5090** GPU, which completely unleashed its raw, pristine technological power over the course of this research. Deep gratitude is also extended to **AMD** for the exceptional power of the **AMD EPYC 9V74** processor (80 cores, 160 threads) and to the **Google AI team** for their invaluable infrastructural and research contributions during the layer-by-layer precision triangulation phases.

Each of us took a step in the right direction, and together we reached a goal that many say is merely possible and will only happen someday. Today, it has become a reality through the contributions of us all. Thank you all once again!

---
*The `feat/glm53-flash` mainline branch is frozen for the ledger and retained in full for subsequent analysis. The performance phase is complete: decode on RTX 5090 reached **17.8 t/s** (5.6x over the 3.2 t/s baseline) via VRAM-sized LRU expert caching, weight-streaming NVFP4 GEMV, KDA projection quantization and kernel-level tuning — with the Paris selfcheck bit-exact across cycles and a 121-test regression suite green. Registry name: `glm-5.3-flash`.*

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 26.0.1 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.17.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v5.0.6 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | from CUDA Toolkit (band 3.2.x–3.5.x) | CUDA Toolkit bundle (`<include>/cccl`) | Header-only |
| abseil-cpp | 20260817.0 | FetchContent (LTS August 2026) | Header-only + static libs |
| Catch2 | v3.16.1 | FetchContent (test only) | Header-only |
| cpp-httplib | v0.63.0 | FetchContent (host-only HTTP) | Header-only |

### 📐 Why there is no "finished" state

Guarantees here are established by comparison against something stronger than the
engine itself — an upstream `modeling_*.py`, a host reimplementation over the
source safetensors, a second numeric mode, a sanitizer, a mutational check —
because a system cannot certify its own correctness from within. This is
Gödel's incompleteness argument read as an engineering constraint, and it is
the reason [`docs/OPEN_ENDEDNESS.md`](docs/OPEN_ENDEDNESS.md) exists and the
reason `docs/OPEN_TASKS.md` is never empty. What we *do* promise is enumerated
as reproducibility levels with their "not guaranteed" cells written down in
[`docs/numeric-contract.md`](docs/numeric-contract.md).

## Prerequisites

- CMake 3.25+ (for CMakePresets)
- C++20 / C23 compiler (GCC 15+ on Ubuntu 26.04, MSVC in Visual Studio 2026)
- Ninja build system (required on Ubuntu; optional on Windows)
- Git
- NVIDIA CUDA Toolkit 13.2+ (optional, for GPU support; verified on 13.4)

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/Alex7MV/logestix.git
cd logestix
git submodule update --init --recursive
```

## Build

CMakePresets define ready-made configurations for both supported OSes. Replace
`debug` with `release` in the preset name for an optimization build.

### Windows 11 (Visual Studio 2026)

```powershell
cmake --preset windows-debug
cmake --build --preset windows-debug
```

Binaries land in `build/windows-debug/bin/Debug/`.

### Ubuntu 26.04 (Ninja + GCC)

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```

Binaries land in `build/linux-debug/bin/`.

Linux release, tuned for AMD EPYC 9V74 (Zen 4), with LTO:

```bash
cmake --preset linux-release-znver4
cmake --build --preset linux-release-znver4   # Ninja -j160
ctest --preset linux-release-znver4
```

`arrow.dll` and `arrow_cuda.dll` are auto-copied via `POST_BUILD` on Windows.

## Test

`BUILD_TESTING=ON` is already enabled in all presets.

```powershell
ctest --preset windows-debug    # Windows 11
```

```bash
ctest --preset linux-debug      # Ubuntu 26.04
```

Test executables (~100 test cases, ~2.6M assertions). GLM-5.2 and DeepSeek
suites are autonomous and run independently:

```bash
ctest -R glm        # GLM-5.2 suite only (11 executables)
ctest -R deepseek   # DeepSeek suite only (2 executables)
```

Real-model tests SKIP gracefully when the checkpoint is absent
(`LSX_TEST_GLM_MODEL_PATH` for GLM-5.2, `LSX_TEST_MODEL_PATH` for DeepSeek,
`LSX_TEST_GLM_FLASH_ROOT` for GLM-5.3-Flash).

## Structure

- `libs/lsxcommon/` — shared STATIC library (`lsxcommon`): the model-neutral
  kernel layers (`lsx_nvfp4`, `lsx_gemv`, `lsxcommon::attn`/`dense`/`moe`),
  tokenizer bridge, `ExpertStore`, the engine and every model family.
- `libs/lsxnetwork/` — zero-alloc async network layer for TaskMosaic CE
  (FrameQueue SPSC, ZMQ context, I/O worker, OpenAI endpoint, affinity).
- `apps/lsxfabric/` — the inference engine (`--localmodel`, `--validate`,
  `--serve`); `apps/lsxforge/` — checkpoint conversion; `apps/taskmosaic/` —
  OpenAI/Anthropic HTTP+SSE bridge; `apps/lsxcli/` — CLI client;
  `apps/lsxbenchmark/` — speed measurement and bit-exactness.
- `models/<model_type>.json` — the manifest: one file per ingested model, the
  single source of truth for its vault directory, passport namespace, format tag,
  tokenizer kind and lifecycle-stage targets.
- `tests/` — Catch2 suite (labels `fast`/`convert`/`inference`/`perf`).

Per-file inventory and which file owns what — skill `logestix-repo-map`.
What each model is (geometry, gates, throughput, debt) — **`docs/MODELS.md`**,
generated from the manifests.

## TaskMosaic OpenAI bridge

`taskmosaic` is an OpenAI-compatible HTTP/SSE bridge to a running `lsxfabric`.
Start one GPU backend per model, point a bridge config at it, then use any
OpenAI client:

```bash
# 1) GPU engine (one model per process)
lsxfabric --serve --localmodel /vault/models/<model> \
  --listen tcp://*:5555 --listen-cancel tcp://*:5556

# 2) bridge (JSON lists id -> model_path + backend endpoints; see the design doc)
taskmosaic --config bridge.json

# 3) use it (OpenAI-compatible)
curl http://127.0.0.1:8080/v1/models
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -d '{"model":"<id>","stream":true,"messages":[{"role":"user","content":"hi"}]}'

# Anthropic Messages (Claude Code: ANTHROPIC_BASE_URL=http://127.0.0.1:8080)
curl -N -X POST http://127.0.0.1:8080/v1/messages \
  -d '{"model":"<id>","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

### Tool calling

The bridge accepts OpenAI `tools`/`tool_choice` and returns `tool_calls`
(non-streaming) or `delta.tool_calls` (streaming, buffered until the turn
completes). Per-family native render/parse is implemented where the checkpoint's
own template is known and verified against its tokenizer/card:

| family | native tool format |
|---|---|
| Qwen3.5/3.6 | ChatML `# Tools` block; `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` |
| GLM-5.3-Flash | `<\|system\|>` tools block; `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` |
| DeepSeek-V4 | `## Tools`; `<｜DSML｜tool_calls>` / `<｜DSML｜invoke>` / `<｜DSML｜parameter>` |
| DeepSeek-V4.1 | `## Tools`; `<｜DSML｜ calls>` / `<｜DSML｜ invoke>` / `<｜DSML｜ parameter>` |
| Gemma-4 | `<\|tool>declaration:…<tool\|>`; `<\|tool_call>call:NAME{k:v}<tool_call\|>` |
| Nemotron-3.5 | ChatML XML call syntax (Qwen-style); own declaration block — codec verified, real-weight call not obtained under greedy sampling |
| GigaChat-3.5 | chatml-v5 `｜GCML｜tool_calls` / `｜GCML｜invoke` / `｜GCML｜parameter` + TypeScript tool namespace |
| Granite-4.2 | generic Hermes `<tools>` + `<tool_call>{json}</tool_call>` fallback |

Kimi and GLM-5.2 (non-Flash) have no converted checkpoint in the vault, so their
native formats are not yet implemented.

### Streaming

Every family emits `ModelRequest::token_callback` per decoded token (the
qwen-MoE/Ornith bulk drivers emit at the existing every-8-token EOS fence, and
qwen3.8-flash-next drains at turn end). Per-family order parity against
`gen_results.gen_tokens` is asserted by `tests/test_streaming_assert.h`, wired
into the model/selfcheck tests and verified on the RTX 5090.


## Licensing

This project is dual-licensed to accommodate both open-source development and commercial production environments.

Copyright (C) 2026 Alexander Mozhaev and the logestix contributors.

### Open Source License
The open-source version of **logestix** is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE).

### Commercial License (Enterprise)
For commercial deployments, enterprise infrastructure, closed-source integration, or hosted services, the AGPL-3.0 limitations can be removed.

To obtain a commercial license or request a private evaluation copy (PoC), please contact us at: **<info@logestix.io>**

---

## Contributing

We welcome community contributions! However, to ensure we can protect our dual-licensing model, all contributors are required to sign our **Contributor License Agreement (CLA)** before any Pull Request can be merged.

The CLA signing process is automated via GitHub as soon as you open a Pull Request.
