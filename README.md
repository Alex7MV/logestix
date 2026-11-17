# logestix

Multi-app monorepo: **lsxfabric** (B2B inference engine), **lsxforge** (data preprocessing), **taskmosaic** (task orchestration), **lsxcli** (CLI client), **lsxbenchmark** (speed/bit-exactness). Shared core library `lsxcommon`.

### Verified end-to-end C++/CUDA inference

logestix is a deterministic C++20/CUDA runtime for native NVFP4 inference,
verified across **39 registered model profiles** and **555 CTest targets**
(see `docs/MODELS.md` and the Test section below). Model ingestion — converter,
frozen geometry, tokenizer bridge, parity goldens — is largely agent-driven,
but every claim rests on a measured oracle, not on the agent's word.

The reference milestone: **GLM-5.3-Flash**, ingested from its raw distributed
`safetensors` checkpoint (306 GB FP8 → 177 GB NVFP4 K64-slab format; 45 layers
of linear and sparse attention; ~13.9 GB VRAM dense layer-block). Its SelfCheck
is bit-exact — token, text and last-logits determinism across ≥ 3 stress cycles
of the canonical "The capital of France is" prompt, decoding `' Paris'` — with
FSI/CODE/DET parity 3/3 against goldens captured from the source checkpoint
(`docs/SELFCHECK_MATRIX.md`). Decode on RTX 5090 reached **17.8 t/s** (5.6× over
the 3.2 t/s baseline) via VRAM-sized LRU expert caching, weight-streaming NVFP4
GEMV and kernel tuning. What is *not* guaranteed is enumerated as
reproducibility levels in [`docs/numeric-contract.md`](docs/numeric-contract.md).

Thanks to the **GLM-5.3-Flash** authors for the architecture, to **NVIDIA** for
Blackwell and the RTX 5090, and to **AMD** for the EPYC 9V74 host. The
`feat/glm53-flash` branch is frozen for the ledger; registry name:
`glm-5.3-flash`.

| Dependency | Version | Source | Type |
|---|---|---|---|
| Apache Arrow | 26.0.1 | FetchContent (BUNDLED) | Shared (`arrow.dll`) |
| spdlog | 1.17.0 | FetchContent | Header-only |
| FlatBuffers | v25.12.19 | submodule (`third_party/flatbuffers`) | Static |
| libzmq | v4.3.5 | submodule (`third_party/libzmq`) | Static |
| simdjson | v5.2.0 | submodule (`third_party/simdjson`) | Static (`.lib`) |
| NVIDIA CCCL | from CUDA Toolkit (band 3.2.x–3.5.x) | CUDA Toolkit bundle (`<include>/cccl`) | Header-only |
| abseil-cpp | 20260817.0 | FetchContent (LTS August 2026) | Header-only + static libs |
| Catch2 | v3.16.1 | FetchContent (test only) | Header-only |
| cpp-httplib | v0.66.0 | FetchContent (host-only HTTP) | Header-only |

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
- **NVIDIA CUDA Toolkit 13.5 — required, not optional.** The build enables
  CUDA unconditionally (`project(... LANGUAGES C CXX CUDA)` +
  `find_package(CUDAToolkit REQUIRED)`); there is no CPU-only build. The
  enforced floor is 13.2 (`test_cuda_toolkit_guard`); 13.5 is the verified
  toolkit. A 13.x driver older than the toolkit is supported through CUDA
  minor-version compatibility (e.g. toolkit 13.5 on a 13.4 driver).
- **NVIDIA Blackwell GPU — required for all device work.** Compute capability
  12.0/12.1 only (`sm_120`/`sm_121`, e.g. RTX 5090). Pre-Blackwell
  architectures (Hopper, Ada, Ampere) are deliberately unsupported and
  rejected at configure time. Device codegen comes from `LSX_CUDA_ARCH`
  (default `120;121`); `-DLSX_CUDA_ARCH="120f;121f"` selects the
  family-specific targets — measured neutral for this repo's weight-streaming
  NVFP4 decode paths, kept as a one-variable experiment for compute-bound
  cases.

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

`BUILD_TESTING=ON` is already enabled in all presets. Every test target
carries exactly one label, and the label decides when it runs
(enforced by `test_ctest_labels_guard`):

| Label | What it is | When it runs |
|---|---|---|
| `fast` | host-only unit and guard tests; parallel-safe, no model weights needed | every `ctest --preset` — the default filter of all test presets |
| `convert` | model converters on synthetic checkpoints | when a converter changed, or manually |
| `inference` | CUDA kernels and the model contour (load / model / selfcheck / fsi / code / det) | when the contour changed, or manually |
| `perf` | throughput baselines on real weights | separate command only, **Release only** (a Debug run SKIPs) |

The one CUDA-aware `fast` target, `test_cuda_toolkit_guard`, allocates no
device memory — it reads toolkit/driver versions and device properties, and
its device case SKIPs where no GPU exists.

As of 2026-11-16 the tree registers **555 CTest targets: 149 `fast`,
41 `convert`, 318 `inference`, 47 `perf`**. Counts move as models are
ingested, so re-read them instead of trusting this paragraph:
`ctest --preset linux-debug -N` (the fast default) or
`ctest --test-dir <build-dir> -N` (the full tree).

```powershell
ctest --preset windows-debug    # Windows 11 — fast label only
```

```bash
ctest --preset linux-debug               # Ubuntu 26.04 — fast label only (149 targets)
ctest --preset linux-debug-convert       # converter sweep
ctest --preset linux-debug-inference     # CUDA / model-contour sweep
ctest --preset linux-release-znver4-perf # perf sweep; Release-only
ctest -L "^(fast|convert|inference|perf)$"   # full run (from the build directory)
```

Every base test preset also exists with a `-fast` / `-convert` / `-inference`
/ `-perf` suffix selecting that label. GPU-labelled targets hold the `gpu`
resource lock: run **one GPU workload at a time**. The lock is per `ctest`
process, not per machine — two concurrent `ctest` invocations both acquire it
and share the GPU, so never start a second GPU-labelled sweep while one is in
flight.

Family sweeps work by name regex, e.g. `ctest -R glm` (GLM-5.2 and
GLM-5.3-Flash) or `ctest -R deepseek` (all DeepSeek variants). The suites are
autonomous and run independently.

### Model vault (`LSX_MODEL_ROOT`)

The engine and the real-model tests resolve converted checkpoints from a
model vault. The root defaults to `/mnt/fast_storage/huggingface` on Linux
and `D:\Programs\huggingface.co` on Windows (compile-time defaults in
`libs/lsxcommon/include/lsxcommon/path_registry.h`). **Set `LSX_MODEL_ROOT`
to an absolute path to relocate the vault** — the runtime override takes
precedence over the platform default:

```bash
export LSX_MODEL_ROOT=/data/models   # your vault root
```

Converted models live in `<LSX_MODEL_ROOT>/<Model>/logestix/` (produced by
`lsxforge`). Individual model tests also accept a per-model root override
declared in `models/<model_type>.json` (`vault.env`, e.g.
`LSX_TEST_GLM_5_2_ROOT`); the full per-model list is in `docs/MODELS.md`.
When the converted tree or a parity golden is absent, the test SKIPs
gracefully (exit code 4) — a SKIP is **not** a pass.

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

We welcome community contributions! However, to ensure we can protect our dual-licensing model, all contributors are required to sign our **Contributor License Agreement (CLA)** before any Pull Request can be merged — confirm agreement in the PR checklist or contact **<info@logestix.io>**.
