# logestix

Multi-app monorepo: **lsxfabric** (B2B inference engine), **lsxforge** (data preprocessing), **taskmosaic** (task orchestration), **lsxcli** (CLI client), **lsxbenchmark** (speed/bit-exactness). Shared core library `lsxcommon`.

### Verified end-to-end C++/CUDA inference

logestix is a deterministic C++20/CUDA runtime for native NVFP4 inference. The
catalogue holds **47 registered model rows across 22 orgs** at the time of
writing, and `docs/MODELS.md` — generated from the per-model manifests — is the
count to trust, not this line. A row describes a model that has been **triaged
and ingested** (converter, frozen geometry, tokenizer bridge, parity goldens);
whether its artifact has actually been **built** is a separate, recorded fact
per codec profile. Model ingestion is largely agent-driven, but every claim
rests on a measured oracle, not on the agent's word.

What an ingestion has to establish — and what each manifest records either as
done or as declared **debt** — is the same for every profile:

- **SelfCheck** — the canonical `"The capital of France is"` probe answers
  `Paris`, with bit-exact token/text/last-logits determinism across ≥ 3 stress
  cycles.
- **Parity gates** — FSI, CODE and DET, each a separate target with its own
  golden. A gate whose golden was never captured is declared **debt** in the
  manifest; it is not silently absent.
- **Native tool round-trip** — the checkpoint's own chat/tool template, rendered
  and parsed on real weights, not the generic fallback.
- **A seeded throughput row** at the canonical 2048-token protocol.

Measured verdicts, including the red and debt cells, live in
[`docs/SELFCHECK_MATRIX.md`](docs/SELFCHECK_MATRIX.md); throughput in
`tests/perf/perf_baselines.json`. **A SKIP is not a pass**, and a registry row
beats anything written in prose here.

The reference milestone is **GLM-5.3-Flash** (registry `glm-5.3-flash`): ingested
from its raw distributed `safetensors` checkpoint into the K64-slab NVFP4
layout, bit-exact SelfCheck, FSI/CODE/DET 3/3 against goldens captured from the
source checkpoint, and decode accelerated by VRAM-sized LRU expert caching,
weight-streaming NVFP4 GEMV and kernel tuning. Its current numbers — and every
other model's — are in the two registries named above. What is *not* guaranteed
is enumerated as reproducibility levels in
[`docs/numeric-contract.md`](docs/numeric-contract.md).

Thanks to the **GLM-5.3-Flash** authors for the architecture, to **NVIDIA** for
Blackwell and the RTX 5090, and to **AMD** for the EPYC 9V74 host.

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

The pins are enforced by `tests/test_core.cpp` → `TEST_CASE("pinned dependency
versions")`; the bump procedure is skill `cuda-cmake-build-gotchas`.

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
- C++20 / C23 compiler (GCC 15+ on Ubuntu 26.04, MSVC in Visual Studio 2026;
  AMD's AOCC clang is supported as an alternative host compiler)
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

Binaries land in `build/windows-debug/bin/Debug/`, and `arrow.dll` /
`arrow_cuda.dll` are auto-copied there via `POST_BUILD`.

### Ubuntu 26.04 (Ninja + GCC)

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```

Binaries land next to their target: `build/linux-debug/apps/<app>/<binary>`
(e.g. `build/linux-debug/apps/lsxfabric/lsxfabric`) and
`build/linux-debug/tests/<test-target>`. On Linux there is no `bin/` — that
directory is a Windows-only `CMAKE_RUNTIME_OUTPUT_DIRECTORY` convention.

Linux release, tuned for AMD EPYC 9V74 (Zen 4), with LTO:

```bash
cmake --preset linux-release-znver4
cmake --build --preset linux-release-znver4   # Ninja -j160
ctest --preset linux-release-znver4
```

The same configuration with **AOCC clang 17 as the host compiler** is
`linux-release-znver4-aocc`. It reads `AOCC_ROOT` from the environment, so the
preset cannot be configured without it:

```bash
export AOCC_ROOT=/path/to/aocc-compiler-5.2.0
cmake --preset linux-release-znver4-aocc
cmake --build --preset linux-release-znver4-aocc -j64
```

Real-weight gates are 5–20× faster on a Release tree than on `linux-debug`, so
selfcheck/parity/perf runs belong here rather than in a debug build.

## Test

`BUILD_TESTING=ON` is already enabled in all presets. Every test target
carries exactly one label, and the label decides when it runs
(enforced by `test_ctest_labels_guard`):

| Label | What it is | When it runs |
|---|---|---|
| `fast` | host-only unit and guard tests; parallel-safe, no model weights needed | every `ctest --preset` — the default filter of all test presets |
| `convert` | model converters on synthetic checkpoints | when a converter changed, or manually |
| `inference` | CUDA kernels and the model contour (load / model / selfcheck / fsi / code / det / mcp) | when the contour changed, or manually |
| `perf` | throughput baselines on real weights | separate command only, **Release only** (a Debug run SKIPs) |

The one CUDA-aware `fast` target, `test_cuda_toolkit_guard`, allocates no
device memory — it reads toolkit/driver versions and device properties, and
its device case SKIPs where no GPU exists.

Counts move with every ingestion, so read them instead of trusting this page:
`ctest --test-dir <build-dir> -N` prints the target total, and `-L <label>`
narrows to one label. For orientation only, the tree on 2026-11-28 registered
**663 targets: 184 `fast`, 48 `convert`, 376 `inference`, 55 `perf`**.

```powershell
ctest --preset windows-debug    # Windows 11 — fast label only
```

```bash
ctest --preset linux-debug               # Ubuntu 26.04 — fast label only
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
flight (`nvidia-smi --query-compute-apps` first).

Narrow a run by name regex when working on one model or family — the regex is
the model's `model_type` (or its family prefix), not a list this file keeps:

```bash
ctest --test-dir build/linux-debug -R "<model_type>" -j 1   # one model's row
ctest --test-dir build/linux-debug -R "<family>" -j 1       # its siblings too
```

Each model's `model_type` and every target it owns are the manifest's
(`models/<org>/<model>.json`, projected into `docs/MODELS.md`), so there is
nothing to look up by hand and no reason to sweep the label to verify one
model.

## Where models come from: two org-nested stores

Nothing is downloaded, converted or loaded by guessing a path. There are two
roots, kept apart on purpose, **neither with a default**, and both laid out the
way Hugging Face lays a download out — **`<org>/<model>`**.

Both must be exported before anything real-weight can run; the binary will not
invent either one:

```bash
export LSX_SOURCE_ROOT=/data/checkpoints   # <org>/<model> checkpoints, read-only
export LSX_ARTIFACT_ROOT=/data/artifacts   # converted trees, one dir per profile
```

| store | contents | env | default |
|---|---|---|---|
| **source** (read-only) | the checkpoint as downloaded: `config.json`, `*.safetensors`, `tokenizer.json`, `chat_template.jinja` | `LSX_SOURCE_ROOT`, then the pre-split `LSX_MODEL_ROOT` | **none** — unset is a configuration error, not a fallback |
| **artifact** (writable) | the converted LogestiX tree, one directory per codec profile | `LSX_ARTIFACT_ROOT` | **none** — unset is a configuration error, not a fallback |
| host cache | per-GPU perf calibration only, never model data | `LSX_CACHE_DIR` | `$XDG_CACHE_HOME/logestix` / `~/.cache/logestix` |

```
<LSX_SOURCE_ROOT>/<org>/<model>                       the download
<LSX_ARTIFACT_ROOT>/<org>/<model>/<profile>           the converted artifact
```

**The profile directory IS the artifact.** There is no `logestix/` segment any
more (that was the pre-split layout, where conversion wrote inside the
checkpoint): a profile directory holds `artifact.fb`, `model.index.json`,
`pinned/`, and — self-sufficiency invariant, so a converted tree loads with the
source store unmounted — the tokenizer and chat template it was built with.
Profile names are `nvfp4` (default), `bf16`, `int8-w8a16`; one model may hold
several, each with its own codecs and its own measured gates, and the load gate
refuses a profile whose codecs the active device cannot execute (naming the
profiles that would work instead of choosing one silently).

### Where an artifact comes from

An artifact exists only because someone converted a checkpoint — the repo ships
no weights:

```bash
export LSX_SOURCE_ROOT=/data/checkpoints        # where the checkpoint lives
export LSX_ARTIFACT_ROOT=/data/lsx_artifact     # where the artifact is written
lsxforge --input <org>/<model> --codec-profile nvfp4 --<family>
# -> $LSX_ARTIFACT_ROOT/<org>/<model>/nvfp4/
```

`--input` is a model **identity**: the store directory (`Qwen3.5-9B`), the
registry `model_type` (`qwen3.5-9b`), an org-qualified name, or an absolute
checkpoint path. The org and model directory in the output path come from
`models/<org>/<model>.json`, not from the path you typed, so an artifact always
lands beside its source. With `--codec-profile` the output is the profile
directory and `--output` is ignored; without it the converter writes into
`--output` directly.

### Which artifacts are built, and how a run finds them

`docs/MODELS.md` is the authority: each row names its store path
(`<org>/<model>`), its env override (`LSX_TEST_<MODEL>_ROOT`) and, per profile,
whether it is **built** — with the codec mask, sha256 and each gate's verdict —
or **not built**. At the time of writing 2 of the registered rows have a built
artifact (`microsoft/phi-4/nvfp4`, `Qwen/Qwen3.5-9B/nvfp4`); the rest are
`not built` because no conversion has been run for them, which is a fact the
registry records rather than a gap in the code.

The apps and the real-weight tests resolve a model by **the registry**, so
`--model`/`--localmodel` accept an artifact directory, a path inside one, a
store directory, a `model_type`, or an `<org>/<model>` name — and a plain
checkpoint is reported as "source-only, convert it" instead of "weights missing".
A per-model `LSX_TEST_<MODEL>_ROOT` (declared as `vault.env` in the manifest)
still wins, e.g. for a tree on another disk. When the artifact or a parity
golden is genuinely absent the test SKIPs (exit code 4) and the message names
the path it looked at — a SKIP is **not** a pass.

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
- `models/<org>/<model>.json` — the manifest: one file per ingested model, laid
  out the way the store is (`microsoft/phi-4.json` describes the checkpoint at
  `<source root>/microsoft/phi-4` and the artifact at
  `<artifact root>/microsoft/phi-4/<profile>`). The single source of truth for
  its store coordinates, passport namespace, format tag, tokenizer kind,
  lifecycle-stage targets and its per-profile build state. Everything derived
  from it (target registration, registry list, `docs/MODELS.md`) is generated at
  configure time; `test_manifest_sync` fails on any drift and the generator
  refuses a row whose path disagrees with its own `vault` block.
- `tests/` — Catch2 suite (labels `fast`/`convert`/`inference`/`perf`).

Per-file inventory and which file owns what — skill `logestix-repo-map`.
What each model *is* (geometry, store coordinates, per-profile build state,
gates, throughput, debt) — **`docs/MODELS.md`**, generated from the manifests.
Per-model narrative — `docs/model-reports/`.

## TaskMosaic OpenAI bridge

`taskmosaic` is an OpenAI-compatible HTTP/SSE bridge to a running `lsxfabric`.
Start one GPU backend per model, point a bridge config at it, then use any
OpenAI client:

```bash
# 1) GPU engine (one model per process); <org>/<model>/<profile> is the artifact
lsxfabric --serve --localmodel <org>/<model>/nvfp4 \
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
completes). Every model renders and parses through **its own** chat template:
a native codec where the checkpoint's template is known and verified against
its tokenizer, the generic Hermes format (`<tools>` block +
`<tool_call>{json}</tool_call>`) as the floor otherwise.

The per-family wire formats are **not** listed here on purpose — they are
checkpoint properties, they change with each ingestion, and there are two
authorities instead:

- `libs/lsxcommon/src/*_tool_format.{h,cpp}` — the codecs, one per family; the
  header's comment states the exact markers and where they came from
  (the checkpoint's `chat_template.jinja` or vendor encoder).
- `tests/test_model_suite_guard.cpp` → `kTooling` — one row per registered
  model: the targets that prove its render+parse round-trip, or an empty row
  meaning "generic fallback, nothing claimed".

Example — DeepSeek-V4.1's native DSML dialect, verified on real weights:

```
<｜DSML｜ calls>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>
```

`ParseToolCalls` returns this as an OpenAI-shaped `ToolCall`; the same codec
renders assistant tool calls and tool results back into the transcript, so a
multi-turn agent loop round-trips through one template.

### Streaming

Every family emits `ModelRequest::token_callback` once per decoded token,
through a `GenerationStream` that also polls `abort_flag` between steps, so a
cancelled request stops instead of running to its budget. Per-family order
parity against `gen_results.gen_tokens` — no invented and no dropped tokens —
is asserted by `tests/test_streaming_assert.h`, wired into each model's
model/selfcheck targets.

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
