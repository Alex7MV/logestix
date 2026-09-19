# Contributing to logestix

Thanks for your interest in contributing!

## Contributor License Agreement (CLA)

logestix is dual-licensed: the open-source edition is AGPL-3.0, and a separate
commercial license is offered for enterprise deployments. To keep that model
viable, **every contributor must sign the Contributor License Agreement (CLA)
before a Pull Request can be merged**. The signing flow is automated via GitHub
as soon as you open a Pull Request.

## Getting started

```bash
git clone --recurse-submodules https://github.com/Alex7MV/logestix.git
cd logestix
git submodule update --init --recursive

cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
```

See `README.md` for the Windows and AOCC release flows, and
`docs/NVIDIA_DOCS.md` for the mandatory NVIDIA documentation workflow before
touching NVFP4/CUDA kernel semantics.

## Development rules

The single source of truth for engineering rules is `AGENTS.md`, with detailed
per-area skills under `.opencode/skills/`. The inviolable parts:

- **Target architecture**: NVIDIA Blackwell (CC 12.0/12.1) only. Pre-Blackwell
  targets (sm_90, sm_80) are rejected; there is no backward compatibility.
- **Logging**: `spdlog` only — no `std::cout`/`printf`.
- **Format strings**: `absl::StrFormat` uses printf style (`%s/%d/%u/%f`), not
  fmt `{}` placeholders (enforced by `test_absl_format_guard`).
- **Containers**: host-only C++ prefers `absl::flat_hash_map`/`btree_map`; headers
  reachable from `.cu`/`.cuh` must not include `<absl/container/*>`.
- **NVFP4**: route matrix math through the model-neutral
  `lsxcommon/lsx_inference_nvfp4.h`; do not add parallel named dispatchers.
- **Tests**: every test carries exactly one CTest label (`fast` / `convert` /
  `inference` / `perf`) assigned in the same commit (enforced by
  `test_ctest_labels_guard`). Guard-test registries must be extended in the same
  commit as the files they cover.

## Commit messages

English, with a type prefix: `build:`, `feat:`, `fix:`, `refactor:`, `test:`,
`docs:`.

## Pull requests

1. Create a topic branch.
2. Keep the change focused; extend the relevant guard tests and label registries
   in the same commit.
3. Make sure the fast suite is green: `ctest --preset linux-debug` (fast-only).
4. Open the PR — the CLA prompt appears automatically.

By contributing you agree that your contributions are licensed under the
project's AGPL-3.0 license and may also be distributed under the commercial
license.
