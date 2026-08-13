# Centralized Metadata Vault (model-metadata/) — Design

## Purpose

Phase 4. Establish a single, platform-uniform root for all model structural JSON
manifests and binary translation passport files. Today every app hardcodes a
model checkpoint path (`D:\Programs\huggingface.co\...` on Windows,
`/mnt/llm_models/models/...` on Linux) and appends per-model `logestix/pinned`
subdirectories ad hoc. This work centralizes that under one invariant
`model-metadata/` block appended automatically to the resolved platform root.

## Platform Selector

Compile-time preprocessor selection in a single new header. No runtime
detection, no registry. Exactly two targets, per CC 12.0 forward-compatible
build matrix:

| Platform | Root base |
|---|---|
| `_WIN32` | `D:\Programs\huggingface.co` |
| else (Linux) | `/mnt/llm_models/models` |

Every metadata read/write query resolves to `model_metadata_root()`:

```
{root base}/model-metadata/
```

`model_metadata_root()` is the concatenation of the selected base + the invariant
relative block `model-metadata/`. It is the only path authority; no consumer
reconstructs the root.

## Resulting vault layout (identical on both platforms)

```
{root}/model-metadata/
    manifests/*.json        structural model manifests
    passport/passport.bin   binary translation passport
```

## Scope

### New files

- `libs/lsxcommon/include/lsxcommon/path_registry.h`
  - compile-time `#if defined(_WIN32)` / `#else` base selection
  - `model_metadata_root()` declaration (returns `std::string`)
- `libs/lsxcommon/src/path_registry.cpp`
  - `model_metadata_root()` definition

### Routing manifest consumers (read/write -> vault root)

- `libs/lsxcommon/src/model_factory.cpp` — `DetectType()` / default `Create()`
  path resolution
- `libs/lsxcommon/src/model_registry.cpp` — registry init reads structural JSON
  manifest from `model_metadata_root()`
- `libs/lsxcommon/include/lsxcommon/model_factory.h` — expose vault-root accessor
- `libs/lsxcommon/src/glm_v5_model.cpp` + `deepseek_v4_flash_model.cpp` —
  tokenizer/metadata load re-rooted
- `libs/lsxcommon/src/inference_load.cu` — `pinned/shared_experts.arrow`
  resolution re-rooted
- `libs/lsxcommon/src/glm_convert.cpp` + `glm_inference_load.cpp` —
  `SaveGlmPinnedTensors` / `WriteGlmModelIndex` output and
  `LoadGlmPinnedWeights` input re-rooted

### Routing binary passport files (translation passport)

- `tools/lsx_token_packer.cpp` — default `--out` resolves to
  `model_metadata_root() + passport/passport.bin`
- `libs/lsxnetwork/include/lsxnetwork/lsx_token_translator.h` — default passport
  path constant declared; `Load()` default routed to vault
- `libs/lsxnetwork/src/lsx_token_translator.cpp` — passport read/write default
  re-rooted to vault

### CLI / app defaults

- `apps/lsxcli/src/main.cpp`, `apps/lsxfabric/src/main.cpp`,
  `apps/lsxforge/src/main.cpp` — hardcoded `D:\Programs\huggingface.co\...`
  defaults replaced by vault-root resolution; `lsxforge` pinned-dir construction
  (`model_path/logestix/pinned`) re-rooted to `model_metadata_root()`

## Error handling

- `model_metadata_root()` never fails; it is pure path construction.
- I/O failures (manifest missing, passport unreadable) surface at the existing
  call sites; no new failure modes introduced.
- `model_metadata_root()` + `path_registry` create directories lazily where the
  caller already creates dirs (`lsxforge`, converters); no new mkdir in the
  selector.

## Testing

- Unit test for `model_metadata_root()` platform selection (compile-time
  branch; assert suffix is `model-metadata/` and prefix matches base).
- Existing real-model tests unaffected: they SKIP when the checkpoint/vault
  is absent. Synthetic fixture tests continue to use explicit temp paths and
  are not re-routed.

## Out of scope

- No migration of on-disk existing checkpoints into the vault.
- No runtime path override mechanism beyond defaults (not YAGNI's requirement).
- No changes to CUDA kernels, inference math, or occupancy.