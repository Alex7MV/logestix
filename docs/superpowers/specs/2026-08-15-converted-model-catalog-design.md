# Design: Catalog of Converted Models (model-metadata registry)

## Purpose

After a successful conversion, the converter must record up-to-date information
about available converted models in the `model-metadata` vault. This gives a
deterministic, machine-readable view of "which models are converted and to what
state", and a source-hash to decide whether a re-conversion is needed.

## Scope

- Only the **base `lsxcommon::Converter`** (the `--input`/`--output` path, format
  `logestix-v1`). GLM/Qwen converters are handled by their own pipelines and are
  **out of scope** for this change.
- Write path, read path (listing), and tests.
- No GPU work.

## Non-goals

- No binary/passport format (passport.bin is a separate token translator table).
- No auto re-conversion on stale hash (the hash is stored as information only).
- No per-model subdirectories; a single JSON manifest is kept.

## Vault

Manifest path: `lsxcommon::lsx_model_metadata("manifests/models.json")`, i.e.
`<base>/model-metadata/manifests/models.json`. This reuses the existing
`manifests/` convention already exercised by `tests/test_path_registry.cpp`
(`lsx_model_metadata("manifests/model.index.json")`).

## Module

New `lsxcommon::converted_models` module:

- `include/lsxcommon/converted_models.h`
- `src/converted_models.cpp`
- Registered in `libs/lsxcommon/CMakeLists.txt` `add_library(lsxcommon ...)`.

### Record

```cpp
struct ConvertedModelRecord {
    std::string model_name;     // fs::path(source_path).filename()
    std::string format;         // "logestix-v1"
    std::string source_path;    // absolute source dir
    std::string output_path;    // absolute converted output dir
    std::string converted_at;   // ISO 8601 UTC timestamp
    int         num_layers;
    int         experts_per_layer;
    int64_t     total_pinned_tensors;
    int64_t     total_experts;
    std::string source_sha256;  // FNV-1a 64, hex-encoded
};
```

### Source hash

Deterministic FNV-1a 64-bit over a canonical serialization of the scanned
`SafetensorsModel`: for each shard in scan order, `"<filename>|<file_size>|"`,
and for each tensor `"<name>|<dtype>|<byte_size>|"`. `scan_directory` is already
deterministic (index-order, else sorted), so the hash is reproducible across
runs. Hex-encoded lowercase string.

### API

```cpp
// Read the manifest; empty vector if absent/unparseable (best-effort).
std::vector<ConvertedModelRecord> ReadConvertedModelManifest();

// Upsert by (model_name, format): existing entry is replaced, never duplicated.
// Rewrites the file sorted by (model_name, format). Returns Status.
arrow::Status UpdateConvertedModelManifest(const ConvertedModelRecord& rec);

// Explicit-path variants (used for deterministic writes; the vault wrappers
// below forward to them with lsx_model_metadata("manifests/models.json")).
arrow::Status WriteManifestTo(const std::filesystem::path& manifest_path,
                              const std::vector<ConvertedModelRecord>& recs);
std::vector<ConvertedModelRecord> ReadManifestFrom(
    const std::filesystem::path& manifest_path);

// Log each record via spdlog::info.
arrow::Status LogConvertedModels(const std::vector<ConvertedModelRecord>& recs);
```

The vault-path wrappers delegate to the explicit-path variants, so the manifest
path is always explicit and never reconstructed twice. `UpdateConvertedModelManifest`
creates `manifests/` if missing. On parse failure of an existing but corrupt file,
the existing content is treated as empty and replaced (a corrupted manifest is not
a reason to keep stale data, but a `spdlog::warn` is emitted).

## Converter integration

At the very end of `Converter::Run` (after `VerifyConvertedModel` succeeds):

1. Build a `ConvertedModelRecord` from the conversion's known values
   (`model_name = fs::path(input).filename()`, `format = "logestix-v1"`,
   source/output paths, geometry counters, FNV hash of the scanned model).
2. Call `UpdateConvertedModelManifest(rec)`.
3. On failure, log `spdlog::warn` only — the model is already valid; the registry
   is best-effort and must not fail the conversion.

## CLI

Add `lsxforge --list-models`: reads and logs the manifest via
`LogConvertedModels(ReadConvertedModelManifest())`. No output-path argument
needed; the vault path is fixed by the registry.

## Tests

New `tests/test_converted_models.cpp` (read-only, no GPU):

1. Round-trip: upsert a record, read back, fields match.
2. Upsert/no-duplicate: upsert two records with the same `(model_name, format)`
   (different timestamps), read back => single entry with the latest values; a
   second distinct `(model_name, format)` produces two entries.
3. Determinism: same `SafetensorsModel`-equivalent serialization yields the same
   hash.
4. Resilience: a missing manifest reads as empty; a corrupt manifest is replaced
   on the next upsert without throwing.

Tests exercise the explicit-path variants against a temp dir, never the real
vault. The vault itself is only asserted via `test_path_registry` (already
existing). Registered in `tests/CMakeLists.txt`.

## Acceptance

- `lsxforge --input <src> --output <dest>` writes/updates
  `<base>/model-metadata/manifests/models.json`.
- Re-converting the same model never grows the record set (upsert).
- `lsxforge --list-models` prints the catalog.
- `tests/test_converted_models` passes under `ctest`.