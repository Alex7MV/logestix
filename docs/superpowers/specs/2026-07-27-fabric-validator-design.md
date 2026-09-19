# Fabric Validator Design

## Purpose

`lsxfabric --validate` runs an integrity check of a logestix model:
the index, pinned tensors, MoE experts (NVFP4 encoding, E4M3 scales,
global scales).

## CLI

```
lsxfabric --localmodel PATH --validate
```

New `--validate` flag. Exit code 0 on success, 1 on errors.

## Modules

New file `apps/lsxfabric/src/validator.h` / `validator.cpp`.

### Structs

```cpp
struct ModelInfo {
    std::string format;
    std::string model_name;
    int num_layers;
    int experts_per_layer;
    int64_t total_pinned;
    int64_t total_experts;
};

struct ValidationResult {
    bool ok = true;
    int errors = 0;
    int warnings = 0;
};
```

### Functions

```cpp
// Parse model.index.json
Result<ModelInfo> LoadModelIndex(const std::string &index_path);

// Load pinned .arrow, verify the schema, log
ValidationResult ValidatePinnedTensors(const std::string &pinned_dir,
                                       int64_t expected_count);

// Load one expert, verify all fields
ValidationResult ValidateExpert(const std::string &arrow_path,
                                int layer, int expert);

// NVFP4 data validation: every nibble ∈ [0x00, 0x0F]
// No invalid codes
bool ValidateNVFP4Data(const uint8_t *data, int64_t nbytes);

// E4M3 scale validation: sign=0, exp4 ∈ [1, 14]
bool ValidateE4M3Scales(const uint8_t *scales, int64_t count);

// Global scale: finite, > 0
bool ValidateGlobalScale(float scale);

// Full validation
ValidationResult RunValidator(const std::string &model_path);
```

### RunValidator flow

1. Reads `model.index.json` → `ModelInfo`
2. Checks the `logestix-v1` format
3. Validates `pinned/`:
   - Loads `pinned/global.arrow`
   - For each layer loads `pinned/layer_NNN.arrow`
   - Checks that the RecordBatch reads without errors
4. Validates experts:
   - For each layer, for the first expert (exp_000.arrow):
     - Reads the RecordBatch
     - Checks the 9 fields in the schema
     - Validates the NVFP4 data w1/w2/w3
     - Validates the E4M3 scale arrays w1/w2/w3
     - Validates global_scale (float) w1/w2/w3
5. Summary: how many experts checked, how many errors, how many pinned

### main.cpp

```cpp
if (validate) {
    auto result = RunValidator(model_path);
    return result.ok ? 0 : 1;
}
```

### Testing

New `apps/lsxfabric/tests/test_validator.cpp` with Catch2:
- `LoadModelIndex` with a synthetic index.json
- `ValidateNVFP4Data` with valid/invalid data
- `ValidateE4M3Scales` with valid/invalid bytes
- `ValidateGlobalScale` with 0, negative, inf, normal
