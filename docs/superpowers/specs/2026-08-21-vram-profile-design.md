# VRAM Requirements Profile — Design

Date: 2026-08-21
Status: Approved (design), pending implementation plan

## Purpose

Every model gets three computed VRAM parameters, calculated during model
conversion and persisted in `model.index.json`:

| Parameter | Meaning |
|---|---|
| `min_bytes` | Minimal VRAM to launch the model: device-resident weights only |
| `optimal_bytes` | Optimal VRAM: weights + KV-cache + workspace at the model's context length |
| `max_bytes` | Maximal VRAM for full placement: optimal + all routed experts staged in device memory |

The values are the foundation for subsequent features (scheduling, capacity
checks, UI display).

## Scope

- New estimation module `lsxcommon::vram_estimate` (host-only, no CUDA).
- All three conversion pipelines write a mandatory `vram` block into their
  `model.index.json`: DeepSeek (`converter.cpp`), GLM (`glm_convert.cpp`),
  Qwen (`qwen_convert.cpp`).
- Runtime exposure via `IModel::VramRequirements()`.

## Non-goals

- Backward compatibility with old `model.index.json` files lacking the
  `vram` block — existing models are reconverted after this change.
- Measured (dry-run load) VRAM numbers; values are analytic estimates.
- Vault-level catalog (`manifests/models.json`) — not part of this design.
- GPU-side or NVCC-compiled code paths.

## 1. Estimation Module

Files:

- `libs/lsxcommon/include/lsxcommon/vram_estimate.h`
- `libs/lsxcommon/src/vram_estimate.cpp`

Added to `lsxcommon` STATIC library sources (host-only C++, never included
from `.cu`/`.cuh` TUs).

```cpp
namespace lsxcommon {

struct VramTensorInfo {
    std::string name;           // safetensors tensor name
    std::int64_t numel = 0;     // element count
    std::int64_t raw_bytes = 0; // original size: dtype_size * numel
};

struct VramGeometry {
    int hidden_dim = 0;
    int kv_dim = 0;             // KV latent dim (attention value width)
    int num_attn_layers = 0;
    int num_moe_layers = 0;
    int experts_per_layer = 0;
    std::int64_t context_length = 0; // from source config.json
};

struct VramProfile {
    std::uint64_t min_bytes = 0;
    std::uint64_t optimal_bytes = 0;
    std::uint64_t max_bytes = 0;
    std::int64_t context_tokens = 0; // seq_len actually used in the estimate
};

// classify_nvfp4: converter marks which tensors it quantizes to NVFP4
// (the converter's quantization rules are authoritative); all other tensors
// keep their original dtype size.
VramProfile EstimateVram(
    absl::Span<const VramTensorInfo> tensors,
    bool (*classify_nvfp4)(const std::string &name),
    const VramGeometry &geom);

} // namespace lsxcommon
```

## 2. Formulas

Converted tensor size:

- NVFP4 tensor: `numel / 2` bytes (packed E2M1 nibbles) + `numel / 16` bytes
  (per-block E4M3 scales, block of 16).
- Other tensors: `raw_bytes` (norms and biases stay BF16/FP16 — precision
  invariant preserved).

Components:

```
kv_tokens  = sum over attention layers of ceil(context_tokens / ratio)
             (ratio = per-layer KV compression from config "compress_ratios";
              ratio <= 1 or absent entry -> full-context storage;
              empty list -> every layer stores the full context)
kv_cache   = kv_tokens * kv_dim * 2 * 2          // K+V, FP16
workspace  = prefill_chunk * hidden_dim * num_attn_layers * 2 * 2   // FP16
             (prefill_chunk = 2048 tokens: decode is single-token and prefill
              runs in bounded chunks, so transient workspace must not scale
              with the full context)
expert_pool = num_moe_layers * experts_per_layer * per_expert_converted_bytes

min_bytes     = sum(converted sizes, excluding routed-expert tensors)
optimal_bytes = min_bytes + kv_cache + workspace
max_bytes     = optimal_bytes + expert_pool
```

`context_tokens` in the profile records the seq_len used. Empty inventory →
zero profile.

## 3. Converter Integration

### Context length source

Each converter reads its source `config.json`; field fallback chain:
`max_position_embeddings` → `context_length` → `seq_length` → `2048`.
Missing fields produce a spdlog warning; conversion does not fail.

### NVFP4 classification per converter

- DeepSeek (`converter.cpp`): `is_staged_tensor()` + shared experts → NVFP4.
- GLM (`glm_convert.cpp`): `TensorKind::RoutedExpert`, `TensorKind::SharedExpert` → NVFP4.
- Qwen (`qwen_convert.cpp`): dense linear matrices → NVFP4; norms/biases — no.

### index.json schema addition (mandatory)

All three writers extend their hand-rolled JSON builders with an identical
block via a shared helper `AppendVramJson(profile)` in `vram_estimate.cpp`:

```json
"vram": {
    "min_bytes": 8734629888,
    "optimal_bytes": 9126804480,
    "max_bytes": 412316860416,
    "context_tokens": 163840
}
```

Writers touched:

- `WriteModelIndex` — `converter.cpp:962` (format `logestix-v1`)
- `WriteGlmModelIndex` — `glm_convert.cpp:365` (format `logestix-glm`)
- `WriteQwenModelIndex` — `qwen_convert.cpp:283` (format `logestix-qwen`,
  nested `logestix/model.index.json` path unchanged)

The block is mandatory: writers always emit it.

### Readers (strict parsing)

All three topology readers parse the `vram` block strictly. A missing or
malformed block is a topology error (`error_msg` set / parse failure):

- `glm::Topology` + `load_topology` — `glm_convert.cpp:213`; `Topology`
  gains `VramProfile vram;`
- Qwen `load_topology` — `qwen_convert.cpp`
- DeepSeek `load_topology` — `deepseek_inference_load.cu:199`

Old index files are invalid by contract; models are reconverted.

## 4. Runtime Surface (IModel)

Additive interface change in `include/lsxcommon/model.h`:

```cpp
#include "lsxcommon/vram_estimate.h"

// VRAM requirements computed at conversion time, read from model.index.json.
virtual const VramProfile &VramRequirements() const { return kEmptyVramProfile; }
```

`kEmptyVramProfile` — static empty profile, keeps test fakes compiling
(same pattern as the `ResidentWeightBytes()` default).

Population: `ModelBase` gains protected member `VramProfile vram_;`. Each
model's `Load()` already parses its `model.index.json` for topology — the
same parse fills `vram_`. `VramRequirements()` is overridden once in
`ModelBase`; concrete models do not duplicate code.

Consumers: lsxcli (`--info` / `--infofull`) displays the three
values formatted via `format_size()` (GiB). The factory may expose profiles
pre-load later (index-only read) if needed.

## 5. Error Handling

- Conversion: missing context field in config.json → fallback 2048 + warning;
  conversion proceeds.
- Model load: missing/malformed `vram` block in index.json → `Load()` returns
  false with an error naming the file (strict contract post-reconversion).
- Estimation: empty inventory → zeroed profile (no error).

## 6. Testing

New test file `tests/test_vram_estimate.cpp` (Catch2):

- NVFP4 packing math (`numel/2 + numel/16`) vs raw-dtype preservation.
- Hierarchy invariant `min <= optimal <= max` on synthetic inventories.
- KV-cache/workspace scale with supplied `context_length`.
- Empty inventory → zero profile.
- Per-converter classify functions on representative names.

Extended tests:

- `tests/test_glm_convert.cpp`: round-trip — synthetic convert produces
  index.json containing `vram` with expected byte values; `load_topology`
  returns the profile; removing the block fails parsing.
- `tests/test_qwen_convert.cpp`: same for `logestix-qwen`.
- `tests/test_deepseek_loader.cpp`: strict-parse coverage for the DeepSeek
  reader.

## Acceptance

1. Converting any supported model emits `model.index.json` with a valid,
   internally consistent `vram` block (`min <= optimal <= max`).
2. Loading a converted model exposes the profile through
   `IModel::VramRequirements()`.
3. All new and extended tests pass under `ctest --preset windows-debug`.
