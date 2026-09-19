# --testprompt: Text Generation Benchmark

## Overview

Add `--testprompt` option to `lsxfabric` — hardcoded-prompt text generation with
basic latency/throughput measurement.  Phased approach: Phase 1 = FFN‑only
(RMSNorm + MoE Router + Shared+Routed Experts + LM Head), Phase 2 = +MLA+HC.

## Phase 1 Scope

- Hardcoded prompt: `"The capital of France is"`
- BPE tokenizer via `tokenizer.json` (simdjson‑parsed)
- Full forward pass without attention: RMSNorm → Router → Shared FFN → Routed FFN
- Autoregressive loop (argmax sampling), configurable token count
- Metrics: TTFT, tokens/s, total time, final loss

## Architecture

### New Files

`libs/lsxcommon/include/lsxcommon/`
- `tokenizer.h`         — BPE encode/decode API
- `inference_gpu.h`     — `RunInference()`, `InferenceConfig`, `InferenceResult`

`libs/lsxcommon/src/`
- `tokenizer.cpp`              — simdjson parser for tokenizer.json, BPE logic
- `inference_load.cu`          — load pinned tensors (global + per-layer) to VRAM
- `inference_norm.cu`          — RMSNorm CUDA kernel
- `inference_moe.cu`           — Router (gate→softmax→top‑k) + Shared+Routed FFN
- `inference_head.cu`          — LM Head GEMM + argmax sampling
- `inference_pipeline.cu`      — orchestrator, autoregressive loop, timing

`apps/lsxfabric/src/`
- `main.cpp` (modified)        — add `--testprompt` flag

### Data Structures

```cpp
struct InferenceConfig {
    std::string model_path;
    int max_gen_tokens = 64;
};

struct InferenceResult {
    double ttft_ms = 0.0;           // time to first token
    double tokens_per_sec = 0.0;
    double total_time_ms = 0.0;
    int num_tokens = 0;
    int num_layers = 0;
    float final_loss = 0.0f;
    std::string output_text;
    std::string error_msg;
    bool ok = false;
};
```

### PinnedWeights (GPU layout)

```
PinnedWeights (device pointers)
├── d_embed       [vocab_size, hidden_dim]    FP16
├── d_head        [vocab_size, hidden_dim]    FP16
├── d_final_norm  [hidden_dim]                FP16
└── layers[43]
    ├── d_ffn_norm      [hidden_dim]          FP16
    ├── d_gate_weight   [hidden_dim, n_routed]  FP16
    ├── d_shared_w1     NVFP4 packed
    ├── d_shared_w2     NVFP4 packed
    ├── d_shared_w3     NVFP4 packed
    └── d_shared_*_scales / global_scale
```

### Tokenizer (`tokenizer.h`)

```cpp
class BpeTokenizer {
public:
    explicit BpeTokenizer(const std::string &tokenizer_json_path);
    std::vector<int32_t> Encode(const std::string &text);
    std::string Decode(const std::vector<int32_t> &ids);
    int vocab_size() const;
    int bos_id() const;
    int eos_id() const;
};
```

Implementation:
- Parse `tokenizer.json` with simdjson: extract `model.vocab`, `model.merges`,
  `added_tokens`, pre‑tokenizer regex
- Pre‑tokenize: GPT‑2 byte‑level splitting
- BPE merge loop per word token
- Add BOS/EOS from added_tokens
- Decode: id→token lookup, byte‑level unescape

### GPU Kernels

**RMSNorm** (`inference_norm.cu`):
```
grid = 1, block = 256
sum(x^2) → block reduce → sqrt(mean + eps) → x / rms * weight
```

**Router** (`inference_moe.cu`):
```
gate_logits = h × gate_weight^T    (cublasGemmEx FP16, [1×h] × [h×256] → [1×256])
softmax(gate_logits) → probabilities
top‑k indices + weights             (custom kernel, k=4 or from model config)
```

**Shared+Routed FFN** (`inference_moe.cu`):
- Reuse `dequant_nvfp4_to_half_kernel` from benchmark
- Reuse `cublasGemmEx CUDA_R_16F` pattern from benchmark
- Shared: `silu(w1×h)*(w3×h) → w2×result`
- Routed: for each of k experts, weighted sum

**LM Head + Argmax** (`inference_head.cu`):
```
logits = h × head_weight^T    (cublasGemmEx FP16, [1×h] × [h×vocab] → [1×vocab])
argmax kernel: grid=(vocab+255)/256, block=256, block reduce → global max
```

**Pipeline** (`inference_pipeline.cu`):
```
1. tokenizer.Encode(prompt) → ids
2. embed[ids[0]] → h     (call only — prefill the first token)
3. for token in 0..max_gen:
     for layer in 0..42:
       rms_norm → router → shared_ffn → routed_ffn → h += shared + routed
     final_norm → lm_head → argmax → next_id
     embed[next_id] → h
     record timing per token boundary
4. TTFT = time to first argmax
5. tokens/s = (max_gen-1) / elapsed_decode
```

### CLI Integration

```cpp
// main.cpp
bool testprompt = false;
if (strcmp(argv[i], "--testprompt") == 0)
    testprompt = true;

if (testprompt) {
    InferenceConfig cfg;
    cfg.model_path = model_path;
    auto result = RunInference(cfg);
    // print results
}
```

Output format:
```
=== --testprompt Results ===
  Generated tokens    : 64
  Time to first token : 1234.5 ms
  Generation speed    : 15.3 t/s
  Total time          : 5678.9 ms
  Final loss          : 2.345
  Output text         : "The capital of France is..."
```

### Router top‑k

Default value: `k = 8` (common for DeepSeek‑style MoE).  Load from
`model.index.json` or hardcoded — TBD during implementation.

### Prefill vs Decode (Phase 1)

Phase 1 has no KV cache, so prefill and decode are the same operation
(seq_len = 1).  The loop is simply:

```
for token in 0..max_gen:
    if token == 0:
        h = embed[prompt_ids[0]]
    else:
        h = embed[prev_token_id]
    for layer in 0..42:
        rms_norm → router → shared_ffn → routed_ffn → h += shared + routed
    final_norm → lm_head → argmax → next_token_id, loss
```

TTFT = time from `loop start` to `first argmax`.

### BF16→FP16 Conversion

Pinned tensors are stored as BF16 (original safetensors format).  Convert on
upload to GPU via element‑wise kernel (`__bfloat162float` → `__float2half`).
cuBLAS GEMM uses `CUDA_R_16F`.

### Limits & Constraints

- Phase 1: no attention → generated text is not coherent (stub = identity)
- Phase 2 adds: MLA attention + KV cache + HC components
- KV cache for Phase 2: `[max_seq_len × num_layers × kv_latent_dim × 2]` FP16
- All pinned tensors stay in VRAM for the lifetime of the process
- Experts loaded via mmap (same as benchmark), streamed per layer

### Testing

- Integration test in `tests/test_core.cpp`: run `--testprompt`, verify output
  contains expected metrics and non‑empty text
- Unit test for tokenizer: known string → known IDs (compare with Python)

### Build Changes

`libs/lsxcommon/CMakeLists.txt`:
- Add `tokenizer.cpp` to `lsxcommon` sources
- Add `inference_load.cu`, `inference_norm.cu`, `inference_moe.cu`,
  `inference_head.cu`, `inference_pipeline.cu` to `lsxcommon_cu` sources
