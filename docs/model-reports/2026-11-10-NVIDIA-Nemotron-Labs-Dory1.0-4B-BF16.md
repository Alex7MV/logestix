# NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 — ingestion report (2026-11-10)

## Publish-ready post

**What was converted.** NVIDIA's *Dory1.0-4B* (`DoryForCausalLM`, a looped
latent-recurrent normalized-GPT decoder) was adapted to a C++20/CUDA inference
engine that targets Blackwell only. The BF16 checkpoint (7.95 GB, 507 tensors)
was quantized to NVFP4 with a second NVFP4 residual plane on every projection,
the 1.3 GB embedding + LM-head pair was kept FP16, and the resulting 4.8 GiB
tree serves 94 tok/s on one RTX 5090 at 2048 generated tokens (canonical
protocol, bit-exact strict path), with 29.7 ms time-to-first-token, 424 W
average draw and a 4.8 GiB resident footprint.

**What the checkpoint forced into the engine.** Dory is not a transformer with
pre-norms; it is a *sphere* model, and four of its semantics silently break any
transformer-shaped assumption:

1. **`justnorm` replaces RMSNorm.** `x · rsqrt(Σx² + 1e-6)` in FP32 — no mean, no
   learned scale, no `sqrt(d)` factor. Every mixer output is renormalized onto
   the unit sphere, including the residual, so the hidden state is a *direction*
   and never a magnitude.
2. **The residual is a spherical interpolation.** `justnorm(justnorm(a) +
   |α·c| · (justnorm(b) − justnorm(a)))`, with `α` a learned per-dimension vector
   and `c = ngpt_alpha_init / initializer_range`. The absolute value is
   load-bearing (the sign is not a direction) and there is a second
   normalization after the interpolation.
3. **The layer stack is applied more than once.** 72 physical layers, each ONE
   mixer (attention xor MLP), partitioned 16 / 40 / 16 into input, recurrent and
   output blocks. The 40-layer recurrent block runs `n_recurrent_loops` (= 2)
   times, re-injecting the input block's output: `h = justnorm(state + input_h)`
   before each pass, `state = justnorm(h)` after it. So one token costs 112 mixer
   passes, and loop 0 starts from `justnorm(2·input_h)` — the loop is not a no-op
   at the first iteration. KV caches are `per_loop`: 56 of them (8 input + 20×2
   recurrent + 8 output).
4. **Attention is rectangular and gated.** `hidden_size` (2560) is not
   `num_heads·head_dim` (4096), so no square-geometry kernel applies. RoPE is
   applied *before* a per-head nGPT qk-norm, q is scaled by
   `(sqk·sqrt(head_dim)/std)²` elementwise after that norm while k is only
   normalized, the attention output is multiplied by
   `sigmoid(g_proj(x)·(s_gate·sqrt(√hidden)/std)²)` and then normalized again
   before `o_proj`. The MLP mirrors it with Suv-scaled SwiGLU and a normalized
   intermediate. There is **no final norm**: logits are
   `(h·Wᵀ) · logit_scale[v]/std`, with a 131072-entry per-vocabulary scale.

The reference used is the vendor implementation submitted to vLLM
(`vllm-project/vllm#60149`, `vllm/model_executor/models/dory.py`); the installed
transformers release (5.15) has no `DoryForCausalLM` and cannot even *build the
tokenizer* from the checkpoint directory, so the reference was reimplemented in
PyTorch from that file and validated against the real weights before any C++ was
written (`tools/dory_oracle.py`). That oracle produced every golden in this
report.

**Artifact-level defects and traps found in the checkpoint.**

- *`model_type: dory` has no transformers class.* `AutoTokenizer.from_pretrained`
  fails with `StrictDataclassClassValidationError` because the config-validation
  path rejects `layer_types` entries `"mlp"`. Suggested fix: ship a
  `configuration_dory.py` with the model, or publish the tokenizer as a separate
  resource. Any consumer that reaches for the generic Auto* path hits this.
- *`mlp` is a layer type, not a sub-module convention.* Fifty percent of the
  "layers" have no attention at all. A stack that assumes `layer_types` names
  only attention variants will build 36 phantom attention modules.
- *`head_dim` is explicit and larger than `hidden_size / num_heads`* (256 vs
  160). Deriving it from the ratio — the common default — silently mis-shapes
  every attention projection.
- *`attention_output_gate_logit_scale_init` is `null`,* and the effective value
  is `sqrt(hidden_size)`. A consumer that treats null as "no gate scale" gets a
  gate that saturates on the first token.
- *`eos_token_id` is a list* (`[</s>, <|im_end|>]`) while `bos_token_id` /
  `pad_token_id` are scalars, and the tokenizer sets `add_bos_token: false`.
  Reading `eos_token_id` as an int yields `</s>` (id 2) instead of the
  tokenizer's own `<|im_end|>` (id 11), and the model then never stops.
- *Reasoning is always on, and it changes answers.* The chat template defaults to
  `enable_thinking=true` and the model card enables a reasoning parser in both
  vLLM and llama.cpp. For the DET letter-match probe the model answers
  `CRITICAL` with thinking off and `CATASTROPHIC` (the correct answer) with it
  on; for the FSI/CODE probes the trace is long and self-repeating (it does not
  close the `<think>` block inside 2048 tokens). The goldens are therefore
  captured in the mode each probe needs — reasoning on for DET, off for FSI and
  CODE — and the engine frames with reasoning on by default, matching how the
  model is served. Suggested fix: document the two modes' behaviour explicitly,
  or publish a recommended `--reasoning` off variant for extraction tasks.

**Measured numbers.**

| measurement | result |
|---|---|
| SelfCheck ("The capital of France is" → "Paris") | PASS, 3 bit-exact cycles, `logits_hash=f100088895adb12d`, top-1 = 6993 (` Paris`) |
| FSI parity gate | 3/3 against source-captured goldens |
| CODE parity gate | 3/3 against source-captured goldens |
| DET determinism gate | 3/3, 3 byte-identical runs per probe (reasoning mode) |
| native tool round-trip | `get_weather {"city":"Paris"}` parsed from the converted weights |
| first 64 tokens vs the source, same prompt | identical for 43 tokens, then one near-tie diverges |
| decode throughput, 2048 tokens, strict path | **94.2 tok/s** (RTX 5090, 435–490 W) |
| decode throughput, 2048 tokens, auto policy | 93.1 tok/s |
| time to first token | 29.7 ms (short prompt) / 131–136 ms (2048-token protocol) |
| resident weights | 4.83 GiB of a 32 GiB card |
| conversion | 56.6 s wall (7.95 GB BF16 read + GPU-side quantization → 4.8 GiB tree) |
| load | 0.43–0.49 s |

**One engine defect this model exposed, and its fix.** The Blackwell W4A4
tensor-core path is tolerance-class per GEMM: measured against the scalar
reference it is 0.06 relative L2 error at one query row, rising to 0.19 at 32
rows. For a magnitude-normalized transformer that is invisible; for Dory it is
catastrophic, because every mixer output is normalized onto the unit sphere, so a
few percent of relative error in a projection becomes a *direction* error at the
very next layer and the 2×40 recurrent passes compound it. Under the per-GPU auto
policy the model answered `# Prompt\n\nWrite a short story about a man who is a
professional photographer…` to "The capital of France is" — fluent, plausible and
completely unlike the source. The family therefore pins the strict scalar path
(`FastMathMode::kOff`) for the duration of every `Infer`, exactly as the granite
and granite-switch families already do, and a regression test drives the
process-global routing to the TC path and requires the strict token stream.
Anyone ingesting another sphere/normalized architecture should expect this: an
"auto policy" that is tolerance-class for a magnitude-carrying residual is not
tolerance-class for a direction-carrying one.

**What was not measured, and why.**

- Long-context (>2048 token) behaviour, and the sliding-window (1024) versus
  full-attention split at long offsets: the perf protocol is fixed at 2048, and
  the checkpoint's own 256K context is not exercisable at that budget on one
  32 GiB card (KV at 262144 positions × 56 slots ≈ 29 GiB, which does not leave
  room for the weights).
- The documented `last_loop` KV-cache mode (one cache per recurrent attention
  layer, overwritten by each loop). Only the checkpoint default `per_loop` is
  ingested; `last_loop` is a memory/accuracy trade-off the vendor documents but
  does not ship as a separate checkpoint.
- Alternate loop counts (1 and 4). The model card presents them as an
  accuracy/compute dial with no retraining; the frozen passport pins 2, and
  supporting the dial would be a new configuration axis rather than an
  ingestion.
- A perplexity sweep against the BF16 source. The parity gates are exact-answer
  probes, not a distributional comparison; the 43/64-token agreement above is
  the only long-form comparison recorded.

## Evidence

### Stage 0 (triage) and the plan

`docs/superpowers/plans/2026-11-10-dory1-4b-ingestion.md` holds the geometry
table, the per-layer-type RoPE schedule, the ballast verdict (none: all 507
tensors classified, an unknown name is a converter error), the feasibility
numbers and the tokenizer verdict (byte-level BPE, pre-tokenizer kind 9,
`add_bos_token: false`, 1000 added tokens of which 12–17 are `special: false`).

### Source oracle

- `tools/dory_oracle.py` — the reference forward (BF16 weights, FP32
  norms/softmax/`logit_scale`), transcribed from `vllm/model_executor/models/
  dory.py` at `vllm-project/vllm#60149`. Verified against the real checkpoint
  before any C++ existed: `"The capital of France is"` (raw, unframed) →
  `" Paris."` in 8 greedy tokens, and the framed chat prompt →
  `"The user is asking a simple factual question … Paris"`.
- `tools/{fsi,code,det}_capture.py` gained a `loader: "dory"` /
  `tokenizer: "fast"` branch that routes through the oracle (the generic
  `AutoModelForCausalLM` path cannot load this checkpoint).

### Code

| piece | file |
|---|---|
| frozen passport (+ `static_assert` invariants, KV-slot count) | `libs/lsxcommon/include/lsxcommon/dory_dense_constants.h` |
| converter (`lsxforge --dory`, format `logestix-dory-nvfp4`) | `libs/lsxcommon/src/dory_dense_convert.cpp` |
| loader (pre-scaled nGPT vectors, per-loop KV slot map) | `libs/lsxcommon/src/dory_dense_load.cpp` |
| kernels (5 nGPT normalizations, sliding-window GQA + split-KV) | `libs/lsxcommon/src/dory_dense_inference.cu` |
| model class `DoryDenseModel` (kType `dory-1.0-4b`) | `libs/lsxcommon/src/dory_dense_model.cpp` |
| tool/chat codec (Nemotron-v3 ChatML + Qwen3-Coder XML) | `libs/lsxcommon/src/dory_tool_format.cpp` |
| manifest (the only hand-written registry) | `models/dory-1.0-4b.json` |

The 72-entry layer pattern is machine-checked twice: as `static_assert`s over
`frozen::LayerIsAttention/LayerIsFullAttention/CountKvSlots()` and at conversion
time against `config.layer_types` (a mismatch fails the conversion).

### Gates

```
test_dory_dense_convert      Passed   (frozen geometry, fused q|k|v + gate|up,
                                      residual plane halves the 4-bit error,
                                      unclassified tensor is an error)
test_dory_dense_loader       Passed   (FP16 edges, F32 pre-scaled nGPT vectors,
                                      KV slot map, residual planes present)
test_dory_dense_model        Passed   (registration, determinism, cooperative
                                      cancel, continuation == fresh prefill,
                                      strict-routing regression, frozen passport)
test_dory_dense_kernels      Passed   (8 host references: justnorm, justnorm-sum,
                                      slerp-alpha, qk-norm, gated-attn-norm,
                                      Suv-SwiGLU-norm, sliding-window attention,
                                      split-KV parity, logit scale)
test_dory_tool_framing       Passed   (byte-exact against the checkpoint's
                                      chat_template.jinja; XML parse round-trip)
test_dory_dense_selfcheck    Passed   (Paris, 3 bit-exact cycles)
test_dory_dense_fsi_selfcheck   Passed  3/3
test_dory_dense_code_selfcheck  Passed  3/3
test_dory_dense_det_selfcheck   Passed  3/3 (3 byte-identical runs per probe)
test_perf_baseline (dory-1.0-4b)  Passed  94.23 t/s r3 / 93.08 t/s perf
```

Guards re-run green after the change: `test_manifest_sync` (1510 assertions),
`test_model_suite_guard` (190), `test_code_naming_guard` (72),
`test_ctest_labels_guard` (39), `test_model_independence_guard` (1668).

The full `fast` label sweep is **141 targets, 140 green**. The one failure,
`test_gpu_perf` case *"memory-bandwidth probe releases its device buffers"*, is a
**pre-existing instrument flake unrelated to this model** — it was reproduced
failing 2 of 3 consecutive sweeps with every `dory` target excluded (`-E dory`),
and it passes 5/5 when run alone. Its assertion compares device-wide free memory
before and after the probe, so any other process releasing memory in that window
fails it (`free went UP 300 MiB`). Filed as `docs/OPEN_TASKS.md` § G9 with two
candidate fixes; deliberately not "fixed" here, because the honest fix is to
measure the process's own held bytes, not to widen the slack while a new model
is being ingested.

### Registry / docs rows

- `docs/SELFCHECK_MATRIX.md` §1: `| dory-1.0-4b | PASS (x3,
  f100088895adb12d) | 3/3 | 3/3 | 3/3 (reasoning mode) |`
- `docs/MODELS.md` § `dory-1.0-4b` (generated).
- `tests/perf/perf_baselines.json`: two rows (`perf` 93.08, `r3` 94.23),
  measured through `test_perf_baseline` on `linux-release-znver4-aocc`.
- `test_model_suite_guard.cpp` tool verdict:
  `{"dory-1.0-4b", {"test_dory_tool_framing", "test_dory_dense_selfcheck"}}`.

### Recorded deviations

1. **Strict routing pinned** (`FastMathMode::kOff` inside `Infer`) — see the
   engine-defect paragraph above. Pinned by
   `test_dory_dense_model.cpp::"dory pins the strict routing"`.
2. **Two capture modes.** FSI and CODE goldens are captured with reasoning off,
   DET with reasoning on; the engine's default framing leaves the think block
   open (the served mode). Both are recorded in the capture scripts' override
   comments and in `docs/SELFCHECK_MATRIX.md`.
3. **43/64-token agreement with the source** on one long-form prompt. The
   divergence is a near-tie in the 4-bit body, not a semantic break: the
   residual plane lifts the effective weight precision to ~8 bits and every
   exact-answer probe matches, but token-for-token agreement across a long
   generation is not guaranteed and is not claimed.

### Debt

None declared. All nine lifecycle stages have real-weight targets and every one
of them ran; there is no SKIP and no template-derived golden in this model's
row.
