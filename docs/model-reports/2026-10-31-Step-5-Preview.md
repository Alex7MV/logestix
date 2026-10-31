# Step-5-Preview — integration report

| | |
|---|---|
| Model | `stepfun-ai/Step-5-Preview` (`model_type = "step5_vl"`, text backbone `Step5ForCausalLM`, `model_type = "step5"`) |
| Converted layout | `logestix-step5-nvfp4` (NVFP4 body, BF16 embed/lm_head, F32 norms + router bias) |
| Source | FP8 safetensors, 606 GB, 24 text shards + 3 MTP/vision shards, 2 704 indexed text tensors |
| Converted tree | 317 GB (8.6 GiB pinned dense graph + 30 976 routed-expert arrows of 10.12 MiB) |
| Hardware | 1 × RTX 5090 (32 GB), 750 GB host RAM, CUDA 13.4 |
| Runtime | token-by-token decode, streamed mmap `ExpertStore` + device slot LRU (1 482–1 583 slots ≈ 15 GiB) |
| Date | 2026-10-31 |
| Reference | the vendor's `inference/model.py` + `inference/kernel.py` (standalone Transformers/Triton path) and `config.json` |

**Verdict at a glance:** a clean 600B/27B-active MoE to port, and the answers
are correct and deterministic on the real weights (Paris ×3 with bit-identical
logits; the native tool round-trip returns `{"city":"Paris"}`). Three
checkpoint-level facts each cost real debugging time and are worth stating in
the model card: the RMSNorm is **zero-centred** `(1 + w)`, **layer 91 is dense**
while `moe_layer_list` runs 3..90, and **layers 88–90 ship BF16 experts** (no
`weight_scale_inv`). The DSA Lightning Indexer is stored but not executed: at
the supported 4 096-token context its top-512-region selection degenerates to
the dense causal path. Full source-parity goldens (FSI/CODE/DET) were **not**
captured because the reference deployment is multi-GPU (4× B300 / 8× H200).
Perf is a correctness-first floor, not a throughput claim.

---

## Publish-ready post

### Notes from an independent NVFP4 port of Step-5-Preview

We integrated Step 5 Preview into our own C++/CUDA inference engine (no PyTorch
at runtime) and converted the FP8 release in place to NVFP4: 606 GB of
safetensors → a 317 GB tree, with the 30 976 routed experts streamed from host
mmap through a device-slot LRU. The dense graph pins at 8.6 GiB; with a
~1 500-slot expert cache the process sits at roughly 23 GiB of VRAM on one
RTX 5090. Thanks for shipping the standalone `inference/` implementation — the
packed-DSA reference made the hybrid schedule and the router arithmetic
unambiguous.

#### What worked well

- Greedy determinism is solid: the framed control prompt answers "Paris" in the
  reasoning block, and three runs produced byte-identical final logits
  (`logits_hash = 63b0f8b2fcf56d8a` over all 128 896 logits), including across a
  full model reload.
- The native tool wire is unambiguous: `<tool_call>` / `<function=NAME>` /
  `<parameter=KEY>` bodies parse cleanly. A real-weight round-trip generated the
  call after the `</think>` marker and parsed to `{"city":"Paris"}`.
- The router recipe is well specified: sigmoid scores, a selection-only bias,
  renormalised `(score − bias)`, then `× 3.0`, with the gated shared expert
  added in FP32.
- The tokenizer bridge reproduces the checkpoint tokenizer exactly (7/7
  encode+decode cases, including the digit/CJK splits and the o200k
  punctuation-run pattern).

#### Three checkpoint facts that cost debugging time

1. **The RMSNorm is zero-centred: `x / rms(x) * (1 + w)`.** The stored weights
   are centred on zero, so treating them as a conventional RMSNorm scale
   multiplies every activation by roughly zero. Shapes and dtypes do not object,
   and the first symptom is fluent-looking but empty output (in our case, a
   stream of newlines). A note in the model card would save any new integrator
   the afternoon it cost us.

2. **`moe_layer_list` is 3..90, but layer 91 is a dense-MLP full-attention
   layer.** Reading the config as "dense prefix then MoE to the end" makes the
   last layer look for 352 experts that do not exist. The mixed layout also
   means an expert store cannot assume a contiguous dense *suffix*; we bound the
   store at layer 91 and keep the flat index `layer × experts + e`.

3. **Layers 88–90 ship BF16 experts.** They are listed in
   `quantization_config.modules_to_not_convert` and have no
   `weight_scale_inv`. A converter that keys on "experts are FP8" silently stops
   at layer 88; it must accept both an FP8 block-scaled stack and a BF16 stack.

#### Things that were easy to get subtly wrong

- The expert projections are **fused stacks** `[352, 1536, 4096]` /
  `[352, 4096, 1536]`, not per-expert tensors; the converter has to split them
  and re-quantise each expert.
- The RoPE span is per layer and the table is **halved** (pair `(d, d + span/2)`,
  NEOX): 192 dims at theta 1e4 on the sliding layers, 64 dims at theta 1e7 on
  the full-attention layers. The `rope_scaling` llama3 block has `factor = 1.0`
  and is the identity.
- The per-head attention gate (`self_attn.g_proj`, one logit per head) is
  applied **before** `o_proj`.
- The final RMSNorm narrows the FP32 residual to BF16 **first**; that rounding
  boundary is part of the reference arithmetic.
- The router gate is BF16 on disk but must be evaluated in FP32.

#### What we did not run

The DSA indexer is stored in the converted tree but not executed. At or below
`topk × region_block_size = 4096` tokens every region is selected, so the dense
causal path *is* the reference path; beyond that our engine refuses the request
rather than silently producing a different answer. Implementing the FP8 region
summaries, the radix top-k with its zero-group tie-break and the region gather
is the remaining work for the 1M-token claim.

We also could not capture FSI/CODE/DET-style parity goldens: the reference
deployment in the model card starts at 4× B300 or 8× H200, and the 606 GB FP8
source does not fit on a single 32 GB card. We validated at the kernel level
against host references instead (zero-centred norm with the BF16 narrowing,
per-head QK norm, partial RoPE, router selection/coefficients, clamped SwiGLU,
windowed GQA attention, FP32 combine). If the authors publish a recommended
NVFP4 recipe — in particular which tensors must stay high precision — that
would be very useful, since the routed experts are ~96% of the bytes.

#### One data point on speed

Our correctness-first path (token-by-token prefill/decode, a router readback per
MoE layer, no CUDA graphs, no prefetch stream) reaches **1.82 t/s at 2 048
generated tokens** (TTFT ≈ 14.5 s) on the RTX 5090, with the 6.6 GB/token
expert stream as the bottleneck. That is a floor, not a verdict on the model's
throughput — it mostly measures our host-to-device staging.

Overall: a well-specified model to port, and the three checkpoint facts above
are the only things that cost real debugging time.

---

## Evidence

### Gates actually run (2026-10-31)

| gate | result | note |
|---|---|---|
| SelfCheck (`Paris`, framed, kOff) | **PASS**, ×3 | `logits_hash = 63b0f8b2fcf56d8a`, 128 896 logits, byte-identical across the three cycles; 8 806 MiB pinned + ~15 000 MiB expert cache |
| SelfCheck, kAuto path | **PASS** | 8 tokens, two full model loads, byte-identical logits |
| Tool round-trip (real weights) | **PASS** | 80 tokens, `</think>` then `<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>`; parser returns `{"city":"Paris"}` |
| Kernel parity vs host reference | **PASS** | zero-centred norm + BF16 narrowing, per-head QK norm, partial NEOX RoPE, sigmoid+selection-bias router, clamped SwiGLU, windowed GQA attention, per-head gate, FP32 combine |
| Tokenizer parity | **PASS 7/7** | encode + decode round-trip against the checkpoint's own `tokenizer.json` |
| Converter / loader / model / tool framing (synthetic) | **PASS** | labels `convert` / `fast` / `inference` |
| FSI / CODE / DET | **SKIP (declared debt)** | source oracle is multi-GPU (4× B300 / 8× H200); rows in `kGoldenDebt`: `step-5-preview:{fsi,code,det}` |
| Perf | **seeded** | 1.82 t/s (perf) / 1.83 t/s (r3), TTFT 14.49 s / 14.35 s, 2 048 tokens, 127.63 W / 129.86 W avg (145.47 W / 147.72 W peak) |

### Defect found by the determinism gate (fixed)

The attention block first took the RoPE span from the frozen passport (192/64
dims) while the synthetic fixture's head is only 32 dims, so the rope kernel
wrote past the head into neighbouring scratch buffers and the model was not
bit-reproducible in-session. The span is now clamped to the actual `head_dim`
(`min(frozen_span, HD)`); production geometry is unchanged because 192/64 ≤ 192.

### Architecture as implemented

- 92 layers, 4096 hidden, 64 Q / 4 KV heads at head_dim 192, GQA with 16 Q per
  KV group; `layer_types` gives 69 sliding (window 512) and 23 full-attention
  layers (every fourth, starting at 3).
- Per-head QK RMSNorm (F32 weights, zero-centred), NEOX partial RoPE, sigmoid
  per-head output gate before `o_proj`.
- Layers 0–2 and 91 dense SwiGLU (I=13 824); layers 3–90 routed MoE (352
  experts, top-8, I=1 536) + one gated shared expert (I=1 536), combined in
  FP32. Router: F32 sigmoid logits from a BF16 table + F32 selection-only bias,
  weights renormalised from `(score − bias)` and scaled by 3.0.
- SwiGLU clamp `min(silu(gate), 7) × clamp(up, −7, 7)`.
- FP32 residual stream; final norm narrows to BF16 before the (BF16) head.
- Stripped and counted: MTP layers 92–94 (`skipped_mtp_tensors = 51`,
  `num_mtp_layers_stripped = 3`) and the 667-tensor vision tower.
- Stored-not-applied: the DSA indexer weights (184 tensors, 834 MB under
  `indexer/`).

### Evidence links

- Open work, the checkpoint traps and the DSA >4096 plan:
  `docs/OPEN_TASKS.md` §R9.
- Gate matrix row: `docs/SELFCHECK_MATRIX.md` (§2, declared debt).
- Perf rows: `tests/perf/perf_baselines.json` (`step-5-preview`, `perf` + `r3`).
- Tests: `tests/test_step_moe_{convert,loader,kernels,model,selfcheck,fsi_selfcheck,code_selfcheck,det_selfcheck,tokenizer,tool_framing}.cpp`.
- Passport + guards: `libs/lsxcommon/src/step_moe_constants.h`,
  `test_model_independence_guard` (`kPassports`/`kLifecycle`),
  `test_model_suite_guard` (`Suites()` + tool verdict + `kGoldenDebt`).
