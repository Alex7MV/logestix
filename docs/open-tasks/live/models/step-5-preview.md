# Step-5-Preview — open items

### R9. Step-5-Preview (600B-A27B) — ingestion landed; DSA >4096 and parity goldens open (2026-10-31)

**What landed.** Independent family `step` (`kType = step-5-preview`, format tag
`logestix-step5-nvfp4`), converted in place from the 606 GB FP8 release
(`stepfun-ai/Step-5-Preview`, 24 text shards + `model-mtp3*` + `model-vit*`)
into a 317 GB tree: 8.8 GiB pinned dense graph + a 307 GB routed-expert store
(30 976 arrows of 10.12 MiB) streamed through the neutral mmap `ExpertStore`,
`first_moe_layer = 3` (`LSX_STEP_EXPERT_MAX_OPEN`, default 4096) and a device
slot LRU (`LSX_STEP_EXPERT_CACHE_SLOTS`, default sized from free VRAM; 1482-1583
slots ≈ 15 GiB on this card). Kernels: FP32 residual stream with zero-centered
`(1+w)` RMSNorm, per-head QK norm, NEOX-paired partial RoPE (192-dim sliding /
64-dim full, theta 1e4 / 1e7), per-head sigmoid output gate, windowed GQA
decode attention, sigmoid + selection-only-bias router (top-8, renorm, ×3.0),
clamped SwiGLU (`min(silu(g),7) * clamp(u,-7,7)`) for the routed experts and the
gated shared expert combined in FP32.

**Geometry traps the source forces (each was a candidate for a silently wrong
token).**
1. **Zero-centered norms.** `Step5RMSNorm` is `x/rms(x) * (1 + w)` with F32
   weights; the checkpoint's weights are centred on zero, so reading them as a
   plain scale multiplies every activation by ~0. First run of the real weights
   emitted newlines; the fix took the Paris gate to green.
2. **`moe_layer_list` is 3..90, and layer 91 is DENSE.** The dense prefix is
   layers 0..2 *and* a trailing dense-MLP full-attention layer 91; the neutral
   `ExpertStore` only knows a contiguous dense prefix, so the loader bounds the
   store at the last MoE layer + 1 (91) and the flat expert index stays
   `layer*experts+e`.
3. **Three expert layers are BF16** (`quantization_config.modules_to_not_convert`
   → layers 88..90, no `weight_scale_inv`). The converter accepts both an FP8
   stack + 128x128 F32 scale grid (GPU dequant) and a BF16 stack (verbatim).
4. **The expert stacks are FUSED** `[352,1536,4096]` / `[352,4096,1536]`; the
   converter splits them per expert into the neutral 9-column `ExpertSchema`.
5. **Partial RoPE span is per layer** and the table is HALVED (pair
   `(d, d+span/2)`, NEOX). The `rope_scaling` llama3 block (`factor 1.0`) is the
   identity and is not applied.
6. **Per-head attention gate runs BEFORE `o_proj`**, from the same normed
   hidden the QKV projections use.
7. **Final RMSNorm narrows the FP32 residual to BF16 first** — part of the
   reference arithmetic, implemented in its own kernel.
8. **MTP layers 92..94 and the 667-tensor vision tower are stripped+counted**
   (`deferred`: `skipped_mtp_tensors = 51`, `skipped_visual_tensors = 667`,
   `num_mtp_layers_stripped = 3`).

**Open item A — the DSA indexer runs only as the dense-equivalent case.** The
Lightning Indexer (`sparse_indexer_{q,k,z,w}`, region 8, proxy 256, top-512
regions) selects at most `topk*region = 4096` tokens per query. At or below that
bound every region is selected, so the dense causal path IS the reference path;
the engine therefore runs dense attention and refuses a request larger than
4096 tokens with an explicit error. The indexer weights ARE converted
(`indexer/layer_N.arrow`, 184 tensors, 834 MB) and the four open items are:
the FP8 region-summary compression, the radix top-k selection with its
zero-group tie-break, the region gather in the attention kernel, and a
region-summary cache in the session. This is the difference between the model's
1M-token claim and the 4096-token supported context (`kMaxContextTokens`).

**Open item B — parity goldens are declared debt.** The reference stack
(`inference/model.py` + `kernel.py`) is the vendor's multi-GPU FP8/Triton path;
the model card's deployment table starts at 4× B300 / 8× H200, so the 606 GB
source cannot be executed on one RTX 5090 and no FSI/CODE/DET golden can be
captured from it (`kGoldenDebt`: `step-5-preview:{fsi,code,det}`). Instruments
used instead: the kernel parity suite against host references (zero-centered
norm + BF16 narrowing, per-head QK norm, partial RoPE, router selection and
coefficients, clamped SwiGLU, windowed attention, FP32 combine) and the
real-weight SelfCheck/determinism/tool round-trip.

**Gates as measured (2026-10-31).** SelfCheck: **Paris x3, bit-exact
`logits_hash = 63b0f8b2fcf56d8a`**, 128 896 logits, kOff, 8.6 GiB pinned +
15.0 GiB expert cache; the kAuto determinism case (8 tokens, two loads) is
byte-identical. Real-weight native tool round-trip: the model emits
`<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>`
and the parser returns `{"city":"Paris"}` (80 generated tokens,
`</think>` then the call). Converter/loader/kernels/model/tool stages green
(`test_step_moe_{convert,loader,kernels,model,tool_framing}`).

**Non-defect found and fixed by the determinism gate.** The attention block
initially took the RoPE span from the frozen passport (192/64) while the
synthetic fixture's head is 32 dims, so the rope kernel wrote past the head into
neighbouring arena buffers and the model was not bit-reproducible. The span is
now clamped to the actual `head_dim` (`min(frozen_span, HD)`); production
geometry is unchanged (192/64 ≤ 192).

**Perf.** Seeded through the harness on the Release/AOCC tree at the canonical
2048-token protocol (`tests/perf/perf_baselines.json`, `step-5-preview`):
**1.82 t/s (perf) / 1.83 t/s (r3)**, TTFT ≈14.5 s, 127.63 W / 129.86 W average
(145.47 W / 147.72 W peak) on the DRAM probe. Both numeric classes land on the
same number because the path is host-store bound (~6.6 GB of expert payload per
token from the 307 GB tree; 8.8 GiB pinned + ~15 GiB expert cache), not compute
bound. The correctness-first shape is token-by-token prefill/decode and a router
readback per MoE layer (88 host round-trips per step), with no prefetch copy
stream. Next levers: the ids-streaming/no-handshake expert path and an expert
prefetch stream (engine work, not correctness).
