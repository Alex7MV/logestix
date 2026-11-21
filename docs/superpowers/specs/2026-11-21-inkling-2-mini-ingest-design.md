# Inkling-2-Mini ingestion — design (2026-11-21)

Approved by the operator on 2026-11-21 ("ок"). Follows the Model Ingestion
Protocol (skill `ingesting-a-model`), TDD per stage, all work on branch
`feat/inkling-2-mini-ingest`. Temp/scratch space: `/mnt/fast_storage/tmp/inkling2-mini`
(fast disk). CUDA docs mirror: `/home/alexmv2025/projects/cuda/docs` (operator-approved
substitute for the nvidia-cuda-docs MCP).

## Intent

Adapt `thinkingmachines/Inkling-2-Mini` (downloaded and sha256-verified at
`/mnt/fast_storage/huggingface/Inkling-2-Mini`) into logestix, text-only, through
every lifecycle stage: convert → load → model → kernels → SelfCheck (Paris,
≥3 bit-exact cycles) → FSI/CODE/DET parity gates → native tool round-trip →
seeded perf row at the canonical 2048-token protocol → publishable model report.

## Stage 0 findings (triage, done 2026-11-21)

**Classification: NEW MODEL in the existing `inkling` family — not a sibling.**
`model_type: inkling2_loop` is a different architecture from `inkling2`
(Inkling-2-Small): a looped transformer. Different geometry AND new semantics,
so it gets its own frozen passport, class, converter, loader, kernels.

### Geometry (from config.json + safetensors headers + hf_quant_config.json)

| field | value |
|---|---|
| hidden_size | 2304 |
| num_hidden_layers | 24 physical; layers [6,18) run 3 passes → 48 applications/token |
| heads / kv heads / head_dim | 24 / 8 / 128 (GQA 3), same for SWA layers |
| vocab / unpadded | 201024 / 200058 (unembed padded; pad rows dropped) |
| moe_intermediate_size | 1152 |
| routed experts / top-k / shared | 128 / 8 / 0 |
| route_scale, scoring | 8.0, sigmoid + selection-only FP32 bias (`mlp.gate.bias`) |
| expert softcaps | gate 4.0, up 25.0 (tanh-capped SwiGLU) |
| final_logit_softcapping | 30.0 |
| hybrid pattern | full attention at (i+1)%6==0 → {5,11,17,23}; sliding window 1024 elsewhere |
| rel bank | d_rel 16, rel_extent 1024, NO RoPE; q/k L2-clip 22.627417 (learned per-head q scale), rel clip 8.0, v RMS-norm no gamma |
| log scaling | **n_floor 8192** (Small: 128000), alpha 0.1, full layers only |
| sconv | kernel 4; static on sliding layers; dynamic (static_w + weight_proj[32,2304]) on the 4 full layers; normalizer sqrt(k+1) static / sqrt(2k+1) dynamic |
| AttnRes | block size 6; boundary appends instead of local residual; full layers use 3 queries (q/k/v); `attn_queries.i` → layer i+1; `mlp_queries.i` → layer i; `output_query` final |
| loop | core_start 6, core_end 18, num_passes 3, gate_rank 64, is_varloop false |
| loop controllers | per loop layer × {attn,k,v,mlp}: down[64,4608], forget[2304,64], gain[2304,64]; MessageGate: `(1-tanh(forget(down)))*h + tanh(gain(down))*msg`, down input = concat(RMSnorm(h), RMSnorm(msg)) |
| cross-pass queries | `middle_cycle_cross_pass_queries` + `..._kv_queries` [24,2304] each → merged (12,4,2304) message_input.query (q,mlp / k,v split per conversion table) |
| loop cache | each pass has its OWN KV + conv state slots: 24 + 2×12 = 48 cache layers; passes 2/3 restart from pre-loop hidden/block states |
| expert codec | **NVFP4 layers 0–19** (U8 packed + E4M3 scales per 16, amax/2688 per expert), **MXFP8 layers 20–23** (F8_E4M3 + ue8m0 per 32). Only hf_quant_config.json + dtype census reveal the split |
| dtypes elsewhere | BF16 dense; FP32: AttnRes queries, router bias, cross-pass queries; sconv weights FP32 in reference (`_keep_in_fp32_modules_strict`) |
| eos / pad | 200006 `<\|content_model_end_sampling\|>` / 199999 `<\|endoftext\|>`; no BOS |

### Ballast (strip + count, never apply)

- vision tower: 422 tensors, 0.59 GiB
- `mtp.safetensors`: 54 tensors, 0.59 GiB, NOT in the index (3-layer MTP head)

### Tokenizer verdict

tiktoken BPE, kind 2 (same as Small). `tokenizer.json` and
`tiktoken/tokenizer.model` are **byte-identical to Inkling-2-Small** (sha256
verified). Chat template differs only in `{% generation %}` markers — rendered
text identical. Tool format: reuse the verified Small semantics in a new
family file `inkling_moe_loop_tool_format.cpp` (independence rule); framing
tests assert against Mini's own template.

### Feasibility (RTX 5090, 32 GiB)

Artifact 18.2 GiB: dense 3.0 + NVFP4 experts 9.6 + MXFP8 experts 5.0 + vision 0.6.
Whole model fits in VRAM (~20 GiB incl. KV at 2048 ctx): expert cache will
effectively pin everything → compute-bound, unlike Small's host streaming.
Full BF16 source (~53 GB dequantized) fits host RAM → **source-derived
FSI/CODE/DET goldens are possible** via transformers 5.24.0
(`models/inkling2_loop`, wheel extracted at
`/mnt/fast_storage/tmp/inkling2-mini/tf/x`), CPU/offload execution.

### Semantics to verify against the reference (not inferred)

1. Loop scheduling: pass 1 over layers 6–17 with growing block stack; passes
   2/3 restart from pre-loop states, receive per-layer messages retrieved by
   `message_input` attention over the PREVIOUS pass's block states + final
   hidden; per-pass cache_offset.
2. MessageGate arithmetic and where gates apply (after attention_input mix,
   before input layernorm; mlp gate after mlp_input mix).
3. Expert slab row-interleave (w13 gate/up alternating) — the Small defect;
   conversion table confirms `Interleave(dim=1)` for Mini too.
4. NVFP4 global scale = amax/2688 per expert; MXFP8 ue8m0 per 32.
5. Attention scale head_dim**-0.5, sink logits, gated output, L2 clip order
   (clip after sconv), log-scaling from 8192.
6. Cross-pass query merge order (q,mlp from one tensor; k,v from the other).

## Approaches considered

1. **New model files `inkling_moe_loop_*` modeled on `inkling_moe_*`** (chosen).
   Own frozen passport/class/converter/loader/kernels; `loop` token added to
   SchemeRegex in the same commit. Full protocol compliance; duplicates the
   inkling2 attention math but the loop machinery touches every layer call,
   KV layout and the session, so sharing would be invasive.
2. Extend InklingMoeModel with runtime loop config — rejected: frozen-geometry
   invariant forbids runtime config reads; guard would fail.
3. Sibling variant reusing Small's class — rejected: different geometry AND
   architecture; model independence forbids borrowed passports.

## Decisions

- Model key `inkling-2-mini`, family `inkling`, class `InklingMoeLoopModel`,
  namespace `lsxcommon::inkling::loop::frozen`, format tag
  `logestix-inkling-2-mini`, forge flag `--inkling_loop`, vault dir
  `Inkling-2-Mini`.
- Mixed-precision preserved byte-for-byte (NVFP4 0–19, MXFP8 20–23), same
  codecs/kernels approach as Small; dense BF16→FP16 device planes as in Small.
- Expert host store: neutral `ExpertStore` views + LRU (R1/R2); tree is small
  enough that the cache pins effectively everything — no private arena.
- Goldens: source-derived via HF 5.24 `inkling2_loop` on CPU (operator
  approved the slower capture). If a probe cannot be captured from source,
  record debt honestly — never rewrite goldens.
- KV layout: pos-major default; loop passes addressed as extra cache layers
  through `KvCacheGeometry`.
- Lifecycle contract in `inkling_moe_loop_model.cpp` +
  `inkling_moe_loop_inference_pipeline.cu`: residency lock, streaming Emit per
  token, abort_flag, teardown, governor lease, result fill.
- Perf: seeded through the harness (Release), 2048 tokens, both modes.

## Goals / non-goals / anti-goals

- Goals: all 9 stages green with real-weight gates; source-derived goldens;
  publishable report with measured numbers.
- Non-goals: vision tower execution, MTP/speculative decoding, contexts >2048
  validation, multi-GPU, varloop (is_varloop=false in this checkpoint).
- Anti-goals: requantizing MXFP8 layers to 4-bit (measured harmful on Small);
  hand-edited goldens/perf rows; runtime geometry; skipping a gate as "pass".

## Constraints

- Blackwell-only (sm_120), CUDA 13.5 toolkit / 13.4 driver minor-version compat.
- One GPU workload at a time (RESOURCE_LOCK gpu; check nvidia-smi before runs).
- spdlog-only logging; no absl containers in .cu-reachable headers;
  printf-style StrFormat; NVFP4 only via lsx_nvfp4; GEMM only via lsx_gemv.
- No subagents (AGENTS.md); builder-check of this doc done by the main agent.
- Converter tests: synthetic checkpoints only, `convert` label, TIMEOUT ≤600 s.

## Left to the builder

- Exact kernel tiling/occupancy choices within nvfp4-blackwell-conventions.
- Message-gate kernel fusion details (rank-64 GEMVs are tiny; fuse per layer).
- Session/cache growth strategy for 48 cache slots (reuse Small's session
  pattern, fix its known EnsureCapacity trap from day one).
- Golden capture prompt framing details per fsi/code/det skills.