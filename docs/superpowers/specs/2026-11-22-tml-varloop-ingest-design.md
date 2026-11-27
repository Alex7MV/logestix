# TML-d24-varloop-research ingestion — design (2026-11-22)

Follows the Model Ingestion Protocol (skill `ingesting-a-model`), TDD per stage,
all work on branch `feat/inkling-2-varloop-ingest`. Temp/scratch space:
`/mnt/fast_storage/tmp/opencode/varloop` (fast disk). CUDA docs mirror:
`/home/alexmv2025/projects/cuda/docs` (operator-approved substitute for the
nvidia-cuda-docs MCP). Host: 160 threads, RTX 5090 32 GiB, 750 GB RAM.

## Intent

Adapt `TML-d24-varloop-research` (Thinking Machines Lab, released 2026-11-20,
at `/mnt/fast_storage/lsx_model/thinkingmachines/TML-d24-varloop-research`) into logestix,
text-only, through every lifecycle stage: convert → load → model → kernels →
SelfCheck (Paris, ≥3 bit-exact cycles) → FSI/CODE/DET parity gates → native
tool round-trip → seeded perf row at the canonical 2048-token protocol →
publishable model report.

## Stage 0 findings (triage, done 2026-11-22)

**Classification: SIBLING VARIANT of `inkling-2-mini`, not a new family.**
`architectures: ["Inkling2LoopForConditionalGeneration"]`,
`model_type: inkling2_loop` — the same architecture as Inkling-2-Mini, with
`text_config.loop_config.is_varloop: true`. The *only* config delta against
Inkling-2-Mini is that one boolean (plus no root-level `dtype` key):

```
DIFF text_config.loop_config.is_varloop   TML=True   Mini=False
DIFF dtype                                TML=None   Mini=bfloat16
```

Every other field, the tensor name set (1244 tensors, index total_size
19 525 870 752 — identical), the two tokenizer files, `chat_template.jinja`,
`tokenizer_config.json`, `processor_config.json` and `hf_quant_config.json` are
**byte-identical** to Inkling-2-Mini's. The weights are a different fine-tune
(shard sha256 `79e60b65…` vs Mini's `65e60f51…`), so the model answers
differently and needs its own goldens.

Per stage 0.1 this is a **sibling**: same architecture family, same geometry,
one new semantic (a runtime-chosen pass count). Therefore:

* own frozen passport (`inkling_moe_varloop_constants.h`) — model independence
  forbids a shared namespace with `inkling-2-mini`;
* the parent's loader / kernels / generation pipeline are REUSED, extended to
  carry a runtime pass count and cache slots sized for the maximum;
* no new kernels, no new expert codec, no new tokenizer, no new tool codec.

### What "varloop" means (verified, not assumed)

The model card: "the number of loops can be chosen at inference time … the
model was trained with 1 to 5 passes … post-trained with 3 passes". The
reference implementation in transformers 5.24 (`models/inkling2_loop`) does
**not** branch on `is_varloop` at all — it consumes `loop_num_passes` from
`loop_config.num_passes` (3 here) and appends `loop_num_passes - 1` deep copies
of the loop layers' cache layers. So `is_varloop` is a *capability declaration*,
not a code path: the same weights can be run with any pass count in `[1, 5]`,
and 3 is the checkpoint's own default.

The engine therefore takes: **default 3 passes (identical to the source's
default), with the pass count selectable at load time** in `[1, 5]`
(`LSX_INKLING_VARLOOP_PASSES`). Cache state for the extra passes is allocated
for the maximum (5) so changing the count never re-allocates mid-session:
`CacheLayers() = 24 + (5-1)*12 = 72` KV/conv slots, of which
`24 + (passes-1)*12` are touched. The slot formula `CacheSlot(layer, pass)` is
unchanged, so an active count below the maximum uses a prefix of the slab.

### Geometry — identical to Inkling-2-Mini (re-verified on this checkpoint)

| field | value |
|---|---|
| hidden_size | 2304 |
| num_hidden_layers | 24 physical; loop [6,18) |
| loop | core_start 6, core_end 18, gate_rank 64, **is_varloop true**, num_passes 3 (default), 1..5 supported |
| heads / kv / head_dim | 24 / 8 / 128 (GQA 3) |
| vocab / unpadded | 201024 / 200058 |
| experts / top-k / shared | 128 / 8 / 0; moe_intermediate 1152 |
| routing | sigmoid + selection-only FP32 bias; renorm then ×8.0 |
| softcaps | gate 4.0, up 25.0, final logit 30.0 |
| hybrid | full at {5,11,17,23}; sliding window 1024 elsewhere |
| rel bank | d_rel 16, extent 1024, no RoPE; qk L2 clip 22.627417, rel clip 8.0 |
| log scaling | n_floor **8192** (not Small's 128000), alpha 0.1 |
| AttnRes | block 6, full layers 3 mix queries, output_query final |
| expert codec | NVFP4 layers 0–19, MXFP8 layers 20–23 (same hf_quant_config) |
| eos / pad | 200006 / 199999; no BOS |

### Tokenizer verdict

tiktoken BPE, kind 2. `tokenizer.json`, `tiktoken/tokenizer.model`,
`tokenizer_config.json` and `chat_template.jinja` are byte-identical to
Inkling-2-Mini's (sha256 verified on all four), so the verified loop tool codec
(`inkling_moe_loop_tool_format`) is the model's native format and needs no
re-derivation. The stage-4b verdict points at the existing
`test_inkling_moe_loop_tool_framing` target — the same precedent as
gemma-4-31b-it (byte-identical template ⇒ parent's target proves it).

### Ballast (strip + count, never apply)

* vision tower: `model.visual.*` (same 422 tensors as Mini) — stripped, counted.
* `mtp.safetensors`: present, not in the index — never read.

### Feasibility (RTX 5090 32 GiB, 750 GB host)

Artifact size ≈ Inkling-2-Mini's 18.2 GiB (dense 3.0 + NVFP4 9.6 + MXFP8 5.0 +
vision 0.6 stripped). Device: dense plane 3.2 GiB + device expert slab 14.71 GiB
+ decode scratch + KV. KV/conv slots grow from 48 to 72: conv rings are
allocated eagerly at load (106 KiB/slot ⇒ 7.6 MiB), KV grows with the request
(pos-major, 2 MiB per slot per 1024 positions ⇒ 144 MiB at 1024, 288 MiB at
2048 for 72 slots). Both fit beside the 18 GiB working set on a 32 GiB card;
a card that cannot hold the expert slab falls back to host staging exactly as
the parent does.

### Semantics that must not be assumed (checked against the parent's verified set)

All of §0.6 of `docs/open-tasks/live/models/inkling-2-mini.md` carries over
unchanged, because the config and the tensor set are identical and the release
code is the same module. The one new question — what a pass count does — is
answered above from the reference source, and stage 8 pins it end to end
(the same prompt at 1/3/5 passes must produce the source's own outputs).

## Plan (TDD, one commit per stage)

| # | stage | red test | green code |
|---|---|---|---|
| 1 | naming | `test_code_naming_guard` case for `inkling_moe_varloop_*` | `varloop` variant token in `SchemeRegex()` |
| 2 | passport | `test_inkling_moe_varloop_constants` | `inkling_moe_varloop_constants.h` (geometry + pass bounds, own namespace) |
| 3 | converter | `test_inkling_moe_varloop_convert` (index carries varloop/passes_max/format tag; the non-varloop pipeline still rejects) | `InklingLoopConvertSpec` threaded through `inkling_moe_loop_convert` + `inkling_moe_varloop_convert.cpp` |
| 4 | loader | `test_inkling_moe_varloop_loader` | topology carries `varloop` + `loop_passes_max`; `CacheLayers()` = max slots, `LayerApplications()` = active |
| 5 | pipeline | `test_inkling_moe_varloop_pipeline.cu` (pass-count sweep, slot isolation, determinism) | runtime pass count in `Forward`/`RunInklingLoopGeneration` |
| 6 | model | `test_inkling_moe_varloop_model` | `InklingMoeVarloopModel` (kType, own frozen check, pass resolution, tool bridge) + factory registration via the manifest |
| 7 | gates | manifest row + selfcheck/FSI/CODE/DET targets | source-captured goldens via `tools/inkling_loop_oracle.py` |
| 8 | perf | `test_perf_baseline_tml_d24_varloop_research` | measured 2048-token row (+ a pass-count sweep in the report) |
| 9 | report | — | `docs/model-reports/2026-11-22-TML-d24-varloop-research.md`, MODELS.md, SELFCHECK_MATRIX.md, OPEN_TASKS |

## Result — pass-count sweep on real weights (measured 2026-11-22)

The knob is not decoration: the loop IS the work, so the count is a direct
throughput dial. 128-token decode, one GPU target at a time, Release/AOCC tree:

| passes | layer applications | decode rate | Paris SelfCheck | `logits_hash` |
|---|---|---|---|---|
| 1 | 24 | **100.8 t/s** | **fails** | `ccc087283987df2d` |
| 3 (default) | 48 | **48.2 t/s** | passes | `b50b3e494ffc7a9f` |
| 5 | 72 | **33.5 t/s** | passes | `f70488dc8822c290` |

Each count produces its own logits hash — expected, because a different number of
layer applications runs — and each is bit-stable across repeated runs (pinned on
the synthetic tree at 1/3/5 by `test_inkling_moe_varloop_pipeline`).

The 1-pass failure is the model card's own caveat made concrete: training sampled
1:2:6:2:1 but post-trained at 3, and at 1 pass this checkpoint does not reach the
"Paris" answer inside a 32-token framed window. That is a property of the WEIGHTS,
not of the engine — the same engine answers correctly at 3 and 5 passes — so the
manifest pins the default at the declared 3 and the report records the 1-pass
result as a limitation rather than hiding it.

Cost per layer application is ~24 ms at every count, which is what a shared-weight
loop should look like: the passes replay the same kernels over the same weights,
so cost scales linearly with the count and neither the slab nor the expert window
moves with it.

Canonical protocol (2048 tokens, `test_perf_baseline_tml_d24_varloop_research`,
cooled card): **32.33 t/s auto / 30.13 bit-exact**, TTFT 0.32/0.45 s, 18 291 MiB
resident — within 3 % of the fixed-pass sibling (34.1 / 32.1 t/s), i.e. the 5-pass
slab costs nothing at the default 3-pass count.
