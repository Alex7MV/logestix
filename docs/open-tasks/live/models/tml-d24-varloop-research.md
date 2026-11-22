# TML-d24-varloop-research ingestion — stage 0 triage, plan and trail

**Status: COMPLETE (2026-11-22).** Branch `feat/inkling-2-varloop-ingest`.
Design: `docs/superpowers/specs/2026-11-22-tml-varloop-ingest-design.md`.

Source checkpoint: `/mnt/fast_storage/huggingface/TML-d24-varloop-research`
(`thinkingmachines/tml-d24-varloop-research`, released 2026-11-20). Single
18.2 GiB shard (`79e60b65…`) + a 0.59 GiB `mtp.safetensors` the index does not
list. Text-only ingestion: the vision tower is stripped, images are out of
scope for this engine.

## 0.1 Classification

**SIBLING VARIANT of `inkling-2-mini`, not a new family.**
`architectures: ["Inkling2LoopForConditionalGeneration"]`, `model_type:
inkling2_loop`. The config diff against Inkling-2-Mini — computed field by field
on both trees — is exactly one boolean:

```
DIFF text_config.loop_config.is_varloop   TML=True   Mini=False
DIFF dtype                                TML=None   Mini=bfloat16   (root key)
```

The 1244-tensor index (total_size 19 525 870 752), `hf_quant_config.json`,
`tokenizer.json`, `tiktoken/tokenizer.model`, `tokenizer_config.json` and
`chat_template.jinja` are **byte-identical** to Inkling-2-Mini's (sha256
verified on all five). The weights are a different fine-tune
(`79e60b65…` vs Mini's `65e60f51…`), so the answers and the logits hash are this
model's own and it needs its own goldens.

Consequences of "sibling": own frozen passport (independence forbids sharing a
namespace), own format tag, own model class — and the parent's loader, kernels,
generation pipeline and tool codec are REUSED, extended only where the new
capability demands it.

## 0.2 The new semantic: what `is_varloop` actually means

Verified in the release's own code, not inferred from the name:

- transformers 5.24 `models/inkling2_loop` **never branches on `is_varloop`** —
  `grep -r varloop` over the module finds it only in the *config* (a stored
  attribute), never in the forward path.
- What the forward path does read is `loop_config.num_passes` (3 here): it
  appends `num_passes - 1` deep copies of the loop layers' cache state
  (`cache.layers += [copy.deepcopy(layer) for layer in loop_cache_layers * (n-1)]`)
  and indexes them with `cache_offset = num_hidden_layers + (loop_idx-1)*num_loop_layers - loop_start`.
- The model card states the range: trained with 1..5 passes sampled 1:2:6:2:1,
  post-trained at 3.

So `is_varloop` is a **capability declaration**: the same weights can run any
count in [1, 5], 3 is the checkpoint's own default, and the reference's own
`CacheSlot(layer, pass)` formula already supports every count. The engine's
addition is (a) a runtime selection, (b) a slab sized for the ceiling so a count
change never re-allocates, (c) the trained range enforced at load.

## 0.3 Geometry (re-verified on THIS checkpoint, not copied)

| field | value |
|---|---|
| hidden_size | 2304 |
| layers | 24 physical; loop [6,18) |
| loop | core_start 6, core_end 18, gate_rank 64, `is_varloop` **true**, num_passes 3 (default), range 1..5 |
| heads / kv / head_dim | 24 / 8 / 128 (GQA 3); SWA identical |
| vocab / unpadded | 201024 / 200058 |
| experts / top-k / shared | 128 / 8 / 0; moe_intermediate 1152 |
| routing | sigmoid + selection-only FP32 bias; renorm → ×8.0 |
| expert softcaps | gate 4.0, up 25.0; final logit softcap 30.0 |
| hybrid | full at {5,11,17,23}; sliding window 1024 elsewhere |
| rel bank | d_rel 16, extent 1024, NO RoPE; qk L2 clip 22.627417, rel clip 8.0, v RMS-norm without gamma |
| log scaling | n_floor **8192** (not Inkling-2-Small's 128000), alpha 0.1, full layers only |
| AttnRes | block 6; full layers 3 mix queries; `output_query` final |
| sconv | k=4; static on sliding, dynamic (`static_w` + `weight_proj`) on the 4 full layers |
| cache slab | 24 + (passes-1)×12 slots; the artifact records `loop_passes_max` = 5 → 72 |
| expert codec | NVFP4 layers 0–19 (U8 + E4M3 per 16, amax/2688), MXFP8 20–23 (E4M3 + ue8m0 per 32) |
| eos / pad | 200006 `<\|content_model_end_sampling\|>` / 199999 `<\|endoftext\|>`; no BOS |

## 0.4 Tokenizer

tiktoken BPE, kind 2. All four tokenizer files byte-identical to Inkling-2-Mini's
(`7f332727…` / `f170372d…` / `bf09342d…` / `53a9035f…`), so the verified loop
tool codec is this model's native format; stage 4b's verdict points at the
existing `test_inkling_moe_loop_tool_framing` (the gemma-4-31b-it precedent for
a byte-identical template), and the runtime glue is now shared rather than
copied (`inkling_moe_loop_tool_format`'s `EncodePromptIds` /
`EncodeConversationIds` / `ParseGeneratedCalls`).

## 0.5 Ballast (strip + count)

- `model.visual.*`: 422 tensors, 0.59 GiB — stripped, counted in the index.
- `mtp.safetensors`: 54 tensors, not in the index — never read.

## 0.6 Feasibility (RTX 5090 32 GiB, 750 GB host, 160 threads)

Artifact 18 GiB (dense 3.0 + NVFP4 9.6 + MXFP8 5.0; vision stripped).
Conversion-time VRAM profile: min 3.00 GiB / optimal 4.83 GiB / max 19.42 GiB at
context 2048 — the max is above the fixed-pass sibling's because the KV/conv slab
is sized for 5 passes (72 slots). Device at run time: dense plane 3.2 GiB +
device expert slab 14.71 GiB + scratch + KV; the 5-pass slab costs 288 MiB of KV
at 2048 tokens versus 192 MiB at 3 passes, which fits beside the working set.

The source reference (53 GB BF16 on CPU) fits the host, so FSI/CODE/DET goldens
are captured by RUNNING the release's own implementation, not derived from its
template.

## 0.7 Plan and trail

| # | stage | result |
|---|---|---|
| 1 | naming guard accepts the `varloop` VARIANT token | red → green (`test_code_naming_guard`, 14 cases) |
| 2 | frozen passport + constants test | red (missing header) → green, 262 assertions |
| 3 | converter spec (card name, format tag, own geometry, ceiling) + `--inkling_varloop` | red → green; `is_varloop` enforced in BOTH directions |
| 4 | loader/topology: `varloop` + `loop_passes_max`, `CacheLayers()` = slab, `LayerApplications()` = active; expected-format parameter | red → green |
| 5 | pipeline: trace sized by applications, slot isolation at 1/3/5 passes | red → green (`test_inkling_moe_varloop_pipeline`, 72 assertions) |
| 6 | model class + generated registry + tool bridge from the shared codec | red → green (5 cases, 103 assertions) |
| 7 | conversion of the real 18 GiB tree | **17 s**, 645 pinned tensors, 3072 experts, format `logestix-inkling-2-varloop`, `loop_passes_max` 5 |
| 8 | SelfCheck on real weights | **green first run**: Paris in 32 tokens, `logits_hash 98a2c3c532c9b013` bit-identical ×3 at passes=3; knowledge probe top-1 `152717` (" Paris"), perplexity 29.98 |
| 8b | FSI/CODE/DET source-captured goldens | **FSI 3/3, CODE 3/3, DET 2/3** — see the DET-02 finding below |
| 8c | perf row at the canonical 2048 protocol | **32.33 t/s auto / 30.13 bit-exact**, TTFT 0.32/0.45 s, 18 291 MiB resident (seeded through `test_perf_baseline_tml_d24_varloop_research`) |
| 9 | report + catalogue + matrix | report written; matrix row added |

## R21. DET-02: the DET gate's extractor resolves the prompt echo, on BOTH sides

**Measured, open, not an engine defect.** `test_inkling_moe_varloop_det_selfcheck`
is 2/3 on the real tree; the red cell is DET-TEST-02.

- the SOURCE (release code, CPU) answers `DET`: the capture warns
  `source answer differs from the prompt template: 'DET' != 'CATASTROPHIC'`;
- the ENGINE answers `DET` too, deterministically (`runs=3 deterministic=true`),
  so the two agree exactly;
- both reasoning streams echo the prompt identifier `DET-Validation-Core-02`
  before stating the correct word `CATASTROPHIC`, and the shared extractor
  (`tools/det_capture.py::_extract_token` on the capture side, the C++ extractor
  on the gate side) returns the first standalone uppercase token — the echo's
  first segment. The `-` that follows it is not a word character, so `DET` looks
  like a complete token.

The golden's `expected` is the prompt template's canonical answer, which neither
the source nor the engine produces under this extractor. Same class as
Granite-Switch 4.1 30B's DET-02 (`R10`) and the FSI extractor defect fixed during
the Inkling-2-Mini ingestion (`docs/model-reports/2026-11-21-Inkling-2-Mini.md`).

**Why it is not fixed here.** Teaching the extractor to reject a hyphen-delimited
identifier segment changes a gate shared by 40 models and would need every
committed DET golden re-validated — the DET prompts embed their own id in the
body, so the class is latent for every family whose model echoes the prompt. The
fix belongs with the other harness work (`K`/`M`), and the sibling precedent
(`R10`) is to leave the cell measured-red and record the mechanism.


## Pass-count sweep on real weights

The knob is a real dial, because the loop is the work. 128-token decode, one GPU
target at a time, Release/AOCC tree:

| passes | applications | decode rate | Paris SelfCheck | `logits_hash` |
|---|---|---|---|---|
| 1 | 24 | **100.8 t/s** | **fails** | `ccc087283987df2d` |
| 3 (default) | 48 | **48.2 t/s** | passes | `b50b3e494ffc7a9f` |
| 5 | 72 | **33.5 t/s** | passes | `f70488dc8822c290` |

Cost per layer application is ~24 ms at every count: the passes replay the same
kernels over the same weights, so the rate scales with the count and neither the
KV/conv slab nor the expert window moves with it.

The 1-pass failure is the model card's caveat made concrete — training sampled
1:2:6:2:1 but post-trained at 3, and at 1 pass this checkpoint does not reach the
answer inside the 32-token framed window. It is a property of the WEIGHTS, not of
the engine (the same engine is correct at 3 and 5), so the default stays the
checkpoint's declared 3 and the report states the limitation.

Engine-side SelfCheck detail (`test_inkling_moe_varloop_selfcheck`, real tree):

```
first ids: 200001 200008 371 150409 171960 149878 150744 151231 1355 150377 25 196980
prompt_len=20  eos=0  passes=3  32 tokens  logits_hash=98a2c3c532c9b013
```

The opener (`200001 200008 371 150409 171960`, i.e. `<|message_model|>
<|content_thinking|>The user asks`) is the same as the sibling's, and the
continuation diverges immediately — different fine-tune, same architecture.
