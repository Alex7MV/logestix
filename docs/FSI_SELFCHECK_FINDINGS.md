# FSISelfCheck — capture and parity findings

- **Date**: 2026-09-26
- **Scope**: pilot + rollout of FSISelfCheck
  (`docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md`)
- **Status**: mechanism complete; three families (Qwen3.6-35B-A3B,
  Qwen3.8-Flash-Next and Granite-4.2-30B) are wired into CI. The remaining
  families' tests SKIP until their goldens are captured and their divergences
  are resolved.

The FSI gate is **model-defined**: the golden is the original safetensors
model's own extracted answer (not the prompt template). The converted model is
required to reproduce that normalized answer at `FastMathMode::kOff`.

## Capture feasibility (original safetensors, HF transformers)

| Family | Result |
|---|---|
| xing4.0-29b-a4b | captured (all 3 exact, real CPU forward of the shipped `modeling_xing4_0.py`) — see the section at the end |
| Qwen3.6-35B-A3B | captured (all 3 exact) |
| Qwen3.8-Flash-Next | captured (all 3 exact; experimental `qwen4_exp` arch loaded with patches, see below) |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | captured (all 3 exact) |
| gemma-4-26B-A4B-it | captured (FSI-01/02 rephrased; template warnings) |
| gemma-4-31B-it | captured (FSI-02/03 exact; FSI-01 template divergence — the 26B class) |
| granite-4.2-30b | captured with the chat template + low-effort thinking (see below); all 3 exact |
| phi-4 | captured with the built-in ChatML template (`tokenizer_config.json`); all 3 exact |
| Qwen3.5-9B | captured (all 3; FSI-01/02 numeric substitution) — see the `qwen3_5` note below |
| ornith-1.5-35B-A3B | captured (all 3 exact) from the upstream **transformers** release — the retired MLX export was not capturable (its `language_model.*` keys MISS against the `model.language_model.` HF layout); see the ornith note below |
| DeepSeek-V4-Flash | recognized (`deepseek_v4`); FP8 loads after installing `accelerate` + `kernels==0.16.0`, but the FP8 linear kernel is CUDA-only and 149 G does not fit the 32 GB GPU (`device_map=auto` OOM) — needs a larger-VRAM host |
| GigaChat3.5-432B | **template-derived golden committed 2026-10-31**; converted **PASS 3/3** at a 1536-token budget (FSI-02 overruns 512). Source not runnable locally: FP8 (416 G) multi-GPU vendor stack + custom `gigachat3_5` code, BF16 body 864 G > host RAM |
| DeepSeek-V4.1-Flash | not capturable locally — `deepseek_v41` is not in transformers 5.15.1 and the release ships its **own inference stack** (`inference/model.py`, `convert.py`, `generate.py`; tilelang FP8/FP4 kernels, tensor-parallel, target 2×80 GB GPU), not HF remote code. Do not confuse it with the flat FP8 + FP4 MoE checkpoint in the vault. |
| GLM-5.3-Flash | not capturable — `glm5_next` unknown to transformers 5.15.1 (template-derived golden; **green 3/3 at a 512-token budget since 2026-10-23**, see the update below) |
| Inkling-2-Small | **template-derived golden 2026-11-15**, converted **PASS 3/3**. Source not runnable locally: 594B, own NVFP4+MXFP8 stack, 500 GB aggregated VRAM. First measured run was 2/3 and the failure was NOT the model: it reasons INLINE (no `<\|content_thinking\|>` terminator, no `</think>`), so its chain and its answer share ONE line, and `ExtractFsiFormula` applied `prefer_last_answer` to the line choice but not to the anchor position on that line — it returned a reasoning paragraph while the correct formula sat last in the stream. Fixed test-first; opt-in per golden case, so no other family's extraction moved. |
| Qwen3.5-35B-A3B | captured (all 3 exact) on **CPU**, no `OVERRIDES` entry — see below |
| GLM-5.2, Kimi-K2.7-Code | source absent from the vault |

### Ornith-1.5-35B-A3B capture recipe (leading system turn, thinking off)

The upstream **transformers** release is the same `qwen3_5_moe` VLM layout as
Qwen3.5-35B-A3B (`model.language_model.*`, `lm_head.weight`, fused
`mlp.experts.*`, `model.visual.*`), so the generic loader works. Unlike
Qwen3.5-35B-A3B it is safety-tuned: with only a user turn,
`enable_thinking=False` **refuses FSI-TEST-02** (reads the "deterministic
routing loop" prompt as money laundering) and answers FSI-TEST-01 with a
non-canonical variant. Ornith is normally served with a leading system turn
(its own model class injects one), and with that system turn + thinking off
the source answers all three prompts directly, in < 40 tokens each. This is
the `{"system": "You are a helpful assistant."}` override in
`tools/fsi_capture.py` (and the CODE/DET scripts). With thinking **on** the
model reasons for > 512 tokens and never closes `</think>` inside the budget,
so `enable_thinking=True` is not usable.

Converted-model parity passes 3/3 (2026-10-07); the goldens are committed.

### Qwen3.5-35B-A3B capture recipe (CPU, no overrides)

The checkpoint is a `qwen3_5_moe` VL MoE (40 layers, 256 experts, hybrid
linear/full attention, 67 GiB bf16) and transformers 5.15.1 loads it natively,
so the **generic** capture path works with no `OVERRIDES` entry:

```bash
python tools/fsi_capture.py --model /mnt/llm_models/models/Qwen3.5-35B-A3B \
    --type qwen3.5-35b-a3b --out tests/fixtures/fsi/qwen3.5-35b-a3b.json
```

The useful operational fact: `GENERIC` leaves `device` at `None`, i.e. the
capture runs on **CPU** and needs no VRAM. All three prompts (3 x 128 new
tokens) take ~2 minutes on a 160-core host with the weights resident in page
cache — far cheaper than sharding onto the GPU, and it leaves the GPU free for
the converted-model verification that follows. Same for `code_capture.py` /
`det_capture.py`; the three goldens for this model were captured in ~4 minutes
total.

All three answers are the source model's own (no template deviation):
`PV=CF*((1-exp(-r*n))/(1-exp(-r)))`,
`Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`,
`Drawdown>Threshold` — and the converted NVFP4 model reproduces all three at
`kOff` (62 s / 52 s / 94 s per gate).

### Qwen3.8-Flash-Next (`qwen4_exp`) capture recipe

The installed transformers 5.15.1 ships `Qwen4Exp*` classes, but the checkpoint
declares `model_type = qwen4_exp` and uses a layer type (`qwen_sparse_attention`)
that this build does not accept/route. Capture works after (documented in
`tools/fsi_capture.py`, override `loader = "qwen4_exp"`):

1. Extend `transformers.configuration_utils.ALLOWED_ATTN_LAYER_TYPES` with
   `qwen_sparse_attention`.
2. Register `qwen_sparse_attention` in `cache_utils.DYNAMIC_LAYER_TYPE_MAPPING`
   / `STATIC_LAYER_TYPE_MAPPING` (→ `Dynamic/StaticIndexedLayer`).
3. Build `Qwen4ExpForCausalLM(text_config)` in BF16 and load the checkpoint
   directly (source prefix `model.language_model.*` → `model.*`; skip the
   absent-from-text `model.visual.*` tower).
4. Fuse the PLE n-gram embedding: 128 `ngram_embedding.shard_N.weight` shards
   ([2500012, 160]) fill `ngram_embedding.weight` ([320001536, 160]) in order.

## Converted-model parity (source golden vs converted, `kOff`)

| Family | Result |
|---|---|
| Qwen3.5-35B-A3B | **PASS 3/3** (wired into CI; 2026-10-03) |
| Qwen3.6-35B-A3B | **PASS 3/3** (wired into CI) |
| Qwen3.8-Flash-Next | **PASS 3/3** (wired into CI) |
| Granite-4.2-30B | **PASS 3/3** (wired into CI) |
| phi-4 | **PASS 3/3** (wired into CI; 2026-10-01) |
| Qwen3.8-27B | **PASS 3/3** (wired into CI; 2026-10-01) |
| Qwen3.5-9B | FSI-02/FSI-03 **PASS**; FSI-01 diverges on a near-tie (golden **not** committed; SKIP; 2026-10-01) |
| gemma-4-26B-A4B-it | FSI-02 PASS after algebraic canonicalization; FSI-01 and FSI-03 diverge |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | FAIL 3/3 — converted model emits incoherent text |
### Granite-4.2-30B capture recipe

The capture initially failed because granite's chat template lives in a
standalone `chat_template.jinja` (absent from `tokenizer_config.json`), so the
generic path silently encoded the raw prompt and the model produced a malformed
FSI-02. The fix (in `tools/fsi_capture.py`): if `tokenizer.chat_template` is
unset, load `chat_template.jinja`; then frame with the template. Granite is a
reasoning model: its non-thinking mode still gets FSI-02 wrong, but
`enable_thinking=True, low_effort=True` answers all three exactly. Extraction
now takes the answer segment after the last `</think>` (reasoning CoT can
contain competing formulas), in both C++ and the Python mirror.

### phi-4 capture recipe

Straightforward: `microsoft/phi-4` (`Phi3ForCausalLM`) loads under
transformers 5.15.1, its chat template is embedded in `tokenizer_config.json`
(`<|im_start|>role<|im_sep|>content<|im_end|>`), and greedy decoding answers all
three FSI prompts. No `OVERRIDES` entry is needed. Golden:
`tests/fixtures/fsi/phi-4.json`.

### Qwen3.8-27B (`qwen3_5` VLM) capture recipe

The checkpoint is `Qwen3_5ForConditionalGeneration` (a VLM) with the text
weights under `model.language_model.*` and a 333-tensor `model.visual.*` tower
plus 15 `mtp.*` tensors. A plain `AutoModelForCausalLM.from_pretrained` is
silently wrong: `qwen3_5` maps to `Qwen3_5ForCausalLM`, whose weights are
prefixed `model.`, so every language weight is dropped and the model is random.
The `OVERRIDES` entry `"qwen3.8-27b": {"loader": "qwen3_5", "device_map":
"auto"}` (all three capture tools) builds a `Qwen3_5TextConfig` from
`config.text_config`, instantiates `Qwen3_5ForCausalLM`, and copies each shard
with `model.language_model.` -> `model.` (`model.visual.*` / `mtp.*` skipped).
27B bf16 (54 GiB) does not fit 32 GiB VRAM, so the model is sharded GPU+CPU with
`accelerate` (`_dispatch_auto`); capture takes ~7 min. The source answers all
three prompts exactly (FSI-01/02 with numeric substitution). Golden:
`tests/fixtures/fsi/qwen3.8-27b.json`; the converted NVFP4 model reproduces all
three at `kOff` (~17 s).

### Qwen3.5-9B (`qwen3_5` VLM) capture recipe + FSI-01 near-tie (2026-10-01)

Same checkpoint shape as the 27B (a `Qwen3_5ForConditionalGeneration` VLM:
`model.language_model.*` text, 333-tensor `model.visual.*` tower, 15 `mtp.*`).
The `OVERRIDES` entry `"qwen3.5-9b": {"loader": "qwen3_5", "device_map":
"auto"}` (all three capture tools) loads it text-only. 9B bf16 (~19 GiB) fits
32 GiB VRAM, so `_dispatch_auto` keeps every layer on GPU. The source answers
FSI-01/02 with numeric substitution and FSI-03 exactly; golden
`tests/fixtures/fsi/qwen3.5-9b.json` was captured but is **not committed**
(see below), so `test_qwen_dense_fsi_selfcheck` SKIPs.

The converted NVFP4 model reproduces FSI-02 and FSI-03 exactly, but **FSI-01
diverges**: source `PV=2500000*((1-exp(-0.0835*7))/(1-exp(-0.0835)))` vs
converted `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` (the prompt's own Expected
Template). This is a **near-tie**, not a runtime defect:

- At the decision point (`prompt` + `PV=`) the source's top-2 are `'2'`
  (logit 23.0) and `'CF'` (22.0) — a 1.0 gap (p≈0.73/0.27).
- The converted model's top-2 collapse to `'2'` 22.09 / `'CF'` 21.86 in
  `kAuto` (gap 0.23, chooses `'2'`, matching the source) and `'CF'` 22.47 /
  `'2'` 21.92 in `kOff` (gap 0.55, flips). The gate runs at `kOff`, so it
  flips.
- The converted distribution is globally close to the source (Pearson r
  ≈0.95 over the full 248320-vocab logits; 43/50 top-50 overlap; top-100 MAE
  0.61). Teacher-forcing the source's `PV=2500000` prefix shows the converted
  model tracks the source afterward.
- A reference audit of the qwen dense contour (RMSNorm offset form, partial
  RoPE / text-only M-RoPE reduction, gated full attention with the
  `qwen3_5` `attn_output_gate` split, GatedDeltaNet recurrence) found no
  discrete defect.

This is the same 4-bit precision-sensitivity class as gemma/Nemotron (a
single near-tie that greedy decoding cascades). Per policy the golden is not
committed and the gate stays SKIP; a "fix" would mean altering precision
semantics for a 0.2-logit margin with regression risk.

### gemma-4-26B-A4B-it

Recheck (2026-09-27) with the `chat_template.jinja` capture fix (gemma's chat
template is not in `tokenizer_config.json`, so the earlier capture used the raw
prompt): the *source* now answers FSI-02 canonically-equivalently and FSI-03
exactly, but the converted model still diverges.

- FSI-01: source `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)` (annuity-due, which
  the prompt actually asks for) vs converted
  `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` (ordinary annuity) — genuinely different
  expression; greedy decoding diverged.
- FSI-03: source `Drawdown>Threshold` vs converted
  `Drawdown>Thresholdthought` — the converted model appended a continuation
  token inside the identifier.

#### GLM-style check (2026-09-27): multi-valued EOS bug found + fixed

Fresh source capture (`tools/fsi_capture.py`, `chat_template.jinja` fix): FSI-01
annuity-due, FSI-02 canonical, FSI-03 exact. Converted model was 1/3.

**Root cause:** gemma-4's `eos_token_id` is a list (`[1, 106, 50]`); the
converter kept only the first element (`1`). A turn actually ends on **106**
(`<turn|>`) — confirmed by replaying the source, whose FSI-03 stream is
`Draw... down > Threshold <turn|>`. The decode loop stopped only on `1`, so it
ran past the turn end and emitted `thought`.

**Fix (GLM-5.3-Flash pattern):** `GemmaTopology.eos_ids` (`std::vector<int>`)
populated from the top-level config array plus the sibling
`generation_config.json`, written as `eos_ids` in `model.index.json`, read back
on load, and checked as a set in `RunGemmaGeneration`. Pinned by a converter
test asserting `eos_ids == {1, 106, 50}` on the synthetic fixture. The converted
model was re-generated so its index carries the full set.

**Result:** FSI-03 → **PASS** (`Drawdown>Threshold`); gemma **1/3 → 2/3**.

**Remaining FSI-01** is not a runtime defect: a teacher-forced top-1 agreement
check (source HF prefix fed back) is **21/22** — the converted model tracks the
source including the trailing `*exp(r)`, diverging on a single parenthesis
near-tie that greedy decoding then cascades. This is the same
quantization-sensitivity profile as Nemotron. gemma stays SKIP (2/3); the golden
is not committed.

### NVIDIA-Nemotron-3.5-Lightning-30B-A3B

Recheck (2026-09-27) with the same `chat_template.jinja` fix: the source now
answers all three exactly, but the converted model is unchanged — still FAIL 3/3.

- FSI-01 raw starts `The correct formula for the present value (PV)
  calculation is:` and the answer line does not match.
- FSI-02 raw is a long run of `0` digits (degenerate).
- FSI-03 raw is ``The requested output is: `0011` ``.

The existing "The capital of France is" -> "Paris" selfcheck passes on this
converted model, so the divergence only appears on the longer/complex FSI
prompts. This is a real converted-model finding, not a gate artifact.

#### Nemotron conversion / acceleration investigation (2026-09-26)

Checked whether the divergence is a bug introduced by the perf/"acceleration"
commits. Evidence (all negative for an acceleration bug):

| Check | Result |
|---|---|
| Routed-expert device LRU (`3fb1dbb`) | **Not the cause** — cache loads fully resident (`2944/2944` slots, only the prewarm misses) and the model still diverges. |
| Batched routed-expert GEMV (`1164799`) | **Not the cause** — at `kOff` it falls back to the pre-acceleration per-expert `RunLsxNvfp4Gemm`; `kAuto` and `kOff` fail identically. |
| Parallel router top-k (`242463f`) | Reviewed: same max/tie-break (lowest index) as the old serial scan; unit-tested (`test_nemotron_moe_kernels`). |
| FP16 relu2 / weighted-sum refactor | Numerically identical (relu^2 and the weighted sum are still computed in FP32; accumulation order unchanged). |
| Recurrent state leak across `Infer` | **No** — `ResetNemotronSession` zeroes mamba/conv/KV state at the start of every `Infer`. |
| KV/context overflow | **No** — session allocated for `kMaxSeq = 4096`. |
| Chat framing / special tokens | **Fine** — chat-framed "The capital of France is" answers correctly; high-id BPE tokens (` markdown` = 121468, ` Boundary` = 130036) match HF exactly. |
| Conversion/kernel/loader integrity | `test_nemotron_moe_{convert,kernels,loader,model,math}` all pass. |

Length sweep vs HF (greedy): exact match at 5 / 68 / 215 prompt tokens; at 425
tokens the converted model emits `"In"` where HF emits `"The"` (a near-tie
flip). On raw prompts the converted model matches HF token-for-token, including
the high-id tokens.

**Conclusion:** no evidence of an acceleration bug. The converted NVFP4 model
matches BF16 on short/medium and many long prompts, but greedy decoding
occasionally diverges when the state has accumulated quantization error, and on
some long/ambiguous FSI prompts that first flip cascades into a different
answer. This is a quantization-sensitivity limitation of the 4-bit conversion,
not the LRU/batched/router optimizations. A definitive proof would require a
higher-precision reference (BF16/FP16 resident) or per-layer hidden-state
parity against HF; the FSI gate correctly surfaces the divergence.

#### Nemotron divergence is quantization — confirmed (2026-09-27)

Ran the GLM-style check (audit the decode path against HF, then measure the
actual divergence) to decide whether a second runtime bug hides here.

- **Audit vs HF `modeling_nemotron_h.py`:** the full-attention path uses
  `total = pos + 1` (`nemotron_moe_inference_pipeline.cu`), i.e. real causal
  attention — the GLM DSA bug (current-token-only) does **not** exist here.
  GQA head mapping, full-dim RoPE (`rotate_half`), the sigmoid top-k router
  (`e_score_correction_bias` selects only, `norm_topk_prob`, `routed_scaling_factor`;
  group routing is a no-op with `n_group == 1`), the gated RMSNorm, the Mamba-2
  recurrence and the residual/shared-expert structure all match the reference.
- **Decisive measurement — teacher-forced top-1 agreement:** feed HF's exact
  greedy prefix back as the prompt and ask the converted model for the next
  token. On a raw 268-token passage it agrees with BF16 on **17/32** positions;
  every miss is a high-probability synonym (`clear`→`certain`,
  `essential`→`foundation`, `bridge`→`engine`, `machine`→`executable`) while
  function words and most content words match exactly. A structurally broken
  attention/Mamba would not track BF16 like this; a distribution that is close
  but not identical does.
- **Output sensitivity:** the FSI text changes chaotically under a tiny numeric
  perturbation (the `dt` clamp below), i.e. greedy is landing on near-ties.

**Verdict:** the Nemotron FSI divergence is 4-bit NVFP4 precision sensitivity on
reasoning prompts, not a runtime defect. No further digging is warranted absent
a BF16/FP16 converted reference.

##### Reference-fidelity fix found along the way: decode `dt` clamp

HF clamps the Mamba-2 `dt` to `time_step_min` (0.001) only inside the
chunk/prefill scan (`mamba2_chunk_scan`); its single-token decode path
(`mamba2_selective_state_update`) does **not** clamp. The runtime clamped both
(the checkpoint has `softplus(dt_bias)` heads down to ~1.1e-6, so this is a
×1000 difference where it triggers). Fixed by threading a `prefill` flag:
`clamp_dt = 1` while consuming the prompt, `0` for greedy decode
(`nemotron_moe_kernels.{h,cu}`, `nemotron_moe_device_math.h`,
`nemotron_moe_inference{,_pipeline}.cu`, `nemotron_moe_model.cpp`), pinned by
`tests/test_nemotron_moe_{math,kernels}`. The FSI outcome is unchanged — the fix
is reference fidelity, not the divergence cause.

## DeepSeek-V4.1-Flash: API-derived golden experiment (2026-09-26)

The source cannot be run locally (own FP8/FP4 tilelang tensor-parallel stack).
As a stopgap the assistant (itself DeepSeek-V4.1-Flash) answered the three FSI
prompts and those answers were used as the golden, with `input_ids` produced by
the release's official `encoding/encoding.py` (`thinking_mode="chat"`) plus
`tokenizer.json`.

Result on the converted model (`kOff`): FSI-03 matched `Drawdown>Threshold`;
FSI-01 was one `)` short of the golden; FSI-02 ignored "use the provided
variable names" and substituted the numeric legs, additionally dropping zeros
(`100000` instead of `100000000`). Using the runtime's own framing (raw BPE via
`EncodeRawPrompt`, the default `BuildPromptInputIds`) was worse (garbage/empty),
so the divergence is model/runtime-level, not framing.

A framing sweep (one load, four encodings: chat / thinking / raw BPE / chat +
system) confirmed it: no framing matched. FSI-01 is consistently one `)` short
(chat and chat+system), FSI-02 consistently evaluates the legs numerically with
the same lost zeros, thinking mode emits a long CoT that does not finish in the
budget, and raw BPE degrades to garbage/empty.

An exhaustive follow-up sweep (framings × `ignore_eos` × prompt variants, plus a
1k-token thinking budget and a digit-echo probe) found no configuration that
reproduces the API-derived golden:

- FSI-01: with `ignore_eos` the model continues and explicitly calls the prompt's
  template unbalanced, then emits its own "corrected" formula.
- FSI-02: even a strict "do not evaluate, echo the variable names" prompt still
  yields `100000` instead of `100000000`.
- Thinking mode never terminates its CoT within 1024 tokens.
- Digit-echo probe (`100000000`, `123456789`, `0.00000001`, ...) reproduces the
  numbers correctly in isolation, so the FSI-02 digit loss is model behavior,
  not a runtime decode bug.

Conclusion: DeepSeek-V4.1 cannot be certified against the API-derived golden on
this runtime; it stays a finding (test SKIPs). A real reference needs the
model's own FP8/FP4 stack on the intended hardware.

The golden was not committed, so `test_deepseek_moe_v41_fsi_selfcheck` SKIPs.
DeepSeek-V4-Flash is the same situation (its vault copy is a flat FP8+FP4
pre-quantized layout consumed by the repo's own loader, not an HF checkpoint).

### GLM-style recheck (2026-09-27)

Redid the comparison the way GLM-5.3-Flash was handled.

- **Fresh API capture:** the assistant (deepseek-v4.1-flash) answered the three
  prompts; the answers equal the Expected Templates
  (`PV=CF*((1-exp(-r*n))/(1-exp(-r)))`, the `Balance_Alpha` retrace,
  `Drawdown>Threshold`). The golden is therefore the template and is **valid** —
  the earlier "no framing matched" was not a bad golden.
- **Runtime audit vs the release's own `inference/model.py`:** the sliding-window
  ring (`get_window_topk_idxs`), `Compressor`, `Indexer`, the two-level
  `select_candidate_blocks`, the sparse attention + inverse RoPE, the Gate
  (`sqrtsoftplus`, bias selects only, `norm_topk_prob`, `route_scale`), the
  Hyper-Connections mix/pre/post order and the YaRN schedule all match. No
  discrete defect of the GLM (`total = 1`) or Nemotron (`dt` clamp) kind was
  found.
- **Converted model (`kOff`), official `encoding/encoding.py` framing:**

  | Case | chat | thinking |
  |---|---|---|
  | FSI-01 | `PV=CF*((1-exp(-r*n))/(1-exp(-r))` (one `)` short) — FAIL | CoT does not finish — FAIL |
  | FSI-02 | numeric substitution, zeros lost (`100000`) — FAIL | CoT — FAIL |
  | FSI-03 | `Drawdown>Threshold` — **PASS** | **PASS** |
  | | **1/3** | **1/3** |

  GLM went 0/3 → 3/3 after its fix; DeepSeek-V4.1 is 1/3 in both modes, so it is
  not wireable as passing — it stays SKIP (Gemma/Nemotron class).

**Interpretation:** since the fresh source capture matches the templates, the
divergence is real on the converted side. With no discrete defect in the audited
path, it is the precision level of the requantization (native FP8 + FP4 experts →
neutral NVFP4), not a rewrite bug; the digit-echo probe and the FSI-03 pass show
the decode is functional. A definitive reference still requires running the
release's own `inference/` stack on the intended 2x80GB hardware.

### DeepSeek-V4.1-Flash: model-defined golden committed (2026-10-07)

The source ships its own FP8/FP4 tilelang tensor-parallel stack and cannot run
locally, so the reference is the model itself: as with GLM-5.3-Flash, the
DeepSeek-V4.1-Flash assistant answered the three prompts and its answers — the
`Expected Templates` — are the golden. `input_ids` come from the release's
`encoding/encoding.py` (`thinking_mode="thinking"`) + `tokenizer.json`, the same
framing as the committed DET golden. The golden is now committed
(`tests/fixtures/fsi/deepseek-v4.1-flash.json`); previously the test SKIPped.

Result on the converted model (E4M3FN NVFP4, `kOff`): **1/3**.

| Case | golden (= the model) | converted `extracted` |
|---|---|---|
| FSI-01 | `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` | `PV=CF*(1-e` (truncated prompt echo) |
| FSI-02 | `Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C` | `Balance_Alpha=100000000` (initial-state echo) |
| FSI-03 | `Drawdown>Threshold` | `Drawdown>Threshold` **PASS** |

Same profile as the 2026-09-27 sweep (no framing reproduced the golden; FSI-01
consistently short, FSI-02 numerically evaluates), and the E4M3FN codec change
did not move it. In thinking mode the CoT does not terminate within the 512-token
budget, so the first-anchor-line extractor latches onto the prompt echo. No
discrete rewrite defect was found in the audited runtime; this is the
requantization precision plus the reasoning model's CoT length against the
extractor, not a kernel bug. Tracked in `OPEN_TASKS.md` section Q. A definitive
reference still needs the release's own `inference/` stack on 2×80 GB.

## GLM-5.3-Flash: API-derived golden experiment (2026-09-27)

GLM-5.3-Flash cannot be run locally either (`glm5_next` is absent from
transformers 5.15.1 and the checkpoint is FP8). As a stopgap the assistant
(switched to GLM-5.3-Flash) answered the three FSI prompts and those answers
were used as the golden; `input_ids` came from the model's `chat_template.jinja`
+ `tokenizer.json` (the template always opens `<|assistant|><think>`).

Result on the converted model (`kOff`): **fails all three, degenerating under
every framing tested** (open `<think>` / immediately closed `<think></think>` /
raw BPE):

- FSI-01: long off-task reasoning trace (wrong annuity formula).
- FSI-02: pathological repetition of `</think>` (4097 chars).
- FSI-03: reasoning trace; the raw framing repeats `True` / `Expected: True`.

The short `"The capital of France is" -> "Paris"` selfcheck still passes
(168 s), so the converted model was fine on short prompts and degenerated only
on the longer FSI inputs.

### Root cause + fix (2026-09-27): DSA layers attended only the current token

This was **not** a quantization limit and not a shared length bug — it was a
real runtime defect in the GLM-5.3-Flash DSA decode path. `RunFlashDsaDecode`
(`glm_moe_flash_inference.cu`) called the MLA attention kernel with the K/V
pointer already offset to the current slot *and* `total = 1`, so the 11 DSA
layers attended only to the current token, discarding all causal history (the
code comment promised "causal over slots 0..slot"; the working GLM-5.2 MLA path
uses `pos + 1`). The KDA layers alone carried context, which is enough for
local recall (Paris) but not for reasoning/retrieval tasks (FSI).

Fix: pass the cache base and `total = slot + 1` (full causal over `0..slot`);
the kernel's score scratch is now sized by `cap` instead of a hardcoded 4096
(the old `total = 1` never reached that bound). With the fix, the converted
model answers FSI-02/FSI-03 exactly and FSI-01 algebraically-equal to the
golden; `test_glm_moe_flash_selfcheck` (Paris) and
`test_glm_moe_flash_{convert,kernels,loader,model}` stay green. Real causal
attention costs throughput, so the reference was re-baselined
(`tests/perf/perf_baselines.json`: perf 33.93 → 14.88 t/s, r3 18.48 → 12.50 t/s;
the old numbers were measured with the broken no-attention path) — details and
the rejected warp-per-head optimization in the GLM ingestion plan, Session 6.

Because the source cannot be run locally, the committed golden
(`tests/fixtures/fsi/glm-5.3-flash.json`) is **template-derived** — `input_ids`
come from the official `chat_template.jinja` + `tokenizer.json`, and `expected`
is the prompt's `Expected Template` formula (recorded in the `source` field).
This is a deliberate deviation from the "golden = source model's own answer"
rule, forced by the unrunnable source; it still guards the converted DSA path
(it was 0/3 before the fix, 3/3 after). `test_glm_moe_flash_fsi_selfcheck` now
runs green (~283 s) instead of SKIPping.

> **Re-conversion regression 2026-10-07.** Re-converting GLM-5.3-Flash after
> the full-E4M3FN (subnormal) codec change makes **FSI-TEST-02 RED**: the model
> deliberates (`Let me analyze …`) and never emits a clean answer line, so the
> extractor picks the reasoning's initial-state echo `Balance_Alpha=100000000`
> instead of the required `Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`.
> It is **not** a token budget (tested at 128, 512 and 2048 — at 512 it drifts
> into numerically evaluating the expression and at 2048 it is still
> deliberating). FSI-01/03 and Paris/CODE/DET stay green. A behavioural
> divergence from the offline requantization; tracked in `OPEN_TASKS.md`
> section P. The template-derived golden is left unchanged.

### Update 2026-10-23: §P retracted — it was the 128-token budget

A fresh conversion of the same source (the vault had no `logestix/` tree at
all) reproduced the 128-token behaviour exactly — and showed that the budget
was the whole story:

| case | 128 tokens (committed golden) | 512 tokens |
|---|---|---|
| FSI-01 | pass | pass |
| FSI-02 | `Balance_Alpha=100000000` (prompt echo) | `Balance_Alpha=((100000000*Rate_A-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C` (pass) |
| FSI-03 | `` — deliberation cut mid-sentence | `Drawdown>Threshold` (pass) |

The 2026-10-07 note above ("not a token budget — reproduces at 512 and 2048")
does **not** reproduce: at 512 every prompt emits its canonical line. The
reason CODE and DET were green all along is that both already used
`max_new_tokens = 512`; FSI's golden was the only one at 128. The golden's
budget is now 512 for all three prompts, the expectations are unchanged, and
`test_glm_moe_flash_fsi_selfcheck` passes 3/3 on real weights.

The lesson worth keeping: for a checkpoint that reasons at Max effort, a
token budget is part of the *harness contract*, not a free parameter — a
reasoning model cut mid-trace produces a plausible anchor line (the echo) that
looks exactly like a semantic divergence.

## Normalization contract implemented here

- Preferred answer line: the first line where the anchor is immediately
  followed (spaces allowed) by `=` (arithmetic) or a comparison operator;
  fall back to the first line containing the anchor.
- Arithmetic answers are canonicalized by parsing the RHS (numbers,
  identifiers, `exp(...)` calls, `+ - * /`, parentheses, unary minus), then
  left-associating equal-precedence chains and re-printing with minimal
  parentheses. Redundant parentheses therefore compare equal, and forms that
  differ only by associativity also compare equal
  (`CF*((1-exp(-r*n))/(1-exp(-r)))` == `CF*(1-exp(-r*n))/(1-exp(-r))`,
  `a*(b/c)` == `(a*b)/c`, `a-(b-c)` == `a-b+c`). Each rewrite is an exact
  algebraic identity, so genuinely different expressions are never equated;
  commutativity is deliberately not applied. Comparison answers are
  whitespace-normalized and bounded to the anchor, operator and one identifier.

As-built delta vs the spec: the golden `expected` is model-defined (spec
§3.2 updated); arithmetic canonicalization and operator-aware line selection
were added after observing real model outputs.

## Next steps

0. FP8 capture requires `accelerate` and `kernels` (`kernels==0.16.0`) installed
   and a host whose VRAM can hold the FP8 layers; pass `--device-map auto` on
   such a host. This session installed `accelerate 1.15.0` and `kernels 0.16.0`.
1. Wire each remaining family into CI by capturing its golden once the source
   is available/framework-loadable, and resolving any genuine divergence.
2. Nemotron: investigate the converted model on the FSI prompts (long-input
   path) independently of FSI.
3. Gemma FSI-01/FSI-03: determine whether the divergence is numeric or
   template-related.

## alice-foundation-80b-a3b (2026-10-01)

**Capture: template-derived (source not runnable locally).** The 80B source is
152 GB BF16 and the shipped `modeling_alice_ai.py` expert loop is O(512)
Python per layer; a CPU forward is ~1–2 min/token, so a real 128–2048-token
capture is hours. `transformers` 5.15.1 also mis-loads the SentencePiece
`tokenizer.model` (needs `protobuf` + `sentencepiece`). Golden `expected` is
taken from `tools/fsi_prompts.json`; `input_ids` are framed with the checkpoint's
`chat_template.jinja`.

**Extraction:** AliceAI's CoT close is `[COT_END]` (not `</think>`); added it as
a generic reasoning-close marker in `ExtractFsiFormula` so the answer after CoT
is selected instead of a reasoning echo.

**Result:** FSI-TEST-01/02/03 PASS at `kOff` with `max_new_tokens=2048` (the
model reasons at length before the answer line). The model reproduces the
canonical formulas/comparison.

## xing4.0-29b-a4b (2026-10-04)

**Capture: real.** The checkpoint ships `modeling_xing4_0.py` and fits CPU
(58 GB BF16), so `tools/{fsi,code,det}_capture.py` ran the unmodified upstream
model on `transformers` 5.15.1 / `torch` 2.13 (bf16, eager attention) and all
three stages captured exactly — FSI 3/3, CODE 3/3, DET 3/3
(`tests/fixtures/{fsi,code,det}/xing4.0-29b-a4b.json`).

**Result: all three gates PASS at `kOff`** (`test_xing_moe_{fsi,code,det}_selfcheck`).

Six defects had to be fixed before parity; each was invisible to the synthetic
fixtures and to SelfCheck's "is Paris the answer" check, so they are recorded
here as the reusable lesson rather than as a Xing-only note:

| Defect | Why the existing tests missed it |
|---|---|
| RoPE launcher took a **caller-supplied scalar** as the head offset, so all 32 heads rotated the same 64 floats | a single-token prefill never reads the cache back — the softmax is trivial |
| **One KV buffer for all 40 layers** (upstream keys `past_key_values.update(..., layer_idx)`) | same single-token blindness; a 2-token prompt already diverges |
| MoE used the **gate weights for the SwiGLU up-branch** and never ran the down projection (the expert view table had 2 of 3 matrices) | `silu(gate)*gate` is a well-formed MLP: finite, plausible, wrong |
| The shared `k_rot` slice was written by the `h == 0` block only | heads 1..31 read **uninitialized** rope dims; surfaced as first-step non-determinism, found with `compute-sanitizer --tool initcheck` |
| Routed-expert accumulation was a read-modify-write of one `acc[i]` from all `top_k` blocks | race, not an overflow — only `racecheck`/multi-step comparison sees it |
| Blocking `cudaMemcpy` for the sampled token **and** for the logits ran on the legacy default stream, which does not synchronize with the context stream (`cudaStreamNonBlocking`) | identical requests sampled different sequences; the numbers looked like tie-breaking noise |

**YaRN:** upstream's `Xing4_0RotaryEmbedding` receives `attention_factor = 1.0`
for this checkpoint (`rope_scaling.mscale_all_dim = 1.0` does not reach it), so
cos/sin satisfy `cos^2 + sin^2 == 1`; `mscale = 0.1*ln(64)+1 = 1.4158883` appears
**only** in the softmax scale, where the attention module squares it
(`kAttnScale = 1.4158883^2 / sqrt(192)`). Scaling the tables as well doubles
every attention score — the model still answers "Paris", but FSI-01/02 truncate.

**Tool-call markers:** `tokenizer_config.json` advertises `<\u200Btool_call>`
(zero-width space) at ids 11/12, but the SentencePiece piece stored at those ids
— and what `tokenizers` actually matches — is the **plain** spelling: encoding
the ZWSP form yields six ordinary pieces (`[612, 127643, 9621, 34, 4039, 124430]`),
not `[11]`. The codec therefore looks up and renders the plain form; re-encoding
upstream's own template output reproduces the marker ids. A test asserts both
facts against the exported vocabulary.

## gemma-4-26b-a4b-it — captured from the source safetensors, and one real divergence

`Gemma4ForConditionalGeneration` / bf16 / `Gemma4Config` in transformers 5.15.1, so the
source **runs locally** and the golden is model-defined, not template-derived. Capture
needed `--device-map auto`: 26B bf16 is ~52 GB against a 32 GB RTX 5090, so the experts
stay offloaded to CPU and generation is PCIe-bound (minutes per case). `chat_template.jinja`
was required and loaded — without it the raw body is encoded and the answers come back
malformed.

The source model does **not** answer with the prompt's template formulas, which is the
point of the gate:

| case | template | source (golden) |
|---|---|---|
| FSI-TEST-01 | `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` | `PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)` |
| FSI-TEST-02 | `(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C` | same, canonicalized |
| FSI-TEST-03 | `Drawdown>Threshold` | same |

FSI-TEST-01 is the annuity-**due** case: the trailing `*exp(r)` is the correct answer and
the template's line is missing it.

**The converted model fails FSI-TEST-01, and it fails on the template's formula:**

```
source  raw: PV=CF*(1-exp(-r*n))/(1-exp(-r))*exp(r)
converted raw: PV=CF*((1-exp(-r*n))/(1-exp(-r)))
```

This is a real behavioural divergence, not a harness artifact — the extracted answer is
clean and balanced, so `ExtractFsiFormula` worked; the converted model simply picked the
plainer formula. FSI-TEST-02 and FSI-TEST-03 pass.

**Not caused by the attention score-dot port.** Re-ran both gates at `7810383` with
`libs/lsxcommon/src/gemma_moe_inference.cu` reverted to its pre-port state: identical
results, 5 passed / 1 failed each. So it is an NVFP4-vs-bf16 quantization difference on a
borderline greedy decision, exactly the class FSI exists to surface, and it predates the
perf work. Open, not masked — the gate stays red.

**Co-existing CODE divergence, different in kind.** `tests/fixtures/code/gemma-4-26b-a4b-it.json`
captured the same way. CODE-TEST-01 and CODE-TEST-03 match; CODE-TEST-02 fails with
`m=(l+(h-l)/2)` against the source's `m=l+((h-l)/2)`. The two are algebraically identical
and CODE applies **no** canonicalization by design, so this is the intended strictness
firing rather than a defect. Recorded in `docs/CODE_SELFCHECK_FINDINGS.md`.

## gemma-4-31b-it (2026-10-09) — the dense member of the Gemma-4 family

Captured from the source safetensors with `Gemma4ForConditionalGeneration` /
bf16 / `Gemma4Config` in transformers 5.15.1, `--device-map auto` (62 GB bf16
against a 32 GB RTX 5090, so the source offloads to CPU and is PCIe-bound).
`chat_template.jinja` was required and loaded; the 31B ships a tokenizer.json
and chat template BYTE-IDENTICAL to the 26B's.

| case | source (golden) | converted | verdict |
|---|---|---|---|
| FSI-TEST-01 | `PV=2500000*((1-exp(-0.0835*7))/(1-exp(-0.0835)))` | `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` | **fail** (template form) |
| FSI-TEST-02 | `Balance_Alpha=(((100000000*Rate_A*(1-Slip_A))*Rate_B*(1-Slip_B))*Rate_C*(1-Slip_C))` | same | pass (canonicalized) |
| FSI-TEST-03 | `Drawdown>Threshold` | same | pass |

FSI-TEST-01 is the same divergence class as the 26B: the source substitutes the
prompt's numbers (the annuity-due form) while the converted model repeats the
template formula. It is an NVFP4-vs-bf16 quantization difference on a borderline
greedy decision, not a harness artifact — `ExtractFsiFormula` returned a clean,
balanced expression. The gate stays red, consistent with the 26B.

The converted model's CODE gate is fully green (3/3) and the DET gate is green
(3/3, `deterministic=true` on every case) — see the CODE/DET findings docs.

## Muse-Glimmer-30B (2026-11-19) — captured at `reasoning_strength: low`, one prompt at a time

The ingestion (2026-11-19, `docs/model-reports/2026-11-19-Muse-Glimmer-30B.md`)
landed every gate except FSI, which was declared debt after two failed captures.
Re-reading the failed run's own raw shows **two** defects, and the model was only
responsible for one:

1. **The extraction read the reasoning tail.** The retained FSI-03 raw from the
   512-token `high` attempt contains `assistant to=userDrawdown>Threshold`
   repeatedly — the model answered — yet the run recorded `empty extraction`,
   because the echoed prompt's `Variables: Drawdown = 14.999991; Threshold =
   15.000000.` gave the extractor a `Drawdown`-plus-`=` line first and the
   comparison pattern then found no identifier. Same defect as the CODE/DET
   gates, fixed by the ATEM first-complete-channel rule.
2. **The framing decided the answer's form.** The checkpoint's
   `chat_template.jinja` renders `Reasoning strength: <value>.` into the system
   block, defaulting to `high` (`reasoning_strength` is the template's own
   kwarg). At `high` the answer channel is not consistently the canonical
   formula: the 512-token attempt's extraction recorded `Balance_Alpha=100000000`
   (the prompt's numeric echo) for FSI-02 — that raw was not retained, so the
   channel-vs-echo origin cannot be separated, which is the argument for reading
   a named channel rather than the tail.

The 1024-token retry could not have fixed either defect, and the loop is **not**
what `reasoning_strength: low` removes: the captured `low` raws still carry
12 / 7 / 37 `assistant to=user` channels (and 0 / 5 / 0 `assistant to=self`) in
512 tokens. What `low` does is make the answer channel the canonical formula for
all three prompts; what the first-complete rule does is read it. Capture ran one
prompt per run through the CPU text tower (`tools/muse_golden_oracle.py`, 627
text tensors, 809 vision skipped), 512 tokens, `~27 min/case` (400-470 CPU-min),
with no "model answer differs from the template" warning on any case.

| case | source golden | converted (kOff, 512) | verdict |
|---|---|---|---|
| FSI-TEST-01 | `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` | `PV=CF*(1-exp(-r*n))/(1-exp(-r))` | pass (canonicalized) |
| FSI-TEST-02 | `Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C` | same (canonicalized) | pass |
| FSI-TEST-03 | `Drawdown>Threshold` | same | pass |

`test_muse_dense_fsi_selfcheck`: **3/3**, 11 s/case.

**The harness change this needed.** Muse writes its answer in an ATEM channel
(`<|start|>assistant to=user<|message|>…<|eot|>`); the markers are special
tokens, so the decoder leaves the headers as literal text, and the model
re-emits the answer after looping, so the LAST answer channel can be truncated
by the budget (measured on its CODE/DET goldens: `PARADOX_ANTINOM`). The golden
therefore carries `answer_channel`/`reasoning_channel`, and the harness reads
the FIRST COMPLETE channel — one implementation,
`atem::SelectAnswerChannel` in `libs/lsxcommon/include/lsxcommon/atem_channel.h`,
shared by FSI/CODE/DET, with the capture-side twin in
`tools/muse_golden_oracle.py::select_answer_channel` (both pinned by tests).

**Capture tool change** (the reason a 3-case run is now safe to kill):
`tools/fsi_capture.py` saves each completed case atomically, `--only` captures a
single id and `--resume` merges into the file an earlier run wrote. A failed
case dumps its raw beside the golden and the process exits 3 with the ids to
retry, keeping the partial file.

**Two artifacts recorded, not masked:** the captured `input_ids` begin with a
doubled `<|begin_of_text|>` (the template already emits `bos_token` and the tool
also asks for special tokens; the CODE/DET goldens have it too, and the verifier
replays the exact ids so parity is unaffected), and no caller sets
`MuseRenderPolicy::reasoning_strength`, so the engine serves `high` while this
golden verifies `low`.

## Inkling-2-Mini (2026-11-21) — the prompt-echo defect, fixed in the core instead of worked around

The second source-run capture in the family (the sibling's goldens are
template-derived), and the first to reach the extractor with a stream whose
answer sits mid-reasoning. FSI-TEST-03 failed on the FIRST engine run with
`extracted=''` while the raw plainly contained `Drawdown>Threshold`.

The cause is the defect recorded above in the Muse section, reached from a
different direction. The stream restates the prompt before answering, so the
chosen line is

```
Variables: Drawdown = 14.999991; Threshold = 15.000000.
```

`Drawdown` there is followed by `=`, an operator, so `SelectAnswerLine` returned
that line as a comparison candidate — but the bound then requires
`anchor <op>+ identifier` and the character after `=` is a digit, so the whole
extraction returned empty. Muse had the same input shape and was rescued by the
ATEM first-complete-channel rule, which is a Muse-shaped fix for a general
defect: a checkpoint without channels had nothing to rescue it.

So the CORE extractor was fixed, red-first
(`tests/test_fsi_selfcheck.cpp`): a comparison candidate now has to survive the
bound, both when the LINE is chosen and when the position inside it is resolved.
This cannot regress a passing case — a line holding a bounded candidate also
contains `anchor` + operator, so it was the line the old scan selected first
anyway — and arithmetic is untouched, because the capture-side arithmetic rule
(`tools/fsi_capture.py`: take the `=` line, then a character run) never searched
for a later candidate, and its historical forward behaviour is pinned by an
existing case.

**Rejected alternative, measured first:** `prefer_last_answer: true` (the
setting the Small and base Inkling goldens carry). It fixes FSI-TEST-03 but
corrupts the other two on this stream — FSI-TEST-02 becomes
`Balance_Alpha=100` (the last `Balance_Alpha=` is inside a partial reasoning
line) and FSI-TEST-03 becomes `Drawdown>ThresholdThe` (the answer runs into the
next sentence without a separator). The option is a model-shape hint, not a
repair, and this entry records why it was not used here.

Capture: `tools/inkling_loop_oracle.py` (release's own transformers 5.24 text
tower, CPU bf16, 64 threads, ~0.9 s/token). 512 tokens per case,
`ignore_eos=true` (the stream does not reach EOS inside the window). All three
cases are the model's own answers, and the converted engine reproduces all
three at `kOff`.
