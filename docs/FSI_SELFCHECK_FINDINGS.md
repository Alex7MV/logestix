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
| Qwen3.6-35B-A3B | captured (all 3 exact) |
| Qwen3.8-Flash-Next | captured (all 3 exact; experimental `qwen4_exp` arch loaded with patches, see below) |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | captured (all 3 exact) |
| gemma-4-26B-A4B-it | captured (FSI-01/02 rephrased; template warnings) |
| granite-4.2-30b | captured with the chat template + low-effort thinking (see below); all 3 exact |
| phi-4 | captured with the built-in ChatML template (`tokenizer_config.json`); all 3 exact |
| Qwen3.5-9B | captured (all 3; FSI-01/02 numeric substitution) — see the `qwen3_5` note below |
| ornith-1.5-35B-A3B | not capturable — config loads as `qwen3_5_moe`, but the MLX checkpoint uses `language_model.*` key names, so all HF params MISSING (needs a key-prefix remap) |
| DeepSeek-V4-Flash | recognized (`deepseek_v4`); FP8 loads after installing `accelerate` + `kernels==0.16.0`, but the FP8 linear kernel is CUDA-only and 149 G does not fit the 32 GB GPU (`device_map=auto` OOM) — needs a larger-VRAM host |
| GigaChat3.5-432B | recognized (`gigachat3_5`) + custom code; FP8 (416 G) — same CUDA/VRAM blocker as DeepSeek-V4-Flash, plus RAM pressure |
| DeepSeek-V4.1-Flash | not capturable locally — `deepseek_v41` is not in transformers 5.15.1 and the release ships its **own inference stack** (`inference/model.py`, `convert.py`, `generate.py`; tilelang FP8/FP4 kernels, tensor-parallel, target 2×80 GB GPU), not HF remote code. Do not confuse it with the flat FP8 + FP4 MoE checkpoint in the vault. |
| GLM-5.3-Flash | not capturable — `glm5_next` unknown to transformers 5.15.1 |
| GLM-5.2, Qwen3.5-35B-A3B, Kimi-K2.7-Code | source absent from the vault |

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
