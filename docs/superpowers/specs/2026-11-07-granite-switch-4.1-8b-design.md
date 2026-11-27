# Granite-Switch 4.1 8B ingestion design (Stage 0 triage + plan)

## Context

`granite-switch-4.1-8b` (IBM, `/mnt/fast_storage/lsx_model/ibm-granite/granite-switch-4.1-8b`,
downloaded 2026-11-06) is the small sibling of the already-ingested
`granite-switch-4.1-30b`. Both are `GraniteSwitchForCausalLM`
(`model_type="granite_switch"`): a Granite 4.1 dense decoder with two embedded
act-LoRA adapters, control-token routing and token exchange, plus switch
counting/memory slots that the reference implements with a Kerdock codebook.

This document is the Stage 0 record for the sibling and the plan for the
remaining stages. It is deliberately short where the 30b design
(`2026-10-31-granite-switch-4.1-30b-design.md`) already answers the question,
and long where the two checkpoints **differ**, because every difference is a
place where reusing the parent's assumptions silently decodes the wrong model.

Read as a process test: the operator asked for the full ingestion cycle to be
run on this model specifically to see whether the protocol still holds end to
end. The audit findings are collected in `docs/open-tasks/` and in the final
report, not here.

## Stage 0.1 — classification: sibling variant, different geometry

- Same `model_type` (`granite_switch`), same tensor naming, same adapter
  technology, same tokenizer, same chat template → **sibling**, parent loader,
  converter, kernels and tool bridge are reused unchanged.
- Geometry differs in **four** fields, three of which are part of the
  arithmetic rather than the shape → **its own frozen passport** (Stage 3).
  This is the "same architecture, different geometry" branch of the triage, not
  the "identical geometry → reuse the parent's class" branch.

## Stage 0.2 — geometry, measured from the checkpoint (not from the card)

Read from `config.json` and cross-checked against the safetensors headers
(4 shards, 804 tensors, 17.55 GB BF16, all BF16 except `codebook` F32 and the
exchange `lut` I64):

| field | 8B | 30B | delta |
|---|---|---|---|
| decoder layers | **40** | 64 | own passport |
| `num_hidden_layers` (config) | 42 | 66 | = decoder + **2 switch cache slots**, same rule |
| hidden_size | 4096 | 4096 | — |
| q/kv heads, head_dim | 32 / 8, 128 | same | — |
| intermediate / shared_intermediate | **12800** | 32768 | own passport |
| vocab_size | 100355 | 100355 | — |
| rms_norm_eps | 1e-5 | 1e-5 | — |
| **rope_theta** | **1e7** | 5e7 | **arithmetic** |
| attention_multiplier | 0.0078125 (1/128) | same | — |
| embedding_multiplier | 12.0 | same | — |
| **residual_multiplier** | **0.22** | 0.175 | **arithmetic** |
| logits_scaling | 16.0 | same | — |
| tie_word_embeddings | **true** (no `lm_head`) | same | — |
| bos / eos / pad | 100257 / 100257 / 100256 | same | — |
| adapters | `guardian-core` r16, `decision` r64 | same | — |
| control / substitute ids | 100352, 100353 → 27, 14711 | same | — |
| `layer_types` | 42 × `full_attention` | 66 × `full_attention` | **no hybrid** |
| mamba_* keys | present, **unused** | present, unused | dead config |
| `num_local_experts` | **0** | 0 | no routed experts |

**The hybrid trap is absent but was checked, not assumed:** `mamba_*`,
`ms_*` (Kerdock counting) and `mamba_n_heads: 128` are all in `config.json`
while `layer_types` is uniformly `full_attention` and no `mamba`/`conv1d` tensor
exists in the index. Treating the presence of `mamba_*` keys as evidence of a
hybrid layer pattern would be wrong; the tensor index is the authority.

**Non-uniform shapes: none.** Every per-layer tensor has the same shape in all
40 layers (`[2,1,64,4096]` A-slices, `[2,1,1024|25600|8192|4096,64]` B-slices,
qkv `[6144,4096]`, o `[4096,4096]`, `input_linear` `[25600,4096]`,
`output_linear` `[4096,12800]`), so there is one layer shape, not 40.

## Stage 0.3 — tokenizer and chat template: byte-identical to the parent

`tokenizer.json`, `merges.txt`, `vocab.json` and `chat_template.jinja` are
**md5-identical** to the 30b checkpoint. Consequences, all of them a decision
already made once:

- `tokenizer_kind = 0`, the parent's arrow bridge applies unchanged (Stage 4).
- Stage 4b has nothing new to reverse-engineer. The parent's verdict stands:
  the model emits a bare JSON tool call, so the `model.h` default already
  parses it and `tool_roundtrip` stays `false` in the manifest (declared, not
  silently skipped) — the same verdict the 30b recorded.
- The control tokens are ordinary vocab ids (100352/100353, `special: false` in
  `added_tokens`), which is why the exchange happens as an id substitution
  before embedding rather than as a text marker.

## Stage 0.4 — ballast: strip and COUNT

| item | in weights? | action |
|---|---|---|
| ASR / speech encoder (470M, `asr_enabled: true`) | **no** — device pipeline config only, `asr_model_id: null` | config keys ignored; nothing to strip |
| `io_configs/` (guardian-core, decision YAML) | no | not read |
| `compose_report.json`, `BUILD.md`, `adapter_index.json` | no | metadata only; `adapter_index.json` cross-checked for adapter names/ranks |
| `mamba_*` / `ms_*` config block | no | ignored; `mamba_chunk_size` etc. never read |
| `num_local_experts: 0` + `num_experts_per_tok: 2` | no experts | the "MoE" keys are the switch, not routed experts; the converter classifies every tensor, and there is no expert tensor to skip |

The converter's existing contract (`classify_granite_switch_tensor` returns
`Unknown` → hard error, `CheckTensorCoverage` + `CheckLayerContiguity`) makes an
unclassified leftover a build failure rather than a silent drop, so "nothing was
forgotten" is machine-checked here rather than asserted.

## Stage 0.5 — feasibility on this machine

Derived from the checkpoint, not from the card:

- quantized base linears: `4 × 4096² + 6144 × 4096 + 25600 × 4096 + 4096 × 12800`
  = 199.2 M params/layer × 40 = **7.97 B** → NVFP4 at 0.5625 B/param ≈ **4.48 GB**;
- LoRA planes stay FP16. The source stores **395.8 M** padded params
  (≈ 0.79 GB); the converter trims each adapter to its real rank, so the tree is
  smaller than that, not larger;
- tied embed/head FP16: `100355 × 4096 × 2` = **0.82 GB**;
- norms/biases FP16: negligible.

**≈ 6.1 GB of weights before the rank trimming**, versus 16.2 GiB for the 30b.
KV cache is the only context-dependent term: `2 × 40 × 8 × 128 × 2 B` =
**0.164 MB/token** → 0.17 GB at 1024 tokens, 21.5 GB at the 131072 maximum. On
the 32 GB RTX 5090 the model is comfortable at any context the gates use
(≤ 2048), and the emitted VRAM profile will record the real numbers rather than
this estimate.

No host-mapped expert store is involved (no routed experts), so the
`ExpertStore` R1/R2 rules do not apply to this family.

## Stage 0.6 — semantics to verify before trusting a token

Carried from the parent because they are properties of the composition, not of
the size, and re-verified per checkpoint where the value differs:

1. **Switch is a prefix scan.** Adapter at every position = the adapter of the
   most recent control token. Verified against the reference on random
   sequences for the 30b; the codebook and `control_to_substitute_lut` are the
   same tensors here `[2048,64]` F32 / `[100355]` I64, so the mechanism is
   unchanged. Must re-verify on the 8b with the parent's adapter-parity oracle.
2. **Token exchange precedes embedding**: 100352→27, 100353→14711, while
   adapter *selection* reads the original ids.
3. **Three multipliers are arithmetic**: embedding ×12, residual ×**0.22**
   (30b: 0.175), logits ÷16. These are 1.0 in Granite 4.2 — copying that path
   decodes a different model. The passport must pin 0.22.
4. **theta is 1e7, not 5e7.** A parent-copied constant costs correctness at
   long positions and nothing at short ones, so a short-prompt gate would pass.
   The passport pins it and `ValidateGeometry` compares it.
5. **Tied head.** No `lm_head`; the head is the embedding with
   `global_scale / 16`.
6. **`num_hidden_layers` includes 2 switch slots**: 42 = 40 + 2, enforced by the
   converter, same rule as 66 = 64 + 2.
7. **`FastMathMode::kOff` stays pinned** for this family: activations reach the
   same ~6e3 scale (embedding ×12), which overflows the NVFP4 TC activation
   quantiser (max ~2.7e3) on long prompts. The 8b is *not* exempt because it is
   smaller; the defect was activation range, not model size.

## Plan for the remaining stages

| stage | work | why it is not "reuse and hope" |
|---|---|---|
| 1 convert | reuse `ConvertGraniteSwitchModel` (0 config-gated geometry rules, fully config-driven) | needs a **convert test at the 8b's geometry** so the `42 = 40 + 2` rule and the 12800 FFN width are exercised, not just the 2-layer synthetic fixture |
| 2 NVFP4 | reuse | residual plane only if a gate goes red per the parent's decision procedure |
| 3 registration | **new passport** `granite_dense_switch_8b_constants.h` (`lsxcommon::granite_switch_8b::frozen`) + `GraniteSwitch41_8BModel` + factory row | frozen geometry must be the 8b's; the parent's `ValidateGeometry` would reject the real checkpoint |
| 4 tokenizer | reuse (identical bytes) | — |
| 4b tool bridge | reuse; verdict stays "generic bare JSON" | declared in the manifest, not skipped |
| 5 kernels | reuse (no `frozen::` reference in the .cu; all geometry from `pw_.top`) | — |
| 6 sandbox | reuse | — |
| 8 harness | shared parity driver + shared selfcheck driver + a `perf` row | goldens must be **captured from the 8b source**, not copied from the 30b |
| 9 report | `docs/model-reports/2026-11-07-Granite-Switch-4.1-8B.md` | numbers measured, not projected |

### Refactor shape for Stage 3

`GraniteSwitch41_30BModel` currently hard-codes the passport inside one class.
Both classes must keep a resolvable `kType` literal and the shared lifecycle
body must stay in one TU (the independence guard concatenates the manifest's
`lifecycle_tu` per model), so:

- `GraniteSwitchDenseModel` (not `final`): the whole implementation, taking a
  `GraniteSwitchFrozen` expectation struct in its constructor. `ValidateGeometry`
  compares against that struct instead of the `frozen::` namespace directly, and
  logs the model name it was told.
- `GraniteSwitch41_30BModel` and `GraniteSwitch41_8BModel`: two `final`
  subclasses, each declaring `static const std::string kType`, each passing its
  own passport values. No template — `test_manifest_sync` resolves `kType` by
  grepping `<model_class>::kType = "..."`.
- The 8b passport is a leaf; the 30b passport is untouched, so its manifest,
  gates and frozen-geometry check keep exactly the values they were verified
  with.

This keeps "unique passport per model" and "no cross-family geometry
borrowing" true while avoiding a copy of a 220-line model body.
