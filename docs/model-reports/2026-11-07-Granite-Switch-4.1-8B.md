# Granite-Switch 4.1 8B — ingestion report

| | |
|---|---|
| Model | `granite-switch-4.1-8b`, `model_type = "granite_switch"`, `GraniteSwitchForCausalLM` |
| Layout | `logestix-granite-switch` (NVFP4 base, FP16 aLoRA planes, FP16 tied embed/head) |
| Source | 17.55 GB BF16, 4 shards, 804 tensors |
| Converted | 5.7 GB on disk; **6.09 GB** device-resident (`6 088 178 432` bytes) |
| VRAM profile | min 6.45 GB / optimal = max 29.27 GB at ctx 131 072 (KV-dominated) |
| Precision | NVFP4 base + a second NVFP4 residual plane on the **attention** projections (`"residual": "attn"` in the index) |
| Hardware | 1 × RTX 5090 (32 GB), 750 GB host RAM, CUDA toolkit 13.5 / driver 13.4 |
| Runtime | token-by-token prefill + persistent KV cache, one stream |
| Date | 2026-11-07 |
| Reference | `granite-switch==0.1.1` + `transformers==5.17.0`, CPU BF16 oracle |

**Verdict at a glance:** the sibling runs end to end on real weights and passes
every gate it can pass — SelfCheck ("Paris") bit-exact across three cycles, FSI
3/3, CODE 3/3, the two embedded aLoRA adapters and the control-token switch
reproduce the source's own first token on all three adapter states, and the native
chat framing reproduces the source ids token for token. DET is 2/3 on DET-02
where the **source itself** answers `CRITICAL` against the prompt's `CATASTROPHIC`
— the same source/prompt disagreement the 30B records, not an engine defect.

FSI-01 was red on the first conversion and is green after a measured precision
change; the ablation that chose it is below, because "make it more precise" is
the last resort and the placement had to be earned.

---

## Publish-ready post

### Granite Switch 4.1 8B, running locally in a C++/CUDA engine

We ported IBM's Granite Switch 4.1 8B checkpoint — the Granite 4.1 dense decoder
composed with two embedded act-LoRA adapters, control-token routing and token
exchange — into our Blackwell inference engine. The release is 17.55 GB of BF16;
we convert it in place to a **5.7 GB** NVFP4 tree that keeps the two adapter
planes in FP16 and occupies **6.09 GB** of device memory at decode. On a single
32 GB consumer card the KV cache, not the weights, is what bounds context.

Nothing is re-exported or re-trained: the same checkpoint the vendor's reference
package serves is read directly, including the composed tensor names
(`base_layer`, `lora_A/B`, `slices`, the switch codebook and the exchange LUT).

#### What this checkpoint is, and what it is not

It is the small sibling of Granite Switch 4.1 30B and shares its architecture,
tokenizer bytes and chat template exactly — the tokenizer files are md5-identical
to the 30B's. It is **not** the 30B with fewer layers: four fields differ and
three of them are part of the arithmetic.

1. **`residual_multiplier` is 0.22, not 0.175.** Every sublayer residual is
   scaled by this. A copy of the 30B's constant decodes a plausible but wrong
   model — fluent text, wrong numbers.
2. **`rope_theta` is 1e7, not 5e7.** Invisible at short prompts, wrong at long
   positions, which is exactly the kind of error a short-prompt smoke test hides.
3. **The FFN width is 12 800, not 32 768**, with 40 decoder layers (the config's
   `num_hidden_layers` is 42: 40 decoder layers plus the switch's counting and
   memory cache slots — the same +2 rule as the 30B's 66 = 64 + 2).
4. **`mamba_*` and `ms_*` keys are present in `config.json` and mean nothing
   here.** `layer_types` is uniformly `full_attention` and the tensor index has
   no recurrent tensors. Treating the presence of those keys as evidence of a
   hybrid layer pattern would be a real trap for an importer that trusts the
   config over the weights. The tensor index is the authority.

The rest of the semantics are the composition's, not the size's, and were
re-verified rather than assumed: the switch is a prefix scan over control tokens
(not a learned layer), token exchange (`<|guardian-core|>`→27, `<|decision|>`→14711)
happens *before* embedding while adapter selection reads the original ids,
embeddings are scaled by 12 and logits divided by 16, and the head is tied
(there is no `lm_head`).

#### Adapters

`guardian-core` is rank 16, `decision` is rank 64; the checkpoint zero-pads both
to rank 64 and pre-scales the LoRA `B` by `alpha/rank`. The converter verifies
the A-tail padding and stores only the real rank. Against the CPU-BF16 source,
the converted engine reproduces the first generated token on all three states:
base `The` (791), guardian-core `{"` (5018), decision `A` (32).

#### Measured

| check | result |
|---|---|
| SelfCheck ("Paris"), 3 cycles | **PASS**, bit-identical, `logits_hash = b7c1aa453d83a86f`, top-1 `12366` ("Paris"), `n = 100 355` |
| Native chat framing → input ids | **exact match** with the source oracle (real weights) |
| aLoRA adapter parity (base / guardian-core / decision) | **3/3**, first token identical to the source |
| Tokenizer (converted arrows vs checkpoint) | **PASS** over the shared corpus |
| FSI | **3/3** |
| CODE | **3/3** |
| DET | **2/3** — DET-02, the source's own answer (see below) |
| Decode rate at 2048 tokens | **not measured** — the perf harness is Release-gated and the Release tree was not rebuilt in this session |

#### FSI-01: red on the first conversion, green after a measured precision change

This is the one gate that needed engineering, and it is worth recording in full
because the tempting fix was the wrong one.

**The fork.** FSI-01 asks for the present-value formula. The source answers
`PV=CF*((1-exp(-r*n))/(1-exp(-r)))`; the first conversion answered
`PV=2500000*((1-exp(-0.0835*7))/(1-exp(-0.0835)))` — the same formula evaluated
with the prompt's own numbers, all three of them correct. The choice is at the
first symbol after `PV=`: the variable name or its value.

**Is it a tie or a bug?** Teacher-forced the golden prefix and compared the two
rows at that position. Source: `CF` +35.75 against `250` +28.88, a margin of
**+6.9 logits**. Converted: `CF` +30.08 against `250` +31.34, **−1.3**. A 8.2-logit
swing is not noise, so the bug checklist ran before any precision work: the tied
head's `global_scale` is applied (the answer's constants are all correct, which a
dropped scale would not leave), `Free()` releases every plane, and the written
NVFP4 records decode back to the source at **relL2 0.094–0.095** on every tensor
class and on layers 0, 20 and 39 — a healthy 4-bit body, not a broken codec
(measured by decoding the engine's own arrows against the BF16 checkpoint).

**So: accumulated 4-bit body error.** The fix is the documented one — a second
NVFP4 plane carrying `w - dequant(base)`, decoded as one more GEMV with
`acc = true`, ~8 effective bits for 2x the bytes on those tensors.

**The placement was measured, not assumed.** The rule is explicit that a recipe
which works on one member does not transfer, and that offline HF-side simulation
is only qualitative. Four trees were converted from the same checkpoint and probed
at the fork (margin = logit(`CF`) − logit(`250`); source +6.88):

| placement | Δ on disk | Δ resident | margin | FSI-01 |
|---|---|---|---|---|
| none | — | — | **−1.27** | red |
| **attn** (qkv + o) | **+0.9 GB** | **+0.94 GB** | **+1.58** | **green** ← shipped |
| mlp (gate_up + down) | +3.3 GB | +3.4 GB | +2.19 | green |
| attn + mlp | +4.2 GB | +4.3 GB | +7.14 | green |

`attn` is the cheapest set that clears the gate **and** the one that leaves the
biggest tensors at 4 bits: the MLP is 79% of this layer's bytes, so the two halves
of the rule ("cheapest that clears", "leave the biggest alone") point the same
way. `attn+mlp` reproduces the source margin almost exactly (+7.14 vs +6.88) but
costs 3.3 GB more for a gate that is already green. The knob is a convert-time env
(`LSX_GSW_RESIDUAL=attn|mlp|attn+mlp|none`), not a runtime flag, so the next member
can re-run the ablation without editing code and the choice is visible in the
index as `"residual": "attn"`.

**Verified after the change:** FSI 3/3, SelfCheck still bit-exact over three cycles
(`b7c1aa453d83a86f`), CODE 3/3, adapter parity 3/3, tool round-trip green, and a
legacy 4-bit-only tree still loads and decodes (the plane is optional in the
loader, so old artifacts are not invalidated).

**Cost, stated honestly:** device-resident 5.14 → 6.09 GB, tree 4.8 → 5.7 GB, and
the attention projections now read twice the bytes per token. The decode-rate
consequence is **not measured** (see the perf note above); it is bounded by the
attention share of the per-token read, which is 21% of this model's quantized
weights, and the family is weight-streaming bound rather than compute bound.

#### DET-02: red because the source is red

DET-02 asks for the word that starts and ends with `C`. The golden records
`CATASTROPHIC` — the prompt's own Expected Template — and the source answers
`CRITICAL`. The capture prints exactly that warning and stores the template's word
deliberately: DET measures instruction-following, not source parity, which is why
its goldens are template-derived. The engine under the residual plane answers
`CRITICAL`, i.e. it now behaves like the source. The 30B's row records the same
2/3 for the same reason.

Worth noting for the opposite reason: with the 4-bit-only tree the engine *passed*
DET-02 by answering `CATASTROPHIC`. That was the quantization error landing
accidentally on the template word — a red flag that a "green" gate can be green
for the wrong reason, and a reason not to grade fidelity by that cell alone.

#### Defects found in the checkpoint's surroundings, not its numbers

- **The conversion pipeline wrote a hard-coded model name.** The index writer
  emitted `"model": "Granite-Switch-4.1-30B"` for every checkpoint it was given,
  so the 8B tree would have claimed to be the 30B. Found by a test written before
  the fix; the identity now comes from the checkpoint directory and is validated
  on load. This is a converter defect that would have silently mislabelled every
  future member of the family.
- **Two capture tools stamped goldens with the 30B's name**, and three stage
  tests were hard-coded to the 30B while the manifest for the 8B claimed them —
  a sibling "covered" by a test that verifies its parent. Both fixed; the parity
  test now rejects a golden whose `model_type` is not its own.
- **A test tried to `dynamic_cast` the model to the 30B class**, which would have
  failed at runtime for the 8B instead of at review.

---

## Evidence

- Stage 0 triage and design:
  `docs/superpowers/specs/2026-11-07-granite-switch-4.1-8b-design.md`.
- Execution trail (commands, order, what each printed):
  `docs/superpowers/plans/2026-11-07-granite-switch-4.1-8b.md`.
- Passport: `libs/lsxcommon/include/lsxcommon/granite_dense_switch_8b_constants.h`
  (`lsxcommon::granite_switch_8b::frozen`), a leaf; the shared comparison
  `GraniteSwitchGeometryMismatch()` lives in `granite_dense_switch_load.cpp` and
  reports every differing field plus the identity.
- Manifest: `models/granite-switch-4.1-8b.json`; catalogue row in
  `docs/MODELS.md`; matrix row in `docs/SELFCHECK_MATRIX.md` §1.
- Gates, 2026-11-07, one test at a time, GPU exclusive
  (`linux-debug`, RTX 5090):

  ```
  test_granite_switch_8b_adapter_parity .....   Passed    2.37 sec
  test_granite_switch_8b_model ..............   Passed    0.70 sec
  test_granite_switch_8b_tokenizer_parity ...   Passed    0.04 sec
  test_granite_switch_8b_selfcheck ..........   Passed    7.77 sec
  test_granite_switch_8b_fsi_selfcheck ......   Passed    2.65 sec
  test_granite_switch_8b_code_selfcheck .....   Passed    2.29 sec
  test_granite_switch_8b_det_selfcheck ......***Failed    4.12 sec  (DET-02, source red too)
  test_perf_baseline_granite_switch_4_1_8b ..***Skipped  0.11 sec  (Release-gated)
  ```

  `test_granite_switch_8b_model` includes the real-weight native tool round-trip:
  `pinned weights loaded: 40 layers (6088178432 bytes resident)`, framing ids
  identical to the captured golden.
- SelfCheck detail: `logits_hash=b7c1aa453d83a86f n=100355 tok=12366`,
  three cycles identical. (The 4-bit-only tree measured `ad9b0c4940bb27a9`; the
  hash changed because the weights did, which is the expected direction and the
  reason a hash is not a contract.)
- Converted index (`logestix/model.index.json`): `format logestix-granite-switch`,
  `model granite-switch-4.1-8b`, `num_layers 40`, `intermediate_size 12800`,
  `rope_theta 10000000.0`, `residual_multiplier 0.22`, `num_adapters 2`,
  `adapter_ranks [16, 64]`, `residual "attn"`, `total_pinned_tensors 803`.
- Goldens, all captured from the 8B source (never copied from the 30B):
  `tests/fixtures/{fsi,code,det}/granite-switch-4.1-8b.json` and
  `tests/fixtures/adapters/granite-switch-4.1-8b.json`.
- Declared gaps: perf row not seeded (Release build not rebuilt). The FSI-01
  residual-plane work is closed and archived:
  `docs/open-tasks/2026-11-07-granite-switch-fsi01-residual.md`.
