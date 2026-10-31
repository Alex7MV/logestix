# Granite-Switch 4.1 30B — integration report

| | |
|---|---|
| Model | `ibm-granite/granite-switch-4.1-30b` (revision 4.1.1), `model_type = "granite_switch"` |
| Converted layout | `logestix-granite-switch` (NVFP4 base, FP16 aLoRA planes, FP16 embed / tied head) |
| Source | 56 GB BF16, 13 shards, 1284 indexed tensors |
| Converted tree | 17 GB on disk; 16.22 GiB device-resident base + adapters |
| Hardware | 1 x RTX 5090 (32 GB), 750 GB host RAM, CUDA 13.4 |
| Runtime | token-by-token prefill/decode, persistent KV cache, single stream |
| Date | 2026-10-31 |
| Reference | `granite-switch==0.1.1` + `transformers==5.17.0`, CPU BF16 |

**Verdict at a glance:** a faithful, fully local run of a composed model. The
base decoder, both embedded aLoRA adapters (`guardian-core`, `decision`) and the
control-token switch work end-to-end: SelfCheck is bit-exact across three
cycles, FSI and CODE are 3/3, adapter parity is 3/3 (including the native chat
`input_ids` token-for-token), and the model decodes at 69 t/s on one consumer
card. DET is 2/3 because the **source** itself answers `CRITICAL` where the
shared DET-02 prompt's Expected Template is `CATASTROPHIC` — a source/prompt
disagreement, recorded red rather than papered over.

---

## Publish-ready post

### Granite Switch 4.1 30B, running locally in a C++/CUDA engine

We ported IBM's composed Granite Switch 4.1 30B checkpoint — the Granite 4.1
30B dense decoder plus two embedded act-LoRA adapters, control-token routing
and token exchange — into our own Blackwell inference engine. The release is
56 GB of BF16; we convert it in place to a 17 GB NVFP4 tree and keep the two
LoRA planes in FP16. It fits a 32 GB RTX 5090 at 16.2 GiB resident and decodes
at **69 tokens/s** at 2 048 generated tokens.

The interesting part is that nothing is re-trained or re-exported: the same
checkpoint that the reference Python package serves is read directly. The
converter recognises the composed tensor names (`base_layer`, `lora_A/B`,
`slices`, the switch codebook and the exchange LUT), quantises the base
projections to NVFP4, trims each adapter to its own rank and bakes the
pre-scaled LoRA `B` in as stored.

#### What the checkpoint forced into the engine

1. **The switch is a prefix scan, not a learned layer.** The reference counts
   control tokens with a one-hot attention head and retrieves the adapter id
   from a Kerdock/DG codebook. On every position the result is exactly the
   adapter of the most recent control token (or base). We verified that
   equivalence against the reference on random sequences and implemented it as
   a host-side scan; the codebook is stored for fidelity but not uploaded.
2. **Token exchange happens before embedding.** `<|guardian-core|>` (100352)
   embeds as id 27 and `<|decision|>` (100353) as id 14711; adapter *selection*
   still reads the original ids. Getting this order wrong routes correctly but
   feeds the decoder the wrong token.
3. **Three multipliers are part of the arithmetic.** Embeddings are scaled by
   12, every sublayer residual by 0.175, and logits divided by 16. They are all
   1.0 in Granite 4.2, so a copy of that path silently decodes the wrong model.
4. **Tied embeddings.** There is no `lm_head` tensor; the head is the embedding
   matrix with `global_scale / 16`. The passport and the loader pin the alias.
5. **`num_hidden_layers` is 66, but there are 64 decoder layers.** The other two
   are the switch's counting and memory cache slots; the converter rejects the
   config unless `tensor_layers + 2 == config.num_hidden_layers`.

#### Adapters

`guardian-core` is rank 16 on q/k/v/o; `decision` is rank 64 on q/k/v/o and the
MLP. The checkpoint zero-pads both to rank 64 and pre-scales `B` by
`alpha/rank`; our converter verifies the A-tail padding and stores only the real
rank. At runtime each projection is the NVFP4 base GEMV plus an FP16
shrink/expand add for tokens whose adapter is active, so base spans pay nothing
and an adapter span pays two small launches per slice.

#### Chat and tools

The native template is `<|start_of_role|>role<|end_of_role|>content<|end_of_text|>`,
with the template's `<tools>` block and bare-JSON `<tool_call>` bodies. We
reproduce the aLoRA insertion points exactly (`<guardian>` → control token +
`guardian>`, `### DECISION` → control token + ` DECISION`) and accept
`chat_template_kwargs.adapter_name` from the OpenAI endpoint, an explicit
control token in the text, or automatic invocation-text detection.

#### What we did not run

The 470M Granite Speech encoder is a separate artifact and is not in this
checkpoint, so `<|audio|>`/ASR is out of scope. The advertised 131k context is
also not representable on a 32 GB card: its KV cache alone is ~34.4 GiB; ~32k
tokens is the practical arithmetic ceiling, and the perf number above is the
canonical 2 048-token protocol.

One parity gate is red for a source-side reason: DET-TEST-02 asks for
`CATASTROPHIC`, and the BF16 source (like our conversion) deterministically
answers `CRITICAL`. We recorded that as a source/prompt disagreement instead of
rewriting the golden or pretending the gate passed.

---

## Evidence

### Gates actually run (2026-10-31, RTX 5090)

| gate | result | note |
|---|---|---|
| SelfCheck (`Paris`, kOff) | **PASS x3** | `logits_hash = fdc632042c30919c` over 100 355 logits, byte-identical across cycles |
| FSI | **3/3** | goldens captured from the CPU-BF16 `granite-switch` oracle |
| CODE | **3/3** | same |
| DET | **2/3, DET-02 RED** | source and converted both answer `CRITICAL` (deterministic x3); prompt template expects `CATASTROPHIC` — `docs/OPEN_TASKS.md` §R10 |
| Adapter parity | **3/3** | base / guardian-core / decision first tokens `The` / `{"` / `A`, routing 0 / 1 / 2 |
| Native chat ids | **exact** | source `input_ids` reproduced token-for-token for all three modes |
| Tool framing (synthetic) | **PASS** | role markers, tools block, aLoRA insertion, bare-JSON parse |
| Converter / loader / model / kernels | **PASS** | labels `convert` / `inference` |
| Fast suite | **PASS 119/119** | includes naming, independence, suite, ctest-label guards |
| Perf | **69.28 t/s (perf) / 70.07 t/s (r3)** | 2 048 tokens, TTFT 49.09 / 190.76 ms, 552.95 / 573.69 W avg, AOCC release |

### Resident / storage

- `resident_bytes = 17 416 338 176` (16.22 GiB): NVFP4 base + FP16 LoRA + FP16
  embed table + head alias.
- Converted tree 17 GB; source 56 GB BF16.
- Perf rows in `tests/perf/perf_baselines.json` (gpu `rtx 5090`, os `linux`).

### Files

- `granite_dense_switch_constants.h` — frozen passport
- `granite_dense_switch_convert.{h,cpp}` — composed-name classifier, NVFP4 base,
  rank-trimmed adapter planes, LUT verification, index + tokenizer
- `granite_dense_switch_load.{h,cpp}` — base + adapter upload, tied head alias,
  host token preparation
- `granite_dense_switch_inference.{h,cu}` — kernels, multiplier/LoRA passes,
  KV cache and generation loop
- `granite_dense_switch_model.{h,cpp}` — `IModel` + `kType`
- `granite_switch_tool_format.{h,cpp}` — native chat/aLoRA/tool codec
- Tests: `test_granite_switch_{convert,loader,prepare,kernels,model,selfcheck,
  fsi_selfcheck,code_selfcheck,det_selfcheck,adapter_parity,tool_framing}`
- Oracle/capture: `tools/granite_switch_oracle.py`,
  `tools/granite_switch_adapter_capture.py`, loader overrides in
  `tools/{fsi,code,det}_capture.py`
