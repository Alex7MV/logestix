# Nemotron-3.5-Lightning-30B-A3B (BF16) — open items

### R20. Nemotron-3.5-Lightning-30B-A3B (BF16) — first lifecycle run, one loader defect, parity red with a measured mechanism (2026-11-13)

Branch `feat/nemotron-lightning-bf16-lifecycle`. The BF16 sibling of the ingested
NVFP4 member had a manifest row, a registered `kType`, a leaf class, a converter
and eleven ctest targets — and no evidence: no converted tree, every gate a SKIP,
no goldens, `tool_roundtrip: false`, perf rows from 2026-09-16. This session
converted it and ran every stage on real weights. Report:
`docs/model-reports/2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md`.

**Sibling, not a new lineage.** Identical frozen geometry and the same
`logestix-nemotron-nvfp4` output layout, so `NemotronModelBase`, the loader, the
kernels, the expert cache and the tool codec carry it unchanged; a second
passport for the same widths is what `test_model_independence_guard` forbids.

**The defect: the BF16 loader accepted the NVFP4 sibling's tree.** Both
distributions write the same format tag, so only the index's `model` field
separates them — and `NemotronNvfp4Model` has enforced that since ingestion while
`NemotronHModel` did not. A mixed-up tree would have loaded and then verified
against the other model's goldens. Fixed with `frozen::kModelName` plus a check
in `LoadAndValidateIndex`; the synthetic fixture's index writer now emits the
converter's string (it wrote `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, which
no converter emits — that is why nothing caught the gap).
Test: "nemotron Load rejects a tree built for the other distribution", verified
RED with the check disabled.

**Conversion, measured.** 44 s from page cache → 18 GB (2.2 GB pinned / 107
Arrow files, 16 GB experts / 23 dirs × 128), 2944 routed experts, 355 pinned
tensors, 270 MTP tensors cut and counted. Per-tensor relL2 against the BF16
source (decoding the written record, `tools/nemotron_bf16_quant_probe.py`):
attention 0.0947, Mamba 0.0950, shared expert 0.0943, routed experts 0.0946;
`lm_head` and `embeddings` are BF16 passthrough (0.000000) — this distribution's
head is NOT quantized, unlike the vendor NVFP4 tree. An ideal E2M1 grid with an
exact float block scale measures 0.0937–0.0939 on the same tensors, so the codec
sits on the 4-bit grid bound and E4M3 block-scale rounding costs ~1 % relative.

**Gates as measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**
- SelfCheck **PASS**: Paris ×3 bit-exact, `logits_hash 0dba89277c5d1250`, top-1 6993.
- Tokenizer parity **22/22** on the real vault (290 assertions).
- Tool round-trip **PASS** on real weights: `get_weather {"city":"Paris"}`;
  the manifest's `tool_roundtrip` is now `true`.
- Perf re-seeded through the harness at 2048 tokens: **127.00 t/s / r3 120.32**,
  TTFT 85 / 89 ms, 250 W mean. The NVFP4 sibling reads 127.55 / 120.39 — the
  expected agreement for the same layout.
- **FSI 2/3, CODE 2/3, DET 1/3** (DET determinism 3/3).

**Parity provenance is better than the sibling's, and that is what makes the red
informative.** This checkpoint loads in `transformers` 5.15.1 directly, so all
three goldens in `tests/fixtures/{fsi,code,det}/nemotron-3.5-lightning-30b-a3b.json`
record the vault path as `source` — the source was **executed** and answers all
nine cases exactly as the prompt templates specify (after 1358 / 734 / 363
thinking tokens; `transformers` 5.15.1 / torch 2.13.0+cu130, CPU bf16). No oracle
script, no template derivation. Alignment is mechanical: the goldens are captured
with `enable_thinking=True`, and their `input_ids` decode **byte-exactly** to the
served frame pinned by `tests/fixtures/nemotron/fsi01_frame.txt` (first id 10,
no bos — `add_bos_token = false`). `tests/test_nemotron_moe_model.cpp` "FSI-capture
frame equals the served frame" fails if the renderer and the template drift (a
thinking-off mutation was verified to fail it).

**The measured mechanism (why 02/03 are red).** Across the three parity runs every
case ran to its **token budget** — the decoded raw ends mid-repetition at the
truncation point and the wall time matches a full-budget decode (FSI-02 at 2048
took 38 s) — while the source stops at 363/734/1358 tokens. Where it reaches the answer it reaches it *correctly* and then
repeats it: FSI-02 logs 15 copies of
`</think>Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`,
CODE-02 logs `</think>m=l+((h-l)/2)` repeatedly. The extractor's last-line
preference therefore lands on the budget-truncated copy (`...Rate_B-S`, `PARAD`)
or rejects the repetition (`''`). **The verdict is budget-sensitive** — at 2048
FSI-03 passes, at 3072 it fails and FSI-02 extracts `Balance_Alpha=(((` — which is
why the row is debt and not a pass. Two other generation paths in the same build
(the tool round-trip and the Paris probe) terminate normally, so "the decode loop
never advances" is not supported; whether the non-termination is only 4-bit body
noise on the `</think>` vs `<|im_end|>` argmax is **not established**.

**The instrument that cannot see what it is asked (keep this).** A grep for
`<|im_end|>` in a decoded raw can never prove that a model failed to emit eos:
`arrow_tokenizer_decode` drops special tokens, so a clean stop is invisible in
the text as well. The observable that does work is the repetition running to the
budget boundary, plus the wall time. The one prompt where the answer alone was
ambiguous (`FSI-01`) is also the one that passes — do not generalise from it.

**Open, explicitly.**
- N1. The residual-plane lever (`gemma-4-26b` recipe: attn + Mamba projections at
  ~8 effective bits, routed experts left 4-bit, ≈ +1.1 GiB resident, doubled
  attn+Mamba read) is the documented next step. Not taken: it changes the tree,
  re-seeds every perf row and makes the two distributions incomparable — an
  operator decision on 2026-11-13 to measure before changing the tree. Trigger:
  any claim that this family's parity can be green on this card without it.
- N2. Session state after a case that exhausts its budget: in one 3072-budget run
  the case following a cap-exhausted case logged `raw=''` while generating ~3000
  tokens (a decode of a non-empty tail returned an empty string); the same case
  in isolation is non-empty. `Infer` resets the session per call, so this is not
  the obvious cache-reuse bug — reproduce it before believing it.
- N3. The capture budget must exceed the SOURCE's own need: at the tools' 384-token
  default all three FSI traces were truncated inside the reasoning and the
  extraction recorded the prompt's own `Expected Template:` line as the answer
  (a tautology). The recorded budget is 2048, measured from the source's
  363–1358. Any future re-capture must check `</think>` is present in the raw.
