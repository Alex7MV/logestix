# GigaChat4-37B-A4B — open items

### G1. GigaChat4-37B-A4B — ingestion complete (2026-11-26)

Branch `feat/model-GigaChat4-37B-A4B`. Report:
`docs/model-reports/2026-11-26-GigaChat4-37B-A4B.md`.

**Real-weight gates (release/AOCC tree).** SelfCheck bit-exact ×3
(`logits_hash 20148869d4404e55`, top-1 `Paris`), FSI 3/3, CODE 3/3,
DET 3/3, tool round-trip `get_weather {"city":"Paris"}`, perf seeded through
the harness: **59.06 t/s** / TTFT 449 ms at the canonical 2 048-token
protocol. 23 GB artifact from a 39 GB FP8/BF16 source; the entire 19.7 GB
expert set is device-resident, so decode is compute-bound, not PCIe-bound.

**The defect this ingestion shipped with, and how it was found.**
`GatedNorm` in this checkpoint computes its rank-16 `silu→sigmoid` gate from
the state **already scaled** by the norm weight (`s = xn·w`,
`out = s·2·sigmoid(Wdown@silu(Wup@s))`), where the 3.5 checkpoint gates from
the normalized vector. Reading 4.0 the 3.5 way is a 36× error on the measured
activation (the `post_self_attn_layernorm` weight has mean ≈ 0) and grows the
residual stream at every layer: the first real-weight SelfCheck answered `ab`
instead of `Paris`, and the tool round-trip produced token salad.

Localisation used the operator-approved source probe, not guesswork:

1. `tools/gigachat4_layer_probe.py` loads the checkpoint's layer prefix
   (FP8 block-dequantized with its scale tiles), runs the reference
   `modeling_gigachat4.py`, and compares per-stage activations against the
   engine's dump (`LSX_GC4_DUMP_LAYERS=<dir>` in `GcPipeline::Step`).
2. The comparison said embedding / norm / GDN block all ≥ 0.996 cos, then
   `post_self_attn_norm` dropped to **0.4284** and the layer residual to
   **0.0757** — i.e. the defect was in the norm, not the indexer, not the
   router, not the rope.
3. Feeding the gate from `xn·w` reproduces the reference module to 4 decimals
   (std 0.00813 vs 0.00813) where the legacy form gave 0.33135.

After the fix the same comparison reads **0.99683** on the layer-0 residual,
bounded by the NVFP4 quantization floor (`relL2 ≈ 0.094` per matrix), and every
gate is green. The variant is now data in `GcPipelineDims`
(`norm_gate_from_scaled`), with the 3.5 entry points asserted bit-identical on
device.

**Second defect, in the goldens, not the engine.** The `<|think|>` closing
marker is a **special** token, which `arrow_tokenizer_decode` drops. Goldens
captured in the checkpoint's default reasoning mode therefore left the gate's
`</think>` cut unable to fire, and the answer extractor latched onto the first
capitalised token of the model's chain-of-thought (DET extracted `DET` from the
quoted context while the model's own last line was the right answer). The
goldens are now captured in the checkpoint's instant mode
(`template_kwargs: {"reasoning": False}`), which is also the frame the engine
serves for single-answer prompts.

**Open, recorded rather than hidden.**

- The FSI/CODE/DET goldens are **template-derived**: the three answer strings
  come from the prompt set, not from executing the source (which needs a
  host-side dequant to 74 GB BF16). What pins "these are the model's answers"
  is the per-layer probe above plus the SelfCheck/tool round-trip, not a
  source-side generation.
- `tools/gigachat4_oracle.py` runs the full-model logits comparison when a
  longer probe is wanted; it needs ~74 GB of host RAM and a GPU dequant pass,
  so it is not part of a gate.
- Context cap is 4 096 (rope and DSA index tables sized to it), not the
  checkpoint's 1M. DSA above 2 048 keys is covered by the indexer kernel tests
  and the 2 048-token perf run, not by a long-document eval.
- MTP (layer 32) and the shipped `dflash_draft` drafter are stored and counted,
  never applied.
