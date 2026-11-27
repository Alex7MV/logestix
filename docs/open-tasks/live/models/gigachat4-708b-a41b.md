# GigaChat4-708B-A41B — open items

Branch `feat/model-GigaChat4-708B-A41B`. Design:
`docs/superpowers/specs/2026-11-27-gigachat4-708b-a41b-ingest-design.md`.
Report: `docs/model-reports/2026-11-27-GigaChat4-708B-A41B.md`.

### G5. GigaChat4-708B-A41B — ingested (2026-11-27), FSI-02 and DET-03 red on extraction

The 37B's big sibling: byte-identical modeling/configuration/tokenizer/chat
template, 64 decoder layers (3 dense FFN + 61 MoE of 256 top-8 experts), 16
gated MLA layers carrying the DSA indexer, 48 GatedDeltaNet layers, one MTP
layer stored and never applied. 710.3 B params / 720 GB FP8 source → **379 GB**
NVFP4 tree (361 GB experts, 14 GB dense, 4.6 GB MTP), of which the expert set
cannot live on the 32 GB card: 61 × top-8 = **488 expert touches per token =
~12 GB of H2D**, 389–392 slots resident out of 15 616.

**Measured real-weight gates (AОCC Release tree, one target at a time).**

| gate | result |
|---|---|
| SelfCheck | PASS — `Paris`, 3 bit-exact cycles, `logits_hash 4c91393c68d0b3e8` (reproduced across 3 processes; identical at 64 and 389 slots) |
| CODE | 3/3 |
| FSI | **2/3** — FSI-02 |
| DET | **2/3** — DET-03, `deterministic=true` over 3 repeats |
| tool round-trip | PASS (shared GCML codec; template + tokenizer byte-identical to the 37B) |
| perf | **3.71 t/s**, TTFT 11 120.85 ms, avg 152.89 W / peak 155.60 W at the canonical 2 048-token protocol; `r3` 3.68 t/s |

**Both reds are the `I3` extractor class, not the model.**
`docs/open-tasks/live/instruments-and-guards.md` §I3:

- FSI-02: the model's raw text carries the correct formula, wrapped across two
  lines (`… *Rate_B-` / `Slip_B) …`); `ExtractFsiFormula` selects one line, and
  the line it resolves is the preamble's `Balance_Alpha=100000000`. Deterministic
  over two full runs. FSI-01 showed the same wrapping once in four runs (a
  redundant-parenthesis variant passes; the wrapped one does not).
- DET-03: asks for `BIT_ZERO`; the model's last line *is* `BIT_ZERO`, but
  `ExtractToken` returns the first standalone uppercase token, which is the
  preamble's `DET` from `DET-Validation-Core-03`. `CutPromptEcho` searches for
  32 verbatim bytes of the prompt body, and this model paraphrases instead of
  echoing, so nothing is cut.

Both extractors are shared by every one of the 43 manifests; repairing them
needs its own before/after sweep and is deliberately not bundled into a model
ingestion. The goldens are template-derived (see below), so the reds are not
evidence about source parity either way.

**Owed / not measured.**

1. **Source-side agreement** rests on the per-layer probe that the 37B ingestion
   built (`tools/gigachat4_layer_probe.py` against `LSX_GC4_DUMP_LAYERS`), not
   on a full-model oracle: executing the source needs a dequant to ~1.4 TB BF16
   against 750 GB of RAM. Not re-run here — the 708B shares every semantic knob
   with the 37B, whose probe is recorded in
   `docs/open-tasks/live/models/gigachat4-37b-a4b.md` §G1.
2. The FSI/CODE/DET goldens are **template-derived** (`--template-only` /
   `--template-golden`), recorded as such in each file's `source`.
3. No long-document evaluation at the 1 M-token window the config declares: the
   session cap is 4 096 (5 GiB of MLA KV); DSA above 2 048 keys is covered by
   `test_gigachat_moe_attn` and `test_gigachat_moe_37b_indexer`, and by the
   2 048-token perf run.
4. MTP layer 64 and `dflash_draft/` stay stored-and-counted, never applied.
5. `test_ctest_labels_guard` does not **link** on the release/AOCC tree
   (`undefined reference to absl::raw_log_internal::internal_log_function`); it
   builds and passes in the debug tree. Pre-existing, unrelated to this diff,
   seen during this ingestion and not filed as new.

**Defects found and fixed while ingesting** (both affect the shipped 37B row):

- `24ea89b3` — the DSA indexer rotated by `2*pos`: the decoder passed an
  already-advanced rope pointer into a callee that selects row `pos` itself.
  Invisible below `index_topk` (the selection is the identity there), wrong and
  out-of-bounds above it.
- `53783bc2` — the converter's VRAM profile classified tensors with the
  one-argument classifier, so on a checkpoint deeper than 32 layers every MoE
  layer past 32 read as MTP ballast; and the converter held the whole expert set
  in one map, which is 387 GB here.
