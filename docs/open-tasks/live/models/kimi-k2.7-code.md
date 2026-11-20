# Kimi-K2.7-Code — open items

### R2. Kimi-K2.7-Code — audit closure (2026-10-26)

The 2026-08 adaptation was re-audited against the checkpoint's own reference and
four blockers were found and fixed (details and every number:
`docs/open-tasks/2026-10-26-kimi-audit-verification.md`):

- **RoPE pairing** was half-split; the vendor de-interleaves before
  `rotate_half`, i.e. adjacent pairs. Cross-position dots differed by O(1).
- **Attention temperature** dropped the YaRN `mscale^2` factor (0.0722 vs 0.1447).
- **Encoder prepended a `[BOS]`** the reference tokenizer never emits, and the
  **stop id was `[EOS]` (163585)** instead of `config.json eos_token_id` =
  `<|im_end|>` (163586), so generation never cut at the end of a turn.
- **tiktoken merge reconstruction was unsound** (9/14 corpus cases diverged);
  the vocabulary itself is the merge table, and added tokens are matched
  literally. Two shared-pretokenizer defects came out of the same gate (Han vs
  the letter alternative, the `'s/'t/'re…` contraction suffix).

Tool-calling now uses the checkpoint's native grammar (stage 4b), the converter
classifies and counts ballast (`skipped_visual_tensors`,
`skipped_shape_tensors`, `mtp_*`) and fails on unknown tensors, and the model
was re-converted (23040 experts, 542 GB). Real-weight gates: SelfCheck Paris ×3
bit-exact; layer oracle relL2 0.0034 (layer 0) → 0.1226 (layer 60), cos ≥ 0.9925.

Open items for this family:

- **K1 — golden debt (FSI/CODE/DET).** Declared, not captured, and now
  **measured as infeasible on this host**: with `compressed-tensors` 0.19 and
  `tiktoken` 0.14 installed, `AutoModelForCausalLM.from_pretrained(...,
  device_map="cpu", attn_implementation="eager")` loads the config, applies the
  quantizer to 69 120 modules and then runs "Decompressing model": it
  materializes bf16 copies of the INT4 experts — 187 GB at 8 % of the module
  count, i.e. **~2.0 TB extrapolated against 750 GB of RAM**. The 581 GB source
  alone mmap-fits (page cache), the decompressed form does not. A vendor-stack
  capture (vLLM/SGLang `--tool-call-parser kimi_k2`, per the checkpoint's
  deployment guide) needs multi-GPU hardware, not this box. The three targets
  therefore SKIP (tree present, golden absent) and the rows stay in
  `kGoldenDebt`; the layer oracle is the parity instrument that *is* available
  (`docs/open-tasks/2026-10-26-kimi-audit-verification.md`). Do **not** derive a
  golden from the engine.
- **K2 — async decode overlap.** The per-step D2H + event wait is not a "park":
  the host needs the token id to stream it and to honour abort, so it is a real
  dependency. The old note about a "cross-cycle divergence on a fully-async
  decode" could not be reproduced after the numeric fixes
  (`LSX_KIMI_ASYNC_STEP=1` compares 6 cycles, identical hashes), and the
  diagnostic path itself is still host-serialized (`cudaStreamSynchronize`).
  Pipelining the readback one step behind is a throughput item, not correctness.
- **K3 — perf rows seeded 2026-10-26 (resolved, archived).** The 2026-09-14
  `perf` row (5.1 t/s) carried no evidence and predated the conversion; it was
  replaced through the harness with **`perf` 4.86 / `r3` 5.04 t/s**. The -5% is
  the numeric fixes changing the routed expert sets, not a regression (the old
  row had no protocol evidence to compare against). Numbers, power and TTFT:
  [`open-tasks/2026-10-26-kimi-audit-verification.md`](open-tasks/2026-10-26-kimi-audit-verification.md).
- **K4 — KV capacity.** `kKimiMaxSlots = 1024` against a 262 144 window; a
  long-context gate is out of reach on one 5090 (the KV cache is 512+64 bf16 per
  token per layer). Unchanged, documented.
