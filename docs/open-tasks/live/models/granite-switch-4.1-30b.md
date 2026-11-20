# Granite-Switch 4.1 30B — open items

### R10. Granite-Switch 4.1 30B (granite dense + 2 aLoRA) — ingestion landed 2026-10-31

Composed checkpoint `ibm-granite/granite-switch-4.1-30b` (revision 4.1.1):
Granite 4.1 30B dense decoder + `guardian-core` (rank 16, q/k/v/o) and
`decision` (rank 64, q/k/v/o + MLP) aLoRA adapters + control-token routing +
token exchange. Format tag `logestix-granite-switch`, `kType`
`granite-switch-4.1-30b`, converted in place with `lsxforge --granite_switch`.
Frozen geometry is 64 decoder layers (config says 66 = 64 + 2 switch cache
slots), 4096H / 32Q / 8KV / 128HD / 32768I / vocab 100355, tied embeddings,
multipliers 12 / 0.175 / 16.

Real-weight results (`linux-debug`, RTX 5090, 2026-10-31):

- **SelfCheck PASS** — Paris, 3 stress cycles, bit-exact logits hash
  `fdc632042c30919c`; 16.22 GiB resident.
- **FSI 3/3, CODE 3/3.**
- **DET 2/3 — DET-TEST-02 RED.** The prompt template's Expected Template is
  `CATASTROPHIC`; the **source** CPU-BF16 oracle itself answers `CRITICAL`
  (`tools/det_capture.py` logged
  `source answer differs from the prompt template: 'CRITICAL' != 'CATASTROPHIC'`).
  The converted engine answers `CRITICAL` too, deterministically across 3 runs,
  i.e. this is a source-vs-prompt disagreement, not a conversion divergence and
  not a near-tie a residual plane can move toward the template. The golden is
  left as captured (template-defined by design); the red row is recorded in
  `docs/SELFCHECK_MATRIX.md` and `docs/DET_SELFCHECK_FINDINGS.md`.
- **Adapter parity 3/3** against the CPU-BF16 oracle: base / guardian-core /
  decision first tokens match (`The` / `{"` / `A`), routing is 0 / 1 / 2, and
  the native chat renderer reproduces the source `input_ids` token-for-token for
  all three modes (`test_granite_switch_adapter_parity`).
- **Perf (AOCC release, 2048-token protocol): 69.28 t/s (perf) / 70.07 t/s
  (r3)**, TTFT 49 / 191 ms, 552.95 W average — `tests/perf/perf_baselines.json`.
  The perf row was re-confirmed after the numeric-policy fix below.
- **TC fast-math is disabled for this family (correctness).** The checkpoint's
  `embedding_multiplier=12` / `residual_multiplier=0.175` produces activations
  up to ~6e3 (measured at layer 17 in BF16). The NVFP4 TC path quantizes
  *activations* to E2M1 with E4M3 block scales; its representable ceiling is
  ~2.7e3, so long prompts clamp. Evidence: the 197-token tool prompt on the TC
  path decoded `x = x e^{x}` instead of the source `<tool_call>...`; the same
  prompt on the FP16-activation path reproduces the source tool call exactly
  (input ids token-for-token, `test_granite_switch_model`). `Infer` therefore
  pins `FastMathMode::kOff` for this family (still consulting
  `EffectiveFastMathMode(cfg_)` for the independence contract). kOff is not a
  throughput loss here: 70.07 t/s r3 vs 69.28 Auto on the measured rows.

Open items / declared scope:

- **DET-TEST-02** as above: red because the source does not follow the prompt's
  Expected Template. Closing it needs either a different prompt (shared prompt
  set — not a per-model change) or a source that answers `CATASTROPHIC`.
- **ASR / `<|audio|>` is out of scope.** The 470M Granite Speech encoder is a
  separate artifact and is not present in this checkpoint; the tokenizer's
  `<|audio|>` added token is carried but never decoded.
- **Long context is card-bound.** KV at the advertised 131072 tokens is
  ~34.4 GiB by itself (64 layers x 131072 x 1024 x 2 B x 2), which does not fit
  beside the 16.2 GiB weight set on a 32 GiB card. ~32k tokens is the practical
  arithmetic ceiling; the perf row is the canonical 2048-token protocol.
- The switch codebook is stored (raw F32) for fidelity but not uploaded: the
  reference coded-memory retrieval is exactly the prefix scan over control
  tokens (verified against `granite_switch.hf.switch.multi` on random
  sequences), and token exchange is the checkpoint's LUT.
