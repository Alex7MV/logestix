# Nemotron-3.5-Lightning-30B-A3B (NVFP4) — open items

### R13. Nemotron-3.5-Lightning-30B-A3B-NVFP4 — lifecycle verified, **NoPE defect fixed** (2026-11-01)

Branch `feat/nemotron-nvfp4-lifecycle`. The family (BF16 passport + NVFP4
sibling) was already ingested; this session **ran the lifecycle on real weights
for the first time** and found a real semantic defect the green suite could not
see.

**The defect: full-attention layers were wrong — they are NoPE.** The
checkpoint's own reference class (`transformers` 5.15.1
`NemotronHAttention`) has **no rotary module**: it never builds cos/sin, has no
`rotary_emb`, and ignores the `position_ids` it is handed. Position in this
hybrid comes from the 23 Mamba-2 layers; `rope_theta` /
`partial_rotary_factor` in `config.json` are vestigial. The engine applied
full-dimension `rotate_half` RoPE to q and k anyway, so every attention layer
injected a position-dependent rotation the weights were never trained with.

Why the old suite hid it: RoPE is the identity at `pos = 0` (cos 1, sin 0), and
the Paris SelfCheck is a ~7-token prompt — the model is robust enough to answer
it with a mild perturbation. The corruption is context-length dependent:

| pos | per-layer relL2 vs the BF16 reference, engine-with-RoPE |
|---|---|
| 0 | 0.09 (pure 4-bit) |
| 3 | 0.28 |
| 43 | 0.60 |
| 284 | **1.20** (cos 0.38 at L51), jumps of 2x at every attention layer |

Symptoms: any tool-framed prompt (~300 tokens) produced HTML-tag soup and no
tool call; the real-weight tool round-trip failed. Fix: delete the two
`RunNemotronRope` calls and the now-dead kernel + its unit test; the NoPE fact
is recorded in both constants headers and next to the attention launch.
After the fix, with the same converted tree:

- per-layer relL2 vs the BF16 tree reference at pos 284: **0.003 (L0), 0.011
  (L5), 0.23 (L51)**, cos ≥ 0.973 — pure quantization accumulation;
- greedy generation matches the source reference **24/24 tokens** on the tool
  prompts (was 0/24), 24/24 on the plain chat prompt, 13/24 on one free-form
  prompt before a near-tie;
- real-weight tool round-trip: `get_weather {"city":"Paris"}`.

**The instrument that found it (new, keep it).**
`tools/nemotron_nvfp4_oracle.py` undoes the ModelOpt encodings (W4A16_NVFP4
`weight`/`weight_scale`/`weight_scale_2`, per-tensor FP8, BF16 passthrough) with
torch and assembles a CPU BF16 `NemotronHForCausalLM` — the source oracle this
family never had. `--from-tree`-equivalent evidence was produced by decoding the
converted `logestix/pinned/*.arrow` back to BF16 (same script's codec): that
tree reference reproduces the *source* reference token-for-token, which is how
"the weights are fine, the runtime is wrong" was separated from "the conversion
is lossy". 8 s per 16-token CPU generation on 160 cores, so it is a usable gate
instrument, not an experiment.

**Gates as measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**
- SelfCheck **PASS**: Paris ×3 bit-exact, `logits_hash fe3b05c75a2033ba`
  (changes from `e5543669c00386f6` — the NoPE fix legitimately moves the logits).
- Tokenizer parity **22/22** (`test_nemotron_moe_tokenizer_parity`, new;
  `tools/nemotron_tokenizer_capture.py` + `tests/fixtures/nemotron/`), kind 9.
- Tool calling (real weights): green, above.
- Perf rows re-seeded through the harness at the canonical 2048-token protocol:
  **perf 127.55 t/s / r3 120.39 t/s**, TTFT 84 / 89 ms, 242 / 241 W. RoPE was
  negligible, so this is a re-seed, not a speed-up.
- FSI/CODE/DET remain declared debt, now with measurements instead of a
  hypothesis: the **source** answers FSI 3/3 per the Expected Template
  (`PV=CF*((1-exp(-r*n))/(1-exp(-r)))`, `Balance_Alpha=…`,
  `Drawdown>Threshold`), so a golden is capturable — but the converted tree
  does not reproduce them (FSI-01 appends `*exp(r)`, FSI-02/03 emit fenced /
  repeatedly-closed bodies), and the source echoes the prompt body on
  CODE-TEST-01 and DET-TEST-01/02, so those two sets have no source-faithful
  answer to record.

**Open, explicitly.**
- N1. FSI parity is red on the converted tree. The measured mechanism is the
  4-bit body (attention projections sit at 9.5% relL2 vs source; the vendor
  kept them BF16 and the Mamba projections FP8). The documented lever is the
  residual plane (attn + Mamba projections at ~8 effective bits, routed experts
  left 4-bit) — the gemma-4-26b recipe — at roughly +1.1 GiB resident and a
  doubled attn+Mamba read. Not attempted here: it changes the tree every
  existing baseline was seeded on and the family is a perf-frontier row.
- N2. The mamba expert store probes `layer_0..` / every non-MoE layer at load
  (3712 failed mmap opens, debug-logged) because `ExpertStore::Load` does not
  consult `moe_index_of_layer`. Harmless (entries stay empty, the compact
  device id map is correct) and ~30 ms per load; noted, not fixed.
