# Kimi-K3 — open items

### R15. Kimi-K3 (2.8T / ~104B active) — ingestion landed; SelfCheck green, parity runs pending (2026-11-03)

Branch `feat/kimi-k3`. A new `kimi` variant (`k3`, `kType = kimi-k3`, format
tag `logestix-kimi-k3-nvfp4`): 93 layers (dense SiTU MLP on layer 0, MoE
elsewhere), 69 **KDA** linear-attention layers interleaved with 24 **gated
NoPE MLA** layers (0-based `{3,7,…,91,92}`), **AttnRes** block 12, **Stable
LatentMoE** (hidden→3584 latent + RMSNorm, 896 experts top-16, 2 shared experts
at 6144), SiTU-GLU (β=4, linear β=25), 7168H, vocab 163840.

**Measured on real weights** (1.5 TB MXFP4 source → 1.5 TB NVFP4 tree: 23 GB
pinned + 8.9 GB streamed plane + 1.4 TB routed experts; RTX 5090, CUDA 13.4):

| gate | verdict |
|---|---|
| SelfCheck (XTML-framed) | PASS x3 cycles bit-exact: `The answer is Paris.`, `logits_hash=5c1e227265d6a27b` on all three, top-1 1008 |
| tokenizer parity | 23/23 against the checkpoint's `tiktoken.model` (`tokenizer_kind 2`) |
| tool codec | byte-exact against `encoding_k3.py` frames (synthetic tokenizer) |
| FSI / CODE / DET | goldens committed (template-derived); **not yet run** on real weights |
| decode rate | **2.13 s/token (perf) / 2.04 (r3)** — harness rows `0.47` / `0.49` t/s at 2048 tokens, seeded 2026-11-06. Earlier figures on this row were log-derived prose: 2.96–2.97 at the shipped configuration (expert window 65 536, no eager warm, 64-token cycles) and 6.95 at the old 512-slot window against a 1472-expert working set — see §R17 P5 and its archive entry `open-tasks/2026-11-05-kimi-k3-decode-window-and-eviction.md` |
| perf row | **seeded 2026-11-06**: `perf` 0.47 t/s, `r3` 0.49 t/s, 2048 tokens, TTFT 131.9 / 102.1 s |

**Open, explicitly.**

- K1. FSI/CODE/DET were not executed to completion. At the measured decode rate
  (see K3 below) the budgets (512–2048 tokens × 3 prompts, DET × 3 stress runs)
  are multi-hour runs; the goldens and tests are in place and SKIP (4) until
  then. Do **not** read the matrix row as green.
- K2. The parity goldens are **template-derived** (`tools/kimi_k3_template_goldens.py`):
  the 1.5 TB source needs `fla-core` and a multi-GPU runner, so no
  model-defined golden can be captured here. `expected` is the prompt set's own
  Expected Template; the frame is the checkpoint's own XTML
  (`encoding_k3.py`), and `answer_start_token_id` cuts at `<|sep|>` (163589).
  Kimi-K3 answers inside its thinking channel, so a text-based cut would read
  the reasoning as the answer — the same trap as Mistral-Large-4 (L1/L3).
- K3. Perf rows are unseeded. The decode path streams 16 experts × 92 layers
  (~27 GB) per token from the 1.4 TB host store plus the 9.5 GB latent/shared
  plane; expect ~0.2–0.4 t/s. Seed through the harness at the canonical
  2048-token protocol before quoting a number.
- K4. `A_log` is read **per channel** (the checkpoint ships 128 entries =
  `head_dim`), while the `fla` reference indexes it per head (96). The
  checkpoint shape is authoritative and SelfCheck is green with it, but a
  source-side comparison (impossible here) is the only definitive check.
- K5. The latent/shared plane is streamed (~103 MiB/layer/step) because pinning
  it needs ~34 GiB and does not fit this card. There is deliberately no
  "pin it all" knob: it cannot succeed here, and a knob that only works on
  unseen hardware is untested code.
- K6. Context is capped at 4096 (`LSX_KIMI_K3_MAX_CONTEXT`). The 1 M-token
  window the release targets needs ~84 GB of latent KV at 24 layers and is
  out of scope for this card.
- K7. A load is flaky under VRAM pressure: with ~8 GiB free before `Load`, the
  24.5 GB pinned graph fails mid-graph (`layer N ... upload failed`, logged).
  The model needs a card that can hold the pinned set plus ~2 GiB headroom; a
  clean "not enough VRAM" pre-check is a follow-up.
