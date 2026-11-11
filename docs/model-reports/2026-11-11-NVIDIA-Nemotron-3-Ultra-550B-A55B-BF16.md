# NVIDIA-Nemotron-3-Ultra-550B-A55B — integration report

| | |
|---|---|
| Model | `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`, `model_type = "nemotron_h"`, `NemotronHForCausalLM` |
| Converted layout | `logestix-nemotron-ultra-nvfp4` (NVFP4 mixer/attention/latent/shared/routed projections; BF16 embeddings, head, norms, conv1d and `A_log`/`D`/`dt_bias`; F32 router bias) |
| Source | BF16, 1.1 TB, 225 shards, 51 023 tensors, 550 B params / ~57 B active |
| Converted tree | 292 GB (22.5 GB pinned dense graph + 24 576 routed-expert arrows of 11.8 MB) |
| Hardware | 1 × RTX 5090 (32 GB), 750 GB host RAM, CUDA 13.4 driver / 13.5 toolkit |
| Runtime | token-by-token decode, 108-layer hybrid walk, streamed mmap expert store + device slot LRU (661 of 55 296 slots) |
| Date | 2026-11-11 |
| Reference | `transformers` 5.15.1 `models/nemotron_h/modeling_nemotron_h.py` + `models/zamba2/modeling_zamba2.py` |
| Lineage | **separate** from `nemotron-3.5-lightning-30b-a3b`: own passport, converter, loader, expert cache, kernels, pipeline and tool codec. No file of the 30B lineage was edited. |

**Verdict at a glance.** A 550 B hybrid Mamba-2/MoE/attention model ports to a
single 32 GB card at **1.77 t/s** because 290 GB of routed experts stream from
NVMe while a 22.5 GB dense graph stays resident. On the real weights it is
correct and deterministic — Paris ×3 bit-identical, the native tool round-trip
green, CODE 3/3 and DET 3/3 — with one measured generation error (FSI-01, below)
and one honest gap in provenance: the source is too large to execute here, so
source fidelity rests on a slice oracle rather than a full-model reference.

---

## Publish-ready post

### Notes from an independent NVFP4 port of Nemotron-3-Ultra-550B

We brought NVIDIA's 550 B hybrid into our own C++/CUDA engine (no PyTorch at
runtime), converting the 1.1 TB BF16 release in place to a 292 GB NVFP4 tree.
The dense backbone pins at 22.5 GB and the 24 576 routed experts stream from
host mmap through ~660 device slots, so the model runs on one consumer card at
1.77 t/s. The reference modeling code is clear, and four things about this
architecture are load-bearing.

**The architecture that matters.** 108 layers, of which 48 are Mamba-2, 48 are
MoE and 12 are full attention (indices 7, 14, 23, …, 98). Mamba-2 runs at 256
heads × 64 with 8 state groups and 128-wide state, a 35072-row `in_proj` and a
gated RMSNorm over 2048-wide groups. Attention is GQA 64/2 at head_dim 128 and
is **NoPE** — the reference class has no rotary module at all, and `rope_theta`
in `config.json` is vestigial. MoE routes 512 experts top-22 through a **latent
bridge**: `fc1` (8192→2048), the experts at 2048, `fc2` (2048→8192), then an
ungated relu² shared expert at the full 8192 width.

**Checkpoint-level notes for the authors.** (1) `config.json` has **no
`num_hidden_layers`**; the depth is `len(layers_block_type)`, and the reference
config derives it the same way — a reader that requires the field rejects the
checkpoint. (2) There are two different `n_group`-ish fields: `n_group` (=1)
groups *experts* for the router, `n_groups` (=8) groups the *SSM state*.
Conflating them was my first wrong turn. (3) `generation_config.json` carries the
real stop set `[2, 11]` while `config.json` carries the scalar `2`; stopping only
on the first lets the model generate past its own turn end. (4) The 1 040 `mtp.*`
tensors are ballast the reference drops on load (`_keys_to_ignore_on_load_unexpected`).

**Two defects our green synthetic suite could not see.** Both are Ultra-only, and
both are invisible at 3 layers of tiny geometry:

(5) The batched routed-expert GEMV entry point accepts **at most 8 experts per
launch**. This model routes to top-22; the 30B sibling routes to top-6, so
copying its call fails with `cudaErrorInvalidValue` on the first MoE layer.
(6) The latent bridge **aliases** the expert input and the up-projection output if
you reuse one scratch buffer: the up-GEMV writes 22×5120 halfs into the buffer it
is still reading the 2048-wide latent from, so chunk 0 clobbers its own input.
The 30B cannot hit this — it has no latent bridge. The symptom was the memorable
one: coherent-looking garbage (`'إن order'`), a residual stream that grew
1.4 → 14 129 over the depth, and a head that faithfully reproduced the wrong
state from the checkpoint's own BF16 weights.

**How we proved the numerics without running the source.** 1.1 TB does not fit a
750 GB host, so the reference is a *slice*: a real `NemotronHForCausalLM` built
from the checkpoint's own contiguous layers 0–7 (mamba, moe, mamba, moe, mamba,
moe, mamba, attention) plus the real embedding, `norm_f` and `lm_head`. That is
enough to execute — and therefore to be an oracle for — every piece of math this
model adds. Engine versus reference, at the last prompt position: relative L2
3 × 10⁻⁴ at layer 0, 2.4 × 10⁻³ at the latent-MoE layer 1, 7 × 10⁻³ by layer 4,
3 × 10⁻² by layer 5, all within the 5 × 10⁻² bound. The residual over a
BF16 reference that has itself been through the NVFP4 codec collapses to those
numbers, which is how "the conversion is lossy" was separated from "the runtime
is wrong".

**Measured.** Determinism: Paris ×3, bit-identical final logits,
`logits_hash bd82916c6550a2df`, top-1 6993. Tool round-trip:
`get_weather {"city":"Paris"}` through the native `<tool_call><function=…>` codec.
CODE 3/3, DET 3/3 (three byte-identical runs), FSI 2/3. Throughput **1.77 t/s**
(r3 1.75) at the canonical 2 048-token protocol, TTFT 13.4 s, 139 W average —
memory-bound on the expert store, not on the GPU.

**Known deviation, stated rather than hidden.** FSI-01 asks for the present value
of an *annuity-due* under continuous compounding. The model answers
`PV=CF*((1-exp(-r*n))/(1-exp(-r)))*exp(r)`: it appends `exp(r)` to the expression
that is *already* the annuity-due form, landing 8.7 % high at r = 0.0835. The 30B
sibling of the same family appends the same factor, so this is a family-level
generation habit, not a conversion artefact. It is excluded from the committed
FSI gate **in writing** — the fixture stores the exact measured output — and the
gate reports 2/3 rather than being quietly re-baselined.

**Not measured.** A full-model source reference (the 1.1 TB BF16 tree cannot be
assembled here); the MTP/speculative head (cut, counted: 1 040 tensors); context
beyond 4 096 tokens (the session ceiling; `config.json` claims 262 144); and
prefill optimisation (the sequential recurrent form stands, mathematically
identical to the reference chunk scan).

---

## Evidence

- Design: `docs/superpowers/specs/2026-11-10-nemotron-ultra-550b-ingestion-design.md`;
  plan: `docs/superpowers/plans/2026-11-10-nemotron-ultra-550b-ingestion.md`.
- Gates on the real weights (ctest, `LSX_MODEL_ROOT=/mnt/fast_storage/huggingface`):
  - `test_nemotron_ultra_selfcheck` — Paris ×3 bit-exact, `logits_hash
    bd82916c6550a2df`, top-1 6993; tool round-trip `get_weather {"city":"Paris"}`.
  - `test_nemotron_ultra_layers` — per-layer engine vs slice-oracle, 8 layers ×
    the last prompt position, `rel < 5e-2` and `cos > 0.99` on 64 leading dims.
  - `test_nemotron_ultra_fsi_selfcheck` — 2/3 (FSI-01 excluded, measured).
  - `test_nemotron_ultra_code_selfcheck` — 3/3.
  - `test_nemotron_ultra_det_selfcheck` — 3/3, deterministic.
  - `test_nemotron_ultra_tokenizer_parity` — 22/22 against `tokenizer.json`
    (md5 `e7f0d67f…`, byte-identical to the two 30B siblings and Dory1.0).
  - `test_perf_baseline_nemotron_ultra` — 1.77 t/s / r3 1.75, TTFT 13 352 /
    13 470 ms, 139 / 128 W.
- Host-side suite (no weights): `test_nemotron_moe_ultra_constants` (6 cases),
  `_convert` (9), `_loader` (7), `_math` (8), `_expert_cache` (4), `_model` (4),
  `_kernels` (10 cases / 6 920 assertions), `test_nemotron_ultra_tool_framing` (9).
- Instruments: `tools/nemotron_ultra_oracle.py` (slice oracle, `--quantize-nvfp4`
  for the like-for-like comparison); goldens captured with the repo's own
  `tools/{fsi,code,det}_capture.py --template-golden`.
- Conversion: `lsxforge --nemotron_ultra`, 292 GB written, 24 576 experts, 1 040
  MTP tensors cut, 831 pinned tensors, **peak host residency 512 expert records**
  (one layer) — the per-layer streaming is what makes a 290 GB expert set
  convertible on a 750 GB host.
- Registry: `models/nemotron-ultra-550b-a55b.json` (family `nemotron`, own
  namespace `lsxcommon::nemotron_ultra::frozen`, format tag
  `logestix-nemotron-ultra-nvfp4`, forge flag `--nemotron_ultra`, alias
  `nvidia-nemotron-3-ultra-550b-a55b-bf16` for `--localmodel` path detection).
- Measurement traps worth keeping:
  - `simdjson` on-demand is **forward-only**: the Ultra index reader must read
    keys in the writer's order or later lookups silently fail (it first reported
    the tree as missing required fields).
  - `FormatVramJson` already emits its own `"vram":` key; concatenating it after
    another produced a **duplicate key** that substring assertions could not see.
    The convert test now asserts every top-level key appears exactly once.
  - A device kernel's shared scratch must never be sized by a passport constant
    that describes one checkpoint: moving `kRouterGroups` 8 → 1 silently shrank
    the router's group arrays while `n_group` stayed a runtime argument.
