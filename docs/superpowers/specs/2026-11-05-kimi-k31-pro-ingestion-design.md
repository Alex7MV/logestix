# Kimi-K3.1-Pro ingestion — design

Date: 2026-11-05
Source: `/mnt/fast_storage/lsx_model/moonshotai/Kimi-K3.1-Pro` (1.5 TB, 96 shards,
497 340 tensors)
Registry: `kimi-k3.1-pro` (family `kimi`, variant `pro`, files `kimi_moe_pro_*`)
Format tag: `logestix-kimi-k31-pro-nvfp4`

## 1. Stage 0 triage (measured from the checkpoint)

### 1.1 Classification

`model_type = kimi_k31`, `architectures = [KimiK31ForConditionalGeneration]`, text
tower `KimiLinearForCausalLM` (`kimi_linear`) — the **same contour as Kimi-K3**
with one delta: the 24 full-attention Gated-MLA layers become **Gated DSA**
(DeepSeek Sparse Attention: the same gated NoPE MLA plus a Lightning Indexer
that selects `index_topk = 2048` keys per query).

Measured tensor deltas against `Kimi-K3/model.safetensors.index.json`:

* 0 tensors removed; 497 220 of 497 340 names are byte-identical.
* +120 tensors = 24 DSA layers × 5, all under `layers.{3,7,...,91,92}.self_attn.indexer.`:
  `wq_b [4096,1536]` (from the 1536-wide normalised q-lora),
  `wk [128,7168]`, `k_norm.weight [128]` + `k_norm.bias [128]` (**LayerNorm with
  bias**, `index_k_norm_type = "layernorm"`), `weights_proj [32,7168]`.
  All BF16; `index_rope_dim = 0` (no rotary on the indexer), 32 heads × 128.
* Geometry, MoE, KDA, AttnRes, SiTU, vocab/ids, tokenizer (`tiktoken.model`,
  2 795 286 bytes, identical), XTML framing: identical to K3.

**The checkpoint ships no modeling code.** `configuration_kimi_k3.py` and
`encoding_k3.py` are present, `modeling_kimi_k31.py` is not (K3 shipped
`modeling_kimi_k3.py` + `modeling_kimi_linear.py`). The indexer's arithmetic is
therefore reconstructed from (a) the tensor shapes, (b) `dsa_config`, and (c) the
Lightning-Indexer prior art already in this repository (`deepseek_moe_pro` CSA,
`mistral_moe_large4` DSA). This is the one semantic risk of the ingestion and is
recorded as debt, not as a verified parity claim.

### 1.2 Geometry (frozen, restated — the passport may not include K3's header)

Identical to Kimi-K3 (see `2026-11-03-kimi-k3-ingestion-design.md` §1.2): 93
layers (layer 0 dense MLP, 1..92 LatentMoE), hidden 7168, 69 KDA + 24 gated MLA
at 0-based `{3,7,...,91,92}`, AttnRes block 12, 896 routed experts top-16 with a
3584 latent + 2 shared experts (I=6144), SiTU-GLU (β 4 / linear β 25), vocab
163 840, bos 163 584, stop 163 586 (+ 163 585), pad 163 839, weight-form RMSNorm
eps 1e-5, `routed_scaling_factor 1.0`, `mla_use_nope`, `mla_use_output_gate`.

New frozen fields:

| field | value |
|---|---|
| `kDsaLayers` | the same 24 layers as `kMlaLayers` (every full-attention layer) |
| `kIndexHeads` | 32 |
| `kIndexHeadDim` | 128 |
| `kIndexTopK` | 2048 |
| `kIndexRopeDim` | 0 |
| `kIndexKvLen` | 7168 (indexer reads the attention input) |
| `kIndexQLora` | 1536 (indexer reads the normalised q-lora) |
| indexer norms | LayerNorm (mean-centred, weight + bias), eps 1e-5, no rotary |
| indexer weights | **never quantized** (BF16): they decide *which* keys are visible |

### 1.3 Feasibility on this machine

RTX 5090 32 GB, 750 GB RAM, 3.5 TB free. K3 measured 23 GB pinned + 8.9 GB
streamed plane + 1.4 TB routed experts (82 432 files, ~20 min of pure I/O).
K3.1-Pro adds 0.36 GB of pinned indexer weights and a 25 MB indexer-k cache at
4096 context — the memory plan is unchanged, the model still fits.

## 2. Approaches considered

**A. Full DSA with a dormant-at-`ctx <= topk` fast path (chosen).** The indexer
is converted, pinned, scored, and used to restrict attention whenever
`n_pos > index_topk`; at `n_pos <= topk` every causal key is inside the top-k, so
the driver passes no selection list and the existing dense MLA kernel runs
unchanged — *bit-identical* to K3, which is the reference path for the canonical
2048-token protocol. Cost: ~700 lines of converter/loader/kernel/pipeline work.

**B. Store-not-apply** (the `step` §R9 precedent): load and account the indexer,
always run dense attention. Cheaper, but the 1M-token context the model exists
for is then wrong, and the debt is larger.

**C. Reject / discard the indexer tensors.** Not viable: the converter classifies
every tensor (an unknown name is an error), and a tree that cannot be rebuilt
from the checkpoint is not a conversion.

A was picked: the gates cannot see the difference (top-k is a no-op at ≤2048),
so the only honest way to spend the effort is on the path the gates cannot cover.

## 3. Design

### 3.1 Files

```
libs/lsxcommon/src/
  kimi_moe_pro_constants.h        frozen passport, ns lsxcommon::kimi_pro::frozen
  kimi_moe_pro_convert.{h,cpp}    lsxforge --kimi_k31_pro entry + variant gate
  kimi_moe_pro_model.{h,cpp}      IModel registration "kimi-k3.1-pro"
  kimi_moe_k3_constants.h         (extended: variant/profile declaration only)
  kimi_moe_k3_convert.cpp         shared converter, +KimiVariant + indexer path
  kimi_moe_k3_load.{h,cpp}        shared loader, + optional indexer weights
  kimi_moe_k3_kernels.{h,cu}      shared kernels, + indexer k/q/score/top-k
  kimi_moe_k3_inference_pipeline.cu shared driver, + DSA branch
apps/lsxforge/                    --kimi_k31_pro flag (cli + main)
tests/                            test_kimi_pro_* + guards
```

Shared-file extension follows the `minimax-m3.1-flash` precedent (variant gets its
own constants + convert entry + model; the parent's loader/kernels/pipeline are
reused). The variant discriminator is `config.text_config.model_type`
(`kimi_k3` vs `kimi_k31`) **and** the presence of the indexer tensors, both ways:
a K3 checkpoint converted with `--kimi_k31_pro` fails (missing indexer), a Pro
checkpoint converted with `--kimi_k3` fails (unknown `indexer.*`), and the
written `model.index.json` carries `"model": "kimi-k3.1-pro"` + `"dsa_layers"`,
which the loader cross-checks against the passport.

### 3.2 Indexer arithmetic (the reconstructed part)

With `h` = the layer's attention input — the **post-`input_norm` hidden**, i.e.
the vector the gated-MLA projections themselves consume (the same input the
DeepSeek DSA contour feeds its indexer; ruling recorded in the SDD ledger:
`dsa_config` names the indexer's tensors but not its input, and the attention
projections' input is the only reading consistent with `qr` being the normalised
q-lora output) — and `qr` = that same normalised q-lora vector the MLA layer
already computes:

```
q[h,:] = (wq_b @ qr)[h*128 : (h+1)*128] * 128^-0.5      h = 0..31
k[s,:] = LayerNorm(wk @ h_s)                            (weight + bias, eps 1e-5)
w[h]   = weights_proj @ h_t * 32^-0.5
score[t,s] = sum_h w[h] * relu(q[h,:] . k[s,:])          s <= t (causal)
topk       = the kIndexTopK largest scores; ties broken by ascending position
```

`k[s,:]` is cached per position per DSA layer (the indexer key cache, FP16,
25 MB at 4096) because the attention input of an earlier position is not
retained. `q` and `w` are per query token. Selection is skipped when
`n_pos <= kIndexTopK`, which is the exact identity.

Open points of the reconstruction (all recorded as debt, none of them reachable
at the canonical 2048-token protocol, where selection is dormant):

* `index_skip_topk_offset = 1` and `index_topk_freq = 1` are not implemented as
  knobs: freq 1 is "select on every DSA layer", and with the offset's value of 1
  the only reading that changes nothing at `n_pos > topk` is "the selection
  starts at the second position", which the `n_pos <= topk` bypass already
  covers for every prompt up to 2049 tokens. A different value would need the
  source to define it.
* The selection does **not** force the current position into the top-k. DeepSeek
  DSA guarantees causality by construction (the scores are causal), and forcing
  a position would be an invented rule; at `n_pos > topk` the current position's
  own score is normally the largest, and the pipeline pins the exact set the
  scores imply.
* The `dsa_layers` table is required to be exactly the MLA table: the source
  never indexes a KDA layer (there is no `indexer` module under `self_attn` of a
  KDA layer) and no full-attention layer lacks one.

### 3.3 Kernels

In the shared `kimi_moe_k3_kernels.{h,cu}` (one more TU is not nameable under the
naming guard: `indexer` is not an allowed subsystem token):

* `RunKimiK3IndexerKey` — `k = LayerNorm(wk @ x)`, one block per position, FP32
  mean/var reduction, writes the FP16 cache row.
* `RunKimiK3IndexerQuery` — `q = (wq_b @ qa) * 128^-0.5` and
  `w = (weights_proj @ x) * 32^-0.5`.
* `RunKimiK3IndexerScores` — `score[s] = sum_h w[h] * relu(q[h]·k[s])`, one block
  per position group, deterministic block reductions, no `atomicAdd`.
* `RunKimiK3IndexerTopK` — exact selection, deterministic (value desc, index asc),
  single block, `k <= n_pos <= max_context`.
* `RunKimiK3MlaAttention` — one new **optional** `(const int32_t *sel, int n_sel)`
  pair; `sel == nullptr` keeps today's ascending-`0..n_pos` loop verbatim.

### 3.4 Tests (TDD: each is written red before its code)

| test | label | assertion |
|---|---|---|
| `test_kimi_pro_constants` | fast | passport identity, DSA table == MLA table, indexer shapes, `kIndexTopK` |
| `test_kimi_pro_convert` | convert, 120 s | synthetic E2E: format tag, `dsa_layers` in the index, indexer tensors stored BF16 verbatim, missing indexer → error, indexer on a KDA layer → error, K3 checkpoint rejected |
| `test_kimi_pro_loader` | fast | topology parse; indexer weights uploaded with the frozen shapes; null when absent; frozen cross-check |
| `test_kimi_pro_indexer` (`lsx_add_cuda_test`) | inference | key LayerNorm vs host ref; score vs host ref (weights and scale included); top-k exactness + tie-break; `sel = [0..n-1]` bit-identical to the dense kernel; masked attention == host reference over an arbitrary selected set |
| `test_kimi_pro_model` | fast | registry/`kType`, frozen mismatch fails `Load`, lifecycle rows |
| `test_kimi_pro_selfcheck` | inference, SKIP 4 | framed Paris, ≥3 bit-exact cycles |
| `test_kimi_pro_{fsi,code,det}_selfcheck` | inference, SKIP 4 | template-derived goldens (source is unrunnable here) |

Guard registries extended in the same commit: `kPassports`, `consumers`,
`kLifecycle`, `Suites()`, the tool-verdict table, the golden case, ctest labels.
Naming needs no change (`pro` is already a variant token).

### 3.5 Real-weight plan

1. Convert in place: `<source>/logestix/` (~1.56 TB, expert I/O in the
   background while the rest of the work continues).
2. `lsxbenchmark --selfcheck` (Paris ×3) and the framed control prompt.
3. Tool round-trip `get_weather {"city":"Paris"}` through the native XTML codec.
4. FSI/CODE/DET: template-derived goldens, run time-boxed; a run that cannot
   finish is recorded as a measured debt, never as a pass.
5. Perf rows seeded **through the harness** at the canonical 2048-token
   protocol (`perf` + `r3`), or recorded as unseeded with the measured reason
   (~2 s/token, host-store bound).
6. Report `docs/model-reports/2026-11-05-Kimi-K3.1-Pro.md`, `AGENTS.md`
   structure entry, `SELFCHECK_MATRIX.md` row, `OPEN_TASKS.md` §R17 (indexer
   semantics inferred; >2048 unvalidated).

## 4. Non-goals

* Vision tower / `MoonViT-V2` + `mm_projector` (401 M) — stripped and counted.
* 1M-token allocation: `kDefaultMaxContext` stays 4096 (the DSA branch is
  exercised above 2048 by the kernel test, not by the default session budget).
* Quantizing the indexer or the router: both change *which* experts/keys are
  chosen, not how precisely a value is carried.

## 5. Risks

1. **Indexer semantics are inferred** (no modeling code in the checkpoint). The
   structure is fully determined by the tensors; the three constants that are not
   (`128^-0.5` on q, `32^-0.5` on the head weights, LayerNorm eps) follow the
   Lightning-Indexer prior art in this repo. Mitigation: the arithmetic is
   isolated in three testable kernels with host references, the no-op identity at
   ≤2048 is bit-exact by construction, and the residual risk is named in
   `OPEN_TASKS.md`.
2. **Conversion cost**: 1.56 TB of I/O on a disk with 3.5 TB free. Mitigated by
   starting it as soon as the converter builds and by the fact that the expert
   path is pure I/O with no GPU or host-RAM spike.
3. **Perf**: unseeded baseline; the decode is host-store bound by construction
   (27 GB of routed experts per token), exactly as K3.
