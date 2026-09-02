# Qwen3.8-Flash-Next Ingestion — Round 1 Status

Date: 2026-09-02
Branch: `feat/model-Qwen3.8-Flash-Next`
Spec: `docs/superpowers/specs/2026-09-02-qwen3.8-flash-next-ingestion-design.md`
Plan: `docs/superpowers/plans/2026-09-02-qwen3.8-flash-next-ingestion.md`

## What landed (Tasks 0-8 complete, 10/10 planned commits)

- Frozen geometry passport (`qwen_moe_exp_constants.h`): 48 layers,
  GDN 16x128 QK / 48x128 V, QSA 24Q/2KV HD256, indexer 4Q/1KV budget 2048
  compress 4, MoE 512 experts top-10 I=640, hc_count=4 lowrank=320,
  PLE layer 2 (ngram 3, 16 heads, head_dim 160, 128 shards), vocab 248320.
  Naming guard accepts `qwen_moe_exp_*` (exp marker between type and role).
- Converter (`ConvertQwen38FlashNextModel`, lsxforge `--qwen38flashnext`):
  full tensor classification (50 kinds), NVFP4 policy, PLE splitmix64
  multipliers + prime head table verified bit-exact against the checkpoint
  (stored layer_multipliers and ngram_heads_* matched the recomputation),
  expert split 48x512, mtp/ple/visual accounting. Real model converted:
  336 GiB BF16 -> 163 GiB (64G experts NVFP4, 96G PLE shards, 2.5G pinned).
- Loader: pinned dense upload (hc norms BF16, projections NVFP4), expert
  store 48x512 slabs (1.35 GiB/layer), unified `gpu::ExpertDeviceCache` LRU
  (9071/24576 slots on RTX 5090), PLE shards via zero-copy Arrow mmap views.
- CUDA kernels: hc GatedResidual (F32 core + NVFP4 production wrapper with
  (1+w) group-RMSNorm), hc-scatter, QSA indexer (degenerate fast-path +
  block top-k), masked GQA attention, PLE inject kernel, router top-10
  (reused parameterized etalon kernel).
- Model: `Qwen38FlashNextModel` registered as `qwen3.8-flash-next`,
  ValidateGeometry (16 fields vs frozen), governor leases, 48-layer Infer
  loop wired through RunExpGeneration (PLE -> hc -> GDN|QSA -> scatter ->
  hc -> MoE top-10 -> scatter -> global mixer -> lm_head).
- Tests: 6 suites; 5/6 green (convert/loader/kernels/model/naming-guard),
  selfcheck harness runs SKIP without the model.

## Debugging round (real-model bring-up)

Fixed on the way to a clean forward pass:
1. `d_qgate` null in the QSA branch -> invalid argument.
2. qwen4_exp has NO input/post layernorms (hc replaces them) — loader
   validation relaxed.
3. Inject-weights scratch buffer undersized (hc*4 vs hc*n*4).
4. Router argument order (H,E,K vs seq,H,E,K).
5. Shared-combine `d_gate_val` nullptr -> illegal access.
6. Arena float slice sizing for PLE rows.
7. `lin_idx`/`full_idx` not reset per step -> OOB KV writes from step 2.
8. hc up-gate scratch aliased the up WEIGHT buffer (dequant output overwrote
   the weights mid-kernel) -> NaN cascade.
9. hc inject weights: stack-buffer async copy -> garbage; replaced with real
   block_inject dequant (2*sigmoid(W_inj @ normed / hc)).
10. PLE host-gather implemented (splitmix64 hash + prime mod + mmap row read).

Current state: end-to-end generation is numerically STABLE (no NaN), logits
bounded [-0.54, 0.55], but the SelfCheck answer is `blo!` instead of
`Paris`. Remaining accuracy gap is in the bit-exact parity of the new
architecture blocks (hc GatedResidual subtleties, GDN at nv=48, PLE conv
state) vs the HF reference.

## Next steps (accuracy round 2)

1. Python parity harness: run HF modeling_qwen4_exp on the first token and
   dump per-stage tensors (embed, hc mixed, GDN out, MoE out, logits);
   diff against device probes to localize the first divergence.
2. QSA indexer: wire the real micro-block mask (currently full causal).
3. PLE conv state: verify the dilated short-conv window update order.
4. hc (1+w) group-RMSNorm: confirm the group layout (stream-major vs
   element-major) against HF unflatten semantics.

## Perf note

Decode speed measurement is deferred until correctness lands; the LRU cache
and TC-batch etalon path are in place for the tuning round.
