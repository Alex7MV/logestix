# Nemotron-3-Ultra-550B-A55B — open items

### R19. Nemotron-3-Ultra-550B-A55B — lifecycle complete, two Ultra-only defects found (2026-11-11)

Branch `feat/nemotron-ultra-550b-ingest`. A **separate lineage** from
`nemotron-3.5-lightning-30b-a3b` (operator decision): own frozen passport, converter,
index/loader, expert store, kernels, pipeline and tool codec under
`nemotron_moe_ultra_*` / `nemotron_ultra_*`. No 30B file was edited. Report:
`docs/model-reports/2026-11-11-NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.md`.

**Geometry reality vs the first read of `config.json`.** Three things a reader gets
wrong, all caught by the real-weight probe and not by the green synthetic suite:

- `num_hidden_layers` is **absent**; depth is `len(layers_block_type)` (=108),
  exactly as `NemotronHConfig` derives it.
- `n_group` (routing) is **1**; the 8 in the same file is `n_groups` (SSM state
  groups). The passport initially froze `kRouterGroups = 8` from the wrong field.
  The grouped-router implementation stays (it is tested at n_group 1, 4 and 8) but
  this checkpoint's group stage is the identity.
- `generation_config.json` holds the real stop set `[2, 11]`; `config.json` holds
  the scalar `2`.

**The two defects the synthetic suite could not see (both fixed, both Ultra-only).**

- **U1 — top-22 exceeds the batched expert GEMV's contract.** 
  `RunLsxNvfp4GemmSeq1GemmVExpertBatch` accepts at most **8** experts per launch;
  the 30B sibling routes to top-6, so copying its call returns
  `cudaErrorInvalidValue` on the first MoE layer. Launches are chunked by 8 now.
- **U2 — the latent bridge aliases its own input.** With `moe_latent_size = 2048`
  the routed up-projection writes `topk*I = 112 640` halfs into the buffer the GEMV
  is still reading the 2048-wide latent from, so chunk 0 clobbers its input. The
  30B cannot hit it (no latent bridge: its expert input is `normed`). Symptom:
  coherent-looking garbage (`'إن order'`), residual L2 growing 1.4 → 14 129 over
  108 layers, and the head faithfully reproducing that wrong state from the
  checkpoint's own BF16 weights. Fixed with a dedicated `s.latent16` buffer.

**Measured (RTX 5090, 32 GB, `linux-release-znver4-aocc`).**

- SelfCheck **PASS**: Paris ×3 bit-exact, `logits_hash bd82916c6550a2df`, top-1 6993.
- Tool round-trip: `get_weather {"city":"Paris"}`.
- Slice oracle (`tools/nemotron_ultra_oracle.py`, real layers 0–7 + real
  embed/norm_f/head): engine vs reference relL2 **3e-4 (L0), 2.4e-3 (L1, latent
  MoE), 7e-3 (L4), 3e-2 (L5)**. It must be a CONTIGUOUS prefix — with layers
  0,1,7 the attention block's input differs by five layers and the comparison is
  meaningless.
- CODE 3/3, DET 3/3 (byte-identical over three runs), tokenizer 22/22.
- Perf: **1.77 t/s** (r3 1.75), TTFT 13.4 s, 139 W at the canonical 2 048-token
  protocol. 661 of 55 296 expert slots; 12.5 GB of expert traffic per token.
- Conversion: 292 GB, 24 576 experts, 1 040 MTP tensors cut, **peak host residency
  512 expert records** (per-layer streaming; the whole pool would be 290 GB).

**Open, explicitly.**

- **O1 — FSI-01 is a real generation error, excluded in writing.** The model
  answers `PV=CF*((1-exp(-r*n))/(1-exp(-r)))*exp(r)`: it appends `exp(r)` to the
  expression that is *already* the annuity-due form, landing 8.7 % high at
  r = 0.0835. The 30B sibling appends the same factor (§R13), so it is a
  family-level habit, not a conversion artefact. The committed FSI golden keeps
  FSI-02/03 (both pass) and records FSI-01's exact output under
  `excluded_cases`; the matrix row therefore reads 2/3, not 3/3.
- **O2 — no full-model source reference.** The 1.1 TB BF16 tree cannot be
  assembled in 750 GB, so the parity goldens are template-derived (recorded in
  every fixture's `source`) and source fidelity rests on the slice oracle's 8
  layers. Extending the slice is the documented lever (each 8-layer step with 3
  MoE layers costs ~65 GB of host RAM).
- **O3 — conversion tooling.** `lsxforge --nemotron_ultra --layers a:b` converts a
  window and marks the tree partial; a partial tree is refused by the loader even
  under `LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1`. Resume works by re-running a later
  window (the index lists the union of layers on disk).
- **O4 — the Ultra converter validates descriptors but does not call
  `integrity::VerifyAllTensorReads`**, which reads all 51 023 tensors into heap
  vectors — a second full 1.1 TB pass. If another family ever needs the same, the
  helper wants a scoped form rather than being skipped ad hoc.
