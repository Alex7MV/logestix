# gemma-4-31b-it (dense) — open items

### R. gemma-4-31b-it (dense): FSI-01 RED (the TC partial-M bug it was found by is closed, 2026-10-10)

The dense 31B member of the Gemma-4 family was ingested (`gemma_dense_*`,
format tag `logestix-gemma-dense`, `lsxforge --gemma4dense`). Real-weight
results: SelfCheck PASS (`Paris.`, bit-exact across stress cycles), CODE 3/3,
DET 3/3, FSI 2/3. Open items:

- **FSI-TEST-01 RED (stays red).** Source (golden):
  `PV=2500000*((1-exp(-0.0835*7))/(1-exp(-0.0835)))`; the converted model
  repeats the prompt template `PV=CF*((1-exp(-r*n))/(1-exp(-r)))`. The fork is
  at step 2 (after `PV=`), and it was re-measured in the engine on 2026-10-24
  after the MoE member's fix (pre-softcap, teacher-forced): src `2` 55.5 vs
  `CF` 42.75, converted `CF` 46.1 vs `2` 43.4.
  **The 26B residual recipe does not transfer and no partial placement fits.**
  Measured in the engine (each variant = a 65 s re-conversion + probe):
  `o`+`down` planes (the offline-simulation winner) -> margin `CF` +6.3 at
  22 GiB; `q,k,v,o`+`down` -> `CF` +10.4 at 25 GiB; all linears -> 33.7 GiB,
  OOM. Adding precision moved *away* from the source at this pair, i.e. the
  required upgrade is broader than the card. The offline HF simulation is only
  qualitative here: its all-4-bit baseline already sat on the source's side of
  the fork (its error is smaller than the engine's), which is why
  "`down`+`o` suffices" did not transfer. The dense residual plumbing was
  therefore reverted -- the tree is back to the 4-bit-only 17 GiB format.
  Getting this gate green needs ~8 bits on every linear (~33.7 GiB resident),
  i.e. a 48 GiB card or a different block format. Full trail:
  `open-tasks/2026-10-24-gemma4-moe-residual-plane-closed.md` §5.
- **Scalar prefill is a choice, not a workaround.** The shared TC M-tile
  defect this gate was originally found by is fixed (2026-10-10, archived);
  gemma keeps the scalar path because running it through TC moves the first
  token and breaks the tool round-trip —
  [`open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md`](open-tasks/2026-10-10-neutral-moe-and-tc-mtile-closed.md).
- **Perf: 47.3 t/s (perf) / 41.0 t/s (r3)** at 2048 tokens on
  `linux-release-znver4-aocc` (16.5 GiB resident; TTFT 337 ms, scalar prefill).
  Profile-guided decode work took it 40.6 -> 47.5: shuffle RMSNorm
  (11.4 -> 3.5 us/call), tied-head logits through the neutral warp-per-row GEMV
  (308 -> ~670 GB/s), unsplit attention at short context.
- **Rejected experiments (each measured, then reverted):**
  - *CUDA-graph replay of the decode step.* Implemented (capture-safe device
    position + KV-write kernel; `LSX_GEMMA_DENSE_CUDA_GRAPH`), capture
    succeeded, but it is SLOWER: 46.8 vs 53.8 t/s at 1024 and 47.6 -> 46.1 at
    2048. The decode is kernel-bound, not launch-bound, and a captured step
    needs a FIXED split count (the position advances in-graph), which
    over-splits short contexts. Default off. The device-pos/KV-kernel refactor
    it required was reverted with it.
  - *Warp-per-head decode attention.* 43.1 vs 47.6 t/s (only ~1k threads in
    flight). Reverted.
- **Comparison to the other dense model, Qwen3.8-27B** (same release tree,
  same prompt, `--max-gen 2048`): 54.9 t/s, but it is 48/64 *linear-attention*
  (GatedDeltaNet) layers, so its per-token work is far lower, and it reads a
  smaller pinned set; gemma-4-31B is 60 full/sliding-window attention layers
  and ~7% more weight bytes (16.1 vs ~15 GiB). Effective HBM bandwidth is
  comparable (gemma ~818 GB/s vs qwen ~884 GB/s). The ~13% gap is layer-type +
  bytes, not a defect; at `--max-gen 1024` they are at parity (53.8 vs 54.1).
