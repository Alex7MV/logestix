# Ling-3.1-flash — open items

### R7. Ling-3.1-flash (560B-A25B) — ingestion landed, parity goldens are declared debt (2026-10-30)

**Landed.** Family `ling` (`logestix-ling-nvfp4`) on the real 535 GiB
`BailingMoeV3ForCausalLM` checkpoint converted in place
(`/mnt/fast_storage/huggingface/Ling-3.1-flash`, **298 GB** tree: 10 GiB pinned
dense side + 308 GB routed-expert arrows). Geometry: 72 main layers + MTP
layer 72 (stripped and counted, `deferred.mtp = 1555`), 63 KDA + 9 gated MLA
(every 8th layer), 68 MoE layers × 512 routed experts top-8 (group-limited
sigmoid + `expert_bias`, renorm ×2.5) and one ungated shared expert, vocab
157 184. The 34 816 routed experts stream from the neutral mmap `ExpertStore`
through a device slot LRU (≈16-17 GiB, 2053-2182 slots).

**What the checkpoint forced into the engine (measured).**
1. **Per-expert global scales.** Every expert carries its own three NVFP4
   `global_scale`s; the first loader pass filled `MoEExpertRecord.gs*` from a
   single probed expert and the output was incoherent English/Chinese
   ("Theodiech"). Visiting all experts in the geometry pass fixed it — the
   control prompt now reasons and answers "The capital of France is **Paris**".
2. **Official SwiGLU limits.** `share_expert_swiglu_limit_list` is 10 for layers
   64..71 and `expert_swiglu_limit_list` is 10 for layers 65..71. The
   reference HF `modeling_bailing_moe_v3.py` ignores them, but SGLang's
   `BailingMLP` applies `min(silu(gate), L) * clamp(up, -L, L)`; without the
   shared clamp the layer-64 activation jumps ~5x and the answer degrades. Both
   clamps are implemented (`RunLingSwiGLU` + the neutral routed FFN's optional
   `swiglu_limit`), and the HF `expert_swiglu_limit_list` is indexed per layer.
3. **Tokenizer kind 5.** The checkpoint's pre-tokenizer is a single letter run
   with an optional prefix, ONE digit per number word and apostrophe
   contractions (`'(?i:[sdmt]|ll|ve|re)|...`), not the o200k case-run split.
   `LingWordStream` + `tokenizer_kind = 5`; parity **27/27** against the
   checkpoint's own tokenizer (golden captured by
   `tools/ling_tokenizer_capture.py`).
4. **Scratch zeroing.** The per-request arena and router slots are zeroed before
   use; a reused `cudaMalloc` block could otherwise surface stale/NaN bits as a
   router `eid = -1` on the second model load in one process.

**Gates.** SelfCheck: **Paris ×3, bit-exact `final_logits`** (kOff, 10 762 MiB
pinned + 17 GiB expert cache), the kAuto token stream is stable across reloads
(raw logits are the TC tolerance class), and the real-weight native tool
round-trip returns `get_weather {"city":"Paris"}`. Tokenizer parity 27/27, tool
framing 51 assertions, converter/loader/kernels/model stages green. Perf rows
seeded through the harness at the canonical 2048-token protocol:
**2.77 t/s (perf) / 2.41 t/s (r3)**, TTFT ≈ 10.3 s, 2048 generated tokens
(`tests/perf/perf_baselines.json`).

**Golden debt (declared).** FSI/CODE/DET cannot be captured from the source on
this card: the 560B BF16/FP8 model needs 8×H200 and the FLA KDA kernels are
GPU-only. A CPU float32 reference over the source safetensors exists for
single-token probes (`/tmp` scratch during this session: pos-0 top-5 and the
20-token prompt's next-token top-5), but a full 9-prompt capture set is hours
per prompt. The three targets therefore SKIP with rows in `kGoldenDebt`
(`ling:fsi`, `ling:code`, `ling:det`); a future capture must come from the
source (SGLang with the official clamps preferred), never from the engine.
