# Beam-1.0-501B-A23B — open items

### R8. Beam-1.0-501B-A23B (501B-A23B) — ingestion landed, parity goldens are declared debt (2026-10-31)

**Landed.** Independent family `beam` (`logestix-beam-nvfp4`) on the real
934 GB `BeamForCausalLM` checkpoint converted in place
(`/mnt/fast_storage/huggingface/Beam-1.0-501B-A23B`, **265 GB** tree: 6.49 GiB
pinned dense graph + the device expert cache, and 12 544 routed-expert arrows
under `experts/layer_{3..51}/`). Geometry: 52 layers (3 dense + 49 MoE × 256
routed experts top-8, no shared expert), 5120H / 48 Q · 8 KV · 128 HD / vocab
151 936. The experts stream from the neutral mmap `ExpertStore` with
`first_moe_layer = 3` (`LSX_BEAM_EXPERT_MAX_OPEN`, default 4096) through a
device slot LRU (`LSX_BEAM_EXPERT_CACHE_SLOTS`, default 512 ≈ 11 GiB).

**What the checkpoint forced into the engine (from
`transformers/models/beam/modeling_beam.py`, read before writing any kernel).**
1. **Gated GQA.** `self_attn.gate_proj` [NH·HD,H] plus **one bias per head**
   (`gate_bias` [48]); `attn_out *= sigmoid(gate + bias[head])` BEFORE `o_proj`.
2. **Four sandwich norms + depth-scaled residual.** `idx = 2*(layer+1)` for the
   attention branch and `idx+1` for the MLP branch:
   `out = branch/sqrt(idx) + residual*sqrt((idx-1)/idx)`, each scaled operand
   rounded before the add.
3. **Hybrid attention.** Sliding layers (43) use rotate-half RoPE over the full
   128 head dims (theta 10000, window 512 INCLUDING the current token); the 9
   full layers {5,11,…,47,51} run **NoPE** with an SSMax query scale
   `(1 + (pos+1)/2048)^0.1` applied to q in FP32.
4. **Group-limited sigmoid router.** F32 logits; sigmoid scores;
   `e_score_correction_bias` only for CHOICE; group score = sum of the two best
   corrected scores in each of 8 groups; keep the top 4 groups; top-8 by
   corrected score; weights = the PLAIN sigmoid scores renormalized, ×2.5 after
   aggregation. No shared expert.
5. **Final logit softcap 30** (`logits/30 → tanh → ×30`, F32).
6. **Per-head q/k RMSNorm** (weight form) before RoPE/SSMax; SwiGLU product in
   FP32.
7. **Tokenizer `kind = 6`** (Qwen2/3 single-run pattern: one digit per number
   word, no case/CJK split) with the Harmony control markers as
   `special:false` added tokens (matched literally). Tool wire: modified
   Harmony, `to={name}` + `<channel|>analysis <|constrain|>xml` +
   `<rfl:parameter name="k">` bodies.

**Two defects the real weights exposed (both fixed).**
- `config.json` carries `eos_token_id: null`; the real stop list lives in
  `generation_config.json` as `[151674, 151675]`. Reading only the first id
  (`<|return|>`) made the model **repeat a tool call forever**: `<|call|>`
  (151675) never terminated generation. The converter now carries the whole
  list, and the index/loader/model propagate it.
- The sampler can reach reserved embedding rows (ids ≥ the tokenizer's largest
  added id). A single such id made the *whole* token-stream decode fail, so a
  perfectly valid native tool call parsed as "no call". `ParseBeamToolCalls`
  now drops ids outside the vocabulary before decoding.

**Gates.** SelfCheck: **Paris ×3, bit-exact `logits_hash`** (kOff, 17 445 MiB
resident = 6.49 GiB dense + 11 GiB expert cache); the real-weight native tool
round-trip returns `get_weather {"city":"Paris"}`. Tokenizer parity 30/30,
tool framing, converter/loader/kernels/model stages green. FSI/CODE/DET are
**declared golden debt**: the 934 GB BF16 source does not fit in this host's
750 GB RAM, so no source-oracle capture is possible (rows in `kGoldenDebt`:
`beam:fsi`, `beam:code`, `beam:det`). The instruments used instead are the
host-reference kernel parity suite (norms, RoPE/SSMax, router selection,
window bounds, gate, depth residual, softcap) and SelfCheck/determinism.

**Perf.** Seeded through the harness on the Release/AOCC tree at the canonical
2048-token protocol (`tests/perf/perf_baselines.json`): **1.61 t/s (perf) /
1.60 t/s (r3)**, TTFT ≈27.1 s, 116.97 W / 119.22 W average on the 1531 GB/s
DRAM probe. That is the correctness-first path: token-by-token prefill/decode,
a per-MoE-layer router readback (49 host round-trips per step, no copy-stream
overlap), and a 512-slot expert cache; the release build is ~2.6x the debug
number. The next levers are the ids-streaming/no-handshake decode path used by
qwen/ling and an expert prefetch copy stream — both are engine work, not
correctness gaps.
