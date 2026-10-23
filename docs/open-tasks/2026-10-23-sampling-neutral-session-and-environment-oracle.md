# Sampling, neutral KV session, hoisted prequant, environment oracle — 2026-10-23

Verbatim sections from `docs/OPEN_TASKS.md` (§E4, §S, the §T cost paragraph and
§W), with the resolutions found in this session.

### E4. `temperature` / `top_p` are accepted and ignored by the OpenAI surface
- **Status 2026-10-17: documented, implementation planned (below).** `ModelRequest`
  (`libs/lsxcommon/include/lsxcommon/model.h:64`) has no sampling fields and
  every family decodes argmax (`sample::GreedySampler`), so a request carrying
  `"temperature": 0.7` through the TaskMosaic bridge is served greedily. Nothing
  regressed; this was always the behaviour.
- **Documented where a client will actually read it:**
  `openai_endpoint.h:112` now spells out that `temperature`, `top_p`, `top_k`,
  `n`, `seed`, `presence_penalty`, `frequency_penalty` and `logprobs` are parsed
  as unknown and DISCARDED, that this is deliberate, and that greedy is the
  numeric class the DET/FSI/CODE goldens pin.
- **Why greedy is the right default to keep, not an oversight:** this engine is
  a bit-exactness/parity machine. A sampling kernel makes generations
  non-reproducible by construction and would invalidate every golden captured
  from the source checkpoints. Any future sampling work must therefore be
  opt-in and numeric-class-aware, not a silent default.
- **Plan when it comes up (sized by reading the code, not guessed):**

  | layer | work | size |
  |---|---|---|
  | `ModelRequest` | add `temperature`, `top_p`, `top_k`, `seed` | trivial |
  | `sample::ISampler` | the real work: a temperature/top-p CUDA sampling kernel (softmax + multinomial + RNG). Today it is a pure argmax interface — this is a new numeric primitive, not a refactor | **the bulk** |
  | call sites | thread the request fields into `Sample()`: **12** driver files (`qwen_*`, `glm_*`, `deepseek_*`, `ornith`, `phi`, `granite`, `gemma_*`, `mistral`, `xing`, `mimo`, `kimi`, `nemotron`) | mechanical but wide |
  | v2 wire protocol | `FabricRequestView` (`fabric_wire.h`) + `Submit()` + server-side `mreq` in `fabric_server.cpp` + client in `chat_bridge.cpp`. **Changes the wire frame**, so v2 backward compatibility needs a decision | medium, needs a version call |
  | numeric-class policy | decide whether sampling is allowed in `kAuto` (production) or only under an explicit mode; DET/FSI/CODE must pin greedy | a decision, not code |

  So: implementing it is a **sampling feature plus a wire-format change**, not a
  small plumbing fix. That is why documenting is the right call for now.
- **Next step (when scheduled):** start from the `ISampler` kernel + the
  numeric-class policy, decide the wire-format version, then the 12 call sites
  are mechanical.

## Resolution

`ModelRequest` now carries `sample::SamplingParams`
(`temperature`/`top_p`/`top_k`/`seed`). `sample::SamplingSampler` is a real
device sampler: multi-block max, exact top-k by four-pass radix selection on the
float bits of `max - v` (ties at the k-th value are kept), a 4096-bin
conservative nucleus for top-p (the boundary bin is included, so the retained
mass is >= p and overshoots by at most one bin), an exact per-block sum and a
deterministic pick; the RNG is counter-based splitmix64 over `seed`+draw index,
so a request replays exactly. Greedy remains the default and is bit-identical:
`SamplerFor(params)` returns `DefaultSampler()` when `temperature <= 0`, and
every family with a fused argmax tail keeps it behind `ISampler::IsGreedy()`
(zero extra launches for greedy). The 4096-bin top-p approximation is documented
in `sampling.h`; DET/FSI/CODE never sample.

Wire: request frames are v3 (40-byte header: temperature, top_p, 64-bit seed,
top_k, reserved). Stream frames stay v2, and a v2 request still decodes as
greedy, so an older client keeps working. `taskmosaic` parses the OpenAI and
Anthropic sampling fields instead of discarding them, `lsxfabric` maps them onto
`ModelRequest`, and `test_fabric_wire` pins both the v3 round-trip and the v2
compatibility path. A guard test (`every family threads the request sampling
policy`) scans every `Infer(const ModelRequest&)` TU and requires
`SamplerFor(req.sampling)` (or the shared `RunInferShared` base) so the
"accepted and ignored" defect cannot come back silently. Tests:
`test_sampling` matches the softmax distribution over 8000 draws, proves the
top-k set, the dominant-token nucleus, and byte-identical sequences for a seed.

## S. Attention layer neutral; only the KV session keeps a qwen name (2026-10-10, updated)

- **What landed (Phase 5, 2026-10-11):** `qwen_inference.{h,cu}` is now
  `lsx_attn_inference.{h,cu}` in `lsxcommon::attn`, and the neutrality guard
  enforces it. `QwenLayerWeights` became the neutral `attn::LayerWeights`
  (`lsx_inference_load.h`); the kernels take a neutral `attn::AttnGeometry`
  instead of the `QwenTopology` passport (they read exactly 12 of its scalars
  and none of its vectors/strings/vram). Two pure-passthrough GEMM shims were
  deleted outright. Bit-identical on real weights; throughput unchanged
  (ornith 167.3 vs 167.06 t/s baseline at 2048).
- **What is still qwen-named on purpose (Phase 5b):** `QwenKvCache`,
  `QwenGenerationResult` and the shared fast-math aliases
  (`QwenTcFastMath`/`SetQwenTcFastMath`/`QwenSeq1Gemv`/`QwenGemmLarge`/
  `QwenFastMathScope`). Three families hold a `QwenKvCache`, so renaming it
  into a qwen-owned header would make Ornith depend on another family --
  worse than today, for a cosmetic gain. `QwenKvCache::Allocate` also still
  takes `QwenPinnedWeights`.
- **Next step (5b):** change `QwenKvCache::Allocate`/`EnsureCapacity` to take
  scalars rather than the pinned-weight bundle, then the cache and result
  structs can go neutral too. Separate change; nothing blocks on it.

## Resolution

`KvCache::Allocate`/`EnsureCapacity` now take a neutral `attn::KvGeometry`
(explicit layer counts and head dims) instead of `QwenPinnedWeights`; the
shallow per-request `QwenPinnedWeights` view is gone from the qwen MoE, qwen MoE
Exp and Ornith drivers. `QwenKvCache`/`QwenGenerationResult` became
`attn::KvCache`/`attn::GenerationResult` in the neutral attention layer
(`lsx_attn_inference.{h,cu}`); the qwen driver header keeps only two `using`
aliases so its call sites stay readable. `test_model_independence_guard` no
longer lists the two names as allowed qwen symbols, and `test_qwen_dense_kv_parity`
stays bit-exact.

  **Cost of the fix, accepted deliberately:** the quantizer is now two launches
  instead of one, so the TC branch pays one extra launch per slot per layer —
  4 slots x 48 layers = 192 extra launches per token, about 1 ms against a
  ~22 ms token. Measured in Release at the canonical 2048-token protocol:
  `perf` (kAuto) 47.13 -> **45.02 t/s (-4.5%)**, re-recorded in
  `perf_baselines.json` at commit `cb60207`; `r3` (kOff) is unaffected at 14.68
  because `kOff` forces `tc=false` and never enters this branch. The guard
  flagged the drop as DEGRADED and refused to move the reference on its own,
  which is the intended behaviour — the cause was identified (launch overhead,
  not bandwidth: the FP32 round-trip is 8 B/element against multi-MB expert
  tiles) and only then was the number changed by hand. A 4.5% decode cost is a
  fair price for a 52x error, but it is a standing cost, so a cheaper shape is
  available if it ever matters: hoist the reduction out of the per-slot loop
  (2 launches per layer instead of 8) by quantizing all K slots in one pass.
  Not done — it is a restructuring of `RunMoERoutedFFN` for 4.5%, which does not
  justify the regression risk across the five families that share this path.

## Resolution

Hoisted. When the caller supplies `MoERoutedTcScratch` (packed nibbles, block
scales, FP32 SwiGLU arena and per-slot alpha) and a `ygu_batch`, the TC routed
path runs w13 for every slot, then **one** batched absmax and **one** batched
quantize for all K slots (`RunAttnSwigluPreScaledQuantBatch`), then the per-slot
down GEMM and fold. Three prequant launches per layer instead of three per slot;
the per-slot alpha, the E4M3/E2M1 quantization and the ascending fold order are
unchanged, so the accumulator is bit-identical — pinned by a new case in
`test_qwen_moe_kernels` that asserts `memcmp == 0` against the per-slot chain
*and* that the hoisted branch actually ran (non-zero alpha). Mistral's session
now owns the scratch, which also removes the per-slot `cudaMallocAsync` pairs.
Measured on the canonical 2048-token protocol: Mistral 41.3 -> 42.2 t/s
(+2.2%) with the same selfcheck hash; `LSX_MOE_TC_HOIST=0` keeps the old path
for bisection. The rest of §T (the three golden gates) stays open: the source is
still unrunnable, so no parity oracle exists.

## W. Throughput baselines read low on a heat-soaked GPU — cooldown added (2026-10-15)

`test_perf_baseline` measured phi-4 r3 at 117.2–118.0 t/s against a 122.26
reference (-3.9%, DEGRADED) and qwen3.8-flash-next perf at 43.14 against 44.85
(-3.8%, DEGRADED) — reproducibly, with a free, idle GPU. Neither is a code or
artifact regression:

- The **same Release-AOCC binary** re-measures the references when the card is
  left to cool: phi r3 back within tolerance; qwen3.8 isolated `lsxbenchmark`
  44.8 (reference 44.85).
- qwen3.8's reference was recorded 2026-10-14 **on this same build**, artifact
  unchanged since 2026-10-07; the driver (`595.91.07 -> 595.99.02`, 2026-10-08),
  kernel (`7.0.0-34 -> -38`, 2026-10-02) and CUDA toolkit (`13.4.2 -> 13.4.3`,
  2026-10-13) updates all predate it, so the "old toolchain" hypothesis is
  retracted by that control.
- Under load the SM clock settles at **~2812 MHz against a 3150 MHz maximum**
  (600 W power cap, 64–76 °C): one 35–45 s decode heat-soaks the card, so
  back-to-back targets read low. A fixed threshold *below* the resting
  temperature never works — returning to the ~40 °C idle floor is a **plateau**
  taking ~2–3 min (traced 70 -> 40 °C).

**Fix:** `lsxcommon::perf::CoolDownGpu` (NVML, resolved at runtime like the
power sampler) waits until the GPU temperature **plateaus at its running
minimum** — as cool as the ambient allows — with a timeout, before each
`RunPerfModel`. It runs once per target, before the perf->r3 pair, so the
within-target protocol is unchanged. Knob `LSX_PERF_COOLDOWN_TIMEOUT_S`
(default 300; 0 disables). Verified: phi-4 and qwen3.8-flash-next pass
back-to-back; fast suite 94/94.

**Second measurement defect, fixed 2026-10-23:** `test_perf_baseline` did not
call `Core::Initialize`, so unlike every app it ran with spdlog's synchronous
default logger. On the same binary, prompt and 2048-token protocol it read
134.4 t/s where `lsxbenchmark` read 146.6 (reproduced both ways; adding the
initialization makes them agree). The harness now initializes the logging core,
so a baseline measures the path users run. Consequence: every other family's
committed row was recorded on the slower path and is ~9% low; the next run of
that target re-seeds it upward automatically (a higher reading is `kUpdate`,
not a failure), so they are conservative floors in the meantime.

**Still open (the §B class):** a genuinely warm room whose plateau sits above
what the reference needs cannot be cooled into tolerance — no oracle separates
"the room is warm" from "the code regressed" at 3%. The cooldown removes the
back-to-back component, not the ambient one.

## Resolution

`perf::MeasureMemoryBandwidthGbs` is a fixed 512 MiB device-to-device copy
(deliberately larger than the 96 MB L2, so it is DRAM-bound; 64 MiB measured L2
speed, 2165 GB/s, and was rejected). The perf harness measures it once per
target after the cooldown and stores it as a synthetic table entry
(`__gpu_probe__`/`probe`, GB/s, protocol in the `prompt` field). A degraded
model reading is now attributed by `perf_baseline::AttributeDegrade`: it is
ENVIRONMENT (reported, not failed) only when the probe is also below 97% *and*
down at least as much as the model; otherwise it stays a code regression, so the
oracle can never mask one. `test_perf_baseline_logic` pins the attribution table
(including the "probe down but not enough" case). The full perf suite was
re-seeded on 2026-10-23 with the probe in place (see the archive row in
`OPEN_TASKS.md`).
