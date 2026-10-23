# Residual closures — 2026-10-23

Verbatim §U from `docs/OPEN_TASKS.md`, with the resolution found in the same
session.

## U. granite-4.2-30b tool round-trip RED again at `31998bf` (found 2026-10-13)

`test_granite_dense_selfcheck`'s second test case ("end-to-end tool
round-trip", real weights) fails at HEAD:

```
CHECK( calls[0].name == "get_weather" )   with expansion: "get_weather.x" == "get_weather"
```

The model emits `get_weather.x` — it puts the `.x` into its **own reasoning
text** first ("We have tool: get_weather.x, …"), then into
`<function>get_weather.x</function>` and into the bare
`{"name": "get_weather.x", …}` object. So the parse path is not at fault: the
divergence is in the greedy generation. The first test case is unaffected
(`Paris`, `top1_tok=12366`, `logits_hash=3d1604aa75c99a1f` stable over 3
cycles), and FSI/CODE/DET are green.

**Not an Arrow regression.** Found while bumping Arrow 25.0.1 -> 26.0.0, so it
was attributed with a control: a second tree at the *same* commit (`31998bf`)
built against **Arrow 25.0.1** produces the identical `get_weather.x` **and the
identical `logits_hash=3d1604aa75c99a1f`** — the Arrow bump is bit-identical
on this path.

**Where the suspicion points instead:** the granite tool round-trip was
"recovered" on 2026-10-07
(`open-tasks/2026-10-07-nemotron-nvfp4-and-codec-closed.md`, archive row
L/M/N), and 9 commits have landed in `libs/lsxcommon` since — including three
that touch the shared attention layer granite decodes through (neutral-layer
split `2b4b41c`/`65db1c7`, `00e22ec` TC mxf4nvf4 GEMM tile fix, `f99008b`
`__host__ __device__` KV-geometry helpers). Which of them moved this token is
**not yet established** — the missing instrument is a generation-level oracle
(the token ids of the 256-token round-trip, plus the logit margin at the flip
step), not another pass/fail on the same assertion.

## Resolution

The defect was **not** in generation arithmetic: the shared attention/GEMM
changes listed above had not moved the token — the `get_weather.x` spelling was
the generic Hermes prompt, not the checkpoint's ChatML template. The native
Granite codec (`granite_tool_format.{h,cpp}`, commit `de0b160`) renders the
checkpoint's `<|im_start|>…<tool_call><function=…>` form, including the
`<think></think>` prefill and the `enable_thinking=false` newline. The real-weight
`test_granite_dense_selfcheck` tool round-trip now returns `get_weather` and the
Paris `logits_hash=3d1604aa75c99a1f` is unchanged, so no golden was re-captured.

## qwen fused-linear attention test fixtures

`test_qwen_dense_fused_linear_attn` had two RED cases at synthetic
`nk=2 kd=8` (`nanf < …`). They were test-fixture defects, not kernel defects:

- `FillNvfp4` generated E4M3 block scales `120 + (i % 12)`, which includes
  `0x7F` — the reserved E4M3FN NaN. The scale cycle stops at `120 + (i % 6)`.
- The host z reference accumulated the same projection twice.
- After the NaN was removed, the tiny full-layer geometry (`gs=0.5`) overflowed
  FP16 at the output; it now uses the production-like `gs=0.02`.

All 43,675 assertions pass.

## §Y — r3/bit-exact performance restored (and qwen3.6 closed)

The X4/F-04 strict path had been made environment-independent by disabling the
large-N streaming GEMV; qwen3.5-9b r3 fell to **19.3 t/s** against the 111.06
reference. Disabling the route was the wrong side of the trade.

The strict scope now sets a process-global `LsxBitExactMode` flag. The
dispatcher reads it as: both the large-N and the size-bounded (<=32 MiB)
streaming GEMV are canonical strict routes, TC/seq1-gemv stay off, and
`LSX_GEMV_LARGE`/`LSX_GEMV_MAX_MIB` are ignored, so the strict hash cannot move
with the environment. qkv and z use the neutral `RunLsxNvfp4Gemm` route in this
mode for every H (a/b stay warpdot at N=32), which is what closes qwen3.6
(H=2048): with the earlier H>=4096 floor it measured 69.6, without the floor
73.4.

Canonical 2048-token measurements and re-seeded references:

| model | old r3 | new r3 |
|---|---|---|
| qwen3.5-9b | 111.06 | **147.0** |
| qwen3.8-27b | 39.03 | **52.27** |
| qwen3.6-35B-A3B | 13.32 | **73.39** |

The perf entries that the harness rewrote on the same runs were kept. Correctness:
qwen3.5-9b, qwen3.8-27b, qwen3.6-35B-A3B and Ornith SelfCheck/FSI/CODE/DET all
pass. `test_perf_baseline_qwen3_5_9b` and `..._qwen3_6_35b_a3b` pass both modes.

Rejected alternative: keep the streaming GEMV off and re-seed r3 at ~19 t/s
(correct but 5.7x slower); it makes the strict mode useless for the main use case
and does not improve determinism.

### Ornith perf-mode 167.06 -> 160.2 (closed as a documented cost)

`test_perf_baseline_ornith` read 160.0/160.1 t/s on two cooled runs against
167.06. Attribution: the reference is from `36eddff` (2026-10-07), before the
neutral routed-MoE migration (`c3ddc4b`) and the overflow-proof batched SwiGLU
quantizer (`8e34314`). That commit explicitly records the cost: "one extra
elementwise launch per layer (Qwen3.6 256-token: 166.7 -> 161.3 t/s)" — the
same ~4% seen here, so the gap is a deliberate correctness fix (massive
activations no longer overflow FP16 and poison the block scale), not a new
regression. The perf entry was re-seeded to **160.20 t/s** with the measured
TTFT/power evidence, and the harness then re-seeded r3 from 12.70 to
**72.02 t/s** on the new strict route.

Router investigation (negative): the decode router is ~10% of GPU time, but
changing its launch from 128 to 64 threads per block moved the isolated kernel
only 11.80 -> 11.73 us and left E2E unchanged (160.5 vs 160.0); reverted.
`ornith.router.decode` had no budget row (the test SKIPped); a row was added
with 11.84 us observed and a 14.0 us max, so the real-shape router is now
guarded instead of silently skipped.


## Red selfchecks re-checked (2026-10-23)

All four currently-red real-weight gates were re-run after the strict-GEMV
work, and each reproduces with the route disabled as well (no routing
artifact):

- `test_gemma_dense_fsi_selfcheck`: FSI-01, the model emits the template
  `PV=CF*(...)` instead of substituting the prompt's numbers (§R).
- `test_gemma_moe_fsi_selfcheck`: FSI-01 drops the annuity-due `exp(r)` and
  FSI-02 substitutes the numeric rates instead of the symbolic variables;
  FSI-03 passes (§J, refreshed).
- `test_gemma_moe_code_selfcheck`: CODE-02 emits `m=l+(h-l)/2` against the
  source's `m=l+((h-l)/2)`; the gate intentionally does not canonicalize
  algebraically (§J).
- `test_phi_dense_det_selfcheck`: DET-02 is the known C-vs-CR near-tie
  (source margin ~1.1 logits); it stays CRITICAL deterministically 3/3 (§O).

These are model-defined divergences at the NVFP4/bf16 boundary, not harness
or routing defects. The only safe fixes would be to change the numeric class
(e.g. keep sensitive heads/embeddings in BF16, which changes the model format
and needs a re-convert plus all gates) or to weaken the oracle, which the
project deliberately rejects. Left red.

The previously-unrun `test_alice_moe_code_selfcheck` and
`test_alice_moe_det_selfcheck` now pass (151 s / 94 s), closing the `n/r`
coverage gap in the audit table.

## C1 — GPU budget tests are now serialized

`LSX_KERNEL_BUDGET_TESTS` (the eight budget suites plus `test_ornith_moe_kernels`,
which gained a router budget) now carry `RESOURCE_LOCK "gpu"` in
`tests/CMakeLists.txt`, alongside `SKIP_RETURN_CODE 4`. A `ctest -j` run can no
longer overlap two latency/bandwidth budgets, which was the measured flake:
five concurrent copies failed four of five samples while 80 solo runs were
green. This is the contention-proofing C1 asked for; the best-of-5 mitigation
stays as the second layer.

## D2 — Xing re-measured

At the canonical 2048-token protocol Xing now measures **123.2 t/s perf /
112.4 t/s r3** (old record 117.2 / 20.1; the r3 jump is the strict
streaming-GEMV route). The 127.1 perf figure that had been in the file did not
reproduce: two cooled runs read 123.3 and 123.2, so the perf entry was
re-seeded with the measured TTFT/power evidence rather than left 3% high. The graphless 128-token nsys profile attributes the
fixed step to expert-batch GEMV (~24%), `xing_rmsnorm_kernel` (~12%, 167
launches/token), mHC gates/proj/collapse (~19%) and the attention/router GEMVs.
A 128-thread `RunLxGemmVec` launch was neutral (123.4 vs 123.3 t/s) and was
reverted. The remaining GEMV-occupancy work is still a numerics project, so D2
stays open as that one item.

## D2 closure — Xing decode (supersedes "D2 — Xing re-measured" above)

**Closed 2026-10-23 at 146.65 t/s perf / 123.92 t/s r3** (was 123.2 / 112.4;
the pre-session direct baseline was 123.3, reproduced twice). Four measured
changes, each with the real-weight gates re-run:

1. **Expert-batch GEMV activation loads.** The expert body still issued sixteen
   scalar `__half2` loads per 32-element chunk; the dense seq1 kernel had long
   used four 128-bit loads. Porting that form is bit-identical (same 32 values,
   same arithmetic) and moved gate/up 13.51 -> 9.24 us and down 12.95 ->
   9.05 us per layer, i.e. ~543/566 -> ~800 GB/s. +6.1% end to end.
2. **The router projection ran twice.** `RunLxGemmVec` wrote `d_router_logits`,
   which nothing ever read; `xing_router_proj_kernel` inside
   `LaunchXingRouter` produces the scores top-k consumes. The dead launch cost
   6.4 us x 38 per token. +2.7%.
3. **mHC gates + collapse merged into one launch.** Warps 1..7 stage the 57 KB
   stream block into shared memory while warp 0 runs the Sinkhorn; one barrier,
   then the collapse reads shared memory instead of re-issuing the same loads.
   Same exp inputs, same per-cell order, same collapse order: bit-identical.
   9.10 -> 7.27 us per mHC block. +2.1%.
4. **RMSNorm was a single-block DRAM-latency kernel.** Cold (128 MB L2 flush)
   the 256-thread scalar form measured 7.99 us because 8 warps could not cover
   the 42 KB with 4-byte loads; float4 + 512 threads measures 4.96 us cold and
   1.75 us in situ (was 4.63). Tolerance class (the partial-sum grouping was
   already tolerance class); SelfCheck/FSI/CODE/DET all green. +6.9%.

**Measurement harness defect found on the way:** `test_perf_baseline` never
called `Core::Initialize`, so it ran with spdlog's synchronous default logger
while every app runs the async logging core. The same binary/prompt measured
134.4 in the harness and 146.6 via `lsxbenchmark` (reproduced both ways); with
the initialization added they agree. This is the first commit that measures the
production path. The other families' committed rows are therefore ~9% low and
will re-seed upward automatically the next time their perf targets run; until
then they are conservative floors, not comparable to Xing's new row.

**Final profile at 2048** (graphless nsys, 141.6 t/s under the profiler):
`xing_attn_split_kernel` 26.8% (about 84 MB of KV read per token - the memory
roofline for this cache), expert GEMV 15.9%, mHC 16.1%, dense GEMVs ~16.7%,
RMSNorm 4.3%, SwiGLU 2.8%. The serial-kernel and expert-GEMV-occupancy items
are done; what remains is context-dependent (KV compression) and prefill
(token-by-token, TTFT), both different projects.

Verbatim D2 section removed from `docs/OPEN_TASKS.md` follows.

### D2. `xing` decode: 14.4 -> 123.2 t/s (nine serial kernels + device dispatch + graph replay)
- **Status: serial-kernel class CLOSED (2026-10-05); re-measured 2026-10-23.
  14.40 -> **123.2 t/s perf / 112.4 t/s r3** at 2048 tokens (the old r3
  record was 20.10; the strict streaming-GEMV route removed the scalar-tile
  bottleneck). Every
  single-block, thread-0-serial and under-occupied kernel on the decode path is
  fixed or measured-and-declined; what is left is the GEMV occupancy, which is a
  different (numerics) project - see the end. Every fix came from the same
  method: nsys + a `--max-gen` sweep to separate fixed from context-dependent
  cost, then a Release-only latency budget at the PRODUCTION shape.
- The three serial costs, in the order nsys found them:

  | kernel | before | after | what it was |
  |---|---|---|---|
  | `xing_hyper_conn_kernel` | 229 us | 61 us | `<<<1,256>>>`, 24 sequential 256-thread tree reductions (192 dependent barriers) |
  | `xing_router_proj_kernel` + `..._topk_kernel` | 129 us | 6.4 us | `<<<1,256>>>`: one SM read all 458 KiB of router weights, and XingSigmoid (a double-precision glibc replica) ran 8x per warp back to back before the top-k could start |
  | `xing_attn_kernel` | 814 us | 209 us | `<<<32,256>>>` (32 of 170 SMs), softmax serial on thread 0, values serial over t on one thread per dim |
  | `xing_attn_split_kernel` | 209 us | 63 us | the fixed grid was 32 blocks - one per head; split-K over position chunks (heads x 8 = 256 blocks) plus a combine kernel |
  | `xing_hyper_conn_gates_kernel` | 36 us | 6.5 us | 20 Sinkhorn iterations + 16 exps serial on thread 0, and `ExpfGlibc` is a DOUBLE-precision replica - a consumer SM retires FP64 at 1/64 rate; then warp-0 registers with shuffles |
  | `xing_hyper_conn_collapse_kernel` | 8 us | 2.0 us | the 57 KB collapse was the tail of the one-block gates kernel (8 warps of loads in flight); its own wide block, no FP64 in it |
  | `xing_attn_combine_kernel` | 6.8 us | 1.8 us | kAttnSplits rescale factors were recomputed per head DIM thread - 16 serial double-precision exps each; now computed once per chunk |

- The attention ones are what the length curve pointed at:
  `us/token = 14258 + 16518 per 1000 cached positions` meant 54% of a 2048-token
  step was context-dependent, all of it in that kernel. After the two attention
  passes the same sweep gives **7016 + 1680**, i.e. the context term shrunk 9.8x
  and is now 19% of the step; the fixed term is 81% and is the whole frontier
  (it was 46% of the step before this series).
- The depth mixer needed TWO passes and the second one had to be re-derived after
  the first looked done. Splitting `<<<1,256>>>` into a projection (24 blocks,
  one per hc_fn row) and a gates block took 61 -> 40 us - and then nsys said the
  gates block alone was 36 of those 40. The cause was arithmetic, not occupancy:
  `ExpfGlibc`/`XingSigmoid` are bit-exact glibc replicas written in double
  precision, so the 16 exps + 4 sigmoids of the Sinkhorn were 20 serial
  double-precision calls on thread 0. One cell per thread (and one row/column
  reciprocal per thread) is numerically IDENTICAL - exp and max are element-wise
  and the Sinkhorn's per-cell updates were already independent - for 40 -> 17 us.
  **The lesson worth keeping: on consumer Blackwell (1/64 FP64) a "tiny" serial
  scalar tail can dominate a kernel, and nsys per-kernel attribution is what
  finds it - the block count looked fine.**
- **2026-10-23 profile and next lever.** nsys on the graphless decode
  (128 tokens) attributes the fixed step to the expert-batch GEMV
  (`gemv_expert_batch_ids`, ~24% combined), `xing_rmsnorm_kernel` (~12%,
  167 launches/token at ~4.6 us), the mHC gates/proj/collapse (~19%) and the
  attention/router GEMVs. A 128-thread `RunLxGemmVec` launch was neutral
  (123.4 vs 123.3 t/s) and reverted. The remaining 8 ms/token is still the
  GEMV/allocation-occupancy project described below, not bandwidth.
- Fixed cost, where the next levers are: the per-token `[vocab]` logits round
  trip is GONE (the sampler now reads `session_->d_logits` in place and only the
  8-byte argmax comes back, with the full row captured just for the final token
  - worth +0.7% end to end, 106.9 -> 107.9 t/s at 128 tokens, so it was never
  the E2E bottleneck, but it removes 1 MB of PCIe traffic and three
  `cudaMalloc`/`cudaFree` per token, which is what concurrent sessions on one
  model pay for). On the GPU side the remaining single-block work is the gates block
  (13 us x 40 layers = 0.5 ms/token) and the attention could still use more SMs
  (one block per head = 32 of 170; a position-split plus a combine kernel would
  open 8x more). `ResolveXingRouted`'s 38 D2H syncs per token were
  MEASURED as free at 128 tokens (64.3 vs 64.3 t/s with them skipped), so they
  are not worth doing before the two above.
- Numeric gates re-run on the real weights after every change (all three kernel
  rewrites are tolerance class): SelfCheck Paris x3, FSI 3/3, CODE 3/3, DET 3/3.
  Each kernel carries a Release-only latency budget and a host-parity case at the
  production shape - the original cases used hidden=64 / 16 experts, which is why
  all three defects survived a green suite.
- **What the remaining 7.0 ms/token is NOT: bandwidth.** Per token the decode
  path streams the 4-of-64 routed experts, the shared expert, the attention
  projections, the two dense layers and the LM head: ~2.68 GB, so at 113.6 t/s
  that is 304 GB/s, 17% of the 1792 GB/s peak -
  and a layer's ~53 MB should take 31 us at peak, while a layer measures ~222 us
  (9.1 ms / 41). So decode is latency/occupancy-bound, not bandwidth-bound, and
  the next lever is the GEMV path itself: `RunLxGemmVec` splits output rows
  across blocks conservatively and never fills 170 SMs, so each of the ~15
  dependent kernels per layer pays launch + drain latency at low occupancy.
  Raising decode further means wider GEMV tiles (more rows per block, split-K
  over the K dimension) - correctness-first no longer, i.e. a real numerics
  project with its own gate, not another serial-kernel fix.
- Effective bandwidth went 31.6 -> 304 GB/s (113.6 t/s, 8.8 ms/token). The memory roofline for this model
  is 2.19 GB/token => **694 t/s at 85% of peak**; at 113.6 t/s the model reads
  304 GB/s of it - which is the point above: the roofline is not the binding
  constraint any more, the GEMV occupancy is.
- **What the next project is, and it is plumbing, not arithmetic.** After the
  router fix the profile is dominated by kernels that are far too slow for their
  bytes: `xing_rmsnorm_kernel` at 4.8 us for a 3584-float row, 163 launches per
  token, and 81 each of the three mHC kernels. All are single-block, so 169 of
  170 SMs idle during a 42 KB reduction, and a decode step is ~400 dependent
  launches. Replacing the RMSNorm's 8-barrier tree with warp shuffles bought
  +0.4% (19292/19323 ms vs 19366/19392) which is the point: **the floor of a
  one-block launch here is ~4 us regardless of what the block does.** Neither
  more ILP nor fewer barriers moves it. The lever that does is CUDA graph
  capture of a whole decode step, and that requires the routed-id dispatch to be
  device-side - so the id->slot table whose sync measured free above becomes a
  PREREQUISITE here, not a win. Second project: the prefill row-batching (below).
- **DONE (2026-10-05): the decode step is captured and replayed as a CUDA graph**
  (114.1 -> 117.2 t/s at 2048, 113.9 -> 117.1 on the repeat; `LSX_XING_CUDA_GRAPH=0`
  restores the legacy per-token launch path). Three things had to become true
  first, and each is a separate commit:
  1. the routed-expert dispatch resolves on the device (`8bb474c`), because the
     host path drains the pipeline once per MoE layer and a drain cannot be
     recorded;
  2. `{pos, token_id}` live in device memory rather than in kernel arguments
     (`f7fa0bb`) - a graph bakes its arguments in, so a per-token value has to
     travel in memory. The embedding gather, the KV write and both RoPE kernels
     read them there now, and the RoPE kernels take the BASE YaRN tables and
     index their own row;
  3. the attention launches a position-independent geometry when captured
     (`3e0ae5a`): both forms at their maximum, excluding each other on the
     device position, with empty chunks publishing `(m = -inf, l = 0)` so the
     combine cannot read the previous position's slots. Same bits, verified at
     pos 2047 / 1000 / 255.

  What the graph does NOT cover: the sampler, the per-token logits readback and
  the KV-layout work below. `XingCaptureDecodeGraph` refuses (rather than
  capturing) when the step is not capture-legal - fast-math off, the layer-debug
  staging on - and a capture that fails after it started is reported instead of
  being papered over, because it leaves the context unusable.
- Tool calling is verified on the REAL weights
  (`test_xing_moe_selfcheck` "end-to-end tool round-trip", the same case nine
  other families run): the model emits its native markers as the control ids
  11/12, `arrow_tokenizer_decode` drops special tokens, so `output_text` reads
  as a bare `get_weather<param_key>city</param_key><param_value>Paris</...>` with
  no marker in it - which is why `xing_moe_tool_format::Parse` walks the ids and
  splices the marker bytes instead of decoding, and why the test asserts the ids
  are present rather than trusting the decoded text. An earlier "emission not
  observed" note was that same artifact, not a defect.
- Remaining per-token costs, in the order they were identified:
1. ~~`ResolveXingRouted` syncs once per MoE layer (38/token)~~ DONE, and the
   earlier "measured free" conclusion was HALF RIGHT. Skipping only the
   `cudaStreamSynchronize` really is free (64.3 vs 64.3 t/s, then 21569 vs
   21555 ms - 0.06%): a decode-loop sync costs only what it drains. What the
   experiment did NOT skip was the host resolve around it - six `assign()`
   calls plus nine pointer fields per selected expert, 38 times per token - and
   moving the whole thing to the device (uploaded per-layer expert pointer
   tables, ids read in the GEMV kernel) is worth **+3.4%**. Both halves had to
   go: the sync was free, the resolve was not.
2. ~~`XingMoeModel::Infer` round-trips the whole `[vocab]` logits row~~ DONE
   (9108ff4): the sampler now reads `session_->d_logits` in place and only the
   8-byte argmax comes back, with the row captured once for `final_logits`. The
   three per-step `cudaMalloc`/`cudaFree` are gone with it.
3. Prefill replays the prompt one token per step (no incremental/parallel
   prefill), so TTFT grows linearly with prompt length.
- **Next target, measured (2026-10-05): the mHC depth mixer, 1.32 ms/token, 15%
  of the step.** It is four launches per mHC block, two blocks per layer, so 320
  per token: proj 5.93 us (24 blocks), gates 7.06 us (ONE block), collapse 1.97 us,
  mix 1.55 us. The gates kernel is the interesting one - it is 0.56 ms/token on
  its own and it is not bandwidth: the Sinkhorn already lives in warp 0's
  registers at the frozen geometry (hc_mult=4, 16 of 32 lanes), so the cost is
  20 strictly serial iterations of shuffle-sums and divides, with the other seven
  warps idle. The obvious fix - merge gates into the wide collapse block so the
  57 KiB stream read overlaps the Sinkhorn - is the one that was already tried
  and reverted: the collapse block runs 1024 threads, whose 64-register cap makes
  the FP64 exp replica spill, and that cost more than the overlap won. The way
  through is a producer/consumer split INSIDE one launch: warps 1..7 prefetch the
  streams into shared memory while warp 0 runs the iterations, one barrier, then
  the collapse - FP64 stays off the wide path and the load overlaps the chain.
  Budget: `xing.mhc.decode` in `kernel_budgets.json` guards the current 12.5-13.0 us
  for proj+gates+collapse, so a regression here is visible immediately.
- The baseline is recorded in `tests/perf/perf_baselines.json` (mode `perf`
  and `r3`); it is still an explicit **floor**, not a tuned number, so the perf
  guard treats a regression against it as a real signal.
- Parity is unaffected: SelfCheck Paris and the FSI/CODE/DET gates pass at
  `kOff`, so this is pure throughput work.


