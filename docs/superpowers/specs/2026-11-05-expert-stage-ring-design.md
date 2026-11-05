# Expert staging ring in the shared `ExpertStore` — design

Status: proposed (awaiting operator approval)
Date: 2026-11-05
Branch: `perf/expert-stage-pipeline`
Supersedes: `docs/OPEN_TASKS.md` §R17 **P6** as written ("overlap the expert
staging with the compute") — see *Why P6 as written is not doable*.

## Why

The shared store is the streaming path for the whole repository. Verified by
grep rather than remembered — 15 `ExpertStore<TEntry>` aliases:

| alias | family / variant | streaming or resident |
|---|---|---|
| `KimiK3ExpertDb` | kimi-k3, kimi-k3.1-pro | **streaming** (1.4 TB tree) |
| `KimiExpertDb` (`kimi_moe_load.h`) | kimi-k2.7 | streaming (542 GB tree) |
| `KimiExpertDb` (`kimi_moe_expert_cache.h`) | kimi-k2 expert cache | streaming |
| `ExpertDb` (`deepseek_moe_v41_load_internal.h`) | deepseek-v4 / v4.1 / pro | **streaming** (811 GB) |
| `GlmExpertDb` | glm-5.3-flash | streaming |
| `MinimaxExpertDb` | minimax-m3, m3.1-flash | streaming |
| `MoEExpertDb` (`lsx_moe_inference.h`) | qwen-moe, gemma-moe, ornith, mistral-small | streaming |
| `GemmaExpertDb` | gemma-4 moe | streaming |
| `GcExpertStore` | gigachat | streaming |
| `NmExpertStore` | nemotron | streaming |
| `MimoExpertStore` | mimo | streaming |
| `AliceExpertDb` | alice | streaming |
| `BeamExpertDb`, `LingExpertDb`, `StepExpertDb`,
`MistralLarge4ExpertDb` | beam / ling / step / mistral-large4 | streaming |

Families whose routed experts are **device-resident** (`flint`, `hy`, `xing`)
never call the store and are out of scope; they are listed here only so the
builder can prove they are untouched.

Every family in the left column is bound by the same path, and the path is
slower than the disk under it by a factor of eight.

Measured, not assumed. Two instruments:

**1. A cold-cache probe against the real Kimi-K3.1-Pro expert files**
(`/tmp/opencode/expert_probe/stage_probe.cu`, 16 experts × 6 planes = one
layer, 283.5 MiB, page cache dropped via `munmap` → `POSIX_FADV_DONTNEED` →
`mmap` before every run; the cold/warm control shows 4.3× so eviction is real):

| arm | ms/layer | GB/s | s/token (92 layers, copies only) |
|---|---|---|---|
| pageable, 6 `cudaMemcpyAsync` — **what the code does today** | 54.8 | 5.2 | 5.05 |
| pageable, 2 copies (two contiguous file spans) | 54.1 | 5.3 | 4.97 |
| `cuMemcpyBatchAsync`(6), `SRC_ACCESS_ORDER_STREAM` | 72.3 | 4.2 | 6.56 |
| `cuMemcpyBatchAsync`(6), `ANY` | 59.8 | 5.0 | 5.47 |
| batch(6), `ANY` + `PREFER_OVERLAP_WITH_COMPUTE` | 61.4 | 5.0 | 5.45 |
| batch(96) per layer | 61.0 | 4.9 | 5.55 |
| pinned: `pread` 1 thread + batch(6) | 66.4 | 4.5 | 6.04 |
| **pinned: `pread` 4 threads + batch(96)** | **26.9** | **10.5** | **2.48** |

Batching the copies is a **loss** in every variant: the cost was never per-call
submission. Within the winning arm, `pread` is 21.35 ms of the 26.9 (13.9 GB/s
at 4 threads, against a measured 39.8 GB/s array ceiling) and the pinned DMA is
the remaining ~5.5 ms (**54 GB/s**).

**2. nsys on a live 64-token run of Kimi-K3.1-Pro** (release build, 321.6 s of
generation, 89 model steps):

| | |
|---|---|
| wall | 341.1 s |
| GPU busy (union) | 181.3 s — **53%** |
| host busy inside CUDA API | 226.7 s — **66%** |
| host/GPU overlap | 66.9 s — 20% |
| **GPU idle** | **159.9 s — 47%** |
| H2D traffic | 3076.3 GiB in 870 542 copies, **19.4 GB/s** |
| kernel time | 10.8 s — **3%** of wall |
| time inside `cudaMemcpyAsync` on the host | 222.6 s of 321.6 s (925 953 calls, 240 µs each) |

The last row is the mechanism: a pageable `cudaMemcpyAsync` is **not
asynchronous**. The driver copies the pageable source into an internal pinned
buffer during the call, so the host blocks for the whole copy and the two halves
(host→page-cache and page-cache→VRAM) never overlap. Copy sizes confirm the
parse: 393 441 × 5.25 MiB (packed planes) + 393 048 × 0.66 MiB (scales) =
1472 experts × 89 steps, i.e. 92 MoE layers × top-16 per token.

So the traffic per token is 37.1 GB against 39.8 GB/s of disk: the floor is
~0.93 s/token, and the model records 0.31 t/s (3.2 s/token). **That is a 3.4×
gap on a path whose only bottleneck is how we move bytes from NVMe to VRAM.**

## Goals

- Move the streaming path onto pinned staging with parallel `pread`, so that the
  decode rate stops being set by pageable-copy serialisation.
- Put it in the **shared** `ExpertStore` so every streaming family benefits
  (15 aliases, table above), with one instrument and one set of hazard rules
  instead of one per family.
- Keep numbers identical: the change is scheduling and plumbing only. Same
  bytes, same kernels, same accumulation order.
- Make the improvement a machine-checked number, not a claim.

## Non-goals

- Changing kernels, the NVFP4 layout, or any family's device staging layout.
- GPUDirect Storage: `nvidia_fs` is not loaded and `libcufile` is absent on this
  host — verified, so it is out of scope here (recorded, not silently dropped).
- The `cuMemcpyBatchAsync` idea: measured a loss, so it is used only where it
  already wins (whole-layer submit, ≤1 call per layer, see below) or not at all.
- Prefetching the *next layer's* experts (impossible — see below).

## Why P6 as written is not doable

"Stage layer N+1 while layer N computes" cannot be implemented, for a reason
that is structural rather than technical: **the expert ids for layer N+1 are a
function of layer N's output.** The router reads the MoE input, which is only
complete after layer N's routed FFN and combine have run. There is nothing to
stage until the layer that would have been overlapped has finished.

What *is* available, and what this design exploits, is pipelining **within** a
layer: while `pread` fills pinned buffer k+1, the DMA of buffer k is in flight,
and both overlap the (small) GEMV of expert k-1. That is where the 2× comes
from, and it is bounded by `37.1 GB / max(pread_bw, dma_bw)` — currently
~2.5 s/token at 4 pread threads, ~1.3 s at 16, ~0.93 s at the array ceiling.

`OPEN_TASKS.md` §R17 P6 has been corrected to say this.

## Approaches considered

**A — ring in the shared `ExpertStore`** (chosen). The store owns pinned buffers,
the `pread` pool, the DMA and the slot lifetime; families keep their device
layout and ask the store to stage an entry into it. One instrument, nine
beneficiaries, one set of hazard rules. Cost: the contract of the most-shared
header in the tree changes, so every consumer must be proven unchanged.

**B — ring local to `kimi_moe_k3_inference_pipeline.cu`.** ~1 day, no shared
risk, proves the number fast. Rejected as the *end state*: eight other families
stay at 5.2 GB/s and a second implementation of the pattern appears, which is
exactly what the neutral-store consolidation removed. Still useful as a spike if
A's measurement is ambiguous.

**C — `pread` into pinned + async DMA, no ring (sequential).** Smallest diff,
~1.7× by estimate (0.31 → ~0.53 t/s) because it removes the pageable
serialisation but keeps read and DMA in lockstep. Rejected: it is A minus the
half that pays for the work.

Within A, two sub-variants, and the design takes the second:

- **A1 whole-layer barrier**: `pread` all 16 experts, then one DMA batch, then
  the GEMVs. Measured (26.9 ms) but it leaves the read and the DMA serialized.
- **A2 chunked pipeline** (chosen): a ring of `depth` slots; `pread` of slot k+1
  is issued while the DMA of slot k is in flight. Same bytes, more overlap. Depth
  and chunk size are *measured* parameters, not guesses — the probe's next run.

Also measured and rejected inside A2: `O_DIRECT` for the `pread` (would halve
the host memory traffic by bypassing the page cache) is kept as an experiment,
not a default: it needs 4 KiB-aligned buffers and its interaction with the
existing `mmap`-based parse is unproven on this host.

## Design

### What the store knows, and what it doesn't

The store already decodes every entry through `ExpertSchema` (9 columns, in
file order `w1,w2,w3,w1s,w2s,w3s`). What it does not know is *where in the file*
those planes live, which is all `pread` needs.

**Plane descriptors.** At `ParseFile` time the store records, per flat index,
six `{file_offset, bytes}` pairs. The offset comes from the Arrow batch's own
buffers: `buffer->address() - mapping_base`, where `mapping_base` is obtained
once per file from a zero-copy `ReadAt(0, n)` — the returned slice's address is
the mapping base because Arrow slices the mmap rather than copying. Reading
stays through `MemoryMappedFile` (Arrow guard rule R-A1 keeps `ReadableFile`
forbidden and raw `mmap` out of consumer code). The base derivation is guarded
by a **sanity check that must fail loudly**: offsets strictly increasing,
`offset + bytes <= file size`, and the packed planes exactly
`3 × packed_bytes` with scales `3 × scale_bytes` — a wrong base produces garbage
in every one of those three checks rather than a silent bad copy.

The descriptors are family-neutral (`ExpertSchema` order), so no family code
changes and no per-family layout table appears.

### The API

Added to `ExpertStore<TEntry>`; no existing method changes behaviour.

```cpp
// Configuration (set before Load).
struct StageRingConfig {
    size_t slots = 0;          // 0 = disabled, the caller keeps its own copies
    int pread_threads = 0;     // 0 = auto (measured default, see below)
    size_t chunk_experts = 0;  // experts per pipeline step; 0 = whole layer
};
StageRingConfig stage_ring;

// Queues staging of one resident entry into the caller's device buffer.
// `planes` names the destination offset of each of the 6 schema planes, so a
// family keeps its own device layout ([w1|w2|w3|s1|s2|s3] for kimi,
// [w1|s1|w2|s2|w3|s3] for others — whatever it already uses).
// Returns false when the ring is disabled or the entry is not resident; the
// caller then falls back to its existing cudaMemcpyAsync path unchanged.
bool EnqueueStage(size_t idx, uint8_t *dst_dev,
                  const int64_t (&dst_off)[6], cudaStream_t stream);

// Blocks until the entry queued by EnqueueStage is readable by `stream`, by
// having the stream wait on the slot's DMA-complete event. Never drains the
// device.
void AwaitStage(size_t idx, cudaStream_t stream);
```

The caller keeps its `EnsureResident` + plane-pointer bookkeeping; only the
"get these 6 spans from the file to VRAM" step moves into the store.

### The ring

- `slots` pinned buffers, each `max_expert_bytes` (per-store; the largest
  resident entry), allocated once with `cudaHostAlloc`. `ulimit -l` is
  `unlimited` here; the store checks it and fails loudly with the sentence that
  names pinning, because the failure mode is otherwise "every later CUDA call
  fails" (already documented in `PinChunked`).
- A worker pool of `pread_threads` issues `pread(fd, slot + off, len, file_off)`
  for each plane. The fd is `MemoryMappedFile::file_descriptor()`, so no second
  open and no descriptor bookkeeping.
- Slot lifetime: **a slot cannot be refilled while a DMA reading it is in
  flight**. Each slot carries a `cudaEventDisableTiming` event; recycle requires
  `cudaEventQuery` on that slot only. This is the same lesson already paid for
  in `RetireSlot`/`ReapRetired`: check the front of a FIFO, never scan — a full
  scan cost a measured 6.6% of a 2048-token mistral-small-4 run.
- DMA: one `cudaMemcpyAsync` per plane per expert, from pinned (the measured
  54 GB/s path), on the caller's stream. **No `cuMemcpyBatchAsync`** — measured
  slower than plain copies at every size and attribute combination tested.
- The pipeline is driven from the family's loop: `EnqueueStage(k+1)` then
  `AwaitStage(k)` then GEMV(k). `AwaitStage` is `cudaStreamWaitEvent`, so the
  host never blocks.

### Interaction with the existing eviction machinery

Staging reads **from the file descriptor**, not from the mapping: `pread` needs
only `{fd, file_offset, length}`, and the tree is immutable. That decouples the
ring from `RetireSlot`/`ReapRetired` entirely — an expert may be evicted, its
mapping unmapped and retired, while a staging `pread` for it is still in flight,
and nothing breaks. The builder must not add a drain hook for staging: the
existing `pending_drain` hook exists for families whose *DMA* sources the mmap
directly and does not apply here.

What does keep the current coupling: `EnqueueStage` requires the entry to be
resident (`entries[idx].host_w1 != nullptr`) so the plane descriptors and sizes
are known. An evicted entry returns `false` and the caller falls back — the
store never re-parses behind the caller's back.

### When slots are fewer than the layer's experts

`slots < experts_per_layer` is legal and expected (the budget is pinned RAM).
`EnqueueStage` then recycles the oldest slot, which requires waiting for that
slot's DMA event — as a **stream wait**, never a host sync, or the win is gone.
If no slot can be recycled without blocking the caller's stream in a way that
would serialize the whole layer, `EnqueueStage` returns `false` and the caller
uses its legacy copy for that expert: a partially engaged ring is slower than
either pure path, so `slots` must be documented as "at least 2, ideally ≥ the
layer's top-k".

### Engagement counters

The store counts `staged_experts`, `staged_bytes`, `legacy_fallbacks` and
`ring_waits`. Rollout needs them: without a counter, "the ring is engaged for
this family" is an assumption, and a family silently falling back would look
like a change that simply did not help. Logged at decode end when
`LSX_EXPERT_STAGE_RING=1`.

### Hazard rules (each becomes a failing test first)

1. **A pinned slot is never refilled while a DMA that reads it is in flight.**
   The test fills a slot, issues a DMA, immediately enqueues a second expert into
   the same ring with `slots = 1`, and requires that the first copy's bytes
   arrive intact — a ring that recycles on the host's schedule corrupts silently,
   which is the worst failure mode this store can have.
2. **A plane descriptor always addresses the plane it names.** Synthetic
   fixture where each plane has a distinct byte pattern; every staged expert is
   compared byte-for-byte against the file after `AwaitStage`. A wrong base
   offset would move `w1_scales` into `w2_packed` and produce a plausible,
   wrong model — this is the check that makes the offset derivation safe.
3. **The ring never changes the staged bytes.** Same expert staged through the
   ring and through the legacy path must be `memcmp`-identical, for the synthetic
   fixture and for one real expert file.
4. **Disabled ring is a no-op.** With `slots = 0` every path behaves exactly as
   today (guards the fallback, and is what every family gets before opt-in).
5. **Bandwidth budget** (`kernel_budgets.json`, kind `min`): staged bytes per
   second through the ring, per `gpu_key`. A bare floor is deliberately avoided —
   the existing helpers' rule is that a number without a machine is worthless.
6. **Numeric neutrality, end to end**: `test_kimi_pro_selfcheck` Paris ×3
   bit-exact (`logits_hash` unchanged), which is the gate for any change touching
   this path, plus the family's determinism gate.

### Knobs

- `LSX_EXPERT_STAGE_RING=0|1` — off by default until the A/B numbers are in.
- `LSX_EXPERT_STAGE_SLOTS=<n>`, `LSX_EXPERT_STAGE_THREADS=<n>`,
  `LSX_EXPERT_STAGE_CHUNK=<n>` — the three measured parameters.
- Knobs resolve through `gpu_perf` like every other perf knob (env > test CLI >
  calibration cache > passport > base), so a narrow-bus GPU can turn the ring
  off where pinned staging does not pay.

### Rollout

1. Ring + tests, disabled by default. **No family is wired yet** — the store
   gains a capability nobody calls, so every `test_<model>_selfcheck` must stay
   green with the flag absent and with it present-but-unwired. That is the
   proof the shared-header blast radius is zero.
2. **Kimi-K3.1-Pro is the only family wired** (`kimi_moe_k3_inference_pipeline.cu`
   calls `EnqueueStage`/`AwaitStage` in place of its 6-copy loop); measure
   `--max-gen 2048` per the canonical protocol; the budget test and the perf row
   disagreeing is a bug in one of them.
3. Thread count, chunk size, depth swept on Kimi (the parameters above).
4. Only then wire the remaining streaming families (14 aliases, table above) and
   decide the default. `flint`/`hy`/`xing` never call the store and must be shown
   untouched by their guards.

## What is left to the builder

Measured, not decided here:

- `pread_threads`: 4 threads gave 13.9 GB/s against a 39.8 GB/s array ceiling.
  Sweep 4/8/16/32 — expect the knee near the array limit.
- `slots` and `chunk_experts`: the trade is pinned RAM (18.6 MB per Kimi slot)
  against overlap. Depth 2 is the minimum for any overlap; more only helps if
  the DMA and the read are within ~1.5× of each other.
- `O_DIRECT`: measure; do not assume.
- Whether the page cache should be told anything at all: the earlier
  `fadvise(WILLNEED)` and page-cache prefetch experiments were both **negative**
  (§R17 P5) and must not be resurrected by accident.

## Risks

- **Shared-header blast radius.** `expert_store.h` is included by 15 families;
  a compile-time or behavioural mistake reaches all of them. Mitigated by: off by
  default, the disabled path byte-identical, and `test_expert_store*` +
  `test_model_suite_guard` running in `fast`.
- **Pinned memory.** A ring of N slots × 18.6 MB (Kimi) per store is small, but a
  family with a much larger entry (mistral-large4's 21.3 MiB × 4 planes) must be
  checked against the governor's VRAM/locked-memory budget. Configuration is
  explicit, never inferred from a percentage.
- **The measured number may land below the estimate.** If pread saturates at
  ~20 GB/s instead of 39.8, the ceiling is 1.9 s/token (0.53 t/s) — still a 1.7×
  win, and still worth landing. This is stated up front so a smaller result is
  read as the array's behaviour, not as a failed change.

---

## Outcome of execution (2026-11-05)

Implemented, tested, bit-exact, and **neutral on this host**. The ring is off by
default; `kimi-k3.1-pro` is wired behind `LSX_EXPERT_STAGE_RING`.

### What was built

- `PreadPool` (`pread_pool.h`) — persistent N-thread pread, short reads not
  counted as complete, `Run` waits for **all** workers.
- Plane locators per resident expert, derived from the decoder's own pointers
  (one owner for the column→plane mapping) and gated three ways, so a mapping
  base off by the Arrow header is loud instead of silent.
- The ring: `InitStageRing` / `PrefetchStage` / `StageFromSlot` / `AwaitStage`,
  per-slot DMA events. Slot reuse waits **host-side** — see "the defect the
  doctoring found" below.
- Env knobs + engagement counters, and a machine-keyed bandwidth floor
  (`expert_store.stage_ring_gbps`, measured 9.36 GB/s, floor 7.49).

### Measured, on the real tree

Paired runs, same binary, same prompt, 32 and 64 tokens, ring off as control:

| arm | 32 tok | 64 tok | notes |
|---|---|---|---|
| ring off (control) | 122.5 s, 124.6 s | 184.1 s, 264.5 s† | †first run of the session, cold cache — not comparable |
| ring on, per-expert refill | 163.9 s | — | 1472 pinned slots = 27 GiB locked; alloc alone 14 s |
| ring on, 64/128 slots | 150.9 / 150.2 s | — | still worse than control |
| **ring on, layer-wide prefetch, 16/32 slots** | **119.4 / 121.2 s** | — | ≈ control, inside noise |

nsys at 32 tokens, ring off → on:

| | off | on |
|---|---|---|
| H2D copies | 558 478 | 558 478 |
| H2D bytes/run | 1978.3 GiB | 1978.3 GiB |
| H2D effective bandwidth | 23.7 GB/s | 22.3 GB/s |
| host time inside `cudaMemcpyAsync` | 118.0 s | **45.3 s** |
| kernel time | 6.8 s | 6.8 s |

The host-side copy cost genuinely drops by 62%. The wall does not move.

### Why neutral — two measured reasons

1. **This host serves the reads from the page cache.** 665 GB of the 750 GB of
   RAM sits in buff/cache, and the whole per-token working set is far smaller
   than that, so the pageable path already runs at 23.7 GB/s — not the 5.2 GB/s
   the cold-cache probe measured, and not the 19.4 GB/s an earlier profile saw in
   a colder state. Pinning cannot beat a copy that is already cache-served.
2. **Expert staging is a minority of the traffic.** 61.8 GiB/token goes H2D;
   routed experts are 25.5 GiB of it (41%) and the remaining **36.4 GiB/token is
   the streamed latent/shared/dense plane**, which never goes through
   `ExpertStore` at all. A *perfect* expert path therefore touches 41% of the
   copy time at most, and on this host that 41% is already at cache speed.

### The two defects the doctoring found

- **Slot reuse cannot be a stream wait.** The refill writes the pinned buffer
  from the host with pread, so `cudaStreamWaitEvent` orders device work while the
  host overwrites the buffer a pending DMA will read. The hazard test passed
  against a ring with no check at all until it was given payloads that differ per
  expert and a busy stream; then the doctored run showed expert 0's buffer
  holding expert 1's bytes. Reuse is now `cudaEventSynchronize`, which is rare
  because slots ≥ top-k makes recycling a once-per-layer event.
- **`warm_on_load = false` means no plane geometry at init.** This family
  disabled the eager pre-parse (§R17 P5), so the ring logged "no plane geometry"
  and every expert silently fell back — a change that looks like it simply did
  not help. One expert is now parsed at ring init; the regression test fails
  without that.

### Not done, stated as debt

- **Canonical 2048-token rows were not run** for either arm. At the measured
  rates that is ~2.1 h per arm, and the 32/64-token pair plus the mechanism above
  already answers the question the rows would answer. They are the quotable
  form, so `§R17 P6` records them as `not run`, not as a pass.
- **GPUDirect Storage** remains unavailable on this host (`nvidia_fs` not
  loaded, no `libcufile`), and it is the one lever that would remove the host
  copy from the 59% instead of optimising the 41%.
- **The cold-cache win is unverified end to end.** The probe's 26.9 ms vs
  54.8 ms per layer is a real cold measurement; reproducing it on the model
  needs a machine whose page cache cannot hold the tree (or root to drop it).
