# Routed-expert host storage: the arena is retired (2026-10-14)

Item **§V** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md), split on the date the
migration landed. The migration itself is closed; the two follow-ups it exposed
stay open in §V there.

## What the defect was

Routed experts are the largest thing a model owns, and four families kept their
**entire** expert set in pinned host RAM for the life of the session:

- `lsx_moe_inference.cu::LoadMoEExpertStore` — one pinned slab per layer, used
  by qwen MoE, Ornith and Mistral;
- `qwen_moe_exp_load.cpp::LoadQwen4ExpExpertStore` — a single pinned arena for
  qwen3.8-flash-next, plus a **second full disk pass** (pass 1 measured the
  records, pass 2 filled the arena).

Retained host bytes ≈ the expert tree on disk: 64 GiB (qwen3.8), 61 GiB
(mistral), 17 GiB each (qwen3.6, Ornith). It survived review because the host
has 750 GiB of RAM, so `cudaMallocHost` of tens of gigabytes succeeds silently.
The same class had already shipped once: Alice's `AliceExpertHost` was an
unbounded owning map, measured as a monotonic RSS climb of 8.3 → 12.7 → 18.0 GB
peak over 32/128/512 generated tokens (~21 MB/token), fixed in `4d98fe2`.

## What was done

`MoEExpertRecord` now holds immutable **geometry only** (six byte counts + three
global scales) — nothing that can dangle. The payload lives in the neutral
`MoEExpertDb` (`ExpertStore<MoEExpertEntry>`), and the device-slot staging copies
six segments from the mmap views. `EnsureMoEExpertView` maps an expert in (LRU
tail retired past `max_open`); `EnsureMoEExpertResident` keeps its old name and
meaning, since staging into a device slot is a different job from mapping.

The invariants the tests now pin:

- geometry survives eviction, payload does not — the device-slot sizing loops
  read `records[]` long after an expert has been retired, and a pointer there
  would be a stale-pointer bug;
- an entry is a view: trivially copyable, ≤128 bytes, no byte buffers;
- the LRU really evicts (which is why `max_open` must be passed at load time —
  `Load()` warms up to the cap, so raising it afterwards asserts nothing; the
  first version of that test did exactly that and failed).

`tests/test_expert_store_guard.cpp` (label `fast`) locks the result: no owning
container of an expert payload type, no `…ExpertHost`/`…ExpertTable`
identifier, every `*ExpertEntry` file aliases `ExpertStore<`, and the legacy
`MoEExpertStore` consumer list is frozen so the design cannot spread by copy.

## Measurements (real weights, same prompt, `/usr/bin/time -v` peak RSS)

| model | before | after | throughput |
|---|---|---|---|
| qwen3.6-35b-a3b | 17.5 GB, load 61 s | **7.0 GB** | 141.4 → 141.1 t/s (flat) |
| qwen3.8-flash-next | 63.6 GB, load 73.7 s | **10.9 GB** (cap 4096), load 9.9 s | 32.1 → **9.2 t/s** |

`max_open` is now the host-memory knob and scales it linearly on qwen3.8:
512 → 1.7 GB, 2048 → 5.6 GB, 4096 → 10.9 GB. Exposed as
`LSX_MOE_EXPERT_MAX_OPEN` / `LSX_QWEN38_EXPERT_MAX_OPEN`.

Gates after the change: fast 94/94; qwen3.8 FSI / CODE / DET / selfcheck all
pass on real weights; Ornith and qwen3.6 selfcheck pass on real weights.

## Two things found while measuring (both left open in §V)

1. **qwen3.8 decode lost 3.5x, and the cause is pinned H2D staging.** Isolated
   by one extra run: mapping all 24576 experts *without* pinning gives 26.7 GB
   and 17.3 t/s, so ~half the loss is evict/re-parse and ~half the DMA path.
   The arena was not only memory. `cudaHostRegister` is not a way to recover it
   — it refuses every entry of a file-backed mapping — so the fix is a bounded
   anonymous pinned arena (`PinChunked` capped at `max_open`), which is the next
   piece of work.
2. **A claimed `max_open` defect did not reproduce.** An early note said Mistral
   peaked at 65.6 GB regardless of the cap. Re-measured twice per point it is
   monotone and saturating (3.2 / 7.0 / 31.1 GB at cap 64 / 512 / 8192), and a
   load with generation disabled peaks at 1.8 GB — the geometry pass costs no
   payload pages, because `DecodeEntry` takes Arrow views. Recorded in §V2 as
   closed rather than quietly dropped: the failure mode was believing a single
   un-repeated measurement.

## Deliberately not migrated: glm-5.3-flash

Its tree deviates from the neutral store's conventions three ways (dense prefix
`kMoeFirstLayer = 3`, store slot = layer − first_moe_layer, zero-padded
`exp_000.arrow` / `layer_003`) and there is no converted tree on this machine to
verify against, so migrating it would be unverifiable. Its arena fields are
renamed `slab_*_off` / `h_slabs` so the arena-ness is visible on sight and can
never be mistaken for the view path.