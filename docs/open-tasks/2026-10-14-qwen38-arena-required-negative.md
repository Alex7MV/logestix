# qwen3.8-flash-next: the arena is required (V1, closed negative, 2026-10-14)

Item **§V1** of [`../OPEN_TASKS.md`](../OPEN_TASKS.md), the follow-up the
ExpertStore migration snapshot left open ("qwen3.8 decode -3.5x"). Resolved
the same day as a negative result: for a model whose decode working set is the
whole expert set, the pinned arena is the correct design and mmap views are a
net loss. Kept verbatim, measurements included.

### V1. RESOLVED (negative): the arena is required when the working set is the whole expert set

Tried the bounded pinned window §V1 called for, measured it, and it does not
work. qwen3.8-flash-next, real weights, `/usr/bin/time -v`, same prompt:

| configuration | peak RSS | throughput |
|---|---|---|
| pinned arena (original) | 63.6 GB | 32.1 t/s |
| views, cap 4096 | 10.9 GB | 9.2 t/s |
| views + pinned window, cap 2048 | 10.9 GB | **9.2 t/s — no speedup** |
| views + pinned window, cap 24576 | 127.2 GB | 29.0 t/s |

The window pins 5400 MiB in 754 ms and buys nothing, because the decode working
set is LARGER than the cap: nearly every touch is an expert that was evicted
and re-mapped, hence unpinned. Speed only returns when essentially everything is
pinned — and then the mmap views are still mapped beside the pinned copy, so it
costs 2x the arena (127 vs 63.6 GB) for slightly less speed. For a model whose
decode working set is the whole expert set the arena is the right design, and
the migration is a net loss. This is a resource constraint, not a defect.

`cudaHostRegister` over the mmap views is not a way back: refused for every
entry (`registered=0 failed=512`), because a file-backed MAP_SHARED mapping is
not registrable. Three sticky-CUDA-error traps on the way there, each of which
presents as "generation dies right after a successful load" with nothing in the
message pointing at pinning: (1) a non-page-aligned start is rejected; (2) the
six payload spans of one expert are adjacent in the file, so aligning each down
to a page makes neighbours overlap and the second registration returns
`cudaErrorHostMemoryAlreadyRegistered`; (3) unregistering a range that was never
registered also raises a sticky error.

qwen3.8 is therefore back on its arena (63.6 GB, 32.0 t/s re-measured). The
neutral store stays for qwen MoE / Ornith / Mistral, whose working set fits the
cap: 17.5 → 7.0 GB at unchanged throughput.
