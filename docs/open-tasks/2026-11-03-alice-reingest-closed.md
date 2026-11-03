# AliceAI-Foundation-80B-A3B-Base re-ingestion — two accounting defects closed (2026-11-03)

The `alice` family was ingested on 2026-10-01 and merged, but its converted
vault tree was not retained on this host (`/mnt/llm_models` is empty; the
source had been re-homed to `/mnt/fast_storage/huggingface`). Re-running the
full lifecycle on the current checkout exposed two engine defects that no
consumer had exercised. Both were fixed test-first; the real-weight gates are
green and the perf rows were re-seeded through the harness.

## Defect 1 — the emitted VRAM profile was inverted

`alice_tensor_is_nvfp4` returned `true` only for the shared-expert projections,
and `ConvertAliceModel` passed `VramTensorPolicy::mem_class = nullptr`. The
converter *does* quantise the fused routed-expert stacks, but the profile
counted them as BF16 and device-resident:

- `min_bytes = 162 355 811 584` — the whole source, in "device-resident
  weights only";
- `max_bytes == optimal_bytes` — no expert pool at all.

The Alice loader ignores the `vram` block, which is why the 2026-10-01 gates
never saw it; any planner reading the index would.

Fix: mark `FusedExpertsGateUp/Down` as NVFP4 and add
`alice_tensor_is_routed_expert` (routed kinds → `VramTensorClass::RoutedExpert`),
and drop the stored-but-never-loaded `mtp.*` tree from the runtime inventory.
The emitted profile is now min 4.44 GB / optimal 31.0 GB / max 74.5 GB at
ctx 262 144.

Pinned by `tests/test_alice_moe_convert.cpp` —
`"alice vram profile streams routed experts as NVFP4"`: the assertions that
`max_bytes - optimal_bytes` equals the routed-expert NVFP4 payload and that the
routed kinds are streamed were both **red first**.

## Defect 2 — `resident_bytes` under-reported the device footprint 8×

`LoadAlicePinnedWeights` re-derived `resident_bytes` from a formula that counted
only embed/head, router gates and shared experts, omitting every KDA and
attention FP16 projection that the same function uploads. On the real
checkpoint it reported **1 187 MiB** while it had allocated **4 231 MiB**. That
number drives the governor's permanent-weight lease and the `resident:` line of
`lsxbenchmark --bench`.

Fix: accumulate at the allocation sites (`H`, `W` and the global uploads) and
assign the total. The loader now matches the converter's independently computed
`vram.min_bytes` exactly.

Pinned by `tests/test_alice_moe_loader.cpp` —
`"alice resident bytes count every uploaded weight"`: on the synthetic fixture
the loader reported `47 616` against the profile's `366 384` **before** the fix,
and the equality is asserted after it.

## Coverage added in the same pass

- `test_alice_moe_selfcheck` now runs the protocol's ≥3 stress cycles with a
  `memcmp` on `final_logits` (`LSX_TEST_ALICE_SELFCHECK_CYCLES`, floored at 3).
  It previously did a single `Infer`.
- Stage 4b gained its real-weight half: a `[alice][tool][real-model]` case using
  the shared `tests/test_tool_roundtrip.h` helper. The native codec emitted and
  parsed `get_weather {"city": "Paris"}`.

## Real-weight result (2026-11-03, RTX 5090, CUDA 13.4, AOCC release)

- Conversion: 151.4 GiB BF16 source → 45.6 GiB `logestix-alice-nvfp4` in 4 min
  05 s (2 745 pinned records + 24 576 expert files).
- SelfCheck "Paris" ×3, bit-exact, `logits_hash f900547d25be7914` (129 024
  logits); FSI 3/3, CODE 3/3, DET 3/3.
- Tool round-trip: `get_weather {"city": "Paris"}`.
- Perf by the harness (measured twice, identical): **45.70 t/s** (`perf`) /
  **45.88 t/s** (`r3`) at 2 048 tokens; TTFT 475 / 470 ms; 181–198 W.
- Residency: 4.23 GiB device-resident, 2.9 s load; routed experts streamed
  from the host store.
- `ctest -L fast` 130/130, `-L convert` 37/37, Alice inference 7/7.

No open debt for this family: all three parity goldens are present and green.
Full report: `docs/model-reports/2026-11-03-AliceAI-Foundation-80B-A3B-Base.md`.
