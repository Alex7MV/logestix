# Golden / Performance matrix (R3 vs default)

Measured 2026-09-15, RTX 5090, AOCC build `linux-release-znver4-aocc`.
Per model:
- **perf** = default (`kAuto`, per-GPU max-speed policy).
- **R3** = `--bit-exact` (`FastMathMode::kOff`, explicit R3/C-exact switch).
- hash = FNV-1a of the selfcheck `final_logits` (see `numeric-conformance-audit.md`).

| Model | perf t/s | R3 t/s | perf hash | R3 hash | tok |
|---|---|---|---|---|---|
| GLM-5.3-Flash | 33.6 | 18.9 | 358d6dcbc80aac34 | b4df9a1e73b05b42 | 12089 |
| DeepSeek-V4-Flash | — | — | dfc943649788e218 | 7f6c3d35966839c6 | — |
| DeepSeek-V4.1-Flash | — | — | 9f4edd4db34ddad7 | 08eeeb86e464e71e | — |
| Kimi-K2.7-Code | — | — | d340d91e89467bcc | f9e51ad5c57eb1ff | — |
| Gemma-4-26B-A4B-it | — | — | c06c0f18ae3cabda | f343ce41dd246552 | — |
| Qwen3.6-35B-A3B | — | — | 5209e2295509aa7d | 4759fe1466ffcf89 | — |
| Qwen3.8-Flash-Next | — | — | 139814718cb71a0c | 7dd22315feb10d6f | — |

Protocol: `--bench --max-gen 2048` and `--selfcheck [--bit-exact]`. Speeds are the
canonical 2048-token decode; hashes are from the 2-token selfcheck.

## Findings

1. **`perf` hash != `R3` hash for every model.** The default (`kAuto`) numeric
   path is distinct from the `--bit-exact` (`kOff`) path on all seven models,
   even though both produce the golden token. Per the contract this classifies
   `kAuto` as **C-fast** for every model (it does not reproduce the `kOff`
   bit-logits), which conflicts with §3.4 ("default `kAuto` must resolve to
   C-exact for FSI models"). **Operator decision required** (see below).
2. **Hashes are build-specific** (host toolchain), as the contract's toolchain
   clause anticipates. Cross-model comparison must use the same build on both
   boxes; the numbers above are the AOCC-build golden.
3. GLM-5.3-Flash shows the R3-vs-perf throughput cost of the strict path:
   33.6 -> 18.9 t/s (~1.8x).

## Cross-model verification procedure (per model, per mode)

On the 5090 record the hash; on the 5060 Ti run the same build and compare:

```
LSX_SELFCHECK_EXPECT_HASH=<5090 hash>  <lsxbenchmark> --model <vault> --selfcheck [--bit-exact]
```

Match -> same numeric path across models (R3 cross-model holds). Mismatch ->
device-dependent numeric path (fix-list #2, e.g. cuBLAS algorithm).

## Operator decision (from finding 1)

- (a) Make the FSI default **be** the `kOff` path (true C-exact, slower:
  GLM 18.9 vs 33.6 t/s), or
- (b) Redefine R3 as "deterministic + golden tokens" (not bit-identical to
  `kOff`), which makes `kAuto` R3-compliant and keeps performance — but weakens
  the "bit-identical logits" wording in §2.
