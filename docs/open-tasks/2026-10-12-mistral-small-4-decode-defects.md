# Mistral-Small-4-119B — the decode defects found on real weights (2026-10-12)

Session branch `feat/mistral-small-4-119b`. Moved out of `docs/OPEN_TASKS.md` §T
on 2026-11-05: §T remains open (its three FSI/CODE/DET goldens are declared debt
because the source cannot run here), but everything below is **resolved**, and
the live list is supposed to carry only open work.

Five defects plus three follow-ups, each isolated against a host NVFP4 reference
built from the source safetensors. They are kept here in full because several are
"the same class as" traps that recur in other families, and because two of them
were found only by *disbelieving an earlier attribution* — the 44x shared-expert
under-contribution and the "neutral router NaN" that was never the router's.

- **Real-weight decode: fixed (2026-10-12).** `test_mistral_moe_selfcheck`
  passes (Paris + tool round-trip + 3 bit-exact stress cycles); `--bench`
  completes 2048 tokens at **14.68 t/s** (Release, `r3` and `perf` both;
  the `linux-debug` preset is 45 t/s but a Debug binary is not a valid perf
  record — `test_perf_baseline` SKIPs outside `NDEBUG`); SelfCheck passes in **both** the auto
  and `--bit-exact` (`FastMathMode::kOff`) modes. The layer-4 "outlier" that
  this entry previously reported was a **measurement artefact** — the expert is
  a genuine massive-activation expert (host reference at layer 4 confirms
  routed max 411.9 vs the 403.8 the GPU produced). Five real defects, each
  isolated against a host NVFP4 reference built from the source safetensors:
  1. **Shared expert through the BF16 GEMV.** `attn::RunAttnMLP` reinterpreted
     the shared expert's E2M1 nibbles as BF16: shared rms 0.0080 vs the host
     reference 0.3495 (**44x** under-contribution). Fixed by running the
     neutral NVFP4 seq==1 GEMV (`MistralSharedExpert`, the Xing pattern —
     `RunMoESharedCombine` is gated and does not apply, this model's shared
     expert is ungated). Regression test
     `test_mistral_moe_kernels` "matches the host NVFP4 reference" (mutation
     checked: reverting the scale makes it fail).
  2. **Autodetect name mismatch.** The converter wrote
     `"model": "Mistral-Small-4-119B-2603"`, which `detect_model_type`
     normalizes to `mistral-small-4-119b-2603` — not the registered kType, so
     `lsxbenchmark`/`lsxfabric` could not resolve the model by path at all.
  3. **Expert-cache VRAM over-commit.** A hand-rolled budget with a 2 GiB
     reserve resolved 1838 slots (~25 GiB) and the 2048-token bench OOMed on
     the KV arena. Now the shared `lifecycle::ExpertCacheSlots` (8 GiB
     calibrated reserve + the `expert_slots_pct` knob) → 1382 slots.
  4. **FP16 SwiGLU overflow (the actual NaN).** This model carries massive
     activations: layer 32 expert 67 reaches `|silu(g)*u| = 2.5e5` against the
     FP16 ceiling of 65504 — exactly 4 values overflow, and the inf propagated
     through the down projection into the residual. The scalar (non-TC) MoE
     path stores that product in FP16. `RunMoERoutedFFN` and
     `MistralSharedExpert` now evaluate the product in FP32 and store it under
     a power-of-two prescale, folded back in the FP32 accumulator
     (`SwigluF32Kernel`/`SwigluAbsMaxKernel`/`SwigluAlphaKernel`/
     `SwigluStoreScaledKernel` in `lsx_kernels_act.cuh`, plus
     `moe_coeff_add_scaled_kernel` / `acc_add_half_scaled_kernel`). It is a
     **provable no-op for every existing model** (alpha == 1 whenever the
     product fits FP16, and the value expression is the same F32 silu). The TC
     path was already immune — it quantizes from FP32 — which is why only
     `--bit-exact` diverged. Regression test: "survives an FP16-overflowing
     SwiGLU" (mutation checked).
  5. **FP16 residual + session leak.** The residual itself reaches ~2.4e5 at
     layer 32 (host reference), past FP16 range, so `d_h` is now FP32 (the
     norms and the `d_post` hand-off are unchanged). And `~MistralSession` now
     frees — without it every `Infer` leaked the dense graph + expert cache and
     the *second* request OOMed. (`FreeMistralSession` resets the pointers in
     place; `s = MistralSession{}` recursed infinitely once the destructor
     owned teardown.)
- **CLOSED — the "neutral router NaN" was a mis-attribution, not a defect
  (2026-10-12).** This entry previously recorded
  `RunMoERouterTopK`'s seq==1 path as returning a NaN logit on finite real
  inputs. It does not. The NaN was the shared-expert FP16 overflow propagating
  into the *next* layer's `d_post`, which is the router's input — i.e. a
  downstream symptom of defect 1, measured before that defect was found. With
  the shared expert fixed, the neutral device router runs this family on real
  weights at its real geometry (E=128, H=4096, top-4) in **both** numeric
  modes; the host-side router workaround has been removed.
  The investigation did surface a **real coverage gap**: the only router
  reference test covered `seq=5`, i.e. the *prefill* kernel. The seq==1 decode
  entry (`moe_router_dots_kernel` + `moe_router_select_kernel`) had no
  reference comparison at all, which is why a wrong answer there would have
  been invisible. Now pinned by "router decode-split (seq==1) matches host
  replica" (`test_qwen_moe_kernels`), sweeping E=128/H=4096, E=128/H=2048,
  E=60 (grid not a multiple of the 128-thread block), E=16, E=8 and H=132
  (H not a multiple of 8, the scalar dot path). That test passes unmodified on
  the current neutral code, which is the evidence that the kernel was never
  the problem.
- **FOUND AND FIXED — the bit-exact MoE path multiplied every layer by alpha
  (2026-10-12).** Chasing the golden gates turned up a real defect in the
  overflow-proof SwiGLU above: it divides the activation by `alpha` once, so the
  down projection returns `out / alpha`, but the FP32 accumulator folded back
  `alpha**2`. Every layer's MoE output was therefore `alpha`=16x too large at
  Mistral's layer 32, compounding down the stack.
  **Why nothing caught it:** the Paris SelfCheck still passed in both modes (a
  memorised completion survives large numeric error), the bench still ran, and
  the new shared-expert regression test asserted `cos > 0.99` — **cosine is
  invariant under a uniform scale, so it returned exactly 1.0 for a 16x-wrong
  answer.** Mutation testing had verified that the *prescale* was present, not
  that its factor was right.
  The tell was end-to-end, not numeric: on "Repeat exactly this text: ABCDEFG"
  bit-exact mode returned `The output output output` while the auto (TC) mode
  returned `HIJKLMNOPQRSTUVWXYZ`. Host-oracle confirmation at layer 32: the GPU
  accumulator was 16.0x the reference with `corr = 0.999997`, i.e. right shape,
  wrong scale. After the one-line fix (`coeff * alpha`, not `coeff * alpha*alpha`)
  both modes agree, and the regression test now pins `|got|max / |ref|max` to
  within 5% so a scale error cannot pass again.
- **FIXED — the tensor-core routed FFN overflowed its SwiGLU activation to
  FP16 inf, costing layer 32 ~52x. Root cause, fix and instrument below.**
  The tensor-core branch of `RunMoERoutedFFN` fused SwiGLU into the NVFP4
  activation quantizer (`RunAttnSwigluPreciseQuantInternal`). That kernel does
  `const __half hv = __float2half(sig * g * u);` — it rounds the product to
  **FP16 before quantizing**, and FP16 tops out at 65504. Mistral-Small-4 layer
  32 expert 67 reaches `|silu(g)*u| = 2.5e5` from entirely ordinary FP16
  projections (max 444 and 568), so the store overflowed to `+inf`, the block
  `absmax` became `inf`, `FloatToE4m3(inf)` saturated the block scale, and every
  nibble in the tile quantized from an infinite value. The scalar path was
  immune because it keeps the product in FP32 under a power-of-two prescale.
  Measured on real weights (Paris, layer 32, routed accumulator vs a host
  oracle over the source safetensors):

  | mode | before | after |
  |---|---|---|
  | `kOff` | ratio 0.9998, corr 1.000000 | unchanged (ratio 0.9998) |
  | `kAuto` (TC) | **ratio 0.0192**, corr 0.975 | **ratio 0.9932, corr 1.000000** |

  Fix: `RunAttnSwigluPreScaledQuantInternal` keeps the product in FP32, reduces a
  global `absmax` with an atomic, prescales by the smallest power of two that
  brings it inside both the FP16 and the E4M3-block-scale span, and publishes
  `alpha` in a second slot so `moe_coeff_add_scaled_kernel` folds one reciprocal
  factor into the FP32 accumulator. Two launches instead of one, no host
  synchronization (`alpha` is device-only while `RunLsxTcGemmPreQuant` takes a
  host float). `alpha == 1` is a bit-exact no-op, so every other family is
  unchanged — pinned by a dedicated bit-identity test, and re-verified on real
  weights for Qwen3.6-35B-A3B, Gemma-4-26B-A4B, Xing4.0-29B-A4B and
  Ornith-1.5-35B-A3B, each in both numeric classes. (GLM-5.3-Flash is present
  only as an unconverted HF checkpoint, so it could not be run here.)
  **Why it survived a green suite:** Paris passes in both modes and the bench
  runs, so the suite was green over a layer whose accumulator was off by 52x.
  Nothing compared a *layer accumulator* against an oracle — only final text.
  That gap is now covered by `test_qwen_nvfp4_gemv`: one case forces a
  Mistral-shaped massive activation and asserts the recovered magnitude, one
  asserts bit-identity in the normal range, one asserts a seq==1 projection is
  invariant to a power-of-two activation prescale. Both of the first two fail
  against the old kernel.

  **Cost of the fix: reduced on 2026-10-23.** The extra prequant launches
  are hoisted (three per layer for all K slots, bit-identical), removing
  the standing cost: Mistral 41.3 -> 42.2 t/s at 2048. The scratch and the
  measurement are in
  [`open-tasks/2026-10-23-sampling-neutral-session-and-environment-oracle.md`](2026-10-23-sampling-neutral-session-and-environment-oracle.md).
  The rest of §T (the three golden gates) is still open.

  Two earlier diagnoses of this entry were wrong and are retracted here, since
  the record is the point: (1) "the E4M3 block scale saturates, attenuating the
  expert 94x" — the scale does saturate, but only *because* the FP16 store
  already went to inf; and (2) "Mistral never takes the `tc_fused` branch, so
  the default path is unaffected" — false, because `qwen::SetQwenTcFastMath` is
  an inline alias for `lsx_nvfp4::SetLsxTcFastMath`, the very global the
  engine-init perf policy sets (`gpu_perf.cpp`), so `kAuto` leaves it `true` and
  `kOff` forces it `false`. Checking whether the *model class* calls the setter
  says nothing about the value in force. `LSX_PERF_AUTO=0` (which pins the
  legacy bit-exact policy, `tc=false`) is the cheap one-flag bisect for "is this
  the fast-math path?", and it puts layer 32 back at ratio 0.9998.
