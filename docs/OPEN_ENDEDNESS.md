# Why this work has no completion state

Status: accepted (2026-10-12)
Companion to [`numeric-contract.md`](numeric-contract.md) (what we guarantee) and
[`OPEN_TASKS.md`](OPEN_TASKS.md) (what is currently open).

This document states the structural reason that LogestiX work does not terminate
in a "done" state, and derives the engineering practices that follow from it. It
is not a disclaimer: it is the justification for practices the repository
already follows, and for the standing obligation that keeps
`docs/OPEN_TASKS.md` non-empty.

---

## 1. The theorems, stated precisely

Gödel (1931) is usually summarized loosely. The precise statements matter,
because the refinements are what actually connect to software work.

### 1.1 First incompleteness theorem

> For every consistent, recursively axiomatizable formal system `T` that is
> sufficiently strong to interpret arithmetic, there is a sentence `G` such that
> neither `G` nor `¬G` is provable in `T`.

Gödel's original construction needed **ω-consistency** to conclude that `¬G`
is unprovable. **Rosser (1936)** removed that hypothesis, yielding the
statement above. The informal "there are true statements that can be neither
proved nor refuted" is the Church formulation and requires trusting the system's
consistency; the Rosser form needs only consistency.

"Sufficiently strong to interpret arithmetic" is not a rhetorical flourish. A
system that cannot do arithmetic cannot state the sentence, so the theorem does
not apply. Section 2 is about which parts of this repository clear that bar.

### 1.2 Second incompleteness theorem

> For every such `T`, if `T` is consistent then `T ⊬ Con(T)`: the system cannot
> prove its own consistency using its own means.

This is the one that bites hardest on verification practice, and its consequence
is stronger than "proof is hard": **no internal argument suffices**. A
self-contained certificate that "this decoder is correct" is not a difficult
certificate that we have failed to write — it is not a certificate that can
exist within the system it certifies.

### 1.3 Two neighbours that matter more, in practice

- **Tarski's undefinability theorem.** `T` cannot define its own truth predicate
  for its own sentences. Whatever plays "the truth" in Theorem 1 is necessarily
  *outside* `T`. For us this is concrete: "the model emitted the right token" is
  a statement about the intended model's semantics, and that semantics lives in
  upstream PyTorch/transformers, not in this codebase.
- **Rice's theorem.** No nontrivial property of the *behaviour* of a program is
  decidable. "Does this NVFP4 kernel compute the same function as the scalar
  reference for all inputs?" is a nontrivial behavioural property, hence
  undecidable in general. This is the operational face of the same obstruction,
  and it is the one that bites on every kernel decision.

Note what these theorems do **not** say. They do not say software is buggy, and
they do not say testing is futile. They say something narrower and more
annoying: **a guarantee of completeness cannot be produced from inside the
system.** The methodological version of that point — testing shows the presence
of bugs, never their absence — is Dijkstra's, not Gödel's. They are different
claims with different uses; conflating them would be convenient and wrong.

---

## 2. The mapping, and where it is legitimate

Gödel's theorems are about formal systems, and a CUDA codebase is not one. The
analogy is worth making only where the correspondence is structural, so let us
be explicit about which parts of this repository satisfy the hypotheses.

| Gödel's role | LogestiX's role | Where it lives |
|---|---|---|
| Formal system `T` | A decode pipeline plus the property language used to reason about it (token ids, tensor indices, scale exponents, addresses, accumulators) | `libs/lsxcommon/` |
| Recursively axiomatizable | Executable, machine-checked by the same kind of derivation checker | kernels, `ctest` |
| Strong enough to interpret arithmetic | The NVFP4 dequant path itself: `ldexpf` prescale, E2M1×E4M3 scaling, `1/gs`, FP16 range checks, address arithmetic in the expert LRU | `lsx_kernels_act.cuh`, `lsx_inference_nvfp4.cu` |
| Proof | A passing assertion, a bit-exact match, a golden comparison | `tests/` |
| Truth in the intended model | Upstream `modeling_*.py` semantics, or a host reimplementation over the source safetensors | `tools/*_capture.py`, external references |
| `Con(T)` | "This pipeline computes the model" | — |

The decisive row is the third one. A codebase with no arithmetic would evade
Theorem 1. Ours does arithmetic everywhere, and specifically does *scale
arithmetic whose intermediate magnitudes are unbounded a priori* — `gs` is
~10⁴–10⁵, the block scale saturates E4M3 at 448, the product of the two is
carried in FP16 whose ceiling is 65504. These are exactly the statements for
which "this value does not overflow" is a property we must prove about all
inputs, and cannot.

The mapping fails, and the theorem does not apply, where the system is finite
and the property is decidable. A schema validator, a guard test that scans for
`printf`, an offset arithmetic check on a fixed layout — these are finite
decidable properties, they can be completed, and treating them as "open-ended"
would be an excuse for sloppiness. Section 6 draws that line explicitly.

---

## 3. What the theorems predict, and what we observed

Theorem 2 says internal self-certification is impossible. The prediction is
that **internal evidence misses whole classes of defect**, and that the only
effective evidence comes from a strictly stronger system.

On 2026-10-12 the Mistral-Small-4-119B ingestion provided a clean instance.
Five defects were found; the relevant finding is not how many, but *where the
evidence came from*:

| Defect | Internal evidence | External evidence that caught it |
|---|---|---|
| Shared expert routed through the BF16 GEMV (`RunAttnMLP` reinterpreting E2M1 nibbles as BF16) | `fast` suite **green (93/93)**. The Mistral kernel test asserted finiteness and cross-run determinism, never agreement with a reference value — so a wrong-but-finite result passed. | Host NVFP4 reference over the source safetensors: rms 0.0080 vs 0.34953 — a 44× under-contribution |
| SwiGLU product overflows FP16 (layer 32, expert 67: `\|silu(g)*u\| = 2.5e5` vs ceiling 65504) | Every layer inspected had finite values; the synthetic fixture's activations are O(1) and never reach the ceiling | Host reference reproducing the same product, finding exactly 4 values past 65504 |
| Residual needs FP32 (reaches ~2.4e5 at layer 32) | Same | Same host reference |
| Converter wrote a `model` name the factory never registered | All synthetic tests green | `detect_model_type` run against the real index |
| 2 GiB VRAM reserve instead of 8 GiB | Model loaded and answered SelfCheck | The 2048-token bench OOMed on the KV arena |

Three points follow, and they are the content of this document.

1. **A fully green internal suite was not evidence.** The shared expert was
   wrong by a factor of 44 while the whole `fast` suite passed. The reason is
   worth stating precisely, because it is a coverage fact and not a mystical
   one: the existing kernel test could only ever detect *non-finite* outputs,
   and 0.0080 is perfectly finite. The synthetic fixtures could not have caught
   it either — they exercise finiteness and determinism, not agreement with a
   reference. That gap is now closed by the host-reference test, which is the
   only instrument in the repository capable of this class of detection.
2. **The internal gate that did fire produced a symptom, not a diagnosis.** The
   real-weight SelfCheck failed — with `NaN` logits and empty output. That is a
   true signal, but it does not localise a cause: the investigation went through
   several hypotheses — a global-scale convention error, which turned out to be
   **real**; a "`NaN` from the neutral `RunMoERouterTopK` seq==1 path", which
   turned out to be **a mis-attribution**: the NaN was defect 1 propagating
   into the router's *input* one layer later, and the neutral router has
   always been correct on this geometry; and a "layer-4 expert outlier", also
   a **measurement artefact** of our own instrumentation — before the
   external reference isolated the actual cause. Two of the three hypotheses
   were wrong, and one of them had been written down as a defect in the neutral
   layer and carried across sessions. Two of the diagnostic probes
   used during that search were themselves wrong — one read a buffer before the
   kernel that fills it, and one raced the non-blocking compute stream — and
   both had to be found and corrected before they reported anything. Every real
   diagnosis of the five defects terminated at an external oracle. Tarski in
   practice: truth has to come from outside.
3. **A false lead survived longer than it should have.** The "layer-4 expert
   outlier" was recorded as a blocking defect and carried across sessions before
   the host reference showed it was simply a genuine massive-activation expert
   (GPU 403.8 against reference 411.9). So even the *bookkeeping* of open work is
   subject to this: the record of what is wrong is itself only as good as the
   strongest evidence available when it was written.

---

## 4. The practices this licenses

Each of these is already in the repository. The point is that they are not
optional diligence — they are the only kind that works.

**Build an external oracle for every numeric claim.** A host reimplementation
over the source checkpoint is the strongest available instrument, because it
is a genuinely different implementation of the same specification. The FSI /
CODE / DET golden gates
([`FSI_SELFCHECK_FINDINGS.md`](FSI_SELFCHECK_FINDINGS.md) and siblings) capture
answers from the *source safetensors* rather than from our own decode, which is
precisely the "truth outside the system" requirement. Families that declare this
as outstanding debt are declaring an unmit requirement, not a missing nicety.

**Treat a green suite as a sample, never as a proof.** Wording matters more than
it looks: the repository's claims are of the form "verified on N cases", and
should stay that way. The 93/93 fast suite is evidence of the *absence of
regressions in the cases it covers*, which is a strictly weaker statement, and
the table above is the receipt for the weaker claim.

**Mutation-check regression tests.** A test that cannot fail is an axiom that
cannot be refuted. Both new Mistral regression tests were verified by reverting
the fix and confirming the failure (the global-scale mutation; the prescale
mutation, which turns the outputs into `inf` and trips the finiteness
assertion). A test not shown to fail is an untested assumption about itself.

**Use differential numeric modes as an oracle pair.** The auto / `--bit-exact`
split is an accidental but valuable instrument: when they disagree, one of them
is wrong, and finding out which requires the external reference. Keeping both
classes green and the disagreement small is stronger than any single-class
green.

**Keep `docs/OPEN_TASKS.md` permanently non-empty.** An empty open-tasks file
would be the claim `Con(T)`. The archive convention
([`docs/open-tasks/README.md`](open-tasks/README.md)) preserves resolved items
with their full investigation trail — including the hypotheses that were wrong —
precisely because the negative results are the part Theorem 1 makes expensive.

**Re-derive, do not re-run.** The `fast` suite is a regression net over what we
already knew. New knowledge comes from a different instrument. When a defect
survives a green suite, the first question is which instrument is missing.

---

## 5. What it means for a claim of "verified"

The README's phrase "100% bit-exact match" is a claim about a measured
comparison, and the comparison is real and worth having. What Gödel's theorems
bound is the *reach* of that claim, not its truth: bit-exactness was established
for the prompt, the run, the build and the hardware set that were exercised. It
is not a proof that no input diverges, and Theorem 2 is the reason no such proof
is available from inside — which is also why the numeric contract
([`numeric-contract.md`](numeric-contract.md)) is defined as a set of
*reproducibility levels with explicitly enumerated "not guaranteed" cells*
rather than as a single correctness guarantee. That document's shape is the
correct response to Theorem 2.

---

## 6. What this does not license

Incompleteness is not an excuse, and the doc is worthless if it is read as one.

- **Decidable, finite properties must still be closed.** Schema validation,
  naming and guard registries, label discipline, memory-layout assertions,
  buffer-size checks: these are finite and decidable, and shipping one
  half-done is ordinary sloppiness wearing a philosophical hat.
- **Suites must not be allowed to rot.** A green suite is the only regression net
  we have; letting it decay is not "accepting Gödel", it is destroying the one
  instrument that does work.
- **Undecidability says nothing about whether a test is well-chosen.** Choosing
  inputs that discriminate between a kernel and its reference is a design
  problem, fully solvable, and we solve it badly when we choose zeros.
- **The theorems are not a licence for nondeterminism or tolerance creep.** The
  R3/R2/R1 contract is a deliberate narrowing of what we promise; Gödel does not
  push it further in either direction.

---

## 7. Summary

The work cannot be certified from inside. Every real guarantee we hold rests on
a comparison against something stronger — an upstream implementation, a host
reimplementation, a second numeric mode, a sanitizer, a mutational check — and
every one of those is a finite sample of an unbounded question. That is not a
failure of discipline or of ambition; it is the shape of the problem. The
engineering consequence is the practice the repository follows: build oracles,
state guarantees as levels with their "not guaranteed" cells written down, and
keep the open list honest, because an empty one would be a claim we cannot make.