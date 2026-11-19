# Muse-Glimmer-30B: FSI golden captured — R21 closed (2026-11-19)

Snapshot of the live item as it stood when this session took it, followed by the
resolution: the item left `docs/OPEN_TASKS.md` when the FSI capture landed.

## R21. Muse-Glimmer-30B — ingestion landed; FSI golden capture open (2026-11-19)

Branch `feat/muse-glimmer-30b-ingest`. New dense family `muse`: 52 layers,
hidden 6656, 32Q/2KV x128 gated GQA, [sliding x3, full-NoPE] x13, centered
norms, weightless embed norm, untied FP16 head, softcap 20 x 0.196116.
BF16 -> NVFP4 (8 projections/layer), embed/head/norms FP16. Report:
`docs/model-reports/2026-11-19-Muse-Glimmer-30B.md`.

**Measured lifecycle:** SelfCheck PASS x3 bit-exact (`logits_hash
3374b7bc94231955`, top-1 13796 "Paris", raw probe, kOff, max_gen 2); native
ATEM tool round-trip `get_weather {"city":"Paris"}`; cooperative cancel;
CODE 3/3 and DET 3/3 (deterministic x3) on goldens captured from the EXECUTED
source via the text-tower CPU oracle (`tools/muse_golden_oracle.py`); perf
52.86 (perf) / 53.34 (r3) t/s @2048, ttft 554 ms, resident 19.5 GB.

**The harness defect found and fixed (TDD):** Muse emits its answer in an
`assistant to=user` ATEM channel, then loops back into `assistant to=self`
reasoning; the LAST answer channel can be truncated by the token budget
mid-answer, and the reasoning tail echoes the prompt body (so `CutPromptEcho`
cut everything and the extraction was empty). `CodeGoldenCase`/`DetGoldenCase`
gained `answer_channel`/`reasoning_channel`; the harness selects the FIRST
COMPLETE answer channel. Two CODE unit cases + one DET unit case written RED
first (`tests/test_code_selfcheck.cpp`, `tests/test_det_selfcheck.cpp`).

**Open: the FSI golden.** The source IS runnable (the CPU oracle answers the
Paris probe and the CODE/DET sets), but the FSI capture was attempted twice
and did not complete: at 512 tokens FSI-03 aborts (the source loops in its
`to=self` reasoning channel and never emits a clean `Drawdown>Threshold`
line — the raw is a wall of repeated "So output Drawdown>Threshold" inside
reasoning); at 1024 tokens the 3-case capture ran >1.5 h on CPU (160 cores,
~1600% CPU) without finishing and was killed. The gate is wired
(`test_muse_dense_fsi_selfcheck`, SKIP_RETURN_CODE 4) and SKIPs for the
golden, not the tree; `golden_debt.fsi` records the attempt. Next step: a
longer-budget capture run unattended (the per-case cost is ~30-50 min at
1024 tokens), or a reasoning-strength-low framing if the checkpoint's
template supports it (the jinja's `reasoning_strength` kwarg).

## Resolution (2026-11-19)

**The two defects were in the harness and in the framing, not in the budget.**
Re-reading the failed run's raw separated them:

1. *Extraction.* The retained FSI-03 raw from the 512-token `high` attempt
   contains `assistant to=userDrawdown>Threshold` repeatedly — the model
   answered — but the run recorded `empty extraction`, because the echoed
   prompt's `Variables: Drawdown = 14.999991; Threshold = 15.000000.` gave the
   extractor a `Drawdown`-plus-`=` line first and the comparison pattern then
   found no identifier. This is the same defect CODE/DET hit.
2. *Framing.* The checkpoint's `chat_template.jinja` renders `Reasoning
   strength: <value>.` into the system block and defaults to `high`;
   `reasoning_strength: low` is the template's own switch, and at `low` the
   answer channel carries the canonical formula for all three prompts (at
   `high` the 512-token attempt extracted `Balance_Alpha=100000000`, the
   prompt's numeric echo, for FSI-02).

A longer budget could fix neither: the 1024-token attempt burned 1.5 h and would
have produced the same answers. The loop itself is **not** removed by `low` —
the captured raws still carry 12 / 7 / 37 `assistant to=user` channels (0 / 5 /
0 `assistant to=self`) in 512 tokens; the first-complete-channel rule is what
makes the repetition harmless.

**Capture recipe (now the documented one):**

```bash
python3 tools/muse_golden_oracle.py     # loader only; the capture tools import it
python3 tools/fsi_capture.py --model /mnt/fast_storage/huggingface/Muse-Glimmer-30B \
    --type muse-glimmer-30b --out tests/fixtures/fsi/muse-glimmer-30b.json \
    --max-new-tokens 512 --only FSI-TEST-01
python3 tools/fsi_capture.py ... --only FSI-TEST-02,FSI-TEST-03 --resume
```

Measured on the reference host (CPU text tower, 627 text tensors loaded, 809
vision tensors skipped): ~27 min per case at 512 tokens (~21 cores, 400-470
CPU-min/case), 512 tokens regardless of `ignore_eos`; the raw is 1.7-2.0 kB and
the FIRST COMPLETE `assistant to=user` channel is 18-71 characters and equals
the prompt's canonical answer in all three cases, so no case needed the "model
answers differently" warning path.

**Tooling changes (TDD, `test_fsi_selfcheck.cpp` RED first):**

1. The ATEM channel rule moved to one place —
   `libs/lsxcommon/include/lsxcommon/atem_channel.h`
   (`atem::SelectAnswerChannel`), used by FSI, CODE and DET; the two
   copy-pasted implementations in `{code,det}_selfcheck.cpp` were deleted.
2. `FsiGoldenCase` gained `answer_channel`/`reasoning_channel`; when present,
   `RunFsiCase` reads the first COMPLETE answer channel instead of the raw tail
   (the last channel can be budget-truncated — measured on Muse CODE/DET:
   `PARADOX_ANTINOM`). Absent → historical behaviour, every existing golden
   untouched.
3. `tools/muse_golden_oracle.py` now owns the checkpoint's capture facts —
   `TEMPLATE_KWARGS = {"reasoning_strength": "low"}`, the channel headers,
   `capture_opts()` and `select_answer_channel()` (the Python twin of the C++
   rule). Its torch/transformers imports are lazy, so `fsi_capture.py --selftest`
   exercises the pure rules without the checkpoint stack.
4. `tools/fsi_capture.py` captures **one prompt at a time and saves each
   completed case atomically**, with `--only` to run a single id and `--resume`
   to merge into what an earlier run wrote. This is the direct lesson of the
   two failed attempts: the old tool wrote the golden only after all three
   cases, so a kill lost everything. A failed case now dumps its full raw to
   `<out>.<ID>.raw.txt`, is skipped, and the run exits 3 keeping the partial
   file.

**Verdicts.** `test_muse_dense_fsi_selfcheck` on the converted NVFP4 tree
(kOff, 512-token budget, `ignore_eos=true`): **3/3 passed**, 11 s/case —
FSI-01 `PV=CF*(1-exp(-r*n))/(1-exp(-r))` (canonical form of the source's
`PV=CF*((1-exp(-r*n))/(1-exp(-r)))`), FSI-02
`Balance_Alpha=((100000000*Rate_A-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C`, FSI-03
`Drawdown>Threshold`. `golden_debt.fsi` is deleted from the manifest (the suite
guard fails a stale debt row), `docs/SELFCHECK_MATRIX.md` §1 now reads 3/3 and
the §2 row is gone.

**Two artifacts recorded, neither masked:**

- The captured `input_ids` start with **two** `<|begin_of_text|>` tokens
  (`add_special_tokens=True` on a template that already emits `bos_token`). The
  model answers correctly and the verifier replays the exact ids, so parity is
  unaffected — but the framing is not canonical. It is the same for the CODE and
  DET goldens (captured by the same path at `high` strength), so it is recorded
  rather than fixed piecemeal; a fix means re-capturing all three stages.
- `MuseRenderPolicy::reasoning_strength` exists in the engine
  (`libs/lsxcommon/src/muse_tool_format.h`) but nothing sets it, so **serving is
  always `high`** — the strength at which the model loops its answer channel.
  The FSI gate therefore verifies parity at `low`, not the served strength. A
  serve-time knob (env or `lsxbenchmark` flag) is the follow-up; the golden must
  be re-captured at that strength if the default ever changes, because the
  framing is part of `input_ids`.

Findings text: `docs/FSI_SELFCHECK_FINDINGS.md` § "Muse-Glimmer-30B".
