# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 — Lifecycle Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (Project `AGENTS.md` forbids subagents, so this plan is executed inline — not subagent-driven.)

**Goal:** Turn the declared-debt `nemotron-3.5-lightning-30b-a3b` manifest row into measured reality: convert the BF16 checkpoint, run every lifecycle stage on real weights, capture parity goldens from the BF16 source, re-seed the perf rows through the harness, and write the stage-9 report.

**Architecture:** No new model lineage. The family's converter (`lsxforge --nemotron`), loader, kernels, expert cache, tool codec and `NemotronModelBase` lifecycle already exist and are shared with the NVFP4 sibling; this plan converts the BF16 *distribution* through that pipeline and produces evidence. The only code added is measurement/alignment tooling (one probe tool, one framing fixture+test) plus manifest and documentation reconciliation.

**Tech Stack:** C++20 / CMake presets (`linux-release-znver4-aocc`, `linux-debug`), CUDA 13.5 on RTX 5090 (CC 12.0), Catch2/ctest, Python 3 (torch CPU + `transformers` 5.15.1) for source capture, NVFP4 codec via the neutral `lsx_inference_nvfp4` layer.

**Spec:** `docs/superpowers/specs/2026-11-13-nemotron-lightning-bf16-lifecycle-design.md`

## Global Constraints

- Manifest `models/nemotron-3.5-lightning-30b-a3b.json` is the **only** hand-written registry; never edit `kModels`/`kLifecycle`/`tests/CMakeLists.txt` registrations or any guard registry for a model target — add the manifest field instead.
- Vault: `$(lsx_model_models_dir)/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (models dir = `/mnt/fast_storage/huggingface`). Converted tree lands in `<vault>/logestix/`.
- Temp workspace: `/mnt/fast_storage/tmp/nemotron_bf16/`.
- One GPU workload at a time: run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before **every** GPU-labelled target and abort if another process holds the card.
- Release preset `linux-release-znver4-aocc` for every measured number (perf rows, gates quoted in the report); `linux-debug` for the fast/guard sweep. A Debug build SKIPs the perf target by design.
- Never convert a real checkpoint inside a ctest target; converter tests stay synthetic (`convert` label, bounded TIMEOUT).
- Never hand-edit `tests/perf/perf_baselines.json`: the perf target writes it.
- Never rewrite a golden to make a gate pass. A SKIP is not a pass; a red gate becomes a measured debt entry in `docs/SELFCHECK_MATRIX.md` + `docs/OPEN_TASKS.md`.
- No CLI speed flags; the only numeric run flag is `--bit-exact`. `FastMathMode::kOff` is used by the parity gates (they pin it) and by the selfcheck only where the manifest already declares `selfcheck_bit_exact`.
- Logging through `lsxcommon::log` only; `absl::StrFormat` is printf-style (`%s`, `%d`).
- Commit messages in English with `build:`/`feat:`/`fix:`/`refactor:`/`test:`/`docs:` prefixes; one stage per commit.
- The six attention layers are **NoPE** and MTP is **cut, never applied** — any measurement that suggests otherwise is a runtime defect, not a config to change.

## Review Focus

1. **A re-downloaded checkpoint whose tensor inventory changed.** The vault was fetched 2026-11-13; if the revision differs from the 2026-09-16 audit (extra tensor, renamed prefix, different MTP depth), the converter must **error** rather than skip it. Owning task: Task 2 (its step asserts the classifier's fatal path on an injected unknown tensor and the real inventory totals).
2. **The served frame vs the captured parity frame.** The engine hardcodes `<|im_start|>assistant\n<think>\n` (thinking on); a capture run with the tool's default `enable_thinking=False` records `<think></think>` and gates a prompt the model is never served. Owning task: Task 6 (frame fixture + red-first assertion).
3. **A tree that names the wrong distribution.** Both siblings write the same `format` tag, so only the index's `model` field separates them; loading the NVFP4 tree as the BF16 model would then verify against the wrong goldens. Owning task: Task 2 (index assertions) and Task 3 (load/geometry probe).
4. **Parity answers that echo the prompt body.** The sibling's source echoed CODE-TEST-01 / DET-TEST-01/02 bodies; recording an echo as `expected` turns the gate into a tautology. Owning task: Task 8/9 (capture step requires an answer distinct from the body, else the case is recorded under `excluded_cases` with the measurement).
5. **Bit-exactness that is only apparent.** `final_logits` must be byte-identical across ≥3 cycles; a nondeterministic reduce or an evicted-expert re-read shows as a hash change, not a failure. Owning task: Task 5 (three cycles, `memcmp` on the logits, hash recorded).

---

### Task 1: Branch, baseline sweep, and the pre-change green line

**Files:** none modified (workspace only).

**Interfaces:**
- Produces: branch `feat/nemotron-lightning-bf16-lifecycle`; a recorded baseline result for the fast sweep so a later red is attributable.

- [ ] **Step 1: Create the branch**

```bash
cd /mnt/fast_storage/projects/cuda/logestix
git status --porcelain            # expect: only the two new docs/superpowers files
git switch -c feat/nemotron-lightning-bf16-lifecycle
```

- [ ] **Step 2: Confirm the Release tree is current**

Run: `cmake --build --preset linux-release-znver4-aocc`
Expected: completes with no compile work or a small increment; `build/linux-release-znver4-aocc/apps/lsxforge/lsxforge` exists.

- [ ] **Step 3: Establish the pre-change guard line (Debug)**

Run: `ctest --preset linux-debug -L fast --output-on-failure -j 8`
Expected: every target green or explicitly skipped. Save the summary to `/mnt/fast_storage/tmp/nemotron_bf16/baseline-fast.txt`.

- [ ] **Step 4: Commit the spec and the plan**

```bash
git add docs/superpowers/specs/2026-11-13-nemotron-lightning-bf16-lifecycle-design.md \
        docs/superpowers/plans/2026-11-13-nemotron-lightning-bf16-lifecycle.md
git commit -m "docs(nemotron-bf16): approved lifecycle-completion design and plan"
```

---

### Task 2: Convert the BF16 checkpoint and verify the inventory

**Files:**
- Modify: none in-tree (converter is reused).
- Evidence: `/mnt/fast_storage/tmp/nemotron_bf16/inventory.txt`
- Output: `<vault>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/logestix/` (`model.index.json`, `pinned/`, `experts/`)

**Interfaces:**
- Consumes: `lsxforge --nemotron --input <vault> --output <vault>`
- Produces: a tree whose index reads `format = "logestix-nemotron-nvfp4"`, `model = "nemotron-3.5-lightning-30b-a3b"`, 2944 routed experts, `deferred.mtp_tensors = 270`.

- [ ] **Step 1: Snapshot the source inventory before converting**

Run:

```bash
python3 - <<'PY' > /mnt/fast_storage/tmp/nemotron_bf16/inventory.txt
import json, re, collections
v = "/mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
d = json.load(open(v + "/model.safetensors.index.json"))
wm = d["weight_map"]
print("tensors", len(wm), "bytes", d["metadata"]["total_size"],
      "params", d["metadata"]["total_parameters"])
mtp = [k for k in wm if k.startswith("mtp.")]
print("mtp", len(mtp))
blocks = json.load(open(v + "/config.json"))["layers_block_type"]
print("blocks", collections.Counter(blocks),
      "attn_idx", [i for i, b in enumerate(blocks) if b == "attention"])
per = collections.Counter(re.sub(r"\.\d+\.", ".N.", k) for k in wm
                          if not k.startswith("mtp."))
for p, c in sorted(per.items()):
    print(f"{c:5d}  {p}")
PY
```
Expected in the printed output: `tensors 6513`, `bytes 65842365568`, `params 31577937344`, `mtp 270`, `blocks Counter({'mamba': 23, 'moe': 23, 'attention': 6})`, `attn_idx [5, 12, 19, 26, 33, 42]`, 23 mixer blocks with 7 tensors each, 23 MoE layers with 260 each, 6 attention layers with 4 each. A different `tensors`/`blocks` count means the vault revision changed — stop and re-do Stage 0 before converting.

- [ ] **Step 2: Confirm the converter's unknown-tensor path is fatal (guard, no code change)**

Run: `ctest --preset linux-release-znver4-aocc -R test_nemotron_moe_convert --output-on-failure`
Expected: PASS, including the case `nemotron converter rejects an unrecognized tensor` — that is the pin for Review Focus 1.

- [ ] **Step 3: Run the real conversion**

```bash
build/linux-release-znver4-aocc/apps/lsxforge/lsxforge \
  --nemotron \
  --input /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --output /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```
Expected: the log reports the converted layout, the routed-expert count and `mtp_tensors 270`; exit 0. Record wall time and the final log lines in the evidence file.

- [ ] **Step 4: Assert the index contents**

Run:

```bash
python3 - <<'PY'
import json
t = "/mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/logestix"
d = json.load(open(t + "/model.index.json"))
assert d["format"] == "logestix-nemotron-nvfp4", d["format"]
assert d["model"] == "Nemotron-3.5-Lightning-30B-A3B", d.get("model")
print("keys:", sorted(d))
print("deferred:", d.get("deferred"))
print("experts:", d.get("total_experts"), "layers:", d.get("num_layers"))
print("pinned:", d.get("total_pinned_tensors"), "attn:", d.get("attn_layer_idx"))
PY
```
Expected: `format`/`model` assertions pass (Review Focus 3), `deferred` shows `{"mtp": "cut", "mtp_tensors": 270}`, `total_experts 2944`, `num_layers 52`, `attn_layer_idx [5, 12, 19, 26, 33, 42]`, `total_pinned_tensors` printed (355 on the NVFP4 sibling's tree — this tree's number is recorded, not assumed, and it has no `fp8_kv_scales_dropped` field). Copy the printed numbers into the evidence file.

- [ ] **Step 5: Confirm the tree sizes**

Run: `du -sh <vault>/logestix <vault>/logestix/pinned <vault>/logestix/experts`
Expected: tree on the order of 18 GB (the sibling's measured figure); record the exact numbers.

- [ ] **Step 6: Commit**

No in-tree change: the tree lives in the vault. If the conversion forced a converter fix, that fix and its failing test land here instead, with message:

```bash
git commit -m "fix(nemotron): classify <tensor> in the BF16 distribution"
```

---

### Task 3: Per-tensor quantization error vs the BF16 source (stage 2 evidence)

**Files:**
- Create: `tools/nemotron_bf16_quant_probe.py`
- Evidence: `/mnt/fast_storage/tmp/nemotron_bf16/quant-error.txt`

**Interfaces:**
- Produces: `python3 tools/nemotron_bf16_quant_probe.py --selftest` (exit 0) and `--model <vault> --tree <vault>/logestix --out <file>`, emitting one line per probed tensor: `name  relL2  ref=source dtype`.
- The probe decodes the converted record back to BF16 with the same codec as `libs/lsxcommon/src/nemotron_moe_nvfp4_dequant.cu` (E2M1 nibble, per-16 E4M3 block scale, F32 global scale) and compares against the same tensor read from the BF16 shards.

- [ ] **Step 1: Write the failing self-test**

Test (in the new file, `--selftest`): quantize a synthetic `[rows, cols]` FP32 tensor with `quant_nvfp4` (block 16, the repo's codec constants), decode it back, and assert (a) `relL2 < 0.12` for a smooth tensor, (b) decoding the packed bytes equals the codec's own reference pair on a hand-built 1×32 case with `global_scale != 1` (a `gs = 1` case cannot see a dropped scale).
Run: `python3 tools/nemotron_bf16_quant_probe.py --selftest`
Expected: FAIL — file/flag does not exist yet.

- [ ] **Step 2: Implement the probe**

Signatures: `quant_nvfp4(w: torch.Tensor, block: int = 16) -> tuple[bytes, torch.Tensor, float]`, `dequant_nvfp4(packed, block_scale, global_scale) -> torch.Tensor`, `probe(model_dir: str, tree_dir: str) -> list[dict]`. Probe one tensor per class: `backbone.layers.5.mixer.q_proj.weight` (attention), `backbone.layers.0.mixer.in_proj.weight` (Mamba), `backbone.layers.1.mixer.experts.0.down_proj.weight` (routed expert), `backbone.layers.1.mixer.shared_experts.up_proj.weight` (shared), `lm_head.weight` (head).

- [ ] **Step 3: Run the self-test**

Run: `python3 tools/nemotron_bf16_quant_probe.py --selftest`
Expected: PASS.

- [ ] **Step 4: Measure the real tree**

Run: `python3 tools/nemotron_bf16_quant_probe.py --model /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --tree /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/logestix --out /mnt/fast_storage/tmp/nemotron_bf16/quant-error.txt`
Expected: five lines. Sanity anchors from the sibling: head ≈ 0.2 %, experts 5–6 %, attention ≈ 9–10 %; a value > 0.3 or ≈ 0 on any line means a wrong decode, not a wrong model — investigate before recording.

- [ ] **Step 5: Commit**

```bash
git add tools/nemotron_bf16_quant_probe.py
git commit -m "test(nemotron-bf16): probe the converted tree against the BF16 source per tensor"
```

---

### Task 4: Loader, kernels, session, tokenizer on the real tree (stages 3, 5, 6)

**Files:**
- Modify: `libs/lsxcommon/src/nemotron_moe_constants.h` (add `kModelName`)
- Modify: `libs/lsxcommon/src/nemotron_moe_model.cpp` (`LoadAndValidateIndex` gains the distribution check)
- Modify: `libs/lsxcommon/src/nemotron_moe_convert.cpp` (index writer uses `kModelName`)
- Test: `tests/test_nemotron_moe_model.cpp` (new case)

**Interfaces:**
- Consumes: `nemotron_moe::NemotronIndex::model` (filled by `LoadNemotronIndex` from the index's `model` field).
- Produces: `lsxcommon::nemotron_moe::frozen::kModelName == "Nemotron-3.5-Lightning-30B-A3B"`; `NemotronHModel::Load` fails on a tree that declares another distribution.
- Why this exists (Review Focus 3): `NemotronNvfp4Model` already refuses a tree whose `model` field is not its own, with the comment "a mixed-up tree would silently verify against the wrong one". The BF16 leaf has no such check, so the NVFP4 tree loads as `nemotron-3.5-lightning-30b-a3b` — harmless while this model had no goldens, a real hazard now that Task 7–9 give it its own.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nemotron_moe_model.cpp`:

```cpp
TEST_CASE("nemotron Load rejects the NVFP4 sibling's tree",
          "[nemotron][fast][model]")
{
    // Both distributions emit `logestix-nemotron-nvfp4`; only the index's
    // `model` field separates them, and each carries its own parity goldens.
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_dist_src");
    const auto dst = nemotrontest::MakeTempDir("nm_dist_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    REQUIRE(nemotron_moe::ConvertNemotronModel(src.string(), dst.string()).ok());

    // Rewrite the index so it declares the sibling distribution.
    const auto idx = dst / "logestix" / "model.index.json";
    RewriteModelField(idx, "Nemotron-3.5-Lightning-30B-A3B-NVFP4");

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    cfg.model_type = NemotronHModel::kType;
    NemotronHModel m(cfg);
    REQUIRE_FALSE(m.Load());
    CHECK(m.Load() == false);   // and it must not "succeed" on a retry
}
```

`RewriteModelField(path, value)` is a file-local helper: read the JSON, replace the `"model": "…"` string, write it back.

- [ ] **Step 2: Run it to see it fail**

Run: `cmake --build --preset linux-debug --target test_nemotron_moe_model && ctest --preset linux-debug -R test_nemotron_moe_model --output-on-failure`
Expected: FAIL — `Load()` returns true because no check exists.

- [ ] **Step 3: Implement the check**

In `nemotron_moe_constants.h`, beside `kType`:

```cpp
// The `model` field this distribution writes into model.index.json. Both
// siblings share the `logestix-nemotron-nvfp4` format tag, so this string is
// the only thing that tells their trees apart.
inline constexpr const char *kModelName = "Nemotron-3.5-Lightning-30B-A3B";
```

In `NemotronHModel::LoadAndValidateIndex`, after `LoadNemotronIndex` succeeds:

```cpp
if (idx_.model != nemotron_moe::frozen::kModelName) {
    error = absl::StrCat("model.index.json declares model '", idx_.model,
                         "' but this loader serves '",
                         nemotron_moe::frozen::kModelName,
                         "' - the NVFP4 sibling's tree is a different model");
    return false;
}
```

In the converter's index writer, replace the literal `"Nemotron-3.5-Lightning-30B-A3B"` with `frozen::kModelName` so the two cannot drift.

- [ ] **Step 4: Run the tests**

Run: `ctest --preset linux-debug -R "test_nemotron_moe_model|test_nemotron_moe_convert|test_nemotron_moe_loader" --output-on-failure`
Expected: PASS, including the new rejection case and the existing synthetic-conversion cases (the fixture still writes `kModelName`).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/nemotron_moe_constants.h libs/lsxcommon/src/nemotron_moe_model.cpp libs/lsxcommon/src/nemotron_moe_convert.cpp tests/test_nemotron_moe_model.cpp
git commit -m "fix(nemotron): refuse the NVFP4 sibling's tree in the BF16 loader"
```

- [ ] **Step 6: Host-side targets (Debug tree, no GPU)**

Run: `ctest --preset linux-debug -R "test_nemotron_moe_(loader|expert_cache|model|math)" --output-on-failure`
Expected: all PASS.

- [ ] **Step 7: GPU check, then the device targets**

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv     # expect: empty
ctest --preset linux-release-znver4-aocc -R "test_nemotron_moe_kernels|test_nemotron_session" --output-on-failure
```
Expected: PASS (not SKIP — the tree exists now). Rebuild the Release tree first (`cmake --build --preset linux-release-znver4-aocc`) so it carries Step 3's change.

- [ ] **Step 8: Tokenizer parity on the real vault**

Run: `ctest --preset linux-release-znver4-aocc -R test_nemotron_moe_tokenizer_parity --output-on-failure`
Expected: PASS with 22/22 corpus lines matching; if it reports any mismatch, the tokenizer arrows are wrong and everything downstream is void — stop and fix before proceeding.

- [ ] **Step 9: Record**

Append the four verdicts to `/mnt/fast_storage/tmp/nemotron_bf16/gates.txt`.

---

### Task 5: SelfCheck on real weights (Paris ×3 bit-exact)

**Files:** none (shared driver `tests/test_selfcheck_driver.cpp`).

**Interfaces:**
- Consumes: manifest `selfcheck_max_gen = 2`, `selfcheck_bit_exact = false`, `selfcheck_frame_prompt = false` (transcribed from the sibling; do **not** normalise).
- Produces: `logits_hash`, top-1 id, and the three-cycle bit-exactness verdict for the matrix row and the report.

- [ ] **Step 1: Run the gate**

Run: `ctest --preset linux-release-znver4-aocc -R test_nemotron_moe_selfcheck --output-on-failure`
Expected: PASS — Paris answered, `selfcheck_passed == true`, three cycles with byte-identical `final_logits`.

- [ ] **Step 2: If red, apply the §G5 rule before touching the probe**

Re-run with the same binary and compare top-1/`n`/hash; only if the BF16 probe genuinely differs from the sibling's recorded probe (2 tokens, kAuto, raw BPE) change a manifest field, and say which one and why in the commit. Never "normalise" all three.

- [ ] **Step 3: Record**

Append `logits_hash`, top-1 id and the cycle count to the evidence file.

- [ ] **Step 4: Commit (only if the probe or a defect changed)**

```bash
git commit -m "fix(nemotron-bf16): <defect> surfaced by the real-weight selfcheck"
```

---

### Task 6: Tool-calling bridge — frame fixture, host test, real-weight round-trip (stage 4b)

**Files:**
- Create: `tests/fixtures/nemotron/fsi01_frame.txt` (the exact prompt string the source tokenizer was fed for FSI-TEST-01)
- Modify: `tests/test_nemotron_moe_model.cpp` (new test case)
- Modify: `tests/CMakeLists.txt:3385-3386` (add `LSX_SOURCE_DIR` to `test_nemotron_moe_model`'s compile definitions)
- Modify: `models/nemotron-3.5-lightning-30b-a3b.json` (`tool_roundtrip: true`)

**Interfaces:**
- Consumes: `static bool NemotronHModel::RenderNemotronConversation(const ChatConversation &conv, std::string &out)` — already public and static (the existing `[tool_framing]` case calls it); it emits `<|im_start|>system\n…<|im_end|>\n` … `<|im_start|>assistant\n<think>\n`.
- Produces: `tests/fixtures/nemotron/fsi01_frame.txt`, referenced by the new test case; `tool_roundtrip: true` backed by a real run.

- [ ] **Step 1: Write the failing frame-alignment test**

Add `LSX_SOURCE_DIR="${PROJECT_SOURCE_DIR}"` beside the existing `TOKENIZER_TEST_DATA_DIR` definition for `test_nemotron_moe_model` (same shape as `test_perf_baseline_logic` at line ~193), then add a new case to `tests/test_nemotron_moe_model.cpp`:

```cpp
TEST_CASE("nemotron FSI-capture frame equals the served frame",
          "[nemotron][fast][prompt_framing]")
{
    // Review Focus 2: the frame the parity capture feeds the SOURCE must be the
    // frame the ENGINE serves. The fixture is written by the capture run; a
    // renderer/template drift shows here, not in the gate.
    const std::filesystem::path frame_path =
        std::filesystem::path(LSX_SOURCE_DIR) /
        "tests/fixtures/nemotron/fsi01_frame.txt";
    REQUIRE(std::filesystem::exists(frame_path));
    ChatConversation conv;   // one user turn, no tools: what FSI capture sends
    conv.messages.push_back({"user", <FSI-TEST-01 body>, "", {}});
    std::string rendered;
    REQUIRE(NemotronHModel::RenderNemotronConversation(conv, rendered));
    REQUIRE(rendered == ReadWholeFile(frame_path));   // byte-exact, tokenizer-free
}
```

with `ReadWholeFile(std::filesystem::path) -> std::string` a two-line `std::ifstream` + `std::istreambuf_iterator` helper defined in the file's anonymous namespace. Compare the raw string (not ids) so the assertion does not depend on which tokenizer is loaded.

- [ ] **Step 2: Run it to see it fail**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_nemotron_moe_model && ctest --preset linux-debug -R test_nemotron_moe_model --output-on-failure`
Expected: FAIL — `fsi01_frame.txt` does not exist. (If it fails earlier on a missing `LSX_SOURCE_DIR`, Step 1's CMake edit was not applied.)

- [ ] **Step 2: Run it to see it fail**

Run: `cmake --build --preset linux-debug --target test_nemotron_moe_model && ctest --preset linux-debug -R test_nemotron_moe_model --output-on-failure`
Expected: FAIL — `fsi01_frame.txt` does not exist.

- [ ] **Step 3: Produce the fixture from the checkpoint's own template**

Run a one-off in `/mnt/fast_storage/tmp/nemotron_bf16/` that applies the vault's `chat_template.jinja` to `[{"role":"user","content":<FSI-TEST-01 body>}]` with `add_generation_prompt=True, enable_thinking=True`, and writes the resulting string to `tests/fixtures/nemotron/fsi01_frame.txt` (no trailing newline added).
Expected: the file opens `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n…` and ends `<|im_end|>\n<|im_start|>assistant\n<think>\n`.

- [ ] **Step 4: Re-run the test**

Run: `ctest --preset linux-debug -R test_nemotron_moe_model --output-on-failure`
Expected: PASS. If it fails, the renderer and the template disagree — fix the **renderer** (shared with the NVFP4 sibling, so re-run that sibling's target too), never the fixture.

- [ ] **Step 5: Flip the manifest and run the real-weight round-trip**

Set `"tool_roundtrip": true` in the manifest, re-configure (`cmake --preset linux-release-znver4-aocc`), rebuild, then:

Run: `ctest --preset linux-release-znver4-aocc -R test_nemotron_moe_selfcheck --output-on-failure`
Expected: PASS including the `selfcheck tool round-trip on the converted weights` case — the decoded call is `get_weather` with `{"city":"Paris"}`.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/nemotron/fsi01_frame.txt tests/test_nemotron_moe_model.cpp models/nemotron-3.5-lightning-30b-a3b.json
git commit -m "test(nemotron-bf16): pin the served frame to the source template and enable the real-weight tool round-trip"
```

---

### Task 7: FSI golden from the BF16 source, then the gate

**Files:**
- Create: `tests/fixtures/fsi/nemotron-3.5-lightning-30b-a3b.json`
- Modify: `tools/fsi_capture.py` (override entry)

**Interfaces:**
- Consumes: `tools/fsi_capture.py --model <vault> --type nemotron-3.5-lightning-30b-a3b --out tests/fixtures/fsi/nemotron-3.5-lightning-30b-a3b.json`
- Produces: a golden whose `source` names the executed BF16 checkpoint, three prompts with `input_ids`, `expected`, `raw`, and `excluded_cases` for any measured generation error.

- [ ] **Step 1: Add the capture override (thinking on — Review Focus 2)**

In `tools/fsi_capture.py`'s `OVERRIDES` add:

```python
# The engine's frame for this checkpoint opens <|im_start|>assistant\n<think>\n
# (thinking ON); the tool default (enable_thinking False) would capture
# <think></think> and gate a prompt the engine never serves.
"nemotron-3.5-lightning-30b-a3b": {"enable_thinking": True},
```

- [ ] **Step 2: Validate the prompt set without a model**

Run: `python3 tools/fsi_capture.py --selftest`
Expected: PASS (the prompt/formula set is intact).

- [ ] **Step 3: Capture from the source**

Run: `python3 tools/fsi_capture.py --model /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --type nemotron-3.5-lightning-30b-a3b --out tests/fixtures/fsi/nemotron-3.5-lightning-30b-a3b.json --max-new-tokens 384`
Expected: three captured entries; `source` records the executed BF16 checkpoint (device CPU, dtype bf16). Record wall time.

- [ ] **Step 4: Audit the captured answers before running the gate**

For each prompt assert the answer is a real formula line, not an echo of the prompt body (Review Focus 4). If a case answers with the body or with an added factor (the sibling appends `*exp(r)` to FSI-01), move it to `excluded_cases` with the exact output and the measured deviation — do not edit `expected`.

- [ ] **Step 5: Run the gate**

Run: `ctest --preset linux-release-znver4-aocc -R test_nemotron_moe_fsi_selfcheck --output-on-failure`
Expected: PASS for every case left in `prompts`. A red result on the 4-bit body is recorded (not silenced): the number goes to Task 11's matrix/OPEN_TASKS entries and the residual plane stays an open item.

- [ ] **Step 6: Commit**

```bash
git add tools/fsi_capture.py tests/fixtures/fsi/nemotron-3.5-lightning-30b-a3b.json
git commit -m "test(nemotron-bf16): capture the FSI golden from the BF16 source and run the gate"
```

---

### Task 8: CODE golden from the BF16 source, then the gate

**Files:**
- Create: `tests/fixtures/code/nemotron-3.5-lightning-30b-a3b.json`
- Modify: `tools/code_capture.py` (same override entry as Task 7)

**Interfaces:**
- Consumes: `tools/code_capture.py --model <vault> --type nemotron-3.5-lightning-30b-a3b --out <path>`
- Produces: three captured cases or an explicit `excluded_cases` record.

- [ ] **Step 1: Add the override**

Same key as Task 7, `{"enable_thinking": True}`, with the same reason comment.

- [ ] **Step 2: Self-test**

Run: `python3 tools/code_capture.py --selftest` → PASS.

- [ ] **Step 3: Capture**

Run: `python3 tools/code_capture.py --model /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --type nemotron-3.5-lightning-30b-a3b --out tests/fixtures/code/nemotron-3.5-lightning-30b-a3b.json --max-new-tokens 384`

- [ ] **Step 4: Apply the echo rule (Review Focus 4)**

Any case whose answer reproduces the prompt body verbatim is recorded under `excluded_cases` with the raw output. A golden of an echo is a tautology, not parity.

- [ ] **Step 5: Run the gate**

Run: `ctest --preset linux-release-znver4-aocc -R test_nemotron_moe_code_selfcheck --output-on-failure`
Expected: PASS on the retained cases; otherwise the measurement is recorded for Task 11.

- [ ] **Step 6: Commit**

```bash
git add tools/code_capture.py tests/fixtures/code/nemotron-3.5-lightning-30b-a3b.json
git commit -m "test(nemotron-bf16): capture the CODE golden from the BF16 source and run the gate"
```

---

### Task 9: DET golden from the BF16 source, then the gate

**Files:**
- Create: `tests/fixtures/det/nemotron-3.5-lightning-30b-a3b.json`
- Modify: `tools/det_capture.py` (same override entry)

**Interfaces:** as Task 8, for `det_capture.py` / `test_nemotron_moe_det_selfcheck`.

- [ ] **Step 1: Add the override** — `{"enable_thinking": True}` with the same reason comment.

- [ ] **Step 2: Self-test** — `python3 tools/det_capture.py --selftest` → PASS.

- [ ] **Step 3: Capture**

Run: `python3 tools/det_capture.py --model /mnt/fast_storage/lsx_model/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --type nemotron-3.5-lightning-30b-a3b --out tests/fixtures/det/nemotron-3.5-lightning-30b-a3b.json --max-new-tokens 384`

- [ ] **Step 4: Apply the echo rule** — as Task 8.

- [ ] **Step 5: Run the gate**

Run: `ctest --preset linux-release-znver4-aocc -R test_nemotron_moe_det_selfcheck --output-on-failure`
Expected: PASS (deterministic across its three runs) on the retained cases.

- [ ] **Step 6: Commit**

```bash
git add tools/det_capture.py tests/fixtures/det/nemotron-3.5-lightning-30b-a3b.json
git commit -m "test(nemotron-bf16): capture the DET golden from the BF16 source and run the gate"
```

---

### Task 10: Re-seed the perf rows through the harness

**Files:**
- Modify: `tests/perf/perf_baselines.json` (written by the target only — never by hand)

**Interfaces:**
- Consumes: `test_perf_baseline_nemotron` (Release; manifest pins `LSX_PERF_VAULT_SUBDIR=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, label `perf`, TIMEOUT 2400, `SKIP_RETURN_CODE 4`).
- Produces: fresh `perf` (kAuto) and `r3` (kOff) rows dated 2026-11-13 with `ttft_ms`, `avg_power_w`, `peak_power_w`, `power_samples`.

- [ ] **Step 1: Verify the card is idle and cool**

Run: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
Expected: no rows. Do not start the target with another GPU workload resident (the ctest `RESOURCE_LOCK` is per-process, not per-machine).

- [ ] **Step 2: Run the target**

Run: `ctest --preset linux-release-znver4-aocc -R test_perf_baseline_nemotron --output-on-failure`
Expected: PASS; both modes measured at 2048 generated tokens; the JSON now holds this run's rows.

- [ ] **Step 3: Read back the rows**

Run: `python3 -c` over `tests/perf/perf_baselines.json`, printing the two `nemotron-3.5-lightning-30b-a3b` rows with `mode`, `tokens_per_sec`, `ttft_ms`, `date`.
Expected: `date` is this run's; the old `2026-09-16` rows (including `r3 24.07`) are gone. If the environment probe reports a degraded card, re-run rather than recording a low number.

- [ ] **Step 4: Commit**

```bash
git add tests/perf/perf_baselines.json
git commit -m "perf(nemotron-bf16): re-seed the rows from the BF16 tree at the canonical protocol"
```

---

### Task 11: Manifest, docs, guards, and the stage-9 report

**Files:**
- Modify: `models/nemotron-3.5-lightning-30b-a3b.json` (`format_tag`, `aliases`, `report`, `notes`, `golden_debt` reasons replaced by measured ones if any remain)
- Create: `docs/model-reports/2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md`
- Modify: `docs/MODELS.md` (regenerated), `docs/SELFCHECK_MATRIX.md` (§4 row removed, §1/§2 row added), `docs/OPEN_TASKS.md` (new section), `docs/model-reports/README.md` (index entry)

**Interfaces:**
- Consumes: every verdict recorded in `/mnt/fast_storage/tmp/nemotron_bf16/*.txt`
- Produces: a manifest that describes the measured state, a regenerated catalogue, and a publishable report.

- [ ] **Step 1: Finalise the manifest**

Set `"format_tag": "logestix-nemotron-nvfp4"`, `"aliases": ["nvidia-nemotron-3.5-lightning-30b-a3b-bf16"]` (must equal the lower-cased vault dir), `"report": "docs/model-reports/2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md"`, and a one-line `notes` carrying the measured perf. Keep `golden_debt` only for gates that genuinely stayed red, with the measured reason as its text; delete the keys that went green.

- [ ] **Step 2: Regenerate the catalogue and validate the registry**

```bash
python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .
python3 tools/gen_model_manifest.py --strict
```
Expected: `--strict` clean (no unclaimed test file, no schema violation); `docs/MODELS.md` diff limited to the nemotron row.

- [ ] **Step 3: Run the guard sweep**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug -R "manifest_sync|model_independence_guard|model_suite_guard|code_naming_guard|ctest_labels_guard|expert_store_guard|logging_guard|absl_format_guard" --output-on-failure
```
Expected: all PASS. `test_manifest_sync` additionally proves the compiled table equals the manifests.

- [ ] **Step 4: Full fast sweep and the label sweeps**

```bash
ctest --preset linux-debug -L fast --output-on-failure -j 8
ctest --preset linux-debug-convert --output-on-failure
ctest --preset linux-release-znver4-aocc-inference --output-on-failure
```
Expected: green or explicitly skipped; any target that executed must be green. Save all three summaries to the evidence file.

- [ ] **Step 5: Update the matrix and the open-tasks record**

`docs/SELFCHECK_MATRIX.md`: delete the §4 row, add the model to §1 with its measured verdicts (SelfCheck hash, tokenizer 22/22, tool round-trip, FSI/CODE/DET outcomes, perf). `docs/OPEN_TASKS.md`: new section recording (a) the BF16-source quantization-error table from Task 3, (b) the golden provenance ("source executed locally" — unlike the NVFP4 sibling's template-derived or oracle-reconstructed goldens), (c) the residual-plane decision as an **open item with its measured trigger**, (d) any defect found.

- [ ] **Step 6: Write the stage-9 report**

`docs/model-reports/2026-11-13-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md`, two parts: a publish-ready English post (what was converted and on what hardware; what the checkpoint forced into the engine — NoPE, relu2, top-6 renormalised with 2.5 routed scale, shared expert outside the routed sum, MTP cut; artifact-level defects with observed failure modes and suggestions to the authors; measured gates/perf; what was **not** measured and why) and an `Evidence` section (spec path, this plan, OPEN_TASKS section, matrix row, gate transcripts, perf row dates). Add the entry to `docs/model-reports/README.md`.

- [ ] **Step 7: Final verification before claiming completion**

```bash
ctest --preset linux-release-znver4-aocc -R "test_nemotron" --output-on-failure
python3 tools/gen_model_manifest.py --check --docs docs/MODELS.md --root .
```
Expected: every nemotron target green (none skipped for an absent tree); catalogue current. Quote the actual output in the summary — no success claim without it.

- [ ] **Step 8: Commit**

```bash
git add models/nemotron-3.5-lightning-30b-a3b.json docs/MODELS.md docs/SELFCHECK_MATRIX.md docs/OPEN_TASKS.md docs/model-reports/
git commit -m "docs(nemotron-bf16): lifecycle verdicts, matrix row and the stage-9 report"
```

---

## Self-review notes (written after the plan, against the spec)

- **Spec coverage:** every stage in the spec's order has a task (1 baseline, 2 convert+inventory, 3 quantization evidence, 4 load/kernels/tokenizer, 5 selfcheck, 6 stage 4b, 7–9 parity gates, 10 perf, 11 docs/report). The spec's "guards green before the report" is Task 11 Step 3–4, ordered before Step 6.
- **Deliberate omissions:** the residual plane (spec non-goal — recorded, not built); MTP application (never); context beyond 4096 (out of scope).
- **Known unknowns carried as measurement steps, not guesses:** pinned-tensor count, `logits_hash`, perf numbers, and which parity cases survive the echo rule. The plan requires recording what the run says instead of pre-writing a number.
