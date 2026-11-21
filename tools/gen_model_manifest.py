#!/usr/bin/env python3
"""Generate every derived representation of the logestix model registry.

`models/<model_type>.json` is the single source of truth for one ingested model.
Everything that used to be hand-maintained in five places is emitted from it:

    --header   -> model_manifest.h        (C++ table for guards and gate drivers)
    --cmake    -> model_targets.cmake     (ctest registration)
    --registry -> model_registry.inc      (LsxModelFactory registration)
    --docs     -> docs/MODELS.md          (the model catalogue)
    --check    -> regenerate the committed docs and fail on any difference

Emitted C++/CMake goes to the build tree and is never committed; only the
markdown is, and `--check` makes hand-editing it impossible.

Proof modes:
    --self-test <dir>   every schema violation is reported (used by ctest)
    --print-selfcheck-targets  one Paris SelfCheck target per line (serial sweep)
    --diff-cmake <base.json>  compare the emitted target set with a captured
                              ctest oracle, so a 35-row transcription cannot
                              drift silently

See docs/superpowers/specs/2026-11-06-model-manifest-design.md.
"""

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ("convert", "load", "model", "kernels", "selfcheck",
          "fsi", "code", "det", "perf")

# stage -> (label, timeout seconds or 0 for none, sets SKIP_RETURN_CODE 4)
#
# These are the MAJORITY values measured across the 510 pre-migration targets
# (tests/fixtures/ctest/2026-11-06-baseline.json), not ideals: most model tests
# carry no timeout and no gpu lock, and gate targets carry no timeout at all.
# A target that differs states it in the manifest's `target_overrides`, and
# --diff-cmake proves the transcription is exact.
STAGE_DEFAULTS = {
    "convert": ("convert", 120, False),
    "load": ("inference", 0, True),
    "model": ("inference", 0, False),
    "kernels": ("inference", 0, True),
    "selfcheck": ("inference", 0, True),
    "fsi": ("inference", 0, True),
    "code": ("inference", 0, True),
    "det": ("inference", 0, True),
    "perf": ("perf", 3000, True),
}

# stage -> default resource lock ("" = none)
# EVERY model stage allocates on the GPU: convert quantizes there, load uploads,
# and model/kernels/selfcheck/fsi/code/det/perf all decode. The table used to be
# empty except where the hand-written CMake happened to set a lock, and the
# result was 263 of 279 GPU-labelled targets unprotected — including 36 of 44
# `perf` targets, against AGENTS.md's claim that "every perf target" takes it.
# A per-stage opinion is now unnecessary: the whole row is one GPU workload.
#
# Scope of what this buys: CTest enforces a resource lock among the tests of ONE
# ctest invocation. Two concurrent `ctest` processes still share the card (see
# docs/OPEN_TASKS.md §G1), so the operator rule remains "one GPU run at a time".
STAGE_LOCKS = {stage: "gpu" for stage in
               ("convert", "load", "model", "kernels", "selfcheck",
                "fsi", "code", "det", "perf")}

# Stages whose tests build a synthetic checkpoint/tree and therefore need the
# geometry override. deviates are expressed in the manifest's target_overrides.
SYNTHETIC_ENV = "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1"
SYNTHETIC_STAGES = ("convert", "load", "model")

PERF_CASE_FMT = "perf baseline: {}"
PERF_BINARY = "test_perf_baseline"

# Every key a manifest may carry. An unknown key is an ERROR, not something to
# skip: `"selfcheck_frame_promt": true` reads as a declaration and behaves as
# `false`, which is precisely the silent-rebaseline class that docs/OPEN_TASKS.md
# G5 records. This tuple was declared but never consulted until 2026-11-07.
MODEL_FIELDS = ("model_type", "display_name", "family", "variant", "model_class",
                "model_header", "passport", "namespace", "format_tag",
                "forge_flag", "tokenizer_kind", "vault", "lifecycle_tu",
                "stages", "golden_debt", "target_overrides", "perf", "aliases",
                "resource_lock", "report", "notes",
                "tool_roundtrip", "auto_path_determinism", "cooperative_cancel",
                "selfcheck_frame_prompt", "selfcheck_max_gen",
                "selfcheck_bit_exact")

# The subset that `--print-model-rows` and the compiled ModelRow carry, named by
# MANIFEST key. It is the emitter's field order, not a copy of it: `row_values`
# below is indexed by this tuple, so a field added here appears in the output and a
# field removed here disappears from it. `namespace` is carried in Python as `ns`
# (the manifest key is the user-facing name, the attribute predates the manifest).
# test_manifest_sync is what catches a field added to the C++ RowKey and not here,
# or the reverse.
ROW_FIELDS = ("model_type", "family", "variant", "model_class", "model_header",
              "passport", "namespace", "format_tag", "vault_dir", "vault_env",
              "tokenizer_kind", "tool_roundtrip", "auto_path_determinism",
              "cooperative_cancel", "selfcheck_frame_prompt",
              "selfcheck_max_gen", "selfcheck_bit_exact")


class ManifestError(Exception):
    def __init__(self, path, where, message):
        super().__init__(f"{path}{(':' + where) if where else ''}: {message}")
        self.path, self.where, self.message = path, where, message


@dataclass
class Target:
    name: str
    stage: str
    model_type: str
    source: str
    label: str
    timeout: int
    skip: bool
    env: list = field(default_factory=list)
    resource_lock: str = "gpu"
    defines: list = field(default_factory=list)
    cuda: bool = False
    form: str = "registration"   # "registration" or "helper" (lsx_add_test)
    perf_case: str = ""
    sources: tuple = ()    # extra TUs beyond the conventional one
    command: str = ""      # executable; defaults to the target name
    filter: str = ""       # Catch2 name/tag filter for a second case


@dataclass
class Manifest:
    model_type: str
    display_name: str
    family: str
    variant: str
    model_class: str
    model_header: str
    passport: str
    ns: str
    format_tag: str
    forge_flag: str
    tokenizer_kind: int
    vault_dir: str
    vault_env: str
    lifecycle_tu: list
    stages: dict
    golden_debt: dict
    target_overrides: dict
    perf: dict
    tool_roundtrip: bool
    auto_path_determinism: bool
    cooperative_cancel: bool
    selfcheck_frame_prompt: bool
    selfcheck_max_gen: int
    selfcheck_bit_exact: bool
    resource_lock: str
    aliases: list
    report: str
    notes: str
    path: str
    root: "Path" = None

    def targets(self):
        out = []
        for stage in STAGES:
            for name in self.stages.get(stage, []):
                out.append(self._target(stage, name))
        return out

    def _target(self, stage, name):
        label, timeout, skip = STAGE_DEFAULTS[stage]
        env = [SYNTHETIC_ENV] if stage in SYNTHETIC_STAGES else []
        lock = self.resource_lock or STAGE_LOCKS[stage]
        # `fast` means host-only (no model load, no device allocation) — those
        # must stay parallel, or a 140-target sweep serialises on a lock they
        # never needed. The label is resolved AFTER the override below, so the
        # check happens there.
        # Root-relative, never cwd-relative: this runs from whatever directory
        # the build (or a ctest) happens to use.
        source = (f"{name}.cu" if (self.root / "tests" / f"{name}.cu").exists()
                  else f"{name}.cpp")
        cuda = source.endswith(".cu")
        defines = [f'LSX_SOURCE_DIR="{TOKEN_SOURCE_DIR}"']
        perf_case = ""
        if stage == "perf":
            perf = self.perf or {}
            timeout = int(perf.get("timeout", timeout))
            lock = perf.get("resource_lock", lock)
            sub = perf.get("vault_subdir", self.vault_dir)
            env = [f"LSX_PERF_VAULT_SUBDIR={sub}"]
            perf_case = PERF_CASE_FMT.format(self.model_type)
        ov = self.target_overrides.get(name, {})
        label = ov.get("label", label)
        if label == "fast" and "resource_lock" not in ov:
            lock = ""          # host-only: see the note above
        timeout = int(ov.get("timeout", timeout))
        if "skip" in ov:
            skip = bool(ov["skip"])
        if "env" in ov:
            env = list(ov["env"])
        if "resource_lock" in ov:
            lock = ov["resource_lock"]
        if "source" in ov:
            source = ov["source"]
            cuda = source.endswith(".cu")
        defines = list(ov.get("defines", defines))
        sources = tuple(ov.get("sources", ()))
        form = ov.get("form", "registration")
        command = ov.get("command", name)
        filter_ = ov.get("filter", "")
        return Target(name=name, stage=stage, model_type=self.model_type,
                      source=source, label=label, timeout=timeout, skip=skip,
                      env=env, resource_lock=lock, defines=defines, cuda=cuda,
                      perf_case=perf_case, command=command, filter=filter_,
                      sources=sources, form=form)


# `LSX_SOURCE_DIR` is emitted literally; the consuming CMake passes the value.
TOKEN_SOURCE_DIR = "${PROJECT_SOURCE_DIR}"


# ── loading and validation ────────────────────────────────────────────────

def _req(doc, key, path, kind=None):
    if key not in doc:
        raise ManifestError(path, key, "required field missing")
    value = doc[key]
    if kind is not None and not isinstance(value, kind):
        raise ManifestError(path, key, f"expected {kind.__name__}")
    return value


def row_values(m) -> dict:
    """Manifest -> {manifest key: emitted value}, for ROW_FIELDS ordering."""
    return {
        "model_type": m.model_type,
        "family": m.family,
        "variant": m.variant,
        "model_class": m.model_class,
        "model_header": m.model_header,
        "passport": m.passport,
        "namespace": m.ns,
        "format_tag": m.format_tag,
        "vault_dir": m.vault_dir,
        "vault_env": m.vault_env,
        "tokenizer_kind": m.tokenizer_kind,
        "tool_roundtrip": int(m.tool_roundtrip),
        "auto_path_determinism": int(m.auto_path_determinism),
        "cooperative_cancel": int(m.cooperative_cancel),
        "selfcheck_frame_prompt": int(m.selfcheck_frame_prompt),
        "selfcheck_max_gen": m.selfcheck_max_gen,
        "selfcheck_bit_exact": int(m.selfcheck_bit_exact),
    }


def load_one(path: Path) -> Manifest:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(path.name, f"line {exc.lineno}", f"invalid JSON: {exc.msg}")

    unknown_keys = sorted(set(doc) - set(MODEL_FIELDS))
    if unknown_keys:
        raise ManifestError(
            path.name, unknown_keys[0],
            f"unknown field(s) {unknown_keys}; a misspelled declaration is "
            f"silently ignored, so this is an error. known: {list(MODEL_FIELDS)}")

    stages = _req(doc, "stages", path.name, dict)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ManifestError(path.name, "stages",
                            f"unknown stage(s) {sorted(unknown)}; "
                            f"known: {list(STAGES)}")
    for stage, names in stages.items():
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ManifestError(path.name, f"stages.{stage}",
                                "expected a list of target names")

    vault = _req(doc, "vault", path.name, dict)
    model_type = _req(doc, "model_type", path.name, str)

    expected_env = "LSX_TEST_" + re.sub(r"[.\-]", "_", model_type).upper() + "_ROOT"
    if vault.get("env") != expected_env:
        raise ManifestError(
            path.name, "vault.env",
            f"vault.env must be {expected_env} (derived from model_type), "
            f"got {vault.get('env')!r}")

    return Manifest(
        model_type=model_type,
        display_name=doc.get("display_name", model_type),
        family=_req(doc, "family", path.name, str),
        variant=doc.get("variant"),
        model_class=_req(doc, "model_class", path.name, str),
        model_header=_req(doc, "model_header", path.name, str),
        passport=_req(doc, "passport", path.name, str),
        ns=_req(doc, "namespace", path.name, str),
        format_tag=doc.get("format_tag", ""),
        forge_flag=doc.get("forge_flag", ""),
        tokenizer_kind=int(doc.get("tokenizer_kind", 0)),
        vault_dir=_req(vault, "dir", path.name + ":vault", str),
        vault_env=vault["env"],
        lifecycle_tu=list(doc.get("lifecycle_tu", [])),
        stages=stages,
        golden_debt=dict(doc.get("golden_debt", {})),
        target_overrides=dict(doc.get("target_overrides", {})),
        perf=dict(doc.get("perf", {})),
        tool_roundtrip=bool(doc.get("tool_roundtrip", False)),
        resource_lock=doc.get("resource_lock", ""),
        aliases=list(doc.get("aliases", [])),
        auto_path_determinism=bool(doc.get("auto_path_determinism", False)),
        cooperative_cancel=bool(doc.get("cooperative_cancel", False)),
        selfcheck_frame_prompt=bool(doc.get("selfcheck_frame_prompt", False)),
        selfcheck_max_gen=int(doc.get("selfcheck_max_gen", 2)),
        selfcheck_bit_exact=bool(doc.get("selfcheck_bit_exact", True)),
        report=doc.get("report", ""),
        notes=doc.get("notes", ""),
        path=path.name,
        root=path.parent.parent,
    )


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def validate(manifests, root: Path, partial: bool = False,
             strict_orphans: bool = False):
    """Filesystem and cross-model checks.

    Returns (errors, warnings). Errors block every emitter; warnings do not.
    The orphan scan is a warning while manifests are authored in clusters and
    an error under `--check`, which is what ships and what the sync test runs.
    """
    errors = []
    warnings = []
    src = root / "libs" / "lsxcommon" / "src"

    def add(m, where, msg):
        errors.append(f"{m.path}:{where}: {msg}")

    for m in manifests:
        # rule 2: every stage is covered or its absence is declared
        for stage in STAGES:
            names = m.stages.get(stage, [])
            if not names and stage not in m.golden_debt and stage != "perf":
                add(m, f"stages.{stage}",
                    f"gap without debt: stage '{stage}' is empty and no "
                    f"golden_debt.{stage} reason is declared")

        # rule 4: the passport exists and declares the owning namespace
        passport = root / m.passport
        if not passport.is_file():
            add(m, "passport", f"passport not found: {m.passport}")
        else:
            text = strip_comments(passport.read_text(errors="replace"))
            last = m.ns.split("::")[-1]
            # Namespaced passports declare `namespace <last>`; the dense qwen
            # passports are CLASS-scoped and declare the model class instead
            # (test_model_independence_guard documents both forms).
            declared = (re.search(r"\bnamespace\s+" + re.escape(last) + r"\b", text)
                        or re.search(r"\b(class|struct)\s+" + re.escape(last) + r"\b",
                                     text))
            if not declared:
                add(m, "namespace",
                    f"passport {m.passport} declares neither 'namespace {last}' "
                    f"nor 'class {last}'")

        # rule 5: the model class exists and its kType resolves to this
        # model_type. Two legitimate spellings: a literal, or an assignment from
        # the passport's own frozen constant (4 of 35 models use the latter).
        # model_header is spelled relative to libs/lsxcommon/src (kPassports
        # convention), so "../include/lsxcommon/x.h" resolves inside the library.
        header = (src / m.model_header).resolve()
        if not header.is_file():
            add(m, "model_header", f"model_class not found in src: {m.model_header}")
        else:
            text = strip_comments(header.read_text(errors="replace"))
            if not re.search(r"\b(class|struct)\s+" + re.escape(m.model_class) + r"\b",
                             text):
                add(m, "model_class",
                    f"model_class not found in src: {m.model_class} "
                    f"(not declared in {m.model_header})")
        ktype, ktype_from = None, None
        for cand in sorted(src.glob("*.cpp")) + sorted(src.glob("*.h")):
            body = cand.read_text(errors="replace")
            hit = re.search(
                re.escape(m.model_class) + r"\s*::\s*kType\s*=\s*\"([^\"]*)\"",
                body)
            if hit:
                ktype, ktype_from = hit.group(1), cand.name
                break
            hit = re.search(
                re.escape(m.model_class) + r"\s*::\s*kType\s*=\s*[\w:]*::(\w+)\s*;",
                body)
            if hit:
                # resolve the constant it aliases, in the passport first
                const = hit.group(1)
                for scope in (passport, *sorted(src.glob("*.h"))):
                    if not scope.is_file():
                        continue
                    res = re.search(
                        r"\b" + re.escape(const) + r"\s*(?:\[[^\]]*\])?\s*=\s*\"([^\"]*)\"",
                        scope.read_text(errors="replace"))
                    if res:
                        ktype, ktype_from = res.group(1), f"{cand.name}->{scope.name}"
                        break
                if ktype:
                    break
        if ktype is None:
            add(m, "model_class", f"model_class not found in src: {m.model_class} "
                                  f"has no resolvable kType (literal or passport alias)")
        elif ktype != m.model_type:
            add(m, "model_class",
                f"model_class {m.model_class} declares kType {ktype!r} "
                f"(via {ktype_from}), manifest says {m.model_type!r}")

        # rule 14: the declared selfcheck probe must be a probe.
        # These three fields decide the recorded logits_hash (final_logits is the
        # LAST decode step, the math mode decides its bits, framing decides the
        # ids), so a typo in one is a silent re-baseline of the one number the
        # matrix calls bit-exact evidence. `selfcheck_max_gen` is range-checked
        # because 0 would generate nothing and 100000 would need an hour.
        if not 2 <= m.selfcheck_max_gen <= 4096:
            add(m, "selfcheck_max_gen",
                f"selfcheck_max_gen={m.selfcheck_max_gen} outside 2..4096 "
                f"(a probe needs at least one token after the first)")

        # rule 13: a declared report must exist. docs/MODELS.md links it, so a
        # typo would be a dead "see the report" for every reader.
        if m.report and not (root / m.report).is_file():
            add(m, "report", f"report not found: {m.report}")

        # rule 6: every stage target resolves to a source file
        for t in m.targets():
            if t.stage == "perf":
                if not (root / "tests" / f"{PERF_BINARY}.cpp").is_file():
                    add(m, "stages.perf", f"no source for target {t.name} "
                                          f"({PERF_BINARY}.cpp missing)")
                continue
            if t.form == "selfcheck":
                if not (root / "tests" / "test_selfcheck_driver.cpp").is_file():
                    add(m, f"stages.{t.stage}",
                        f"no source for target {t.name}: "
                        f"test_selfcheck_driver.cpp missing")
                continue
            if t.form == "parity":
                # One shared driver TU serves every model's FSI/CODE/DET gate.
                if not (root / "tests" / "test_parity_gate.cpp").is_file():
                    add(m, f"stages.{t.stage}",
                        f"no source for target {t.name}: test_parity_gate.cpp "
                        f"missing")
                continue
            if t.command != t.name or t.filter:
                # A second Catch2 case over an existing binary (perf rows,
                # "[tool]*" cases): the source is the command's, not its own.
                if not (root / "tests" / f"{t.command}.cpp").is_file() and \
                   not (root / "tests" / f"{t.command}.cu").is_file():
                    add(m, f"stages.{t.stage}",
                        f"no source for target {t.name}: shared binary "
                        f"{t.command} has no test source")
                continue
            if not (root / "tests" / t.source).is_file():
                add(m, f"stages.{t.stage}",
                    f"no source for target {t.name}: tests/{t.source} does not exist")

    if not partial:
        seen_type, seen_ns, fam_variant, seen_vault = {}, {}, {}, {}
        for m in manifests:
            if m.model_type in seen_type:
                errors.append(f"{m.path}: model_type {m.model_type!r} duplicated "
                              f"by {seen_type[m.model_type]}")
            seen_type[m.model_type] = m.path
            if m.ns in seen_ns:
                errors.append(f"{m.path}: namespace {m.ns!r} already owned by "
                              f"{seen_ns[m.ns]} (rule 9: unique per model)")
            seen_ns[m.ns] = m.path
            key = (m.family, m.variant)
            if m.variant and key in fam_variant:
                errors.append(f"{m.path}: family {m.family!r} variant "
                              f"{m.variant!r} duplicated by {fam_variant[key]}")
            fam_variant[key] = m.path
            # rule 12: an alias is a case-folded spelling of the vault directory
            # (that is what path auto-detection produces), never a second name.
            for alias in m.aliases:
                if alias != m.vault_dir.lower():
                    errors.append(
                        f"{m.path}: alias {alias!r} is not the lower-cased "
                        f"vault directory {m.vault_dir.lower()!r}")
            # rule 10: one vault directory per model. Two models pointing at the
            # same tree means one of them silently loads the other's weights.
            if m.vault_dir in seen_vault:
                errors.append(
                    f"{m.path}: vault.dir {m.vault_dir!r} already used by "
                    f"{seen_vault[m.vault_dir]} (one tree cannot be two models)")
            seen_vault[m.vault_dir] = m.path
        (errors if strict_orphans else warnings).extend(
            check_orphans(manifests, root))
        (errors if strict_orphans else warnings).extend(
            check_matrix(manifests, root))
        errors.extend(check_doc_budget(manifests))
    return errors, warnings


def claimed_targets(manifests):
    """Target names AND the TUs they build: a second source of an existing
    target (test_kimi_pro_constants.cpp in test_kimi_k3_constants) is claimed."""
    names = set()
    for m in manifests:
        for stage in STAGES:
            for target in m.stages.get(stage, []):
                names.add(target)
                for src in m.target_overrides.get(target, {}).get("sources", ()):
                    names.add(Path(src).stem)
    return names


DOC_SECTION_BUDGET = 15      # lines
DOC_SECTION_CHARS = 900      # and characters: 15 long lines are not a catalogue


def check_doc_budget(manifests):
    """One model's section must fit in DOC_SECTION_BUDGET lines.

    docs/MODELS.md is a catalogue, not a second AGENTS.md: the whole point of
    moving the narrative out of AGENTS.md was that a per-model essay grows without
    bound. The budget is enforced on the EMITTED text, so a long `notes` or a long
    debt reason fails generation instead of quietly inflating the file.
    """
    problems = []
    for m in manifests:
        section = emit_model_section(m, {}).rstrip("\n")
        lines = len(section.splitlines())
        if lines > DOC_SECTION_BUDGET:
            problems.append(
                f"{m.path}: docs/MODELS.md section is {lines} lines "
                f"(budget {DOC_SECTION_BUDGET}) - shorten `notes`/`golden_debt`; "
                f"the detail belongs in the model report")
        elif len(section) > DOC_SECTION_CHARS:
            # A single 600-char note is one line and an essay at once.
            problems.append(
                f"{m.path}: docs/MODELS.md section is {len(section)} characters "
                f"(budget {DOC_SECTION_CHARS}) - shorten `notes`/`golden_debt`; "
                f"the detail belongs in the model report")
    return problems


def check_orphans(manifests, root: Path):
    """Every model test file is claimed by a manifest (Review Focus #1)."""
    claimed = claimed_targets(manifests)
    allowed = set(non_model_tests(root))
    errors = []
    for path in sorted((root / "tests").glob("test_*.cpp")) + \
            sorted((root / "tests").glob("test_*.cu")):
        stem = path.stem
        if stem in claimed or stem in allowed:
            continue
        errors.append(f"tests/{path.name}: unclaimed model test file - no manifest "
                      f"lists {stem!r} in a stage (add it, or add it to "
                      f"NON_MODEL_TESTS)")
    return errors


def non_model_tests(root: Path):
    """Test files that belong to no model: infrastructure and guards.

    Frozen on purpose: a new infrastructure test is added here in the same
    commit, which keeps the orphan check (Review Focus #1) honest.
    """
    claimed = set()
    for path in sorted((root / "models").glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        overrides = doc.get("target_overrides", {})
        for names in doc.get("stages", {}).values():
            for target in names:
                claimed.add(target)
                for src in overrides.get(target, {}).get("sources", ()):
                    claimed.add(Path(src).stem)
    explicit = _NON_MODEL_TESTS
    return sorted(claimed | explicit)


_NON_MODEL_TESTS = frozenset({
    # Shared model-test drivers: one TU serving every model of a stage kind.
    # They are named by the manifests' `form`, not by a stage list.
    "test_parity_gate",
    "test_selfcheck_driver",
    # Inkling-2-Mini ingestion in flight: claimed here until the manifest row
    # lands with the model class; DELETE these entries in that same commit.
    "test_inkling_moe_loop_constants",
    "test_inkling_moe_loop_convert",
    "test_inkling_moe_loop_loader",
    "test_inkling_moe_loop_kernels",
    "test_absl_container_guard", "test_absl_format_guard", "test_anthropic_endpoint", "test_arrow_tree_integrity",
    "test_arrow_usage_guard", "test_arrow_validation", "test_attnres_kernels", "test_blackwell_micro_gemm", "test_blackwell_perf_deepseek",
    "test_blackwell_perf_dense", "test_blackwell_perf_glm", "test_blackwell_perf_kimi", "test_blackwell_perf_moe",
    "test_blackwell_tc_mma", "test_code_naming_guard", "test_code_selfcheck", "test_converter_helpers", "test_tool_format_json", "test_config_json",
    "test_core", "test_ctest_labels_guard", "test_cuda_event_wait", "test_cuda_toolkit_guard",
    "test_det_selfcheck", "test_diagnostics", "test_expert_store", "test_expert_store_guard",
    "test_expert_store_stage", "test_fabric_client", "test_fabric_wire", "test_fast_load",
    "test_fast_math_guard_cross_tu", "test_fast_math_guard_other_tu", "test_fsi_selfcheck", "test_generation_cli",
    "test_generation_stream", "test_geometry_guard", "test_gpu_perf", "test_gpu_power",
    "test_hardware_profile", "test_http_server", "test_incremental_detokenizer", "test_infrastructure_guard",
    "test_integrated_pipeline", "test_json_escape", "test_kernels_common", "test_kv_cache_layout", "test_kv_cache_layout_device",
    "test_license_guard", "test_lifecycle_plan", "test_logging_guard", "test_lsx_gemv",
    "test_lsx_kernels", "test_lsx_nvfp4_dispatch", "test_manifest_sync", "test_model_engine",
    "test_model_factory", "test_model_independence_guard", "test_model_integrity", "test_model_runtime",
    "test_model_suite_guard", "test_mosaic_frontend_route", "test_mosaic_network", "test_nvfp4_pack",
    "test_nvfp4_scale", "test_parity_selfcheck_naming_guard", "test_path_registry", "test_perf_baseline",
    "test_perf_baseline_logic", "test_perf_knobs_apply", "test_pread_pool", "test_prompt_bridge",
    "test_resource_governor", "test_runtime_tuner", "test_safetensors", "test_sampling",
    "test_schema_views", "test_stream_read_ceiling", "test_stream_scratch", "test_tokenizer",
    "test_tokenizer_arrows", "test_tokenizer_vocab_contract", "test_tool_format", "test_vram_estimate",
})


# ── emitters ──────────────────────────────────────────────────────────────

def cstr(value):
    if value is None:
        return "nullptr"
    out = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{out}"'


def emit_header(manifests) -> str:
    models = sorted(manifests, key=lambda m: m.model_type)
    rows = []
    for m in models:
        rows.append(
            "    {%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %d, %s, %s, %s, %s, %d, %s},"
            % (cstr(m.model_type), cstr(m.family), cstr(m.variant),
               cstr(m.model_class), cstr(m.model_header), cstr(m.passport),
               cstr(m.ns), cstr(m.format_tag), cstr(m.vault_dir),
               cstr(m.vault_env), m.tokenizer_kind,
               "true" if m.tool_roundtrip else "false",
               "true" if m.auto_path_determinism else "false",
               "true" if m.cooperative_cancel else "false",
               "true" if m.selfcheck_frame_prompt else "false",
               m.selfcheck_max_gen,
               "true" if m.selfcheck_bit_exact else "false"))
    targets = []
    for m in models:
        for t in sorted(m.targets(), key=lambda x: (x.stage, x.name)):
            targets.append(
                "    {%s, %s, %s, %s, %s, %d, %s, %s, %s},"
                % (cstr(t.model_type), cstr(t.stage), cstr(t.name),
                   cstr(t.source), cstr(t.label), t.timeout,
                   "true" if t.skip else "false",
                   cstr(";".join(t.env)) if t.env else "nullptr",
                   cstr(t.resource_lock)))
    lifecycle = []
    for m in models:
        for tu in m.lifecycle_tu:
            lifecycle.append("    {%s, %s}," % (cstr(m.model_type), cstr(tu)))

    debt = []
    for m in models:
        for stage in sorted(m.golden_debt):
            debt.append("    {%s, %s, %s}," % (
                cstr(m.model_type), cstr(stage), cstr(m.golden_debt[stage])))

    return f"""// generated by tools/gen_model_manifest.py from models/*.json -- do not edit
#pragma once
#include <cstddef>
namespace lsxmanifest {{
struct ModelRow {{
    const char *model_type; const char *family; const char *variant;
    const char *model_class; const char *model_header; const char *passport;
    const char *ns; const char *format_tag; const char *vault_dir;
    const char *vault_env; int tokenizer_kind; bool tool_roundtrip;
    bool auto_path_determinism; bool cooperative_cancel;
    bool selfcheck_frame_prompt; int selfcheck_max_gen;
    bool selfcheck_bit_exact;
}};
struct TargetRow {{
    const char *model_type; const char *stage; const char *target;
    const char *source; const char *label; int timeout; bool skip;
    const char *env; const char *resource_lock;
}};
struct LifecycleRow {{ const char *model_type; const char *tu; }};
struct DebtRow {{ const char *model_type; const char *stage; const char *reason; }};
inline constexpr ModelRow kModels[] = {{
{chr(10).join(rows)}
}};
inline constexpr TargetRow kTargets[] = {{
{chr(10).join(targets)}
}};
inline constexpr LifecycleRow kLifecycle[] = {{
{chr(10).join(lifecycle)}
}};
inline constexpr DebtRow kDebt[] = {{
{chr(10).join(debt)}
}};
inline constexpr std::size_t kModelCount = {len(models)};
inline constexpr std::size_t kLifecycleCount = {len(lifecycle)};
inline constexpr std::size_t kDebtCount = {len(debt)};
inline constexpr std::size_t kTargetCount = {len(targets)};
}}  // namespace lsxmanifest
"""


def _env_list(env):
    return ";".join(env)


def emit_cmake(manifests) -> str:
    """Registration only: add_test + set_tests_properties, never add_executable.

    Build wiring (add_executable, the CUDA OBJECT libraries, per-target include
    dirs and defines) stays in tests/CMakeLists.txt: it is per-target compile
    reality, not a projection of the model registry. What IS derived here is the
    part that drifted silently - labels, timeouts, skip codes, resource locks and
    environments - which is exactly what the ctest oracle records.

    A target claimed by several models (test_deepseek_tool_framing is named by all
    three deepseek rows) is registered ONCE; re-registering a name would make one
    manifest's label and timeout silently win over another's.
    """
    emitted = set()
    out = ["# generated by tools/gen_model_manifest.py from models/*.json -- do not edit", ""]
    for m in sorted(manifests, key=lambda x: x.model_type):
        out.append(f"# ---- {m.model_type}" + (f" ({m.notes})" if m.notes else ""))
        for t in sorted(m.targets(), key=lambda x: (STAGES.index(x.stage), x.name)):
            if t.name in emitted:
                continue
            emitted.add(t.name)
            if t.stage == "perf":
                out.append(f"add_test(NAME {t.name} COMMAND {PERF_BINARY} "
                           f"\"{t.perf_case}\")")
                props = [f"LABELS {t.label}"]
                if t.timeout:
                    props.append(f"TIMEOUT {t.timeout}")
                if t.skip:
                    props.append("SKIP_RETURN_CODE 4")
                if t.resource_lock:
                    props.append(f'RESOURCE_LOCK "{t.resource_lock}"')
                out.append(f"set_tests_properties({t.name} PROPERTIES "
                           + " ".join(props) + ")")
                if t.env:
                    out.append(f"set_tests_properties({t.name} PROPERTIES "
                               f'ENVIRONMENT "{_env_list(t.env)}")')
                out.append("")
                continue

            if t.command != t.name or t.filter:
                props = [f"LABELS {t.label}"]
                if t.timeout:
                    props.append(f"TIMEOUT {t.timeout}")
                if t.skip:
                    props.append("SKIP_RETURN_CODE 4")
                if t.resource_lock:
                    props.append(f'RESOURCE_LOCK "{t.resource_lock}"')
                quoted = f' "{t.filter}"' if t.filter else ""
                out.append(f"add_test(NAME {t.name} COMMAND {t.command}{quoted})")
                out.append(f"set_tests_properties({t.name} PROPERTIES "
                           + " ".join(props) + ")")
                if t.env:
                    out.append(f"set_tests_properties({t.name} PROPERTIES "
                               f'ENVIRONMENT "{_env_list(t.env)}")')
                out.append("")
                continue

            if t.form == "selfcheck":
                out.append(f"add_test(NAME {t.name} COMMAND {t.name})")
                props = []
                if t.timeout:
                    props.append(f"TIMEOUT {t.timeout}")
                if t.skip:
                    props.append("SKIP_RETURN_CODE 4")
                if t.resource_lock:
                    props.append(f'RESOURCE_LOCK "{t.resource_lock}"')
                if props:
                    out.append(f"set_tests_properties({t.name} PROPERTIES "
                               + " ".join(props) + ")")
                out.append(f"set_tests_properties({t.name} PROPERTIES "
                           f"LABELS {t.label})")
                out.append("")
                continue

            if t.form == "parity":
                # One driver, one binary per (model, kind): the model and the
                # gate kind are compile-time constants, everything else comes
                # from the manifest row.
                out.append(
                    f"add_test(NAME {t.name} COMMAND {t.name})")
                props = []
                if t.timeout:
                    props.append(f"TIMEOUT {t.timeout}")
                if t.skip:
                    props.append("SKIP_RETURN_CODE 4")
                if t.resource_lock:
                    props.append(f'RESOURCE_LOCK "{t.resource_lock}"')
                if props:
                    out.append(f"set_tests_properties({t.name} PROPERTIES "
                               + " ".join(props) + ")")
                out.append(f"set_tests_properties({t.name} PROPERTIES "
                           f"LABELS {t.label})")
                out.append("")
                continue

            if t.form == "helper":
                # A target that builds its own executable through the helper
                # (its build wiring is one statement, not a block).
                all_sources = [t.source, *t.sources]
                args = [f"NAME {t.name}", "SOURCES " + " ".join(all_sources),
                        "LINK lsxcommon",
                        f'INCLUDE "${{PROJECT_SOURCE_DIR}}/libs/lsxcommon/src"']
                if t.defines:
                    args.append("DEFINE " + " ".join(
                        '"' + d.replace('"', '\\"') + '"' for d in t.defines))
                args.append(f"LABEL {t.label}")
                if t.timeout:
                    args.append(f"TIMEOUT {t.timeout}")
                if t.skip:
                    args.append("SKIP")
                if t.env:
                    args.append(f'ENV "{_env_list(t.env)}"')
                emit = "lsx_add_cuda_test" if t.cuda else "lsx_add_test"
                out.append(emit + "(" + " ".join(args) + ")")
                if t.resource_lock:
                    out.append(f'set_tests_properties({t.name} PROPERTIES '
                               f'RESOURCE_LOCK "{t.resource_lock}")')
                out.append("")
                continue

            # Registration only; the build wiring stays in tests/CMakeLists.txt.
            out.append(f"add_test(NAME {t.name} COMMAND {t.name})")
            props = []
            if t.timeout:
                props.append(f"TIMEOUT {t.timeout}")
            if t.skip:
                props.append("SKIP_RETURN_CODE 4")
            if t.resource_lock:
                props.append(f'RESOURCE_LOCK "{t.resource_lock}"')
            if props:
                out.append(f"set_tests_properties({t.name} PROPERTIES "
                           + " ".join(props) + ")")
            if t.env:
                out.append(f"set_tests_properties({t.name} PROPERTIES "
                           f'ENVIRONMENT "{_env_list(t.env)}")')
            out.append(f"set_tests_properties({t.name} PROPERTIES "
                       f"LABELS {t.label})")
            out.append("")
    return "\n".join(out)


def emit_registry(manifests) -> str:
    """A complete TU fragment: includes at file scope, one registration function.

    It must be included from FILE scope, not inside a function: the per-model
    headers cannot be expanded inside a lambda body.
    """
    models = sorted(manifests, key=lambda m: m.model_type)
    lines = ["// generated by tools/gen_model_manifest.py -- do not edit",
             "// Included at file scope by libs/lsxcommon/src/model_registry.cpp.",
             ""]
    # One include per HEADER, not per model: siblings can share a model header
    # (granite-switch 30B and 8B are two classes in one TU), and a duplicate
    # `#include` line reads like a copy-paste bug in a file nobody may edit by
    # hand. `pragma once` makes it harmless; the line is removed for clarity.
    seen_headers = set()
    for m in models:
        if m.model_header in seen_headers:
            continue
        seen_headers.add(m.model_header)
        lines.append(f'#include "{m.model_header}"')
    lines += ["", "namespace lsxcommon {", "namespace {", "",
              "void RegisterAllModels()", "{"]
    for m in models:
        lines.append(f"    LsxModelFactory::Register<{m.model_class}>();")
    if any(m.aliases for m in models):
        lines += ["",
                  "    // Vault-normalized aliases: path auto-detection lower-cases the",
                  "    // directory name, so a tree stored as \"Beam-1.0-501B-A23B\" must",
                  "    // also resolve as \"beam-1.0-501b-a23b\". Declared per manifest row",
                  "    // (validation: an alias is exactly the lower-cased vault dir)."]
        for m in models:
            for alias in m.aliases:
                lines.append(
                    f'    LsxModelFactory::Register("{alias}", [](const '
                    f"ModelInitConfig &c) {{ return "
                    f"std::make_unique<{m.model_class}>(c); }});")
    lines += ["}", "", "}  // namespace", "}  // namespace lsxcommon", ""]
    return "\n".join(lines)


def emit_model_section(m, baseline) -> str:
    """The catalogue block for one model. Kept separate so the line budget is
    measured on the same text that gets written (see check_doc_budget)."""
    gates = []
    for stage in ("selfcheck", "fsi", "code", "det"):
        names = m.stages.get(stage, [])
        gates.append(
            f"{stage}: "
            f"{'debt' if stage in m.golden_debt else ('yes' if names else 'no')}")
    perf = (baseline or {}).get(m.model_type, {})
    perf_row = (f"{perf.get('tokens_per_sec')} t/s ({perf.get('mode')}, "
                f"{perf.get('date')})" if perf else "not seeded")
    out = [
        f"## {m.model_type}",
        "",
        f"- family `{m.family}`" + (f", variant `{m.variant}`" if m.variant else ""),
        # The passport IS the geometry source (constexpr under static_assert),
        # so the line says so rather than leaving a reader to wonder whether the
        # numbers were omitted by accident.
        f"- class `{m.model_class}`, namespace `{m.ns}`",
        f"- geometry: `{m.passport}` (frozen constants, static_assert-checked)",
    ]
    if m.format_tag or m.forge_flag:
        out.append(
            (f"- format `{m.format_tag}`" if m.format_tag else "- format not "
             f"declared")
            + (f", forge `{m.forge_flag}`" if m.forge_flag else ""))
    if m.tokenizer_kind:
        out.append(f"- tokenizer kind {m.tokenizer_kind}")
    out.append(f"- vault `{m.vault_dir}` (override `{m.vault_env}`)")
    out.append(f"- lifecycle TUs: "
               f"{', '.join(f'`{t}`' for t in m.lifecycle_tu) or 'none declared'}")
    out.append(f"- gates: {'; '.join(gates)}")
    out.append(f"- perf: {perf_row}")
    if m.golden_debt:
        out.append("- debt: " + "; ".join(
            f"{k} — {v}" for k, v in sorted(m.golden_debt.items())))
    # Absence is stated, not silent: the report is Model Ingestion Protocol
    # stage 9, and a catalogue that simply omits the line lets a missing one read
    # as "not applicable".
    out.append(f"- report: `{m.report}`" if m.report
               else "- report: **not written** (stage-9 debt)")
    if m.notes:
        out.append(f"- notes: {m.notes}")
    return "\n".join(out) + "\n"


def emit_docs(manifests, baseline) -> str:
    lines = [
        "# Models — generated catalogue",
        "",
        "<!-- generated by tools/gen_model_manifest.py from models/*.json;",
        "     do not edit. `ctest -R test_manifest_sync` fails on any drift. -->",
        "",
        f"{len(manifests)} registered models. Per-model narrative, rationale and",
        "history live in `docs/model-reports/`; gate verdicts in",
        "`docs/SELFCHECK_MATRIX.md`; throughput in `tests/perf/perf_baselines.json`.",
        "",
        "Legend: `gates: <stage>: yes` means the stage target and its golden exist — not that every probe passed (measured verdicts, including documented red cells, are in `docs/SELFCHECK_MATRIX.md`); `debt` means the golden was never captured.",
        "",
    ]
    for m in sorted(manifests, key=lambda x: (x.family, x.model_type)):
        lines.append(emit_model_section(m, baseline).rstrip("\n"))
        lines.append("")
    return "\n".join(lines) + "\n"


def check_matrix(manifests, root):
    """The matrix holds measured verdicts, so it is not generated.

    What IS checked: every model row in the matrix exists in the manifests, and
    every `debt` cell corresponds to a declared `golden_debt` reason. That is the
    drift direction that matters - a manifest row renamed or a debt resolved
    without the matrix following it.
    """
    path = root / "docs" / "SELFCHECK_MATRIX.md"
    if not path.is_file():
        return []
    text = path.read_text()
    # Only the model verdict tables: they are the ones with this header. Other
    # tables in the file describe what a gate MEANS, not a model.
    tail = text.split("| model (passport) |", 1)
    if len(tail) < 2:
        return []
    rows = []
    for block in re.split(r"\n## ", tail[1]):
        for line in block.splitlines():
            m = re.match(r"^\|\s*\**([a-z0-9][a-z0-9.\-]*)\**\s*\|", line)
            if m and m.group(1) != "model":
                rows.append(m.group(1))
    known = {m.model_type for m in manifests}
    return [f"docs/SELFCHECK_MATRIX.md: row {name!r} has no manifest row"
            for name in rows
            if name not in known and name not in ("model (passport)",)]


def load_manifests(root: Path, partial=False):
    mdir = root / "models"
    paths = sorted(mdir.glob("*.json")) if mdir.is_dir() else []
    manifests = [load_one(p) for p in paths]
    return manifests


def perf_baseline(root: Path):
    path = root / "tests" / "perf" / "perf_baselines.json"
    if not path.is_file():
        return {}
    entries = json.loads(path.read_text()).get("entries", [])
    latest = {}
    for e in entries:
        key = e.get("model_type", "")
        if not key or key.startswith("__"):
            continue
        if key not in latest or e.get("date", "") >= latest[key].get("date", ""):
            latest[key] = e
    return latest


def write(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(f"wrote {p} ({len(text.splitlines())} lines)", file=sys.stderr)


def project_targets(manifests):
    """The shape the ctest oracle records, for --diff-cmake."""
    out = {}
    for m in manifests:
        for t in m.targets():
            out[t.name] = {
                "name": t.name,
                "labels": [t.label],
                "timeout": t.timeout or None,
                "skip_return_code": 4 if t.skip else None,
                "environment": list(t.env),
                "resource_lock": t.resource_lock or None,
            }
    return out


def diff_cmake(manifests, baseline_path):
    doc = json.loads(Path(baseline_path).read_text())
    base = {t["name"]: t for t in doc["tests"]}
    mine = project_targets(manifests)
    norm = lambda t: json.dumps({"labels": sorted(t["labels"]),
                                 "timeout": t["timeout"],
                                 "skip_return_code": t["skip_return_code"],
                                 "environment": sorted(t["environment"]),
                                 "resource_lock": t["resource_lock"]},
                                sort_keys=True)
    problems, compared = [], 0
    for name, want in sorted(base.items()):
        got = mine.get(name)
        if got is None:
            continue          # not a model target (or not authored yet)
        compared += 1
        if norm(want) != norm(got):
            problems.append(f"{name}: baseline {norm(want)} != manifest {norm(got)}")
    missing = sorted(set(mine) - set(base))
    for name in missing:
        problems.append(f"{name}: manifest target absent from the baseline")
    print(f"compared {compared} targets, {len(problems)} difference(s)",
          file=sys.stderr)
    for p in problems:
        print(p)
    return 0 if not problems else 1


def self_test(root: Path):
    """Every schema rule is reported, proven against a LIVE manifest.

    The base is a real `models/*.json` row, mutated in memory. A checked-in copy
    of one was the obvious first design and it went stale the first time the tree
    moved (the deleted per-family gate files made the "valid" fixture invalid). A
    mutation of the real base cannot drift: if the base changes shape, this fails
    loudly.
    """
    base_path = root / "models" / "flint.json"
    if not base_path.is_file():
        print(f"error: {base_path} is missing (self-test needs a real base row)",
              file=sys.stderr)
        return 2
    base = json.loads(base_path.read_text())

    cases = []

    def add(name, mutate, needle, cross_model=False):
        cases.append((name, mutate, needle, cross_model))

    def unknown_stage(d):
        d["stages"]["mystery"] = ["test_x"]

    add("unknown_stage", unknown_stage, "unknown stage")

    def missing_passport(d):
        d["passport"] = "libs/lsxcommon/src/nope.h"

    add("missing_passport", missing_passport, "passport not found")

    def ktype_mismatch(d):
        d["model_class"] = "NotAFlintModel"

    add("ktype_mismatch", ktype_mismatch, "model_class not found in src")

    def orphan_target(d):
        d["stages"]["model"].append("test_does_not_exist")

    add("orphan_target", orphan_target, "no source for target")

    def undeclared_gap(d):
        d["stages"]["fsi"] = []
        d.get("golden_debt", {}).pop("fsi", None)

    add("undeclared_gap", undeclared_gap, "gap without debt")

    def vault_env(d):
        d["vault"]["env"] = "LSX_TEST_WRONG_ROOT"

    add("vault_env", vault_env, "vault.env must be LSX_TEST_FLINT_ROOT")

    def duplicate_vault(d):
        d["vault"]["dir"] = "GLM-5.2"

    # Cross-model rules only run with more than one manifest in play.
    add("duplicate_vault", duplicate_vault, "already used by", cross_model=True)

    def namespace_clash(d):
        d["namespace"] = "lsxcommon::glm"

    add("namespace_clash", namespace_clash, "already owned by", cross_model=True)

    # The selfcheck probe fields decide the recorded logits_hash, so a typo in one
    # is a silent re-baseline. Only the integer needs range-checking; the three
    # booleans are coerced by load_one and cannot be out of range.
    def misspelled_field(d):
        d["selfcheck_frame_promt"] = True       # one letter short

    add("misspelled_field", misspelled_field, "unknown field")

    def probe_budget_zero(d):
        d["selfcheck_max_gen"] = 0

    add("probe_budget_zero", probe_budget_zero, "outside 2..4096")

    def probe_budget_huge(d):
        d["selfcheck_max_gen"] = 100000

    add("probe_budget_huge", probe_budget_huge, "outside 2..4096")

    # The unmutated base must validate: that is the control, and it is what keeps
    # this mode honest (a validator that rejected everything would still report
    # "8/8").
    errs, _warn = validate([load_one(base_path)], root, partial=True)
    if errs:
        print("error: the base manifest does not validate:", file=sys.stderr)
        for e in errs:
            print("  " + e, file=sys.stderr)
        return 1

    peer_path = root / "models" / "glm-5.2.json"
    peer = load_one(peer_path) if peer_path.is_file() else None
    if peer is None:
        print("error: self-test needs models/glm-5.2.json as the second model",
              file=sys.stderr)
        return 2

    ok = 0
    for name, mutate, needle, cross_model in cases:
        doc = json.loads(json.dumps(base))
        mutate(doc)
        tmp = root / "models" / f".self-test-{name}.json"
        try:
            tmp.write_text(json.dumps(doc))
            m = load_one(tmp)
            manifests = [m, peer] if cross_model else [m]
            errs, _warn = validate(manifests, root, partial=not cross_model)
        except ManifestError as exc:
            errs = [str(exc)]
        finally:
            tmp.unlink(missing_ok=True)
        if any(needle in e for e in errs):
            ok += 1
        else:
            print(f"{name}: expected a message containing {needle!r}, got:",
                  file=sys.stderr)
            for e in errs:
                print("  " + e, file=sys.stderr)
    print(f"{ok}/{len(cases)} schema violations reported (base: "
          f"{base_path.name}, live)")
    return 0 if ok == len(cases) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--header")
    ap.add_argument("--cmake")
    ap.add_argument("--registry")
    ap.add_argument("--docs")
    ap.add_argument("--check", action="store_true",
                    help="regenerate the committed docs and fail on drift")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on a test file no manifest claims "
                         "(run with --check once every model is manifested)")
    ap.add_argument("--self-test", action="store_true",
                    help="mutate a live manifest and assert every schema rule "
                         "is reported")
    ap.add_argument("--diff-cmake", metavar="BASELINE_JSON")
    ap.add_argument("--print-model-rows", action="store_true")
    ap.add_argument("--print-non-model-tests", action="store_true")
    ap.add_argument("--print-selfcheck-targets", action="store_true",
                    help="one Paris SelfCheck target per line, for a serial sweep")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()

    if args.self_test:
        return self_test(root)

    if args.print_non_model_tests:
        for name in non_model_tests(root):
            print(name)
        return 0

    try:
        manifests = load_manifests(root)
        errors, warnings = validate(manifests, root,
                                    strict_orphans=args.strict)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if warnings:
        print(f"{len(warnings)} warning(s):", file=sys.stderr)
        for w in warnings[:5]:
            print(f"warning: {w}", file=sys.stderr)
        if len(warnings) > 5:
            print(f"warning: ... and {len(warnings) - 5} more "
                  f"(run with --check to see them all)", file=sys.stderr)
    if warnings and args.strict:
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} manifest error(s):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 2

    if args.print_selfcheck_targets:
        # Exists so a sweep is derived from the manifests instead of a hand-typed
        # regex: the 2026-11-07 audit found a hand-written filter that silently
        # omitted 15 of the 37 SelfCheck targets, and an omitted gate reads exactly
        # like a passing one. Sorted by model_type so a diff is stable.
        for m in sorted(manifests, key=lambda x: x.model_type):
            for t in m.stages.get("selfcheck", []):
                print(t)
        return 0

    if args.print_model_rows:
        for m in sorted(manifests, key=lambda x: x.model_type):
            values = row_values(m)
            print("\t".join(str(values[f]) if values[f] is not None else ""
                            for f in ROW_FIELDS))
        return 0

    if args.header:
        write(args.header, emit_header(manifests))
    if args.cmake:
        write(args.cmake, emit_cmake(manifests))
    if args.registry:
        write(args.registry, emit_registry(manifests))
    if args.docs:
        write(args.docs, emit_docs(manifests, perf_baseline(root)))

    if args.check:
        rc = 0
        docs = root / "docs" / "MODELS.md"
        want = emit_docs(manifests, perf_baseline(root))
        got = docs.read_text() if docs.is_file() else ""
        if got != want:
            print("docs/MODELS.md is stale - regenerate with --docs:",
                  file=sys.stderr)
            for line in list(difflib.unified_diff(
                    got.splitlines(), want.splitlines(),
                    "committed", "generated", lineterm=""))[:40]:
                print(line, file=sys.stderr)
            rc = 1
        return rc

    if args.diff_cmake:
        return diff_cmake(manifests, args.diff_cmake)

    return 0


if __name__ == "__main__":
    sys.exit(main())
