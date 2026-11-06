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
STAGE_LOCKS = {
    "convert": "", "load": "", "model": "", "kernels": "", "selfcheck": "",
    "fsi": "", "code": "", "det": "", "perf": "gpu",
}

# Stages whose tests build a synthetic checkpoint/tree and therefore need the
# geometry override. deviates are expressed in the manifest's target_overrides.
SYNTHETIC_ENV = "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1"
SYNTHETIC_STAGES = ("convert", "load", "model")

PERF_CASE_FMT = "perf baseline: {}"
PERF_BINARY = "test_perf_baseline"

MODEL_FIELDS = ("model_type", "family", "variant", "model_class",
                "model_header", "passport", "namespace", "format_tag",
                "vault_dir", "vault_env", "tokenizer_kind",
                "tool_roundtrip", "auto_path_determinism")


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
    perf_case: str = ""


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
    resource_lock: str
    report: str
    notes: str
    path: str

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
        source = f"{name}.cu" if (Path("tests") / f"{name}.cu").exists() else f"{name}.cpp"
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
        return Target(name=name, stage=stage, model_type=self.model_type,
                      source=source, label=label, timeout=timeout, skip=skip,
                      env=env, resource_lock=lock, defines=defines, cuda=cuda,
                      perf_case=perf_case)


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


def load_one(path: Path) -> Manifest:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(path.name, f"line {exc.lineno}", f"invalid JSON: {exc.msg}")

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
        auto_path_determinism=bool(doc.get("auto_path_determinism", False)),
        report=doc.get("report", ""),
        notes=doc.get("notes", ""),
        path=path.name,
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
            if not re.search(r"\bnamespace\s+" + re.escape(last) + r"\b", text):
                add(m, "namespace",
                    f"passport {m.passport} does not declare 'namespace {last}'")

        # rule 5: the model class exists and its kType is this model_type
        header = src / m.model_header if "/" not in m.model_header else root / m.model_header
        if not header.is_file():
            add(m, "model_header", f"model_class not found in src: {m.model_header}")
        else:
            text = strip_comments(header.read_text(errors="replace"))
            if not re.search(r"\b(class|struct)\s+" + re.escape(m.model_class) + r"\b",
                             text):
                add(m, "model_class",
                    f"model_class not found in src: {m.model_class} "
                    f"(not declared in {m.model_header})")
        ktype = None
        for cand in sorted(src.glob("*.cpp")) + sorted(src.glob("*.h")):
            hit = re.search(
                re.escape(m.model_class) + r"\s*::\s*kType\s*=\s*\"([^\"]*)\"",
                cand.read_text(errors="replace"))
            if hit:
                ktype = hit.group(1)
                break
        if ktype is None:
            add(m, "model_class", f"model_class not found in src: {m.model_class} "
                                  f"has no kType literal")
        elif ktype != m.model_type:
            add(m, "model_class",
                f"model_class {m.model_class} declares kType {ktype!r}, "
                f"manifest says {m.model_type!r}")

        # rule 6: every stage target resolves to a source file
        for t in m.targets():
            if t.stage == "perf":
                if not (root / "tests" / f"{PERF_BINARY}.cpp").is_file():
                    add(m, "stages.perf", f"no source for target {t.name} "
                                          f"({PERF_BINARY}.cpp missing)")
                continue
            if not (root / "tests" / t.source).is_file():
                add(m, f"stages.{t.stage}",
                    f"no source for target {t.name}: tests/{t.source} does not exist")

    if not partial:
        seen_type, seen_ns, fam_variant = {}, {}, {}
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
        (errors if strict_orphans else warnings).extend(
            check_orphans(manifests, root))
    return errors, warnings


def claimed_targets(manifests):
    names = set()
    for m in manifests:
        for stage in STAGES:
            names.update(m.stages.get(stage, []))
    return names


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
        for names in doc.get("stages", {}).values():
            claimed.update(names)
    explicit = _NON_MODEL_TESTS
    return sorted(claimed | explicit)


_NON_MODEL_TESTS = frozenset()


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
            "    {%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %d, %s, %s},"
            % (cstr(m.model_type), cstr(m.family), cstr(m.variant),
               cstr(m.model_class), cstr(m.model_header), cstr(m.passport),
               cstr(m.ns), cstr(m.format_tag), cstr(m.vault_dir),
               cstr(m.vault_env), m.tokenizer_kind,
               "true" if m.tool_roundtrip else "false",
               "true" if m.auto_path_determinism else "false"))
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

    return f"""// generated by tools/gen_model_manifest.py from models/*.json -- do not edit
#pragma once
#include <cstddef>
namespace lsxmanifest {{
struct ModelRow {{
    const char *model_type; const char *family; const char *variant;
    const char *model_class; const char *model_header; const char *passport;
    const char *ns; const char *format_tag; const char *vault_dir;
    const char *vault_env; int tokenizer_kind; bool tool_roundtrip;
    bool auto_path_determinism;
}};
struct TargetRow {{
    const char *model_type; const char *stage; const char *target;
    const char *source; const char *label; int timeout; bool skip;
    const char *env; const char *resource_lock;
}};
struct LifecycleRow {{ const char *model_type; const char *tu; }};
inline constexpr ModelRow kModels[] = {{
{chr(10).join(rows)}
}};
inline constexpr TargetRow kTargets[] = {{
{chr(10).join(targets)}
}};
inline constexpr LifecycleRow kLifecycle[] = {{
{chr(10).join(lifecycle)}
}};
inline constexpr std::size_t kModelCount = {len(models)};
inline constexpr std::size_t kTargetCount = {len(targets)};
}}  // namespace lsxmanifest
"""


def _env_list(env):
    return ";".join(env)


def emit_cmake(manifests) -> str:
    out = ["# generated by tools/gen_model_manifest.py from models/*.json -- do not edit", ""]
    for m in sorted(manifests, key=lambda x: x.model_type):
        out.append(f"# ---- {m.model_type}" + (f" ({m.notes})" if m.notes else ""))
        if m.model_type != m.variant and m.family in ("qwen", "gemma", "mistral", "kimi"):
            pass
        for t in sorted(m.targets(), key=lambda x: (STAGES.index(x.stage), x.name)):
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

            args = [f"NAME {t.name}", f"SOURCES {t.source}", "LINK lsxcommon"]
            args.append(f'INCLUDE "${{PROJECT_SOURCE_DIR}}/libs/lsxcommon/src"')
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
            if t.cuda:
                out.append("lsx_add_cuda_test(" + " ".join(args) + ")")
            else:
                out.append("lsx_add_test(" + " ".join(args) + ")")
            if t.resource_lock:
                out.append(f'set_tests_properties({t.name} PROPERTIES '
                           f'RESOURCE_LOCK "{t.resource_lock}")')
            out.append("")
    return "\n".join(out)


def emit_registry(manifests) -> str:
    models = sorted(manifests, key=lambda m: m.model_type)
    lines = ["// generated by tools/gen_model_manifest.py -- do not edit"]
    for m in models:
        lines.append(f'#include "{m.model_header}"')
    lines.append("")
    for m in models:
        lines.append(f"        LsxModelFactory::Register<{m.model_class}>();")
    return "\n".join(lines) + "\n"


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
    ]
    for m in sorted(manifests, key=lambda x: (x.family, x.model_type)):
        gates = []
        for stage in ("selfcheck", "fsi", "code", "det"):
            names = m.stages.get(stage, [])
            gates.append(f"{stage}: {'debt' if stage in m.golden_debt else ('yes' if names else 'no')}")
        perf = baseline.get(m.model_type, {})
        perf_row = (f"{perf.get('tokens_per_sec')} t/s ({perf.get('mode')}, "
                    f"{perf.get('date')})" if perf else "not seeded")
        lines += [
            f"## {m.model_type}",
            "",
            f"- family `{m.family}`" + (f", variant `{m.variant}`" if m.variant else ""),
            f"- class `{m.model_class}`, passport `{m.passport}`, "
            f"namespace `{m.ns}`",
            f"- format `{m.format_tag}`, forge `{m.forge_flag}`, "
            f"tokenizer kind {m.tokenizer_kind}",
            f"- vault `{m.vault_dir}` (override `{m.vault_env}`)",
            f"- lifecycle TUs: {', '.join(f'`{t}`' for t in m.lifecycle_tu) or 'none declared'}",
            f"- gates: {'; '.join(gates)}",
            f"- perf: {perf_row}",
            f"- debt: " + ("; ".join(f"{k} — {v}" for k, v in sorted(m.golden_debt.items()))
                           or "none"),
        ]
        if m.report:
            lines.append(f"- report: `{m.report}`")
        if m.notes:
            lines.append(f"- notes: {m.notes}")
        lines.append("")
    return "\n".join(lines) + "\n"


def emit_matrix_rows(manifests) -> str:
    out = ["| model (passport) | SelfCheck | FSI | CODE | DET |", "|---|---|---|---|---|"]
    for m in sorted(manifests, key=lambda x: x.model_type):
        cells = []
        for stage in ("selfcheck", "fsi", "code", "det"):
            if stage in m.golden_debt:
                cells.append("debt")
            elif m.stages.get(stage):
                cells.append("?" if stage != "selfcheck" else "PASS")
            else:
                cells.append("—")
        out.append(f"| {m.model_type} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ── modes ─────────────────────────────────────────────────────────────────

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


def self_test(fixtures: Path, root: Path):
    valid = fixtures / "valid" / "flint.json"
    manifests = [load_one(valid)]
    errs, _warn = validate(manifests, root, partial=True)
    if errs:
        print("valid fixture FAILED validation:", file=sys.stderr)
        for e in errs:
            print("  " + e, file=sys.stderr)
        return 1
    expected = {
        "unknown_stage": "unknown stage",
        "missing_passport": "passport not found",
        "ktype_mismatch": "model_class not found in src",
        "orphan_target": "no source for target",
        "undeclared_gap": "gap without debt",
        "vault_env": "vault.env must be LSX_TEST_FLINT_ROOT",
    }
    ok = 0
    for stem, needle in expected.items():
        path = fixtures / "invalid" / f"{stem}.json"
        try:
            m = load_one(path)
            errs, _warn = validate([m], root, partial=True)
        except ManifestError as exc:
            errs = [str(exc)]
        hit = any(needle in e for e in errs)
        if hit:
            ok += 1
        else:
            print(f"{stem}: expected a message containing {needle!r}, got:",
                  file=sys.stderr)
            for e in errs:
                print("  " + e, file=sys.stderr)
    print(f"{ok}/{len(expected)} schema violations reported")
    return 0 if ok == len(expected) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--header")
    ap.add_argument("--cmake")
    ap.add_argument("--registry")
    ap.add_argument("--docs")
    ap.add_argument("--matrix")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", metavar="FIXTURES")
    ap.add_argument("--diff-cmake", metavar="BASELINE_JSON")
    ap.add_argument("--print-model-rows", action="store_true")
    ap.add_argument("--print-non-model-tests", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()

    if args.self_test:
        return self_test(Path(args.self_test), root)

    if args.print_non_model_tests:
        for name in non_model_tests(root):
            print(name)
        return 0

    try:
        manifests = load_manifests(root)
        errors, warnings = validate(manifests, root,
                                    strict_orphans=args.check)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} manifest error(s):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 2

    if args.print_model_rows:
        for m in sorted(manifests, key=lambda x: x.model_type):
            print("\t".join(str(v) if v is not None else ""
                            for v in (m.model_type, m.family, m.variant,
                                      m.model_class, m.model_header, m.passport,
                                      m.ns, m.format_tag, m.vault_dir,
                                      m.vault_env, m.tokenizer_kind,
                                      int(m.tool_roundtrip),
                                      int(m.auto_path_determinism))))
        return 0

    if args.header:
        write(args.header, emit_header(manifests))
    if args.cmake:
        write(args.cmake, emit_cmake(manifests))
    if args.registry:
        write(args.registry, emit_registry(manifests))
    if args.docs:
        write(args.docs, emit_docs(manifests, perf_baseline(root)))
    if args.matrix:
        write(args.matrix, emit_matrix_rows(manifests))

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
