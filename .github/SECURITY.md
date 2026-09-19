# Security Policy

## Supported Versions

logestix is pre-1.0 and ships from `main`. Security fixes are applied to the
latest `main` only.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security problems. Instead,
report them privately to **<info@logestix.io>** with:

- a description of the issue and its impact;
- the affected component (file/commit if known);
- a minimal reproduction or proof of concept;
- the environment (OS, CUDA toolkit, GPU).

You can expect an acknowledgement within a few business days. Once a fix is
available it will be released on `main` and, when appropriate, a GitHub Security
Advisory will be published.

## Scope Notes

- The model vault, converted checkpoints and reference outputs are **not** part
  of this repository; do not report issues that require those assets unless the
  problem is in the code path that consumes them.
- Vendored third-party code lives under `third_party/` (submodules) and is
  tracked in `DEPENDENCIES.md` / `THIRD_PARTY_NOTICES`; report upstream issues to
  the respective projects.
