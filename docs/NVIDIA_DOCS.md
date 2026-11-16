# NVIDIA CUDA documentation — external reference (MCP)

The NVIDIA documentation that used to be mirrored in this directory
(`NVIDIA_DOCS_NVFP4_MEMORY.md`, `NVIDIA_DOCS_HOPPER.md`,
`NVIDIA_DOCS_AMPERE.md`, `NVIDIA_DOCS_RUBIN.md`) was exported from the
`nvidia-cuda-docs` MCP server and has been **removed from the repository** to
avoid redistributing NVIDIA copyrighted material.

The actual, up-to-date documentation is **not vendored**. Before changing any
NVFP4/CUDA kernel, load the current NVIDIA documentation from the
`nvidia-cuda-docs` MCP server:

- Query the MCP for the specific topic: PTX ISA, CUDA C++ Programming Guide,
  `cuda_fp4`/`cuda_fp8` header semantics, CUTLASS notes, TMA/bulk copies,
  tcgen05 block-scaled MMA, stream-ordered memory, CC 12.x limits.
- If the MCP is not available on the machine, **stop and obtain access to the
  current documentation before proceeding** — do not rely on stale excerpts.
- When the MCP yields facts the code depends on, pin the derived invariant
  in-code (comment/regression test) in the same commit, because no offline
  mirror exists.

Source of the removed export: `mcp: nvidia-cuda-docs`.

## Offline copy kept outside the repository

The operator keeps the same export **outside** the tree, in a private
untracked directory (`NVIDIA_DOCS_NVFP4_MEMORY.md`,
`NVIDIA_DOCS_HOPPER.md`, `NVIDIA_DOCS_AMPERE.md`, `NVIDIA_DOCS_RUBIN.md`, plus
copies of `numeric-contract.md` / `numeric-conformance-audit.md`).

This section is a **pointer only**: no NVIDIA text is reproduced here, the
dumps stay untracked and outside `logestix/`, and the names above are the same
filenames the removal notice already lists — they locate the material, they do
not redistribute it. Anything derived from those files that a kernel depends on
gets re-derived in first-party words and pinned in-code/in a regression test,
which is where the obligation actually lands (see the MCP bullet above).

Read the offline copy when the MCP is unavailable, but treat it as the
**2026-09/10 snapshot** it is: its CCCL entry is 3.4.2 and its numbers were
verified on that date, so it is a cross-check, never a substitute for a current
MCP query.
