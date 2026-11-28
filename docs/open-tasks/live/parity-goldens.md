# Parity goldens — declared debt and how a golden is captured

### K. Golden-debt: the declared list, and why it is not a hard failure (2026-10-06)

Vector 2 of the three "will it recur" holes is now closed. `test_model_suite_guard`
carries a new case that pairs every `Suites()` row against `tests/fixtures/{fsi,code,det}`
and is machine-enforced in **both** directions:

- a parity golden that is missing **and not declared** in `kGoldenDebt` fails;
- a `kGoldenDebt` row whose golden now exists fails too, so the list cannot outlive
  the problem and rot;
- a `kGoldenDebt` row matching no live gap fails, so renames do not leave dead entries.

Current state: **39 declared debt entries** across thirteen model rows that have
none (or only some) of the three goldens. That number is the honest coverage figure, and
until now it was only obtainable by diffing two directories by hand.

The design choice worth recording: this is **declared debt, not a hard requirement**.
A hard requirement would paint nine families red on the next fast run and demand
capturing goldens for every one — including models with no local safetensors, which
is hours of work plus vault access we do not have here. Declared debt gets the
property that actually matters (a new family cannot be added quietly without
goldens, and the remaining gap is enumerable from test output) without pretending the
existing debt is smaller than it is. A missing golden stays a SKIP; what changes is
that it can no longer be invisible.

The last single-stage entry here, `granite-4.2-30b:code`, closed 2026-11-03
(its golden had existed since 2026-10-07; archived with the inference defect
found alongside it). The mechanism remains: a family can look fully wired across
all four parity stages and still have a hole in one of them.

§L (the MCP rollout) is closed — see
[`../2026-11-28-mcp-gate-rollout-closed.md`](../2026-11-28-mcp-gate-rollout-closed.md).
The `mcp` stage is now a required lifecycle stage with 39 goldens and 9 declared
debt rows, so the mechanism described above applies to it unchanged.
