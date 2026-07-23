# Coding guidelines

Binding for all code in `src/`, `tests/`, and `experiments/`.

## Types are documentation

- Strict typing is the primary documentation. A precise signature replaces a
  docstring that restates it.
- Never write a comment or docstring that repeats what the name and type
  already say.
- Prefer narrowing types (`Literal`, discriminated unions, …) over prose
  describing valid values.

## Comments

- Comments are for the user-facing API surface and for *why*, never *what*.
- Public entry points (things an experiment or another module imports) may
  carry a short docstring: one line, plus args only when a contract is not
  expressible in the type (units, coordinate frames, side effects).
- Internal helpers get no docstrings. If an internal needs explaining, rename
  or restructure it first.
- Inline comments only where the code is surprising (workarounds, API churn,
  numerical subtleties) — and they state the reason, ideally with a reference.

## Docstrings describe purpose, not history

- A docstring states what the abstraction is *for*, at the level of the
  abstraction itself — the role it plays for its caller, not how it is built.
- Never cite design history: no decision numbers, plan/spec section references,
  upstream-project comparisons, or "this replaces X". Git history is the
  record; a docstring reader is a user, not a reviewer.
- Implementation specifics (algorithms, masking tricks, reference-parity
  subtleties) belong as inline comments next to the code they explain — and
  only where that code is surprising — never in the docstring.

## Naming

- A name must read correctly at the call site, without the surrounding
  context that named it (`step_one_farm`, not `one`).
- Spell out domain words (`layout`, `state`, `action` — not `lay`, `st`,
  `act`). Single letters only for tight math indices that mirror a formula.
- If an internal needs a comment to explain its role, rename it instead.

## Abstractions

- Clear, small, single-purpose modules; one concept per file.
- Depend on interfaces (`Protocol`) at boundaries; concrete classes inside.
- No speculative generality: build the abstraction when the second consumer
  exists, not before.
- No dummy/placeholder configs, no hardcoded paths, no dead code.

## Tests

- Test names state the behavior; no docstrings in tests.
- Assert real behavior, not "runs without raising".
- Few, sharp tests. No trivial tests (constructor-runs, is-callable,
  shape-only when a value assertion is available).
- Prefer invariant tests (equivariance, feasibility, determinism,
  round-trip equality) over smoke tests. At most one end-to-end smoke
  test per subsystem; everything else exercises one behavior.
