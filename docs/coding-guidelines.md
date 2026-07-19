# Coding guidelines

Binding for all code in `src/`, `tests/`, and `experiments/`.

## Types are documentation

- Strict typing is the primary documentation. A precise signature replaces a
  docstring that restates it.
- Never write a comment or docstring that repeats what the name and type
  already say.
- Prefer narrowing types (`Literal`, discriminated unions, `NDArray[np.float64]`,
  jaxtyping-style shape conventions where adopted) over prose describing valid
  values.

## Comments

- Comments are for the user-facing API surface and for *why*, never *what*.
- Public entry points (things an experiment or another module imports) may
  carry a short docstring: one line, plus args only when a contract is not
  expressible in the type (units, coordinate frames, side effects).
- Internal helpers get no docstrings. If an internal needs explaining, rename
  or restructure it first.
- Inline comments only where the code is surprising (workarounds, API churn,
  numerical subtleties) — and they state the reason, ideally with a reference.

## Abstractions

- Clear, small, single-purpose modules; one concept per file.
- Depend on interfaces (`Protocol`) at boundaries; concrete classes inside.
- No speculative generality: build the abstraction when the second consumer
  exists, not before.
- No dummy/placeholder configs, no hardcoded paths, no dead code.

## Tests

- Test names state the behavior; no docstrings in tests.
- Assert real behavior, not "runs without raising".
