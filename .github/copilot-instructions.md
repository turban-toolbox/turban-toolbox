# Copilot Code Review Instructions

This is a scientific Python library for oceanographic turbulence data processing.

## Review focus

When reviewing a pull request, pay particular attention to:

### Tests (tests/)
- Every new public function or class in `turban/` should have a corresponding test in `tests/`.
- Tests should cover the main use case and at least one edge case (e.g. empty input, mismatched shapes, boundary values).
- New fixtures or shared data paths must be registered in `tests/filepaths.py`.
- Flag if only trivial "smoke tests" are present for non-trivial logic.

### Documentation (docs/)
- New public API (functions, classes, parameters) must be reflected in the relevant file under `docs/`.
- Docstrings should follow the NumPy docstring convention already used in the codebase.
- Flag if a new feature has no worked example or usage note in the docs.

### General
- Flag numerical/scientific code that lacks a reference or explanation of the algorithm used.
- Flag type annotations that are missing or inconsistent with existing code style.
