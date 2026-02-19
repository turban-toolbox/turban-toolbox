[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Turbulence Analysis Toolbox (Turban-Toolbox)

Welcome to TURBAN. You can find documentation on []

# Overview


Currently, only shear processing is fully functional.

## Type checking

TURBAN uses extensive type annotations. Types are currently (for easier development) runtime-checked with `beartype`. This can easily be disabled by commenting out the two lines in `turban/__init__.py`.

## PEP8

TURBAN uses black for the code style. Settings are given in `pyproject.toml` (and should be auto-discovered by black). Code editors like VSCode have extensions that can format files individually, another option is running black on the command line: `$ black .`.

## Tests

TURBAN uses `pytest` for unit testing:
```bash
python -m pytest
```
Settings are defined in `pyproject.toml`.

Markdown documents with python snippets can be tested as well, e.g.:
```bash
python -m pytest --markdown-docs README.md
```

To generate a test coverage report:
```bash
python -m pytest --markdown-docs README.md --cov=turban --cov-report html
```
