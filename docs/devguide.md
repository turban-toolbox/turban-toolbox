# Developer's guide

## Type checking

TURBAN uses extensive type annotations. Types are currently (for easier development) runtime-checked with `beartype`. This can easily be disabled by commenting out the two lines in `turban/__init__.py`.

## PEP8

TURBAN uses black for the code style. Settings are given in `pyproject.toml` (and should be auto-discovered by black). Code editors like VSCode have extensions that can format files individually, another option is running black on the command line: `$ black .`.

## Tests

TURBAN uses `pytest` for unit testing:
```bash
python -m pytest --cov=turban --cov-report html
```
Settings are defined in `pyproject.toml`.

Markdown documents with python snippets can be tested as well, e.g.:
```bash
python -m pytest --markdown-docs --markdown-docs-syntax=superfences docs/
```

To generate a test coverage report:
```bash
python -m pytest --markdown-docs --markdown-docs-syntax=superfences 
```

## Code base overview

### Directory structure


TURBAN handles a variety of instruments and a variety of methods of analysing them.
- `turban/instruments/`: Instruments provide a way of getting raw data from an instrument up to level 1 (converted to physical units).
- `turban/process/`: Analysis methods provide a way of getting data from level 1 to level 4
- `turban/.../generic/`: Each of `instruments/` and `process/` has one folder `generic` for base and helper classes, in addition to one folder per instrument or process type.
- `turban/.../.../api.py`: Define high-level objects that handle data loading, writing, and processing between levels. These are the objects the end user preferably works with.
- `turban/.../.../config.py`: Define configuration objects that store parameters about processing pipelines or instruments such as sampling rate, high-pass filter cutoff frequencies, and the like.
- `tests/`: Defines tests. The directory structure mirrors that of `turban/`.
- `data/`: Provides some data, mostly for test routines. The directory structure mirrors that of `turban/`.
- `docs/` is where the documentation resides (and can be browsed from readthedocs).

### Variable names

Variables in TURBAN code (are supposed to) follow, wherever reasonable, the variable names outlined in `turban/variables.py`.

