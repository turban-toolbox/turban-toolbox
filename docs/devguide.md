# Developer's guide

## Installation

Install the package in editable mode with the development dependency group:

```bash
python -m pip install -e . --group dev
```

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

## Logging

In order to facilitate the use of dedicated loggers with fine-grained
control over the formatting and logging level of various loggers used
in the code tree and in dependency modules, TURBAN provides a
`LoggerManager`.

For a specific module, a logger can be created by
```python
from turban.utils.logging import LoggerManager
logger = LoggerManager().get_logger(__name__)
```
or more explicit
```python
from turban.utils.logging import LoggerManager
logger_manager = LoggerManager()
logger = logger_manager.get_logger(__name__)
```

The class ```LoggerManager``` is implemented as a singleton, so that
any instance creation returns always the same object, irrespective if
an instance was created in some other module. The ```logger```
instance can then be used as usual
```python
logger.debug("This is a debug message")
```

In the application code it is now easy to set the logging level for
one or more loggers. 
```python
import turban.process.shear 
import turban.instruments.microrider.rsIO
import turban.utils.logging as turban_logging


logger_manager = turban_logging.LoggerManager()
logger_manager.set_level("info", "turban.*")
logger_manager.set_level('debug', 'rsIO')
```
This would set all turban loggers to the INFO level, apart from the
module ```rsIO```, which will be logging DEBUG messages as well.

If you are unsure how a logger is called, you can get all loggers
managed by the ```logger_manager``` (all TURBAN loggers) by
```python
loggers = logger_manager.list_loggers()
```
To get access to all loggers, including those of dependency packages,
you would use
```python
all_loggers = logger_manager.list_all_loggers()
```

