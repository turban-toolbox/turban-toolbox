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

### Use in modules
Using a logger instance, dedicated to a specific module allows for
logger messages, that can be traced to a specific module. Typically
for each module a logger instance is created. For example
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
or, as a convenience function
```python
from turban import get_logger
logger = get_logger(__name__)
```
In all cases, a logger instance is returned with a default
configuration, which is accessible through the LoggerManager.

Typical use of a logger is to inject feedback information in the code,
for example
```python
logger.debug("This is a debug message.")
logger.info(f"{number_of_files} files have been read...")
logger.warning("This is a warning")
logger.error("Fatal!")
```
Which of these messages are displayed depends on the setting of the
log-level. This is best determined from the top script, see below.

By default, logger messages are sent to /dev/stderr, and are shown in
the terminal screen. Both the format and the destination of the
messages can be controlled using the LoggerConfig object. An instance
of the LoggerConfig can (optionally) be supplied to the LoggerManager.get\_logger()
method.

The example below demonstrates how a LoggerConfig object can be used
to log the messages to a file, rather than /dev/sdterr.

```python
from turban.utils.logging import LoggerManager, LoggerConfig

logger_manager = LoggerManager()

# define a custom handler
handler = logging.FileHandler("/tmp/mylogs.txt")
logger_config = LoggerConfig(handler=handler)

# create a logger with modified logging behaviour
logger = logger_manager.get_logger(__name__, logger_config)

:
:
logger.debug("This message goes to a file")
```

### Setting the log level

By default, there are four levels of logging:  debug, info, warning, error.
Best practice is to define the log level in the top script. This
avoids that log messages are processed, because the log level is set
on the module level, even if they are not desired. Also, you may not
want all modules to output logger messages at the same level.

The default logging level is set to "warning". This applies to all
loggers; the loggers defined in the TURBAN modules, as well as the
loggers defined in imported packages and modules. The log level for a
specific logger or group of loggers can be controlled by the
LoggerManager.set\_level() method. The method requires one parameter,
the log level, and an optional regular expression that is used to
match existing loggers. (The name or identifier of the logger is
determined on creation, and set by the first argument of
LoggerManager.get\_logger(), see the examples above.) Without a
regular expression, the logger level setting is applied to all
loggers.

If different logger levels are required, it is best practice is to set
the logger level in several steps, becoming more specific. In the
example below, we leave the logger level for all loggers set at
"warning" (default), but set the logger level for all TURBAN loggers
to "info", and for one specific module the logger level is set to
"debug".

```python
from turban.utils.logging import LoggerManager
logger_manager = LoggerManager()
# or
# from turban import logger_manager

# Set the level for all loggers to "error"
logger_manager.set_level("error")

# all TURBAN logger names start with "turban" to "warning
logger_manager.set_level("warning", "^turban")

# to select the loggers of a given package to "info"
logger_manager.set_level("warning","^turban\.instruments\.microrider")

# set the logger level for a specific logger/module to "debug"
logger_manager.set_level("warning","^turban.*rsIO")
# which matches the logger turban.instruments.microrider.rsIO
```


### LoggerManager


The class ```LoggerManager``` is implemented as a singleton, so that
any instance creation returns always the same object, irrespective if
an instance was created in some other module.

If you are unsure how a logger is called, you can get all loggers
managed by the ```logger_manager``` (all TURBAN loggers) by
```python
from turban.utils.logging import LoggerManager
logger_manager = LoggerManager()
loggers = logger_manager.list_loggers()
```
To get access to all loggers, including those of dependency packages,
you would use
```python
from turban.utils.logging import LoggerManager
logger_manager = LoggerManager()
all_loggers = logger_manager.list_all_loggers()
```

