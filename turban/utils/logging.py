import logging
import re
from typing import Self


class LoggerConfig:
    """Configuration class for loggers

    Parameters
    ----------
    handler : logging.Handler | None
        A handler to process debug messages. If None, a default handler is used.
    formatter : logging.Formatter | None
        A formatter to format debug messages. If None, a default formatter is used.
    """

    def __init__(
        self,
        handler: logging.Handler | None = None,
        formatter: logging.Formatter | None = None,
    ):
        self._handler = handler or logging.StreamHandler()
        self._formatter = formatter or logging.Formatter(
            "{asctime} | {levelname:<7s} | {name} | {funcName:<20s} | {filename:>10s}:{lineno:>4d} | {message}",
            "%Y-%m-%d %H:%M:%S",
            style="{",
        )

    def set_formatter(self, formatter: logging.Formatter) -> None:
        """Sets the formatter for debugging messages

        Parameters
        ----------
        formatter: logging.Formatter
            A formatter object defining how to format debug messages
        """
        self._formatter = formatter

    def set_handler(self, handler: logging.Handler) -> None:
        """Sets the handler

        Parameters
        ----------
        handler: logging.Handler
            A handler object defining how to output debug messages
        """
        self._handler = handler

    @property
    def handler(self) -> logging.Handler:
        """Returns the formatted handler

        Returns
        -------
        logging.Handler
            formatted handler object.
        """
        self._handler.setFormatter(self._formatter)
        return self._handler


class LoggerManager:
    """Logger manager

    A class to manage various loggers created by the application, as
    well as in imported modules.  The class is implemented as a
    singleton, which means that multiple instantiations return in fact
    the same object.

    For example:

    >>> lm = LoggerManager()
    >>> LM = LoggerManager()
    >>> lm is LM # -> True

    The class LoggerManager can be called with a debug level (as a
    string), which sets the default level for all loggers. The
    behaviour of specific loggers can be modified by the method
    set_level(..., filter_pattern), using the filter_pattern (regular
    expression).

    Parameters
    ----------
    default_level: str | None
        default debug level. This will override any default level set
        previously. If not specified, the singleton is return without
        modifying the default level.

    """

    _instance: Self | None = None
    _loggers: dict[str, logging.Logger] = {}
    _levels: dict[str, int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    _log_levels: list[tuple[re.Pattern[str] | None, int]] = []

    def __new__(cls, default_level: str | None = None) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        if not default_level is None:
            cls._instance.set_level(default_level)
        return cls._instance

    def get_logger(
        self, name: str, config: LoggerConfig | None = None
    ) -> logging.Logger:
        """Returns a logger created previously with given name, or,
        creates a new logger, if no such logger exists.

        Parameters
        ----------
        name : str
            name of logger (for example __name__)
        config: LoggerConfig | None
            Configuration object that specifies the handlers and
            formatters. If None, a default LoggerConfig is used.

        Returns
        -------
        logging.Logger
            A logger object.
        """
        try:
            logger = self._loggers[name]
        except KeyError:
            logger = logging.getLogger(name)
            config = config or LoggerConfig()
            logger.addHandler(config.handler)
            self._loggers[name] = logger
        else:
            if not config is None:
                logger.info(
                    f"Supplied configuration is not applied to an already configured logger."
                )
        # makes any new logger's level set to any requirements set already.
        for _log_level in self._log_levels:
            self._set_level(*_log_level, name, logger)
        return logger

    def _set_level(
        self,
        regex: re.Pattern[str] | None,
        level: int,
        logger_name: str,
        logger: logging.Logger,
    ) -> None:
        if regex:
            if regex.search(logger_name):
                logger.setLevel(level)
        else:
            logger.setLevel(level)

    def set_level(self, level: int | str, filter_pattern: str | None = None) -> None:
        """Set the level for all or some loggers

        Parameters
        ----------
        level: int | str
            debug level (for example logging.DEBUG or "debug")
        filter_pattern : str | None

            A regular expression to limit the setting to matching
            loggers. If None, all loggers' levels are set.
        """
        if isinstance(level, str):
            _level = self._levels[level.lower()]
        else:
            _level = level

        if not filter_pattern is None:
            regex = re.compile(filter_pattern)
        else:
            regex = None
        for k, v in self._loggers.items():
            self._set_level(regex, _level, k, v)
        self._log_levels.append((regex, _level))

    def list_loggers(self) -> list[str]:
        """Lists all loggers created under the auspicien of the LoggerManager

        Returns
        -------
        list[str]
            list of names of all loggers, under management of the LoggerManager

        See also list_all_loggers() for listing all loggers known in this session.
        """
        return [k for k, v in self._loggers.items()]

    def list_all_loggers(self) -> list[str]:
        """Lists all loggers created in this session

        Returns
        -------
        list[str]
            list of names of all loggers known in this session

        See also list_loggers() for listing all loggers managed by the LoggerManager.
        """
        loggers = [
            name
            for name, logger in logging.Logger.manager.loggerDict.items()
            if isinstance(logger, logging.Logger)
        ]
        return loggers


def get_logger(name: str) -> logging.Logger:
    """Get or create a named logger with the configured stream handler.

    If the logger has no handlers, the pre-configured handler with custom
    formatter is attached automatically.

    Parameters
    ----------
    name : str
        The name of the logger to retrieve or create.

    Returns
    -------
    logging.Logger
        The logger instance with the configured handler attached if needed.

    """
    manager = LoggerManager()
    logger = manager.get_logger(name)
    return logger


def set_turban_loglevel(level: int | str, pattern: str = "turban") -> None:
    """Set the log level for all loggers matching a name pattern.

    Parameters
    ----------
    level : int or str
        The log level to set. Can be a numeric level (e.g. logging.INFO or 20)
        or a string (e.g. "INFO").
    pattern : str, optional
        Logger name prefix to match. Only loggers whose names start with this
        pattern will be affected. Default is "turban".

    """
    manager = LoggerManager()
    regex_pattern = rf"^{pattern}.*"
    manager.set_level(level, regex_pattern)
