import logging
import re
from typing import Self


class LoggerConfig():

    def __init__(self,
                 handler:logging.Handler|None = None,
                 formatter: logging.Formatter|None = None):
        self._handler = handler or logging.StreamHandler()
        self._formatter = formatter or logging.Formatter(
            "{asctime} | {levelname:<7s} | {name} | {funcName:<20s} | {filename:>10s}:{lineno:>4d} | {message}",
            "%Y-%m-%d %H:%M:%S",
            style="{",
        )
        
    def set_formatter(self, formatter: logging.Formatter) -> None:
        self._formatter = formatter

    def set_handler(self, handler: logging.Handler) -> None:
        self._handler = handler

    @property
    def handler(self) -> logging.Handler:
        self._handler.setFormatter(self._formatter)
        return self._handler

class LoggerManager():
    _instance: Self | None = None
    _loggers: dict[str, logging.Logger] = {}
    _levels:dict[str, int] = {"debug":logging.DEBUG,
                              "info": logging.INFO,
                              "warning": logging.WARNING,
                              "error": logging.ERROR}
    _log_levels:list[tuple[re.Pattern|None, int]] = []
    
    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_logger(self, name, config: LoggerConfig | None=None) -> logging.Logger:
        try:
            logger = self._loggers[name]
        except KeyError:
            print("configuring logger")
            logger = logging.getLogger(name)
            config = config or LoggerConfig()
            logger.addHandler(config.handler)
            self._loggers[name] = logger
        else:
            if not config is None:
                logger.info(f"Supplied configuration is not applied to an already configured logger.")
        # makes any new logger's level set to any requirements set already.
        for _log_level in self._log_levels:
            print("Setting log level")
            self._set_level(*_log_level, name, logger)
        return logger

    def _set_level(self,
                       regex: re.Pattern|None,
                       level: int,
                       logger_name: str,
                       logger: logging.Logger) -> None:
            if regex:
                if regex.match(logger_name):
                    logger.setLevel(level)
            else:
                logger.setLevel(level)

    def set_level(self,
                      level: int | str,
                      filter_pattern: str | None = None) -> None:
        if isinstance(level, str):
            _level = self._levels[level]
        else:
            _level = level
        
        if not filter_pattern is None:
            regex = re.complile(filter_pattern)
        else:
            regex = None
        for k, v in self._loggers.items():
            self._set_level(regex, _level, k, v)
        self._log_levels.append((regex, _level))
        

    def list_loggers(self):
        return [k for k, v in self._loggers.items()]


    def list_all_loggers(self):
        pass




def get_logger(name: str):
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
    all_loggers = logging.Logger.manager.loggerDict
    for name in all_loggers:
        if name.startswith(pattern):
            logger = logging.getLogger(name)
            logger.setLevel(level)
