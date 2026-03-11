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

class Loggers():
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
            logger = logging.getLogger(name)
            config = config or LoggerConfig()
            logger.addHandler(config.handler)
            self._loggers[name] = logger
        else:
            if not config is None:
                logger.info(f"Supplied configuration is not applied to an already configured logger.")
            # makes any new logger's level set to any requirements set already.
            for _log_level in self._log_levels:
                self._set_log_level(*_log_level, name, logger)
        return logger

    def _set_log_level(self,
                       regex: re.Pattern|None,
                       level: int,
                       logger_name: str,
                       logger: logging.Logger) -> None:
            if regex:
                if regex.match(logger_name):
                    logger.setLevel(level)
            else:
                logger.setLevel(level)

    def set_log_level(self,
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
            self._set_log_level(regex, _level, k, v)
        self._log_levels.append((regex, _level))
        

    def list_loggers(self):
        return [k for k, v in self._loggers.items()]


    def list_all_loggers(self):
        pass
            
                
        
        
# logger = logging.getLogger("turban")
# logger.setLevel(logging.INFO)

# handler = logging.StreamHandler()
# formatter = logging.Formatter(
#     "{asctime} | {levelname:<7s} | {name} | {funcName:<20s} | {filename:>10s}:{lineno:>4d} | {message}",
#     "%Y-%m-%d %H:%M:%S",
#     style="{",
# )
# handler.setFormatter(formatter)
# logger.addHandler(handler)


