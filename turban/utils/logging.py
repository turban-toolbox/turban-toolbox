import logging

HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(
    "{asctime} | {levelname:<7s} | {name} | {funcName:<20s} | {filename:>10s}:{lineno:>4d} | {message}",
    "%Y-%m-%d %H:%M:%S",
    style="{",
)
HANDLER.setFormatter(FORMATTER)

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
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(HANDLER)
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
