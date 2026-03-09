import logging

logger = logging.getLogger("turban")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "{asctime} | {levelname:<7s} | {name} | {funcName:<20s} | {filename:>10s}:{lineno:>4d} | {message}",
    "%Y-%m-%d %H:%M:%S",
    style="{",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
