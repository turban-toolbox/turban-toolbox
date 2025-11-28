import logging

logger = logging.getLogger('turban')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('{asctime} | {levelname:<8s} | {name:<20s} | {message}', style='{')
handler.setFormatter(formatter)
logger.addHandler(handler)