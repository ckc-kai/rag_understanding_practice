import logging
from colorlog import ColoredFormatter




def setup_logger(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s] "
        "%(name)s: %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger