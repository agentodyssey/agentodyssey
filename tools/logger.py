import logging
from termcolor import colored


class ColoredFormatter(logging.Formatter):

    COLORS = {
        'DEBUG': ('cyan', None, []),
        'INFO': ('white', None, []),
        'WARNING': ('yellow', None, ['bold']),
        'EXCEPTION': ('red', None, ['bold']),
        'ERROR': ('red', None, ['bold']),
        'CRITICAL': ('magenta', None, ['bold']),
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, ('white', None, []))
        base_msg = super().format(record)
        return colored(base_msg, *color)


def get_logger(name="EnvLogger", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(filename)s:%(funcName)s:%(lineno)d)"
    handler.setFormatter(ColoredFormatter(fmt, "%H:%M:%S"))
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = get_logger("EnvLogger")
    # logger = get_logger("ProvLogger")

    logger.debug("Current env step: 42")
    logger.info("Model loaded successfully")
    logger.warning("vllm server is not online")
    logger.error("Model response is empty")
    logger.critical("Can't load the model")