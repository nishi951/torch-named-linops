import logging

from textwrap import indent

__all__ = ["setup_console_logger", "Indenter", "INDENT"]


def setup_console_logger(
    logger,
    log_level: int,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handler_cls=logging.StreamHandler,
):
    logger.setLevel(log_level)
    formatter = logging.Formatter(fmt)
    for handler in logger.handlers:
        if isinstance(handler, handler_cls):
            handler.setFormatter(formatter)
            return logger
    handler = handler_cls()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Indenter:
    """Special context manager for indented formatting

    Example usage:

    with Indenter() as ident:
        ident.print("[DRY RUN]") # Not indented
        with ident:
            ident.print(pipeline.pprint()) # Indented one level
    """

    def __init__(self, start_level: int = -1):
        self.level = start_level

    def indent(self, text):
        for _ in range(self.level):
            text = indent(text, "    ")
        return text

    def print(self, text):
        print(self.indent(text))

    def __enter__(self):
        self.level += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.level -= 1


# Singleton for global indentation control
INDENT = Indenter(start_level=0)
