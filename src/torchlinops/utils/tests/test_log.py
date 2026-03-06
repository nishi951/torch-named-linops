import logging

from torchlinops.utils._log import INDENT, Indenter, setup_console_logger


class TestIndenter:
    def test_indenter_init(self):
        indenter = Indenter(start_level=0)
        assert indenter.level == 0

    def test_indenter_init_negative(self):
        indenter = Indenter(start_level=-1)
        assert indenter.level == -1

    def test_indenter_enter_exit(self):
        indenter = Indenter(start_level=0)
        assert indenter.level == 0
        with indenter:
            assert indenter.level == 1
        assert indenter.level == 0

    def test_indenter_nested(self):
        indenter = Indenter(start_level=0)
        with indenter:
            with indenter:
                assert indenter.level == 2
            assert indenter.level == 1
        assert indenter.level == 0

    def test_indent_single_line(self):
        indenter = Indenter(start_level=0)
        indenter.level = 1
        result = indenter.indent("hello")
        assert "    hello" in result

    def test_indent_multiple_lines(self):
        indenter = Indenter(start_level=0)
        indenter.level = 2
        result = indenter.indent("line1\nline2")
        assert "        line1" in result
        assert "        line2" in result

    def test_indent_zero_level(self):
        indenter = Indenter(start_level=0)
        indenter.level = 0
        result = indenter.indent("hello")
        assert result == "hello"


class TestSetupConsoleLogger:
    def test_setup_default(self):
        logger = logging.getLogger("test_default")
        logger.handlers = []
        result = setup_console_logger(logger)
        assert result is not None
        assert len(result.handlers) > 0

    def test_setup_custom_level(self):
        logger = logging.getLogger("test_level")
        logger.handlers = []
        result = setup_console_logger(logger, log_level=logging.DEBUG)
        assert result is not None

    def test_setup_custom_fmt(self):
        logger = logging.getLogger("test_fmt")
        logger.handlers = []
        fmt = "%(name)s - %(message)s"
        result = setup_console_logger(logger, fmt=fmt)
        assert result is not None

    def test_setup_existing_handler(self):
        logger = logging.getLogger("test_existing")
        logger.handlers = []
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        result = setup_console_logger(logger)
        assert result is not None
        assert len(logger.handlers) == 1

    def test_setup_no_handler(self):
        logger = logging.getLogger("test_no_handler")
        logger.handlers = []
        result = setup_console_logger(logger)
        assert len(logger.handlers) > 0


class TestIndentSingleton:
    def test_indent_singleton_exists(self):
        assert INDENT is not None

    def test_indent_singleton_type(self):
        assert isinstance(INDENT, Indenter)
