"""Unit tests for the logger configuration module.

This module contains tests for the LoggerConfigurator class which handles
logging configuration for the application.
"""

# Standard library imports
import logging
import logging.handlers
import os
import sys
from io import StringIO
from typing import Generator, Optional
from unittest.mock import MagicMock, patch

# Third-party imports
import pytest

# Local application imports
from utils.logger import LoggerConfigurator


@pytest.fixture
def capture_logs() -> Generator[StringIO, None, None]:
    """Capture log output during tests.

    Yields:
        StringIO: A StringIO object containing the captured log output.
    """
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)

    # Get the root logger and add our handler
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    root_logger.handlers = [handler]

    try:
        yield log_stream
    finally:
        # Restore original handlers
        root_logger.handlers = original_handlers
        handler.close()


def test_initialization_with_default_level() -> None:
    """Test that LoggerConfigurator initializes with default log level."""
    # Exercise
    configurator = LoggerConfigurator()

    # Verify
    assert hasattr(configurator, "log_level")
    assert isinstance(configurator.log_level, str)


def test_initialization_with_custom_level() -> None:
    """Test that LoggerConfigurator initializes with custom log level."""
    # Exercise
    configurator = LoggerConfigurator(log_level="DEBUG")

    # Verify
    assert configurator.log_level == "DEBUG"


@patch("logging.basicConfig")
def test_setup_logging_with_valid_level(mock_basic_config: MagicMock, capture_logs: StringIO) -> None:
    """Test that setup_logging works with a valid log level."""
    # Setup
    configurator = LoggerConfigurator(log_level="INFO")

    # Exercise
    configurator.setup_logging()

    # Verify
    mock_basic_config.assert_called_once()
    call_args = mock_basic_config.call_args[1]
    assert call_args["level"] == logging.INFO
    assert "%(asctime)s [%(levelname)s] %(name)s: %(message)s" in call_args["format"]
    assert "%Y-%m-%d %H:%M:%S" in call_args["datefmt"]
    assert call_args["stream"] == sys.stdout


def test_setup_logging_with_invalid_level() -> None:
    """Test that setup_logging raises ValueError for invalid log level."""
    # Setup
    configurator = LoggerConfigurator(log_level="INVALID_LEVEL")

    # Exercise & Verify
    with pytest.raises(ValueError, match="Invalid log level"):
        configurator.setup_logging()


@patch("logging.basicConfig")
def test_setup_logging_calls_basic_config(mock_basic_config: MagicMock, capture_logs: StringIO) -> None:
    """Test that setup_logging calls logging.basicConfig with correct params."""
    # Setup
    configurator = LoggerConfigurator(log_level="DEBUG")

    # Exercise
    configurator.setup_logging()

    # Verify
    mock_basic_config.assert_called_once()
    call_args = mock_basic_config.call_args[1]
    assert call_args["level"] == logging.DEBUG
    assert "%(asctime)s [%(levelname)s] %(name)s: %(message)s" in call_args["format"]
    assert "%Y-%m-%d %H:%M:%S" in call_args["datefmt"]
    assert call_args["stream"] == sys.stdout


@patch("logging.basicConfig")
def test_log_output_format(mock_basic_config: MagicMock, capture_logs: StringIO) -> None:
    """Test that setup_logging sets the correct format params for INFO level."""
    # Setup
    configurator = LoggerConfigurator(log_level="INFO")

    # Exercise
    configurator.setup_logging()

    # Verify
    mock_basic_config.assert_called_once()
    call_args = mock_basic_config.call_args[1]
    assert "%(asctime)s [%(levelname)s] %(name)s: %(message)s" in call_args["format"]
    assert "%Y-%m-%d %H:%M:%S" in call_args["datefmt"]
    assert call_args["level"] == logging.INFO
    assert call_args["stream"] == sys.stdout


@patch("logging.basicConfig")
def test_log_level_filtering(mock_basic_config: MagicMock, capture_logs: StringIO) -> None:
    """Test that setup_logging sets the correct params for WARNING level."""
    # Setup
    configurator = LoggerConfigurator(log_level="WARNING")

    # Exercise
    configurator.setup_logging()

    # Verify
    mock_basic_config.assert_called_once()
    call_args = mock_basic_config.call_args[1]
    assert call_args["level"] == logging.WARNING
    assert "%(asctime)s [%(levelname)s] %(name)s: %(message)s" in call_args["format"]
    assert "%Y-%m-%d %H:%M:%S" in call_args["datefmt"]
    assert call_args["stream"] == sys.stdout
