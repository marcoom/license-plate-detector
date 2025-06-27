"""
Logger configuration module for the project.

Provides a LoggerConfigurator class to set up logging based on the config-defined log level.
"""
from typing import Optional
import logging
import sys
from src.config import LOG_LEVEL

class LoggerConfigurator:
    """
    Configures and provides the project-wide logger.
    """
    def __init__(self, log_level: Optional[str] = None) -> None:
        """
        Initialize the LoggerConfigurator.
        Args:
            log_level (Optional[str]): Log level as a string (e.g., 'INFO', 'DEBUG').
        """
        self.log_level = log_level or LOG_LEVEL

    def setup_logging(self) -> None:
        """
        Set up logging configuration for the project.
        """
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.log_level}")
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout
        )
        logging.getLogger().info(f"Logger initialized with level: {self.log_level.upper()}")
