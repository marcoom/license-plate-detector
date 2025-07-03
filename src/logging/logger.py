# This file is part of License Plate Detection System with YOLO and OCR.
# Copyright (C) 2025 Marco Mongi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Logger configuration module for the project.

Provides a LoggerConfigurator class to set up logging based on the config-defined log level.
"""
from typing import Optional
import logging
import sys
from config import LOG_LEVEL

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
