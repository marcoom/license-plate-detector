"""Video handling and processing for the license plate detection system.

This package provides functionality for reading from video sources (files or webcams)
and writing processed video output, including handling different video formats and
frame processing.

"""

from .video_handler import VideoHandler

__all__ = ["VideoHandler"]
