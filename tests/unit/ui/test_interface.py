"""Unit tests for the video interface module.

This module contains tests for the VideoInterface class, which handles
video input/output operations and processing.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Optional, Tuple

# Third-party imports
import cv2
import pytest

# Local application imports
from ui.interface import VideoInterface


def test_setup_video_writer(tmp_path: Path) -> None:
    """Test that the video writer is properly configured for output.

    Verifies that:
    - The output path is correctly generated from the input path
    - The writer object is created successfully
    - The output path follows the expected naming convention

    Args:
        tmp_path: Pytest fixture providing a temporary directory
    """
    # Setup
    input_video = tmp_path / "test.mp4"
    input_video.touch()  # Create empty test file
    width, height, fps = 1280, 720, 30.0

    # Exercise
    writer, output_path = VideoInterface.setup_video_writer(str(input_video), width, height, fps)

    # Verify
    assert writer is not None, "Video writer should be created"
    assert output_path == str(tmp_path / "test_processed.mp4"), "Output path should follow the expected naming convention"

    # Test with a different filename pattern
    input_video2 = tmp_path / "sample_video.avi"
    input_video2.touch()
    writer2, output_path2 = VideoInterface.setup_video_writer(str(input_video2), width, height, fps)
    assert output_path2 == str(tmp_path / "sample_video_processed.avi"), "Should handle different file extensions correctly"


def test_setup_video_writer_with_invalid_path(tmp_path: Path) -> None:
    """Test that setup_video_writer handles invalid input paths gracefully.

    Verifies that:
    - Non-existent input paths raise an appropriate exception
    - Invalid file extensions are handled properly
    """
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        VideoInterface.setup_video_writer(str(tmp_path / "nonexistent.mp4"), 640, 480, 30.0)

    # Test with invalid file extension
    invalid_file = tmp_path / "test.txt"
    invalid_file.touch()
    with pytest.raises(ValueError, match="Unsupported file format"):
        VideoInterface.setup_video_writer(str(invalid_file), 640, 480, 30.0)
