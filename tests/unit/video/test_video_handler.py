"""Unit tests for the VideoHandler class.

This module contains tests for video input/output operations including
reading frames from a video source and writing frames to an output file.
"""

# Standard library imports
import os
import tempfile
from pathlib import Path
from typing import Generator, Tuple

# Third-party imports
import numpy as np
import pytest

# Local application imports
from video.video_handler import VideoHandler


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample frame for testing.

    Returns:
        np.ndarray: A small test frame (2x2 pixels, 3 color channels).
    """
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_video_handler_initialization() -> None:
    """Test that VideoHandler initializes with default parameters."""
    # Exercise
    handler = VideoHandler(0)  # Using 0 for default camera

    # Verify
    assert handler is not None
    assert handler.cap is not None
    assert handler.writer is None  # No output path provided

    # Cleanup
    handler.release()


def test_video_handler_read_frame_returns_frame(monkeypatch) -> None:
    """Test that read_frame returns a valid frame.

    Verifies that:
    - The function returns a tuple of (success, frame)
    - The success flag is True when a frame is read
  
    This test uses a mock video capture to simulate a video source.
    """
    # Create a mock video capture class
    class MockVideoCapture:
        def __init__(self, *args, **kwargs):
            self.is_opened_flag = True
            
        def isOpened(self) -> bool:
            return self.is_opened_flag
            
        def read(self):
            # Return a simple black frame
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
            
        def release(self):
            self.is_opened_flag = False
    
    # Patch cv2.VideoCapture to use our mock
    monkeypatch.setattr('cv2.VideoCapture', MockVideoCapture)
    
    # Setup
    handler = VideoHandler("test_source")  # Source can be anything since we're mocking
    
    try:
        # Exercise
        ret, frame = handler.read_frame()
        
        # Verify
        assert ret is True, "Frame should be read successfully"
        assert frame is not None, "Frame should not be None"
        assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
        assert len(frame.shape) == 3, "Frame should have 3 dimensions (H, W, C)"
        
    finally:
        # Cleanup
        handler.release()


@pytest.fixture
def temp_video_file() -> Generator[str, None, None]:
    """Create a temporary video file for testing and clean up afterward.

    Yields:
        str: Path to the temporary video file.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        yield temp_path
    finally:
        # Clean up: remove the test file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_video_handler_writer_initialization(temp_video_file: str) -> None:
    """Test that VideoHandler initializes the video writer correctly."""
    # Setup
    frame_width, frame_height, fps = 320, 240, 30

    # Exercise
    handler = VideoHandler(
        0, frame_width=frame_width, frame_height=frame_height, fps=fps, output_path=temp_video_file  # camera index
    )

    try:
        # Verify
        assert handler.writer is not None, "Video writer should be initialized"
        assert handler.output_path == temp_video_file

    finally:
        # Cleanup
        handler.release()


def test_write_frame_to_video(temp_video_file: str, sample_frame: np.ndarray) -> None:
    """Test writing frames to a video file.

    Verifies that:
    - Frames can be written to the output file
    - The output file is created and has content
    - The writer handles frame dimensions correctly
    """
    # Setup
    frame_width, frame_height = sample_frame.shape[1], sample_frame.shape[0]

    # Exercise
    handler = VideoHandler(
        0, frame_width=frame_width, frame_height=frame_height, fps=30, output_path=temp_video_file  # camera index
    )

    try:
        # Write multiple frames to ensure the writer works properly
        for _ in range(5):
            handler.write_frame(sample_frame)

        # Verify the writer was used
        assert handler.writer is not None

        # Release to ensure the file is written
        handler.release()

        # Check if the output file was created and has content
        assert os.path.exists(temp_video_file), f"Output file {temp_video_file} was not created"
        assert os.path.getsize(temp_video_file) > 0, "Output file is empty"

    finally:
        # Cleanup is handled by the fixture
        pass


def test_release_resources() -> None:
    """Test that resources are properly released."""
    # Setup
    handler = VideoHandler(0)

    # Exercise
    handler.release()

    # Verify
    assert handler.cap.isOpened() is False, "Video capture should be released"
    assert handler.writer is None, "Video writer should be released"


def test_failure_paths(monkeypatch, tmp_path: Path):
    """Cover VideoHandler.__init__ error log and read_frame warning."""

    # Dummy cap simulating failure -----------------------------------------------------------
    class _FailCap:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return False

        def read(self):
            return False, None

        def release(self):
            self.released = True

    monkeypatch.setattr("cv2.VideoCapture", lambda *_a, **_k: _FailCap())

    handler = VideoHandler(str(tmp_path / "nonexistent.mp4"))
    with pytest.raises(AssertionError):
        ret, frame = handler.read_frame()
    handler.release()


def test_is_opened_returns_bool(monkeypatch):
    """Test if VideoHandler is opened."""

    # Patch cv2.VideoCapture to simulate successful open
    class _DummyCap:
        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", lambda *_a, **_k: _DummyCap())

    handler = VideoHandler(0)
    assert handler.is_opened() is True


def test_show_and_wait_key_and_destroy(monkeypatch):
    """Cover show_frame, wait_key, destroy_all_windows (lines 100, 111, 118)."""
    calls = {}

    def _imshow(win, frame):
        calls["imshow"] = (win, frame.shape if hasattr(frame, "shape") else None)

    def _waitkey(delay=1):
        calls["waitkey"] = delay
        return 27

    def _destroy():
        calls["destroy"] = True

    monkeypatch.setattr("cv2.imshow", _imshow)
    monkeypatch.setattr("cv2.waitKey", _waitkey)
    monkeypatch.setattr("cv2.destroyAllWindows", _destroy)

    import numpy as np

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    handler = VideoHandler(0)

    handler.show_frame("win", frame)
    key = handler.wait_key(5)
    VideoHandler.destroy_all_windows()

    assert calls.get("imshow") is not None
    assert calls.get("waitkey") == 5
    assert calls.get("destroy") is True
    assert key == 27
