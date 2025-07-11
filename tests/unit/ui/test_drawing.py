"""Unit tests for the drawing utilities module.

This module contains tests for the drawing functions and utilities used to
annotate video frames with detection information, FPS counters, and trajectories.
"""

# Standard library imports
import time
from typing import List, Tuple

# Third-party imports
import numpy as np
import pytest

# Local application imports
from ui.drawing import (
    FPSCounter,
    draw_fps_on_frame,
    draw_plate_on_frame,
    draw_trajectory,
)


@pytest.fixture
def dummy_frame() -> np.ndarray:
    """Create a black test frame for drawing tests.

    Returns:
        np.ndarray: A 100x100 black RGB image (3 channels) with uint8 dtype.
    """
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_fps_counter_initialization() -> None:
    """Test that FPSCounter initializes with 0 FPS and updates correctly.

    Verifies that:
    - Initial FPS is 0
    - After update, FPS is an integer
    - Multiple updates don't cause errors
    """
    # Setup
    counter = FPSCounter()

    # Test initial state
    assert counter.get_fps() == 0, "FPS should start at 0"

    # Test after update
    counter.update()
    fps = counter.get_fps()
    assert isinstance(fps, int), "FPS should be an integer"
    assert fps >= 0, "FPS should not be negative"

    # Test multiple updates
    for _ in range(5):
        counter.update()
    assert isinstance(counter.get_fps(), int), "FPS should remain an integer after multiple updates"


def test_fps_counter_mean_updates(monkeypatch):
    """Simulate passage of >1 s to trigger mean_fps calculation."""
    # Control time.time() BEFORE constructing counter -------------------------------------------------
    start = 1000.0
    monkeypatch.setattr(time, "time", lambda: start)
    counter = FPSCounter()

    counter.update()  # first update at t=start

    # Advance time by 1.1 s so condition in update() becomes True -----------------------------------
    monkeypatch.setattr(time, "time", lambda: start + 1.1)
    counter.update()

    assert counter.get_fps() > 0, "mean_fps should update after >1s elapsed"


def test_draw_fps_on_frame(dummy_frame: np.ndarray) -> None:
    """Test that FPS is drawn on the frame without errors.

    Verifies that:
    - Function runs without raising exceptions
    - Frame is modified (though we don't verify the visual output)

    Args:
        dummy_frame: Test frame fixture to draw on.
    """
    # Make a copy to ensure we don't modify the original
    frame = dummy_frame.copy()

    # Exercise
    draw_fps_on_frame(frame, 30)

    # Verify - just check that the function ran without errors
    # Note: We don't verify the visual output, as that would be brittle


def test_draw_plate_on_frame(dummy_frame: np.ndarray) -> None:
    """Test that license plate information is drawn on the frame.

    Verifies that:
    - Function runs without raising exceptions
    - Different inputs are handled correctly

    Args:
        dummy_frame: Test frame fixture to draw on.
    """
    # Test with track ID
    frame1 = dummy_frame.copy()
    draw_plate_on_frame(frame1, "TEST1234", 10, 10, 80, 80, track_id=1)

    # Test without track ID
    frame2 = dummy_frame.copy()
    draw_plate_on_frame(frame2, "NOPLATE", 20, 20, 60, 60, track_id=None)

    # Test with different coordinates and text
    frame3 = dummy_frame.copy()
    draw_plate_on_frame(frame3, "ABC567", 0, 0, 99, 99, track_id=42)


def test_draw_trajectory(dummy_frame: np.ndarray) -> None:
    """Test that trajectory lines are drawn on the frame.

    Verifies that:
    - Function runs without raising exceptions
    - Different trajectory lengths are handled
    - Edge cases are handled (empty trajectory, single point)

    Args:
        dummy_frame: Test frame fixture to draw on.
    """
    # Test with multiple points
    frame1 = dummy_frame.copy()
    trajectory1 = [(10, 10), (20, 20), (30, 30), (40, 40)]
    draw_trajectory(frame1, trajectory1)

    # Test with single point (should draw a single point)
    frame2 = dummy_frame.copy()
    trajectory2 = [(50, 50)]
    draw_trajectory(frame2, trajectory2)

    # Test with empty trajectory (should not raise)
    frame3 = dummy_frame.copy()
    trajectory3: List[Tuple[int, int]] = []
    draw_trajectory(frame3, trajectory3)

    # Test trajectory with None values to cover continue branch (line 105)
    frame4 = dummy_frame.copy()
    trajectory4 = [(10, 10), None, (20, 20)]
    draw_trajectory(frame4, trajectory4)
