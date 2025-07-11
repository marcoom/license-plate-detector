"""Unit tests for the tracking module.

This module contains tests for the Tracker class, which is responsible
for tracking objects across video frames and maintaining their state.
"""

# Standard library imports
from types import SimpleNamespace
from typing import Any, List, Tuple

# Third-party imports
import numpy as np
import pytest

# Local application imports
from tracking.tracker import Tracker


class _StubTrack:
    """Test double for a track object with minimal implementation for testing."""

    def __init__(self, track_id: int, bbox: List[float]) -> None:
        """Initialize a stub track with the given ID and bounding box.

        Args:
            track_id: Unique identifier for the track.
            bbox: Bounding box coordinates in [x1, y1, x2, y2] format.
        """
        self.track_id = track_id
        self._bbox = bbox  # ltrb format: [x1, y1, x2, y2]
        self.time_since_update = 0

    def to_ltrb(self) -> List[float]:
        """Return the bounding box in [x1, y1, x2, y2] format.

        Returns:
            List[float]: The bounding box coordinates.
        """
        return self._bbox

    def is_confirmed(self) -> bool:
        """Return whether this track is confirmed.

        Returns:
            bool: Always returns True for testing purposes.
        """
        return True


class _StubOCRReader:
    """Test double for OCRReader with predictable output."""

    def readtext(self, _img: np.ndarray) -> List[Tuple[List[int], str, float]]:
        """Return a fixed OCR result for testing.

        Args:
            _img: Input image (ignored in this stub).

        Returns:
            List containing a single OCR result with dummy values.
        """
        return [([0, 0, 0, 0], "XYZ987", 0.9)]


def _dummy_frame() -> np.ndarray:
    """Create a dummy black frame for testing.

    Returns:
        np.ndarray: A 200x200 black image with 3 color channels.
    """
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_process_single_track_updates_history() -> None:
    """Test that process_single_track updates the OCR history correctly.

    Verifies that:
    - Processing a track adds its ID to the OCR history
    - The OCR result is stored in the history for the correct track ID
    """
    # Setup
    tracker = Tracker()
    frame = _dummy_frame()
    track = _StubTrack(track_id=5, bbox=[10, 20, 50, 60])
    ocr_reader = _StubOCRReader()

    # Exercise
    tracker.process_single_track(track, frame, ocr_reader)

    # Verify
    assert track.track_id in tracker.ocr_history, f"Track ID {track.track_id} should be in OCR history"
    assert "XYZ987" in tracker.ocr_history[track.track_id], "OCR result should be stored in history"


def test_update_trajectory_appends_points() -> None:
    """Test that update_trajectory correctly appends trajectory points.

    Verifies that:
    - Trajectory points are added to the correct track's trajectory
    - Multiple updates for the same track extend its trajectory
    - The trajectory contains the expected number of points
    """
    # Setup
    tracker = Tracker()
    frame = _dummy_frame()
    track_id = 3
    track = _StubTrack(track_id=track_id, bbox=[0, 0, 100, 100])

    # Exercise - add two trajectory updates
    tracker.update_trajectory(track, frame)

    # Update track position and add another point
    track._bbox = [10, 10, 110, 110]
    tracker.update_trajectory(track, frame)

    # Verify
    assert track_id in tracker.trajectories, f"Track ID {track_id} should have a trajectory"
    assert (
        len(tracker.trajectories[track_id]) == 2
    ), f"Expected 2 trajectory points, got {len(tracker.trajectories[track_id])}"


def test_update_trajectory_trims_length(monkeypatch):
    """Ensure that update_trajectory trims trajectories longer than cfg.TRAJECTORY_LENGTH (line 102)."""
    # Reduce trajectory length for faster test -------------------------------------------------------
    import config as cfg

    monkeypatch.setattr(cfg, "TRAJECTORY_LENGTH", 5, raising=False)

    # Setup ------------------------------------------------------------------------------------------
    tracker = Tracker()
    frame = _dummy_frame()
    track = _StubTrack(track_id=1, bbox=[0, 0, 10, 10])

    # Exercise: add more than TRAJECTORY_LENGTH points -----------------------------------------------
    for i in range(10):
        track._bbox = [i, i, i + 10, i + 10]  # Move track
        tracker.update_trajectory(track, frame)

    # Verify -----------------------------------------------------------------------------------------
    assert (
        len(tracker.trajectories[track.track_id]) == cfg.TRAJECTORY_LENGTH
    ), "Trajectory should be trimmed to configured length"


def test_process_detections_calls_internal_methods(monkeypatch):
    """Cover lines 119–120 ensuring process_single_track and update_trajectory are invoked."""
    tracker = Tracker()

    # Stub update_tracks to skip heavy DeepSort ------------------------------------------------------
    monkeypatch.setattr(tracker, "update_tracks", lambda *_a, **_k: None)

    # Prepare stub track -----------------------------------------------------------------------------
    track = _StubTrack(track_id=7, bbox=[0, 0, 10, 10])
    track.time_since_update = 0  # Ensure branch passes

    # Attach stub track list -------------------------------------------------------------------------
    tracker.tracker.tracker.tracks = [track]

    # Monitor calls ----------------------------------------------------------------------------------
    called = {}

    def _spy_process_single(track_obj, *_a, **_k):
        called["single"] = track_obj.track_id

    def _spy_update_trajectory(track_obj, *_a, **_k):
        called["traj"] = track_obj.track_id

    monkeypatch.setattr(tracker, "process_single_track", _spy_process_single)
    monkeypatch.setattr(tracker, "update_trajectory", _spy_update_trajectory)

    # Exercise --------------------------------------------------------------------------------------
    tracker.process_detections([], _dummy_frame(), _StubOCRReader())

    # Verify ----------------------------------------------------------------------------------------
    assert called.get("single") == track.track_id
    assert called.get("traj") == track.track_id
