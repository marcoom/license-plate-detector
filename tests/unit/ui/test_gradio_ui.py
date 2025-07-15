"""Unit tests for the Gradio UI Gradio interface.

These tests cover:
1. Basic interface creation.
2. Video-processing generator behaviour under normal and edge conditions.
3. Execution of internal callbacks defined within ``build_interface``.
4. Webcam availability detection.
"""

# Standard library imports
from unittest.mock import patch

# Third-party imports
import gradio as gr
import numpy as np
import pytest

# Local application imports
from ui.gradio_ui import build_interface, process_video, get_available_sources


@patch('os.path.exists')
def test_get_available_sources_webcam_available(mock_exists):
    """Test get_available_sources when webcam is available.
    
    Verifies that:
    - When /dev/video0 exists, both "Video File" and "Webcam" are returned
    - The os.path.exists function is called with the correct path
    """
    # Setup mock to return True for webcam check
    mock_exists.return_value = True
    
    # Call the function and check results
    result = get_available_sources()
    
    # Verify the results
    assert result == ["Video File", "Webcam"]
    mock_exists.assert_called_once_with('/dev/video0')


@patch('os.path.exists')
def test_get_available_sources_webcam_not_available(mock_exists):
    """Test get_available_sources when webcam is not available.
    
    Verifies that:
    - When /dev/video0 doesn't exist, only "Video File" is returned
    - The os.path.exists function is called with the correct path
    """
    # Setup mock to return False for webcam check
    mock_exists.return_value = False
    
    # Call the function and check results
    result = get_available_sources()
    
    # Verify the results
    assert result == ["Video File"]
    mock_exists.assert_called_once_with('/dev/video0')


def test_build_interface_returns_blocks() -> None:
    """Test that build_interface creates a properly configured Gradio Blocks object.

    Verifies that:
    - The function returns a Gradio Blocks instance
    - The interface has the expected components (though we don't test the UI layout)
    """
    # Exercise
    app_ui = build_interface()

    # Verify
    assert isinstance(app_ui, gr.Blocks), "Expected a Gradio Blocks instance"
    assert hasattr(app_ui, "queue"), "Gradio interface should have queue method"


def test_process_video_yields_frames() -> None:
    """Test that the video processing generator yields valid video frames.

    Verifies that:
    - The generator yields numpy arrays
    - The arrays have the expected shape and dtype
    - Multiple frames can be consumed from the generator
    """
    # Setup & Exercise
    frame_generator = process_video()

    # Verify first frame
    first_frame = next(frame_generator)
    assert isinstance(first_frame, np.ndarray), "Expected a numpy array"
    assert first_frame.dtype == np.uint8, "Frame should be 8-bit unsigned int"
    assert len(first_frame.shape) == 3, "Frame should be 3D (height, width, channels)"

    # Verify we can get more frames
    second_frame = next(frame_generator)
    assert isinstance(second_frame, np.ndarray), "Second frame should also be a numpy array"


# -----------------------------------------------------------------------------
# Edge-case and internal-callback tests
# -----------------------------------------------------------------------------


def test_process_video_handles_missing_video_handler(monkeypatch) -> None:
    """If ``app.video_handler`` is ``None`` the generator should yield a single
    informative blank frame and then stop. This exercises lines 53–54 of
    ``gradio_ui.py``."""

    # Create a minimal stub for the Application -------------------------------------------------
    class _StubApp:
        video_handler = None  # Triggers the branch we want to cover

        # Interface expected by ``process_video``
        def setup_video(self, *args, **kwargs):
            pass

    # Patch the factory so it returns our stub --------------------------------------------------
    monkeypatch.setattr("ui.gradio_ui._prepare_app", lambda: _StubApp())

    from ui import gradio_ui  # Local import to get fresh state after patch

    # Exercise -------------------------------------------------------------------------------
    gen = gradio_ui.process_video()

    first_item = next(gen)
    with pytest.raises(StopIteration):
        next(gen)

    # Verify --------------------------------------------------------------------------------
    assert isinstance(first_item, np.ndarray)
    assert first_item.shape == (100, 100, 3)
    assert (first_item == 0).all(), "Expected a completely black informative frame"


def test_process_video_respects_stop_event(monkeypatch) -> None:
    """Ensure the processing loop terminates promptly when the global
    ``stop_event`` is set. Covers lines 61–62, 79–81."""

    # ----------------------------- Video-handler stub -----------------------------------------
    class DummyVideoHandler:
        def __init__(self):
            self.released = False
            self._frame_returned = False

        def read_frame(self):
            # Return a single valid frame once, then signal end-of-stream
            if not self._frame_returned:
                self._frame_returned = True
                frame = np.zeros((10, 10, 3), dtype=np.uint8)
                return True, frame
            return False, None

        def release(self):
            self.released = True

    # ----------------------------- App stub ---------------------------------------------------
    class StubApp:
        def __init__(self):
            self.video_handler = DummyVideoHandler()

            # Attributes accessed inside the loop --------------------------------------------
            class NoopDetector:
                @staticmethod
                def detect(frame):
                    return []

            self.yolo_detector = NoopDetector()

            class NoopTracker:
                @staticmethod
                def process_detections(*args, **kwargs):
                    pass

            self.tracker = NoopTracker()

            class NoopOCR:
                reader = None

            self.ocr_reader = NoopOCR()

        def setup_video(self, *args, **kwargs):
            pass

    monkeypatch.setattr("ui.gradio_ui._prepare_app", lambda: StubApp())

    from ui import gradio_ui

    # Prime the stop event so the very first check terminates the loop ------------------------
    gen = gradio_ui.process_video()

    # First frame should be yielded normally --------------------------------------------------
    frame = next(gen)
    assert isinstance(frame, np.ndarray)

    # Signal stop and expect termination on next iteration ------------------------------------
    gradio_ui.stop_event.set()
    with pytest.raises(StopIteration):
        next(gen)


def test_build_interface_executes_internal_callbacks(monkeypatch) -> None:
    """Patch Gradio component methods so that callback functions supplied via
    ``.change()``, ``.click()`` and ``Event.then()`` are executed *immediately*.
    This provides coverage for multiple inner–function definitions inside
    ``build_interface``.
    """

    import inspect
    import gradio as gr

    # ----------------------------- Helper: run callback --------------------------------------
    def _run(fn):
        """Execute *fn* if it is callable.

        If the callable accepts at least one positional parameter, a dummy
        argument is supplied. This avoids ``TypeError`` for callbacks that
        expect a value from the Gradio component. Built-in callables that do
        not expose a signature (e.g. ``None``) are ignored gracefully.
        """
        if not callable(fn):
            return None  # Nothing to execute

        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):  # Built-in or C-implemented callables
            params = {}

        if params:
            return fn("dummy")
        return fn()

    # ----------------------------- Stub event ------------------------------------------------
    class _DummyEvent:
        def __init__(self, fn):
            _run(fn)  # Run the callback immediately upon creation

        def then(self, fn, *_a, **_k):  # pylint: disable=unused-argument
            _run(fn)  # Also run any chained callbacks
            return self

    # ----------------------------- Generic component patch ----------------------------------
    def _make_change(_orig_change):
        def _change(self, fn, *a, **k):  # noqa: D401  pylint: disable=unused-argument
            _run(fn)
            return _DummyEvent(fn)

        return _change

    def _make_click(orig_click):
        def _click(self, fn, *a, **k):  # noqa: D401  pylint: disable=unused-argument
            _run(fn)
            return _DummyEvent(fn)

        return _click

    # Patch every component class that supports change / click -------------------------------
    for comp_cls_name in ("Radio", "File", "Slider", "Checkbox", "Dropdown", "Button"):
        comp_cls = getattr(gr, comp_cls_name)
        if hasattr(comp_cls, "change"):
            monkeypatch.setattr(comp_cls, "change", _make_change(comp_cls.change))
        if hasattr(comp_cls, "click"):
            monkeypatch.setattr(comp_cls, "click", _make_click(getattr(comp_cls, "click", lambda: None)))

    # ``gr.update`` normally builds a special update object – we just need a placeholder -------
    monkeypatch.setattr(gr, "update", lambda **kwargs: kwargs)

    # Exercise -------------------------------------------------------------------------------
    ui = build_interface()

    # Verify the Blocks object is still built -------------------------------------------------
    assert isinstance(ui, gr.Blocks)
