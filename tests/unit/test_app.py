"""
Unit tests for the main application module.

These tests cover the LicensePlateDetectorApp class, the main application class
for the license plate detection system.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import pytest
import app as app_module


class DummyVideoCapture:
    """Stub for cv2.VideoCapture with configurable state."""

    def __init__(self, *_args, opened: bool = True, fps: float = 30.0, width: int = 640, height: int = 480, **_kwargs):
        self._opened = opened
        self._fps = fps
        self._width = width
        self._height = height
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, prop_id):
        import cv2

        mapping = {
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_WIDTH: self._width,
            cv2.CAP_PROP_FRAME_HEIGHT: self._height,
        }
        return mapping[prop_id]

    def release(self):
        self.released = True


class DummyVideoHandler:
    """Minimal stub mimicking the public interface used in app.py."""

    def __init__(self, *_args, **_kwargs):
        self.writer = None
        self.released = False
        self._calls = 0  # Only one frame is returned, then stops

    def read_frame(self) -> Tuple[bool, object]:
        self._calls += 1
        if self._calls == 1:
            return True, object()
        return False, None

    def write_frame(self, _frame):
        return None

    def release(self):
        self.released = True


class DummyYOLO:
    def detect(self, _frame):
        return ["plate"]


class DummyTracker:
    def process_detections(self, detections, frame, reader):
        self.called = True


class DummyFPSCounter:
    def __init__(self):
        self.count = 0

    def update(self):
        self.count += 1

    def get_fps(self):
        return 60.0


@pytest.fixture(autouse=True)
def patch_app_dependencies(monkeypatch):
    """Patch heavy external dependencies so tests run fast and deterministically."""
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_kw: DummyVideoCapture(*_a, **_kw), raising=True)
    monkeypatch.setattr(app_module, "VideoHandler", DummyVideoHandler, raising=True)
    monkeypatch.setattr(app_module, "YOLODetector", lambda: DummyYOLO(), raising=True)
    monkeypatch.setattr(app_module, "Tracker", lambda: DummyTracker(), raising=True)
    monkeypatch.setattr(app_module, "draw_fps_on_frame", lambda *_a, **_kw: None, raising=True)
    monkeypatch.setattr(app_module, "FPSCounter", DummyFPSCounter, raising=True)

    def _fake_thread(*_args, **_kwargs):
        target = _kwargs.get("target")
        if target is not None:
            target(**_kwargs.get("kwargs", {}))
        return SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(app_module.threading, "Thread", _fake_thread, raising=True)
    monkeypatch.setattr(app_module, "build_interface", lambda: SimpleNamespace(launch=lambda **_kw: None), raising=True)


def test_app_initialization() -> None:
    """Test initialization of the main application class and its components."""
    app = app_module.LicensePlateDetectorApp()
    assert app.yolo_detector is not None, "YOLO detector should be initialized"
    assert app.tracker is not None, "Tracker should be initialized"
    assert app.ocr_reader is not None, "OCR reader should be initialized"
    assert app.video_handler is None, "Video handler should be None until setup"


def test_setup_video_with_webcam() -> None:
    """Test video setup with webcam configuration."""
    app = app_module.LicensePlateDetectorApp()
    app.setup_video(WEBCAM=True, INPUT_VIDEO=None)
    assert app.video_handler is not None, "Video handler should be initialized"
    assert hasattr(app, "video_handler"), "Video handler should be set"


def test_setup_video_with_file(tmp_path: Path) -> None:
    """Test video setup with file input configuration."""
    app = app_module.LicensePlateDetectorApp()
    test_video = "./tests/assets/test_video.mp4"
    app.setup_video(WEBCAM=False, INPUT_VIDEO=test_video)
    assert app.video_handler is not None, "Video handler should be initialized"
    assert hasattr(app, "video_handler"), "Video handler should be set"


def test_setup_video_failure(monkeypatch):
    """Test that SystemExit is raised when video cannot be opened."""
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_kw: DummyVideoCapture(opened=False), raising=True)
    app = app_module.LicensePlateDetectorApp()
    with pytest.raises(SystemExit):
        app.setup_video(WEBCAM=False, INPUT_VIDEO="non_existent.mp4")


def test_setup_video_saves_output(monkeypatch):
    """Test that output path is set when SAVE_TO_VIDEO is True."""
    import config as config_module

    monkeypatch.setattr(config_module, "SAVE_TO_VIDEO", True, raising=False)
    monkeypatch.setattr(app_module, "SAVE_TO_VIDEO", True, raising=False)
    app = app_module.LicensePlateDetectorApp()
    app.setup_video(WEBCAM=False, INPUT_VIDEO="sample.mp4")
    assert app.output_video_path == "sample_processed.mp4"
    assert app.video_handler is not None


def test_launch_gradio_runs() -> None:
    """Test that launch_gradio executes without exceptions."""
    app = app_module.LicensePlateDetectorApp()
    app.launch_gradio()


def test_run_main_loop(monkeypatch):
    """Test the main loop with dummy components."""
    dummy_handler = DummyVideoHandler()

    def _fake_setup_video(self, *args, **kwargs):
        self.video_handler = dummy_handler
        self.output_video_path = None
        return None

    monkeypatch.setattr(app_module.LicensePlateDetectorApp, "setup_video", _fake_setup_video, raising=True)
    app = app_module.LicensePlateDetectorApp()
    app.run()
    assert dummy_handler.released is True, "release() should be called at end of run()"


def test_main_function(monkeypatch):
    """Test that main() builds the interface and launches it."""
    launch_called = {}

    def _fake_launch(**_kwargs):
        launch_called["yes"] = True

    monkeypatch.setattr(app_module, "build_interface", lambda: SimpleNamespace(launch=_fake_launch), raising=True)
    app_module.main()
    assert launch_called.get("yes") is True


def test_run_with_no_video_handler(monkeypatch):
    """Ensure run() exits early when video_handler is not initialized (lines 94–95)."""
    from importlib import reload

    # Patch setup_video to leave video_handler as ``None`` ----------------------------------
    def _fake_setup(self, *a, **k):
        self.video_handler = None  # Explicitly leave it uninitialised
        return None

    monkeypatch.setattr(app_module.LicensePlateDetectorApp, "setup_video", _fake_setup, raising=True)

    # Exercise -----------------------------------------------------------------------------
    app = app_module.LicensePlateDetectorApp()
    app.run()  # Should return quickly without raising


def test_run_with_writer_and_no_detections(monkeypatch):
    """Cover the branch where there are *no* detections, writer is present, and
    an output path is set (lines 110, 117 & 121)."""

    # --------------- Dummy components -----------------------------------------------------
    class _DummyVideoHandler(DummyVideoHandler):
        def __init__(self):
            super().__init__()
            self.wrote = False
            # Provide a writer attribute so the branch is taken
            self.writer = True  # Truthy value is enough

        def write_frame(self, _frame):
            # Override to record the write invocation
            self.wrote = True

    class _NoDetYOLO:
        @staticmethod
        def detect(_frame):
            return []  # Force *no* detections branch

    # --------------- Patching -------------------------------------------------------------
    monkeypatch.setattr(app_module, "YOLODetector", lambda: _NoDetYOLO(), raising=True)

    dummy_handler = _DummyVideoHandler()

    def _fake_setup(self, *a, **k):
        self.video_handler = dummy_handler
        self.output_video_path = "dummy_out.mp4"
        return None

    monkeypatch.setattr(app_module.LicensePlateDetectorApp, "setup_video", _fake_setup, raising=True)

    # --------------- Exercise -------------------------------------------------------------
    app = app_module.LicensePlateDetectorApp()
    app.run()

    # --------------- Verify ---------------------------------------------------------------
    assert dummy_handler.released is True, "Handler should be released"
    assert dummy_handler.wrote is True, "Writer.write() should have been invoked"
