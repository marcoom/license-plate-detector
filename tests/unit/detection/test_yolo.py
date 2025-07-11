"""Unit tests for the YOLO detector module.

Tests for the YOLODetector class, responsible for detecting objects in images using the YOLO model.
"""

import logging
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from detection.yolo import YOLODetector


def test_detect_returns_expected_format(dummy_frame: np.ndarray) -> None:
    """Should return detections in the expected format."""
    with patch("detection.yolo.YOLO") as mock_yolo:
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = [[10, 20, 110, 120, 0.9, 0.0]]
        mock_yolo.return_value.predict.return_value = [mock_result]
        detector = YOLODetector()
        detections = detector.detect(dummy_frame)
        assert isinstance(detections, list)
        assert detections
        bbox, score, class_id = detections[0]
        assert len(bbox) == 4
        assert all(isinstance(coord, (int, float)) for coord in bbox)
        assert 0.0 <= score <= 1.0
        assert isinstance(class_id, (float, int))


def test_threshold_filtering(dummy_frame: np.ndarray) -> None:
    """Should filter detections based on confidence threshold."""
    with patch("detection.yolo.YOLO") as mock_yolo:
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = [[10, 20, 110, 120, 0.9, 0.0]]
        mock_yolo.return_value.predict.return_value = [mock_result]
        detector = YOLODetector()
        detector.threshold = 0.94
        detections = detector.detect(dummy_frame)
        assert detections == []


def test_init_sets_model_and_threshold(monkeypatch):
    """Should initialize model and threshold from config."""
    with patch("detection.yolo.YOLO") as mock_yolo, patch("detection.yolo.cfg") as mock_cfg:
        mock_cfg.MODEL_PATH = "dummy_path.pt"
        mock_cfg.YOLO_THRESHOLD = 0.5
        detector = YOLODetector()
        mock_yolo.assert_called_once_with("dummy_path.pt", task="detect", verbose=False)
        assert detector.threshold == 0.5


def test_detect_empty_results(monkeypatch, dummy_frame):
    """Should return empty list when model returns no detections."""
    with patch("detection.yolo.YOLO") as mock_yolo:
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = []
        mock_yolo.return_value.predict.return_value = [mock_result]
        detector = YOLODetector()
        out = detector.detect(dummy_frame)
        assert out == []


def test_detect_at_threshold(monkeypatch, dummy_frame):
    """Should not return detections at exactly the threshold."""
    with patch("detection.yolo.YOLO") as mock_yolo, patch("detection.yolo.cfg") as mock_cfg:
        mock_cfg.MODEL_PATH = "dummy_path.pt"
        mock_cfg.YOLO_THRESHOLD = 0.9
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = [[0, 0, 10, 10, 0.9, 1.0]]
        mock_yolo.return_value.predict.return_value = [mock_result]
        detector = YOLODetector()
        assert detector.threshold == 0.9
        out = detector.detect(dummy_frame)
        assert out == []


def test_detect_malformed_model_output(monkeypatch, dummy_frame):
    """Should handle malformed model output gracefully."""
    with patch("detection.yolo.YOLO") as mock_yolo:
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = [[0, 0, 10, 10, 0.95]]
        mock_yolo.return_value.predict.return_value = [mock_result]
        detector = YOLODetector()
        out = detector.detect(dummy_frame)
        assert out == []


def test_detect_invalid_frame_type():
    """Should raise if frame is not a numpy array."""
    with patch("detection.yolo.YOLO"):
        detector = YOLODetector()
        with pytest.raises(Exception):
            detector.detect("not_a_frame")


def test_detect_model_predict_exception(dummy_frame):
    """Should handle model predict exceptions gracefully."""
    with patch("detection.yolo.YOLO") as mock_yolo:
        mock_yolo.return_value.predict.side_effect = RuntimeError("Model error")
        detector = YOLODetector()
        with pytest.raises(RuntimeError):
            detector.detect(dummy_frame)


def test_logging_on_init_and_detect(dummy_frame, caplog):
    """Should log on init and detect."""
    with patch("detection.yolo.YOLO") as mock_yolo, patch("detection.yolo.cfg") as mock_cfg:
        mock_cfg.MODEL_PATH = "dummy_path.pt"
        mock_cfg.YOLO_THRESHOLD = 0.5
        with caplog.at_level(logging.INFO):
            detector = YOLODetector()
            assert any("YOLO model loaded from" in m for m in caplog.messages)
        mock_result = MagicMock()
        mock_result.boxes.data.tolist.return_value = [[10, 20, 110, 120, 0.95, 0.0]]
        mock_yolo.return_value.predict.return_value = [mock_result]
        with caplog.at_level(logging.DEBUG):
            detector.detect(dummy_frame)
            assert any("Detected" in m for m in caplog.messages)
