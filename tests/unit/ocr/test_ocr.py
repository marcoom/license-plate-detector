"""Unit tests for the OCR (Optical Character Recognition) module.

This module contains tests for the OCRReader class, which is responsible
for reading text from license plate images.
"""

# Standard library imports
from typing import Dict, List

# Third-party imports
import numpy as np
import pytest

# Local application imports
from ocr.ocr import OCRReader


def _stub_easyocr_reader(
    monkeypatch: pytest.MonkeyPatch, texts: List[str], conf: float = 0.9
) -> None:  # noqa: D401, E501
    """Patch ``easyocr.Reader`` so that it returns *deterministic* results.

    The function replaces ``easyocr.Reader`` with a lightweight dummy
    implementation whose ``readtext`` method yields tuples in the same shape
    produced by *easyocr*::

        [(bbox, text, confidence), ...]

    Only the *text* and *confidence* fields are relevant for our tests, so a
    placeholder ``None`` bbox is sufficient.

    Args:
        monkeypatch: Pytest fixture used to monkeypatch Python objects.
        texts: Sequence of strings that should be returned – in order – each
            time ``readtext`` is invoked. When exhausted the last value is
            repeated.
        conf: Confidence to attach to every returned detection.
    """

    class _DummyReader:  # pylint: disable=too-few-public-methods
        def __init__(self, _: List[str]):
            self._texts = texts

        def readtext(self, _img):
            if self._texts:
                text = self._texts.pop(0)
            else:
                text = texts[-1]
            return [(None, text, conf)] if text is not None else []

    monkeypatch.setattr("easyocr.Reader", _DummyReader)


def test_read_plate_returns_expected_text(monkeypatch: pytest.MonkeyPatch, dummy_plate: np.ndarray, dummy_ocr_history: Dict[int, List[str]]) -> None:
    """Should return the most common text from history."""
    reader = OCRReader(languages=["en"])
    reader.ocr_history = dummy_ocr_history
    result = reader.read_plate(img_license=dummy_plate, track_id=1, confidence_threshold=0.5)
    assert result == "ABC 12345"


def test_get_history_returns_accumulated_reads(monkeypatch: pytest.MonkeyPatch, dummy_plate: np.ndarray) -> None:
    """Should accumulate OCR reads for each track ID."""
    reader = OCRReader()
    track_id = 42
    monkeypatch.setattr(reader.reader, "readtext", lambda img: [(None, "XYZ 9876", 0.9)])
    reader.read_plate(img_license=dummy_plate, track_id=track_id, confidence_threshold=0.5)
    reader.read_plate(img_license=dummy_plate, track_id=track_id, confidence_threshold=0.5)
    history = reader.get_history()
    assert track_id in history
    assert len(history[track_id]) >= 1
    history_entry = history[track_id][0]
    assert isinstance(history_entry, str)
    assert len(history_entry) > 0


def test_read_plate_empty_ocr_results(monkeypatch, dummy_plate):
    """Should return 'Plate' if OCR returns no results."""
    reader = OCRReader()
    monkeypatch.setattr(reader.reader, "readtext", lambda img: [])
    result = reader.read_plate(img_license=dummy_plate, track_id=99, confidence_threshold=0.5)
    assert result == "Plate"
    assert reader.ocr_history[99] == []


def test_read_plate_low_confidence(monkeypatch, dummy_plate):
    """Should not add low-confidence results to history."""
    reader = OCRReader()
    monkeypatch.setattr(reader.reader, "readtext", lambda img: [(None, "BAD", 0.1)])
    result = reader.read_plate(img_license=dummy_plate, track_id=77, confidence_threshold=0.5)
    assert result == "Plate"
    assert reader.ocr_history[77] == []


def test_read_plate_non_string_text(monkeypatch, dummy_plate):
    """Should ignore non-string OCR results."""
    reader = OCRReader()
    monkeypatch.setattr(reader.reader, "readtext", lambda img: [(None, 12345, 0.9)])
    result = reader.read_plate(img_license=dummy_plate, track_id=55, confidence_threshold=0.5)
    assert result == "Plate"
    assert reader.ocr_history[55] == []


def test_get_most_common_plate_no_history():
    """Should return 'Plate' if no history exists."""
    assert OCRReader.get_most_common_plate({}, 1) == "Plate"
    assert OCRReader.get_most_common_plate({2: []}, 2) == "Plate"


def test_get_most_common_plate_returns_most_frequent():
    """Should return the most common value."""
    history = {3: ["A", "B", "A", "A", "B"]}
    assert OCRReader.get_most_common_plate(history, 3) == "A"


def test_ocrreader_init_custom_languages(monkeypatch):
    """Should initialize with custom languages."""
    called = {}
    def dummy_reader(langs):
        called["langs"] = langs
        class Dummy:
            def readtext(self, img):
                return []
        return Dummy()
    monkeypatch.setattr("easyocr.Reader", dummy_reader)
    _ = OCRReader(languages=["es", "fr"])
    assert called["langs"] == ["es", "fr"]


def test_multiple_track_ids(monkeypatch, dummy_plate):
    """Should handle multiple track IDs independently."""
    reader = OCRReader()
    monkeypatch.setattr(reader.reader, "readtext", lambda img: [(None, "PLATE1", 0.8)])
    reader.read_plate(img_license=dummy_plate, track_id=1, confidence_threshold=0.5)
    monkeypatch.setattr(reader.reader, "readtext", lambda img: [(None, "PLATE2", 0.8)])
    reader.read_plate(img_license=dummy_plate, track_id=2, confidence_threshold=0.5)
    assert reader.ocr_history[1][0] == "PLATE1"
    assert reader.ocr_history[2][0] == "PLATE2"
    """Test that get_history returns accumulated OCR reads for each track ID.
    
    Verifies that:
    - Multiple reads for the same track ID are accumulated
    - The history contains the expected track ID
    - The history contains at least one entry for the track ID
        Args:
        monkeypatch: Pytest fixture for modifying objects during testing.
        dummy_plate: Test image of a license plate.
    """
    # Setup
    reader = OCRReader()
    track_id = 42

    # Exercise - perform multiple reads for the same track ID
    reader.read_plate(img_license=dummy_plate, track_id=track_id, confidence_threshold=0.5)
    reader.read_plate(img_license=dummy_plate, track_id=track_id, confidence_threshold=0.5)

    # Verify
    history = reader.get_history()
    assert track_id in history, f"Expected track ID {track_id} in history"
    assert len(history[track_id]) >= 1, f"Expected at least one history entry for track ID {track_id}"

    # Verify history entry format
    history_entry = history[track_id][0]
    assert isinstance(history_entry, str), "History entries should be strings"
    assert len(history_entry) > 0, "History entries should not be empty"


def test_read_plate_appends_history_and_returns_mode(monkeypatch: pytest.MonkeyPatch, dummy_plate: np.ndarray) -> None:
    """Positive path: a valid, high-confidence result is aggregated.

    The test calls :py:meth:`OCRReader.read_plate` three times for the same
    ``track_id``. Two calls return the same text (``ABC123``) thereby
    becoming the mode; one call returns another text (``XYZ987``).
    The function should keep all entries *and* return the most frequent one.
    """

    _stub_easyocr_reader(monkeypatch, texts=["ABC123", "XYZ987", "ABC123"])
    reader = OCRReader()

    # Exercise – three reads for identical track-ID.
    for _ in range(3):
        reader.read_plate(img_license=dummy_plate, track_id=7, confidence_threshold=0.5)

    # Verify – history length & returned mode.
    history: Dict[int, List[str]] = reader.get_history()
    assert history[7] == ["ABC123", "XYZ987", "ABC123"]
    assert OCRReader.get_most_common_plate(history, 7) == "ABC123"


def test_read_plate_creates_history_for_new_track(monkeypatch: pytest.MonkeyPatch, dummy_plate: np.ndarray) -> None:
    """``read_plate`` must create a new list when encountering a fresh ID."""

    _stub_easyocr_reader(monkeypatch, texts=["NEW 001"])
    reader = OCRReader()

    result = reader.read_plate(img_license=dummy_plate, track_id=99, confidence_threshold=0.5)

    assert result == "NEW 001", "Should immediately return freshly detected plate"
    assert reader.ocr_history[99] == ["NEW 001"], "History should contain the new entry"


@pytest.mark.parametrize(
    "history,track_id,expected",
    [
        ({1: ["A", "B", "A"]}, 1, "A"),  # clear winner
        ({2: ["X", "Y"]}, 2, "X"),  # tie → Counter keeps insertion order
    ],
)
def test_get_most_common_plate_various(
    history: Dict[int, List[str]], track_id: int, expected: str
) -> None:  # noqa: D401, E501
    """Unit-check the static helper *without* relying on an ``OCRReader`` instance."""

    assert OCRReader.get_most_common_plate(history, track_id) == expected


def test_read_plate_ignores_results_below_threshold(
    monkeypatch: pytest.MonkeyPatch, dummy_plate: np.ndarray
) -> None:  # noqa: D401, E501
    """A detection with insufficient confidence must **not** be stored."""

    _stub_easyocr_reader(monkeypatch, texts=["LOWCONF"], conf=0.1)
    reader = OCRReader()

    output = reader.read_plate(img_license=dummy_plate, track_id=5, confidence_threshold=0.5)

    # The text is ignored → default "Plate" returned and history remains empty list.
    assert output == "Plate"
    assert reader.ocr_history[5] == []
