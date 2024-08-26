# src/ocr/ocr.py
from collections import Counter

def get_most_common_plate(ocr_history, track_id):
    """
    Retorna la patente más frecuente en el historial de OCR para un track_id.
    """
    if ocr_history[track_id]:
        return Counter(ocr_history[track_id]).most_common(1)[0][0]
    return "Plate"