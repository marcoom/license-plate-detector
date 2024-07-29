# src/ocr/ocr.py
import easyocr

class EasyOCRRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def recognize(self, image):
        text = self.reader.readtext(image)
        return text[0][-2] if text else ""
