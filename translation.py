# translation.py
import cv2
import pytesseract
import logging
from googletrans import Translator
from config import TRANSLATOR

logger = logging.getLogger(__name__)

class TranslationSystem:
    def __init__(self):
        self.translator = TRANSLATOR
        self.ocr_config = r'--oem 3 --psm 6'
    
    def translate_text(self, text: str, dest_lang: str = 'es') -> str:
        try:
            return self.translator.translate(text, dest=dest_lang).text
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text
    
    def translate_image(self, img_path: str) -> str:
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Image not found or unreadable.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config=self.ocr_config)
            return self.translate_text(text)
        except Exception as e:
            logger.error(f"Image translation failed: {str(e)}")
            return ""
