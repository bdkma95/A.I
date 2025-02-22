# translation.py
import cv2
import pytesseract
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from functools import lru_cache
from googletrans import Translator, LANGUAGES
from pydantic import validate_arguments
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """Custom exception for translation failures"""

class OCRError(Exception):
    """Custom exception for OCR-related failures"""

class UnsupportedLanguageError(ValueError):
    """Exception for unsupported target languages"""

class TranslationSystem:
    """
    Enhanced translation system with OCR capabilities and advanced features
    
    Features:
    - Multiple OCR preprocessing techniques
    - Translation caching
    - Language validation
    - Adaptive image processing
    - Retry mechanism for API calls
    """
    
    def __init__(
        self,
        translator: Optional[Translator] = None,
        ocr_config: str = r'--oem 3 --psm 6',
        preprocess_steps: Tuple[str, ...] = ('grayscale', 'denoise')
    ):
        """
        Initialize translation system
        
        Args:
            translator: Optional preconfigured translator instance
            ocr_config: Tesseract OCR configuration
            preprocess_steps: Tuple of image preprocessing steps to apply
        """
        self.translator = translator or Translator()
        self.ocr_config = ocr_config
        self.preprocess_steps = preprocess_steps
        self.supported_languages = LANGUAGES

    @validate_arguments
    def translate_text(
        self,
        text: str,
        dest_lang: str = 'es',
        src_lang: Optional[str] = None
    ) -> str:
        """
        Translate text with validation and caching
        
        Args:
            text: Text to translate (1-5000 characters)
            dest_lang: Target language code (default: Spanish)
            src_lang: Optional source language code
            
        Returns:
            Translated text or original on failure
        """
        self._validate_language_code(dest_lang)
        return self._cached_translation(text, dest_lang, src_lang)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def _cached_translation(
        self,
        text: str,
        dest_lang: str,
        src_lang: Optional[str]
    ) -> str:
        """Retryable translation method with caching"""
        try:
            result = self.translator.translate(
                text,
                dest=dest_lang,
                src=src_lang
            )
            return result.text
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise TranslationError("Text translation failed") from e

    @validate_arguments
    def translate_image(
        self,
        img_path: Path,
        dest_lang: str = 'es',
        ocr_config: Optional[str] = None
    ) -> str:
        """
        Translate text from image with enhanced OCR processing
        
        Args:
            img_path: Path to image file
            dest_lang: Target language code
            ocr_config: Optional custom OCR configuration
            
        Returns:
            Translated text or empty string on failure
        """
        try:
            self._validate_image_path(img_path)
            processed_img = self._preprocess_image(img_path)
            text = self._ocr_extract(processed_img, ocr_config)
            return self.translate_text(text, dest_lang)
        except Exception as e:
            logger.error(f"Image translation failed: {str(e)}")
            return ""

    def _preprocess_image(self, img_path: Path) -> np.ndarray:
        """Apply configured image preprocessing steps"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise OCRError("Failed to read image file")
            
        for step in self.preprocess_steps:
            if step == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif step == 'denoise':
                img = cv2.fastNlMeansDenoisingColored(img)
            elif step == 'threshold':
                img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            elif step == 'upscale':
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
        return img

    def _ocr_extract(
        self,
        image: np.ndarray,
        custom_config: Optional[str]
    ) -> str:
        """Perform OCR with error handling"""
        try:
            return pytesseract.image_to_string(
                image,
                config=custom_config or self.ocr_config
            )
        except pytesseract.TesseractError as e:
            raise OCRError(f"OCR processing failed: {str(e)}") from e

    def _validate_language_code(self, lang_code: str) -> None:
        """Validate language code against supported languages"""
        if lang_code not in self.supported_languages:
            raise UnsupportedLanguageError(f"Unsupported language: {lang_code}")

    def _validate_image_path(self, path: Path) -> None:
        """Validate image path exists and is readable"""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Image file too large")

    @lru_cache(maxsize=1000)
    def cached_translate_text(self, text: str, dest_lang: str) -> str:
        """Cached version of translate_text for frequent requests"""
        return self.translate_text(text, dest_lang)
