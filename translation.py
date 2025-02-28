# translation.py
import cv2
import pytesseract
import logging
import httpx
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, ValidationError
from cachetools import TTLCache
from config import AsyncConfigManager
from exceptions import TranslationError, ImageTranslationError

logger = logging.getLogger(__name__)

class TranslationRequest(BaseModel):
    """Pydantic model for translation request validation"""
    text: str
    target_lang: str
    source_lang: Optional[str] = 'auto'
    format: Optional[str] = 'text'
    fallback: Optional[bool] = True

class TranslationResult(BaseModel):
    """Structured translation result model"""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: Optional[float]
    detected_lang: Optional[str]
    metadata: Dict[str, Any]

class TranslationSystem:
    """
    Async translation system with OCR and text translation capabilities
    
    Features:
    - Async context manager pattern
    - Multi-engine support (Google, DeepL)
    - Advanced image preprocessing
    - Configurable caching
    - Automatic fallback handling
    """
    
    def __init__(self, config: AsyncConfigManager):
        self.config = config
        self._client = httpx.AsyncClient()
        self.cache = TTLCache(maxsize=1000, ttl=timedelta(hours=1))
        self._executor = ThreadPoolExecutor()
        self.ocr_config = r'--oem 3 --psm 6'
        self._deepl_url = "https://api.deepl.com/v2/translate"

    async def __aenter__(self):
        """Async initialization"""
        await self._verify_credentials()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        await self._client.aclose()
        self._executor.shutdown(wait=False)

    async def _verify_credentials(self):
        """Validate translation service credentials"""
        if not self.config.translator:
            raise TranslationError("Translation service not configured")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def translate_text(self, request: TranslationRequest) -> TranslationResult:
        """
        Translate text with enhanced error handling and fallback
        
        Args:
            request: TranslationRequest with parameters
            
        Returns:
            TranslationResult with detailed metadata
        """
        try:
            # Check cache first
            cache_key = f"{request.source_lang}-{request.target_lang}-{hash(request.text)}"
            if cache_key in self.cache:
                logger.debug("Returning cached translation")
                return self.cache[cache_key]

            # Preprocess text
            cleaned_text = await self._preprocess_text(request.text)

            # Choose translation engine
            result = await self._translate_with_engine(cleaned_text, request)

            # Validate and cache result
            validated = self._validate_result(result, request)
            self.cache[cache_key] = validated
            return validated

        except Exception as e:
            if request.fallback:
                return await self._fallback_translation(request)
            raise TranslationError(f"Translation failed: {str(e)}") from e

    async def _translate_with_engine(self, text: str, request: TranslationRequest):
        """Select translation engine based on configuration"""
        if self.config.settings.deepl_api_key:
            return await self._translate_deepl(text, request)
        return await self._translate_google(text, request)

    async def _translate_deepl(self, text: str, request: TranslationRequest):
        """Translate using DeepL API"""
        params = {
            "auth_key": self.config.settings.deepl_api_key,
            "text": text,
            "target_lang": request.target_lang.upper(),
            "source_lang": request.source_lang.upper()
        }
        
        response = await self._client.post(
            self._deepl_url,
            data=params,
            timeout=self.config.settings.translation_timeout
        )
        response.raise_for_status()
        
        data = response.json()
        return TranslationResult(
            source_text=text,
            translated_text=data['translations'][0]['text'],
            source_lang=data['translations'][0]['detected_source_language'].lower(),
            target_lang=request.target_lang,
            confidence=data['translations'][0]['confidence'],
            metadata={"engine": "deepl"}
        )

    async def _translate_google(self, text: str, request: TranslationRequest):
        """Translate using Google's service"""
        loop = asyncio.get_running_loop()
        try:
            translated = await loop.run_in_executor(
                self._executor,
                lambda: self.config.translator.translate(
                    text,
                    dest=request.target_lang,
                    src=request.source_lang
                )
            )
            return TranslationResult(
                source_text=text,
                translated_text=translated.text,
                source_lang=translated.src.lower(),
                target_lang=translated.dest.lower(),
                detected_lang=translated.src.lower(),
                metadata={"engine": "google"}
            )
        except Exception as e:
            raise TranslationError("Google translation failed") from e

    async def translate_image(self, image_path: Path) -> TranslationResult:
        """
        Translate text in images with advanced preprocessing
        
        Args:
            image_path: Path to image file
            
        Returns:
            TranslationResult with image metadata
        """
        try:
            if not await self._check_image_exists(image_path):
                raise ImageTranslationError("Image file not found")

            # Async image processing
            loop = asyncio.get_running_loop()
            img = await loop.run_in_executor(
                self._executor,
                self._preprocess_image,
                image_path
            )

            # OCR processing
            text = await loop.run_in_executor(
                self._executor,
                pytesseract.image_to_string,
                img,
                config=self.ocr_config
            )

            # Translate extracted text
            translation = await self.translate_text(
                TranslationRequest(
                    text=text,
                    target_lang=self.config.settings.target_language
                )
            )

            return TranslationResult(
                **translation.dict(),
                metadata={
                    **translation.metadata,
                    "image_path": str(image_path),
                    "ocr_confidence": self._calculate_ocr_confidence(img)
                }
            )

        except Exception as e:
            raise ImageTranslationError(f"Image translation failed: {str(e)}") from e

    def _preprocess_image(self, image_path: Path):
        """Advanced image preprocessing for OCR"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ImageTranslationError("Invalid image file")

        # Preprocessing pipeline
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        resized = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, threshold = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return threshold

    async def _check_image_exists(self, path: Path) -> bool:
        """Async file existence check"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            path.exists
        )

    def _calculate_ocr_confidence(self, image: np.ndarray) -> float:
        """Calculate weighted OCR confidence score using Tesseract data"""
        try:
            # Get OCR data with confidence scores
            data = pytesseract.image_to_data(
                image, 
                config=self.ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = []
            word_lengths = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                if text and conf > 0:
                    confidences.append(conf)
                    word_lengths.append(len(text))
                    
            if not confidences:
                return 0.0
                
            # Calculate weighted average by word length
            total_length = sum(word_lengths)
            weighted_conf = sum(
                (conf * length) / total_length
                for conf, length in zip(confidences, word_lengths)
            )
            
            # Normalize to 0-1 scale (Tesseract returns 0-100)
            return round(weighted_conf / 100, 2)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.0

    def _validate_result(self, result: TranslationResult, request: TranslationRequest):
        """Validate translation result against request"""
        if len(result.translated_text) == 0:
            raise ValidationError("Empty translation result")
        return result

    async def _preprocess_text(self, text: str) -> str:
        """Comprehensive text cleaning and normalization"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_preprocess_text,
            text
        )

    def _sync_preprocess_text(self, text: str) -> str:
        """CPU-bound text preprocessing"""
        import unicodedata
        import re
        
        # Normalize Unicode (NFKC for compatibility composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace problematic characters
        replacements = {
            '“': '"', '”': '"', '‘': "'", '’': "'",
            '—': '-', '–': '-', '…': '...', ' ': ' '  # Replace non-breaking space
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        # Remove control characters (except tab/newline)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate to API limits
        max_length = self.config.settings.max_text_length
        if len(text) > max_length:
            logger.warning(f"Truncating text from {len(text)} to {max_length} characters")
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
        
        # Language-specific cleanup
        if self.config.settings.target_language in ['ar', 'he']:
            text = re.sub(r'[\u200e\u200f]', '', text)  # Remove RTL/LTR marks
        
        return text

    async def _fallback_translation(self, request: TranslationRequest):
        """Fallback translation strategy"""
        logger.warning("Using fallback translation strategy")
        return TranslationResult(
            source_text=request.text,
            translated_text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            confidence=0.0,
            metadata={"fallback": True}
        )
