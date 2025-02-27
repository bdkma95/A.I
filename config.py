# config.py
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseSettings, Field, ValidationError, validator
import spacy
from spacy.util import get_lang_class
from transformers import pipeline
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Enhanced application settings with async support"""
    # API Configuration
    safetipin_api_key: str = Field(..., env="SAFETIPIN_API_KEY")
    openweather_api_key: str = Field(..., env="OPENWEATHER_API_KEY")
    deepl_api_key: Optional[str] = Field(None, env="DEEPL_API_KEY")
    
    # System Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    model_cache_dir: Path = Path("./models")
    test_image_path: Path = Path("./data/sample_sign.jpg")
    max_text_length: int = 5000
    text_preview_length: int = 30
    
    # Recommendation System
    recommendation_top_k: int = 5
    recommendation_max_distance: float = 0.4
    default_preferences: Dict[str, int] = {'adventure': 5, 'culture': 3, 'budget': 4}
    
    # Safety System
    safety_api_retries: int = 3
    safety_api_timeout: int = 10
    safety_test_texts: List[str] = [
        "This area is perfectly safe during daylight hours",
        "Avoid this neighborhood after dark, frequent incidents reported",
        "Mixed reviews about safety, generally okay with precautions"
    ]
    
    # Translation System
    translation_timeout: int = 10
    target_language: str = "es"
    translation_test_cases: List[Tuple[str, str]] = [
        ("Hello, where is the nearest hospital?", "es"),
        ("Emergency exit on the right side", "fr"),
        ("Vegetarian food options available", "de")
    ]
    
    # Social System
    social_clustering_algorithm: str = "kmeans"
    social_test_interests: List[str] = [
        "hiking mountain trails",
        "visiting art museums",
        "budget backpacking trips",
        "luxury hotel stays",
        "family-friendly activities"
    ]
    social_algorithm_params: Dict[str, Any] = {
        'n_clusters': 2,
        'random_state': 42,
        'eps': 0.5,
        'min_samples': 2
    }
    
    @validator('log_level')
    def validate_log_level(cls, value):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if value not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return value
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True

class AsyncConfigManager:
    """Async context manager for resource-intensive components"""
    def __init__(self):
        self.settings = Settings()
        self._components = {}
        self._validate_directories()
        
    async def __aenter__(self):
        """Async initialization of heavy resources"""
        await self._load_models_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        await self._unload_models_async()
        
    def _validate_directories(self):
        """Ensure required directories exist"""
        self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings.test_image_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def _load_models_async(self):
        """Async model loading with progress tracking"""
        logger.info("Async loading of ML models...")
        self._components['nlp'] = await self._load_spacy_model()
        self._components['sentiment'] = await self._load_sentiment_model()
        self._components['translator'] = Translator()
        
    async def _unload_models_async(self):
        """Async model unloading"""
        logger.info("Unloading ML models...")
        for component in self._components.values():
            if hasattr(component, '_model') and hasattr(component._model, 'close'):
                await component._model.close()
                
    async def _load_spacy_model(self):
        """Async load spaCy model with fallback"""
        try:
            return spacy.load(
                self.settings.default_spacy_model,
                disable=["parser", "ner"]
            )
        except OSError:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download(self.settings.default_spacy_model)
            return spacy.load(
                self.settings.default_spacy_model,
                disable=["parser", "ner"]
            )
            
    async def _load_sentiment_model(self):
        """Async load sentiment analysis model"""
        return pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=0 if spacy.prefer_gpu() else -1
        )
        
    @property
    def nlp(self):
        return self._components.get('nlp')
        
    @property
    def sentiment_analyzer(self):
        return self._components.get('sentiment')
        
    @property
    def translator(self):
        return self._components.get('translator')

# Async configuration initialization
async def get_config() -> AsyncConfigManager:
    """Async configuration factory"""
    try:
        async with AsyncConfigManager() as config:
            yield config
    except ValidationError as e:
        logger.critical(f"Configuration error: {str(e)}")
        raise RuntimeError("Invalid configuration") from e
