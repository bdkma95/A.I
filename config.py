# config.py
import os
import logging
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseSettings, Field, ValidationError
import spacy
from spacy.util import get_lang_class
from transformers import pipeline
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings with environment variable validation
    
    Uses pydantic BaseSettings for automatic environment variable parsing
    """
    safetipin_api_key: str = Field(..., env="SAFETIPIN_API_KEY")
    openweather_api_key: str = Field(..., env="OPENWEATHER_API_KEY")
    model_cache_dir: Path = Path("./models")
    default_spacy_model: str = "en_core_web_sm"
    max_text_length: int = 5000
    translation_timeout: int = 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class Config:
    """
    Central configuration class with lazy-loaded components
    
    Implements singleton pattern for resource efficiency
    """
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration with validation"""
        try:
            self.settings = Settings()
            self._validate_directories()
            self._components = {
                'nlp': None,
                'sentiment': None,
                'translator': None
            }
            logger.info("Configuration initialized successfully")
        except ValidationError as e:
            logger.critical(f"Configuration error: {str(e)}")
            raise RuntimeError("Invalid configuration") from e
    
    def _validate_directories(self):
        """Ensure required directories exist"""
        self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def nlp(self):
        """Lazy-load spaCy NLP model"""
        if self._components['nlp'] is None:
            try:
                logger.info("Loading spaCy model...")
                self._components['nlp'] = spacy.load(
                    self.settings.default_spacy_model,
                    disable=["parser", "ner"]
                )
            except OSError:
                logger.warning("spaCy model not found, downloading...")
                spacy.cli.download(self.settings.default_spacy_model)
                self._components['nlp'] = spacy.load(
                    self.settings.default_spacy_model,
                    disable=["parser", "ner"]
                )
        return self._components['nlp']
    
    @property
    def sentiment_analyzer(self):
        """Lazy-load sentiment analysis pipeline"""
        if self._components['sentiment'] is None:
            logger.info("Initializing sentiment analyzer...")
            self._components['sentiment'] = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=0 if spacy.prefer_gpu() else -1
            )
        return self._components['sentiment']
    
    @property
    def translator(self):
        """Lazy-load translation component"""
        if self._components['translator'] is None:
            logger.info("Initializing translator...")
            self._components['translator'] = Translator()
        return self._components['translator']
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get supported translation languages"""
        return get_lang_class('en').Defaults.languages

# Initialize configuration
try:
    config = Config()
except RuntimeError as e:
    logger.critical("Failed to initialize configuration")
    raise
