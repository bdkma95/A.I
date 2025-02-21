# config.py
import os
import logging
import spacy
from transformers import pipeline
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables and API keys
API_KEYS = {
    "safetipin": os.getenv("SAFETIPIN_API_KEY"),
    "openweather": os.getenv("OPENWEATHER_API_KEY")
}

# Global initializations
NLP_MODEL = spacy.load("en_core_web_sm")
SENTIMENT_ANALYZER = pipeline(
    "text-classification", 
    model="cardiffnlp/twitter-roberta-base-sentiment"
)
TRANSLATOR = Translator()
