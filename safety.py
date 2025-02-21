import requests
from transformers import pipeline
from typing import Optional, Dict
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class SafetyAnalyzer:
    """
    Analyzes safety risks using geospatial data and sentiment analysis
    """
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "text-classification", 
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )

    def get_safety_data(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            response = requests.get(
                f"https://api.safetipin.com/safe?lat={lat}&lon={lon}",
                headers={"Authorization": f"Bearer {settings.safetipin_api_key}"},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Safety API error: {str(e)}")
            return None

    def analyze_risk(self, text: str) -> Dict:
        try:
            result = self.sentiment_analyzer(text)
            return {"risk_level": "high" if result[0]['label'] == "negative" else "low"}
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {"risk_level": "unknown"}
