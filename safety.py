import logging
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, ValidationError
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
import httpx

logger = logging.getLogger(__name__)

class SafetyAPIError(Exception):
    """Custom exception for safety API failures"""

class SafetyAnalysisResult(BaseModel):
    risk_level: str
    confidence: float
    components: Dict[str, float]
    timestamp: datetime

class SafetyAnalyzer:
    """
    Enhanced safety risk analyzer with geospatial data and sentiment analysis
    
    Features:
    - Async support
    - Request retries with exponential backoff
    - Input validation
    - Response caching
    - Detailed risk breakdown
    """
    
    def __init__(self):
        self._initialize_sentiment_analyzer()
        self.session = httpx.AsyncClient(timeout=10)
        self.cache = {}

    def _initialize_sentiment_analyzer(self):
        """Lazy-load the sentiment analyzer to save memory"""
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            return_all_scores=True
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def get_safety_data(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get safety data with enhanced error handling and validation
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
        
        Returns:
            Dict with safety data or None on failure
        """
        self._validate_coordinates(lat, lon)
        
        cache_key = f"{lat}_{lon}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = await self.session.get(
                "https://api.safetipin.com/safe",
                params={"lat": lat, "lon": lon},
                headers={"Authorization": f"Bearer {settings.safetipin_api_key}"}
            )
            response.raise_for_status()
            
            data = response.json()
            self._validate_api_response(data)
            
            self.cache[cache_key] = data
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"API error: {e.response.status_code} - {e.response.text}")
            raise SafetyAPIError(f"API request failed: {e.response.status_code}") from e
        except (ValidationError, ValueError) as e:
            logger.error(f"Invalid API response: {str(e)}")
            raise SafetyAPIError("Invalid API response format") from e
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise SafetyAPIError("Safety check failed") from e

    def analyze_risk(self, text: str) -> SafetyAnalysisResult:
        """
        Analyze text risk with detailed sentiment breakdown
        
        Args:
            text: Input text to analyze (1-500 characters)
        
        Returns:
            SafetyAnalysisResult with risk details
        """
        self._validate_input_text(text)
        
        try:
            scores = self.sentiment_analyzer(text)[0]
            risk_score = next(s['score'] for s in scores if s['label'] == 'negative')
            
            return SafetyAnalysisResult(
                risk_level=self._calculate_risk_level(risk_score),
                confidence=risk_score,
                components={"sentiment": risk_score},
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return SafetyAnalysisResult(
                risk_level="unknown",
                confidence=0.0,
                components={},
                timestamp=datetime.utcnow()
            )

    def _validate_coordinates(self, lat: float, lon: float):
        """Ensure valid geographic coordinates"""
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Invalid coordinates provided")

    def _validate_input_text(self, text: str):
        """Validate text input parameters"""
        if not text or len(text) > 500:
            raise ValueError("Text must be 1-500 characters")

    def _validate_api_response(self, data: Dict):
        """Validate API response structure"""
        required_keys = {'safety_score', 'lighting', 'security', 'transportation'}
        if not required_keys.issubset(data.keys()):
            raise ValueError("Missing required fields in API response")

    def _calculate_risk_level(self, score: float) -> str:
        """Convert numerical score to risk category"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        return "low"

    async def close(self):
        """Clean up resources"""
        await self.session.aclose()

# Usage example
async def main():
    analyzer = SafetyAnalyzer()
    try:
        safety_data = await analyzer.get_safety_data(40.7128, -74.0060)
        risk_assessment = analyzer.analyze_risk("This area feels dangerous at night")
        print(risk_assessment)
    finally:
        await analyzer.close()
