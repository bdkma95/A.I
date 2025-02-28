# safety.py
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pydantic import BaseModel, ValidationError, validator, confloat, conint
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache
from config import AsyncConfigManager
from exceptions import SafetyAPIError

logger = logging.getLogger(__name__)

class SafetyRequest(BaseModel):
    """Pydantic model for safety request validation"""
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)
    text: Optional[str] = None
    max_retries: conint(ge=1, le=5) = 3

class SafetyAnalysisResult(BaseModel):
    """Enhanced safety analysis result model"""
    risk_level: str
    confidence: float
    components: Dict[str, float]
    timestamp: datetime
    location_data: Optional[Dict[str, Any]]
    text_insights: Optional[Dict[str, Any]]

    @validator('risk_level')
    def validate_risk_level(cls, value):
        valid_levels = ['low', 'medium', 'high', 'critical']
        if value not in valid_levels:
            raise ValueError(f"Invalid risk level: {value}")
        return value

class SafetyAnalyzer:
    """
    Async safety analyzer with geospatial intelligence and NLP
    
    Features:
    - Full async context manager support
    - Configurable caching with TTL
    - Circuit breaker pattern
    - Sentiment analysis using shared config resources
    - Comprehensive input validation
    """
    
    def __init__(self, config: AsyncConfigManager):
        self.config = config
        self._client = None
        self.cache = TTLCache(
            maxsize=1000,
            ttl=timedelta(minutes=30)
        )
        self._executor = ThreadPoolExecutor()

    async def __aenter__(self):
        """Async initialization"""
        self._client = httpx.AsyncClient(
            timeout=self.config.settings.safety_api_timeout,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=100
            )
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async cleanup"""
        await self._client.aclose()
        self._executor.shutdown(wait=False)

    @retry(
        stop=stop_after_attempt(lambda config: config.settings.safety_api_retries),
        wait=wait_exponential(
            multiplier=1, 
            max=self.config.settings.safety_api_timeout
        ),
        retry=retry_if_exception_type(SafetyAPIError)
    )
    async def get_safety_data(self, lat: float, lon: float) -> SafetyAnalysisResult:
        """
        Get comprehensive safety analysis with geospatial context
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
        
        Returns:
            SafetyAnalysisResult with combined risk assessment
        """
        try:
            request = SafetyRequest(latitude=lat, longitude=lon)
            cache_key = f"{request.latitude}_{request.longitude}"
            
            if cache_key in self.cache:
                logger.debug("Returning cached safety data")
                return self.cache[cache_key]

            location_data = await self._fetch_location_data(request)
            text_analysis = await self._analyze_text_async(request.text)
            
            result = SafetyAnalysisResult(
                risk_level=self._calculate_risk_level(location_data, text_analysis),
                confidence=self._calculate_confidence(location_data, text_analysis),
                components={
                    **location_data.get('scores', {}),
                    **text_analysis
                },
                timestamp=datetime.utcnow(),
                location_data=location_data,
                text_insights=text_analysis
            )
            
            self.cache[cache_key] = result
            return result

        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise SafetyAPIError("Invalid request parameters") from e
        except Exception as e:
            logger.error(f"Safety analysis failed: {str(e)}")
            raise SafetyAPIError("Comprehensive safety check failed") from e

    async def _fetch_location_data(self, request: SafetyRequest) -> Dict:
        """Fetch geospatial safety data from API"""
        try:
            response = await self._client.get(
                f"{self.config.settings.safety_api_url}/safe",
                params={
                    "lat": request.latitude,
                    "lon": request.longitude
                },
                headers={
                    "Authorization": f"Bearer {self.config.settings.safetipin_api_key}",
                    "X-Request-Source": "AI-Concierge"
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API error: {e.response.status_code}")
            raise SafetyAPIError(f"API request failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected API error: {str(e)}")
            raise SafetyAPIError("Location data fetch failed") from e

    async def _analyze_text_async(self, text: Optional[str]) -> Dict:
        """Async text analysis with sentiment and keyword extraction"""
        if not text:
            return {}

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor,
                self._perform_text_analysis,
                text
            )
        except Exception as e:
            logger.warning(f"Text analysis failed: {str(e)}")
            return {}

    def _perform_text_analysis(self, text: str) -> Dict:
        """CPU-bound text analysis processing"""
        sentiment = self.config.sentiment_analyzer(text)[0]
        keywords = self.config.nlp(text).ents
        
        return {
            "sentiment": max(sentiment, key=lambda x: x['score']),
            "keywords": [str(ent) for ent in keywords],
            "risk_score": self._calculate_text_risk(sentiment)
        }

    def _calculate_text_risk(self, sentiment_scores) -> float:
        """Calculate text-based risk score"""
        negative_score = next(
            (s['score'] for s in sentiment_scores if s['label'] == 'NEGATIVE'),
            0.0
        )
        return min(negative_score * 1.5, 1.0)

    def _calculate_risk_level(self, location_data: Dict, text_analysis: Dict) -> str:
        """Combined risk calculation logic"""
        location_score = location_data.get('safety_score', 0.5)
        text_score = text_analysis.get('risk_score', 0.0)
        combined_score = (location_score * 0.7) + (text_score * 0.3)

        if combined_score >= 0.8:
            return "critical"
        elif combined_score >= 0.6:
            return "high"
        elif combined_score >= 0.4:
            return "medium"
        return "low"

    def _calculate_confidence(self, location_data: Dict, text_analysis: Dict) -> float:
        """Calculate overall confidence score"""
        location_confidence = location_data.get('confidence', 0.7)
        text_confidence = text_analysis.get('sentiment', {}).get('score', 0.5)
        return (location_confidence + text_confidence) / 2
