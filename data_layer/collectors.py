import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from pyrate_limiter import Duration, Rate, Limiter
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, List, Tuple
from functools import lru_cache
from datetime import datetime
import hashlib
from utils.logger import logger
from config.settings import settings
from monitoring.metrics import API_REQUESTS, REQUEST_DURATION
from utils.decorators import circuit_breaker
from cryptography.fernet import Fernet

# Rate limiting: 10 requests/minute
API_RATE_LIMITER = Limiter(Rate(10, Duration.MINUTE))

class APIRateLimitExceeded(Exception): pass
class DataValidationError(Exception): pass

class MatchData(BaseModel):
    id: int
    utcDate: datetime
    homeTeam: Dict[str, str]
    awayTeam: Dict[str, str]
    score: Optional[Dict[str, int]]

class WeatherResponse(BaseModel):
    temp: float
    humidity: float
    wind_speed: float
    conditions: str

class FootballDataCollector:
    def __init__(self):
        self.circuit_state = {"failures": 0, "last_failure": None}
        self.request_cache = {}

    @circuit_breaker(max_failures=3, reset_timeout=60)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @API_RATE_LIMITER.ratelimit("football_api", delay=True)
    def fetch_football_data(self, competition_code: str = "PL") -> pd.DataFrame:
        """Fetch paginated match data with validation and caching"""
        cache_key = f"matches_{competition_code}_{datetime.now().date()}"
        if cached := self.request_cache.get(cache_key):
            return cached

        try:
            with REQUEST_DURATION.labels(endpoint="matches").time():
                headers = {
                    "X-Auth-Token": settings.FOOTBALL_DATA_API_KEY,
                    "X-Response-Control": "minified"
                }
                params = {"competitions": competition_code}
                
                logger.info(f"Fetching matches for competition {competition_code}")
                response = requests.get(
                    "https://api.football-data.org/v4/matches",
                    headers=headers,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                
                API_REQUESTS.labels(api="football").inc()
                data = self._validate_and_process(response.json())
                
                self.request_cache[cache_key] = data
                return data
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            self._update_circuit_state()
            raise
        except ValidationError as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise DataValidationError from e

    def _validate_and_process(self, raw_data: Dict) -> pd.DataFrame:
        """Validate API response structure and process data"""
        try:
            matches = [MatchData(**m).dict() for m in raw_data.get("matches", [])]
            df = pd.DataFrame(matches)
            
            # Add derived features
            if not df.empty:
                df['match_date'] = pd.to_datetime(df['utcDate'])
                df['total_goals'] = df['score'].apply(lambda x: x.get('fullTime', {}).get('home', 0) + 
                                                     x.get('fullTime', {}).get('away', 0))
            return df
            
        except KeyError as e:
            logger.error(f"Missing key in API response: {str(e)}")
            raise DataValidationError from e

class WeatherDataCollector:
    _CACHE_SIZE = 100  # Store last 100 requests
    _ENCRYPTION_KEY = Fernet.generate_key()

    def __init__(self):
        self.circuit_state = {"failures": 0, "last_failure": None}

    @circuit_breaker(max_failures=5, reset_timeout=300)
    @lru_cache(maxsize=_CACHE_SIZE)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_weather_data(self, city: str = "London", country: str = "UK") -> WeatherResponse:
        """Fetch encrypted weather data with caching and validation"""
        cache_key = self._generate_cache_key(city, country)
        logger.debug(f"Weather request for {city}, {country}")

        try:
            with REQUEST_DURATION.labels(endpoint="weather").time():
                url = (
                    f"http://api.openweathermap.org/data/2.5/weather?"
                    f"q={city},{country}&appid={settings.OPENWEATHERMAP_API_KEY}&units=metric"
                )
                
                response = requests.get(url, timeout=8)
                response.raise_for_status()
                
                API_REQUESTS.labels(api="weather").inc()
                return self._process_weather_data(response.json())
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API failed: {str(e)}")
            self._update_circuit_state()
            raise
        except ValidationError as e:
            logger.error(f"Invalid weather data: {str(e)}")
            raise DataValidationError from e

    def _process_weather_data(self, raw_data: Dict) -> WeatherResponse:
        """Process and validate weather data"""
        encrypted_data = self._encrypt_sensitive(raw_data)
        return WeatherResponse(
            temp=encrypted_data['main']['temp'],
            humidity=encrypted_data['main']['humidity'],
            wind_speed=encrypted_data['wind']['speed'],
            conditions=encrypted_data['weather'][0]['description']
        )

    def _generate_cache_key(self, city: str, country: str) -> str:
        """Generate SHA256 cache key"""
        return hashlib.sha256(f"{city}_{country}".encode()).hexdigest()

    def _encrypt_sensitive(self, data: Dict) -> Dict:
        """Encrypt sensitive weather measurements"""
        cipher = Fernet(self._ENCRYPTION_KEY)
        if 'temp' in data.get('main', {}):
            data['main']['temp'] = cipher.encrypt(str(data['main']['temp']).encode()).decode()
        return data

    def batch_fetch_weather(self, locations: List[Tuple[str, str]]) -> Dict:
        """Fetch weather data for multiple locations in parallel"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.fetch_weather_data, city, country): (city, country)
                for city, country in locations
            }
            return {
                future.result().location: future.result()
                for future in as_completed(futures)
            }
