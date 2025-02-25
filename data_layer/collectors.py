import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from pyrate_limiter import Duration, Rate, Limiter
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache
from datetime import datetime
import hashlib
import json
import websockets
import asyncio
from graphql import parse, print_ast
from utils.logger import logger
from config.settings import settings
from monitoring.metrics import API_REQUESTS, REQUEST_DURATION
from utils.decorators import circuit_breaker
from cryptography.fernet import Fernet
from concurrent.futures import ThreadPoolExecutor
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Rate limiting: 10 requests/minute
API_RATE_LIMITER = Limiter(Rate(10, Duration.MINUTE))

class APIRateLimitExceeded(Exception): pass
class DataValidationError(Exception): pass
class SchemaVersionMismatch(Exception): pass

# ======================
# Versioned Schemas
# ======================
class MatchDataV1(BaseModel):
    id: int
    utcDate: datetime
    homeTeam: Dict[str, str] = Field(..., alias="homeTeam")
    awayTeam: Dict[str, str] = Field(..., alias="awayTeam")
    score: Optional[Dict[str, int]]
    schema_version: str = "1.0.0"

class MatchDataV2(MatchDataV1):
    venue: Optional[Dict[str, str]]
    officials: List[Dict[str, str]]
    schema_version: str = "2.0.0"

class WeatherResponseV1(BaseModel):
    temp: float
    humidity: float
    wind_speed: float = Field(..., alias="wind.speed")
    conditions: str = Field(..., alias="weather.0.description")
    schema_version: str = "1.0.0"

    @validator('temp')
    def validate_temp(cls, v):
        if not (-50 <= v <= 60):
            raise ValueError("Invalid temperature value")
        return v

# ======================
# OAuth2 Authentication
# ======================
class OAuth2Authenticator:
    def __init__(self, token_url: str, client_id: str, client_secret: str):
        self.token_url = token_url
        self.client = BackendApplicationClient(client_id=client_id)
        self.oauth = OAuth2Session(client=self.client)
        self.client_secret = client_secret
        self.token = None

    def get_token(self):
        if not self.token or self.token.expired:
            self.token = self.oauth.fetch_token(
                token_url=self.token_url,
                client_id=self.client.client_id,
                client_secret=self.client_secret
            )
        return self.token

# ======================
# GraphQL Client
# ======================
class GraphQLClient:
    def __init__(self, endpoint: str, authenticator: OAuth2Authenticator):
        self.endpoint = endpoint
        self.authenticator = authenticator

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def execute_query(self, query: str, variables: Dict = None) -> Dict:
        parsed_query = parse(query)
        return self._send_gql_request(print_ast(parsed_query), variables)

    def _send_gql_request(self, query: str, variables: Dict) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.authenticator.get_token()['access_token']}",
            "Content-Type": "application/json"
        }
        payload = {'query': query, 'variables': variables or {}}
        
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

# ======================
# Streaming Handler
# ======================
class FootballStreamHandler:
    def __init__(self, uri: str, callback: callable):
        self.uri = uri
        self.callback = callback
        self.running = False

    async def _connect(self):
        return await websockets.connect(self.uri)

    async def start(self):
        self.running = True
        while self.running:
            try:
                async with self._connect() as websocket:
                    async for message in websocket:
                        await self.callback(json.loads(message))
            except Exception as e:
                logger.error(f"Stream connection error: {str(e)}")
                await asyncio.sleep(5)

    def stop(self):
        self.running = False

# ======================
# Enhanced Collectors
# ======================
class FootballDataCollector:
    SCHEMA_VERSIONS = {"1.0.0": MatchDataV1, "2.0.0": MatchDataV2}

    def __init__(self, authenticator: OAuth2Authenticator):
        self.circuit_state = {"failures": 0, "last_failure": None}
        self.request_cache = {}
        self.authenticator = authenticator
        self.graphql = GraphQLClient(
            endpoint=settings.FOOTBALL_GRAPHQL_ENDPOINT,
            authenticator=authenticator
        )

    @circuit_breaker(max_failures=3, reset_timeout=60)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @API_RATE_LIMITER.ratelimit("football_api", delay=True)
    def fetch_football_data(self, competition_code: str = "PL", version: str = "2.0.0") -> pd.DataFrame:
        """Fetch data with schema versioning support"""
        cache_key = f"matches_{competition_code}_{datetime.now().date()}_{version}"
        if cached := self.request_cache.get(cache_key):
            return cached

        try:
            token = self.authenticator.get_token()
            headers = {
                "Authorization": f"Bearer {token['access_token']}",
                "X-API-Version": version
            }
            
            # Rest of implementation remains similar but uses versioned models
            # ...
            
            return self._process_with_version(data, version)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            self._update_circuit_state()
            raise

    def _process_with_version(self, data: Dict, version: str) -> pd.DataFrame:
        model = self.SCHEMA_VERSIONS.get(version)
        if not model:
            raise SchemaVersionMismatch(f"Unsupported version {version}")
            
        matches = [model(**m).dict() for m in data.get("matches", [])]
        return pd.DataFrame(matches)

    def stream_live_matches(self, callback: callable):
        """Start streaming live match updates"""
        handler = FootballStreamHandler(
            uri=settings.FOOTBALL_WS_ENDPOINT,
            callback=callback
        )
        asyncio.run(handler.start())
