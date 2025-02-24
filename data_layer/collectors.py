import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger
from config.settings import settings

class FootballDataCollector:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_football_data(self):
        """Fetch match data from Football-Data.org API"""
        headers = {"X-Auth-Token": settings.FOOTBALL_DATA_API_KEY}
        response = requests.get("https://api.football-data.org/v4/matches", headers=headers)
        response.raise_for_status()
        return pd.DataFrame(response.json()['matches'])

class WeatherDataCollector:
    def fetch_weather_data(self, city="London"):
        """Fetch weather data from OpenWeatherMap API"""
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={settings.OPENWEATHERMAP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
