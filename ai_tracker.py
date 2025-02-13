import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import boto3
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gym
from stable_baselines3 import PPO
import joblib
import streamlit as st
import plotly.express as px
from cryptography.fernet import Fernet
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import start_http_server, Counter, Gauge
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, List
import logging
from psycopg2.pool import SimpleConnectionPool
from celery import Celery
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Generate encryption key
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Encrypt sensitive data
encrypted_db_pw = cipher_suite.encrypt(os.getenv("DB_PASSWORD").encode())
decrypted_pw = cipher_suite.decrypt(encrypted_db_pw).decode()

# Constants
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": decrypted_pw,
    "host": os.getenv("POSTGRES_HOST"),
}
AWS_CONFIG = {
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "bucket_name": os.getenv("AWS_BUCKET_NAME"),
}

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
Swagger(app)

# Initialize Prometheus metrics
API_REQUESTS = Counter('api_requests_total', 'Total API requests')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total predictions made')

# Initialize Celery for async tasks
celery = Celery('tasks', broker='redis://localhost:6379/0')

# Initialize PostgreSQL connection pool
postgres_pool = SimpleConnectionPool(1, 10, **POSTGRES_CONFIG)

# Initialize ThreadPoolExecutor for concurrent tasks
executor = ThreadPoolExecutor(max_workers=10)

# ---------------------
# Data Models
# ---------------------

class PlayerData(BaseModel):
    player_id: int
    goals: int
    assists: int
    minutes_played: int

# ---------------------
# Data Collection Layer
# ---------------------

class DataCollector:
    def __init__(self, football_data_api_key: str, openweathermap_api_key: str):
        self.football_data_api_key = football_data_api_key
        self.openweathermap_api_key = openweathermap_api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_football_data(self) -> pd.DataFrame:
        """Fetch match data from Football-Data.org API."""
        url = "https://api.football-data.org/v4/matches"
        headers = {"X-Auth-Token": self.football_data_api_key}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            API_REQUESTS.inc()
            return pd.DataFrame(response.json()['matches'])
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            raise

    @lru_cache(maxsize=128)
    def fetch_weather_data(self, city: str = "London") -> Dict[str, float]:
        """Fetch weather data from OpenWeatherMap API."""
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.openweathermap_api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            weather_data = response.json()
            return {
                "temperature": weather_data['main']['temp'],
                "humidity": weather_data['main']['humidity'],
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API Error: {str(e)}")
            raise

# ---------------------
# Data Storage Layer
# ---------------------

class DataStorage:
    def __init__(self, postgres_config: Dict, aws_config: Dict):
        self.postgres_config = postgres_config
        self.aws_config = aws_config

    def save_to_postgres(self, data: pd.DataFrame, table_name: str) -> None:
        """Save data to PostgreSQL using connection pooling."""
        conn = postgres_pool.getconn()
        cur = conn.cursor()
        try:
            columns = data.columns.tolist()
            template = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({','.join(['%s']*len(columns))})"
            execute_batch(cur, template, data.values.tolist())
            conn.commit()
        except Exception as e:
            logger.error(f"Database Error: {str(e)}")
            conn.rollback()
        finally:
            cur.close()
            postgres_pool.putconn(conn)

    def upload_to_s3(self, file_name: str, bucket_name: str, object_name: Optional[str] = None) -> None:
        """Upload a file to AWS S3."""
        if object_name is None:
            object_name = os.path.basename(file_name)
        s3 = boto3.client('s3', **self.aws_config)
        try:
            s3.upload_file(file_name, bucket_name, object_name)
            logger.info(f"File {file_name} uploaded to S3 bucket {bucket_name} as {object_name}.")
        except Exception as e:
            logger.error(f"S3 Upload Error: {str(e)}")

# ---------------------
# Data Processing Layer
# ---------------------

class DataProcessor:
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the data."""
        data['xg_contribution'] = data['expected_goals'] + data['expected_assists']
        data['pass_accuracy'] = data['successful_passes'] / data['total_passes']
        data.fillna(0, inplace=True)
        return data

    def train_injury_prediction_model(self, data: pd.DataFrame) -> RandomForestClassifier:
        """Train a Random Forest model to predict injury risk."""
        X = data[['distance_covered', 'sprint_speed', 'tackle_success_rate']]
        y = data['injury_risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def train_player_rating_model(self, data: pd.DataFrame) -> RandomForestRegressor:
        """Train a regression model to predict player ratings."""
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
        X = data[features]
        y = data['player_rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Model MSE: {mse:.2f}")
        return model

# ---------------------
# API Endpoints
# ---------------------

@app.route('/passing-suggestions', methods=['POST'])
@limiter.limit("10 per minute")
def passing_suggestions():
    """API endpoint for passing suggestions."""
    try:
        data = PlayerData(**request.json)
        insights = {"message": "Improve passing accuracy under pressure."}
        return jsonify(insights), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

# ---------------------
# Main Execution
# ---------------------

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(8000)

    # Initialize components
    data_collector = DataCollector(FOOTBALL_DATA_API_KEY, OPENWEATHERMAP_API_KEY)
    data_storage = DataStorage(POSTGRES_CONFIG, AWS_CONFIG)
    data_processor = DataProcessor()

    # Fetch and process data
    matches_data = data_collector.fetch_football_data()
    processed_data = data_processor.process_data(matches_data)

    # Save data to PostgreSQL and S3
    data_storage.save_to_postgres(processed_data, "matches")
    data_storage.upload_to_s3("processed_matches_data.csv", AWS_CONFIG['bucket_name'])

    # Train machine learning models
    player_data = pd.read_csv("player_data.csv")
    injury_model = data_processor.train_injury_prediction_model(player_data)
    player_rating_model = data_processor.train_player_rating_model(player_data)

    # Run Flask app with SocketIO
    socketio.run(app, debug=True)
