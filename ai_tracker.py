import requests
import pandas as pd
import psycopg2
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gym
from stable_baselines3 import PPO
import joblib
import streamlit as st
import plotly.express as px

# Constants
FOOTBALL_DATA_API_KEY = "your_football_data_api_key"
OPENWEATHERMAP_API_KEY = "your_openweathermap_api_key"
POSTGRES_CONFIG = {
    "dbname": "your_dbname",
    "user": "your_username",
    "password": "your_password",
    "host": "your_host"
}
AWS_CONFIG = {
    "aws_access_key_id": "your_access_key",
    "aws_secret_access_key": "your_secret_key",
    "bucket_name": "your_bucket_name"
}

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# ---------------------
# Data Collection Layer
# ---------------------

class DataCollector:
    def __init__(self, football_data_api_key, openweathermap_api_key):
        self.football_data_api_key = football_data_api_key
        self.openweathermap_api_key = openweathermap_api_key

    def fetch_football_data(self):
        """Fetch match data from Football-Data.org API."""
        url = "https://api.football-data.org/v4/matches"
        headers = {"X-Auth-Token": self.football_data_api_key}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return pd.DataFrame(response.json()['matches'])
        else:
            raise Exception(f"Failed to fetch football data. Status Code: {response.status_code}")

    def fetch_weather_data(self, city="London"):
        """Fetch weather data from OpenWeatherMap API."""
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.openweathermap_api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            weather_data = response.json()
            return {
                "temperature": weather_data['main']['temp'],
                "humidity": weather_data['main']['humidity']
            }
        else:
            raise Exception(f"Failed to fetch weather data. Status Code: {response.status_code}")

# ---------------------
# Data Storage Layer
# ---------------------

class DataStorage:
    def __init__(self, postgres_config, aws_config):
        self.postgres_config = postgres_config
        self.aws_config = aws_config

    def save_to_postgres(self, data, table_name="players"):
        """Save data to a PostgreSQL database."""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cur = conn.cursor()
            
            # Example: Insert data into a table
            for _, row in data.iterrows():
                cur.execute(
                    f"INSERT INTO {table_name} (player_id, goals, assists) VALUES (%s, %s, %s)",
                    (row['player_id'], row['goals'], row['assists'])
                )
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")

    def upload_to_s3(self, file_name, bucket_name, object_name=None):
        """Upload a file to AWS S3."""
        if object_name is None:
            object_name = os.path.basename(file_name)
        
        s3 = boto3.client('s3', **self.aws_config)
        try:
            s3.upload_file(file_name, bucket_name, object_name)
            print(f"File {file_name} uploaded to S3 bucket {bucket_name} as {object_name}.")
        except Exception as e:
            print(f"Error uploading to S3: {e}")

# ---------------------
# Data Processing Layer
# ---------------------

class DataProcessor:
    def process_data(self, data):
        """Process and clean the data."""
        data['xg_contribution'] = data['expected_goals'] + data['expected_assists']
        data['pass_accuracy'] = data['successful_passes'] / data['total_passes']
        data.fillna(0, inplace=True)  # Handle missing values
        return data

    def train_injury_prediction_model(self, data):
        """Train a Random Forest model to predict injury risk."""
        X = data[['distance_covered', 'sprint_speed', 'tackle_success_rate']]
        y = data['injury_risk']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        
        return model

    def train_player_rating_model(self, data):
        """Train a regression model to predict player ratings."""
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
        X = data[features]
        y = data['player_rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        
        return model

    def cluster_players(self, data):
        """Cluster players based on performance metrics."""
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score']
        X = data[features]
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['cluster'] = kmeans.fit_predict(X)
        
        # Visualize clusters
        plt.scatter(X['goals'], X['assists'], c=data['cluster'], cmap='viridis')
        plt.xlabel('Goals')
        plt.ylabel('Assists')
        plt.title('Player Clusters')
        plt.show()
        
        return data

    def calculate_pass_accuracy_under_pressure(self, data):
        """Calculate pass accuracy under pressure."""
        data['pass_accuracy_under_pressure'] = (
            data['successful_passes_under_pressure'] / data['total_passes_under_pressure']
        )
        data['pass_accuracy_under_pressure'].fillna(0, inplace=True)  # Handle division by zero
        return data

    def train_xg_model(self, shot_data):
        """Train a simple xG model using shot data."""
        X = shot_data[['distance_to_goal', 'angle', 'body_part']]
        y = shot_data['goal']  # 1 if goal, 0 otherwise
        
        # Encode categorical features
        X = pd.get_dummies(X, columns=['body_part'], drop_first=True)
        
        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(X, y)
        
        return model

    def calculate_xg(self, data, xg_model):
        """Calculate expected goals (xG) for each shot."""
        X = data[['distance_to_goal', 'angle', 'body_part']]
        X = pd.get_dummies(X, columns=['body_part'], drop_first=True)
        data['xg'] = xg_model.predict_proba(X)[:, 1]  # Probability of being a goal
        return data

    def calculate_defensive_score(self, data):
        """Calculate a defensive contribution score."""
        data['defensive_score'] = (
            data['interceptions'] + data['clearances'] + data['blocks']
        ) / data['minutes_played']
        data['defensive_score'].fillna(0, inplace=True)  # Handle division by zero
        return data

    def train_movement_model(self, movement_data):
        """Train a neural network to analyze player movement."""
        X = movement_data[['x', 'y', 'speed', 'acceleration']]
        y = movement_data['outcome']  # Target variable (e.g., successful pass, shot, etc.)
        
        # Build a simple neural network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        
        return model

    def train_reinforcement_learning_agent(self):
        """Train a reinforcement learning agent to simulate player decision-making."""
        env = gym.make('CustomFootballEnv-v0')
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10000)
        return model

    def retrain_model(self, new_data_path="new_player_data.csv", model_path="player_rating_model.pkl"):
        """Retrain the player rating model with new data."""
        new_data = pd.read_csv(new_data_path)
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
        X = new_data[features]
        y = new_data['player_rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"New Model Mean Squared Error: {mse:.2f}")
        
        joblib.dump(model, model_path)

    def monitor_model_performance(self, test_data_path="test_data.csv", model_path="player_rating_model.pkl", threshold=5.0):
        """Monitor the performance of the deployed model."""
        test_data = pd.read_csv(test_data_path)
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
        X = test_data[features]
        y = test_data['player_rating']
        
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f"Current Model Mean Squared Error: {mse:.2f}")
        
        if mse > threshold:
            print("Warning: Model performance has degraded significantly!")

    def identify_undervalued_players(self, performance_data, market_value_data):
        """Identify undervalued players based on performance metrics and market value."""
        data = pd.merge(performance_data, market_value_data, on="player_id")
        
        # Normalize performance metrics
        for col in ['goals', 'assists', 'defensive_score']:
            data[f'normalized_{col}'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        
        # Calculate value score
        data['value_score'] = (
            data['normalized_goals'] + data['normalized_assists'] + data['normalized_defensive_score']
        ) / data['market_value']
        
        # Rank players by value score
        return data.sort_values(by="value_score", ascending=False)

    def train_fantasy_points_model(self, data):
        """Train a regression model to predict fantasy points."""
        features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
        X = data[features]
        y = data['fantasy_points']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        
        return model

    def recommend_players(self, model, player_data):
        """Recommend players based on predicted fantasy points."""
        predictions = model.predict(player_data)
        player_data['predicted_fantasy_points'] = predictions
        return player_data.sort_values(by="predicted_fantasy_points", ascending=False)

# ---------------------
# Analytics & Insights Layer
# ---------------------

@app.route('/passing-suggestions', methods=['POST'])
def passing_suggestions():
    """API endpoint for passing suggestions."""
    data = request.json
    insights = {"message": "Improve passing accuracy under pressure."}
    return jsonify(insights)

def check_fatigue_risk(player_data):
    """Check for fatigue risk based on distance covered."""
    if player_data['distance_covered'] > 12000:  # Example threshold
        return "High fatigue risk detected!"
    return None

@app.route('/')
def index():
    """Render the dashboard homepage."""
    return render_template('index.html')

@socketio.on('update')
def handle_update(data):
    """Handle real-time updates via WebSocket."""
    socketio.emit('real_time_update', data)

# ---------------------
# Visualization Layer
# ---------------------

def generate_positioning_heatmap(tracking_data):
    """Generate a heatmap for player positioning."""
    fig = px.density_heatmap(
        tracking_data, x='x', y='y', nbinsx=20, nbinsy=20,
        title="Player Positioning Heatmap"
    )
    fig.show()

def analyze_movement_patterns(tracking_data):
    """Analyze movement patterns (e.g., passing lanes, defensive coverage)."""
    avg_speed = tracking_data['speed'].mean()
    total_distance = tracking_data['distance_covered'].sum()
    
    print(f"Average Speed: {avg_speed:.2f} m/s")
    print(f"Total Distance Covered: {total_distance:.2f} meters")

# ---------------------
# Main Execution
# ---------------------

if __name__ == '__main__':
    # Initialize components
    data_collector = DataCollector(FOOTBALL_DATA_API_KEY, OPENWEATHERMAP_API_KEY)
    data_storage = DataStorage(POSTGRES_CONFIG, AWS_CONFIG)
    data_processor = DataProcessor()

    # Fetch and process data
    matches_data = data_collector.fetch_football_data()
    processed_data = data_processor.process_data(matches_data)
    
    # Save data to PostgreSQL and S3
    data_storage.save_to_postgres(processed_data)
    data_storage.upload_to_s3("processed_matches_data.csv", AWS_CONFIG['bucket_name'])
    
    # Train machine learning models
    player_data = pd.read_csv("player_data.csv")
    injury_model = data_processor.train_injury_prediction_model(player_data)
    player_rating_model = data_processor.train_player_rating_model(player_data)
    
    # Run Flask app with SocketIO
    socketio.run(app, debug=True)
