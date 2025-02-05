import requests
import pandas as pd
import psycopg2
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
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

def fetch_football_data():
    """Fetch match data from Football-Data.org API."""
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return pd.DataFrame(response.json()['matches'])
    else:
        raise Exception(f"Failed to fetch football data. Status Code: {response.status_code}")

def fetch_weather_data(city="London"):
    """Fetch weather data from OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}"
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

def save_to_postgres(data, table_name="players"):
    """Save data to a PostgreSQL database."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
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

def upload_to_s3(file_name, bucket_name, object_name=None):
    """Upload a file to AWS S3."""
    if object_name is None:
        object_name = os.path.basename(file_name)
    
    s3 = boto3.client('s3', **AWS_CONFIG)
    try:
        s3.upload_file(file_name, bucket_name, object_name)
        print(f"File {file_name} uploaded to S3 bucket {bucket_name} as {object_name}.")
    except Exception as e:
        print(f"Error uploading to S3: {e}")

# ---------------------
# Data Processing Layer
# ---------------------

def process_data(data):
    """Process and clean the data."""
    data['xg_contribution'] = data['expected_goals'] + data['expected_assists']
    data['pass_accuracy'] = data['successful_passes'] / data['total_passes']
    data.fillna(0, inplace=True)  # Handle missing values
    return data

def train_injury_prediction_model(data):
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

# ---------------------
# Visualization Layer
# ---------------------

@app.route('/')
def index():
    """Render the dashboard homepage."""
    return render_template('index.html')

@socketio.on('update')
def handle_update(data):
    """Handle real-time updates via WebSocket."""
    socketio.emit('real_time_update', data)

# ---------------------
# Main Execution
# ---------------------

if __name__ == '__main__':
    # Fetch and process data
    matches_data = fetch_football_data()
    processed_data = process_data(matches_data)
    
    # Save data to PostgreSQL and S3
    save_to_postgres(processed_data)
    upload_to_s3("processed_matches_data.csv", AWS_CONFIG['bucket_name'])
    
    # Train machine learning model
    player_data = pd.read_csv("player_data.csv")
    train_injury_prediction_model(player_data)
    
    # Run Flask app with SocketIO
    socketio.run(app, debug=True)

def calculate_pass_accuracy_under_pressure(data):
    """
    Calculate pass accuracy under pressure.
    """
    data['pass_accuracy_under_pressure'] = (
        data['successful_passes_under_pressure'] / data['total_passes_under_pressure']
    )
    data['pass_accuracy_under_pressure'].fillna(0, inplace=True)  # Handle division by zero
    return data

def train_xg_model(shot_data):
    """
    Train a simple xG model using shot data.
    """
    # Features: distance_to_goal, angle, body_part (encoded), etc.
    X = shot_data[['distance_to_goal', 'angle', 'body_part']]
    y = shot_data['goal']  # 1 if goal, 0 otherwise
    
    # Encode categorical features
    X = pd.get_dummies(X, columns=['body_part'], drop_first=True)
    
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X, y)
    
    return model

def calculate_xg(data, xg_model):
    """
    Calculate expected goals (xG) for each shot.
    """
    X = data[['distance_to_goal', 'angle', 'body_part']]
    X = pd.get_dummies(X, columns=['body_part'], drop_first=True)
    data['xg'] = xg_model.predict_proba(X)[:, 1]  # Probability of being a goal
    return data

def calculate_defensive_score(data):
    """
    Calculate a defensive contribution score.
    """
    data['defensive_score'] = (
        data['interceptions'] + data['clearances'] + data['blocks']
    ) / data['minutes_played']
    data['defensive_score'].fillna(0, inplace=True)  # Handle division by zero
    return data

def train_player_rating_model(data):
    """
    Train a regression model to predict player ratings.
    """
    # Features: goals, assists, pass_accuracy, defensive_score, etc.
    features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
    X = data[features]
    y = data['player_rating']  # Target variable
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    
    return model

def train_injury_risk_model(data):
    """
    Train a classification model to predict injury risks.
    """
    # Features: distance_covered, sprint_speed, heart_rate, etc.
    features = ['distance_covered', 'sprint_speed', 'heart_rate', 'tackle_success_rate']
    X = data[features]
    y = data['injury_risk']  # Target variable (1 if injured, 0 otherwise)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

def cluster_players(data):
    """
    Cluster players based on performance metrics.
    """
    # Features: goals, assists, pass_accuracy, defensive_score, etc.
    features = ['goals', 'assists', 'pass_accuracy', 'defensive_score']
    X = data[features]
    
    # Use KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)
    
    # Visualize clusters
    plt.scatter(X['goals'], X['assists'], c=data['cluster'], cmap='viridis')
    plt.xlabel('Goals')
    plt.ylabel('Assists')
    plt.title('Player Clusters')
    plt.show()
    
    return data

def train_movement_model(movement_data):
    """
    Train a neural network to analyze player movement.
    """
    # Features: x, y coordinates, speed, acceleration, etc.
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

def train_reinforcement_learning_agent():
    """
    Train a reinforcement learning agent to simulate player decision-making.
    """
    # Create a custom environment (e.g., football simulation)
    env = gym.make('CustomFootballEnv-v0')
    
    # Use PPO algorithm
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    
    return model

# Load data
data = pd.read_csv("player_data.csv")

# Feature Engineering
data = calculate_pass_accuracy_under_pressure(data)
data = calculate_defensive_score(data)

# Train xG model
shot_data = pd.read_csv("shot_data.csv")
xg_model = train_xg_model(shot_data)
data = calculate_xg(data, xg_model)

# Train Player Rating Model
player_rating_model = train_player_rating_model(data)

# Train Injury Risk Model
injury_risk_model = train_injury_risk_model(data)

# Cluster Players
data = cluster_players(data)

# Train Movement Analysis Model
movement_data = pd.read_csv("movement_data.csv")
movement_model = train_movement_model(movement_data)

# Train Reinforcement Learning Agent
rl_model = train_reinforcement_learning_agent()

# Load the trained model
model = joblib.load("player_rating_model.pkl")

app = Flask(__name__)

@app.route('/predict-rating', methods=['POST'])
def predict_rating():
    """
    API endpoint to predict player ratings.
    """
    data = request.json
    input_data = pd.DataFrame([data])
    
    # Make predictions
    prediction = model.predict(input_data)
    
    return jsonify({"predicted_rating": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

# Title
st.title("Football Player Performance Dashboard")

# Fetch data from the Flask/FastAPI API
response = requests.post("http://localhost:5000/predict-rating", json={
    "goals": 5,
    "assists": 3,
    "pass_accuracy": 85,
    "defensive_score": 7.5,
    "xg_contribution": 4.2
})

if response.status_code == 200:
    prediction = response.json()["predicted_rating"]
    st.write(f"**Predicted Player Rating:** {prediction:.2f}")
else:
    st.write("Failed to fetch prediction.")

# Display raw data
data = pd.read_csv("player_data.csv")
st.write("### Player Data")
st.write(data)

def retrain_model():
    """
    Retrain the player rating model with new data.
    """
    # Load new data
    new_data = pd.read_csv("new_player_data.csv")
    
    # Features and target
    features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
    X = new_data[features]
    y = new_data['player_rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a new model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"New Model Mean Squared Error: {mse:.2f}")
    
    # Save the updated model
    joblib.dump(model, "player_rating_model.pkl")

# Schedule retraining (e.g., using cron or a task scheduler)
retrain_model()

def monitor_model_performance():
    """
    Monitor the performance of the deployed model.
    """
    # Load test data
    test_data = pd.read_csv("test_data.csv")
    
    # Features and target
    features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
    X = test_data[features]
    y = test_data['player_rating']
    
    # Load the deployed model
    model = joblib.load("player_rating_model.pkl")
    
    # Evaluate the model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Current Model Mean Squared Error: {mse:.2f}")
    
    # Set up alerts for significant changes
    if mse > 5.0:  # Example threshold
        print("Warning: Model performance has degraded significantly!")

# Schedule monitoring (e.g., daily)
monitor_model_performance()

def identify_undervalued_players(performance_data, market_value_data):
    """
    Identify undervalued players based on performance metrics and market value.
    """
    # Merge performance and market value data
    data = pd.merge(performance_data, market_value_data, on="player_id")
    
    # Normalize performance metrics
    data['normalized_goals'] = (data['goals'] - data['goals'].min()) / (data['goals'].max() - data['goals'].min())
    data['normalized_assists'] = (data['assists'] - data['assists'].min()) / (data['assists'].max() - data['assists'].min())
    data['normalized_defensive_score'] = (data['defensive_score'] - data['defensive_score'].min()) / (data['defensive_score'].max() - data['defensive_score'].min())
    
    # Calculate value score
    data['value_score'] = (
        data['normalized_goals'] + data['normalized_assists'] + data['normalized_defensive_score']
    ) / data['market_value']
    
    # Rank players by value score
    data = data.sort_values(by="value_score", ascending=False)
    
    return data[['player_id', 'player_name', 'value_score', 'market_value', 'goals', 'assists', 'defensive_score']]

def train_injury_risk_model(data):
    """
    Train a classification model to predict injury risks.
    """
    # Features: distance_covered, sprint_speed, heart_rate, minutes_played
    features = ['distance_covered', 'sprint_speed', 'heart_rate', 'minutes_played']
    X = data[features]
    y = data['injury_risk']  # Target variable (1 if injured, 0 otherwise)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

def predict_injury_risk(model, player_data):
    """
    Predict injury risk for a player.
    """
    prediction = model.predict(player_data)
    return "High Risk" if prediction[0] == 1 else "Low Risk"

def generate_positioning_heatmap(tracking_data):
    """
    Generate a heatmap for player positioning.
    """
    fig = px.density_heatmap(
        tracking_data, x='x', y='y', nbinsx=20, nbinsy=20,
        title="Player Positioning Heatmap"
    )
    fig.show()

def analyze_movement_patterns(tracking_data):
    """
    Analyze movement patterns (e.g., passing lanes, defensive coverage).
    """
    # Example: Calculate average speed and distance covered
    avg_speed = tracking_data['speed'].mean()
    total_distance = tracking_data['distance_covered'].sum()
    
    print(f"Average Speed: {avg_speed:.2f} m/s")
    print(f"Total Distance Covered: {total_distance:.2f} meters")

def train_fantasy_points_model(data):
    """
    Train a regression model to predict fantasy points.
    """
    # Features: goals, assists, pass_accuracy, defensive_score, xg_contribution
    features = ['goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']
    X = data[features]
    y = data['fantasy_points']  # Target variable
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    
    return model

def recommend_players(model, player_data):
    """
    Recommend players based on predicted fantasy points.
    """
    predictions = model.predict(player_data)
    player_data['predicted_fantasy_points'] = predictions
    return player_data.sort_values(by="predicted_fantasy_points", ascending=False)

# Recommend players
player_data = fantasy_data[['player_id', 'player_name', 'goals', 'assists', 'pass_accuracy', 'defensive_score', 'xg_contribution']]
recommended_players = recommend_players(fantasy_model, player_data)
print(recommended_players.head())

# Player Scouting
undervalued_players = identify_undervalued_players(performance_data, market_value_data)

# Injury Prevention
injury_model = train_injury_risk_model(biometric_data)
risk_level = predict_injury_risk(injury_model, player_data)

# Tactical Analysis
generate_positioning_heatmap(tracking_data)
analyze_movement_patterns(tracking_data)

# Fantasy Football
fantasy_model = train_fantasy_points_model(fantasy_data)
recommended_players = recommend_players(fantasy_model, player_data)
