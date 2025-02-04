import requests
import pandas as pd
import psycopg2
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import os

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
