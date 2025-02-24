from flask import Flask
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger
from prometheus_client import start_http_server
from config import FOOTBALL_DATA_API_KEY, OPENWEATHERMAP_API_KEY, POSTGRES_CONFIG, AWS_CONFIG
from data.collector import DataCollector
from data.processor import DataProcessor
from data.storage import DataStorage
from api.endpoints import api_bp
from logging_setup import setup_logging

logger = setup_logging()

# Initialize Flask app and extensions
app = Flask(__name__)
socketio = SocketIO(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
Swagger(app)

app.register_blueprint(api_bp)

# Initialize other components (Celery, DB pools, etc.) here

if __name__ == '__main__':
    start_http_server(8000)  # Start Prometheus metrics server

    # Initialize components
    data_collector = DataCollector(FOOTBALL_DATA_API_KEY, OPENWEATHERMAP_API_KEY)
    data_storage = DataStorage(POSTGRES_CONFIG, AWS_CONFIG)
    data_processor = DataProcessor()

    # Example operations
    matches_data = data_collector.fetch_football_data()
    processed_data = data_processor.process_data(matches_data)
    data_storage.save_to_postgres(processed_data, "matches")

    # Run the app
    socketio.run(app, debug=True)
