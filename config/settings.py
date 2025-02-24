import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database configuration
    POSTGRES_CONFIG = {
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "host": os.getenv("POSTGRES_HOST"),
    }
    
    # AWS configuration
    AWS_CONFIG = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "bucket_name": os.getenv("AWS_BUCKET_NAME"),
    }
    
    # API Keys
    FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

settings = Settings()
