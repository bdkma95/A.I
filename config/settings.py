import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseSettings, SecretStr, PostgresDsn, AnyUrl, Field

load_dotenv()

class DatabaseSettings(BaseSettings):
    """Database configuration settings with validation."""
    name: str = Field(..., env="POSTGRES_DB")
    user: str = Field(..., env="POSTGRES_USER")
    password: SecretStr = Field(..., env="POSTGRES_PASSWORD")
    host: str = Field("localhost", env="POSTGRES_HOST")
    port: int = Field(5432, env="POSTGRES_PORT")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    
    @property
    def dsn(self) -> PostgresDsn:
        return AnyUrl.build(
            scheme="postgresql+asyncpg",
            user=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=str(self.port),
            path=f"/{self.name}"
        )

class AWSSettings(BaseSettings):
    """AWS configuration with automatic secret handling."""
    access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    secret_access_key: SecretStr = Field(..., env="AWS_SECRET_ACCESS_KEY")
    region_name: str = Field("us-east-1", env="AWS_REGION")
    bucket_name: str = Field(..., env="AWS_BUCKET_NAME")
    endpoint_url: Optional[str] = Field(None, env="AWS_ENDPOINT_URL")

class APISettings(BaseSettings):
    """External API configuration with secret management."""
    football_data: SecretStr = Field(..., env="FOOTBALL_DATA_API_KEY")
    openweathermap: SecretStr = Field(..., env="OPENWEATHERMAP_API_KEY")

        
class Settings(BaseSettings):
    class AppConfig:
        env: str = "production"
        version: str = "1.0.0"
        description: str = "AI-powered football tracking and analysis system"
        secret_key: SecretStr
        
    class ServerConfig:
        host: str = "0.0.0.0"
        port: int = 5000
        debug: bool = False
        use_reloader: bool = False
        
    class SocketConfig:
        allowed_origins: List[str] = ["http://localhost:3000"]
        debug: bool = False
        
    class CeleryConfig:
        enabled: bool = False
        broker_url: str = "redis://localhost:6379/0"
        result_backend: str = "redis://localhost:6379/1"
        
    class AIConfig:
        model_path: str = "/models/football-ai-v1"
        cache_dir: str = "/tmp/ai-cache"
        
    metrics: dict = {"port": 9100}
    rate_limits: dict = {"default": "200/day,50/hour"}

settings = Settings()
