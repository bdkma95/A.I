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
    """Main application settings with nested configurations."""
    database: DatabaseSettings = DatabaseSettings()
    aws: AWSSettings = AWSSettings()
    api: APISettings = APISettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True

    @property
    def show_redacted(self) -> dict:
        """Return settings with sensitive data redacted"""
        return {
            "database": self.database.dict(exclude={"password"}),
            "aws": self.aws.dict(exclude={"secret_access_key"}),
            "api": self.api.dict(exclude={"football_data", "openweathermap"})
        }

settings = Settings()
