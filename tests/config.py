import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class Config:
    # --- Twitter Configuration ---
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    TWITTER_RATE_LIMIT = int(os.getenv("TWITTER_RATE_LIMIT", "300"))  # API calls/hour

    # --- Solana Configuration ---
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    SOLANA_WS_URL = os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
    SOLANA_COMMITMENT = os.getenv("SOLANA_COMMITMENT", "confirmed")
    WALLET_PUBKEY = os.getenv("WALLET_PUBKEY")
    WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")
    
    # Transaction Settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    RETRY_BASE_DELAY = int(os.getenv("RETRY_BASE_DELAY", "2"))
    TX_TIMEOUT = int(os.getenv("TX_TIMEOUT", "300"))  # 5 minutes
    MIN_SOL_BALANCE = int(os.getenv("MIN_SOL_BALANCE", "1000000000"))  # 1 SOL
    
    # --- AI/ML Configuration ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "150"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Sentiment Analysis
    SENTIMENT_POS_THRESHOLD = float(os.getenv("SENTIMENT_POS_THRESHOLD", "0.25"))
    SENTIMENT_NEG_THRESHOLD = float(os.getenv("SENTIMENT_NEG_THRESHOLD", "-0.25"))

    # --- Application Settings ---
    APP_ENV = os.getenv("APP_ENV", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    REPORTING_INTERVAL = int(os.getenv("REPORTING_INTERVAL", "300"))  # 5 minutes

    # --- Meme Coin Settings ---
    COIN_SYMBOL = os.getenv("COIN_SYMBOL", "$MEME")
    COIN_NAME = os.getenv("COIN_NAME", "MemeCoin")
    AIRDROP_AMOUNT = int(os.getenv("AIRDROP_AMOUNT", "1000000"))  # In lamports
    AIRDROP_PROBABILITY = float(os.getenv("AIRDROP_PROBABILITY", "0.1"))

    # --- Security Settings ---
    SECRET_KEY = os.getenv("SECRET_KEY", "default-insecure-secret")
    JWT_SECRET = os.getenv("JWT_SECRET", "default-jwt-secret")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    RATE_LIMITS = {
        'api': os.getenv("RATE_LIMIT_API", "100/5 minute"),
        'auth': os.getenv("RATE_LIMIT_AUTH", "10/minute")
    }

    # --- Monitoring & Alerting ---
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
    ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK")
    ADMIN_EMAILS = os.getenv("ADMIN_EMAILS", "").split(",")

    # --- Cache & Performance ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10000"))

    # --- Network Settings ---
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))
    KEEP_ALIVE = int(os.getenv("KEEP_ALIVE", "5"))

    # --- Spam Protection ---
    SPAM_KEYWORDS: List[str] = os.getenv("SPAM_KEYWORDS", "free,win,click").split(",")
    SPAM_THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.7"))
    USER_WHITELIST = os.getenv("USER_WHITELIST", "").split(",")

    # --- Feature Flags ---
    FEATURE_AIRDROP = os.getenv("FEATURE_AIRDROP", "true").lower() == "true"
    FEATURE_ENGAGEMENT = os.getenv("FEATURE_ENGAGEMENT", "true").lower() == "true"
    FEATURE_ANALYTICS = os.getenv("FEATURE_ANALYTICS", "false").lower() == "true"

    @classmethod
    def validate(cls):
        """Basic configuration validation"""
        required = [
            'TWITTER_API_KEY', 'TWITTER_API_SECRET',
            'WALLET_PUBKEY', 'WALLET_PRIVATE_KEY'
        ]
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing required configuration: {var}")

# Validate configuration on import
Config.validate()
