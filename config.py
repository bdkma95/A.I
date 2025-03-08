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
    TWITTER_RATE_LIMIT = int(os.getenv("TWITTER_RATE_LIMIT", "300"))

    # --- Solana Configuration --- (UPDATED SECTION)
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    SOLANA_WS_URL = os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
    SENDER_WALLET_PRIVATE_KEY = os.getenv("SENDER_WALLET_PRIVATE_KEY")
    SENDER_WALLET_PUBKEY = os.getenv("SENDER_WALLET_PUBKEY")
    
    # Transaction Configuration (NEW PARAMS)
    COMPUTE_UNIT_LIMIT = int(os.getenv("COMPUTE_UNIT_LIMIT", "200000"))  # Default: 200,000
    PRIORITY_FEE_MICRO_LAMPORTS = int(os.getenv("PRIORITY_FEE_MICRO_LAMPORTS", "500"))
    FALLBACK_FEE = int(os.getenv("FALLBACK_FEE", "5000"))  # lamports
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    RETRY_BASE_DELAY = int(os.getenv("RETRY_BASE_DELAY", "2"))
    CONFIRMATION_TIMEOUT = int(os.getenv("CONFIRMATION_TIMEOUT", "300"))  # 5 minutes
    MIN_BALANCE_RESERVE = int(os.getenv("MIN_BALANCE_RESERVE", "1000000000"))  # 1 SOL
    HISTORY_RETENTION = int(os.getenv("HISTORY_RETENTION", "86400"))  # 24 hours

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
    REPORTING_INTERVAL = int(os.getenv("REPORTING_INTERVAL", "300"))

    # --- Meme Coin Settings --- (UPDATED)
    MEME_COIN_SYMBOL = os.getenv("MEME_COIN_SYMBOL", "MEME")
    MEME_COIN_NAME = os.getenv("MEME_COIN_NAME", "MemeCoin")
    AIRDROP_AMOUNT = int(os.getenv("AIRDROP_AMOUNT", "1000000"))
    AIRDROP_PROBABILITY = float(os.getenv("AIRDROP_PROBABILITY", "0.1"))
    REPLY_CACHE_SIZE = 1000
    REPLY_CACHE_TTL = 3600  # 1 hour
    FALLBACK_TEMPLATES = {
        'positive': ["@{user} Thanks for the support! 🚀"],
        'neutral': ["@{user} Appreciate your engagement!"],
        'negative': ["@{user} We value your feedback!"]
    }

    # --- Monitoring & Alerting --- (UPDATED)
    NOTIFICATION_WEBHOOK = os.getenv("NOTIFICATION_WEBHOOK")
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
    ALERT_WEBHOOK = os.getenv("ALERT_WEBHOOK")
    ADMIN_EMAILS = os.getenv("ADMIN_EMAILS", "").split(",")

    # --- Security Settings ---
    SECRET_KEY = os.getenv("SECRET_KEY", "default-insecure-secret")
    JWT_SECRET = os.getenv("JWT_SECRET", "default-jwt-secret")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    RATE_LIMITS = {
        'api': os.getenv("RATE_LIMIT_API", "100/5 minute"),
        'auth': os.getenv("RATE_LIMIT_AUTH", "10/minute")
    }

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
    
    # Dashboard Configuration
    DASHBOARD_SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY",     "default-dashboard-secret")
    DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "securepassword123")
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))

    @classmethod
    def validate(cls):
        """Updated configuration validation"""
        required = [
            'TWITTER_API_KEY', 'TWITTER_API_SECRET',
            'SENDER_WALLET_PUBKEY', 'SENDER_WALLET_PRIVATE_KEY',
            'OPENAI_API_KEY'
        ]
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing required configuration: {var}")

        if cls.APP_ENV == "production" and cls.DEBUG:
            raise ValueError("Debug mode cannot be enabled in production")

# Validate configuration on import
Config.validate()
