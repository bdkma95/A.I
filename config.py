import os
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from a .env file

class Config:
    # Twitter API credentials
    API_KEY = os.getenv("API_KEY")
    API_SECRET_KEY = os.getenv("API_SECRET_KEY")
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
    BEARER_TOKEN = os.getenv("BEARER_TOKEN")  # For Twitter API v2

    # OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Solana configuration
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    SENDER_WALLET_PUBLIC_KEY = os.getenv("SENDER_WALLET_PUBLIC_KEY")
    SENDER_WALLET_PRIVATE_KEY = os.getenv("SENDER_WALLET_PRIVATE_KEY")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    RETRY_BASE_DELAY = int(os.getenv("RETRY_BASE_DELAY", "2"))
    MIN_BALANCE_RESERVE = int(os.getenv("MIN_BALANCE_RESERVE", "890880"))  # ~0.89 SOL
    PRIORITY_FEE_MICRO_LAMPORTS = int(os.getenv("PRIORITY_FEE_MICRO_LAMPORTS", "1000"))
    COMPUTE_UNIT_LIMIT = int(os.getenv("COMPUTE_UNIT_LIMIT", "100000"))

    # Meme coin and engagement settings
    MEME_COIN_SYMBOL = os.getenv("MEME_COIN_SYMBOL", "$MEMECOIN")
    MEME_COIN_NAME = os.getenv("MEME_COIN_NAME", "QuantumDoge")
    AIRDROP_AMOUNT = int(os.getenv("AIRDROP_AMOUNT", "1000000"))
    ACTION_COOLDOWN = int(os.getenv("ACTION_COOLDOWN", "300"))  # 5 minutes
    AIRDROP_PROBABILITY = float(os.getenv("AIRDROP_PROBABILITY", "0.1"))
    SPAM_KEYWORDS = os.getenv("SPAM_KEYWORDS", "giveaway,free,click,win").split(",")
    SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.15"))

    # Celery and task configuration
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    MAX_AIRDROP_RETRIES = int(os.getenv("MAX_AIRDROP_RETRIES", "5"))
    NOTIFICATION_WEBHOOK = os.getenv("NOTIFICATION_WEBHOOK")
    AIRDROP_HISTORY_TTL = int(os.getenv("AIRDROP_HISTORY_TTL", "7"))  # Days

    # Dashboard and monitoring
    DASHBOARD_SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY", "dev-secret")
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
    DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
    MIN_SOL_BALANCE = int(os.getenv("MIN_SOL_BALANCE", "1000000000"))  # 1 SOL
    ERROR_RATE_THRESHOLD = float(os.getenv("ERROR_RATE_THRESHOLD", "0.1"))
    ENGAGEMENT_SPIKE_THRESHOLD = int(os.getenv("ENGAGEMENT_SPIKE_THRESHOLD", "100"))

    # Reply generation
    REPLY_CACHE_SIZE = int(os.getenv("REPLY_CACHE_SIZE", "1000"))
    REPLY_CACHE_TTL = int(os.getenv("REPLY_CACHE_TTL", "3600"))  # Seconds
    AIRDROP_TEMPLATES = [
        "🎉 {user} You've won {coin} tokens! DM wallet 🚀",
        "🏆 {coin} champ detected! DM address for rewards! 💰"
    ]

    # Main application
    MAIN_LOOP_INTERVAL = int(os.getenv("MAIN_LOOP_INTERVAL", "300"))  # 5 minutes
    MAX_BACKOFF = int(os.getenv("MAX_BACKOFF", "600"))  # 10 minutes
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
