import os
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from a .env file

class Config:
    # Twitter API credentials
    API_KEY = os.getenv("API_KEY")
    API_SECRET_KEY = os.getenv("API_SECRET_KEY")
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

    # OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Solana configuration
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    SENDER_WALLET_PUBLIC_KEY = os.getenv("SENDER_WALLET_PUBLIC_KEY")
    SENDER_WALLET_PRIVATE_KEY = os.getenv("SENDER_WALLET_PRIVATE_KEY")

    # Meme coin and airdrop settings
    MEME_COIN_SYMBOL = os.getenv("MEME_COIN_SYMBOL", "$MEMECOIN")
    AIRDROP_AMOUNT = int(os.getenv("AIRDROP_AMOUNT", "1000000"))

    # Celery configuration
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
