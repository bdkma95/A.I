from celery import Celery
import logging
from solana_client import SolanaClient
from config import Config

logger = logging.getLogger(__name__)

celery_app = Celery('tasks', broker=Config.CELERY_BROKER_URL)

@celery_app.task
def send_tokens_async(receiver_wallet, amount):
    client = SolanaClient()
    result = client.send_tokens(receiver_wallet, amount)
    if result:
        logger.info(f"Airdrop sent successfully to {receiver_wallet}")
    else:
        logger.error(f"Airdrop failed for {receiver_wallet}")
    return result
