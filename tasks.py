from celery import Celery
import logging
import time
from typing import Dict, Any
from datetime import datetime
from solana_client import SolanaClient, SolanaClientError
from config import Config
from dashboard import log_api_call, log_engagement

logger = logging.getLogger(__name__)

celery_app = Celery('tasks', broker=Config.CELERY_BROKER_URL)
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_acks_late=True,
)

class AirdropTracker:
    _history = {}
    
    @classmethod
    def record_airdrop(cls, task_id: str, receiver: str, amount: int):
        cls._history[task_id] = {
            'receiver': receiver,
            'amount': amount,
            'status': 'pending',
            'tx_id': None,
            'retries': 0,
            'created_at': datetime.utcnow(),
            'completed_at': None
        }
    
    @classmethod
    def update_status(cls, task_id: str, status: str, tx_id: str = None):
        if task_id in cls._history:
            cls._history[task_id].update({
                'status': status,
                'tx_id': tx_id,
                'completed_at': datetime.utcnow(),
                'retries': cls._history[task_id].get('retries', 0) + 1
            })

@celery_app.task(
    bind=True,
    autoretry_for=(SolanaClientError,),
    retry_backoff=30,
    retry_backoff_max=600,
    max_retries=Config.MAX_AIRDROP_RETRIES,
    acks_late=True
)
def send_tokens_async(self, receiver_wallet: str, amount: int):
    """Send tokens with enhanced monitoring and retry logic"""
    client = SolanaClient()
    task_id = self.request.id
    airdrop_data = {
        'receiver': receiver_wallet,
        'amount': amount,
        'task_id': task_id
    }
    
    try:
        # Record airdrop attempt
        AirdropTracker.record_airdrop(task_id, receiver_wallet, amount)
        
        # Execute transaction
        tx_id = client.send_tokens(receiver_wallet, amount)
        
        if tx_id:
            # Monitor transaction confirmation
            status = client.get_transaction_status(tx_id)
            while status == 'pending':
                time.sleep(10)
                status = client.get_transaction_status(tx_id)
                
            if status == 'confirmed':
                AirdropTracker.update_status(task_id, 'success', tx_id)
                logger.info(f"Airdrop succeeded to {receiver_wallet}")
                log_engagement('airdrops_sent')
                log_api_call('solana', True)
                return {'status': 'success', 'tx_id': tx_id}
            
            AirdropTracker.update_status(task_id, 'failed')
            logger.error(f"Transaction failed for {receiver_wallet}")
            self.retry(countdown=60 * (2 ** self.request.retries))
        else:
            raise SolanaClientError("Transaction ID not received")

    except Exception as e:
        logger.error(f"Airdrop failed for {receiver_wallet}: {str(e)}")
        AirdropTracker.update_status(task_id, 'failed')
        log_api_call('solana', False)
        self.notify_failure(airdrop_data, str(e))
        raise self.retry(exc=e)

def notify_failure(self, data: Dict[str, Any], error: str):
    """Send failure notification to monitoring system"""
    try:
        # Example webhook notification
        if Config.NOTIFICATION_WEBHOOK:
            import requests
            payload = {
                'text': f"Airdrop failure: {data['receiver']}",
                'fields': [
                    {'title': 'Amount', 'value': data['amount']},
                    {'title': 'Error', 'value': error},
                    {'title': 'Retries', 'value': self.request.retries}
                ]
            }
            requests.post(Config.NOTIFICATION_WEBHOOK, json=payload)
            
        # Update dashboard metrics
        self.update_dashboard_metrics(data, error)
    except Exception as e:
        logger.error(f"Notification failed: {str(e)}")

def update_dashboard_metrics(self, data: Dict[str, Any], error: str):
    """Update monitoring dashboard with failure details"""
    # Implement your dashboard-specific metrics updates here
    pass

@celery_app.task
def cleanup_airdrop_history(days: int = 7):
    """Clean up old airdrop records"""
    cutoff = datetime.utcnow() - timedelta(days=days)
    AirdropTracker._history = {
        k: v for k, v in AirdropTracker._history.items()
        if v['created_at'] > cutoff
    }
