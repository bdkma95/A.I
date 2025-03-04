from celery import Celery
import logging
import time
import threading
from typing import Dict, Any
from datetime import datetime, timedelta
from solana_client import SolanaClient, SolanaClientError
from config import Config
from dashboard import log_api_call, log_engagement
import requests

logger = logging.getLogger(__name__)

celery_app = Celery('tasks', broker=Config.CELERY_BROKER_URL)
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)

class AirdropTracker:
    _history = {}
    _lock = threading.Lock()
    
    @classmethod
    def record_airdrop(cls, task_id: str, receiver: str, amount: int):
        with cls._lock:
            cls._history[task_id] = {
                'receiver': receiver,
                'amount': amount,
                'status': 'pending',
                'tx_id': None,
                'retries': 0,
                'created_at': datetime.utcnow(),
                'completed_at': None,
                'errors': []
            }
    
    @classmethod
    def update_status(cls, task_id: str, status: str, tx_id: str = None, error: str = None):
        with cls._lock:
            if task_id in cls._history:
                entry = cls._history[task_id]
                entry.update({
                    'status': status,
                    'tx_id': tx_id,
                    'completed_at': datetime.utcnow() if status in ['success', 'failed'] else None,
                    'retries': entry.get('retries', 0) + (1 if status == 'retrying' else 0)
                })
                if error:
                    entry['errors'].append(error[-500:])  # Truncate long errors

@celery_app.task(
    bind=True,
    autoretry_for=(SolanaClientError,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={'max_retries': Config.MAX_AIRDROP_RETRIES},
    acks_late=True,
    max_retries=Config.MAX_AIRDROP_RETRIES
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
        # Validate input parameters
        if not client.validate_wallet(receiver_wallet):
            raise ValueError(f"Invalid wallet address: {receiver_wallet}")
        
        if amount <= 0 or amount > Config.MAX_AIRDROP_AMOUNT:
            raise ValueError(f"Invalid airdrop amount: {amount}")

        # Record airdrop attempt
        AirdropTracker.record_airdrop(task_id, receiver_wallet, amount)
        
        # Execute transaction
        tx_id = client.send_tokens(receiver_wallet, amount)
        
        if tx_id:
            # Monitor transaction confirmation with timeout
            start_time = time.time()
            while time.time() - start_time < Config.TRANSACTION_TIMEOUT:
                status = client.get_transaction_status(tx_id)
                if status == 'confirmed':
                    AirdropTracker.update_status(task_id, 'success', tx_id)
                    logger.info(f"Airdrop succeeded to {receiver_wallet}")
                    log_engagement('airdrops_sent')
                    log_api_call('solana', True)
                    return {'status': 'success', 'tx_id': tx_id}
                if status == 'failed':
                    break
                time.sleep(10)
            
            # If we get here, transaction didn't confirm
            raise SolanaClientError("Transaction confirmation timeout")
            
        raise SolanaClientError("Transaction ID not received")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Airdrop failed for {receiver_wallet}: {error_msg}")
        AirdropTracker.update_status(task_id, 'failed', error=error_msg)
        log_api_call('solana', False)
        self.notify_failure(airdrop_data, error_msg)
        
        if self.request.retries < Config.MAX_AIRDROP_RETRIES:
            AirdropTracker.update_status(task_id, 'retrying')
            raise self.retry(exc=e)
        
        logger.critical(f"Permanent failure for {receiver_wallet} after {Config.MAX_AIRDROP_RETRIES} attempts")
        return {'status': 'failed', 'error': error_msg}

def notify_failure(self, data: Dict[str, Any], error: str):
    """Send failure notification with retry logic"""
    try:
        # Webhook notification
        if Config.NOTIFICATION_WEBHOOK:
            self.send_webhook_notification(data, error)
            
        # Email/SMS notifications
        if Config.ADMIN_EMAILS:
            self.send_email_alert(data, error)
            
        # Update dashboard metrics
        self.update_dashboard_metrics(data, error)
    except Exception as e:
        logger.error(f"Notification failed: {str(e)}")

def send_webhook_notification(self, data: Dict, error: str):
    """Send webhook with retry logic"""
    for attempt in range(3):
        try:
            response = requests.post(
                Config.NOTIFICATION_WEBHOOK,
                json={
                    'event': 'airdrop_failure',
                    'data': data,
                    'error': error,
                    'attempt': self.request.retries,
                    'timestamp': datetime.utcnow().isoformat()
                },
                timeout=5
            )
            response.raise_for_status()
            return
        except Exception as e:
            if attempt == 2:
                raise
            sleep_time = 2 ** attempt + random.random()
            time.sleep(sleep_time)

def update_dashboard_metrics(self, data: Dict, error: str):
    """Update metrics with thread-safe operations"""
    try:
        from dashboard import metrics
        error_entry = {
            'timestamp': datetime.utcnow(),
            'receiver': data['receiver'],
            'amount': data['amount'],
            'error': error[:200],  # Truncate long errors
            'retries': self.request.retries
        }
        
        with metrics.lock:
            # Error tracking
            metrics.metrics['error_details'].append(error_entry)
            if len(metrics.metrics['error_details']) > Config.MAX_ERROR_HISTORY:
                metrics.metrics['error_details'].pop(0)
            
            # Failure counters
            metrics.metrics['engagement']['airdrops_failed'] += 1
            metrics.metrics['solana']['failures'] += 1
            
            # Critical alert handling
            if self.request.retries >= Config.MAX_AIRDROP_RETRIES - 1:
                alert_msg = (f"Critical failure to {data['receiver'][:6]}... "
                            f"Amount: {data['amount']/1e9:.4f} SOL")
                metrics.metrics['alerts'].insert(0, {
                    'type': 'critical',
                    'message': alert_msg,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
            # Error rate calculation
            total = (metrics.metrics['api_usage']['solana'].get('success', 0) +
                    metrics.metrics['api_usage']['solana'].get('error', 0))
            if total > 0:
                metrics.metrics['solana']['error_rate'] = (
                    metrics.metrics['api_usage']['solana']['error'] / total
                )
    except Exception as e:
        logger.error(f"Metrics update failed: {str(e)}")

@celery_app.task
def cleanup_airdrop_history():
    """Clean up old airdrop records with size limits"""
    cutoff = datetime.utcnow() - timedelta(days=Config.HISTORY_RETENTION_DAYS)
    with AirdropTracker._lock:
        AirdropTracker._history = {
            k: v for k, v in AirdropTracker._history.items()
            if v['created_at'] > cutoff
        }
        # Enforce max history size
        if len(AirdropTracker._history) > Config.MAX_HISTORY_ENTRIES:
            oldest = sorted(AirdropTracker._history.keys(), 
                          key=lambda x: AirdropTracker._history[x]['created_at'])[:-Config.MAX_HISTORY_ENTRIES]
            for key in oldest:
                del AirdropTracker._history[key]

# Periodic tasks
celery_app.conf.beat_schedule = {
    'cleanup-airdrop-history': {
        'task': 'tasks.cleanup_airdrop_history',
        'schedule': timedelta(hours=6),
    },
}
