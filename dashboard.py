from flask import Flask, render_template
from flask_socketio import SocketIO
import logging
import time
import threading
from collections import defaultdict
from typing import Dict, Any
from config import Config
from twitter_client import TwitterClient
from solana_client import SolanaClient
from reply_generator import ReplyGenerator
import json
from datetime import datetime

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.DASHBOARD_SECRET_KEY
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'engagement': defaultdict(int),
            'solana': {
                'balance': 0.0,
                'transactions': defaultdict(int),
                'airdrops': defaultdict(int),
                'error_rate': 0.0
            },
            'api_usage': {
                'twitter': defaultdict(int),
                'solana': defaultdict(int),
                'llm': defaultdict(int)
            },
            'alerts': [],
            'sentiment': {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'average': 0.0
            },
            'performance': {
                'response_times': [],
                'throughput': 0
            },
            'cache': {
                'hits': 0,
                'misses': 0,
                'size': 0
            }
        }
        self.lock = threading.Lock()
        self._running = True
        self.thread = threading.Thread(target=self._update_metrics)
        self.thread.daemon = True
        self.thread.start()

    def _update_metrics(self):
        """Background metrics collection with error handling"""
        while self._running:
            try:
                with self.lock:
                    # Update Solana metrics
                    try:
                        balance = SolanaClient().get_balance(
                            SolanaClient().sender_pubkey
                        )
                        self.metrics['solana']['balance'] = balance / 1e9  # Convert to SOL
                    except Exception as e:
                        logger.error(f"Balance check failed: {str(e)}")

                    # Calculate error rates
                    solana_total = sum(self.metrics['api_usage']['solana'].values())
                    self.metrics['solana']['error_rate'] = (
                        self.metrics['api_usage']['solana'].get('error', 0) / 
                        (solana_total or 1)
                    )

                    # Calculate sentiment averages
                    total = self.metrics['sentiment']['positive'] + \
                            self.metrics['sentiment']['neutral'] + \
                            self.metrics['sentiment']['negative']
                    if total > 0:
                        self.metrics['sentiment']['average'] = (
                            self.metrics['sentiment']['positive'] * 1 +
                            self.metrics['sentiment']['neutral'] * 0 +
                            self.metrics['sentiment']['negative'] * -1
                        ) / total
                    
                    # Check system alerts
                    self._check_alerts()
                    
                    # Broadcast updates
                    socketio.emit('metrics_update', json.dumps(self.metrics))
                    
                time.sleep(Config.DASHBOARD_UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Metrics update error: {str(e)}")
                time.sleep(10)

    def _check_alerts(self):
        """Generate system alerts with severity levels"""
        alerts = []
        
        # Solana balance alert
        if self.metrics['solana']['balance'] < Config.MIN_SOL_BALANCE / 1e9:
            alerts.append({
                'type': 'critical',
                'message': f"Low balance: {self.metrics['solana']['balance']:.4f} SOL",
                'timestamp': datetime.utcnow().isoformat()
            })
            
        # Engagement spike detection
        engagement_rate = self.metrics['engagement'].get('total', 0) / \
                         (Config.DASHBOARD_UPDATE_INTERVAL / 60)
        if engagement_rate > Config.ENGAGEMENT_SPIKE_THRESHOLD:
            alerts.append({
                'type': 'warning',
                'message': f"High engagement: {engagement_rate:.1f} actions/min",
                'timestamp': datetime.utcnow().isoformat()
            })
            
        # Cache efficiency warning
        cache_total = self.metrics['cache']['hits'] + self.metrics['cache']['misses']
        if cache_total > 0 and self.metrics['cache']['hits'] / cache_total < 0.2:
            alerts.append({
                'type': 'notice',
                'message': "Low cache hit rate (<20%)",
                'timestamp': datetime.utcnow().isoformat()
            })
            
        self.metrics['alerts'] = alerts[-Config.MAX_ALERTS_HISTORY:]  # Keep last N alerts

    def shutdown(self):
        """Graceful shutdown of metrics collection"""
        self._running = False
        self.thread.join()

metrics = MetricsCollector()

@app.route("/")
def dashboard():
    return render_template("dashboard.html",
                         coin_symbol=Config.MEME_COIN_SYMBOL,
                         coin_name=Config.MEME_COIN_NAME,
                         metrics_config=json.dumps({
                             'min_balance': Config.MIN_SOL_BALANCE / 1e9,
                             'alert_thresholds': {
                                 'engagement': Config.ENGAGEMENT_SPIKE_THRESHOLD,
                                 'error_rate': Config.ERROR_RATE_THRESHOLD
                             }
                         }))

@app.route("/health")
def health_check():
    return json.dumps({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@socketio.on("connect")
def handle_connect():
    with metrics.lock:
        socketio.emit('initial_metrics', json.dumps(metrics.metrics))

def log_engagement(action_type: str):
    with metrics.lock:
        metrics.metrics['engagement'][action_type] += 1
        metrics.metrics['engagement']['total'] += 1
        metrics.metrics['api_usage']['twitter']['total'] += 1

def log_sentiment(polarity: float):
    with metrics.lock:
        if polarity > 0.2:
            metrics.metrics['sentiment']['positive'] += 1
        elif polarity < -0.2:
            metrics.metrics['sentiment']['negative'] += 1
        else:
            metrics.metrics['sentiment']['neutral'] += 1

def log_api_call(service: str, success: bool):
    with metrics.lock:
        status = 'success' if success else 'error'
        metrics.metrics['api_usage'][service][status] += 1
        metrics.metrics['api_usage'][service]['total'] += 1

def log_cache(hit: bool):
    with metrics.lock:
        if hit:
            metrics.metrics['cache']['hits'] += 1
        else:
            metrics.metrics['cache']['misses'] += 1

if __name__ == "__main__":
    try:
        socketio.run(app, 
                    host=Config.DASHBOARD_HOST,
                    port=Config.DASHBOARD_PORT,
                    debug=Config.DEBUG_MODE,
                    use_reloader=False)
    finally:
        metrics.shutdown()
