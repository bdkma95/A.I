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

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.DASHBOARD_SECRET_KEY
socketio = SocketIO(app, async_mode='gevent')

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'engagement': defaultdict(int),
            'solana': {},
            'api_usage': defaultdict(int),
            'alerts': [],
            'sentiment': defaultdict(float)
        }
        self.twitter_client = TwitterClient()
        self.solana_client = SolanaClient()
        self.reply_gen = ReplyGenerator()
        
        # Start background metrics updater
        self._running = True
        self.thread = threading.Thread(target=self._update_metrics)
        self.thread.daemon = True
        self.thread.start()
        
    def _update_metrics(self):
        """Background task to update metrics"""
        while self._running:
            try:
                # Update Solana metrics
                self.metrics['solana'] = {
                    'balance': self.solana_client.get_balance(
                        self.solana_client.sender_pubkey
                    ),
                    'transactions': self.metrics['api_usage'].get('solana_success', 0),
                    'airdrops': self.metrics['engagement'].get('airdrops_sent', 0)
                }
                
                # Check for alerts
                self._check_alerts()
                
                # Broadcast updates
                socketio.emit('update_metrics', self.metrics)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Metrics update failed: {e}")

    def _check_alerts(self):
        """Generate system alerts based on metrics"""
        alerts = []
        
        # Low balance alert
        if self.metrics['solana'].get('balance', 0) < Config.MIN_SOL_BALANCE:
            alerts.append({
                'type': 'critical',
                'message': f"Low Solana balance: {self.metrics['solana']['balance']/1e9:.2f} SOL"
            })
            
        # API error rate alert
        total_calls = sum(self.metrics['api_usage'].values())
        error_rate = self.metrics['api_usage'].get('errors', 0) / (total_calls or 1)
        if error_rate > Config.ERROR_RATE_THRESHOLD:
            alerts.append({
                'type': 'warning',
                'message': f"High API error rate: {error_rate:.1%}"
            })
            
        # Suspicious engagement spike
        recent_engagements = sum(self.metrics['engagement'].values())
        if recent_engagements > Config.ENGAGEMENT_SPIKE_THRESHOLD:
            alerts.append({
                'type': 'suspicious',
                'message': f"Engagement spike detected: {recent_engagements} actions/min"
            })
            
        self.metrics['alerts'] = alerts

metrics = MetricsCollector()

@app.route("/")
def dashboard():
    return render_template("dashboard.html", 
                          coin_symbol=Config.MEME_COIN_SYMBOL,
                          coin_name=Config.MEME_COIN_NAME)

@socketio.on("connect")
def handle_connect():
    socketio.emit('update_metrics', metrics.metrics)

def log_engagement(action_type: str):
    metrics.metrics['engagement'][action_type] += 1
    metrics.metrics['api_usage']['twitter'] += 1

def log_sentiment(polarity: float):
    metrics.metrics['sentiment']['total'] += polarity
    metrics.metrics['sentiment']['count'] += 1

def log_api_call(service: str, success: bool):
    metrics.metrics['api_usage'][f'{service}_{"success" if success else "error"}'] += 1

if __name__ == "__main__":
    socketio.run(app, 
                port=Config.DASHBOARD_PORT, 
                host=Config.DASHBOARD_HOST,
                debug=Config.DEBUG_MODE)
