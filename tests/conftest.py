import pytest
from unittest.mock import Mock, AsyncMock, patch, call
from datetime import datetime, timedelta
from solana.rpc.async_api import AsyncClient
from tweepy import API
import time

@pytest.fixture
def mock_solana_client():
    """Enhanced Solana client mock with transaction lifecycle support"""
    client = Mock()
    
    # Basic methods
    client.get_balance = AsyncMock(return_value=10**9)  # 1 SOL
    client.validate_wallet = Mock(side_effect=lambda addr: len(addr) > 30)
    
    # Transaction simulation
    client.send_tokens = AsyncMock(
        side_effect=[
            Exception("Timeout"),
            "tx_retry_success_123"
        ]
    )
    client.estimate_fees = Mock(return_value=5000)
    client.get_transaction_status = AsyncMock(
        side_effect=lambda txid: "confirmed" if "success" in txid else "failed"
    )
    
    # Versioned transaction support
    client.create_versioned_transaction = Mock(return_value=Mock())
    client.simulate_transaction = AsyncMock(return_value={"result": {"err": None}})
    
    return client

@pytest.fixture
def mock_twitter_client():
    """Twitter client mock with engagement tracking and spam detection"""
    client = Mock()
    
    # Search and engagement
    client.search_tweets = AsyncMock(return_value=[
        {"id": "1", "text": "Great project!", "author": "user1", "spam_score": 0.1},
        {"id": "2", "text": "FREE MONEY", "author": "spambot", "spam_score": 0.9}
    ])
    
    # Engagement tracking
    client.engage_with_tweet = AsyncMock()
    client._like_tweet = AsyncMock()
    client._retweet = AsyncMock()
    client._reply_to_tweet = AsyncMock()
    
    # Spam detection
    client.is_spam = Mock(
        side_effect=lambda tweet: any(
            kw in tweet["text"].lower() 
            for kw in ["free", "click", "win"]
        )
    )
    
    # Sentiment analysis
    client.analyze_sentiment = Mock(
        side_effect=lambda text: 0.8 if "great" in text else -0.5
    )
    
    # Cooldown management
    client._action_cooldown = Mock(return_value=False)
    
    return client

@pytest.fixture
def mock_openai():
    """OpenAI mock with safety checks and response validation"""
    openai = Mock()
    
    # Completion responses
    openai.ChatCompletion.create = Mock(
        return_value={
            "choices": [{
                "message": {
                    "content": "@user1 🚀 Thanks for the support! #MemeCoinToTheMoon"
                }
            }]
        }
    )
    
    # Moderation system
    openai.Moderation.create = Mock(
        return_value={
            "results": [{"flagged": False}]
        }
    )
    
    # Error simulation
    openai.ChatCompletion.create.side_effect = [
        Exception("API Overload"),
        {"choices": [{"message": {"content": "Fallback response"}}]}
    ]
    
    return openai

@pytest.fixture
def mock_metrics():
    """Metrics collector mock for dashboard testing"""
    class Metrics:
        def __init__(self):
            self.metrics = {
                'engagement': defaultdict(int),
                'solana': {'balance': 10**9, 'transactions': 0},
                'api_usage': defaultdict(int),
                'alerts': []
            }
            
        def log_engagement(self, action_type):
            self.metrics['engagement'][action_type] += 1
            
        def log_api_call(self, service, success):
            key = f"{service}_{'success' if success else 'error'}"
            self.metrics['api_usage'][key] += 1
    
    return Metrics()

@pytest.fixture
def mock_main_app(mock_twitter_client, mock_solana_client, mock_metrics):
    """Main application instance with injected dependencies"""
    app = Mock()
    app.twitter_client = mock_twitter_client
    app.solana_client = mock_solana_client
    app.metrics = mock_metrics
    app.process_tweets = AsyncMock()
    app.process_airdrops = AsyncMock()
    return app
