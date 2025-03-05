import pytest
from unittest.mock import Mock, AsyncMock
from solana.rpc.async_api import AsyncClient
from tweepy import API

@pytest.fixture
def mock_solana_client():
    client = Mock()
    client.get_balance = Mock(return_value=1000000000)  # 1 SOL
    client.send_tokens = Mock(return_value="tx123")
    client.validate_wallet = Mock(return_value=True)
    return client

@pytest.fixture
def mock_twitter_client():
    client = Mock()
    client.search_tweets = Mock(return_value=[{"id": "1", "text": "test"}])
    client.engage_with_tweet = Mock()
    return client

@pytest.fixture
def mock_openai():
    openai = Mock()
    openai.ChatCompletion.create = Mock(return_value={
        "choices": [{"message": {"content": "Test reply"}}]
    })
    return openai
