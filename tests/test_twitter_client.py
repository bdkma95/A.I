import pytest
from twitter_client import TwitterClient
from unittest.mock import patch

@pytest.mark.asyncio
async def test_search_tweets(mock_twitter_client):
    with patch('twitter_client.TwitterClient') as mock:
        instance = mock.return_value
        instance.search_tweets.return_value = [{"id": "1", "text": "test"}]
        
        client = TwitterClient()
        tweets = await client.search_tweets("test")
        assert len(tweets) > 0
        instance.search_tweets.assert_called_once_with("test")

def test_spam_detection():
    client = TwitterClient()
    tweet = {"text": "FREE MONEY http://spam.com", "author": "spambot123"}
    assert client.is_spam(tweet) is True

def test_sentiment_analysis():
    client = TwitterClient()
    positive_text = "Loving the new features!"
    negative_text = "This is terrible!"
    
    assert client.analyze_sentiment(positive_text) > 0.2
    assert client.analyze_sentiment(negative_text) < -0.2
