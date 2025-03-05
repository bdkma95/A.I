import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from twitter_client import TwitterClient
from exceptions import TwitterAPIError

@pytest.fixture
def mock_twitter_client():
    """Fixture providing a mocked TwitterClient with async support."""
    with patch('twitter_client.TwitterClient') as mock:
        client = mock.return_value
        # Configure async methods
        client.search_tweets = AsyncMock()
        client.retweet = AsyncMock()
        client.reply_to_tweet = AsyncMock()
        yield client

@pytest.mark.asyncio
async def test_search_tweets_success(mock_twitter_client):
    """Test successful tweet search with various parameters."""
    # Configure mock response
    mock_twitter_client.search_tweets.return_value = [
        {"id": "1", "text": "Positive test"},
        {"id": "2", "text": "Another test"}
    ]
    
    client = TwitterClient()
    results = await client.search_tweets(
        query="test",
        max_results=100,
        include_retweets=False
    )
    
    # Verify response handling
    assert len(results) == 2
    mock_twitter_client.search_tweets.assert_awaited_once_with(
        query="test",
        max_results=100,
        include_retweets=False
    )

@pytest.mark.asyncio
async def test_search_api_error_handling(mock_twitter_client):
    """Test API error handling and retry logic."""
    mock_twitter_client.search_tweets.side_effect = TwitterAPIError("Rate limited", 429)
    
    with pytest.raises(TwitterAPIError) as exc_info:
        await TwitterClient().search_tweets("test", retries=3)
    
    assert "Rate limited" in str(exc_info.value)
    assert mock_twitter_client.search_tweets.await_count == 3

@pytest.mark.parametrize("tweet, expected", [
    ({"text": "FREE $$$ http://spam.com", "author": "spambot"}, True),
    ({"text": "Legit post", "author": "real_user"}, False),
    ({"text": "FOLLOW ME @spam1 @spam2 @spam3", "author": "spammer"}, True),
    ({"text": "Normal tweet with https://valid.com", "author": "user"}, False),
    ({"text": "Work from home!!!", "author": "job_spam"}, True),
    ({"text": "   ", "author": "empty_tweet"}, True),  # Empty content
])
def test_spam_detection_variants(tweet, expected):
    """Test spam detection with various input scenarios."""
    client = TwitterClient()
    assert client.is_spam(tweet) is expected

@pytest.mark.parametrize("text, expected_range", [
    ("I love this project!", (0.7, 1.0)),      # Strong positive
    ("This is awful", (-1.0, -0.6)),           # Strong negative
    ("It's okay, I guess", (-0.2, 0.2)),       # Neutral
    ("", (0.0, 0.0)),                          # Empty text
    ("🔥🔥🔥", (0.3, 1.0)),                    # Emoji handling
    ("MIXED feelings...", (-0.5, 0.5)),        # Mixed sentiment
])
def test_sentiment_analysis_ranges(text, expected_range):
    """Test sentiment analysis across different text types."""
    score = TwitterClient().analyze_sentiment(text)
    assert expected_range[0] <= score <= expected_range[1]

@pytest.mark.asyncio
async def test_retweet_flow(mock_twitter_client):
    """Test successful retweet flow with confirmation."""
    mock_twitter_client.retweet.return_value = {"id": "rt123"}
    
    result = await TwitterClient().retweet("tweet123")
    
    assert "rt123" in result["id"]
    mock_twitter_client.retweet.assert_awaited_once_with(
        "tweet123",
        enforce_policy=True
    )

@pytest.mark.asyncio
async def test_reply_validation(mock_twitter_client):
    """Test reply content validation and formatting."""
    client = TwitterClient()
    
    # Test mention auto-addition
    await client.reply_to_tweet("tweet456", "user123", "Thanks for feedback!")
    mock_twitter_client.reply_to_tweet.assert_awaited_once_with(
        "tweet456",
        "@user123 Thanks for feedback!",
        check_sentiment=True
    )
    
    # Test content length enforcement
    with pytest.raises(ValueError):
        long_text = "x" * 281  # Exceeds Twitter character limit
        await client.reply_to_tweet("tweet789", "user456", long_text)

def test_input_validation():
    """Test various input validation checks."""
    client = TwitterClient()
    
    # Test empty query handling
    with pytest.raises(ValueError):
        client.validate_search_params("")
    
    # Test invalid user ID format
    with pytest.raises(ValueError):
        client.get_user_profile("invalid@user")
