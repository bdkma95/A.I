import pytest
from unittest.mock import AsyncMock, call, MagicMock
from main_app import MainApp
from exceptions import TwitterAPIError, SolanaClientError

@pytest.fixture
def mock_app_components():
    """Fixture providing mocked dependencies for MainApp"""
    with patch('main_app.TwitterClient') as mock_twitter, \
         patch('main_app.SolanaClient') as mock_solana:
        
        # Configure mock clients
        twitter_client = MagicMock()
        solana_client = MagicMock()
        
        # Set up async methods
        twitter_client.search_tweets = AsyncMock()
        twitter_client.reply_to_tweet = AsyncMock()
        solana_client.send_tokens = AsyncMock()
        
        yield {
            'twitter': twitter_client,
            'solana': solana_client
        }

@pytest.mark.asyncio
async def test_main_processing_flow(mock_app_components):
    """Test complete tweet processing workflow with successful transactions"""
    # Configure mock responses
    mock_tweets = [
        {"id": "1", "text": "Great project!", "author": "user1"},
        {"id": "2", "text": "Amazing work", "author": "user2"}
    ]
    mock_app_components['twitter'].search_tweets.return_value = mock_tweets
    mock_app_components['solana'].send_tokens.return_value = "tx123"
    
    # Initialize and run application
    app = MainApp()
    app.twitter_client = mock_app_components['twitter']
    app.solana_client = mock_app_components['solana']
    
    await app.process_tweets("test query", reward_amount=500000)
    
    # Verify search parameters
    mock_app_components['twitter'].search_tweets.assert_awaited_once_with(
        "test query",
        max_results=100,
        include_retweets=False
    )
    
    # Verify reward distribution
    assert mock_app_components['solana'].send_tokens.await_count == 2
    assert mock_app_components['solana'].send_tokens.await_args_list == [
        call("user1", 500000, priority_fee=0.0005),
        call("user2", 500000, priority_fee=0.0005)
    ]
    
    # Verify engagement actions
    mock_app_components['twitter'].reply_to_tweet.assert_has_awaits([
        call("1", "user1", "Thanks for your support! 🚀"),
        call("2", "user2", "Thanks for your support! 🚀")
    ], any_order=True)

@pytest.mark.asyncio
async def test_error_handling_and_retries(mock_app_components):
    """Test error recovery and retry mechanisms"""
    # Configure failure scenario
    mock_app_components['twitter'].search_tweets.side_effect = [
        TwitterAPIError("Rate limited"),
        [{"id": "3", "text": "Retry success", "author": "user3"}]
    ]
    
    app = MainApp()
    app.twitter_client = mock_app_components['twitter']
    app.solana_client = mock_app_components['solana']
    
    await app.process_tweets("error test", max_retries=3)
    
    # Verify retry behavior
    assert mock_app_components['twitter'].search_tweets.await_count == 2
    mock_app_components['solana'].send_tokens.assert_awaited_once_with(
        "user3", 1000000, priority_fee=0.0005
    )

@pytest.mark.parametrize("tweets, expected_actions", [
    ([], 0),  # No tweets found
    ([{"text": "SPAM", "author": "spambot"}], 0),  # Spam filtered
    ([{"text": "Valid", "author": "user1"}, {"text": "SPAM", "author": "spambot"}], 1)
])
@pytest.mark.asyncio
async def test_spam_filtering(mock_app_components, tweets, expected_actions):
    """Test spam detection and filtering logic"""
    mock_app_components['twitter'].search_tweets.return_value = tweets
    app = MainApp()
    app.twitter_client = mock_app_components['twitter']
    
    await app.process_tweets("filter test")
    
    assert mock_app_components['solana'].send_tokens.await_count == expected_actions
    assert len(app.stats['spam_blocked']) == len(tweets) - expected_actions

@pytest.mark.asyncio
async def test_concurrent_processing(mock_app_components):
    """Test parallel tweet processing capabilities"""
    # Generate 100 test tweets
    mock_tweets = [{"id": str(i), "text": f"Test {i}", "author": f"user{i}"} 
                 for i in range(100)]
    mock_app_components['twitter'].search_tweets.return_value = mock_tweets
    
    app = MainApp()
    app.twitter_client = mock_app_components['twitter']
    app.solana_client = mock_app_components['solana']
    
    await app.process_tweets("load test", batch_size=20)
    
    # Verify all tweets processed
    assert mock_app_components['solana'].send_tokens.await_count == 100
    assert len(app.stats['processed']) == 100
    assert app.stats['total_rewards'] == 100 * 1000000  # Default reward

def test_config_validation():
    """Test application configuration safeguards"""
    app = MainApp()
    
    with pytest.raises(ValueError):
        app.configure(max_retries=-1)
    
    with pytest.raises(TypeError):
        app.configure(reward_amount="invalid")

@pytest.mark.asyncio
async def test_metrics_tracking(mock_app_components):
    """Test runtime metrics collection and reporting"""
    mock_app_components['twitter'].search_tweets.return_value = [
        {"id": "1", "text": "Test", "author": "user1"}
    ]
    
    app = MainApp()
    app.twitter_client = mock_app_components['twitter']
    await app.process_tweets("metrics test")
    
    assert app.stats['processed'] == 1
    assert app.stats['rewards_sent'] == 1
    assert 'processing_time' in app.stats
