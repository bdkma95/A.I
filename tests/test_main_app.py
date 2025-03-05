import pytest
from main_app import MainApp

@pytest.mark.asyncio
async def test_main_loop(mock_twitter_client, mock_solana_client):
    app = MainApp()
    app.twitter_client = mock_twitter_client
    app.solana_client = mock_solana_client
    
    await app.process_tweets("test")
    mock_twitter_client.search_tweets.assert_called()
