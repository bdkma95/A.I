import pytest
from solana_client import SolanaClient, SolanaClientError

def test_valid_wallet_address():
    client = SolanaClient()
    assert client.validate_wallet("VALID_WALLET_ADDRESS") is True

def test_invalid_wallet_address():
    client = SolanaClient()
    assert client.validate_wallet("INVALID_ADDRESS") is False

@pytest.mark.asyncio
async def test_send_tokens_success(mock_solana_client):
    mock_solana_client.send_tokens.return_value = "tx123"
    result = await mock_solana_client.send_tokens("valid_wallet", 1000000)
    assert result == "tx123"

@pytest.mark.asyncio
async def test_send_tokens_insufficient_balance(mock_solana_client):
    mock_solana_client.get_balance.return_value = 500000
    with pytest.raises(SolanaClientError):
        await mock_solana_client.send_tokens("valid_wallet", 1000000)
