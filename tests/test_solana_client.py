import pytest
from unittest.mock import AsyncMock, patch
from solana_client import SolanaClient, SolanaClientError

@pytest.fixture(scope="module")
def solana_client():
    """Fixture providing a configured SolanaClient instance."""
    return SolanaClient(rpc_url="https://api.mainnet-beta.solana.com")

@pytest.fixture
def mock_solana_client():
    """Fixture providing a mocked SolanaClient for async tests."""
    with patch('solana_client.AsyncClient') as mock:
        client = SolanaClient()
        client._async_client = mock.return_value
        yield client

@pytest.mark.parametrize("address, expected", [
    ("VALID_WALLET_ADDRESS_123", True),
    ("B62qk1...LONG_PUBKEY", True),  # Valid compressed address
    ("invalid$char@address", False),
    ("", False),  # Empty string
    ("X" * 44, True),  # Valid length
    ("X" * 43, False),  # Invalid length
    (" \t\nVALID_WITH_WHITESPACE\t\n", True),  # With whitespace
])
def test_wallet_validation(solana_client, address, expected):
    """Test wallet address validation with various edge cases."""
    assert solana_client.validate_wallet(address) == expected
    # Verify normalization of valid addresses
    if expected:
        assert solana_client.validate_wallet(address.strip()) is True

@pytest.mark.asyncio
async def test_send_tokens_success(mock_solana_client):
    """Test successful token transfer flow with confirmation."""
    # Configure mock responses
    mock_solana_client.get_balance = AsyncMock(return_value=10**9)  # 1 SOL
    mock_solana_client.send_tokens = AsyncMock(return_value="tx123")
    mock_solana_client.confirm_transaction = AsyncMock(return_value=True)
    
    result = await mock_solana_client.send_tokens(
        destination="valid_wallet",
        amount=500_000,  # 0.5 SOL
        retries=3
    )
    
    # Verify transaction flow
    assert result == "tx123"
    mock_solana_client.get_balance.assert_awaited_once_with("valid_wallet")
    mock_solana_client.send_tokens.assert_awaited_once_with(
        "valid_wallet",
        500_000,
        priority_fee=0.0005,
        max_retries=3
    )
    mock_solana_client.confirm_transaction.assert_awaited_once_with("tx123")

@pytest.mark.asyncio
async def test_send_tokens_insufficient_balance(mock_solana_client):
    """Test balance check prevention and proper error formatting."""
    mock_solana_client.get_balance.return_value = 100_000  # 0.1 SOL
    
    with pytest.raises(SolanaClientError) as exc_info:
        await mock_solana_client.send_tokens("valid_wallet", 500_000)
    
    assert "Insufficient balance" in str(exc_info.value)
    assert "0.0001" in str(exc_info.value)  # Formatted balance
    assert "0.0005" in str(exc_info.value)  # Formatted amount
    mock_solana_client.send_tokens.assert_not_awaited()

@pytest.mark.asyncio
async def test_network_failure_retries(mock_solana_client):
    """Test network failure retry logic with exponential backoff."""
    from asyncstdlib import itertools
    mock_solana_client.send_tokens = AsyncMock(
        side_effect=itertools.repeat(SolanaClientError("Network error"), 3
    )
    
    with pytest.raises(SolanaClientError) as exc_info:
        await mock_solana_client.send_tokens("valid_wallet", 100_000, retries=3)
    
    assert "Failed after 3 retries" in str(exc_info.value)
    assert mock_solana_client.send_tokens.await_count == 3

def test_fee_calculation(solana_client):
    """Test dynamic fee calculation based on network conditions."""
    # Mock recent prioritization fees
    solana_client._get_recent_priority_fees = lambda: [0.0005, 0.0006, 0.0007]
    
    calculated_fee = solana_client.calculate_priority_fee(percentile=90)
    assert calculated_fee == pytest.approx(0.00065, rel=0.1)

@pytest.mark.asyncio
async def test_transaction_confirmation(mock_solana_client):
    """Test transaction confirmation with status validation."""
    mock_solana_client.confirm_transaction = AsyncMock(
        side_effect=[False, False, True]
    )
    
    result = await mock_solana_client.confirm_transaction(
        "tx123",
        max_retries=3,
        delay=0.1
    )
    
    assert result is True
    assert mock_solana_client.confirm_transaction.await_count == 3

@pytest.mark.asyncio
async def test_concurrent_transactions(mock_solana_client):
    """Test parallel transaction handling and nonce management."""
    from asyncio import gather, Lock
    from unittest.mock import AsyncMock

    # Setup tracking variables
    nonces = []
    nonce_lock = Lock()
    transaction_order = []

    async def mock_send(destination, amount, nonce, **kwargs):
        """Mock send function with delay and nonce tracking"""
        async with nonce_lock:
            nonces.append(nonce)
            transaction_order.append(("start", nonce))
        
        # Simulate network delay based on nonce
        await asyncio.sleep(0.1 * (nonce % 2))
        
        async with nonce_lock:
            transaction_order.append(("complete", nonce))
        return f"tx_{nonce}"

    # Configure mock with sequential nonce generation
    mock_solana_client.send_tokens = AsyncMock(side_effect=mock_send)
    mock_solana_client._generate_nonce = AsyncMock(
        side_effect=range(100, 1000)
    )

    # Send 5 concurrent transactions
    results = await gather(
        mock_solana_client.send_tokens("wallet1", 100_000),
        mock_solana_client.send_tokens("wallet2", 200_000),
        mock_solana_client.send_tokens("wallet3", 300_000),
        mock_solana_client.send_tokens("wallet4", 400_000),
        mock_solana_client.send_tokens("wallet5", 500_000),
    )

    # Verify results
    assert len(results) == 5
    assert all(r.startswith("tx_") for r in results)
    
    # Check nonce sequencing
    assert len(nonces) == 5
    assert sorted(nonces) == list(range(100, 105)), "Nonces should be sequential"
    assert nonces == list(range(100, 105)), "Nonces should be ordered"

    # Verify transaction processing order
    start_order = [n for action, n in transaction_order if action == "start"]
    complete_order = [n for action, n in transaction_order if action == "complete"]
    
    assert start_order == list(range(100, 105)), "Transactions started in order"
    assert complete_order != start_order, "Should complete out-of-order due to simulated delays"
    assert set(complete_order) == set(start_order), "All transactions completed"
