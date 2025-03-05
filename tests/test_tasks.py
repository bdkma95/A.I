import pytest
from unittest.mock import patch, AsyncMock
from tasks import send_tokens_async
from exceptions import SolanaClientError
from celery.exceptions import Retry

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies for task testing"""
    with patch('tasks.SolanaClient') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.send_tokens = AsyncMock()
        mock_instance.confirm_transaction = AsyncMock()
        yield mock_instance

@pytest.fixture
def celery_config():
    """Configure Celery for testing"""
    return {
        'broker_url': 'memory://',
        'result_backend': 'cache+memory://',
        'task_always_eager': True,  # Run tasks synchronously for testing
    }

def test_send_tokens_success(mock_dependencies):
    """Test successful token transfer task execution"""
    # Configure mock responses
    mock_dependencies.send_tokens.return_value = "tx_123"
    mock_dependencies.confirm_transaction.return_value = True
    
    result = send_tokens_async.delay(
        destination="valid_wallet",
        amount=500000,
        priority_fee=0.0005
    ).get()
    
    # Verify task result structure
    assert result == {
        'status': 'confirmed',
        'tx_id': 'tx_123',
        'confirmed': True
    }
    
    # Verify proper client calls
    mock_dependencies.send_tokens.assert_awaited_once_with(
        "valid_wallet",
        500000,
        priority_fee=0.0005,
        max_retries=3
    )
    mock_dependencies.confirm_transaction.assert_awaited_once_with("tx_123")

def test_task_retry_logic(mock_dependencies):
    """Test automatic retries for transient errors"""
    from celery import current_app
    
    # Configure mock to fail twice then succeed
    mock_dependencies.send_tokens.side_effect = [
        SolanaClientError("Temporary error"),
        SolanaClientError("Temporary error"),
        "tx_456"
    ]
    
    # Set up retry policy
    current_app.conf.task_retry_backoff = True
    current_app.conf.task_retry_backoff_max = 600
    
    result = send_tokens_async.apply(args=["wallet", 100000]).get()
    
    # Verify retry attempts and final success
    assert mock_dependencies.send_tokens.await_count == 3
    assert result['tx_id'] == "tx_456"

@pytest.mark.parametrize("exception, max_retries", [
    (SolanaClientError("Permanent error", permanent=True), 0),
    (ConnectionError("Transient error"), 3)
])
def test_error_handling_strategies(mock_dependencies, exception, max_retries):
    """Test different error handling strategies"""
    mock_dependencies.send_tokens.side_effect = exception
    
    with pytest.raises(SolanaClientError if max_retries == 0 else Retry):
        send_tokens_async.delay("wallet", 100000, max_retries=max_retries).get()
    
    assert mock_dependencies.send_tokens.await_count == (1 if max_retries == 0 else 3)

def test_input_validation():
    """Test task input validation"""
    with pytest.raises(ValueError):
        send_tokens_async.delay("invalid_wallet!", -100).get()

def test_state_tracking(mock_dependencies):
    """Test task state updates during execution"""
    from celery.result import AsyncResult
    
    mock_dependencies.send_tokens.side_effect = [
        SolanaClientError("Error"),
        "tx_789"
    ]
    
    result = send_tokens_async.delay("wallet", 100000)
    async_result = AsyncResult(result.id)
    
    # Verify state transitions
    assert async_result.state == 'SUCCESS'
    assert async_result.info['retries'] == 1

def test_concurrent_execution(mock_dependencies):
    """Test parallel task execution handling"""
    from concurrent.futures import ThreadPoolExecutor
    
    mock_dependencies.send_tokens.side_effect = lambda *a, **k: f"tx_{hash(a)}"
    
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                send_tokens_async.delay(f"wallet{i}", 100000).get
            ) for i in range(5)
        ]
        
        results = [f.result() for f in futures]
    
    assert len({r['tx_id'] for r in results}) == 5
    assert mock_dependencies.send_tokens.await_count == 5
