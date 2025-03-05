import pytest
from tasks import send_tokens_async

def test_send_tokens_task(mock_solana_client):
    result = send_tokens_async.delay("valid_wallet", 1000000).get()
    assert "tx123" in result['tx_id']

def test_task_retry_logic(mock_solana_client):
    mock_solana_client.send_tokens.side_effect = Exception("Error")
    with pytest.raises(Exception):
        send_tokens_async.delay("valid_wallet", 1000000).get()
