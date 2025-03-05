import pytest
from unittest.mock import call, patch
from reply_generator import ReplyGenerator
from exceptions import APIError, ContentPolicyViolation

@pytest.fixture
def mock_openai():
    with patch('reply_generator.openai.ChatCompletion') as mock:
        yield mock

@pytest.fixture
def generator():
    return ReplyGenerator()

def test_reply_generation_success(generator, mock_openai):
    """Test successful reply generation with proper API calls and formatting."""
    # Configure mock response
    mock_openai.create.return_value = {
        "choices": [{"message": {"content": "Hi @testuser, thanks for your support! 🚀"}}]
    }
    
    test_message = "Awesome project! When launch?"
    username = "testuser"
    tweet_id = "12345"
    
    reply = generator.generate_reply(test_message, username, tweet_id)
    
    # Validate response formatting
    assert username in reply
    assert reply.startswith(f"Hi @{username}")
    assert "🚀" in reply  # Verify emoji preservation
    
    # Verify API call parameters
    mock_openai.create.assert_called_once()
    call_args = mock_openai.create.call_args[1]
    assert call_args['temperature'] == 0.7
    assert username in call_args['messages'][0]['content']
    assert test_message in call_args['messages'][0]['content']

def test_cache_behavior(generator, mock_openai):
    """Test response caching mechanism with different cache key scenarios."""
    # Configure mock to return unique responses
    mock_openai.create.side_effect = [
        {"choices": [{"message": {"content": "Response 1"}}]},
        {"choices": [{"message": {"content": "Response 2"}}]}
    ]
    
    # Same parameters should cache
    reply1 = generator.generate_reply("Same message", "user1", "t1")
    reply2 = generator.generate_reply("Same message", "user1", "t1")
    assert reply1 == reply2
    assert mock_openai.create.call_count == 1
    
    # Different tweet ID should bypass cache
    reply3 = generator.generate_reply("Same message", "user1", "t2")
    assert reply3 != reply1
    assert mock_openai.create.call_count == 2

def test_fallback_mechanisms(generator, mock_openai):
    """Test various failure scenarios and appropriate fallback responses."""
    # Test empty input fallback
    with pytest.raises(ValueError):
        generator.generate_reply("", "user", "t1")
    
    # Test API error fallback
    mock_openai.create.side_effect = APIError("Service unavailable")
    reply = generator.generate_reply("Valid message", "testuser", "t1")
    assert "testuser" in reply
    assert any(phrase in reply for phrase in generator.fallback_responses['neutral'])
    
    # Test content policy violation
    mock_openai.create.side_effect = ContentPolicyViolation("Policy violation")
    reply = generator.generate_reply("Bad content", "user", "t1")
    assert reply in generator.fallback_responses['policy_violation']

def test_response_safety(generator, mock_openai):
    """Test response sanitization and safety measures."""
    # Test HTML stripping
    mock_openai.create.return_value = {
        "choices": [{"message": {"content": "<script>alert('xss')</script>Hi @user"}}]
    }
    reply = generator.generate_reply("Test", "user", "t1")
    assert "<script>" not in reply
    assert "Hi @user" in reply
    
    # Test excessive mention prevention
    mock_openai.create.return_value = {
        "choices": [{"message": {"content": "@spam1 @spam2 @spam3"}}]
    }
    reply = generator.generate_reply("Test", "user", "t1")
    assert reply.count("@") <= 2

def test_rate_limiting(generator, mock_openai):
    """Test rate limiting and retry logic."""
    from time import time
    start_time = time()
    
    # Generate 5 replies quickly (assuming 3 requests/second rate limit)
    for _ in range(5):
        generator.generate_reply("test", "user", str(time()))
    
    duration = time() - start_time
    assert duration >= 1.5  # Verify rate limiting added delay
    assert mock_openai.create.call_count == 5
