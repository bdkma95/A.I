import pytest
from reply_generator import ReplyGenerator

def test_reply_generation(mock_openai):
    generator = ReplyGenerator()
    reply = generator.generate_reply("Great project!", "user123", "tweet123")
    assert "user123" in reply
    mock_openai.ChatCompletion.create.assert_called_once()

def test_cache_usage():
    generator = ReplyGenerator()
    text = "Cached message"
    reply1 = generator.generate_reply(text, "user1", "t1")
    reply2 = generator.generate_reply(text, "user1", "t1")
    assert reply1 == reply2

def test_fallback_reply():
    generator = ReplyGenerator()
    with pytest.raises(Exception):
        generator.generate_reply("", "user123", "t1")
    assert "user123" in generator.fallback_responses['neutral'][0]
