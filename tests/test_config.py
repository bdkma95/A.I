import os
import pytest
from config import Config

def test_config_defaults(monkeypatch):
    """Test that default values are correctly set when no environment variables are present."""
    # Remove all relevant environment variables
    monkeypatch.delenv("TWITTER_API_KEY", raising=False)
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    
    cfg = Config()
    
    # Verify default values
    assert cfg.SOLANA_RPC_URL == "https://api.mainnet-beta.solana.com"
    assert cfg.LOG_LEVEL == "INFO"
    # Verify optional field is not set
    assert getattr(cfg, 'TWITTER_API_KEY', None) is None

def test_config_environment_vars(monkeypatch):
    """Test that environment variables override default values correctly."""
    # Set test values for all environment variables
    test_rpc = "https://testnet.solana.com"
    test_key = "test_api_key_123"
    test_log_level = "DEBUG"
    
    monkeypatch.setenv("SOLANA_RPC_URL", test_rpc)
    monkeypatch.setenv("TWITTER_API_KEY", test_key)
    monkeypatch.setenv("LOG_LEVEL", test_log_level)
    
    cfg = Config()
    
    # Verify environment values take precedence
    assert cfg.SOLANA_RPC_URL == test_rpc
    assert cfg.TWITTER_API_KEY == test_key
    assert cfg.LOG_LEVEL == test_log_level
