import os
import pytest
from config import Config

def test_config_defaults(monkeypatch):
    monkeypatch.delenv("TWITTER_API_KEY", raising=False)
    cfg = Config()
    
    assert cfg.SOLANA_RPC_URL == "https://api.mainnet-beta.solana.com"
    assert cfg.LOG_LEVEL == "INFO"

def test_config_environment_vars(monkeypatch):
    monkeypatch.setenv("TWITTER_API_KEY", "test_key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    
    cfg = Config()
    assert cfg.TWITTER_API_KEY == "test_key"
    assert cfg.LOG_LEVEL == "DEBUG"
