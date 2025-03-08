from solders.keypair import Keypair
from config import Config

try:
    key = Keypair.from_base58_string(Config.SENDER_WALLET_PRIVATE_KEY)
    print(f"✅ Valid key! Public address: {key.pubkey()}")
except Exception as e:
    print(f"❌ Validation failed: {str(e)}")
