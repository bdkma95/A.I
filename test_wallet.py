from solders.keypair import Keypair
from config import Config

try:
    keypair = Keypair.from_base58_string(Config.SENDER_WALLET_PRIVATE_KEY)
    print(f"✅ Valid keypair! Balance check address: {keypair.pubkey()}")
except Exception as e:
    print(f"❌ Invalid keypair: {str(e)}")
