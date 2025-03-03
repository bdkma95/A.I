import time
import logging
from solana.rpc.api import Client
from solana.publickey import PublicKey
from solana.system_program import TransferParams, transfer
from solana.transaction import Transaction
from config import Config

logger = logging.getLogger(__name__)

class SolanaClient:
    def __init__(self):
        self.client = Client(Config.SOLANA_RPC_URL)
        self.sender_wallet = PublicKey(Config.SENDER_WALLET_PUBLIC_KEY)
        # Note: Ensure that the private key is securely managed.
        self.sender_private_key = Config.SENDER_WALLET_PRIVATE_KEY

    def validate_solana_wallet(self, address):
        try:
            PublicKey(address)
            return True
        except Exception as e:
            logger.error(f"Invalid Solana wallet address: {address} - {e}")
            return False

    def send_tokens(self, receiver_wallet, amount, retries=3):
        for attempt in range(retries):
            try:
                balance = self.client.get_balance(self.sender_wallet)
                if balance["result"]["value"] < amount:
                    logger.error("Insufficient funds in sender wallet.")
                    return False

                txn = Transaction().add(
                    transfer(
                        TransferParams(
                            from_pubkey=self.sender_wallet,
                            to_pubkey=PublicKey(receiver_wallet),
                            lamports=amount,
                        )
                    )
                )
                response = self.client.send_transaction(txn, self.sender_private_key)
                logger.info(f"Tokens sent to {receiver_wallet}. Transaction ID: {response.get('result')}")
                return True
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        return False
