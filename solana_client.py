import time
import logging
from typing import Optional, Tuple
from solders.keypair import Keypair
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from solana.exceptions import SolanaRpcException
from solana.transaction import TransactionExpiredBlockheightExceededError
from solana.rpc.core import RPCException
from solana import compute_budget
from config import Config

logger = logging.getLogger(__name__)

class SolanaClientError(Exception):
    """Custom exception for Solana client errors"""
    pass

class SolanaClient:
    def __init__(self):
        self.client = Client(Config.SOLANA_RPC_URL, timeout=30)
        self.sender_keypair = self._load_keypair()
        self.sender_pubkey = self.sender_keypair.pubkey()
        self.min_balance_reserve = 890880  # Minimum balance reserve for account
        self.max_retries = 5
        self.retry_delay = 2
        
    def _load_keypair(self) -> Keypair:
        """Securely load sender keypair from config"""
        try:
            return Keypair.from_base58_string(Config.SENDER_WALLET_PRIVATE_KEY)
        except ValueError as e:
            logger.critical("Invalid private key format")
            raise SolanaClientError("Invalid private key configuration") from e

    def validate_wallet(self, address: str) -> bool:
        """Validate Solana wallet address with improved checks"""
        try:
            Pubkey.from_string(address)
            return True
        except (ValueError, AttributeError):
            return False

    def get_balance(self, pubkey: Pubkey, commitment: Confirmed = Confirmed) -> int:
        """Get balance with proper error handling"""
        try:
            resp = self.client.get_balance(pubkey, commitment=commitment)
            return resp.value
        except (SolanaRpcException, RPCException) as e:
            logger.error(f"Balance check failed: {str(e)}")
            raise SolanaClientError("Failed to check balance") from e

    def estimate_fees(self, message: MessageV0) -> int:
        """Estimate transaction fees including priority fees"""
        try:
            # Get fee for message including priority fees
            fee_response = self.client.get_fee_for_message(message)
            if not fee_response.value:
                return 5000  # Fallback base fee
            
            # Add priority fee (micro-lamports per CU)
            fee = fee_response.value
            priority_fee = compute_budget.ComputeBudgetPriorityFee(
                compute_budget.ComputeBudgetPriorityFeeConfig(micro_lamports=1000)
            )
            return fee + priority_fee.value
        except (SolanaRpcException, RPCException) as e:
            logger.warning(f"Fee estimation failed: {str(e)}")
            return 10000  # Conservative fallback fee

    def create_transfer_instructions(self, receiver: Pubkey, lamports: int) -> Tuple[MessageV0, int]:
        """Create versioned transfer message with compute budget"""
        # Get recent blockhash
        recent_blockhash = self.client.get_latest_blockhash(commitment=Confirmed).value
        
        # Create transfer instruction
        transfer_instruction = transfer(
            TransferParams(
                from_pubkey=self.sender_pubkey,
                to_pubkey=receiver,
                lamports=lamports,
            )
        )
        
        # Create message with compute budget
        message = MessageV0.try_compile(
            payer=self.sender_pubkey,
            instructions=[
                compute_budget.ComputeBudgetSetComputeUnitLimit(100_000),
                transfer_instruction
            ],
            address_lookup_table_accounts=[],
            recent_blockhash=recent_blockhash.blockhash,
        )
        
        return message, recent_blockhash.last_valid_block_height

    def send_tokens(self, receiver: str, amount: int) -> Optional[Signature]:
        """Send tokens with retry logic and versioned transactions"""
        if not self.validate_wallet(receiver):
            logger.error(f"Invalid receiver address: {receiver}")
            return None

        receiver_pubkey = Pubkey.from_string(receiver)
        
        for attempt in range(self.max_retries):
            try:
                # Check sender balance
                sender_balance = self.get_balance(self.sender_pubkey)
                required_balance = amount + self.min_balance_reserve
                
                if sender_balance < required_balance:
                    logger.error(f"Insufficient balance: {sender_balance} < {required_balance}")
                    return None

                # Create transfer instructions
                message, last_valid_block_height = self.create_transfer_instructions(receiver_pubkey, amount)
                
                # Estimate fees
                fee = self.estimate_fees(message)
                logger.info(f"Estimated transaction fee: {fee} lamports")

                # Create and sign transaction
                transaction = VersionedTransaction(message, [self.sender_keypair])
                
                # Send transaction
                opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
                txid = self.client.send_transaction(transaction, opts).value
                logger.info(f"Transaction submitted: {txid}")
                
                # Confirm transaction
                confirmation = self.client.confirm_transaction(
                    txid,
                    commitment=Confirmed,
                    last_valid_block_height=last_valid_block_height,
                )
                
                if confirmation.value[0].err:
                    logger.error(f"Transaction failed: {confirmation.value[0].err}")
                    raise SolanaClientError("Transaction confirmation failed")
                
                return txid

            except TransactionExpiredBlockheightExceededError:
                logger.warning(f"Blockhash expired, retrying (attempt {attempt+1})")
                time.sleep(self.retry_delay * (2 ** attempt))
                continue
            except (SolanaRpcException, RPCException) as e:
                logger.error(f"Transaction failed: {str(e)}")
                if "Blockhash not found" in str(e):
                    logger.info("Refreshing blockhash and retrying...")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise SolanaClientError("Transaction failed permanently") from e
        
        logger.error(f"Transaction failed after {self.max_retries} attempts")
        return None
