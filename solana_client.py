import time
import logging
from typing import Optional, Tuple, Dict, List
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
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class SolanaClientError(Exception):
    """Custom exception for Solana client errors"""
    pass

class SolanaClient:
    def __init__(self):
        self.client = Client(Config.SOLANA_RPC_URL, timeout=30)
        self.sender_keypair = self._load_keypair()
        self.sender_pubkey = self.sender_keypair.pubkey()
        self.transaction_history: List[Dict] = []
        self.min_balance_reserve = Config.MIN_BALANCE_RESERVE
        self._last_blockhash = None
        self._last_blockheight = None

    def _load_keypair(self) -> Keypair:
        """Securely load sender keypair from config"""
        try:
            return Keypair.from_base58_string(Config.SENDER_WALLET_PRIVATE_KEY)
        except ValueError as e:
            logger.critical("Invalid private key format")
            raise SolanaClientError("Invalid private key configuration") from e

    def validate_wallet(self, address: str) -> bool:
        """Validate Solana wallet address with network check"""
        try:
            pubkey = Pubkey.from_string(address)
            return self.client.get_account_info(pubkey).value is not None
        except (ValueError, AttributeError, RPCException):
            return False

    def get_balance(self, pubkey: Pubkey) -> int:
        """Get balance with error handling and caching"""
        try:
            resp = self.client.get_balance(pubkey, commitment=Confirmed)
            return resp.value
        except (SolanaRpcException, RPCException) as e:
            logger.error(f"Balance check failed: {str(e)}")
            raise SolanaClientError("Failed to check balance") from e

    def estimate_fees(self, message: MessageV0) -> int:
        """Estimate transaction fees with priority and fallback"""
        try:
            fee_response = self.client.get_fee_for_message(message)
            if not fee_response.value:
                return Config.FALLBACK_FEE
            
            priority_fee = compute_budget.ComputeBudgetPriorityFee(
                compute_budget.ComputeBudgetPriorityFeeConfig(
                    micro_lamports=Config.PRIORITY_FEE_MICRO_LAMPORTS
                )
            )
            return fee_response.value + priority_fee.value
        except (SolanaRpcException, RPCException) as e:
            logger.warning(f"Fee estimation failed: {str(e)}")
            return Config.FALLBACK_FEE

    def _get_recent_blockhash(self) -> Tuple[str, int]:
        """Get recent blockhash with caching"""
        if not self._last_blockhash or time.time() > self._last_blockhash_expiry:
            response = self.client.get_latest_blockhash(commitment=Confirmed)
            self._last_blockhash = response.value.blockhash
            self._last_blockheight = response.value.last_valid_block_height
            self._last_blockhash_expiry = time.time() + 60  # Refresh every 60 seconds
        return self._last_blockhash, self._last_blockheight

    def create_transfer_instructions(self, receiver: Pubkey, lamports: int) -> MessageV0:
        """Create versioned transfer message with compute budget"""
        blockhash, _ = self._get_recent_blockhash()
        
        transfer_instruction = transfer(
            TransferParams(
                from_pubkey=self.sender_pubkey,
                to_pubkey=receiver,
                lamports=lamports,
            )
        )
        
        return MessageV0.try_compile(
            payer=self.sender_pubkey,
            instructions=[
                compute_budget.ComputeBudgetSetComputeUnitLimit(
                    Config.COMPUTE_UNIT_LIMIT
                ),
                transfer_instruction
            ],
            address_lookup_table_accounts=[],
            recent_blockhash=blockhash,
        )

    def send_tokens(self, receiver: str, amount: int) -> Optional[Signature]:
        """Send tokens with enhanced retry logic and monitoring"""
        if not self.validate_wallet(receiver):
            logger.error(f"Invalid receiver address: {receiver}")
            return None

        receiver_pubkey = Pubkey.from_string(receiver)
        tx_entry = self._create_transaction_entry(receiver, amount)
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                self._validate_balance(amount)
                message = self.create_transfer_instructions(receiver_pubkey, amount)
                fee = self.estimate_fees(message)
                
                tx_entry.update({
                    'attempt': attempt + 1,
                    'fee': fee,
                    'status': 'pending'
                })
                
                transaction = VersionedTransaction(message, [self.sender_keypair])
                txid = self._send_transaction(transaction)
                tx_entry['txid'] = txid
                
                if self._confirm_transaction(txid):
                    tx_entry['status'] = 'confirmed'
                    self._notify_success(tx_entry)
                    return txid
                
            except TransactionExpiredBlockheightExceededError:
                self._handle_blockhash_expired(tx_entry, attempt)
            except (SolanaRpcException, RPCException) as e:
                self._handle_transaction_error(e, tx_entry, attempt)
                
            self._apply_retry_delay(attempt)
        
        self._handle_final_failure(tx_entry)
        return None

    def _create_transaction_entry(self, receiver: str, amount: int) -> Dict:
        """Create transaction history entry"""
        entry = {
            'receiver': receiver,
            'amount': amount,
            'timestamp': datetime.utcnow(),
            'attempts': 0,
            'status': 'initiated',
            'txid': None,
            'fee': 0
        }
        self.transaction_history.append(entry)
        return entry

    def _validate_balance(self, amount: int):
        """Validate sender balance with reserve"""
        sender_balance = self.get_balance(self.sender_pubkey)
        required_balance = amount + self.min_balance_reserve
        if sender_balance < required_balance:
            raise SolanaClientError(
                f"Insufficient balance: {sender_balance} < {required_balance}"
            )

    def _send_transaction(self, transaction: VersionedTransaction) -> Signature:
        """Send transaction with error handling"""
        try:
            opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
            response = self.client.send_transaction(transaction, opts)
            return response.value
        except RPCException as e:
            logger.error(f"Transaction submission failed: {str(e)}")
            raise

    def _confirm_transaction(self, txid: Signature) -> bool:
        """Confirm transaction with timeout"""
        start_time = time.time()
        while time.time() - start_time < Config.CONFIRMATION_TIMEOUT:
            status = self.client.get_transaction(txid).value
            if status and status.transaction.meta.err is None:
                return True
            time.sleep(2)
        return False

    def _handle_blockhash_expired(self, tx_entry: Dict, attempt: int):
        """Handle blockhash expiration scenario"""
        logger.warning(f"Blockhash expired, retrying (attempt {attempt+1})")
        tx_entry['status'] = 'retrying'
        self._last_blockhash = None  # Force refresh

    def _handle_transaction_error(self, error: Exception, tx_entry: Dict, attempt: int):
        """Handle transaction errors"""
        logger.error(f"Transaction failed: {str(error)}")
        tx_entry['errors'] = tx_entry.get('errors', []) + [str(error)]
        if "Blockhash not found" in str(error):
            self._last_blockhash = None

    def _apply_retry_delay(self, attempt: int):
        """Apply exponential backoff with jitter"""
        delay = Config.RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
        time.sleep(delay)

    def _handle_final_failure(self, tx_entry: Dict):
        """Handle final transaction failure"""
        logger.error(f"Transaction failed after {Config.MAX_RETRIES} attempts")
        tx_entry['status'] = 'failed'
        tx_entry['final_failure'] = True
        self._notify_failure(tx_entry)

    def _notify_success(self, tx_entry: Dict):
        """Handle successful transaction notification"""
        logger.info(f"Transaction confirmed: {tx_entry['txid']}")
        if Config.NOTIFICATION_WEBHOOK:
            self._send_webhook(tx_entry, success=True)

    def _notify_failure(self, tx_entry: Dict):
        """Handle failed transaction notification"""
        logger.error(f"Transaction failed: {tx_entry['receiver']}")
        if Config.NOTIFICATION_WEBHOOK:
            self._send_webhook(tx_entry, success=False)

    def _send_webhook(self, tx_entry: Dict, success: bool):
        """Send transaction notification to webhook"""
        try:
            import requests
            payload = {
                'status': 'success' if success else 'error',
                'receiver': tx_entry['receiver'],
                'amount': tx_entry['amount'],
                'txid': str(tx_entry.get('txid')),
                'attempts': tx_entry['attempt'],
                'timestamp': tx_entry['timestamp'].isoformat()
            }
            requests.post(Config.NOTIFICATION_WEBHOOK, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send webhook: {str(e)}")

    def clean_transaction_history(self, max_age: int = Config.HISTORY_RETENTION):
        """Clean up old transaction history entries"""
        cutoff = datetime.utcnow() - timedelta(seconds=max_age)
        self.transaction_history = [
            tx for tx in self.transaction_history
            if tx['timestamp'] > cutoff
        ]
