from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, Optional, overload
import os
import base64
import logging
from datetime import datetime, timedelta
from threading import RLock
from config.settings import settings

logger = logging.getLogger(__name__)

class CryptoManager:
    _key_registry = {}
    _current_version = 1
    _lock = RLock()
    
    def __init__(self):
        raise NotImplementedError("This class should not be instantiated")

    @classmethod
    def initialize(cls, key_rotation_days: int = 90):
        """Initialize with key rotation policy and existing keys"""
        cls._key_rotation_interval = timedelta(days=key_rotation_days)
        cls._load_keys_from_env()
        cls._rotate_key_if_needed()

    @classmethod
    def _load_keys_from_env(cls):
        """Load encryption keys from environment variables"""
        key_data = os.getenv("ENCRYPTION_KEYS")
        if key_data:
            for version, key in json.loads(key_data).items():
                cls._key_registry[int(version)] = key.encode()

    @classmethod
    def _rotate_key_if_needed(cls):
        """Automatically rotate keys based on rotation policy"""
        with cls._lock:
            if not cls._key_registry or \
               datetime.now() - cls._last_rotation > cls._key_rotation_interval:
                new_version = max(cls._key_registry.keys(), default=0) + 1
                new_key = Fernet.generate_key()
                cls._key_registry[new_version] = new_key
                cls._current_version = new_version
                cls._last_rotation = datetime.now()
                logger.info(f"Rotated to new key version {new_version}")

    @classmethod
    def _get_fernet(cls, version: Optional[int] = None) -> Fernet:
        """Get Fernet instance for specified or current key version"""
        version = version or cls._current_version
        return Fernet(cls._key_registry[version])

    @overload
    @classmethod
    def encrypt(cls, data: str) -> bytes: ...
    
    @overload
    @classmethod
    def encrypt(cls, data: bytes) -> bytes: ...
    
    @overload
    @classmethod
    def encrypt(cls, data: dict) -> bytes: ...
    
    @classmethod
    def encrypt(cls, data: Union[str, bytes, dict]) -> bytes:
        """Encrypt data with current key version"""
        with cls._lock:
            try:
                if isinstance(data, dict):
                    data = json.dumps(data).encode()
                elif isinstance(data, str):
                    data = data.encode()
                
                fernet = cls._get_fernet()
                token = fernet.encrypt(data)
                versioned_token = f"{cls._current_version}:{base64.urlsafe_b64encode(token).decode()}"
                logger.debug(f"Encrypted data with key version {cls._current_version}")
                return versioned_token.encode()
            except Exception as e:
                logger.error(f"Encryption failed: {str(e)}")
                raise EncryptionError from e

    @classmethod
    def decrypt(cls, token: bytes, version: Optional[int] = None) -> Union[str, bytes, dict]:
        """Decrypt data with automatic version detection"""
        with cls._lock:
            try:
                version_str, token_data = token.decode().split(":", 1)
                version = int(version_str)
                fernet = cls._get_fernet(version)
                decrypted = fernet.decrypt(base64.urlsafe_b64decode(token_data))
                
                try:
                    return json.loads(decrypted)
                except json.JSONDecodeError:
                    return decrypted.decode()
            except Exception as e:
                logger.error(f"Decryption failed: {str(e)}")
                raise DecryptionError from e

    @classmethod
    def reencrypt_old_data(cls, old_token: bytes) -> bytes:
        """Re-encrypt data with current key version"""
        decrypted = cls.decrypt(old_token)
        return cls.encrypt(decrypted)

    @classmethod
    def derive_key_from_password(cls, password: str, salt: bytes = None) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        salt = salt or os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

class EncryptionError(Exception):
    """Custom exception for encryption failures"""

class DecryptionError(Exception):
    """Custom exception for decryption failures"""

# Initialize with settings from configuration
CryptoManager.initialize(
    key_rotation_days=settings.KEY_ROTATION_DAYS
)
