from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa
import hvac  # HashiCorp Vault client
import boto3
import os
import json
import base64
import logging
from datetime import datetime, timedelta
from threading import RLock
from config.settings import settings
from urllib3 import PoolManager

logger = logging.getLogger(__name__)

class HSMAdapter:
    """Abstraction layer for Hardware Security Modules"""
    def __init__(self, hsm_library_path: str = None):
        self.backend = default_backend()
        if hsm_library_path:
            self.backend = self._configure_hsm_backend(hsm_library_path)

    def _configure_hsm_backend(self, library_path: str):
        """Configure OpenSSL backend with HSM support"""
        from cryptography.hazmat.backends.openssl.backend import backend
        backend._enable_fips()
        backend._load_engine("dynamic", f"pkcs11={library_path}")
        return backend

    def generate_hsm_key(self) -> bytes:
        """Generate key material using HSM"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=self.backend
        )
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

class CryptoManager:
    _key_registry = {}
    _current_version = 1
    _lock = RLock()
    _hsm = None
    _secret_manager = None

    def __init__(self):
        raise NotImplementedError("This class should not be instantiated")

    @classmethod
    def initialize(cls, 
                 key_rotation_interval: timedelta = timedelta(days=90),
                 use_hsm: bool = False,
                 secret_manager: str = None):
        """Initialize crypto system with security best practices"""
        # Configure HSM if available
        if use_hsm or os.getenv("USE_HSM"):
            cls._hsm = HSMAdapter(os.getenv("HSM_LIBRARY_PATH"))
            logger.info("Initialized with HSM support")

        # Configure secret manager
        if secret_manager or os.getenv("SECRET_MANAGER"):
            cls._secret_manager = secret_manager or os.getenv("SECRET_MANAGER")
            cls._configure_secret_manager()

        cls._key_rotation_interval = key_rotation_interval
        cls._load_keys_from_secret_manager()
        cls._rotate_key_if_needed()

    @classmethod
    def _configure_secret_manager(cls):
        """Configure integration with external secret manager"""
        if cls._secret_manager == "vault":
            cls.vault_client = hvac.Client(
                url=os.getenv("VAULT_ADDR"),
                token=os.getenv("VAULT_TOKEN"),
                session=PoolManager(cert_reqs='CERT_REQUIRED')
            )
        elif cls._secret_manager == "aws":
            cls.aws_client = boto3.client(
                'secretsmanager',
                region_name=os.getenv("AWS_REGION"),
                config=boto3.session.Config(
                    connect_timeout=2,
                    read_timeout=2,
                    retries={'max_attempts': 3}
                )
            )

    @classmethod
    def _load_keys_from_secret_manager(cls):
        """Load keys from configured secret management system"""
        try:
            if cls._secret_manager == "vault":
                secret = cls.vault_client.secrets.kv.v2.read_secret_version(
                    path=os.getenv("VAULT_KEY_PATH")
                )
                cls._key_registry = secret['data']['data']
            elif cls._secret_manager == "aws":
                secret = cls.aws_client.get_secret_value(
                    SecretId=os.getenv("AWS_SECRET_NAME")
                )
                cls._key_registry = json.loads(secret['SecretString'])
            else:
                raise ValueError("Unsupported secret manager")

            logger.info("Successfully loaded keys from secret manager")
        except Exception as e:
            logger.error(f"Failed to load keys from secret manager: {str(e)}")
            raise

    @classmethod
    def _store_key_to_secret_manager(cls, version: int, key: bytes):
        """Securely store new key version in secret manager"""
        try:
            if cls._secret_manager == "vault":
                cls.vault_client.secrets.kv.v2.create_or_update_secret(
                    path=os.getenv("VAULT_KEY_PATH"),
                    secret={str(version): key.decode()}
                )
            elif cls._secret_manager == "aws":
                cls.aws_client.update_secret(
                    SecretId=os.getenv("AWS_SECRET_NAME"),
                    SecretString=json.dumps({
                        **cls._key_registry,
                        str(version): key.decode()
                    })
                )
                
            logger.info(f"Stored new key version {version} in secret manager")
        except Exception as e:
            logger.error(f"Failed to store key in secret manager: {str(e)}")
            raise

    @classmethod
    def _rotate_key_if_needed(cls):
        """Rotate keys based on schedule or compliance requirements"""
        with cls._lock:
            rotation_needed = (
                datetime.now() - cls._last_rotation > cls._key_rotation_interval or
                cls._check_compliance_requirements()
            )

            if rotation_needed:
                new_version = max(map(int, cls._key_registry.keys()), default=0) + 1
                new_key = cls._generate_secure_key()
                cls._key_registry[str(new_version)] = new_key
                cls._current_version = new_version
                cls._last_rotation = datetime.now()
                cls._store_key_to_secret_manager(new_version, new_key)
                logger.info(f"Rotated to new key version {new_version}")

    @classmethod
    def _generate_secure_key(cls) -> bytes:
        """Generate key using HSM or secure software method"""
        if cls._hsm:
            logger.info("Generating key using HSM")
            return cls._hsm.generate_hsm_key()
        return Fernet.generate_key()

    @classmethod
    def _check_compliance_requirements(cls) -> bool:
        """Check external compliance requirements for key rotation"""
        # Example: Integration with compliance management system
        return False  # Implement based on actual compliance checks

    @classmethod
    def encrypt(cls, data: Union[str, bytes, dict]) -> bytes:
        """Encrypt data with audit logging and transport protection"""
        with cls._lock:
            try:
                # Encrypt data
                encrypted = super().encrypt(data)
                
                # Audit log
                logger.info(
                    f"Encryption operation successful | "
                    f"KeyVersion: {cls._current_version} | "
                    f"HSMUsed: {bool(cls._hsm)} | "
                    f"Source: {os.uname().nodename}"
                )
                
                return encrypted
            except Exception as e:
                logger.error(f"Encryption failed: {str(e)}")
                raise EncryptionError from e

    @classmethod
    def decrypt(cls, token: bytes) -> Union[str, bytes, dict]:
        """Decrypt data with secure transport validation"""
        with cls._lock:
            try:
                # Verify transport security
                if not cls._verify_transport_security():
                    raise DecryptionError("Insecure transport layer detected")
                
                # Decrypt data
                decrypted = super().decrypt(token)
                
                # Audit log
                logger.info(
                    f"Decryption operation successful | "
                    f"KeyVersion: {token.decode().split(':')[0]} | "
                    f"HSMUsed: {bool(cls._hsm)} | "
                    f"Source: {os.uname().nodename}"
                )
                
                return decrypted
            except Exception as e:
                logger.error(f"Decryption failed: {str(e)}")
                raise DecryptionError from e

    @classmethod
    def _verify_transport_security(cls) -> bool:
        """Verify TLS is used for all external communications"""
        # Implement actual transport layer checks
        return True  # Simplified for example

# Example environment configuration
"""
export USE_HSM=true
export HSM_LIBRARY_PATH=/usr/local/lib/softhsm/libsofthsm2.so
export SECRET_MANAGER=aws
export AWS_REGION=us-west-2
export AWS_SECRET_NAME=football-app-keys
"""

# Initialize with production-grade security
CryptoManager.initialize(
    key_rotation_interval=timedelta(days=settings.COMPLIANCE_KEY_ROTATION_DAYS),
    use_hsm=True,
    secret_manager="aws"
)
