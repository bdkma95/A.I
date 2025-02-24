from cryptography.fernet import Fernet

class CryptoManager:
    _key = Fernet.generate_key()
    
    @classmethod
    def encrypt(cls, data: str) -> bytes:
        return Fernet(cls._key).encrypt(data.encode())
    
    @classmethod
    def decrypt(cls, token: bytes) -> str:
        return Fernet(cls._key).decrypt(token).decode()
