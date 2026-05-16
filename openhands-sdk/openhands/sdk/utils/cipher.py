"""
Cipher utility for preventing accidental secret disclosure in serialized data

SECURITY WARNINGS:
- The secret key is a string for ease of use but should contain at least 256
  bits of entropy
"""

import hashlib
from base64 import b64encode
from typing import Final

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr


# Fernet token prefix used to distinguish ciphertext from legacy plaintext.
# Do not shorten: a 5-char prefix collides with realistic base64 plaintext.
FERNET_TOKEN_PREFIX: Final[str] = "gAAAAA"


_MIN_SECRET_KEY_LENGTH = 12


class Cipher:
    """
    Simple encryption utility for preventing accidental secret disclosure.

    The secret key is stored internally as ``SecretStr`` to prevent
    accidental disclosure via ``vars(cipher)`` or debug introspection.

    The key must be at least ``_MIN_SECRET_KEY_LENGTH`` characters to
    provide meaningful brute-force resistance when combined with the
    fast SHA256-based key derivation.
    """

    def __init__(self, secret_key: str):
        if len(secret_key) < _MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"Secret key must be at least "
                f"{_MIN_SECRET_KEY_LENGTH} characters "
                f"(got {len(secret_key)})."
            )
        self._secret_key = SecretStr(secret_key)
        self._fernet: Fernet | None = None

    def __repr__(self) -> str:
        return f"Cipher(id={id(self)})"

    def encrypt(self, secret: SecretStr | None) -> str | None:
        if secret is None:
            return None
        secret_value = secret.get_secret_value().encode()
        fernet = self._get_fernet()
        result = fernet.encrypt(secret_value).decode()
        return result

    def decrypt(self, secret: str | None) -> SecretStr | None:
        """
        Decrypt a secret value, returning None if decryption fails.

        This handles cases where existing conversations were serialized with different
        encryption keys or contain invalid encrypted data. A warning is logged when
        decryption fails and a None is returned. This mimics the case where
        no cipher was defined so secrets where redacted.
        """
        if secret is None:
            return None
        try:
            fernet = self._get_fernet()
            decrypted = fernet.decrypt(secret.encode()).decode()
            return SecretStr(decrypted)
        except InvalidToken as e:
            # Import here to avoid circular imports
            from openhands.sdk.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"Failed to decrypt secret value (setting to None): {e}. "
                "This may occur when loading conversations encrypted with a different "
                "key or when upgrading from older versions."
            )
            return None

    def try_decrypt_str(self, value: str) -> str | None:
        """Decrypt to a string, or ``None`` on InvalidToken (no logging)."""
        try:
            return self._get_fernet().decrypt(value.encode()).decode()
        except InvalidToken:
            return None

    def _get_fernet(self):
        if self._fernet is None:
            secret_key = self._secret_key.get_secret_value().encode()
            # Hash the key to make sure we have a 256 bit value
            fernet_key = b64encode(hashlib.sha256(secret_key).digest())
            self._fernet = Fernet(fernet_key)
        return self._fernet
