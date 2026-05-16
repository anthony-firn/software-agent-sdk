"""Tests for the Cipher utility class."""

from base64 import urlsafe_b64encode

from cryptography.fernet import Fernet
from pydantic import SecretStr

from openhands.sdk.utils.cipher import Cipher


def test_cipher_encrypt_decrypt():
    """Test basic encryption and decryption functionality."""
    # Generate a proper Fernet key
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    secret = SecretStr("my-secret-api-key")

    # Test encryption
    encrypted = cipher.encrypt(secret)
    assert encrypted is not None
    assert encrypted != secret.get_secret_value()
    assert isinstance(encrypted, str)

    # Test decryption
    decrypted = cipher.decrypt(encrypted)
    assert decrypted is not None
    assert decrypted.get_secret_value() == secret.get_secret_value()


def test_cipher_encrypt_none():
    """Test that encrypting None returns None."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    result = cipher.encrypt(None)
    assert result is None


def test_cipher_decrypt_none():
    """Test that decrypting None returns None."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    result = cipher.decrypt(None)
    assert result is None


def test_cipher_decrypt_invalid_data():
    """Test that decrypting invalid data returns None and logs warning."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    # Test with completely invalid data
    result = cipher.decrypt("invalid-encrypted-data")
    assert result is None

    # Test with malformed base64
    result = cipher.decrypt("not-base64!")
    assert result is None


def test_cipher_decrypt_wrong_key():
    """Test that decrypting with wrong key returns None and logs warning."""
    # Create two different keys
    key1 = urlsafe_b64encode(b"a" * 32).decode("ascii")
    key2 = urlsafe_b64encode(b"b" * 32).decode("ascii")

    cipher1 = Cipher(key1)
    cipher2 = Cipher(key2)

    secret = SecretStr("test-secret")

    # Encrypt with first cipher
    encrypted = cipher1.encrypt(secret)
    assert encrypted is not None

    # Try to decrypt with second cipher (wrong key)
    result = cipher2.decrypt(encrypted)
    assert result is None


def test_cipher_fernet_caching():
    """Test that Fernet instance is cached properly."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    # Get Fernet instance twice
    fernet1 = cipher._get_fernet()
    fernet2 = cipher._get_fernet()

    # Should be the same instance (cached)
    assert fernet1 is fernet2
    assert isinstance(fernet1, Fernet)


def test_cipher_with_real_fernet_key():
    """Test cipher with a real Fernet-generated key."""
    # Generate a proper Fernet key
    fernet_key = Fernet.generate_key()
    key = fernet_key.decode("ascii")

    cipher = Cipher(key)
    secret = SecretStr("test-api-key-12345")

    # Test round-trip encryption/decryption
    encrypted = cipher.encrypt(secret)
    decrypted = cipher.decrypt(encrypted)

    assert decrypted is not None
    assert decrypted.get_secret_value() == secret.get_secret_value()


def test_cipher_multiple_encryptions_different():
    """Test that multiple encryptions of the same value produce different results."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    secret = SecretStr("same-secret")

    # Encrypt the same secret multiple times
    encrypted1 = cipher.encrypt(secret)
    encrypted2 = cipher.encrypt(secret)

    # Results should be different (due to Fernet's built-in randomness)
    assert encrypted1 != encrypted2

    # But both should decrypt to the same value
    decrypted1 = cipher.decrypt(encrypted1)
    decrypted2 = cipher.decrypt(encrypted2)

    assert decrypted1 is not None
    assert decrypted2 is not None

    assert decrypted1.get_secret_value() == secret.get_secret_value()
    assert decrypted2.get_secret_value() == secret.get_secret_value()


def test_cipher_empty_string():
    """Test encryption/decryption of empty string."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    secret = SecretStr("")

    encrypted = cipher.encrypt(secret)
    assert encrypted is not None
    assert encrypted != ""

    decrypted = cipher.decrypt(encrypted)
    assert decrypted is not None
    assert decrypted.get_secret_value() == ""


def test_cipher_unicode_content():
    """Test encryption/decryption of unicode content."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    secret = SecretStr("🔐 Secret with émojis and ñoñ-ASCII chars! 中文")

    encrypted = cipher.encrypt(secret)
    decrypted = cipher.decrypt(encrypted)

    assert decrypted is not None
    assert decrypted.get_secret_value() == secret.get_secret_value()


def test_cipher_long_content():
    """Test encryption/decryption of long content."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    # Create a long secret (1KB)
    long_secret = "x" * 1024
    secret = SecretStr(long_secret)

    encrypted = cipher.encrypt(secret)
    decrypted = cipher.decrypt(encrypted)

    assert decrypted is not None
    assert decrypted.get_secret_value() == long_secret


def test_cipher_decrypt_only_catches_invalid_token():
    """decrypt() must only catch InvalidToken, not arbitrary exceptions.

    Catching ``Exception`` is too broad and can silently swallow bugs
    like MemoryError or TypeError from programming errors.
    """
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    # Verify decrypt catches InvalidToken (legitimate crypto failure)
    result = cipher.decrypt("garbage-data-that-cant-be-decrypted")
    assert result is None  # gracefully handled

    # Verify that a simulated arbitrary Exception is NOT caught.
    # We use a mock to prove the function doesn't have except Exception.
    original = cipher._get_fernet

    class _BadFernet:
        def decrypt(self, data):
            raise RuntimeError("simulated non-crypto failure")

    cipher._get_fernet = lambda: _BadFernet()  # type: ignore[assignment]

    try:
        cipher.decrypt("any-value")
    except RuntimeError:
        pass  # expected — broad except would have swallowed this
    else:
        raise AssertionError(
            "decrypt() should NOT catch arbitrary RuntimeError exceptions. "
            "The except clause is still too broad (likely 'except Exception')."
        )
    finally:
        cipher._get_fernet = original


def test_cipher_secret_key_not_plaintext_in_vars():
    """The encryption key must not be readable as plaintext from ``vars()``.

    Storing the secret key as a bare ``str`` on the instance means anyone
    with access to the Cipher object can extract the key via ``vars(cipher)``
    or ``cipher.__dict__``. The key should be stored via ``SecretStr`` so
    that accidental logging, debugging, or introspection cannot trivially
    expose it.
    """
    from pydantic import SecretStr

    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    # The key must NOT appear as a plain string anywhere in __dict__.
    for attr, value in vars(cipher).items():
        if isinstance(value, str) and key in value:
            raise AssertionError(
                f"Cipher.{attr} exposes the secret key as plain str: {value!r}"
            )

    # The key must be stored as SecretStr (masked by repr).
    stored = vars(cipher).get("_secret_key")
    assert isinstance(stored, SecretStr), (
        f"Cipher._secret_key must be SecretStr, got {type(stored).__name__}"
    )
    # SecretStr repr must not expose the value.
    assert key not in repr(stored), (
        f"SecretStr repr leaks the key: {repr(stored)}"
    )


def test_cipher_repr_does_not_leak_key():
    """``repr(cipher)`` must not expose the encryption key."""
    key = urlsafe_b64encode(b"a" * 32).decode("ascii")
    cipher = Cipher(key)

    r = repr(cipher)
    assert key not in r, (
        f"repr(cipher) leaks the secret key: {r}"
    )
