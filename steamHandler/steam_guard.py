"""Steam Guard 2FA code generation - AUTO-STEAM-RENT style."""
import base64
import hmac
import hashlib
import struct
import time


def generate_2fa_code(shared_secret: str) -> str:
    """
    Generate Steam Guard 2FA code from shared secret.
    AUTO-STEAM-RENT style implementation.
    """
    try:
        secret_bytes = base64.b64decode(shared_secret)
        timestamp = int(time.time()) // 30
        msg = struct.pack(">Q", timestamp)
        hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
        offset = hmac_hash[-1] & 0x0F
        code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
        alphabet = "23456789BCDFGHJKMNPQRTVWXY"
        code = ""
        for _ in range(5):
            code += alphabet[code_int % len(alphabet)]
            code_int //= len(alphabet)
        return code
    except Exception as e:
        raise ValueError(f"Failed to generate 2FA code: {e}")
