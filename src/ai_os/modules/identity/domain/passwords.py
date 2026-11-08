import hashlib
import hmac
import secrets

MIN_PASSWORD_LENGTH = 12
# scrypt from the standard library: memory-hard, and no extra dependency to audit.
_N, _R, _P = 2**14, 8, 1
_SCHEME = "scrypt"


class WeakPassword(ValueError):
    pass


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPassword(f"Passwords need at least {MIN_PASSWORD_LENGTH} characters")
    salt = secrets.token_bytes(16)
    digest = _derive(password, salt, _N, _R, _P)
    return f"{_SCHEME}${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != _SCHEME:
            return False
        actual = _derive(password, bytes.fromhex(salt), int(n), int(r), int(p))
    except ValueError:
        return False
    return hmac.compare_digest(actual, bytes.fromhex(digest))


# Verified against when the user doesn't exist, so timing doesn't reveal valid usernames.
DUMMY_HASH = f"{_SCHEME}${_N}${_R}${_P}${'00' * 16}$" + "00" * 64


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, maxmem=64 * 1024 * 1024, dklen=64
    )
