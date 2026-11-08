from dataclasses import dataclass
from datetime import datetime, timedelta

ROLES = frozenset({"rep", "team_lead", "manager"})
MAX_FAILED_LOGINS = 5
LOCKOUT = timedelta(minutes=15)


@dataclass(frozen=True)
class User:
    subject: str
    display_name: str
    role: str
    password_hash: str
    active: bool = True
    # Tokens issued before this moment are refused: "log out everywhere".
    sessions_valid_after: datetime | None = None
    failed_logins: int = 0
    locked_until: datetime | None = None

    def is_locked(self, now: datetime) -> bool:
        return self.locked_until is not None and now < self.locked_until


@dataclass(frozen=True)
class IssuedToken:
    token: str
    subject: str
    role: str
    display_name: str
    expires_at: datetime
