from ai_os.shared.auth import Authenticator, Principal, Role, issue_token

SECRET = "test-secret-long-enough-for-hs256-signing"
AUTH = Authenticator(lambda: SECRET)


def bearer(subject: str = "rep-a", role: Role = Role.REP) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(SECRET, subject, role)}"}


async def allow_all(user: Principal, lead_id: str) -> bool:
    return True


async def deny_all(user: Principal, lead_id: str) -> bool:
    return False
