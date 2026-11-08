class InvalidCredentials(Exception):
    """One message for unknown users, wrong passwords and disabled accounts."""

    def __init__(self) -> None:
        super().__init__("Wrong username or password")


class AccountLocked(Exception):
    def __init__(self) -> None:
        super().__init__("Too many failed logins. Try again later")


class UserAlreadyExists(Exception):
    def __init__(self, subject: str) -> None:
        super().__init__(f"User {subject} already exists")


class UserNotFound(Exception):
    def __init__(self, subject: str) -> None:
        super().__init__(f"No user {subject}")


class InvalidUser(ValueError):
    pass
