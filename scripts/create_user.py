"""Creates a login. The first manager is made here; after that, managers use /auth/users.

Usage: python scripts/create_user.py manager-a "Mona Manager" manager
The password is read from AI_OS_NEW_PASSWORD, or asked for without echo.
"""

import argparse
import asyncio
import getpass
import os

from ai_os.bootstrap.identity import build_identity
from ai_os.modules.identity.domain.errors import InvalidUser, UserAlreadyExists
from ai_os.modules.identity.domain.passwords import WeakPassword
from ai_os.shared.auth import Role
from ai_os.shared.config import Settings


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("display_name")
    parser.add_argument("role", choices=[role.value for role in Role])
    arguments = parser.parse_args()
    password = os.environ.get("AI_OS_NEW_PASSWORD") or getpass.getpass("Password: ")
    try:
        user = await build_identity(Settings()).create_user(
            arguments.username, arguments.display_name, arguments.role, password
        )
    except (InvalidUser, WeakPassword, UserAlreadyExists, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Created {user.subject} ({user.role})")


if __name__ == "__main__":
    asyncio.run(main())
