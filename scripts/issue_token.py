"""Issues an API token without a password, for scripts and local debugging.

Users log in at POST /auth/login. When a database is configured, the API refuses tokens
for usernames that have no account, so create one first with scripts/create_user.py.

Usage: python scripts/issue_token.py rep-a rep --hours 8
Reads AUTH_SECRET from the environment or .env.
"""

import argparse
from datetime import timedelta

from ai_os.shared.auth import Role, issue_token
from ai_os.shared.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    parser.add_argument("role", choices=[role.value for role in Role])
    parser.add_argument("--hours", type=float, default=8)
    arguments = parser.parse_args()

    secret = Settings().auth_secret
    if secret is None:
        raise SystemExit("Set AUTH_SECRET first")
    print(
        issue_token(
            secret.get_secret_value(),
            arguments.subject,
            Role(arguments.role),
            timedelta(hours=arguments.hours),
        )
    )


if __name__ == "__main__":
    main()
