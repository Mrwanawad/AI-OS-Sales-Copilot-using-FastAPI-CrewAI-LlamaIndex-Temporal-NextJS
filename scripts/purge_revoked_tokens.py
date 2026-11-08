"""Deletes logout records for tokens that have expired anyway. Run daily."""

import asyncio

from ai_os.bootstrap.identity import build_identity
from ai_os.shared.config import Settings


async def main() -> None:
    print("purged:", await build_identity(Settings()).purge_expired_revocations())


if __name__ == "__main__":
    asyncio.run(main())
