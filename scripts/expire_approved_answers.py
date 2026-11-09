"""Takes approved answers past their review period out of search. Run daily."""

import asyncio

from ai_os.bootstrap.escalation import build_escalations
from ai_os.shared.config import Settings


async def main() -> None:
    expired = await build_escalations(Settings()).expire_due()
    print("expired:", len(expired))


if __name__ == "__main__":
    asyncio.run(main())
