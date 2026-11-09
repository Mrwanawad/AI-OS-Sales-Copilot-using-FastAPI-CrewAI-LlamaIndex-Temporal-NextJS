"""Deletes personal data past its PDPL retention period. Run daily."""

import asyncio

from ai_os.bootstrap.crm import build_privacy
from ai_os.shared.config import Settings


async def main() -> None:
    print(await build_privacy(Settings()).apply_retention())


if __name__ == "__main__":
    asyncio.run(main())
