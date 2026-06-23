import os
from contextlib import asynccontextmanager

import asyncpg
from dotenv import load_dotenv

load_dotenv()

_pool: asyncpg.Pool | None = None


async def init_pool(dsn: str | None = None) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(dsn or os.environ["DATABASE_URL"], min_size=2, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised")
    return _pool


@asynccontextmanager
async def acquire():
    async with get_pool().acquire() as conn:
        yield conn
