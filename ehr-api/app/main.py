"""EHR HTTP API for the Prosper Health clinic.

Run: ``uv run uvicorn app.main:app --reload --port 8000``
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.database import acquire, close_pool, init_pool
from app.routers import appointments, patients

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "001_initial.sql"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    # Apply the schema on startup so the app is self-bootstrapping for dev.
    async with acquire() as conn:
        await conn.execute(MIGRATION.read_text())
    yield
    await close_pool()


app = FastAPI(title="Prosper Health EHR", version="0.1.0", lifespan=lifespan)
app.include_router(patients.router)
app.include_router(appointments.router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
