"""
db.py — Async SQLAlchemy setup + ORM models for users and simulation runs.

If DATABASE_URL is unset the module exposes no-op helpers and `init_db`
returns immediately.  This keeps local dev working without Postgres
while production deploys get full history + auth.
"""
import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from sqlalchemy import (Column, String, Integer, Float, DateTime,
                        ForeignKey, Index, UniqueConstraint, select)
from sqlalchemy.dialects.postgresql import UUID, JSONB, insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
_ENABLED = bool(DATABASE_URL)

Base = declarative_base()
_engine = None
_Session: Optional[async_sessionmaker[AsyncSession]] = None


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email         = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    runs = relationship("SimulationRun", back_populates="user",
                        cascade="all, delete-orphan")


class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True),
                         ForeignKey("users.id", ondelete="CASCADE"),
                         index=True, nullable=False)
    session_id  = Column(String, index=True, nullable=False)
    engine_id   = Column(String, nullable=False)
    started_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen   = Column(DateTime, default=datetime.utcnow, nullable=False)
    config_snap = Column(JSONB, nullable=True)

    user   = relationship("User", back_populates="runs")
    cycles = relationship("SimulationCycle", back_populates="run",
                          cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("user_id", "session_id", name="uq_runs_user_session"),
    )


class SimulationCycle(Base):
    __tablename__ = "simulation_cycles"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    run_id        = Column(UUID(as_uuid=True),
                           ForeignKey("simulation_runs.id", ondelete="CASCADE"),
                           nullable=False)
    cycle_idx     = Column(Integer, nullable=False)
    fault_class   = Column(Integer)
    fault_name    = Column(String)
    confidence    = Column(Float)
    lambda_cur    = Column(Float)
    lambda_pred   = Column(Float)
    fuel_trim     = Column(Float)
    spark_adv     = Column(Float)
    twin_approved = Column(String)
    converged     = Column(String)
    request_body  = Column(JSONB)
    response_body = Column(JSONB)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("SimulationRun", back_populates="cycles")

    __table_args__ = (
        Index("ix_cycles_run_idx", "run_id", "cycle_idx"),
    )


# ── Engine bootstrap ──────────────────────────────────────────────────────────

async def init_db() -> None:
    global _engine, _Session
    if not _ENABLED:
        print("[db] DATABASE_URL not set — persistence + auth disabled.")
        return

    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    _engine = create_async_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    _Session = async_sessionmaker(_engine, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[db] schema ready.")


@asynccontextmanager
async def get_session():
    """Async context manager yielding an AsyncSession.  Raises if DB is disabled."""
    if not _ENABLED or _Session is None:
        raise RuntimeError("Database not initialised.")
    async with _Session() as s:
        yield s


# ── Persistence helpers ───────────────────────────────────────────────────────

# Serialize writes per session_id so the get-or-create step doesn't race
# under the simulation loop's burst of /simulate calls.  asyncio.Lock is
# scoped to this Python process, which is sufficient on uvicorn single-
# worker (HF Spaces' default).  The DB-level uq_runs_user_session
# constraint is the backstop if we ever scale to multiple workers.
_session_locks: dict[str, asyncio.Lock] = {}

def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


async def _upsert_run(s: AsyncSession, user_id: uuid.UUID,
                      session_id: str, engine_id: str) -> uuid.UUID:
    """Insert the run if it doesn't exist, else bump last_seen.

    Uses Postgres ON CONFLICT keyed on the unique constraint so this
    survives even without the in-process lock — and never produces dupes.
    """
    stmt = (
        pg_insert(SimulationRun)
        .values(user_id=user_id, session_id=session_id, engine_id=engine_id)
        .on_conflict_do_update(
            constraint="uq_runs_user_session",
            set_={"last_seen": datetime.utcnow()},
        )
        .returning(SimulationRun.id)
    )
    res = await s.execute(stmt)
    return res.scalar_one()


async def persist_cycle(user_id: uuid.UUID, session_id: str, engine_id: str,
                        cycle_idx: int, request_body: dict,
                        response_body: dict) -> None:
    if not _ENABLED or _Session is None:
        return
    try:
        async with _lock_for(session_id), _Session() as s:
            run_id = await _upsert_run(s, user_id, session_id, engine_id)
            r = response_body
            cycle = SimulationCycle(
                run_id        = run_id,
                cycle_idx     = cycle_idx,
                fault_class   = r.get("fault_class"),
                fault_name    = r.get("fault_name"),
                confidence    = r.get("fault_confidence"),
                lambda_cur    = r.get("lambda_current"),
                lambda_pred   = r.get("lambda_predicted"),
                fuel_trim     = (r.get("control_action") or {}).get("fuel_trim"),
                spark_adv     = (r.get("control_action") or {}).get("spark_advance"),
                twin_approved = str((r.get("twin") or {}).get("approved")),
                converged     = str(r.get("converged")),
                request_body  = request_body,
                response_body = response_body,
            )
            s.add(cycle)
            await s.commit()
    except Exception as e:
        print(f"[db] persist_cycle failed: {e}")


async def list_runs_for_user(user_id: uuid.UUID, limit: int = 50) -> list[dict]:
    if not _ENABLED or _Session is None:
        return []
    async with _Session() as s:
        res = await s.execute(
            select(SimulationRun)
                .where(SimulationRun.user_id == user_id)
                .order_by(SimulationRun.started_at.desc())
                .limit(limit)
        )
        return [{
            "id":         str(r.id),
            "session_id": r.session_id,
            "engine_id":  r.engine_id,
            "started_at": r.started_at.isoformat(),
            "last_seen":  r.last_seen.isoformat(),
        } for r in res.scalars().all()]


async def list_cycles_for_user(user_id: uuid.UUID, session_id: str,
                               limit: int = 500) -> list[dict]:
    if not _ENABLED or _Session is None:
        return []
    async with _Session() as s:
        run_res = await s.execute(
            select(SimulationRun).where(
                SimulationRun.session_id == session_id,
                SimulationRun.user_id == user_id,
            )
        )
        run = run_res.scalar_one_or_none()
        if run is None:
            return []
        res = await s.execute(
            select(SimulationCycle)
                .where(SimulationCycle.run_id == run.id)
                .order_by(SimulationCycle.cycle_idx.asc())
                .limit(limit)
        )
        return [{
            "cycle_idx":     c.cycle_idx,
            "fault_class":   c.fault_class,
            "fault_name":    c.fault_name,
            "confidence":    c.confidence,
            "lambda_cur":    c.lambda_cur,
            "lambda_pred":   c.lambda_pred,
            "fuel_trim":     c.fuel_trim,
            "spark_adv":     c.spark_adv,
            "twin_approved": c.twin_approved == "True",
            "converged":     c.converged == "True",
            "created_at":    c.created_at.isoformat(),
        } for c in res.scalars().all()]
