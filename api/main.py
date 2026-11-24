"""Event log API: the vision pipeline POSTs each counted defect here.

Storage is SQLAlchemy, so the same code runs on SQLite (quick local runs, tests)
and PostgreSQL (docker compose). Pick the database with DATABASE_URL, e.g.
    postgresql+psycopg://user:pass@localhost:5432/steelvision

Run locally (from repo root):
    python -m uvicorn api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
Operator dashboard: http://127.0.0.1:8000/
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_URL = "sqlite:///./steelvision.db"


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)  # stored as UTC
    camera: Mapped[str] = mapped_column(String(32), index=True)
    defect_class: Mapped[str] = mapped_column(String(32), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    frame: Mapped[float] = mapped_column(Float)          # frame index at which the line was crossed
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class EventIn(BaseModel):
    camera: str = Field(min_length=1, max_length=32, examples=["cam1"])
    defect_class: str = Field(min_length=1, max_length=32, examples=["scratches"])
    track_id: int
    frame: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    ts: datetime | None = Field(default=None, description="defaults to server time (UTC)")


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: datetime
    camera: str
    defect_class: str
    track_id: int
    frame: float
    confidence: float | None


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _out(e: Event) -> EventOut:
    return EventOut(id=e.id, ts=_utc(e.ts), camera=e.camera, defect_class=e.defect_class,
                    track_id=e.track_id, frame=e.frame, confidence=e.confidence)


def create_app(database_url: str | None = None) -> FastAPI:
    url = database_url or os.environ.get("DATABASE_URL", DEFAULT_URL)
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool  # one shared in-memory DB across requests
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)

    def get_session():
        with session_factory() as session:
            yield session

    app = FastAPI(title="steel-vision event log", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/events", response_model=EventOut, status_code=201)
    def add_event(body: EventIn, db: Session = Depends(get_session)):
        e = Event(ts=_utc(body.ts) if body.ts else datetime.now(timezone.utc), camera=body.camera,
                  defect_class=body.defect_class, track_id=body.track_id, frame=body.frame,
                  confidence=body.confidence)
        db.add(e)
        db.commit()
        return _out(e)

    @app.get("/events", response_model=list[EventOut])
    def list_events(db: Session = Depends(get_session), limit: int = Query(50, ge=1, le=1000),
                    camera: str | None = None, defect_class: str | None = None):
        q = select(Event).order_by(Event.id.desc()).limit(limit)
        if camera:
            q = q.where(Event.camera == camera)
        if defect_class:
            q = q.where(Event.defect_class == defect_class)
        return [_out(e) for e in db.scalars(q)]

    @app.get("/stats")
    def stats(db: Session = Depends(get_session)):
        by_class = dict(db.execute(select(Event.defect_class, func.count()).group_by(Event.defect_class)).all())
        by_camera = dict(db.execute(select(Event.camera, func.count()).group_by(Event.camera)).all())
        return {"total": sum(by_class.values()), "by_class": by_class, "by_camera": by_camera}

    dashboard_file = Path(__file__).parent / "static" / "dashboard.html"

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return FileResponse(dashboard_file, media_type="text/html")

    return app


app = create_app()