
"""Minimal SQLite backend used to obtain a SQLAlchemy engine for JASPAR.

This is purposely small because the DL backend only needs the shared engine.
"""
import os
from .base import BackendBase

try:
    from sqlalchemy import create_engine
except Exception as e:
    raise RuntimeError("sqlalchemy is required for the SQLite backend") from e


def _default_sqlite_path():
    """Resolve a default JASPAR SQLite path.
    Order of preference:
      1) Environment variable JASPAR_DB_PATH
      2) A file named 'JASPAR2026.sqlite' next to this package (two dirs up)
      3) A file named 'JASPAR.sqlite' next to this package (fallback)
    """
    env = os.getenv("JASPAR_DB_PATH")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(__file__)
    candidates = [
        os.path.abspath(os.path.join(here, "../data", "JASPAR2026.sqlite")),
        #os.path.abspath(os.path.join(here, "..", "JASPAR.sqlite")),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Last resort: do not fail here—return the first candidate; engine creation may fail later.
    return candidates[0]


class SQLiteBackend(BackendBase):
    """Core SQLite backend—exposes a shared SQLAlchemy engine."""
    def __init__(self, engine=None, **opts):
        if engine is None:
            db_path = _default_sqlite_path()
            engine = create_engine(f"sqlite:///{db_path}")
        super().__init__(engine=engine, **opts)
