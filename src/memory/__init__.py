from __future__ import annotations

from .provider.provider import DatabaseProvider
from .sqlite.database import SQLiteDatabase
from .sqlite.config import StorageConfig, DEFAULT_CONFIG

def get_database() -> DatabaseProvider:
    """Helper factory to obtain the active database provider."""
    return SQLiteDatabase()

__all__ = [
    "DatabaseProvider",
    "SQLiteDatabase",
    "StorageConfig",
    "DEFAULT_CONFIG",
    "get_database",
]
