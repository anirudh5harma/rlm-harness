from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Optional

from rlm_harness.memory.embed import (
    Embedder,
    HashingEmbedder,
    tokenize,
)

SCHEMA_VERSION = 1
VALID_RECALL_ROLES = {"user", "assistant", "tool", "reflection", "system"}


class MemoryError(RuntimeError):
    pass


class MemoryValidationError(MemoryError, ValueError):
    pass


@dataclass(frozen=True)
class CoreMemory:
    key: str
    value: str
    updated_at: int


@dataclass(frozen=True)
class RecallEvent:
    id: int
    thread_id: str
    ts: int
    role: str
    content: str
    tokens: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ArchivalMemory:
    id: int
    kind: str
    source_thread: Optional[str]
    ts: int
    content: str
    tokens: int
    metadata: dict[str, Any]
    embedding_model: str
    embedding_dim: int


@dataclass(frozen=True)
class MemorySearchResult:
    memory: ArchivalMemory
    score: float


class Memory:
    def __init__(
        self,
        path: Path,
        embedder: Optional[Embedder] = None,
        now: Optional[Callable[[], float]] = None,
    ):
        self.path = path
        self.embedder = embedder or HashingEmbedder()
        self._now = now or time.time
        self._lock = threading.RLock()

        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        try:
            self._initialize_database()
        except (sqlite3.DatabaseError, MemoryError) as exc:
            self._connection.close()
            raise MemoryError(f"failed to initialize memory database at {path}: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def vector_backend(self) -> str:
        return "sqlite-vec"

    def core_set(self, key: str, value: str) -> CoreMemory:
        key = self._validate_nonempty("key", key)
        value = self._validate_nonempty("value", value)
        updated_at = self._timestamp()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO core (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value = excluded.value,
                  updated_at = excluded.updated_at
                """,
                (key, value, updated_at),
            )
        return CoreMemory(key=key, value=value, updated_at=updated_at)

    def core_get(self, key: str) -> Optional[str]:
        key = self._validate_nonempty("key", key)
        row = self._connection.execute("SELECT value FROM core WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def core_item(self, key: str) -> Optional[CoreMemory]:
        key = self._validate_nonempty("key", key)
        row = self._connection.execute(
            "SELECT key, value, updated_at FROM core WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else CoreMemory(**dict(row))

    def recall_append(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        ts: Optional[int] = None,
    ) -> RecallEvent:
        thread_id = self._validate_nonempty("thread_id", thread_id)
        role = self._validate_role(role)
        content = self._validate_nonempty("content", content)
        timestamp = self._timestamp(ts)
        tokens = count_tokens(content)
        metadata_json = self._encode_metadata(metadata)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recall (thread_id, ts, role, content, tokens, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, timestamp, role, content, tokens, metadata_json),
            )
            event_id = int(cursor.lastrowid)
        return RecallEvent(
            id=event_id,
            thread_id=thread_id,
            ts=timestamp,
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata or {},
        )

    def recall_page(self, thread_id: str, query: str = "", k: int = 5) -> list[RecallEvent]:
        thread_id = self._validate_nonempty("thread_id", thread_id)
        limit = self._validate_limit(k)
        rows = self._connection.execute(
            """
            SELECT id, thread_id, ts, role, content, tokens, metadata_json
            FROM recall
            WHERE thread_id = ?
            ORDER BY ts DESC, id DESC
            LIMIT ?
            """,
            (thread_id, max(limit * 8, 20)),
        ).fetchall()
        events = [recall_from_row(row) for row in rows]
        if not query.strip():
            return events[:limit]

        query_tokens = set(tokenize(query))
        scored = [
            (recall_relevance(event.content, query_tokens), event.ts, event.id, event)
            for event in events
        ]
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [event for _, _, _, event in scored[:limit]]

    def archival_add(
        self,
        kind: str,
        content: str,
        source_thread: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        ts: Optional[int] = None,
    ) -> ArchivalMemory:
        kind = self._validate_nonempty("kind", kind)
        content = self._validate_nonempty("content", content)
        source_thread = source_thread.strip() if source_thread else None
        timestamp = self._timestamp(ts)
        tokens = count_tokens(content)
        metadata_json = self._encode_metadata(metadata)
        embedding = self._embed(content)
        embedding_blob = encode_vector(embedding)
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO archival_meta (
                  kind, source_thread, ts, content, tokens, metadata_json,
                  embedding_model, embedding_dim
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    source_thread,
                    timestamp,
                    content,
                    tokens,
                    metadata_json,
                    self.embedder.model,
                    self.embedder.dim,
                ),
            )
            archival_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT OR REPLACE INTO archival_vec(rowid, embedding) VALUES (?, ?)",
                (archival_id, embedding_blob),
            )
        return ArchivalMemory(
            id=archival_id,
            kind=kind,
            source_thread=source_thread,
            ts=timestamp,
            content=content,
            tokens=tokens,
            metadata=metadata or {},
            embedding_model=self.embedder.model,
            embedding_dim=self.embedder.dim,
        )

    def archival_get(self, memory_id: int) -> Optional[ArchivalMemory]:
        row = self._connection.execute(
            """
            SELECT id, kind, source_thread, ts, content, tokens, metadata_json,
                   embedding_model, embedding_dim
            FROM archival_meta
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()
        return None if row is None else archival_from_row(row)

    def archival_search(
        self,
        query: str,
        k: int = 5,
        kind: Optional[str] = None,
        source_thread: Optional[str] = None,
    ) -> list[MemorySearchResult]:
        query = self._validate_nonempty("query", query)
        limit = self._validate_limit(k)
        query_embedding = self._embed(query)
        query_tokens = set(tokenize(query))

        return self._archival_search_sqlite_vec(
            query_embedding,
            query_tokens,
            limit,
            kind=kind,
            source_thread=source_thread,
        )

    def _configure_connection(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if str(self.path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
        self._load_sqlite_vec()

    def _load_sqlite_vec(self) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:
            raise MemoryError(
                "sqlite-vec is required for memory search; install it with "
                "`python -m pip install sqlite-vec`"
            ) from exc

        extension_loading_enabled = False
        try:
            self._connection.enable_load_extension(True)
            extension_loading_enabled = True
            sqlite_vec.load(self._connection)
            self._connection.enable_load_extension(False)
            extension_loading_enabled = False
            self._connection.execute("SELECT vec_version()").fetchone()
        except AttributeError as exc:
            raise MemoryError(
                "sqlite-vec requires a Python sqlite3 build with extension loading support"
            ) from exc
        except (RuntimeError, sqlite3.DatabaseError) as exc:
            raise MemoryError(f"failed to load sqlite-vec extension: {exc}") from exc
        finally:
            if extension_loading_enabled:
                try:
                    self._connection.enable_load_extension(False)
                except (AttributeError, sqlite3.DatabaseError):
                    pass

    def _initialize_database(self) -> None:
        attempts = 6
        for attempt in range(attempts):
            try:
                self._configure_connection()
                self._migrate()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == attempts - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def _migrate(self) -> None:
        schema = resources.files("rlm_harness.memory").joinpath("schema.sql").read_text()
        with self._lock:
            try:
                self._connection.executescript(schema)
                self._connection.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS archival_vec
                    USING vec0(embedding float[{self.embedder.dim}])
                    """
                )
                applied = self._connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (SCHEMA_VERSION,),
                ).fetchone()
                if applied is None:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                        VALUES (?, ?)
                        """,
                        (SCHEMA_VERSION, self._timestamp()),
                    )
                self._connection.commit()
            except sqlite3.DatabaseError:
                self._connection.rollback()
                raise

    def _archival_search_sqlite_vec(
        self,
        query_embedding: list[float],
        query_tokens: set[str],
        limit: int,
        kind: Optional[str] = None,
        source_thread: Optional[str] = None,
    ) -> list[MemorySearchResult]:
        candidate_limit = max(limit * 20, 50)
        rows = self._connection.execute(
            """
            SELECT rowid, distance
            FROM archival_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (encode_vector(query_embedding), candidate_limit),
        ).fetchall()
        if not rows:
            return []

        results = []
        for row in rows:
            memory = self.archival_get(int(row["rowid"]))
            if memory is None:
                continue
            if kind and memory.kind != kind:
                continue
            if source_thread and memory.source_thread != source_thread:
                continue
            semantic_score = 1.0 / (1.0 + float(row["distance"]))
            lexical_score = recall_relevance(memory.content, query_tokens) * 0.05
            results.append(MemorySearchResult(memory=memory, score=semantic_score + lexical_score))
            if len(results) >= limit:
                break
        return results

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._connection
                self._connection.commit()
            except sqlite3.DatabaseError as exc:
                self._connection.rollback()
                raise MemoryError(f"memory database operation failed: {exc}") from exc

    def _embed(self, content: str) -> list[float]:
        vector = [float(value) for value in self.embedder.embed(content)]
        if len(vector) != self.embedder.dim:
            raise MemoryValidationError(
                f"embedder returned {len(vector)} values, expected {self.embedder.dim}"
            )
        return vector

    def _timestamp(self, ts: Optional[int] = None) -> int:
        if ts is not None:
            if ts < 0:
                raise MemoryValidationError("timestamp must be non-negative")
            return int(ts)
        return int(self._now())

    @staticmethod
    def _validate_nonempty(name: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise MemoryValidationError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _validate_role(role: str) -> str:
        role = Memory._validate_nonempty("role", role)
        if role not in VALID_RECALL_ROLES:
            allowed = ", ".join(sorted(VALID_RECALL_ROLES))
            raise MemoryValidationError(f"role must be one of: {allowed}")
        return role

    @staticmethod
    def _validate_limit(k: int) -> int:
        if k <= 0:
            raise MemoryValidationError("limit must be positive")
        return int(k)

    @staticmethod
    def _encode_metadata(metadata: Optional[dict[str, Any]]) -> str:
        try:
            return json.dumps(metadata or {}, sort_keys=True)
        except TypeError as exc:
            raise MemoryValidationError("metadata must be JSON serializable") from exc


def count_tokens(content: str) -> int:
    return max(1, len(tokenize(content)))


def recall_relevance(content: str, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    if not content_tokens:
        return 0.0
    overlap = query_tokens.intersection(content_tokens)
    return len(overlap) / len(query_tokens)


def encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def recall_from_row(row: sqlite3.Row) -> RecallEvent:
    return RecallEvent(
        id=int(row["id"]),
        thread_id=str(row["thread_id"]),
        ts=int(row["ts"]),
        role=str(row["role"]),
        content=str(row["content"]),
        tokens=int(row["tokens"]),
        metadata=json.loads(row["metadata_json"]),
    )


def archival_from_row(row: sqlite3.Row) -> ArchivalMemory:
    return ArchivalMemory(
        id=int(row["id"]),
        kind=str(row["kind"]),
        source_thread=row["source_thread"],
        ts=int(row["ts"]),
        content=str(row["content"]),
        tokens=int(row["tokens"]),
        metadata=json.loads(row["metadata_json"]),
        embedding_model=str(row["embedding_model"]),
        embedding_dim=int(row["embedding_dim"]),
    )
