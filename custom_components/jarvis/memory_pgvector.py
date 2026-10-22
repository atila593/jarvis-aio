"""Opt-in pgvector backend for JARVIS long-term memory.

SQLite FTS (and ChromaDB when present) stay the zero-infra default. This backend
is used ONLY when the owner explicitly opts in — ``memory_backend: "pgvector"``
plus a ``memory_pgvector_dsn`` — for a household whose memory has outgrown
SQLite. It keeps JARVIS's promise: ``psycopg`` is a deliberately OPTIONAL
dependency (NOT in the manifest), so a stock install is unchanged, and if the
driver isn't installed or Postgres/pgvector is unreachable, every entry point
here fails soft and ``memory.py`` falls back to the default backend.

It runs in the executor (``memory.py`` is always called off the event loop via
``async_add_executor_job``), so the blocking psycopg calls and the blocking
embedding request are both fine here. Vectors are passed as ``'[…]'::vector``
text literals, so only ``psycopg`` is needed — not the ``pgvector`` helper
package.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

_LOGGER = logging.getLogger(__name__)

_DEFAULT_DIM = 768          # nomic-embed-text; overridden by the first real embed
_TABLE = "jarvis_memory"

_ready = False
_dim: Optional[int] = None


def _cfg(key: str, default):
    try:
        from . import jarvis_config
        val = jarvis_config.get(key, default)
        return val if val is not None else default
    except Exception:
        return default


def _dsn() -> str:
    """The Postgres DSN for the pgvector store (e.g.
    ``postgresql://user:pass@host:5432/jarvis``). Empty when unset."""
    return str(_cfg("memory_pgvector_dsn", "") or "").strip()


def configured() -> bool:
    """True only when the owner has explicitly selected pgvector AND given a DSN.
    Pure config read — never touches the network or the driver."""
    return (str(_cfg("memory_backend", "auto")).strip().lower() == "pgvector"
            and bool(_dsn()))


def _vec_literal(vec: list[float]) -> str:
    """A pgvector text literal: ``[0.1,0.2,…]`` (cast to ::vector in SQL)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _embed(text: str) -> Optional[list[float]]:
    """Embed one string synchronously via the same Ollama endpoint/model the
    async embeddings layer uses. Executor-safe (blocking urllib). None on any
    failure so the caller fails soft."""
    try:
        import urllib.request
        from . import embeddings
        base = embeddings._ollama_base()
        model = embeddings._model()
        if not base:
            return None
        payload = json.dumps({"model": model, "input": [text]}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/embed", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        embs = data.get("embeddings")
        if embs and isinstance(embs, list) and embs[0]:
            return [float(x) for x in embs[0]]
    except Exception as exc:
        _LOGGER.debug("pgvector embed failed: %s", exc)
    return None


def _connect():
    """Open a psycopg connection. Tries psycopg (v3) then psycopg2. Raises on
    failure (callers guard)."""
    try:
        import psycopg
        return psycopg.connect(_dsn())
    except ImportError:
        import psycopg2
        return psycopg2.connect(_dsn())


def init() -> bool:
    """Create the extension + table if needed and mark the backend ready. Returns
    False (fail-soft) when not opted in, the driver is missing, or Postgres is
    unreachable — memory.py then falls back to its default backend."""
    global _ready, _dim
    if not configured():
        return False
    try:
        # Fix the vector dimension once: from a probe embedding if we can get one,
        # else the configured/default. A vector column's size is set at creation.
        probe = _embed("jarvis memory dimension probe")
        dim = len(probe) if probe else int(_cfg("memory_pgvector_dim", _DEFAULT_DIM))
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
                    "  id text PRIMARY KEY,"
                    "  content text NOT NULL,"
                    "  role text,"
                    "  device_id text,"
                    "  conversation_id text,"
                    "  ts timestamptz,"
                    f"  embedding vector({dim})"
                    ")")
            conn.commit()
        finally:
            conn.close()
        _dim = dim
        _ready = True
        _LOGGER.info("JARVIS memory: pgvector backend ready (dim=%d)", dim)
        return True
    except Exception as exc:
        _LOGGER.warning("JARVIS memory: pgvector init failed (%s) — falling back", exc)
        _ready = False
        return False


def available() -> bool:
    return _ready


def store(doc_id: str, text: str, metadata: dict) -> bool:
    """Embed and upsert one memory row. False on any failure (caller falls back)."""
    if not _ready:
        return False
    try:
        vec = _embed(text)
        if not vec:
            return False
        if _dim is not None and len(vec) != _dim:
            _LOGGER.debug("pgvector store skipped: dim %d != %d", len(vec), _dim)
            return False
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {_TABLE} "
                    "(id, content, role, device_id, conversation_id, ts, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::vector) "
                    "ON CONFLICT (id) DO NOTHING",
                    (doc_id, text, metadata.get("role", ""),
                     metadata.get("device_id", ""), metadata.get("conversation_id", ""),
                     metadata.get("timestamp") or None, _vec_literal(vec)))
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as exc:
        _LOGGER.debug("pgvector store failed: %s", exc)
        return False


def search(query: str, k: int = 5, hours: Optional[int] = None) -> list[dict]:
    """Nearest-neighbour search by cosine distance. [] on any failure."""
    if not _ready:
        return []
    try:
        vec = _embed(query)
        if not vec:
            return []
        params: list = [_vec_literal(vec)]
        where = ""
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            where = "WHERE ts >= %s"
            params.append(cutoff)
        params.append(_vec_literal(vec))
        params.append(int(k))
        sql = (f"SELECT content, role, ts, 1 - (embedding <=> %s::vector) AS score "
               f"FROM {_TABLE} {where} "
               "ORDER BY embedding <=> %s::vector LIMIT %s")
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()
        out = []
        for row in rows:
            content, role, ts, score = row[0], row[1], row[2], row[3]
            out.append({
                "text": content,
                "role": role or "",
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else (ts or ""),
                "score": round(float(score), 3) if score is not None else 0,
            })
        return out
    except Exception as exc:
        _LOGGER.debug("pgvector search failed: %s", exc)
        return []


def count() -> int:
    if not _ready:
        return 0
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {_TABLE}")
                row = cur.fetchone()
        finally:
            conn.close()
        return int(row[0]) if row else 0
    except Exception as exc:
        _LOGGER.debug("pgvector count failed: %s", exc)
        return 0


def reset() -> None:
    """Test helper — clear the ready/dim module state."""
    global _ready, _dim
    _ready = False
    _dim = None
