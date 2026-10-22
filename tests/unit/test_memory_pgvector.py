"""Opt-in pgvector memory backend.

SQLite/Chroma stay the default; pgvector engages only when the owner selects it.
These tests use a fake psycopg connection and a fixed embedding so the SQL and
the fail-soft behaviour are exercised without a real Postgres, plus the
memory.py routing that prefers pgvector when active and falls back otherwise.
"""
import pytest


@pytest.fixture
def mp(load):
    m = load("memory_pgvector")
    m.reset()
    return m


class _FakeCursor:
    def __init__(self, rows=None):
        self.executed = []            # (sql, params)
        self._rows = rows or []
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def execute(self, sql, params=None):
        self.executed.append((sql, params))
    def fetchall(self):
        return list(self._rows)
    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, rows=None):
        self.cur = _FakeCursor(rows)
        self.committed = False
        self.closed = False
    def cursor(self):
        return self.cur
    def commit(self):
        self.committed = True
    def close(self):
        self.closed = True


def _config(mp, monkeypatch, **cfg):
    monkeypatch.setattr(mp, "_cfg", lambda k, d: cfg.get(k, d))


# ── configured() — pure config gate ───────────────────────────────────────────

def test_configured_requires_backend_and_dsn(mp, monkeypatch):
    _config(mp, monkeypatch, memory_backend="pgvector",
            memory_pgvector_dsn="postgresql://u@h/db")
    assert mp.configured() is True


def test_not_configured_when_backend_is_auto(mp, monkeypatch):
    _config(mp, monkeypatch, memory_backend="auto",
            memory_pgvector_dsn="postgresql://u@h/db")
    assert mp.configured() is False


def test_not_configured_without_dsn(mp, monkeypatch):
    _config(mp, monkeypatch, memory_backend="pgvector", memory_pgvector_dsn="")
    assert mp.configured() is False


# ── vector literal ─────────────────────────────────────────────────────────────

def test_vec_literal_format(mp):
    assert mp._vec_literal([0.1, 0.2, 0.3]).startswith("[") and mp._vec_literal([1, 2]).endswith("]")
    assert mp._vec_literal([1, 2]) == "[1.0,2.0]"


# ── init ───────────────────────────────────────────────────────────────────────

def test_init_creates_extension_and_table(mp, monkeypatch):
    _config(mp, monkeypatch, memory_backend="pgvector",
            memory_pgvector_dsn="postgresql://u@h/db")
    monkeypatch.setattr(mp, "_embed", lambda t: [0.1, 0.2, 0.3, 0.4])
    conn = _FakeConn()
    monkeypatch.setattr(mp, "_connect", lambda: conn)
    assert mp.init() is True
    assert mp.available() is True and mp._dim == 4
    sql_joined = " ".join(s for s, _ in conn.cur.executed)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql_joined
    assert "vector(4)" in sql_joined


def test_init_noop_when_not_configured(mp, monkeypatch):
    _config(mp, monkeypatch, memory_backend="auto")
    assert mp.init() is False and mp.available() is False


def test_init_fail_soft_when_driver_missing(mp, monkeypatch):
    _config(mp, monkeypatch, memory_backend="pgvector",
            memory_pgvector_dsn="postgresql://u@h/db")
    monkeypatch.setattr(mp, "_embed", lambda t: [0.1, 0.2])
    def _boom():
        raise ImportError("no psycopg")
    monkeypatch.setattr(mp, "_connect", _boom)
    assert mp.init() is False and mp.available() is False


# ── store ───────────────────────────────────────────────────────────────────────

def _ready(mp, monkeypatch, conn, dim=3):
    monkeypatch.setattr(mp, "_embed", lambda t: [0.1] * dim)
    monkeypatch.setattr(mp, "_connect", lambda: conn)
    mp._ready = True
    mp._dim = dim


def test_store_inserts_vector(mp, monkeypatch):
    conn = _FakeConn()
    _ready(mp, monkeypatch, conn, dim=3)
    ok = mp.store("id1", "hello world", {"role": "user", "timestamp": "2026-10-22T00:00:00"})
    assert ok is True and conn.committed
    sql, params = conn.cur.executed[-1]
    assert "INSERT INTO jarvis_memory" in sql and "%s::vector" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql
    assert params[0] == "id1" and params[-1] == "[0.1,0.1,0.1]"


def test_store_rejects_dim_mismatch(mp, monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(mp, "_connect", lambda: conn)
    monkeypatch.setattr(mp, "_embed", lambda t: [0.1, 0.1])   # len 2
    mp._ready = True
    mp._dim = 3                                               # expects 3
    assert mp.store("id1", "x", {}) is False
    assert conn.cur.executed == []                            # nothing written


def test_store_false_when_not_ready(mp):
    assert mp.store("id", "text", {}) is False


# ── search ───────────────────────────────────────────────────────────────────────

def test_search_orders_by_distance_and_maps_rows(mp, monkeypatch):
    rows = [("past msg", "assistant", "2026-10-21T10:00:00", 0.91)]
    conn = _FakeConn(rows=rows)
    _ready(mp, monkeypatch, conn, dim=3)
    out = mp.search("query", k=5)
    sql, params = conn.cur.executed[-1]
    assert "ORDER BY embedding <=> %s::vector" in sql and "LIMIT %s" in sql
    assert out == [{"text": "past msg", "role": "assistant",
                    "timestamp": "2026-10-21T10:00:00", "score": 0.91}]


def test_search_adds_time_window(mp, monkeypatch):
    conn = _FakeConn(rows=[])
    _ready(mp, monkeypatch, conn, dim=3)
    mp.search("q", k=3, hours=24)
    sql, _ = conn.cur.executed[-1]
    assert "WHERE ts >= %s" in sql


def test_search_empty_when_not_ready(mp):
    assert mp.search("q") == []


def test_count(mp, monkeypatch):
    conn = _FakeConn(rows=[(42,)])
    _ready(mp, monkeypatch, conn, dim=3)
    assert mp.count() == 42


# ── memory.py routing ──────────────────────────────────────────────────────────

def test_memory_prefers_pgvector_when_active(load, monkeypatch):
    memory = load("memory")
    mp = load("memory_pgvector")
    monkeypatch.setattr(memory, "_pgvector_available", True)
    monkeypatch.setattr(mp, "search",
                        lambda q, k=5, hours=None: [{"text": "from pg", "role": "user",
                                                     "timestamp": "", "score": 1.0}])
    out = memory.search_memory("anything", k=3)
    assert out == [{"text": "from pg", "role": "user", "timestamp": "", "score": 1.0}]


def test_memory_store_routes_to_pgvector(load, monkeypatch):
    memory = load("memory")
    mp = load("memory_pgvector")
    monkeypatch.setattr(memory, "_pgvector_available", True)
    calls = []
    monkeypatch.setattr(mp, "store",
                        lambda doc_id, text, meta: calls.append((doc_id, text)) or True)
    assert memory.store_memory("a real memory line") is True
    assert len(calls) == 1
