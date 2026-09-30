"""The queue: claiming work from `vehicle_intake` and writing results back.

This module is the whole of the worker's contact with PostgreSQL, and it touches exactly one
table and exactly the machine half of it -- ``processing_status``, ``processing_error``,
``processed_at``, ``claimed_at``, ``attempts`` and the ``ocr_*`` columns. It never reads or
writes ``review_status``, ``vehicle_id`` or anything else a CSR owns, and it never touches
``vehicles``. That boundary is the only thing keeping a two-writer table honest, so:

**If you add a column here, add it to the Java entity too.** `ddl-auto: validate` on the API
will catch a Java field with no column; nothing at all will catch the reverse. The schema is
defined in one place -- ``vehicleManagement/src/main/resources/db/changelog/changes/
005-vehicle-intake.sql`` -- and this file must follow it.

The claim is the reason this file exists rather than an HTTP call. FreightDesk's worker sets
``PROCESSING`` with no condition on the current status and is safe only because exactly one
thread in one uvicorn process ever calls it; its own docstring says not to scale it without
fixing that first. Polling from a separate process *is* that scaling, so the guard is no longer
optional.
"""
import json
import logging
import threading
from contextlib import contextmanager
from typing import List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ocr import config

log = logging.getLogger(__name__)

# Declared in ocr/config.py, with every other setting and the reasoning behind this one.
# Read through this name because the whole module already refers to it by it.
STALE_CLAIM_MINUTES = config.STALE_CLAIM_MINUTES

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def dsn() -> str:
    """The connection string.

    Assembled by :func:`ocr.config.dsn` from either ``VM_DB_URL`` or the ``VM_DATABASE_*``
    parts, both of which may come from the ``.env`` file. Kept as a function rather than a
    constant so a test can monkeypatch the settings and re-open a pool.
    """
    return config.dsn()


def pool() -> ConnectionPool:
    """A small pool, opened once.

    Small on purpose: this process OCRs one photo at a time on a CPU, so more connections would
    only be idle ones holding server-side slots.

    **Locked, despite the worker being single-threaded.** A bare check-then-set here lets two
    threads each build a pool; the loser is garbage-collected, and psycopg_pool's finaliser then
    tries to join its own worker thread from inside that thread and raises. The concurrency test
    in tests/test_db.py found exactly that, and a future threaded worker would leak a pool of
    connections every time it raced.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(dsn(), min_size=1, max_size=3, open=True,
                                       kwargs={"row_factory": dict_row})
    return _pool


@contextmanager
def _tx():
    with pool().connection() as conn:
        with conn.transaction():
            yield conn


def claim(limit: int = 1) -> List[dict]:
    """Take the next batch of work, exclusively.

    Three things happen in **one transaction**, and the order matters:

    1. Rows whose worker died holding them are put back. Without this a crash strands work in
       PROCESSING until someone notices; FreightDesk only recovers such rows when its whole
       process restarts, which in practice means "at the next deploy".
    2. ``SELECT ... FOR UPDATE SKIP LOCKED`` locks the rows it returns and steps straight over
       any another transaction already holds. **This is the line that makes two workers safe.**
       Without SKIP LOCKED the second worker would block on the first's rows and then process
       them again; without FOR UPDATE it would not even block.

    3. The same transaction marks them PROCESSING. It has to be the same one -- the row locks
       are released at commit, so a claim that selected in one transaction and updated in
       another has a window where a second worker can select the same ids.

    Returns the claimed rows, each a dict with ``id`` and ``image_keys``.
    """
    with _tx() as conn:
        reclaimed = conn.execute(
            """
            UPDATE vehicle_intake
               SET processing_status = 'QUEUED', claimed_at = NULL
             WHERE processing_status = 'PROCESSING'
               AND claimed_at < now() - make_interval(mins => %s)
            """,
            (STALE_CLAIM_MINUTES,)).rowcount
        if reclaimed:
            log.warning("re-queued %d row(s) from a worker that stopped mid-job", reclaimed)

        rows = conn.execute(
            """
            SELECT id, image_keys
              FROM vehicle_intake
             WHERE processing_status = 'QUEUED'
             ORDER BY created_at
             FOR UPDATE SKIP LOCKED
             LIMIT %s
            """,
            (max(1, min(limit, 10)),)).fetchall()
        if not rows:
            return []

        ids = [r["id"] for r in rows]
        conn.execute(
            """
            UPDATE vehicle_intake
               SET processing_status = 'PROCESSING', claimed_at = now()
             WHERE id = ANY(%s)
            """,
            (ids,))
        return rows


def record_success(intake_id: int, fields: dict) -> None:
    """Write what OCR read and mark the job DONE.

    **DONE means "the machine has finished", not "the machine found something".** A photo that
    yielded nothing still lands here, because a CSR can very often read a plate the engine
    could not, and a row that never reached the worklist would simply be lost.

    Values are stored as read. Normalising them into the shape ``vehicles`` demands is the
    completion step's job, on the Java side, where a human is about to check them: a plate like
    ``MHI2AB1234`` is worth showing precisely because a CSR can see at a glance that the I wants
    to be a 1. Silently dropping it would leave them typing it off the photo instead.
    """
    with _tx() as conn:
        conn.execute(
            """
            UPDATE vehicle_intake
               SET processing_status = 'DONE',
                   processing_error  = NULL,
                   claimed_at        = NULL,
                   processed_at      = now(),
                   ocr_plate         = %s,
                   ocr_mobiles       = %s::jsonb,
                   ocr_company       = %s,
                   ocr_confidence    = %s,
                   ocr_raw           = %s::jsonb
             WHERE id = %s
            """,
            (_fit(fields.get("plate"), 32),
             # Every number found, not just the best-read one. A truck carries several and a
             # telecaller who cannot reach the first needs the second.
             json.dumps([n for n in (fields.get("phones") or []) if n]),
             _fit(fields.get("company"), 160),
             _fit(fields.get("plate_confidence"), 8),
             json.dumps(fields.get("raw") or {}),
             intake_id))


def record_failure(intake_id: int, message: str) -> None:
    """Give up on this photo. One attempt, no retry.

    **A failure here is final.** Whatever went wrong -- the photo would not decode, it was not in
    the store, the engine raised -- the row goes straight to FAILED and no worker will look at it
    again. Nothing automatic will pick it back up.

    That is a deliberate choice and it costs something: a transient fault, an object store
    unreachable for one second, permanently fails a photo that would have read perfectly a moment
    later. Two things make it tolerable. A row a worker was *killed* holding is not a failure at
    all -- it never reached this function, and the staleness reclaim returns it to QUEUED with
    `attempts` untouched. And a human can still press Read again in the worklist once whatever
    broke is fixed, which is what `attempts` counts.

    The alternative was worse in practice. Automatic retry only helps if the retry is spaced out,
    spacing means a row sits QUEUED for minutes looking like ordinary work, and a photo the engine
    simply cannot read -- about one in four of them -- pays that cost three times over to reach
    the same verdict. A CSR looking at the picture settles it in seconds.
    """
    with _tx() as conn:
        conn.execute(
            """
            UPDATE vehicle_intake
               SET processing_status = 'FAILED',
                   attempts          = attempts + 1,
                   processing_error  = %s,
                   claimed_at        = NULL,
                   processed_at      = now()
             WHERE id = %s
            """,
            (_fit(message, 500), intake_id))


def _fit(value, length: int):
    """Truncate to what the column holds. A write that overflows loses the whole result."""
    if value is None:
        return None
    text = str(value).strip()
    return text[:length] if text else None


def check() -> str:
    """Prove the connection and the schema at startup, and say what was found.

    Deliberately loud and early. A worker that starts cleanly and only discovers at the first
    job that it cannot see the table looks, from the outside, exactly like a quiet queue.
    """
    try:
        with pool().connection() as conn:
            row = conn.execute(
                """
                SELECT count(*) FILTER (WHERE processing_status = 'QUEUED')     AS queued,
                       count(*) FILTER (WHERE processing_status = 'PROCESSING') AS processing,
                       count(*)                                                 AS total
                  FROM vehicle_intake
                """).fetchone()
    except psycopg.errors.UndefinedTable as e:
        raise SystemExit(
            "vehicle_intake does not exist in that database. Has vehicleManagement run its "
            f"Liquibase changelog against it?\n  {e}")
    except psycopg.OperationalError as e:
        # Names where we tried, without the password: "connection refused" against an
        # unstated host is the least useful error this worker can produce.
        raise SystemExit(f"Cannot reach the database at {config.describe_db()}:\n  {e}")
    return (f"{row['total']} intake row(s): {row['queued']} queued, "
            f"{row['processing']} in progress")


def close() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
