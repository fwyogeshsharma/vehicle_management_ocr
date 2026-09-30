"""Every setting this worker reads, declared once.

This is the counterpart to ``vehicleManagement/src/main/resources/application.yml``, and it is
deliberately the same shape: one place that names each setting, gives it a default, and says
what it is for. Before this existed the eight settings were spread across ``db.py``,
``storage.py`` and ``worker.py`` as bare ``os.environ.get`` calls, so the only way to find out
what the worker could be configured with was to grep for it.

**Precedence, highest first** -- the same order Spring uses:

1. a real environment variable
2. a line in the ``.env`` file at the repo root
3. the default declared here

That order is what makes ``.env`` safe to keep on a developer's machine: a container or a
systemd unit sets real variables and they win, so nobody ships a laptop's password by accident.
``.env`` itself is gitignored; ``.env.example`` is the committed copy with no secrets in it.

**Two ways to say where the database is**, because the Java side has two and the single-string
form has a trap in it:

- ``VM_DB_URL`` -- one libpq connection string. Wins outright when set. Anything special in the
  password has to be percent-encoded, so ``faber@123`` must be written ``faber%40123``; get it
  wrong and libpq reads the ``@`` as the start of the host and the error names a host nobody
  typed.
- ``VM_DATABASE_HOST`` / ``_PORT`` / ``_NAME`` / ``_USER`` / ``_PASSWORD`` -- the parts, mirroring
  ``spring.datasource`` on the Java side. Assembled with :func:`psycopg.conninfo.make_conninfo`,
  which quotes each value itself, so the password is written exactly as it is and the encoding
  trap cannot happen. **Prefer this form.**

Nothing here reaches the network or the disk beyond reading ``.env``; importing this module is
free and safe from a test.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo

# The repo root: this file is <root>/ocr/config.py.
ROOT = Path(__file__).resolve().parent.parent

# Loaded at import, before any module reads a setting -- which is why every other module gets
# its values from here rather than from os.environ directly. `override=False` is what puts a
# real environment variable above the file.
#
# VM_ENV_FILE points somewhere else when one machine runs two workers against two databases.
ENV_FILE = Path(os.environ.get("VM_ENV_FILE") or (ROOT / ".env"))
ENV_FILE_LOADED = load_dotenv(ENV_FILE, override=False)


def _str(name: str, default: str = "") -> str:
    """A setting as text. Blank and unset are the same thing -- an empty line in a .env file is
    someone clearing a value, not setting it to the empty string."""
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _int(name: str, default: int) -> int:
    raw = _str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a whole number, not {raw!r}.")


def _float(name: str, default: float) -> float:
    raw = _str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number, not {raw!r}.")


# ── the database ────────────────────────────────────────────────────────────────────────────

DB_URL = _str("VM_DB_URL")
DB_HOST = _str("VM_DATABASE_HOST", "127.0.0.1")
# 127.0.0.1 rather than localhost, for the reason application.yml gives: on Windows localhost
# resolves to IPv6 ::1 first and a Docker-published Postgres listens on IPv4 only.
DB_PORT = _int("VM_DATABASE_PORT", 5432)
DB_NAME = _str("VM_DATABASE_NAME", "vehicle_management")
DB_USER = _str("VM_DATABASE_USER", "postgres")
DB_PASSWORD = _str("VM_DATABASE_PASSWORD")


def dsn() -> str:
    """The connection string, assembled or passed through.

    **No default for the credentials**, which is the one place this file departs from
    application.yml. Spring may default the password to ``postgres`` because a Spring Boot app
    that reaches the wrong database fails loudly at startup: Liquibase checksums the changelog
    and Hibernate validates every entity against the schema. This worker does neither. It would
    find a ``vehicle_intake`` table of the right shape in a copied or stale database and start
    claiming rows out of it, and nothing would look wrong until somebody asked where the results
    went. So an unset password is an error with instructions, not a guess.
    """
    if DB_URL:
        return DB_URL
    if not DB_PASSWORD:
        raise SystemExit(
            "No database credentials.\n"
            f"Set them in {ENV_FILE} (copy .env.example), or in the environment:\n"
            "  VM_DATABASE_PASSWORD=...          with the host/port/name/user defaults, or\n"
            "  VM_DB_URL=postgresql://user:pass@127.0.0.1:5432/vehicle_management\n"
            "This worker reads and writes vehicle_intake directly, so it will not guess.")
    # make_conninfo, not an f-string: it quotes each value, so a password with an @ or a space
    # in it needs no percent-encoding and cannot be misread as part of the host.
    return make_conninfo(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                         user=DB_USER, password=DB_PASSWORD)


def describe_db() -> str:
    """Where we are connecting, with no password in it. Safe to log."""
    if DB_URL:
        # Only ever the shape, never the string: a VM_DB_URL carries its password inline.
        return "VM_DB_URL (set)"
    return f"{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# The scratch database for pytest. A separate name on purpose -- the suite truncates
# vehicle_intake, and one variable for both would eventually do that to a real one.
TEST_DB_URL = _str("VM_TEST_DATABASE_URL")


# ── the queue ───────────────────────────────────────────────────────────────────────────────

# How long a claim may be held before the row is assumed abandoned. Must comfortably exceed the
# worst case for one upload, or a slow job is handed to a second worker while the first is still
# on it. Measured at ~10-14s per photo and at most five photos, so fifteen minutes is generous.
#
# This is NOT a retry. A worker killed mid-job never got a verdict on the photo, so its row goes
# back to QUEUED untouched -- `attempts` is not incremented and nothing is recorded against it.
# Without that, every deploy would permanently fail whatever was in flight.
STALE_CLAIM_MINUTES = _int("VM_STALE_MINUTES", 15)

POLL_SECONDS = _float("VM_POLL_SECONDS", 5.0)
BATCH = _int("VM_BATCH", 1)


# ── the photos ──────────────────────────────────────────────────────────────────────────────

# Must match vehicle-management.intake.storage-backend on the Java side: the API writes the
# keys and this worker reads them, so two different backends means a worker that fails every
# job for "photos no longer available" while the photos sit safely in the other one.
IMAGE_BACKEND = _str("VM_IMAGE_BACKEND", "local").lower()
IMAGE_DIR = _str("VM_IMAGE_DIR", "./uploads")
GCS_BUCKET = _str("VM_GCS_BUCKET")
GCS_PREFIX = _str("VM_GCS_PREFIX").strip("/")


def describe() -> str:
    """One line for the startup log, so a misconfigured worker says so before it claims
    anything. Deliberately includes the .env path: "I edited .env and nothing changed" is
    almost always a worker that loaded a different one, or none."""
    where = f"{ENV_FILE} ({'loaded' if ENV_FILE_LOADED else 'not found'})"
    store = (f"gcs://{GCS_BUCKET}/{GCS_PREFIX}".rstrip("/")
             if IMAGE_BACKEND == "gcs" else f"local {IMAGE_DIR}")
    return (f"config from {where}\n"
            f"  database  {describe_db()}\n"
            f"  photos    {store}\n"
            f"  polling   every {POLL_SECONDS}s, {BATCH} at a time, "
            f"stale claims reclaimed after {STALE_CLAIM_MINUTES}m")
