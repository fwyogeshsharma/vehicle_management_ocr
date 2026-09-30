"""Reading photo bytes back out of wherever the API put them.

A port of FreightDesk's ``pipeline/storage.py``, trimmed to what a worker needs: this process
only ever **reads**. Nothing here writes or deletes an object, and that is deliberate — the API
owns what goes into the store, the bucket's lifecycle rule owns what leaves it, and a worker
that could also delete would be a third opinion about a photo a telecaller may still need.

Two backends, chosen by ``VM_IMAGE_BACKEND`` and matching the Java side's two names exactly:

- ``local`` (dev) — files under ``VM_IMAGE_DIR``. Only works when this process shares a
  filesystem with the API, which in practice means one machine.
- ``gcs`` (prod) — a Google Cloud Storage bucket. ``VM_GCS_BUCKET`` required,
  ``VM_GCS_PREFIX`` optional. Authenticates with Application Default Credentials, so on GCP
  there is no key file to hand around.

**Keys are opaque.** This module stores and fetches exactly what it is handed, and never parses
or rebuilds one. That discipline is what let FreightDesk change its object layout twice with no
migration, and it is what lets the same key resolve here and in the Java service.
"""
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class LocalStorage:
    """Photos on local disk. Only correct when the API writes to the same filesystem."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base = Path(base_dir or os.environ.get("VM_IMAGE_DIR") or "./uploads").resolve()

    def get(self, key: str) -> Optional[bytes]:
        p = (self.base / key).resolve()
        # A key comes out of the database, which now has two writers. Refuse one that climbs
        # out of the base directory rather than trusting the row.
        if not str(p).startswith(str(self.base)):
            log.warning("refusing unsafe storage key: %r", key)
            return None
        return p.read_bytes() if p.is_file() else None

    def describe(self) -> str:
        return f"local disk at {self.base}"


class GCSStorage:
    """Photos in a Google Cloud Storage bucket. The same bucket the Java API writes to."""

    def __init__(self, bucket: Optional[str] = None, prefix: Optional[str] = None):
        bucket = bucket or os.environ.get("VM_GCS_BUCKET")
        if not bucket:
            raise ValueError("VM_GCS_BUCKET must be set when VM_IMAGE_BACKEND=gcs")
        # Imported here, not at module scope, so local development needn't install the SDK.
        from google.cloud import storage

        self._bucket_name = bucket
        self._bucket = storage.Client().bucket(bucket)
        self.prefix = (prefix if prefix is not None
                       else os.environ.get("VM_GCS_PREFIX", "")).strip("/")

    def get(self, key: str) -> Optional[bytes]:
        name = f"{self.prefix}/{key}" if self.prefix else key
        blob = self._bucket.blob(name)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def describe(self) -> str:
        return f"gs://{self._bucket_name}/{self.prefix}" if self.prefix else f"gs://{self._bucket_name}"


_storage = None


def get_storage():
    """The backend named by VM_IMAGE_BACKEND, built once.

    An unrecognised value raises rather than falling back to local disk. A silent fallback would
    mean a worker that starts cleanly, finds nothing in an empty local directory, and fails
    every job for "photos no longer available" while the photos sit safely in the bucket.
    """
    global _storage
    if _storage is None:
        backend = os.environ.get("VM_IMAGE_BACKEND", "local").strip().lower()
        if backend == "gcs":
            _storage = GCSStorage()
        elif backend == "local":
            _storage = LocalStorage()
        else:
            raise ValueError(
                f"unknown VM_IMAGE_BACKEND: {backend!r} (expected 'local' or 'gcs')")
        log.info("photo storage: %s", _storage.describe())
    return _storage


def reset_storage() -> None:
    """Drop the cached backend. Tests, and config changes within one process."""
    global _storage
    _storage = None
