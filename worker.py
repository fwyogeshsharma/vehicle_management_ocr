"""The OCR worker: take rows out of the queue, read the photos, write the results back.

**It is a poller, not a web service.** The `vehicle_intake` row *is* the queue, so nothing is
lost when this process dies mid-job: the claim goes stale, the next poll re-queues it, and the
work is picked up again. FreightDesk's equivalent keeps its queue in memory inside the web
process, and its own docstring admits the consequence -- a row nobody picked up is invisible
until the application restarts.

**It talks to PostgreSQL and the object store directly.** No API in between. That means one
table with two writers in two languages, which is a real cost and is spelled out at the top of
``ocr/db.py``; what it buys is a worker that can be scaled, restarted and deployed with no
coupling to the API's request path, and no multi-megabyte photo travelling base64-encoded
through an HTTP response on its way to being decoded again.

Everything it needs to be told lives in ``ocr/config.py`` -- one declaration per setting,
with its default and the reason for it, which is the same job ``application.yml`` does on the
Java side. Put the values in a ``.env`` file at the repo root (copy ``.env.example``) or set
real environment variables, which win over the file. ``python worker.py --check`` prints what
it resolved to and stops.

There is no retry setting. A photo gets **one** attempt; if it fails, the row is FAILED and
nothing automatic will look at it again -- only a human pressing Read again in the worklist.
See ``ocr/db.py:record_failure`` for why.
"""
import logging
import signal
import sys
import time
from typing import Optional

from ocr import config
from ocr import db
from ocr.engine import PaddleEngine
from ocr.pipeline import read_truck
from ocr.storage import get_storage

log = logging.getLogger("worker")

POLL_SECONDS = config.POLL_SECONDS
BATCH = config.BATCH

_running = True


def _stop(signum, _frame):
    """Finish the job in hand, then exit.

    A claim abandoned mid-flight is reclaimed after the staleness cutoff anyway, but exiting
    cleanly means the common case -- a deploy -- does not leave a row waiting fifteen minutes.
    """
    global _running
    log.info("signal %s received; finishing the current job then stopping", signum)
    _running = False


def fetch_images(keys) -> list:
    """The bytes behind an intake's keys, skipping any that no longer resolve.

    Skipping rather than failing: one unreadable photo out of five should not lose the other
    four. An intake with *none* left is a failure, and `process` treats it as one.
    """
    store = get_storage()
    out = []
    for key in (keys or []):
        data = store.get(key)
        if data:
            out.append(data)
        else:
            log.warning("photo missing from the store: %s", key)
    return out


def process(engine, job: dict) -> None:
    intake_id = job["id"]
    try:
        images = fetch_images(job.get("image_keys"))
        if not images:
            db.record_failure(intake_id, "none of the uploaded photos could be read from storage")
            log.warning("intake %s failed: no usable photos", intake_id)
            return
        started = time.time()
        fields = read_truck(engine, images)
        db.record_success(intake_id, fields)
        log.info("intake %s DONE in %.1fs (plate=%s phone=%s)", intake_id,
                 time.time() - started, fields.get("plate"), fields.get("phone"))
    except Exception as e:
        # One bad photo must never stop the worker. The row absorbs the failure, and that is
        # the end of it -- there is no second attempt.
        log.exception("intake %s failed", intake_id)
        try:
            db.record_failure(intake_id, str(e))
        except Exception:
            # Could not even record the failure, so the row is still PROCESSING and this worker
            # is the only one that knows. Leaving it is right: the claim goes stale and the next
            # poll returns it to QUEUED for a fresh attempt. That is not a retry of a failure --
            # no verdict was ever recorded against this photo.
            log.exception("could not record the failure of intake %s", intake_id)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    check_only = "--check" in (sys.argv[1:] if argv is None else argv)

    # What we resolved to, always -- including which .env was read. "I changed .env and nothing
    # happened" is nearly always a worker that loaded a different one, or none at all, and that
    # is invisible unless the worker says so out loud.
    log.info("%s", config.describe())

    # Prove the database and the object store BEFORE loading a gigabyte of models: both of these
    # are one-line misconfigurations, and finding out about them after a 30-second model load is
    # a slow way to learn you typed the bucket name wrong.
    log.info("database: %s", db.check())
    log.info("photo storage: %s", get_storage().describe())

    if check_only:
        # --check stops here on purpose: everything above is configuration, everything below
        # costs half a minute of model loading and then starts taking work off the queue.
        log.info("--check: configuration is good, not starting the loop")
        db.close()
        return 0

    # Load the models before the first poll, so a missing or broken model is a loud failure at
    # startup rather than a per-job error nobody reads. FreightDesk loads lazily and silently
    # downloads on first use, which turns a network blip into "every report failed".
    log.info("loading OCR models...")
    engine = PaddleEngine()

    log.info("polling every %.1fs (batch %d, one attempt per photo, %dm stale cutoff)",
             POLL_SECONDS, BATCH, db.STALE_CLAIM_MINUTES)
    idle_logged = False

    while _running:
        try:
            jobs = db.claim(BATCH)
        except Exception as e:
            log.warning("could not reach the database (%s); retrying in %.1fs", e, POLL_SECONDS)
            time.sleep(POLL_SECONDS)
            continue

        if not jobs:
            if not idle_logged:
                log.info("nothing queued")
                idle_logged = True
            time.sleep(POLL_SECONDS)
            continue

        idle_logged = False
        for job in jobs:
            if not _running:
                # Leave it claimed. The staleness cutoff puts it back; re-queueing it here would
                # race the shutdown and is not worth the extra write path.
                break
            process(engine, job)

    db.close()
    log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
