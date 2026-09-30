"""The queue, against a real PostgreSQL.

**Not mocked, and not SQLite.** Every behaviour worth testing here is a property of PostgreSQL
itself -- ``FOR UPDATE SKIP LOCKED``, the row locks a transaction holds until commit, an UPDATE
that reads its own ``attempts`` column. A mock would assert that we sent the strings we sent.

Point ``VM_TEST_DATABASE_URL`` at a **scratch database**: these tests TRUNCATE ``vehicle_intake``.
It must never be the database the application is using, and never production.

    set VM_TEST_DATABASE_URL=postgresql://postgres:pw@127.0.0.1:5432/vm_ocr_test
    .venv\\Scripts\\python -m pytest tests/test_db.py -v

The schema comes from vehicleManagement's Liquibase changelog; create it with

    mvn liquibase:update -Dliquibase.url=jdbc:postgresql://127.0.0.1:5432/vm_ocr_test ...

Skipped entirely when the variable is unset, so ``pytest tests/`` still runs everywhere.
"""
import json
import threading

import pytest

from ocr import config

# Through config, so a scratch database can be named in .env like everything else -- and so
# this stays the ONLY variable the suite will connect to. Pointing it at VM_DB_URL would
# truncate a real vehicle_intake.
TEST_URL = config.TEST_DB_URL

pytestmark = pytest.mark.skipif(
    not TEST_URL, reason="VM_TEST_DATABASE_URL is not set (needs a scratch PostgreSQL)")

psycopg = pytest.importorskip("psycopg")


@pytest.fixture()
def db(monkeypatch):
    """A clean table and a fresh module-level pool for every test."""
    monkeypatch.setenv("VM_DB_URL", TEST_URL)
    from ocr import db as module

    module.close()
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute("TRUNCATE vehicle_intake RESTART IDENTITY CASCADE")
    yield module
    module.close()


def queue_one(keys=("intake/x/0.jpg",), **columns) -> int:
    """Insert one QUEUED row straight into the table and return its id."""
    cols = {"image_keys": json.dumps(list(keys)), **columns}
    names = ", ".join(cols)
    holders = ", ".join(["%s"] * len(cols))
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        row = conn.execute(
            f"INSERT INTO vehicle_intake ({names}) VALUES ({holders}) RETURNING id",
            list(cols.values())).fetchone()
    return row[0]


def read(intake_id: int) -> dict:
    from psycopg.rows import dict_row
    with psycopg.connect(TEST_URL, autocommit=True, row_factory=dict_row) as conn:
        return conn.execute(
            "SELECT * FROM vehicle_intake WHERE id = %s", (intake_id,)).fetchone()


class TestClaim:

    def test_a_claim_takes_the_row_and_marks_it_processing(self, db):
        intake_id = queue_one()
        claimed = db.claim(1)
        assert [j["id"] for j in claimed] == [intake_id]
        assert claimed[0]["image_keys"] == ["intake/x/0.jpg"]

        row = read(intake_id)
        assert row["processing_status"] == "PROCESSING"
        assert row["claimed_at"] is not None

    def test_the_oldest_row_goes_first(self, db):
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            older = conn.execute(
                "INSERT INTO vehicle_intake (image_keys, created_at) "
                "VALUES ('[\"a\"]'::jsonb, now() - interval '1 hour') RETURNING id").fetchone()[0]
            conn.execute("INSERT INTO vehicle_intake (image_keys) VALUES ('[\"b\"]'::jsonb)")
        assert [j["id"] for j in db.claim(1)] == [older]

    def test_an_empty_queue_is_an_empty_list_not_an_error(self, db):
        assert db.claim(5) == []

    def test_two_workers_never_get_the_same_row(self, db):
        """The one test FreightDesk's design cannot pass.

        Its worker sets PROCESSING with no condition on the current status, so two of them
        polling at the same instant both take the same row and both OCR it. Here SKIP LOCKED
        makes the batches disjoint: one thread may legitimately get everything, but no id may
        appear on both sides.
        """
        ids = {queue_one(keys=(f"intake/{i}/0.jpg",)) for i in range(6)}
        taken: list[list[int]] = []
        barrier = threading.Barrier(2)

        def grab():
            barrier.wait()
            taken.append([j["id"] for j in db.claim(6)])

        threads = [threading.Thread(target=grab) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first, second = taken
        assert not set(first) & set(second), "the same row was handed to both workers"
        assert set(first) | set(second) == ids

    def test_a_row_a_worker_died_holding_comes_back(self, db):
        """Without this a crash strands work until someone notices.

        FreightDesk recovers such rows only when its whole process restarts, which in practice
        means "at the next deploy".
        """
        intake_id = queue_one()
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute(
                "UPDATE vehicle_intake SET processing_status = 'PROCESSING', "
                "claimed_at = now() - make_interval(mins => %s) WHERE id = %s",
                (db.STALE_CLAIM_MINUTES + 1, intake_id))

        assert [j["id"] for j in db.claim(1)] == [intake_id]

    def test_a_claim_still_within_the_cutoff_is_left_alone(self, db):
        intake_id = queue_one()
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute(
                "UPDATE vehicle_intake SET processing_status = 'PROCESSING', claimed_at = now() "
                "WHERE id = %s", (intake_id,))
        assert db.claim(1) == []


class TestResults:

    def test_success_writes_the_reads_and_marks_it_done(self, db):
        intake_id = queue_one()
        db.claim(1)
        db.record_success(intake_id, {
            "plate": "PB10ES8185", "phones": ["9878770969", "8437036313"],
            "company": "GURU NANAK ROADCARRIER",
            "plate_confidence": "LOW", "raw": {"body_texts": ["GURU NANAK"]},
        })

        row = read(intake_id)
        assert row["processing_status"] == "DONE"
        assert row["ocr_plate"] == "PB10ES8185"
        # Both numbers off the side of the truck, not just the best-read one.
        assert row["ocr_mobiles"] == ["9878770969", "8437036313"]
        assert row["ocr_confidence"] == "LOW"
        assert row["ocr_raw"] == {"body_texts": ["GURU NANAK"]}
        assert row["claimed_at"] is None
        assert row["processed_at"] is not None

    def test_reading_nothing_is_still_done(self, db):
        """DONE means the machine finished, not that it found something.

        On real field photos a plate comes back on about one in four. A row that never reached
        the worklist because OCR read nothing would simply be lost -- and a CSR can very often
        read a plate the engine could not.
        """
        intake_id = queue_one()
        db.claim(1)
        db.record_success(intake_id, {"plate": None, "phones": [], "company": None,
                                      "plate_confidence": "NONE", "raw": {}})
        row = read(intake_id)
        assert row["processing_status"] == "DONE"
        assert row["ocr_plate"] is None
        assert row["ocr_mobiles"] == [], "no numbers is an empty list, never null"

    def test_a_value_too_long_for_its_column_is_cut_not_dropped(self, db):
        """A single overlong read must not cost the whole result.

        Without the truncation in `_fit` the UPDATE raises, `record_success` never lands, and
        the row is retried until it FAILS -- losing the plate and the phone number as well.
        """
        intake_id = queue_one()
        db.claim(1)
        db.record_success(intake_id, {
            "plate": "PB10ES8185", "phones": ["9878770969"], "company": "X" * 400,
            "plate_confidence": "LOW", "raw": {},
        })
        row = read(intake_id)
        assert len(row["ocr_company"]) == 160
        assert row["ocr_plate"] == "PB10ES8185"

    def test_one_failure_is_final(self, db):
        """No automatic retry: whatever went wrong, the row is FAILED on the first attempt.

        Retrying was attractive because most faults are transient. It was not worth keeping
        because a photo the engine cannot read stays unreadable -- and on real field photos that
        is about one in four. Three spaced-out attempts only delayed the same verdict, while a
        CSR with the picture open settles it in seconds.
        """
        intake_id = queue_one()
        db.claim(1)
        db.record_failure(intake_id, "the store blinked")

        row = read(intake_id)
        assert row["processing_status"] == "FAILED"
        assert row["attempts"] == 1
        assert row["processing_error"] == "the store blinked"
        assert row["processed_at"] is not None
        assert row["claimed_at"] is None

        assert db.claim(1) == [], "a failed row must never be picked up again on its own"

    def test_only_a_human_can_put_a_failed_row_back(self, db):
        """What the Retry button writes. Nothing in this module does it."""
        intake_id = queue_one()
        db.claim(1)
        db.record_failure(intake_id, "no")
        assert db.claim(1) == []

        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute("UPDATE vehicle_intake SET processing_status = 'QUEUED', "
                         "processing_error = NULL, processed_at = NULL WHERE id = %s",
                         (intake_id,))

        assert [j["id"] for j in db.claim(1)] == [intake_id]
        assert read(intake_id)["attempts"] == 1, "the history of asking is kept, not reset"

    def test_a_worker_killed_mid_job_is_not_a_failed_attempt(self, db):
        """The one case that DOES come back on its own, and must.

        A worker killed by a deploy never reached a verdict, so its row is unfinished work
        rather than a failure: the staleness reclaim returns it to QUEUED with `attempts`
        untouched and no error recorded. Counting it as an attempt would mean every deploy
        permanently failed whatever happened to be in flight.
        """
        intake_id = queue_one()
        db.claim(1)
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute("UPDATE vehicle_intake SET claimed_at = now() - make_interval(mins => %s)"
                         " WHERE id = %s", (db.STALE_CLAIM_MINUTES + 1, intake_id))

        assert [j["id"] for j in db.claim(1)] == [intake_id]
        row = read(intake_id)
        assert row["attempts"] == 0, "an abandoned claim is not a failed attempt"
        assert row["processing_error"] is None

    def test_a_failed_row_is_not_claimed_again(self, db):
        intake_id = queue_one()
        db.claim(1)
        db.record_failure(intake_id, "unreadable")
        assert read(intake_id)["processing_status"] == "FAILED"
        assert db.claim(1) == []

    def test_a_long_error_is_truncated_rather_than_raising(self, db):
        intake_id = queue_one()
        db.claim(1)
        db.record_failure(intake_id, "!" * 900)
        assert len(read(intake_id)["processing_error"]) == 500


class TestBoundary:
    """The worker owns the machine half of the row and nothing else.

    This is the invariant that makes a two-writer table survivable, and it is worth an explicit
    test rather than a comment: nothing in ocr/db.py may write review_status, vehicle_id or any
    column a CSR owns.
    """

    def test_the_worker_never_touches_the_human_columns(self, db):
        intake_id = queue_one(reported_by="a field executive", reported_plate="PB10ES8185")
        before = read(intake_id)

        db.claim(1)
        db.record_success(intake_id, {"plate": "WC32KN7996", "phones": ["9878770969"],
                                      "company": None, "plate_confidence": "LOW", "raw": {}})
        after = read(intake_id)

        for column in ("review_status", "reviewed_by", "reviewed_at", "review_note",
                       "vehicle_id", "reported_plate", "reported_by", "image_keys",
                       "captured_at", "created_at"):
            assert after[column] == before[column], f"the worker wrote {column}"
        assert after["review_status"] == "PENDING"

    def test_a_row_a_csr_has_finished_is_never_re_read(self, db):
        """A completed row is not queue work, whatever its processing_status says.

        The two dimensions are independent, so this cannot be enforced by the claim's status
        filter alone -- it holds because nothing ever puts a reviewed row back to QUEUED.
        """
        intake_id = queue_one()
        # Reaching COMPLETED needs a real vehicle, which is the Java side's job (and its own
        # IntakeIT covers it). The part this test can assert alone is that a row the machine has
        # finished with and a human has decided on is not claimable.
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute("UPDATE vehicle_intake SET processing_status = 'DONE', "
                         "review_status = 'DISCARDED' WHERE id = %s", (intake_id,))
        assert db.claim(1) == []


def test_check_reports_the_queue_depth(db):
    queue_one()
    queue_one()
    summary = db.check()
    assert "2 intake row(s)" in summary
    assert "2 queued" in summary


def test_check_fails_loudly_when_the_table_is_missing(monkeypatch):
    """A worker that only discovers this at the first job looks exactly like a quiet queue."""
    from ocr import db as module
    module.close()
    monkeypatch.setenv("VM_DB_URL", TEST_URL)
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute("ALTER TABLE vehicle_intake RENAME TO vehicle_intake_hidden")
    try:
        with pytest.raises(SystemExit) as caught:
            module.check()
        assert "vehicle_intake does not exist" in str(caught.value)
    finally:
        module.close()
        with psycopg.connect(TEST_URL, autocommit=True) as conn:
            conn.execute("ALTER TABLE vehicle_intake_hidden RENAME TO vehicle_intake")
