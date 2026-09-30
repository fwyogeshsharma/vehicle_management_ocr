"""The settings module: precedence, parsing, and the two ways to name a database.

No database and no network -- ``ocr.config`` only ever reads ``.env`` and the environment, so
every case here is a reload with a different environment.
"""
import importlib
import os

import pytest

import ocr.config


# Every variable the module looks at. Cleared before each reload so a value in the developer's
# own shell -- or in the .env file that certainly exists on the machine running this -- cannot
# quietly decide the outcome of a test.
SETTINGS = [
    "VM_ENV_FILE", "VM_DB_URL",
    "VM_DATABASE_HOST", "VM_DATABASE_PORT", "VM_DATABASE_NAME",
    "VM_DATABASE_USER", "VM_DATABASE_PASSWORD", "VM_TEST_DATABASE_URL",
    "VM_STALE_MINUTES", "VM_POLL_SECONDS", "VM_BATCH",
    "VM_IMAGE_BACKEND", "VM_IMAGE_DIR", "VM_GCS_BUCKET", "VM_GCS_PREFIX",
]


@pytest.fixture
def load(monkeypatch, tmp_path):
    """Reload ocr.config with a controlled environment and an optional .env file."""
    def _load(env_file: str = "", **environ):
        # Cleared through monkeypatch, which matters: load_dotenv writes straight into
        # os.environ, so a value from one test's .env outlives the reload that read it.
        for name in SETTINGS:
            monkeypatch.delenv(name, raising=False)
        # Point at an empty file by default, so the repo's own .env never reaches a test.
        # env_file=None asks for a path that does not exist at all.
        path = tmp_path / ("nonexistent.env" if env_file is None else ".env")
        if env_file is not None:
            path.write_text(env_file, encoding="utf-8")
        monkeypatch.setenv("VM_ENV_FILE", str(path))
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(ocr.config)
    yield _load
    # Leave the module as the rest of the suite expects to find it.
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(ocr.config)


class TestPrecedence:
    """Real environment beats .env beats default -- the order application.yml uses."""

    def test_the_default_applies_when_nothing_says_otherwise(self, load):
        assert load().DB_NAME == "vehicle_management"

    def test_the_env_file_beats_the_default(self, load):
        assert load("VM_DATABASE_NAME=from_the_file").DB_NAME == "from_the_file"

    def test_a_real_variable_beats_the_env_file(self, load):
        # The one that matters in production: a container sets real variables and a stray .env
        # on the image must not win.
        cfg = load("VM_DATABASE_NAME=from_the_file", VM_DATABASE_NAME="from_the_environment")
        assert cfg.DB_NAME == "from_the_environment"

    def test_a_missing_env_file_is_not_an_error(self, load):
        cfg = load(env_file=None)

        assert cfg.ENV_FILE_LOADED is False
        assert cfg.DB_NAME == "vehicle_management"

    def test_an_empty_value_means_unset_not_empty_string(self, load):
        # A cleared line in a .env file is someone removing a value, not setting it to "".
        assert load("VM_DATABASE_NAME=   ").DB_NAME == "vehicle_management"


class TestTheDatabase:

    def test_the_parts_are_assembled_into_a_connection_string(self, load):
        cfg = load(VM_DATABASE_HOST="db.internal", VM_DATABASE_PORT="6543",
                   VM_DATABASE_NAME="vm", VM_DATABASE_USER="vmuser",
                   VM_DATABASE_PASSWORD="secret")
        dsn = cfg.dsn()

        assert "host=db.internal" in dsn
        assert "port=6543" in dsn
        assert "dbname=vm" in dsn
        assert "user=vmuser" in dsn

    def test_a_password_with_an_at_sign_needs_no_encoding(self, load):
        """The trap the split form exists to remove.

        Written into a URL, `faber@123` has to become `faber%40123` or libpq reads the @ as the
        start of the host. Passed as a part, it is quoted for us and written as it is.
        """
        cfg = load(VM_DATABASE_PASSWORD="faber@123")

        assert "%40" not in cfg.dsn()
        assert "faber@123" in cfg.dsn()

    @pytest.mark.parametrize("password", ["has space", "has'quote", "has\\backslash", "a@b:c/d"])
    def test_awkward_passwords_survive_the_round_trip(self, load, password):
        from psycopg.conninfo import conninfo_to_dict

        cfg = load(VM_DATABASE_PASSWORD=password)

        assert conninfo_to_dict(cfg.dsn())["password"] == password

    def test_a_full_url_wins_outright(self, load):
        cfg = load(VM_DB_URL="postgresql://u:p@host/db", VM_DATABASE_HOST="ignored")

        assert cfg.dsn() == "postgresql://u:p@host/db"

    def test_no_credentials_is_an_error_with_instructions(self, load):
        cfg = load()

        with pytest.raises(SystemExit) as raised:
            cfg.dsn()

        message = str(raised.value)
        assert "VM_DATABASE_PASSWORD" in message
        assert "VM_DB_URL" in message
        # The path to the file they are meant to edit, not just the variable names.
        assert ".env" in message

    def test_the_test_database_is_a_separate_setting(self, load):
        """One variable for both would eventually truncate a real vehicle_intake."""
        cfg = load(VM_DATABASE_PASSWORD="live", VM_TEST_DATABASE_URL="postgresql://x/scratch")

        assert cfg.TEST_DB_URL == "postgresql://x/scratch"
        assert "scratch" not in cfg.dsn()


class TestNothingLeaksThePassword:

    def test_describe_db_names_the_target_but_not_the_secret(self, load):
        cfg = load(VM_DATABASE_HOST="db.internal", VM_DATABASE_USER="vmuser",
                   VM_DATABASE_PASSWORD="hunter2")

        described = cfg.describe_db()

        assert "hunter2" not in described
        assert "db.internal" in described and "vmuser" in described

    def test_a_url_is_never_echoed_because_its_password_is_inline(self, load):
        cfg = load(VM_DB_URL="postgresql://u:hunter2@host/db")

        assert "hunter2" not in cfg.describe_db()

    def test_the_startup_summary_is_safe_to_log(self, load):
        cfg = load(VM_DATABASE_PASSWORD="hunter2", VM_IMAGE_BACKEND="gcs",
                   VM_GCS_BUCKET="vm-photos")

        summary = cfg.describe()

        assert "hunter2" not in summary
        assert "vm-photos" in summary


class TestParsing:

    def test_numbers_are_converted(self, load):
        cfg = load(VM_POLL_SECONDS="2.5", VM_BATCH="4", VM_STALE_MINUTES="30")

        assert (cfg.POLL_SECONDS, cfg.BATCH, cfg.STALE_CLAIM_MINUTES) == (2.5, 4, 30)

    @pytest.mark.parametrize("name,value", [("VM_BATCH", "two"), ("VM_STALE_MINUTES", "15m"),
                                            ("VM_POLL_SECONDS", "soon")])
    def test_a_number_that_is_not_one_fails_at_startup(self, load, name, value):
        # Not silently defaulted: VM_BATCH=1O (letter O) would otherwise run at a batch of 1
        # forever and look like the setting simply did nothing.
        with pytest.raises(SystemExit, match=name):
            load(**{name: value})

    def test_the_backend_is_lowercased_and_the_prefix_unslashed(self, load):
        cfg = load(VM_IMAGE_BACKEND="GCS", VM_GCS_PREFIX="/photos/")

        assert cfg.IMAGE_BACKEND == "gcs"
        assert cfg.GCS_PREFIX == "photos"


def test_the_committed_example_names_every_setting(load):
    """.env.example is documentation that rots silently. This is what stops it.

    A setting added to config.py and not to the example is one a deployer cannot discover
    without reading the source.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(ocr.config.__file__)))
    with open(os.path.join(root, ".env.example"), encoding="utf-8") as f:
        example = f.read()

    missing = [name for name in SETTINGS
               if name not in ("VM_ENV_FILE",) and name not in example]

    assert not missing, f"not mentioned in .env.example: {missing}"
