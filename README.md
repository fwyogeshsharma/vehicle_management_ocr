# Vehicle Management — OCR worker

Reads number plates, phone numbers and company names off photos of trucks, for
[`vehicleManagement`](../vehicleManagement).

Python 3.12 · PaddleOCR · CPU only.

---

## What it is

A **polling worker**, not a web service. It takes rows out of `vehicle_intake`, OCRs the photos,
writes the results back, and sleeps. The database row *is* the queue, so nothing is lost if this
process dies mid-job: the claim goes stale, the next poll re-queues it, and the work is picked
up again.

```
                      vehicleManagement (Java)            vehicle_intake
field app ──POST /api/intake/photos──►  photos to the store,  ──►  QUEUED
                                        then INSERT the row          │
                                                                     │
  this worker ──claim, FOR UPDATE SKIP LOCKED──────────────────►  PROCESSING
              ──ocr_plate / ocr_mobile / ocr_company─────────────►  DONE | FAILED
                                                                     │
     CSR ──the worklist in vehicleManagementUI───────────────────►  a real vehicle
```

It reads and writes PostgreSQL directly, and reads the photos straight from the object store.
There is no API in between.

### The cost of that, stated plainly

**`vehicle_intake` has two writers in two languages.** This worker owns the machine half of the
row — `processing_status`, `processing_error`, `processed_at`, `claimed_at`, `attempts` and the
`ocr_*` columns. The Java service owns everything else. Nothing enforces that boundary except
the code in `ocr/db.py` and a test that asserts it (`TestBoundary`), so:

- a column added to the Java entity must be added here too, and `ddl-auto: validate` will not
  catch the reverse;
- the schema is defined in **one** place, `vehicleManagement/.../005-vehicle-intake.sql`, and
  this repository follows it;
- this worker now holds database credentials and object-store credentials. It did not before.

What it buys is a worker that scales, restarts and deploys with no coupling to the API's request
path, and photos that are not base64-encoded through an HTTP response on their way to being
decoded again.

## Running it

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

set VM_DB_URL=postgresql://postgres:pw@127.0.0.1:5432/vehicle_management
.venv\Scripts\python worker.py
```

| Variable | Default | |
|---|---|---|
| `VM_DB_URL` | *(required)* | the database holding `vehicle_intake`. No default on purpose — a worker that quietly pointed at the wrong database would find a table of the right shape |
| `VM_IMAGE_BACKEND` | `local` | `local` or `gcs`. **Must match what the API is configured with** |
| `VM_IMAGE_DIR` | `./uploads` | where the `local` backend reads |
| `VM_GCS_BUCKET` | | required when the backend is `gcs` |
| `VM_GCS_PREFIX` | | optional key prefix inside the bucket |
| `VM_POLL_SECONDS` | `5` | how often to look for work when idle |
| `VM_BATCH` | `1` | rows per claim |
| `VM_STALE_MINUTES` | `15` | when a held claim is assumed abandoned |

The worker proves the database and the object store **before** loading the models, because both
are one-line misconfigurations and finding out after a 30-second model load is a slow way to
learn you typed the bucket name wrong.

### There is no automatic retry

**One attempt per photo.** If it fails — the file would not decode, it was not in the store, the
engine raised — the row goes to `FAILED` and no poll will take it again. There is no attempt
limit to configure because there is nothing to limit.

This costs something and it is worth being clear about it: an object store unreachable for one
second permanently fails a photo that would have read perfectly a moment later. It is still the
right trade. A photo the engine cannot read does not become readable on a second pass, and on
real field photos that is about one in four; automatic retry only helps if the attempts are
spaced out, spacing means the row sits in `QUEUED` for minutes looking like ordinary work, and
the end state is the same `FAILED` either way. A CSR with the picture open settles it in seconds,
and the **Read again** button is there for the case where the cause has actually been fixed.

Two things are deliberately *not* treated as failures:

- **A worker killed mid-job.** It never reached a verdict, so the staleness reclaim returns the
  row to `QUEUED` with `attempts` untouched and no error recorded. Counting it would mean every
  deploy permanently failed whatever was in flight.
- **Reading nothing.** OCR that runs cleanly and finds no plate, phone or company is `DONE`, not
  `FAILED`, and reaches the CSR worklist like any other row. The machine finished; it simply had
  nothing to offer, and the human can still see the photo.

```bat
.venv\Scripts\python -m pytest tests/ -q     :: 31 tests, no database or models needed

:: The queue tests need a SCRATCH PostgreSQL -- they TRUNCATE vehicle_intake.
:: Never point this at the database the application is using.
set VM_TEST_DATABASE_URL=postgresql://postgres:pw@127.0.0.1:5432/vm_ocr_test
.venv\Scripts\python -m pytest tests/test_db.py -v

.venv\Scripts\python probe_paddle.py          :: prove the OCR stack, print a latency number
```

---

## Why these exact versions

**Do not change `requirements.txt` without re-running `probe_paddle.py`.** Two of the pins are
there because of measurements, not taste.

### oneDNN is broken in paddlepaddle 3.3.1, so it is switched off

With `enable_mkldnn` left at its default, **every** PaddleOCR model version fails at inference —
not at construction — with:

```
NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
not support [pir::ArrayAttribute<pir::DoubleAttribute>]
    (at paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc:118)
```

It reads like a data problem and is a build problem. `enable_mkldnn=False` in `ocr/engine.py` is
what makes this work at all. Revisit on the next paddlepaddle upgrade — and re-measure, because
oneDNN is also the CPU acceleration being given up.

### PP-OCRv5 **mobile**, not the default

Measured here on a 1280×720 image, oneDNN off:

| models | per image | reads |
|---|---|---|
| `PP-OCRv6_medium` (the default) | **21.4 – 23.4 s** | correct |
| `PP-OCRv5` server | 43.1 s | correct, but split into single words |
| **`PP-OCRv5_mobile`** | **2.90 s** | correct |

A 7× difference for no loss. End to end through the worker, a single 56 KB photo takes **~3.0 s**
including decode.

### Resolution is the accuracy lever

`MAX_SIDE = 2048` in `ocr/engine.py`. Measured on 8 real field photos (4032px-wide shots of
trucks taken from across the road):

| long side | mean time | plates | phones | raw text reads |
|---|---|---|---|---|
| 1280 | 4.46 s | 2 / 8 | 4 / 8 | 57 |
| **2048** | **10.57 s** | **4 / 8** | 4 / 8 | **117** |
| 3200 | 16.05 s | 4 / 8 | 5 / 8 | 148 |

1280 is FreightDesk's number, and it is right *for FreightDesk* — there it caps a crop of the
**vehicle bounding box**, not the whole frame. Applied to an uncropped wide shot it shrinks the
truck to a third of the frame and the painted text with it, which is exactly the garbling their
own note warns about. 2048 doubles the plate hit rate. 3200 costs another 50% for almost nothing.

10 s a photo is affordable because this is a **batch worker** — nobody is waiting on a form. If
this ever moves into a request path, crop to the vehicle first rather than lowering this.

---

## The real-photo baseline

**952 field photos, 20 sampled evenly across the folder.** 7.5 megapixels and 3.0 MB each,
CPU only, oneDNN off, models loaded once (5.3 s at startup, not per photo).

| | |
|---|---|
| mean | **13.9 s** |
| median | **10.2 s** |
| min / max | 6.3 s / 48.3 s |
| p90 | 22.5 s |

The mean is dragged up by two photos at 40 s and 48 s; with those excluded it is 10.5 s. An
earlier run of the same 20 on a quieter machine gave mean 12.2 s, median 11.8 s, max 21.3 s —
so **treat ~10–14 s per photo as the figure, and expect outliers when the machine is busy.**
At 10 s a photo one worker clears ~350 photos an hour; the 952 in this folder take about
3 hours. Scale by adding workers — the claim is already safe for that.

### What it read

| | of 20 | |
|---|---|---|
| any text at all | 20 | 100% |
| a phone number | 13 | 65% |
| a company name | 7 | 35% |
| a plate | 5 | 25% |

**Plate accuracy is the weak number, and it is worth knowing exactly why.** All five reads came
back `LOW` confidence, and checking them against the photos:

- `PB10ES8185` — correct.
- `WC32KN7996` — the truck is **UP32 KN 7996**. Digits right, state code wrong, on an ornate
  hand-painted plate. This is the dangerous failure: a wrong plate that looks perfectly valid.
- `UP32K`, `UP86T`, `RJ11GC` — all the **first line of a two-line painted plate**. For
  `UP32K` the second line `N0936` is sitting right there in the raw reads; the extractor simply
  never joins them.

So the ceiling is not the OCR — it reads the characters. Two things stand between 25% and
something much higher, and both are named under *Known limits* below.

Reproduce with:

```bat
.venv\Scripts\python benchmark.py "D:\images\For Purushottam\For Purushottam" --sample 20
```

It is read-only; it never writes to, moves or modifies the source images.

---

## What was ported, and what was not

`ocr/fields.py` is lifted from FreightDesk's `pipeline/ocr_engine.py` and `pipeline/extract.py`.
**It is the valuable part.** Every regex, threshold and blacklist entry there was paid for by
someone watching real OCR output go wrong:

- `TRANSPORT` is not a number plate, even though `TR` is the state code for Tripura — and
  neither is `TRANSPOR`, which is why the blacklist matches as a substring both ways.
- `ASHOK` (a Leyland badge) starts with `AS` for Assam, so a plate must contain a digit.
- `981i008120` is a phone number: a digit-confusion map repairs letters inside digit-dominated
  strings, and is deliberately **not** applied to ordinary text, where it would turn every `I`
  and `O` in a company name into a `1` and a `0`.
- A phone number needs one read at ≥ 0.5 confidence before it is offered, because somebody is
  going to ring it.

Copied rather than shared as a library: FreightDesk is a separate product on its own trajectory,
and a shared package would couple their release cycles for no benefit.

**Deliberately not ported:**

- **YOLO vehicle and plate detection.** FreightDesk crops to a detected vehicle box and has a
  fine-tuned plate detector (`models/best.pt`) for a dedicated high-resolution plate crop. Both
  cost a PyTorch install (~1 GB) that this worker does not currently need — PaddleOCR reads the
  whole frame perfectly well, and FreightDesk's own code notes that on a close-up field photo
  YOLO frequently fails to see "a vehicle" and falls back to the full frame anyway. Adding the
  plate detector later is a new module and a config flag: everything downstream consumes
  `(text, confidence)` tuples.
- **Website, city, vehicle-type and other-text extraction.** The intake form has three fields;
  inventing more would be inventing work for a CSR.
- **The "longest clean phrase" fallback for company names.** It guesses, and a wrong company name
  silently attaches a truck to the wrong fleet. Here a human is on the phone and can simply ask.

---

## What it fixes relative to FreightDesk

FreightDesk's worker is honest about being a prototype; its own docstrings name most of these.

| FreightDesk | Here |
|---|---|
| In-process `queue.Queue` — a pending row is invisible until the app restarts | The database is the queue; this polls it |
| No claim: `UPDATE ... SET 'PROCESSING'` with no condition on the current status | `FOR UPDATE SKIP LOCKED` on the server; two workers get disjoint batches |
| A crashed worker strands rows until the next restart | `claimed_at` + a staleness cutoff re-queues them, and that is **not** counted as a failed attempt |
| `FAILED` is terminal and cannot be re-run even by a restart | Also terminal for the machine, but a **Read again** button puts it back in the queue when a human asks |
| No timeout anywhere | `VM_HTTP_TIMEOUT` on every call |
| Models downloaded silently at first use | Loaded at startup, so a broken install fails loudly and immediately |
| Every digit on a line is concatenated, so `[6-9]\d{9}` matches a phone number straddling two real ones — survivable only because they vote across video frames | `_digit_runs` splits the line first. A single field photo gets no vote, and a plausible wrong number is a stranger who gets the call |

---

## Known limits

- **Two-line painted plates are truncated.** Indian truck plates are commonly painted in two
  lines, and only the first is kept — `UP32K` where the truck is `UP32KN0936`. The second line
  is already in the raw reads. Fixing it properly needs the **bounding boxes** PaddleOCR returns
  and `ocr/engine.py` currently discards: join two fragments only when one sits directly below
  the other and the pair forms a valid plate. Joining them by guesswork instead would invent
  plates, which is the same class of bug as the phone number described in `_digit_runs`. This is
  the single highest-value change left, and it is most of the gap between 25% and a useful rate.
- **A wrong state code reads as a valid plate.** `UP32KN7996` came back as `WC32KN7996`. `WC` is
  not an Indian state code at all, so the cheap half of this is rejecting — or at least
  down-ranking — a plate whose first two letters are not on the list; `ocr/fields.py` already
  has the list, and only uses it to *recognise* plates, never to reject them.
- **A plate is never read from the plate itself.** Every read above came off text painted on the
  body. FreightDesk's fine-tuned plate detector (`models/best.pt`) exists for exactly this and
  was deliberately not ported; if the rate needs to go higher than the two fixes above get it,
  that is the next thing to bring across.
- **One worker at a time is assumed but not required.** The claim is safe for several; nothing
  has been run that way.
- **No image pre-processing.** No deskew, no contrast enhancement, no plate crop. FreightDesk
  defines those functions too and never calls them either.
- **English only** (`lang="en"`). A Devanagari plate or company name will not be read.
