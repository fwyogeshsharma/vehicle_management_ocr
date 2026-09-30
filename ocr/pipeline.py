"""Photos of one truck in, three fields out.

The whole OCR path, deliberately small. Several photos of the same truck collapse into one
result: they are independent shots, not consecutive frames, so every read from every photo goes
into one pool and the extractors vote across the lot. FreightDesk does the same thing in
`pipeline/image_api.py` and for the same reason.

**No vehicle or plate detection here.** FreightDesk crops to a YOLO-detected vehicle box first,
and has a fine-tuned plate detector for a dedicated high-resolution plate crop. Both are worth
having eventually, and both cost a PyTorch install (~1 GB) that this service currently does not
need — PaddleOCR reads the whole frame perfectly well, and FreightDesk's own code notes that on
a close-up field photo YOLO frequently fails to see "a vehicle" at all and falls back to the
full frame anyway. Adding the plate detector later is a new module and a config flag, not a
rewrite: everything downstream consumes (text, confidence) tuples.
"""
import logging
import time
from typing import List, Tuple

from .engine import decode
from .fields import extract_fields

log = logging.getLogger(__name__)

MAX_IMAGES = 5


def read_truck(engine, images: List[bytes]) -> dict:
    """OCR every photo of one truck and reduce them to one set of fields.

    An undecodable photo is skipped rather than failing the batch — one corrupt upload out of
    four should not cost the reads from the other three. All of them undecodable raises, because
    then there is genuinely nothing to report.
    """
    reads: List[Tuple[str, float]] = []
    decoded = 0
    started = time.time()

    for index, data in enumerate(images[:MAX_IMAGES]):
        image = decode(data)
        if image is None:
            log.warning("photo %d could not be decoded; skipping", index)
            continue
        decoded += 1
        reads.extend(engine.read(image))

    if decoded == 0:
        raise ValueError("none of the uploaded photos could be decoded as images")

    fields = extract_fields(reads)
    fields["raw"]["images_read"] = decoded
    fields["raw"]["seconds"] = round(time.time() - started, 2)
    log.info("read %d photo(s) in %.2fs: plate=%s phone=%s company=%s",
             decoded, time.time() - started, fields["plate"], fields["phone"],
             fields["company"])
    return fields
