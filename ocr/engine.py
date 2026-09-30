"""The PaddleOCR engine, and the reasons for every argument it is built with.

**Read this before changing the constructor.** Each setting below was chosen from a measurement
on this machine, and three of them are not obvious.
"""
import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

TextResult = Tuple[str, float]

# Cap on the long side before OCR. Resolution is the single biggest accuracy lever here, and
# this number was raised from 1280 after measuring 8 real field photos (4032px wide side-shots
# of trucks taken from across the road):
#
#     side   mean time   plates found   phones found   raw text reads
#     1280      4.46s        2 / 8          4 / 8            57
#     2048     10.57s        4 / 8          4 / 8           117
#     3200     16.05s        4 / 8          5 / 8           148
#
# 1280 came from FreightDesk, where it caps a crop of the *vehicle bounding box*, not the whole
# frame. Applied to an uncropped wide shot it shrinks the truck to a third of the frame and the
# painted text with it -- exactly the garbling their own note warns about. 2048 doubles the
# plate hit rate; 3200 costs another 50% for almost nothing.
#
# 10s per photo is fine here because this is a BATCH worker: nobody is waiting on a form. If
# this ever moves into a request path, crop to the vehicle first (see README) rather than
# lowering this.
MAX_SIDE = 2048


class PaddleEngine:
    """PaddleOCR behind a one-method interface, so the engine stays replaceable.

    FreightDesk keeps the same seam (an `OCREngine` ABC with EasyOCR and PaddleOCR behind it),
    and that seam is why swapping the engine here cost nothing downstream: everything in
    `fields.py` consumes `(text, confidence)` tuples and neither knows nor cares which library
    produced them.
    """

    def __init__(self, models_dir: str = None):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang="en",

            # PP-OCRv5 **mobile**, measured on this machine against the v6_medium default:
            #   v6_medium  21.4 - 23.4 s per 1280x720 image
            #   v5_mobile   2.90 s, identical reads
            # A 7x difference for no loss on the test image. Re-measure if you change these.
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",

            # oneDNN is BROKEN in paddlepaddle 3.3.1 on this platform. With it enabled every
            # model version raises:
            #   NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
            #   not support [pir::ArrayAttribute<pir::DoubleAttribute>]
            # ...at inference time, not construction, so it looks like a data problem rather
            # than a build problem. Disabling it is the whole reason this works. Revisit when
            # paddlepaddle is upgraded -- and re-measure, because it is also the CPU
            # acceleration we are giving up.
            enable_mkldnn=False,

            # These three pipelines are for scanned paperwork: page rotation, dewarping and
            # per-line orientation. A photo of a truck needs none of them and they are not free.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        log.info("PaddleOCR ready (PP-OCRv5_mobile, oneDNN off)")

    def read(self, image: np.ndarray) -> List[TextResult]:
        """Every text run PaddleOCR finds, as (text, confidence)."""
        prepared = _downscale(image)
        results = self._ocr.predict(prepared)
        out: List[TextResult] = []
        for page in results:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            for text, score in zip(texts, scores):
                text = str(text).strip()
                if text:
                    out.append((text, float(score)))
        return out


def _downscale(image: np.ndarray) -> np.ndarray:
    """Shrink to MAX_SIDE, never enlarge.

    INTER_AREA because it is the right filter for shrinking; the cubic upscale FreightDesk uses
    is for tiny plate crops, which this pipeline does not produce.
    """
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_SIDE:
        return image
    scale = MAX_SIDE / long_side
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def decode(data: bytes) -> np.ndarray:
    """Bytes to a BGR array, or None if it is not a decodable image.

    "Does OpenCV decode it" is the only validity test worth applying -- the file extension and
    the declared content type are both things a client asserts, not things that are true.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
