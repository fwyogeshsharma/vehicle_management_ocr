"""Prove the PaddleOCR stack installs, loads and reads text — before any porting happens.

This is deliberately the first thing written in this repo. FreightDesk's PaddleOCR support is
dead code written against a 2.x API that no longer exists, so the result shape has to be
discovered rather than assumed. Run this, read the output, then write the engine against what
it actually printed.

NOTE: the image here is SYNTHETIC. It proves the stack works; it proves nothing about accuracy
on a photo of a real truck in the rain. That needs real photos — see the plan's Risk 4.
"""
import time

import cv2
import numpy as np


def synthetic_truck_rear(width=1280, height=720):
    """A crude stand-in for the back of a truck: a plate, a phone number, a company name."""
    img = np.full((height, width, 3), 70, dtype=np.uint8)

    # White plate panel with black text, roughly where a plate sits.
    cv2.rectangle(img, (430, 470), (860, 580), (255, 255, 255), -1)
    cv2.rectangle(img, (430, 470), (860, 580), (0, 0, 0), 4)
    cv2.putText(img, "MH12AB1234", (450, 550), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (0, 0, 0), 5, cv2.LINE_AA)

    # Painted body text, the thing FreightDesk actually cares about.
    cv2.putText(img, "KUMAR ROADWAYS", (300, 180), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (240, 240, 240), 5, cv2.LINE_AA)
    cv2.putText(img, "PH 9811008120", (330, 300), cv2.FONT_HERSHEY_SIMPLEX,
                1.8, (240, 240, 240), 5, cv2.LINE_AA)
    return img


def main():
    from paddleocr import PaddleOCR

    img = synthetic_truck_rear()
    cv2.imwrite("samples/synthetic_truck.jpg", img)
    print("wrote samples/synthetic_truck.jpg")

    print("\nloading PaddleOCR (first run downloads models)...")
    t0 = time.time()
    # The 3.x constructor. The three doc-* pipelines are for scanned paperwork and cost real
    # time on a photo of a truck, so they are off.
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print(f"loaded in {time.time() - t0:.1f}s")

    print("\npredicting...")
    t0 = time.time()
    results = ocr.predict(img)
    elapsed = time.time() - t0
    print(f"predicted in {elapsed:.2f}s")

    print(f"\nresult type: {type(results)}  len: {len(results)}")
    for i, res in enumerate(results):
        print(f"\n--- result[{i}] type={type(res)} ---")
        keys = list(res.keys()) if hasattr(res, "keys") else "(no keys())"
        print(f"keys: {keys}")
        texts = res.get("rec_texts") if hasattr(res, "get") else None
        scores = res.get("rec_scores") if hasattr(res, "get") else None
        if texts is not None:
            print("\nREADS:")
            for t, s in zip(texts, scores or []):
                print(f"  {s:.3f}  {t!r}")

    print(f"\nper-image latency: {elapsed:.2f}s (synthetic 1280x720, CPU)")


if __name__ == "__main__":
    main()
