"""Measure the pipeline against real photographs.

Everything else in this repo was measured on a synthetic image drawn with `cv2.putText` — the
exact trap FreightDesk fell into, where OCR "worked" for months against one 160x120 placeholder.
This script exists to replace that with a number taken from real phone photos of real trucks.

    python benchmark.py "D:\\images\\For Purushottam\\For Purushottam" --sample 20

Read-only: it never writes to, moves or modifies the source images.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import cv2

from ocr.engine import MAX_SIDE, PaddleEngine
from ocr.pipeline import read_truck

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--sample", type=int, default=20,
                        help="how many photos to read (evenly spread, not the first N)")
    parser.add_argument("--seed", type=int, default=7, help="so a run is reproducible")
    parser.add_argument("--json", help="write the per-photo detail here")
    args = parser.parse_args()

    folder = Path(args.folder)
    photos = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not photos:
        print(f"no images found in {folder}")
        return 1

    # Spread the sample across the whole folder rather than taking the first N: these are named
    # by timestamp, so the first N is one morning in one place and tells you nothing general.
    if args.sample < len(photos):
        random.seed(args.seed)
        step = len(photos) / args.sample
        chosen = [photos[int(i * step)] for i in range(args.sample)]
    else:
        chosen = photos

    print(f"{len(photos)} photos in the folder; reading {len(chosen)}")
    print("loading models...")
    t0 = time.time()
    engine = PaddleEngine()
    print(f"models ready in {time.time() - t0:.1f}s (paid once per worker, not per photo)\n")

    print(f"{'photo':<26} {'megapixels':>10} {'seconds':>8}  plate / phone / company")
    print("-" * 100)

    rows = []
    for path in chosen:
        data = path.read_bytes()
        image = cv2.imread(str(path))
        mp = 0.0 if image is None else (image.shape[0] * image.shape[1]) / 1e6

        started = time.time()
        try:
            fields = read_truck(engine, [data])
            elapsed = time.time() - started
            error = None
        except Exception as e:
            elapsed = time.time() - started
            fields = {"plate": None, "phone": None, "company": None,
                      "plate_confidence": "NONE", "raw": {"body_texts": []}}
            error = str(e)[:60]

        summary = " / ".join(str(fields[k] or "-") for k in ("plate", "phone", "company"))
        print(f"{path.name:<26} {mp:>10.1f} {elapsed:>8.2f}  {summary}"
              + (f"   ERROR {error}" if error else ""))

        rows.append({
            "file": path.name,
            "megapixels": round(mp, 1),
            "bytes": len(data),
            "seconds": round(elapsed, 2),
            "plate": fields["plate"],
            "phone": fields["phone"],
            "company": fields["company"],
            "confidence": fields["plate_confidence"],
            "reads": len(fields["raw"].get("body_texts", [])),
            "error": error,
        })

    report(rows, engine)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nper-photo detail written to {args.json}")
    return 0


def report(rows, engine) -> None:
    times = [r["seconds"] for r in rows]
    n = len(rows)
    got = lambda key: sum(1 for r in rows if r[key])

    print("\n" + "=" * 100)
    print(f"TIMING over {n} real photos, downscaled to {MAX_SIDE}px, CPU, oneDNN off")
    print(f"  mean   {statistics.mean(times):6.2f}s"
          f"   median {statistics.median(times):6.2f}s"
          f"   min {min(times):5.2f}s   max {max(times):5.2f}s")
    if n > 2:
        print(f"  p90    {sorted(times)[int(n * 0.9) - 1]:6.2f}s"
              f"   total  {sum(times):6.1f}s")
    mp = statistics.mean(r["megapixels"] for r in rows)
    print(f"  source photos average {mp:.1f} megapixels "
          f"({statistics.mean(r['bytes'] for r in rows) / 1e6:.1f} MB)")

    print(f"\nWHAT IT READ  (of {n} photos)")
    print(f"  a plate         {got('plate'):>3}   {got('plate') / n:>5.0%}")
    print(f"  a phone number  {got('phone'):>3}   {got('phone') / n:>5.0%}")
    print(f"  a company name  {got('company'):>3}   {got('company') / n:>5.0%}")
    print(f"  any text at all {sum(1 for r in rows if r['reads']):>3}"
          f"   {sum(1 for r in rows if r['reads']) / n:>5.0%}")
    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"  FAILED          {len(errors):>3}")

    print("\nNote: 'read a plate' means something plate-shaped came back, NOT that it is the "
          "right plate.\nChecking correctness needs a human with the photos open.")


if __name__ == "__main__":
    raise SystemExit(main())
