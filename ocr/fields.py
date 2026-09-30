"""Turning raw OCR reads into a plate, a phone number and a company name.

Ported from FreightDesk's `pipeline/ocr_engine.py` and `pipeline/extract.py`. **This is the
valuable part of the port** — every threshold, regex and blacklist entry here was paid for by
someone watching real OCR output go wrong, and the reasons are kept in the comments. The OCR
engine itself is replaceable; this is not.

Deliberately narrower than FreightDesk's version. It drops website, city, vehicle-type and
other-text extraction, because vehicleManagement's intake form has three fields and inventing
more would be inventing work for a CSR.

Everything here is a pure function over `(text, confidence)` tuples — no models, no I/O — which
is what makes it the cheapest thing in either project to test.
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

TextResult = Tuple[str, float]

# ── plate shapes ──────────────────────────────────────────────────────────────

_PLATE_PATTERNS = [
    # Indian modern: MH12AB1234, with optional separators
    re.compile(r"^[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}$"),
    # Indian older / partial: MH 12 A 1234
    re.compile(r"^[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z][\s\-]?\d{4}$"),
]

_INDIAN_STATE_RE = re.compile(
    r"^(AP|AR|AS|BR|CG|CH|DL|DN|GA|GJ|HR|HP|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|NL|OD|PB|PY|"
    r"RJ|SK|TN|TR|TS|UK|UP|WB)",
    re.IGNORECASE,
)

# Truck words that happen to begin with an Indian state code, or garble into one.
# TRANSPORT starts "TR" = Tripura; ASHOK starts "AS" = Assam; CARRIER misreads as GARRIER,
# and "GA" is Goa. Without this list every one of them is classified as a number plate.
_PLATE_WORD_BLACKLIST = frozenset([
    "TRANSPORT", "TRANSPORTS", "TRAVELS", "TRAILER", "TRAILOR", "TANKER",
    "GOODS", "ARRIER", "CARRIER", "CARRIERS", "ROADWAYS", "ROADLINES",
    "LOGISTICS", "UPDATE", "KAPURTHALA", "CHENNAI", "HRSCHOOL", "ASSAM",
    "GANDHI", "ASHOK", "ASHOKLEYLAND", "LEYLAND", "APOLLO", "MAHINDRA",
])


def looks_like_plate(text: str) -> bool:
    """Whether a read is a number plate rather than something painted on the truck.

    Strict regex for clean reads, with a state-code fallback for garbled ones — OCR routinely
    turns MH14 into MHI4, which no plate regex will match but which is obviously still a plate.
    """
    t_clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    tu = text.upper().strip()

    # Substring both ways, so a truncated read like "TRANSPOR" is caught too.
    if len(t_clean) >= 5 and any(t_clean in w or w in t_clean for w in _PLATE_WORD_BLACKLIST):
        return False

    has_letters = bool(re.search(r"[A-Z]", t_clean))
    has_digits = bool(re.search(r"[0-9]", t_clean))

    # A state code plus at least one digit is almost certainly a plate. The digit requirement
    # is load-bearing: letters-only reads are words (ASHOK, JAIPUR) and useless as plate data.
    if _INDIAN_STATE_RE.match(tu) and has_digits and 5 <= len(t_clean) <= 12:
        return True

    if not (has_letters and has_digits and 5 <= len(t_clean) <= 11):
        return False
    return any(p.match(tu) for p in _PLATE_PATTERNS)


# ── camera overlays ───────────────────────────────────────────────────────────

# Burned-in DVR clocks and watermarks. A field executive's phone photo has none of these, but
# the patterns cost nothing and the moment anyone points this at CCTV footage they matter.
_OSD_PATTERNS = [
    re.compile(r"\d{4}[/\-]\d{2}"),
    re.compile(r"\d{2}\s*:\s*\d{2}"),
    re.compile(r"\d{2}[.]\d{2}[.]\d{2}"),
    re.compile(r"\d{2}[.]\d{2}\s*:"),
    re.compile(r"^\d{1,2}[\s:]\s*\d{2}[\s:]\s*\d{2}"),
    re.compile(r"^20\d\d"),
    re.compile(r"^I?PC$", re.IGNORECASE),
    re.compile(r"^ch\d+$", re.IGNORECASE),
    re.compile(r"^\d{1,2}[.:]\d{4,6}$"),
]


def is_osd(text: str) -> bool:
    """Whether a read is a camera overlay rather than something on the truck."""
    return any(p.search(text) for p in _OSD_PATTERNS)


# ── normalising and de-duplicating ────────────────────────────────────────────

_LEADING_NOISE = re.compile(r"^[^A-Za-z0-9]+")
_TRAILING_NOISE = re.compile(r"[^A-Za-z0-9.)]+$")
_WHITESPACE = re.compile(r"\s+")
_PHONE_RE = re.compile(r"([6-9]\d{9})")


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").strip().upper())


def clean_token(text: str) -> str:
    t = _LEADING_NOISE.sub("", text or "").strip()
    return _TRAILING_NOISE.sub("", t).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _is_noise(text: str) -> bool:
    """Too short, or a bare digit run that carries no phone number.

    A 4+ digit run is kept: on a deliberately-taken photo (no clock overlay) it is far more
    likely a partial plate or series fragment than noise. FreightDesk gates this behind a flag
    because its video frames *do* carry clocks; every photo here is a field report, so the
    permissive branch is simply the behaviour.
    """
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(clean) < 2:
        return True
    if re.match(r"^\d+$", clean) and not _PHONE_RE.search(clean):
        return len(clean) < 4
    return False


def deduplicate(texts: List[TextResult]) -> List[str]:
    """Normalise, drop noise, then collapse near-duplicates at edit distance 1.

    Several photos of one truck produce the same painted text read slightly differently each
    time; without this the company name appears three times with one letter different.
    """
    best: Dict[str, float] = {}
    for text, conf in texts:
        n = normalize_text(clean_token(text))
        if _is_noise(n):
            continue
        if n not in best or conf > best[n]:
            best[n] = conf

    unique = list(best.keys())
    drop = set()
    for i, a in enumerate(unique):
        if a in drop:
            continue
        for j, b in enumerate(unique):
            if i == j or b in drop:
                continue
            if abs(len(a) - len(b)) <= 1 and levenshtein(a, b) <= 1:
                drop.add(a if len(a) <= len(b) else b)
    return sorted(t for t in unique if t not in drop)


# ── phone numbers ─────────────────────────────────────────────────────────────

# Applied only inside digit-dominated strings, where a letter is almost certainly a misread
# digit: "981i008120" is a phone number, not a word.
_DIGIT_CONFUSION = str.maketrans("IilL|OoSsBZz", "111110058822")

# A single low-confidence misread should not surface as a number someone will ring.
_PHONE_MIN_CONF = 0.5


# Letters OCR mistakes for digits. A run of these mixed with digits is a misread number;
# any other letter is part of a word, and therefore the end of the number.
_CONFUSABLE_LETTERS = "IilLOoSsBZz"

# A maximal run of things that can appear inside a written phone number: digits, the separators
# people actually use, and the confusable letters above. Everything else -- M, N, a comma, a
# slash -- terminates the run.
_NUMBER_RUN = re.compile(r"[0-9+\-.·–— " + _CONFUSABLE_LETTERS + r"]+")


def _digit_runs(text: str) -> List[str]:
    r"""Split a line into runs that could each be one written number.

    **This is what stops a phone number being invented across a boundary.** A real read from a
    truck's side looks like::

        GURU NANAK ROADCARRIER, TRPORT NAGAR.LUDHIANA.141003, MOB.NO.98787.70969,84370-36313

    Concatenating every digit on that line gives `101114100308098787709698437036313`, in which
    `[6-9]\d{9}` happily matches `8098787709` -- a number straddling the mangled "M0B.N0." and
    the start of the real one. It looks exactly like an Indian mobile and belongs to a stranger,
    and a CSR would ring it. Splitting first yields the pincode and the two real numbers
    separately.

    Confusable letters stay *inside* a run so that `98IIO08I20` survives as one candidate;
    everything else breaks it, so `MOB.NO.` cannot fuse itself onto the number that follows.

    FreightDesk has this bug and is saved by voting across many video frames -- the spurious
    number appears once, the real one dozens of times. A single field photo gets no such vote.
    """
    return [run for run in _NUMBER_RUN.findall(text) if any(c.isdigit() for c in run)]


def extract_phones(texts: List[TextResult]) -> List[str]:
    """**Every** Indian mobile number read off the truck, most-read first.

    A truck routinely carries more than one: the owner's, the driver's, the transport office's,
    often painted in a row. Returning only the best-read one threw away numbers a telecaller
    genuinely needs — if the first does not answer, the second is right there on the same photo.

    Variants within edit distance 2 collapse to whichever was read most often: OCR gets a
    different digit wrong on different shots, and the majority read is almost always the true
    one. That collapsing is also what stops this returning six near-copies of one number.
    """
    counts: Counter = Counter()
    best_conf: Dict[str, float] = {}
    for text, conf in texts:
        # Split BEFORE repairing confusions, so the repair cannot manufacture the digits that
        # join two runs together: "MOB.NO." must stay a letter run, not become "M08.N0.".
        for run in _digit_runs(text):
            # Within a run that is mostly digits already, a stray letter is a misread digit.
            if sum(c.isdigit() for c in run) >= 6:
                run = run.translate(_DIGIT_CONFUSION)
            digits = re.sub(r"\D", "", run)
            digits = re.sub(r"^0091|^91(?=\d{10})|^0(?=\d{10})", "", digits)
            for m in _PHONE_RE.finditer(digits):
                num = m.group(1)
                counts[num] += 1
                best_conf[num] = max(best_conf.get(num, 0.0), conf)

    kept: List[str] = []
    for num, _ in counts.most_common():
        if best_conf.get(num, 0.0) < _PHONE_MIN_CONF:
            continue
        if all(levenshtein(num, k) > 2 for k in kept):
            kept.append(num)
    return kept


def extract_phone(texts: List[TextResult]) -> Optional[str]:
    """The best-read number alone, or None. A convenience over :func:`extract_phones`."""
    phones = extract_phones(texts)
    return phones[0] if phones else None


# ── company names ─────────────────────────────────────────────────────────────

_COMPANY_SUFFIX_RE = re.compile(
    r"(TRANSPORT(S)?|LOGISTICS|ROADWAYS|ROADWAY|CARRIERS?|SERVICES?|"
    r"ENTERPRISES?|INDUSTRIES|INDUSTRY|MOVERS?|PACKERS?|TRADERS?|"
    r"AGENCY|AGENCIES|TOURS?|TRAVELS?|MOTORS?|AUTOMOTIVE|"
    r"PVT\.?\s*LTD\.?|LTD\.?|LIMITED|CORPORATION|CORP\.?|"
    r"INTERNATIONAL|NATIONAL|EXPRESS|FREIGHT|CARGO|LINES?)\s*$"
)

_NOISE_STANDALONE = frozenset([
    "PH", "PHONE", "MOB", "MOBILE", "CONTACT", "TEL", "OFF", "RES", "HO",
    "EMAIL", "MAIL", "GST", "GSTIN", "NO", "CELL",
])


def extract_company(texts: List[str]) -> Optional[str]:
    """Best-effort company name: the longest read ending in a corporate suffix.

    Only the suffix rule is used, unlike FreightDesk which also falls back to "longest clean
    phrase". That fallback guesses, and a wrong company name silently attaches a truck to the
    wrong fleet — here the CSR is on the phone anyway and can simply be asked.
    """
    candidates = []
    for raw in texts:
        t = clean_token(raw)
        if not t or not t[0].isalpha():
            continue
        clean_alpha = re.sub(r"[^A-Z0-9]", "", t.upper())
        if len(clean_alpha) < 4 or t.upper() in _NOISE_STANDALONE:
            continue
        # Digit-dominated strings are phone fragments, not names.
        if sum(c.isdigit() for c in clean_alpha) > sum(c.isalpha() for c in clean_alpha):
            continue
        # Three consecutive digits means a plate or fleet number read, e.g. "RDTAPE 2350".
        if re.search(r"\d{3}", t):
            continue
        if _COMPANY_SUFFIX_RE.search(t.upper()):
            candidates.append(t.upper())
    return max(candidates, key=len) if candidates else None


# ── the whole thing ───────────────────────────────────────────────────────────

def extract_fields(reads: List[TextResult]) -> dict:
    """Every read from every photo of one truck, reduced to the three fields intake needs.

    `confidence` counts how many independent reads agreed on the plate — HIGH at three or more.
    It is a vote count, deliberately not a probability: the engine's own score says how sure it
    was about some pixels, which is a different and much less useful question than whether
    several photos said the same thing.
    """
    usable = [(t, c) for t, c in reads if not is_osd(t)]
    body = deduplicate(usable)

    plate_votes: Counter = Counter()
    for text, _conf in usable:
        n = normalize_text(clean_token(text))
        if looks_like_plate(n):
            plate_votes[re.sub(r"[^A-Z0-9]", "", n)] += 1

    plate = plate_votes.most_common(1)[0][0] if plate_votes else None
    votes = sum(plate_votes.values())
    confidence = "NONE" if votes == 0 else ("HIGH" if votes >= 3 else "LOW")

    # A plate read is not a company name.
    body_without_plate = [t for t in body if not looks_like_plate(t)]

    phones = extract_phones(usable)

    return {
        "plate": plate,
        "plate_confidence": confidence,
        # Every number found, best-read first. `phone` is phones[0] and is kept only because a
        # single "the number" reads better in a log line than a list of one.
        "phones": phones,
        "phone": phones[0] if phones else None,
        "company": extract_company(body_without_plate),
        "raw": {
            "plate_candidates": dict(plate_votes),
            "body_texts": body,
        },
    }
