"""Tests for the pure text logic FreightDesk never covered.

Every case here is a real failure recorded in FreightDesk's session log — these are not
hypotheticals, they are the bugs that were found by watching OCR output and fixed one at a
time. No models, no images, no I/O: the cheapest tests in the project.
"""
from ocr.fields import (
    extract_phones,deduplicate, extract_company, extract_fields, extract_phone, is_osd,
                        looks_like_plate)


class TestLooksLikePlate:
    def test_a_clean_indian_plate(self):
        assert looks_like_plate("MH12AB1234")
        assert looks_like_plate("MH 12 AB 1234")
        assert looks_like_plate("RJ14CA1234")

    def test_a_garbled_plate_still_counts_via_its_state_code(self):
        # OCR turns MH14 into MHI4 constantly. No plate regex matches, but it is still a plate.
        assert looks_like_plate("MHI4CD5678")

    def test_transport_is_not_a_plate_even_though_TR_is_tripura(self):
        # The original bug: every truck with TRANSPORT painted on it got it read as a plate.
        assert not looks_like_plate("TRANSPORT")
        assert not looks_like_plate("TRANSPORTS")

    def test_a_truncated_read_of_a_blacklisted_word_is_also_refused(self):
        # Substring matching both ways exists for exactly this.
        assert not looks_like_plate("TRANSPOR")

    def test_letters_only_is_never_a_plate(self):
        # ASHOK (Leyland badge) starts with AS = Assam, and is useless as plate data.
        assert not looks_like_plate("ASHOK")
        assert not looks_like_plate("ASSAM")

    def test_a_bare_number_is_not_a_plate(self):
        assert not looks_like_plate("9811008120")


class TestIsOsd:
    def test_camera_clock_fragments(self):
        assert is_osd("2023/03")
        assert is_osd("12:45")
        assert is_osd("12.44.30")
        assert is_osd("12.44303")
        assert is_osd("2024-01-01")

    def test_watermarks(self):
        assert is_osd("IPC")
        assert is_osd("PC")
        assert is_osd("ch4")

    def test_a_real_plate_is_not_an_overlay(self):
        assert not is_osd("MH12AB1234")
        assert not is_osd("KUMAR ROADWAYS")


class TestPhone:
    def test_a_plain_number(self):
        assert extract_phone([("9811008120", 0.9)]) == "9811008120"

    def test_letters_misread_as_digits_are_repaired(self):
        # The digit-confusion map: 981i008120 -> 9811008120.
        assert extract_phone([("981i008120", 0.9)]) == "9811008120"
        assert extract_phone([("98IIO08I20", 0.9)]) == "9811008120"

    def test_the_confusion_map_is_only_applied_to_digit_heavy_strings(self):
        # Guarded at 6+ digits on purpose. Applied to ordinary text it would turn every
        # I and O in a company name into a 1 and a 0, inventing phone numbers out of words.
        assert extract_phone([("SOLIS", 0.9)]) is None
        assert extract_phone([("LOGISTICS", 0.9)]) is None

    def test_a_country_code_is_stripped(self):
        assert extract_phone([("+91 98110 08120", 0.9)]) == "9811008120"
        assert extract_phone([("09811008120", 0.9)]) == "9811008120"

    def test_surrounding_text_does_not_matter(self):
        assert extract_phone([("HO,98110-08120", 0.9)]) == "9811008120"

    def test_a_low_confidence_read_is_not_offered_as_callable(self):
        # Somebody would actually ring this number. One smudged read is not enough.
        assert extract_phone([("9811008120", 0.3)]) is None

    def test_the_majority_read_wins_over_a_one_digit_variant(self):
        reads = [("9811008120", 0.9), ("9811008120", 0.9), ("9811008121", 0.9)]
        assert extract_phone(reads) == "9811008120"

    def test_a_number_that_is_not_an_indian_mobile_is_ignored(self):
        assert extract_phone([("1234567890", 0.9)]) is None  # must start 6-9

    def test_every_number_on_the_truck_comes_back(self):
        """A truck usually carries more than one, and a telecaller needs all of them.

        The real read below is from the side of a Ludhiana truck: two numbers painted in a row.
        Returning only the best-read one meant that if the first did not answer, the second --
        visible in the same photo -- was simply lost.
        """
        line = ("GURU NANAK R0ADCARRIER，TRP0RT NAGAR.LUDHIANA.141003,"
                "M0B.N0.98787.70969,84370· 36313")
        assert set(extract_phones([(line, 0.9)])) == {"9878770969", "8437036313"}

    def test_near_duplicate_reads_of_one_number_collapse(self):
        """Otherwise a CSR is offered six versions of the same number to choose between."""
        reads = [("9811008120", 0.9), ("9811008120", 0.9), ("9811008121", 0.9)]
        assert extract_phones(reads) == ["9811008120"]

    def test_no_numbers_is_an_empty_list_not_none(self):
        assert extract_phones([("SHARMA LOGISTICS", 0.9)]) == []

    def test_a_number_is_never_invented_across_a_boundary(self):
        # A real read off the side of a truck in Ludhiana. Concatenating every digit on the
        # line gives 101114100308098787709698437036313, in which [6-9]\d{9} matches
        # "8098787709" -- half the mangled "M0B.N0." and half the real number. It looks like a
        # perfectly ordinary mobile and belongs to a stranger who would get the call.
        line = ("GURU NANAK R0ADCARRIER，TRP0RT NAGAR.LUDHIANA.141003,"
                "M0B.N0.98787.70969,84370· 36313")
        assert extract_phone([(line, 0.9)]) in {"9878770969", "8437036313"}

    def test_a_pincode_is_not_a_phone_number(self):
        # Six digits next to a city name, which is what a pincode looks like.
        assert extract_phone([("LUDHIANA.141003", 0.9)]) is None


class TestCompany:
    def test_a_corporate_suffix(self):
        assert extract_company(["KUMAR ROADWAYS"]) == "KUMAR ROADWAYS"
        assert extract_company(["PATEL FREIGHT LINES"]) == "PATEL FREIGHT LINES"

    def test_the_longest_candidate_wins(self):
        assert extract_company(["SHARMA LOGISTICS", "ABC TRANSPORT"]) == "SHARMA LOGISTICS"

    def test_a_plate_or_fleet_number_is_not_a_company(self):
        # Three consecutive digits: "RDTAPE 2350" was a real false positive.
        assert extract_company(["RDTAPE 2350"]) is None
        assert extract_company(["FRJASCR2455"]) is None

    def test_a_phone_fragment_is_not_a_company(self):
        assert extract_company(["HO 9833727900"]) is None

    def test_a_bare_label_is_not_a_company(self):
        assert extract_company(["PH"]) is None
        assert extract_company(["MOBILE"]) is None


class TestDeduplicate:
    def test_near_duplicates_collapse(self):
        # The same painted name read twice, one letter apart.
        out = deduplicate([("KUMAR ROADWAYS", 0.9), ("KUMAR ROADWAY5", 0.8)])
        assert len(out) == 1

    def test_distinct_texts_survive(self):
        out = deduplicate([("KUMAR ROADWAYS", 0.9), ("MH12AB1234", 0.9)])
        assert len(out) == 2


class TestExtractFields:
    def test_a_whole_truck(self):
        reads = [("KUMAR ROADWAYS", 0.99), ("PH 9811008120", 0.98), ("MH12AB1234", 0.99)]
        got = extract_fields(reads)
        assert got["plate"] == "MH12AB1234"
        assert got["phone"] == "9811008120"
        assert got["company"] == "KUMAR ROADWAYS"
        assert got["plate_confidence"] == "LOW"      # one vote

    def test_three_agreeing_reads_are_HIGH(self):
        reads = [("MH12AB1234", 0.9)] * 3
        assert extract_fields(reads)["plate_confidence"] == "HIGH"

    def test_nothing_readable_is_an_ordinary_outcome(self):
        got = extract_fields([])
        assert got["plate"] is None
        assert got["plate_confidence"] == "NONE"
        assert got["phone"] is None
        assert got["company"] is None

    def test_camera_overlays_do_not_reach_the_fields(self):
        got = extract_fields([("2023/03/14", 0.9), ("12:45:33", 0.9), ("MH12AB1234", 0.9)])
        assert got["plate"] == "MH12AB1234"
        assert all("2023" not in t for t in got["raw"]["body_texts"])

    def test_the_plate_is_not_also_offered_as_the_company(self):
        got = extract_fields([("MH12AB1234", 0.9)])
        assert got["company"] is None
