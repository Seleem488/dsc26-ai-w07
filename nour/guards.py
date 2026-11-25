
"""PII detection and redaction for pharmacist notes."""
from __future__ import annotations

import re
from datetime import date


_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

_DIGIT_CHARS = r"0-9٠-٩۰-۹"
_DIGIT_RE = re.compile(rf"[{_DIGIT_CHARS}]")

GOVERNORATE_CODES = {
    "01", "02", "03", "04",
    "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "31", "32", "33", "34", "35",
    "88",
}


def _normalize_digits(text: str) -> str:
    """Convert Arabic-Indic and Persian digits to ASCII without changing length."""
    return text.translate(_DIGIT_TRANSLATION)


def _valid_national_id(value: str) -> bool:
    """Validate the date and governorate portion of a 14-digit national ID."""
    if len(value) != 14 or value[0] not in {"2", "3"}:
        return False

    century = 1900 if value[0] == "2" else 2000

    try:
        year = century + int(value[1:3])
        month = int(value[3:5])
        day = int(value[5:7])
        date(year, month, day)
    except ValueError:
        return False

    governorate = value[7:9]
    if governorate not in GOVERNORATE_CODES:
        return False

    return True


def find_pii(text: str) -> list[dict]:
    """Find Egyptian national IDs and mobile phone numbers in text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = _normalize_digits(text)
    findings = []

    # National IDs must be exactly 14 consecutive digits.
    for match in re.finditer(rf"[{_DIGIT_CHARS}]+", text):
        original_digits = match.group()
        start, end = match.span()

        if len(original_digits) != 14:
            continue

        # A national ID cannot touch another digit.
        if start > 0 and _DIGIT_RE.fullmatch(text[start - 1]):
            continue
        if end < len(text) and _DIGIT_RE.fullmatch(text[end]):
            continue

        digits = _normalize_digits(original_digits)

        if _valid_national_id(digits):
            findings.append({
                "type": "NATIONAL_ID",
                "start": start,
                "end": end,
            })

    # A separator may appear between digit groups.
    separator = r"[ -]?"
    digit = rf"[{_DIGIT_CHARS}]"

    # Optional country prefix: +2, 002, or 2.
    # The local mobile number is 01[0125] followed by eight digits.
    phone_pattern = re.compile(
        rf"(?<![{_DIGIT_CHARS}+])"
        rf"(?:(?:\+{separator}2|0{separator}0{separator}2|2){separator})?"
        rf"0{separator}1{separator}[0125]"
        rf"(?:{separator}{digit}){{8}}"
        rf"(?![{_DIGIT_CHARS}])"
    )

    for match in phone_pattern.finditer(normalized):
        start, end = match.span()

        findings.append({
            "type": "PHONE",
            "start": start,
            "end": end,
        })

    # National IDs take priority over overlapping phone matches.
    national_ids = [
        item for item in findings
        if item["type"] == "NATIONAL_ID"
    ]

    phones = [
        item for item in findings
        if item["type"] == "PHONE"
        and not any(
            item["start"] < national_id["end"]
            and item["end"] > national_id["start"]
            for national_id in national_ids
        )
    ]

    return sorted(
        national_ids + phones,
        key=lambda item: (item["start"], item["end"]),
    )


def redact(text: str) -> tuple[str, list[dict]]:
    """Replace detected PII with labeled placeholders."""
    findings = find_pii(text)

    if not findings:
        return text, findings

    parts = []
    cursor = 0

    for item in findings:
        start = item["start"]
        end = item["end"]
        pii_type = item["type"]

        parts.append(text[cursor:start])
        parts.append(f"[{pii_type}]")
        cursor = end

    parts.append(text[cursor:])

    return "".join(parts), findings
