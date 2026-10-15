"""Extract URL-only features from raw URL text.

The feature names and definitions mirror the URL-derived columns in the PhiUSIIL
CSV (host_len, sld_len, ...). Only the text of the URL is used; no network
request is ever made. Definitions were checked against the CSV for internal
consistency (e.g. n_labels == n_dots + 1, digit_ratio * host_len == n_digit,
vowel_ratio * sld_len is an integer). The keyword list is an assumption because
the original extraction code is not part of the project.
"""
from __future__ import annotations

import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlsplit

URL_FEATURES = [
    "host_len", "sld_len", "n_dots", "n_hyphen", "n_digit", "digit_ratio",
    "host_entropy", "sld_entropy", "is_ip", "longest_label", "n_labels",
    "tld_len", "vowel_ratio", "max_consec_consonants", "kw_hits",
    "has_at", "has_punycode",
]

KEYWORDS = (
    "login", "signin", "secure", "account", "update", "verify", "bank",
    "confirm", "password", "wallet", "billing", "support",
)
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_VOWELS = set("aeiou")


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def _max_consonant_run(text: str) -> int:
    best = run = 0
    for ch in text:
        if ch.isalpha() and ch not in _VOWELS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def normalise_url(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("Enter a URL as text.")
    raw = raw.strip()
    if not raw:
        raise ValueError("Please enter a URL.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    return raw


def extract_features(raw_url: str) -> dict:
    """Return the 17 URL features for ``raw_url``. Raises ValueError if unusable."""
    url = normalise_url(raw_url)
    try:
        host = (urlsplit(url).hostname or "").lower()
        _ = urlsplit(url).port  # force validation of any explicit port
    except ValueError as exc:
        raise ValueError("That does not look like a valid web address.") from exc
    host = host.strip(".")
    if not host or any(ch.isspace() for ch in host):
        raise ValueError("That does not look like a valid web address.")
    # The dataset's host column has a leading "www." removed (2-label hosts are
    # mostly legitimate, 3-label hosts mostly phishing), so do the same here.
    if host.startswith("www.") and host.count(".") >= 2:
        host = host[4:]
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0
    labels = [label for label in host.split(".") if label]
    if not is_ip and (len(labels) < 2 or any(not label for label in labels)):
        raise ValueError("Enter a full web address with a domain, e.g. https://example.com")
    if is_ip and len(labels) != 4:
        labels = [host]

    sld = labels[-2] if len(labels) >= 2 else labels[0]
    tld = labels[-1] if len(labels) >= 2 else ""
    digits = sum(ch.isdigit() for ch in host)
    lowered = url.lower()
    return {
        "host_len": len(host),
        "sld_len": len(sld),
        "n_dots": host.count("."),
        "n_hyphen": host.count("-"),
        "n_digit": digits,
        "digit_ratio": digits / len(host),
        "host_entropy": _entropy(host),
        "sld_entropy": _entropy(sld),
        "is_ip": is_ip,
        "longest_label": max(len(label) for label in labels),
        "n_labels": len(labels),
        "tld_len": len(tld),
        "vowel_ratio": sum(ch in _VOWELS for ch in sld) / len(sld),
        "max_consec_consonants": _max_consonant_run(sld),
        "kw_hits": sum(k in lowered for k in KEYWORDS),
        "has_at": int("@" in url),
        "has_punycode": int("xn--" in host),
    }
