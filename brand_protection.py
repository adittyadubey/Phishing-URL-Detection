"""Deterministic lookalike-domain checks to complement the ML classifier.

This is a defense-in-depth heuristic, not a universal reputation service. It flags
high-confidence typo-squats of a curated set of frequently impersonated brands.
Keep this separate from model features so its decisions are transparent.
"""
from __future__ import annotations
import ipaddress
import re
from urllib.parse import urlsplit

# Canonical registrable domains for frequently impersonated brands. Extend this
# list from a maintained threat-intelligence feed for production use.
BRAND_DOMAINS = {
    "amazon": {"amazon.com"},
    "apple": {"apple.com"},
    "facebook": {"facebook.com", "fb.com"},
    "google": {"google.com", "google.co.uk", "google.co.in"},
    "instagram": {"instagram.com"},
    "linkedin": {"linkedin.com"},
    "microsoft": {"microsoft.com"},
    "netflix": {"netflix.com"},
    "paypal": {"paypal.com"},
    "steam": {"steampowered.com", "steamcommunity.com"},
    "whatsapp": {"whatsapp.com"},
    "yahoo": {"yahoo.com"},
    "adobe": {"adobe.com"},
    "dropbox": {"dropbox.com"},
    "docusign": {"docusign.com"},
    "coinbase": {"coinbase.com"},
    "binance": {"binance.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "instagram": {"instagram.com"},
}

# Common digit/character substitutions used in visual impersonation.
_TRANSLATE = str.maketrans({"0":"o", "1":"l", "3":"e", "4":"a", "5":"s", "7":"t", "8":"b"})


def _host(raw_url: str) -> str:
    raw = raw_url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    return (urlsplit(raw).hostname or "").lower().rstrip(".").encode("idna").decode("ascii")


def _registrable_label(host: str) -> str:
    """Approximate registrable label; avoids treating subdomains as the brand."""
    labels = host.split(".")
    if len(labels) < 2:
        return host
    # Common two-part public suffixes; no external PSL dependency required.
    two_part = {"co.uk", "org.uk", "ac.uk", "com.au", "net.au", "co.in", "firm.in", "com.br", "co.jp", "com.sg", "co.nz"}
    suffix2 = ".".join(labels[-2:])
    return labels[-3] if suffix2 in two_part and len(labels) >= 3 else labels[-2]


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    if a == b:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            if len(a) == len(b):
                i += 1; j += 1
            elif len(a) > len(b):
                i += 1
            else:
                j += 1
    if i < len(a) or j < len(b):
        edits += 1
    return edits == 1


def inspect_brand_impersonation(raw_url: str) -> dict | None:
    """Return a high-confidence brand lookalike finding, else None.

    Flags only near-exact brand spellings in the registrable domain, not a brand
    merely mentioned in a path/query or a legitimate subdomain of its official host.
    """
    try:
        host = _host(raw_url)
        ipaddress.ip_address(host)
        return None
    except ValueError:
        pass
    if not host or "." not in host:
        return None
    # An official domain or any subdomain of it is not a lookalike finding.
    for brand, official_domains in BRAND_DOMAINS.items():
        if any(host == d or host.endswith("." + d) for d in official_domains):
            return None
    label = _registrable_label(host).lower()
    normalized = label.translate(_TRANSLATE)
    for brand, official_domains in BRAND_DOMAINS.items():
        # Brand's canonical label can differ from its domain (e.g. Wells Fargo).
        canonical_labels = {brand}
        canonical_labels.update(_registrable_label(d) for d in official_domains)
        for canonical in canonical_labels:
            canon_norm = canonical.translate(_TRANSLATE)
            if normalized == canon_norm or _edit_distance_at_most_one(normalized, canon_norm):
                # Avoid flagging a short generic label based only on one edit.
                if len(canon_norm) >= 6 or normalized == canon_norm:
                    return {"brand": brand, "host": host, "label": label,
                            "reason": f"The domain '{host}' closely imitates the known brand '{brand}' but is not one of its configured official domains."}
    # Brand plus credential/payment bait in the registrable label is a strong signal.
    bait = ("login", "signin", "verify", "secure", "account", "support", "wallet", "billing", "password")
    for brand, official_domains in BRAND_DOMAINS.items():
        canonical_labels = {brand} | {_registrable_label(d) for d in official_domains}
        for canonical in canonical_labels:
            if canonical in label and any(token in label for token in bait) and label not in canonical_labels:
                return {"brand": brand, "host": host, "label": label,
                        "reason": f"The domain combines the brand '{brand}' with credential-related wording outside its configured official domain."}
    return None
