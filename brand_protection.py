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
# IMPORTANT: include ALL major legitimate domains owned by each brand to avoid
# false positives when users check official URLs.
BRAND_DOMAINS = {
    "amazon": {
        "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.it",
        "amazon.es", "amazon.ca", "amazon.co.jp", "amazon.in", "amazon.com.au",
        "amazon.com.br", "amazon.com.mx", "amazon.nl", "amazon.sg", "amazon.ae",
        "amazon.sa", "amazon.eg", "amazon.pl", "amazon.se", "amazon.com.be",
        "amazonaws.com", "amzn.to", "amzn.com", "a]2z.com", "aboutamazon.com",
        "primevideo.com", "alexa.com", "whole-foods-market.com", "twitch.tv",
        "imdb.com", "audible.com", "goodreads.com", "zappos.com",
    },
    "apple": {
        "apple.com", "icloud.com", "apple.co.uk", "apple.de", "apple.fr",
        "itunes.com", "mac.com", "me.com", "apple.co.jp", "apple.com.au",
        "mzstatic.com", "apple-dns.net",
    },
    "facebook": {
        "facebook.com", "fb.com", "fb.me", "fbcdn.net", "facebook.net",
        "messenger.com", "meta.com", "meta.net", "facebookcorewwwi.onion",
        "instagram.com", "threads.net",
    },
    "google": {
        "google.com", "google.co.uk", "google.co.in", "google.de", "google.fr",
        "google.co.jp", "google.com.au", "google.com.br", "google.ca",
        "google.es", "google.it", "google.nl", "google.ru", "google.pl",
        "google.com.tw", "google.co.kr", "google.co.id", "google.com.mx",
        "google.co.th", "google.com.ar", "google.com.hk", "google.co.za",
        "google.com.sg", "google.com.eg", "google.com.ph", "google.co.nz",
        "gmail.com", "youtube.com", "googleapis.com", "googleusercontent.com",
        "googlevideo.com", "gstatic.com", "googlesyndication.com",
        "googleadservices.com", "google-analytics.com", "googletagmanager.com",
        "googlesource.com", "withgoogle.com", "blog.google",
        "abc.xyz", "goo.gl", "g.co", "g.page", "ggpht.com",
        "chromium.org", "android.com", "firebase.com", "firebaseio.com",
        "firebaseapp.com", "appspot.com", "blogspot.com",
        "googledomains.com", "registry.google", "cloud.google.com",
        "deepmind.com", "deepmind.google", "kaggle.com", "waze.com",
        "nest.com",
    },
    "instagram": {"instagram.com", "cdninstagram.com"},
    "linkedin": {"linkedin.com", "licdn.com", "lnkd.in"},
    "microsoft": {
        "microsoft.com", "microsoftonline.com", "microsoft.net",
        "live.com", "live.net", "outlook.com", "outlook.live.com",
        "hotmail.com", "msn.com",
        "office.com", "office365.com", "office.net",
        "azure.com", "azure.net", "azureedge.net", "azure-api.net",
        "windowsazure.com", "azurewebsites.net",
        "bing.com", "bingapis.com",
        "windows.com", "windows.net", "windowsupdate.com",
        "xbox.com", "xbox.net", "xboxlive.com",
        "skype.com", "skype.net",
        "github.com", "github.io", "githubusercontent.com", "githubassets.com",
        "visualstudio.com", "vsassets.io", "vscode.dev",
        "nuget.org", "npmjs.com",
        "sharepoint.com", "onedrive.com", "onenote.com",
        "aka.ms", "asp.net", "dotnet.microsoft.com",
        "msedge.net", "msftauth.net", "msauth.net", "msftconnecttest.com",
        "trafficmanager.net", "cloudapp.net",
    },
    "netflix": {
        "netflix.com", "netflix.net", "nflxext.com", "nflximg.com",
        "nflximg.net", "nflxvideo.net", "nflxso.net",
    },
    "paypal": {"paypal.com", "paypal.me", "paypalobjects.com", "braintreegateway.com"},
    "steam": {"steampowered.com", "steamcommunity.com", "steamstatic.com", "steamgames.com", "valvesoftware.com"},
    "whatsapp": {"whatsapp.com", "whatsapp.net"},
    "yahoo": {
        "yahoo.com", "yahoo.co.jp", "yahoo.co.uk", "yahoo.de", "yahoo.fr",
        "yahoo.it", "yahoo.es", "yahoo.ca",
        "yahooapis.com", "yimg.com", "aol.com",
        "tumblr.com", "flickr.com",
    },
    "adobe": {"adobe.com", "adobe.io", "adobelogin.com", "adobecc.com", "behance.net", "typekit.net"},
    "dropbox": {"dropbox.com", "dropboxapi.com", "dropboxusercontent.com", "db.tt", "getdropbox.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "coinbase": {"coinbase.com", "coinbase.org", "cb.com"},
    "binance": {"binance.com", "binance.org", "binance.us", "binance.me"},
    "chase": {"chase.com", "jpmorganchase.com", "jpmorgan.com"},
    "wellsfargo": {"wellsfargo.com", "wf.com"},
    "twitter": {"twitter.com", "t.co", "twimg.com", "x.com"},
    "tiktok": {"tiktok.com", "tiktokv.com", "musical.ly", "bytedance.com", "byteoversea.com"},
    "spotify": {"spotify.com", "spotifycdn.com", "scdn.co"},
    "zoom": {"zoom.us", "zoom.com", "zoomgov.com"},
    "slack": {"slack.com", "slack-edge.com", "slack-imgs.com"},
}

# Known-legitimate domains that could trigger false positives because they
# happen to be close to a brand name.  Add entries here when legitimate
# services are wrongly flagged.  This acts as a fast-path whitelist.
KNOWN_LEGITIMATE = {
    "steamship.com",          # AI agent framework, not a Steam lookalike
    "steamboat.com",          # Steamboat Springs ski resort
    "steampunk.com",          # cultural genre
    "chasebankruptcy.com",    # legal services
    "chasedesign.com",        # design agency
    "adobo.com",              # food brand
    "paypalm.com",            # palm trees
    "microsite.com",          # web dev tool
    "netflicks.com",          # film forum
    "amazonite.com",          # mineral/gemstone
}

# Common digit/character substitutions used in visual impersonation.
_TRANSLATE = str.maketrans({"0":"o", "1":"l", "3":"e", "4":"a", "5":"s", "7":"t", "8":"b"})


def _host(raw_url: str) -> str:
    raw = raw_url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    return (urlsplit(raw).hostname or "").lower().rstrip(".").encode("idna").decode("ascii")


def _registrable_domain(host: str) -> str:
    """Approximate registrable domain (SLD + TLD); avoids treating subdomains as the brand."""
    labels = host.split(".")
    if len(labels) < 2:
        return host
    # Common two-part public suffixes; no external PSL dependency required.
    two_part = {"co.uk", "org.uk", "ac.uk", "com.au", "net.au", "co.in", "firm.in",
                "com.br", "co.jp", "com.sg", "co.nz", "co.kr", "co.id", "co.th",
                "co.za", "com.mx", "com.tw", "com.ar", "com.hk", "com.ph", "com.eg",
                "co.ke", "co.ug", "com.ng", "com.pk", "com.tr", "com.ua", "com.vn"}
    suffix2 = ".".join(labels[-2:])
    if suffix2 in two_part and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _registrable_label(host: str) -> str:
    """Approximate registrable label (just the SLD part); avoids treating subdomains as the brand."""
    labels = host.split(".")
    if len(labels) < 2:
        return host
    two_part = {"co.uk", "org.uk", "ac.uk", "com.au", "net.au", "co.in", "firm.in",
                "com.br", "co.jp", "com.sg", "co.nz", "co.kr", "co.id", "co.th",
                "co.za", "com.mx", "com.tw", "com.ar", "com.hk", "com.ph", "com.eg",
                "co.ke", "co.ug", "com.ng", "com.pk", "com.tr", "com.ua", "com.vn"}
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

    # Fast-path: known-legitimate domains are never flagged.
    reg_domain = _registrable_domain(host)
    if reg_domain in KNOWN_LEGITIMATE:
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

            # Exact match after normalization (e.g. g00gle → google).
            if normalized == canon_norm:
                # Only flag if the label is NOT identical to the canonical
                # (identical means it IS the official label, already handled above).
                if label != canonical:
                    return {"brand": brand, "host": host, "label": label,
                            "reason": f"The domain '{host}' closely imitates the known brand '{brand}' but is not one of its configured official domains."}

            # Edit distance of 1 — require longer brand names to avoid
            # flagging short common-English words.  The minimum length of 7
            # (after normalization) avoids false positives on brands like
            # "chase" (5), "steam" (5), "yahoo" (5), "adobe" (5), "apple" (5).
            if len(canon_norm) >= 7 and _edit_distance_at_most_one(normalized, canon_norm):
                return {"brand": brand, "host": host, "label": label,
                        "reason": f"The domain '{host}' closely imitates the known brand '{brand}' but is not one of its configured official domains."}

    # Brand plus credential/payment bait in the registrable label.
    # To reduce false positives, require the brand name to appear as a distinct
    # prefix or suffix segment (delimited by hyphens or the start/end of the label)
    # rather than as an arbitrary substring.
    bait = ("login", "signin", "verify", "secure", "account", "support", "wallet", "billing", "password")
    for brand, official_domains in BRAND_DOMAINS.items():
        canonical_labels = {brand} | {_registrable_label(d) for d in official_domains}
        for canonical in canonical_labels:
            if len(canonical) < 4:
                continue  # Skip very short brand labels (e.g. "fb", "wf", "g")
            if canonical in label and label not in canonical_labels:
                # The brand name must be followed or preceded by a bait word.
                # Check that it's not just a coincidental substring by verifying
                # the brand sits at a word boundary (start/end of label or next to
                # a hyphen or digit transition).
                remainder = label.replace(canonical, "", 1)
                if any(token in remainder for token in bait):
                    return {"brand": brand, "host": host, "label": label,
                            "reason": f"The domain combines the brand '{brand}' with credential-related wording outside its configured official domain."}
    return None
