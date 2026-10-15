"""Download a page's HTML (safely, once, without running it) and extract the
26 page-content features that appear in the PhiUSIIL CSV.

The page is never rendered and no JavaScript is executed. Only plain HTML text is
downloaded (max 2 MB, short timeout, private/internal addresses refused).

The original extraction code behind the dataset is not part of the project, so
these definitions are reconstructed from the column names and value ranges.
Treat live predictions as approximate even though the dataset metrics are exact.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import warnings
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CONTENT_FEATURES = [
    "LineOfCode", "LargestLineLength", "HasTitle", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup", "NoOfiFrame",
    "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
    "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage",
    "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
]

MAX_BYTES = 2_000_000
TIMEOUT = 8
MAX_REDIRECTS = 5
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PhishingUrlResearch/1.0)", "Accept": "text/html,*/*;q=0.5"}
SOCIAL = re.compile(r"(facebook|twitter|x\.com|instagram|linkedin|youtube|pinterest|tiktok|whatsapp|telegram|reddit)\.", re.I)


class FetchError(Exception):
    pass


def _host(url: str) -> str:
    h = (urlsplit(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def _check_public(url: str) -> None:
    """Refuse non-http(s) schemes and hosts that resolve to private/internal addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Only http and https addresses can be opened.")
    if parts.username is not None or parts.password is not None:
        raise FetchError("URLs containing embedded usernames or passwords are not allowed.")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise FetchError("The URL contains an invalid port.") from exc
    if not 1 <= port <= 65535:
        raise FetchError("The URL contains an invalid port.")
    try:
        infos = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError("The domain name could not be found (the site may be offline).") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise FetchError("That address points to a private or internal network, so it was not opened.")


def fetch_page(url: str) -> dict:
    """Return {'html', 'final_url', 'redirects', 'self_redirect', 'status'}; raises FetchError."""
    session = requests.Session()
    # Do not inherit proxy settings from the host environment. Keep TLS certificate
    # verification enabled; an SSL failure is a safe fetch failure, not a reason to disable it.
    session.trust_env = False
    current, hops, self_redirect = url, 0, False
    while True:
        _check_public(current)
        try:
            resp = session.get(current, headers=HEADERS, timeout=(3, TIMEOUT), stream=True,
                               allow_redirects=False, verify=True)
        except requests.exceptions.SSLError as exc:
            raise FetchError("HTTPS certificate verification failed; the page was not opened. URL-only analysis will be used.") from exc
        except requests.exceptions.RequestException as exc:
            raise FetchError("The page could not be reached (the site may be offline or blocking requests). URL-only analysis will be used.") from exc
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location", "")
            target = urljoin(current, location) if location else ""
            resp.close()
            hops += 1
            if hops > MAX_REDIRECTS or not target:
                raise FetchError("The page redirected too many times or returned an invalid redirect.")
            # Validate each redirect target before issuing the next request.
            try:
                _check_public(target)
            except FetchError as exc:
                raise FetchError("The redirect points to an invalid or private/internal destination and was blocked.") from exc
            if _host(target) == _host(current):
                self_redirect = True
            current = target
            continue
        break
    try:
        ctype = resp.headers.get("Content-Type", "").lower()
        # Only treat a normal successful page response as usable model input.
        # HTTP 202 often means the request was accepted for processing, not that
        # the site's ordinary homepage was returned (e.g. bot/challenge pages).
        if resp.status_code != 200:
            raise FetchError(
                f"The site returned HTTP {resp.status_code}, not a normal HTTP 200 page. "
                "Page features are inconclusive; URL-only analysis will be used."
            )
        if ctype and "html" not in ctype and "text" not in ctype:
            raise FetchError("The address does not return a web page (not HTML).")
        raw = b""
        content_length = resp.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_BYTES:
                    raise FetchError("The page is larger than the allowed download limit.")
            except ValueError:
                pass
        for chunk in resp.iter_content(65536):
            if not chunk:
                continue
            remaining = MAX_BYTES - len(raw)
            raw += chunk[:remaining]
            if len(raw) >= MAX_BYTES:
                break
        enc = resp.encoding or "utf-8"
        html = raw.decode(enc, errors="replace")
    finally:
        resp.close()
    if not html.strip():
        session.close()
        raise FetchError("The page was empty; URL-only analysis will be used.")

    # Do not feed common anti-bot/challenge responses to the full-feature model.
    # Their tiny HTML is not representative of the destination site's real page.
    challenge_markers = (
        "verify you are human", "checking your browser", "captcha", "cf-challenge",
        "attention required! | cloudflare", "robot or human", "automated requests",
        "enable javascript and cookies to continue",
    )
    sample = html[:100_000].lower()
    if any(marker in sample for marker in challenge_markers):
        session.close()
        raise FetchError(
            "The server returned a bot-check or challenge page rather than the normal website. "
            "Page features are inconclusive; URL-only analysis will be used."
        )
    if len(raw) < 1024:
        session.close()
        raise FetchError(
            "The server returned an unusually small HTML response, which may be an error or block page. "
            "Page features are inconclusive; URL-only analysis will be used."
        )

    status_code = resp.status_code
    session.close()
    return {"html": html, "final_url": current, "redirects": hops, "self_redirect": self_redirect,
            "status": status_code, "size": len(raw)}


def extract_content_features(page: dict) -> dict:
    html, base = page["html"], page["final_url"]
    base_host = _host(base)
    soup = BeautifulSoup(html, "html.parser")
    lines = html.splitlines() or [""]
    text = soup.get_text(" ", strip=True).lower()

    def meta(name):
        return soup.find("meta", attrs={"name": re.compile(f"^{name}$", re.I)})

    desc = meta("description")
    inputs = soup.find_all("input")
    itype = lambda t: any((i.get("type") or "").lower() == t for i in inputs)

    anchors = [a.get("href", "") for a in soup.find_all("a", href=True)]
    refs = anchors + [t.get("href", "") for t in soup.find_all("link", href=True)] + [
        t.get("src", "") for t in soup.find_all(["script", "img", "iframe"], src=True)]
    self_ref = empty = external = 0
    for ref in refs:
        ref = ref.strip()
        if ref in ("", "#") or ref.lower().startswith(("javascript:", "#")):
            empty += 1
        elif ref.lower().startswith(("mailto:", "tel:", "data:")):
            continue
        else:
            h = _host(urljoin(base, ref))
            if h == base_host or not h:
                self_ref += 1
            else:
                external += 1

    ext_form = False
    for form in soup.find_all("form", action=True):
        act = form["action"].strip()
        if act and urlsplit(urljoin(base, act)).scheme in ("http", "https") and _host(urljoin(base, act)) != base_host:
            ext_form = True

    submit = itype("submit") or itype("image") or any(
        (b.get("type") or "submit").lower() == "submit" for b in soup.find_all("button"))
    return {
        "LineOfCode": len(lines),
        "LargestLineLength": max(len(l) for l in lines),
        "HasTitle": int(bool(soup.title and soup.title.get_text(strip=True))),
        "HasFavicon": int(any("icon" in " ".join(l.get("rel", [])).lower() for l in soup.find_all("link"))),
        "Robots": int(meta("robots") is not None),
        "IsResponsive": int(meta("viewport") is not None),
        "NoOfURLRedirect": int(page["redirects"] > 0),
        "NoOfSelfRedirect": int(page["self_redirect"]),
        "HasDescription": int(bool(desc and desc.get("content", "").strip())),
        "NoOfPopup": len(re.findall(r"window\.open\s*\(", html)),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": int(ext_form),
        "HasSocialNet": int(any(SOCIAL.search(a) for a in anchors)),
        "HasSubmitButton": int(submit),
        "HasHiddenFields": int(itype("hidden")),
        "HasPasswordField": int(itype("password")),
        "Bank": int("bank" in text),
        "Pay": int("pay" in text),
        "Crypto": int(bool(re.search(r"crypto|bitcoin|ethereum", text))),
        "HasCopyrightInfo": int("\u00a9" in text or "copyright" in text),
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": len([l for l in soup.find_all("link") if "stylesheet" in " ".join(l.get("rel", [])).lower()]),
        "NoOfJS": len(soup.find_all("script")),
        "NoOfSelfRef": self_ref,
        "NoOfEmptyRef": empty,
        "NoOfExternalRef": external,
    }
