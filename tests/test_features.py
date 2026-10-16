import pytest
from phishing_features import URL_FEATURES, extract_features, normalise_url
from brand_protection import inspect_brand_impersonation

def test_feature_schema_and_types():
    features = extract_features("https://example.com/login")
    assert list(features) == URL_FEATURES
    assert all(isinstance(value, (int, float)) for value in features.values())

@pytest.mark.parametrize("url", [
    "https://micros0ft.com",
    "https://paypa1.com",
    "https://g00gle.com",
    "https://amaz0n.com",
    "https://go0gle.com",
    "https://facebok.com",
    "https://netfliix.com",
    "https://linkedln.com",
    "https://microsoftt.com",
])
def test_known_lookalikes_flagged(url):
    assert inspect_brand_impersonation(url) is not None

@pytest.mark.parametrize("url", [
    # Official domains must never be flagged
    "https://microsoft.com",
    "https://www.microsoft.com/en-us/",
    "https://amazon.com",
    "https://example.com/path/microsoft",
    # Official subdomains of brand domains
    "https://login.microsoftonline.com",
    "https://accounts.google.com/signin",
    "https://mail.google.com",
    "https://docs.google.com",
    "https://console.aws.amazon.com",
    "https://signin.aws.amazon.com",
    "https://outlook.live.com",
    "https://www.paypal.me",
    "https://developer.apple.com",
    "https://support.apple.com",
    "https://www.facebook.com",
    "https://www.instagram.com",
    "https://www.linkedin.com",
    "https://www.netflix.com",
    "https://www.youtube.com",
    "https://github.com",
    "https://store.steampowered.com",
    # Known-legitimate domains that could be confused with brands
    "https://steamship.com",
    "https://adobo.com",
    # Generic domains that just happen to have a short edit distance from short brands
    # (brands < 7 chars should NOT be flagged by edit distance)
    "https://chased.com",     # 1 edit from "chase" but "chase" is only 5 chars
    "https://yahooo.com",     # 1 edit from "yahoo" but "yahoo" is only 5 chars
    "https://applet.com",     # 1 edit from "apple" but "apple" is only 5 chars
    "https://steams.com",     # 1 edit from "steam" but "steam" is only 5 chars
])
def test_official_domains_and_legitimate_sites_not_flagged(url):
    result = inspect_brand_impersonation(url)
    assert result is None, f"Expected None but got {result} for {url}"

def test_brand_bait_word_combo_flagged():
    """Brand name + credential bait word outside official domain → flagged."""
    result = inspect_brand_impersonation("https://google-login.xyz")
    assert result is not None

def test_brand_bait_word_short_brand_skipped():
    """Very short brand labels (< 4 chars) are not used for the bait-word check."""
    # "fb" or "wf" are too short to trigger the bait-word combo
    result = inspect_brand_impersonation("https://fblogin.xyz")
    # "fb" is only 2 chars, should not trigger
    assert result is None

def test_kw_hits_hostname_only():
    """kw_hits should count keywords in the hostname, not the URL path."""
    feats_with_path_keyword = extract_features("https://example.com/login/account/verify")
    # "login", "account", "verify" are all in the PATH, not hostname
    assert feats_with_path_keyword["kw_hits"] == 0

    feats_with_host_keyword = extract_features("https://login-secure.example.com/page")
    # "login" and "secure" are in the hostname
    assert feats_with_host_keyword["kw_hits"] == 2

def test_missing_scheme_is_normalized():
    assert normalise_url("example.com") == "http://example.com"

@pytest.mark.parametrize("url", ["", "not a url", "https://"])
def test_invalid_urls_rejected(url):
    with pytest.raises(ValueError):
        extract_features(url)
