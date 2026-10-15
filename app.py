"""Phishing URL Detection Using Machine Learning - Streamlit interface."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from urllib.parse import urlsplit

from page_features import FetchError, extract_content_features, fetch_page
from phishing_features import URL_FEATURES, extract_features, normalise_url
from brand_protection import inspect_brand_impersonation
from train_model import METADATA_PATH, MODEL_PATH, train

st.set_page_config(page_title="Phishing URL Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

ACCENT = "#4c8dff"

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding: 2.4rem clamp(2rem, 4vw, 5rem) 4rem; max-width: 1600px; width: 100%;}
@media (min-width: 1200px) {
  .block-container {padding-top: 2.6rem;}
  .hero {align-items: center; gap: 20px;}
  .hero-icon {width: 56px; height: 56px; border-radius: 14px;}
  .hero h1 {font-size: 2.05rem; letter-spacing: -.025em;}
  .hero p {font-size: .95rem; max-width: 1000px; line-height: 1.7;}
  .rule {margin: 1.8rem 0 2rem;}
  .sec {font-size: 1.3rem; margin-top: 2rem;}
  .sec-cap {font-size: .9rem; max-width: 1000px; margin-bottom: 1.2rem;}
  [data-testid="stForm"] {padding: 1.5rem 1.6rem 1.2rem; border-radius: 14px;}
  [data-testid="stForm"] input {font-size: 1rem !important; min-height: 3.2rem;}
  [data-testid="stForm"] button {font-size: .95rem; padding: .7rem 1.2rem;}
  .cards {gap: 18px; margin-top: .8rem;}
  .card {padding: 20px 22px 18px; border-radius: 14px;}
  .card .k {font-size: .82rem;}
  .card .v {font-size: 2rem; margin: .35rem 0 .45rem;}
  .card .d {font-size: .76rem;}
  .verdict {padding: 24px 28px; border-radius: 14px; margin-top: 1.4rem;}
  .verdict .t {font-size: 1.35rem;}
  .verdict .s, .agree {font-size: .9rem;}
  .status {font-size: .82rem;}
  .flags {font-size: .88rem;}
  .mp th, .cmp th {font-size: .78rem; padding: 10px 12px;}
  .mp td, .cmp td {font-size: .9rem; padding: 12px;}
  .pill {font-size: .78rem; padding: 4px 12px;}
  .minibar {width: 130px; height: 7px;}
  .sub {font-size: 1rem;}
  .read li {font-size: .86rem; padding: 12px 6px 12px 14px;}
  .note, .cap {font-size: .78rem; max-width: 900px;}
  .info td {font-size: .9rem; padding: 12px 8px;}
  .foot {font-size: .78rem;}
}
@media (min-width: 1500px) {
  .block-container {max-width: 1720px;}
}
html, body, [class*="css"] {color: #e6edf7;}

/* Desktop-first responsive workspace */
[data-testid="stAppViewContainer"] {background: #080e1c;}
[data-testid="stHeader"] {background: rgba(8,14,28,.88);}
@media (min-width: 1200px) {
  [data-testid="stAppViewContainer"] .main .block-container {max-width: 1720px;}
  .stMarkdown {overflow-wrap: anywhere;}
}
/* header */
.hero {display: flex; gap: 14px; align-items: flex-start;}
.hero-icon {flex: 0 0 auto; width: 44px; height: 44px; border-radius: 10px; background: #0f1f3d;
  border: 1px solid #1f3b73; display: flex; align-items: center; justify-content: center;}
.hero h1 {font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin: 0; padding: 0; color: #f1f5fb;}
.hero p {font-size: .82rem; line-height: 1.65; color: #7d8fb0; margin: .55rem 0 0 0; max-width: 560px;}
.rule {border: 0; border-top: 1px solid #1a2744; margin: 1.4rem 0 1.6rem 0;}

/* section titles */
.sec {font-size: 1.08rem; font-weight: 700; margin: 1.9rem 0 .35rem 0; color: #f1f5fb;}
.sec-cap {font-size: .78rem; line-height: 1.55; color: #c3cde0; margin: 0 0 .9rem 0; max-width: 560px;}
.sub {font-size: .8rem; font-weight: 700; margin: 0 0 .5rem 0; color: #f1f5fb;}

/* URL form */
[data-testid="stForm"] {background: #0f1a30; border: 1px solid #1e2d4d; border-radius: 10px; padding: 1rem 1rem .8rem 1rem;}
[data-testid="stForm"] input {font-family: "Source Code Pro", ui-monospace, Menlo, Consolas, monospace !important;
  font-size: .82rem !important; background: #0a1428 !important;}
[data-testid="stForm"] [data-baseweb="input"], [data-testid="stForm"] [data-baseweb="base-input"] {background: #0a1428 !important; border-color: #14203a !important;}
[data-testid="stForm"] button {background: transparent; border: 0; color: #f1f5fb; font-size: .78rem; font-weight: 600; padding-left: .6rem;}
[data-testid="stForm"] button:hover {color: #4c8dff; background: transparent;}

/* metric cards */
.cards {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: .4rem;}
.card {background: #0f1a30; border: 1px solid #1e2d4d; border-radius: 10px; padding: 14px 14px 12px 14px;}
.card .k {font-size: .68rem; color: #8ea0c2;}
.card .v {font-size: 1.55rem; font-weight: 700; color: #4c8dff; margin: .15rem 0 .35rem 0; letter-spacing: -.01em;}
.card .d {font-size: .62rem; line-height: 1.4; color: #6f84ab;}
@media (max-width: 640px) {.cards {grid-template-columns: repeat(2, 1fr);}}

/* confusion matrix */
.cm-wrap {background: #0f1a30; border-radius: 8px; padding: 10px 10px 8px 10px;}
.cm {display: grid; grid-template-columns: 22px 78px 1fr 1fr; grid-template-rows: 1fr 1fr 26px 22px; gap: 4px; aspect-ratio: 1.25 / 1;}
.cm .ylab {grid-column: 1; grid-row: 1 / span 2; display: flex; align-items: center; justify-content: center;
  writing-mode: vertical-rl; transform: rotate(180deg); font-size: .68rem; color: #6f84ab;}
.cm .rl {grid-column: 2; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: .74rem; color: #dfe7f5;}
.cm .c {display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 700; color: #fff;}
.cm .cl {display: flex; align-items: center; justify-content: center; font-size: .72rem; color: #dfe7f5;}
.cm .xlab {grid-column: 3 / span 2; display: flex; align-items: center; justify-content: center; font-size: .68rem; color: #6f84ab;}

/* how to read */
.read {list-style: none !important; margin: 0 !important; padding: 0 !important;}
.read li {margin-left: 0 !important;}
.read li {font-size: .72rem; line-height: 1.45; color: #e6edf7; padding: 9px 4px 9px 12px; border-bottom: 1px solid #1a2744;}
.read li b {color: #4c8dff; font-weight: 700;}
.note {font-size: .64rem; line-height: 1.5; color: #8ea0c2; margin-top: 8px;}
.cap {font-size: .66rem; line-height: 1.5; color: #8ea0c2; margin: .6rem 0 0 0; max-width: 520px;}

/* verdict */
.verdict {border-radius: 10px; padding: 16px 18px; margin-top: 1rem; border: 1px solid; }
.verdict.bad {background: #2a1220; border-color: #6b2440;}
.verdict.ok {background: #0e2430; border-color: #1d5560;}
.verdict .t {font-size: 1.1rem; font-weight: 700;}
.verdict.bad .t {color: #ff7b8c;} .verdict.ok .t {color: #4fe0b0;}
.verdict .s {font-size: .78rem; color: #c3cde0; margin-top: .25rem; line-height: 1.5;}
.bar {height: 8px; border-radius: 6px; background: #14203a; margin-top: .8rem; overflow: hidden;}
.bar > div {height: 100%; border-radius: 6px;}
.flags {font-size: .76rem; color: #c3cde0; margin: .7rem 0 0 0; padding-left: 1.1rem; line-height: 1.6;}

.status {font-size: .72rem; color: #8ea0c2; margin-top: .8rem;}
.status.warn {color: #ffcf7a;}
/* per-model predictions */
.mp {width: 100%; border-collapse: collapse; margin-top: .9rem;}
.mp th {font-size: .64rem; font-weight: 600; color: #8ea0c2; text-align: left; padding: 6px 6px; border-bottom: 1px solid #1a2744;}
.mp td {font-size: .76rem; padding: 8px 6px; border-bottom: 1px solid #1a2744; color: #e6edf7; vertical-align: middle;}
.mp tr:last-child td {border-bottom: 0;}
.mp .hl td:first-child {font-weight: 700;}
.pill {display: inline-block; font-size: .66rem; font-weight: 700; padding: 2px 9px; border-radius: 999px;}
.pill.bad {background: #3a1626; color: #ff7b8c;} .pill.ok {background: #0f2f33; color: #4fe0b0;}
.minibar {display: inline-block; vertical-align: middle; width: 90px; height: 6px; border-radius: 4px; background: #14203a; margin-right: 8px; overflow: hidden;}
.minibar > div {height: 100%;}
.agree {font-size: .76rem; color: #c3cde0; margin: .8rem 0 0 0;}
.cmp {width: 100%; border-collapse: collapse; margin-top: .5rem;}
.cmp th {font-size: .64rem; font-weight: 600; color: #8ea0c2; text-align: right; padding: 7px 8px; border-bottom: 1px solid #1a2744;}
.cmp th:first-child, .cmp td:first-child {text-align: left;}
.cmp td {font-size: .76rem; text-align: right; padding: 8px; border-bottom: 1px solid #1a2744; color: #dfe7f5;}
.cmp tr.hl td {color: #4c8dff; font-weight: 700;}
.cmp, .cmp th, .cmp td, .mp, .mp th, .mp td {border-left: 0 !important; border-right: 0 !important; border-top: 0 !important;}

/* project info */
.info {width: 100%; border-collapse: collapse; margin-top: .4rem; border: 0 !important;}
.info td {font-size: .78rem; padding: 9px 6px; border: 0 !important; border-bottom: 1px solid #1a2744 !important; vertical-align: top; color: #dfe7f5;}
.info td:first-child {width: 28%; color: #8ea0c2;}
.foot {font-size: .68rem; color: #6f84ab; margin-top: 2rem; border-top: 1px solid #1a2744; padding-top: .9rem; line-height: 1.6;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SHIELD = (
    f'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01'
    'C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5'
    'a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>'
)


@st.cache_resource(show_spinner="Preparing the models...")
def load_assets():
    """Load the saved models; (re)train from the bundled CSV if missing or unreadable."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        try:
            meta = train()
            sets = joblib.load(MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(f"Model files are missing and automatic training failed: {exc}. Run `python train_model.py` and check the training log.") from exc
        return sets, meta
    try:
        sets = joblib.load(MODEL_PATH)
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if not (isinstance(sets, dict) and "full" in sets and "url_only" in sets
                and "url_only_models" in meta and "tld_lookup" in meta and "features" in meta):
            raise ValueError("model artifact schema is incomplete")
        if list(meta["url_features"]) != list(URL_FEATURES):
            raise ValueError("saved URL feature order does not match the extractor")
        if any(set(meta["features"]) != set(sets["full"][name].feature_names_in_) for name in sets["full"] if hasattr(sets["full"][name], "feature_names_in_")):
            raise ValueError("saved model feature schema does not match metadata")
    except Exception as exc:
        raise RuntimeError(f"Could not load the saved phishing models ({exc}). To rebuild intentionally, run `python train_model.py` from the project folder and review its output; the app will not silently retrain on every loading error.") from exc
    return sets, meta


try:
    model_sets, meta = load_assets()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
models = model_sets["full"]                     # all 44 features (matches the Colab notebook)
model = models[meta["selected_model"]]
m = meta["metrics"]
(tn, fp), (fn, tp) = meta["confusion_matrix"]
ALL_FEATURES = meta["features"]


def phishing_score(clf, row):
    return float(clf.predict_proba(row)[0, list(clf.classes_).index(0)])


def tld_freq(url):
    lk = meta["tld_lookup"]
    host = (urlsplit(normalise_url(url)).hostname or "").lower().strip(".")
    tld = host.rsplit(".", 1)[-1]
    if tld == "com":
        return lk["com"]
    return lk["by_length"].get(str(len(tld)), lk["default"])


# ---------------------------------------------------------------- header
st.markdown(
    f'<div class="hero"><div class="hero-icon">{SHIELD}</div><div>'
    "<h1>Phishing URL Detection Using Machine Learning</h1>"
    "<p>Phishing attacks use fake links to trick people into giving away passwords, card numbers or other "
    f"private data. A {meta['selected_model']} model, trained on labelled URLs from the {meta['dataset']} dataset, "
    "looks at the structure of a web address and the page behind it, and estimates whether it resembles known phishing links.</p>"
    "</div></div><hr class='rule'>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- check a URL
st.markdown(
    '<div class="sec">Check a URL</div>'
    '<div class="sec-cap">Enter a web address. The URL text is analysed and the page\'s HTML is downloaded once to read its structure. '
    "The page is never opened in a browser or run.</div>",
    unsafe_allow_html=True,
)
with st.form("url_form", border=False):
    url_text = st.text_input("URL", placeholder="https://example.com/login", label_visibility="collapsed")
    submitted = st.form_submit_button("Check URL")

if submitted:
    if not url_text.strip():
        st.warning("Please enter a URL first.")
    else:
        try:
            url_feats = extract_features(url_text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            # 1) Try to download the HTML once (never rendered or executed).
            page, fetch_error = None, None
            with st.spinner("Reading the page..."):
                try:
                    page = fetch_page(normalise_url(url_text))
                except FetchError as exc:
                    fetch_error = str(exc)
            full_mode = page is not None
            if full_mode:
                content = extract_content_features(page)
                feats = {**content, **url_feats, "tld_freq": tld_freq(url_text)}
                active, active_names, scores_meta = models, ALL_FEATURES, meta["models"]
                status = (f"Page opened (HTTP {page['status']}, {page['size']/1024:.0f} KB). "
                          f"All {len(ALL_FEATURES)} features were used. Page-feature definitions are reconstructed approximations.")
            else:
                feats = url_feats
                active_names, scores_meta = URL_FEATURES, meta["url_only_models"]
                active = model_sets["url_only"]
                status = (f"{fetch_error} The result below uses the URL text only "
                          f"({len(URL_FEATURES)} features, about {scores_meta[meta['selected_model']]['accuracy']:.0%} accuracy), "
                          "so it is less reliable.")
            missing = [f for f in active_names if f not in feats]
            if missing:
                st.error("Feature extraction did not produce the required inputs: " + ", ".join(missing))
                st.stop()
            row = pd.DataFrame([[feats[f] for f in active_names]], columns=active_names)
            # Explicitly preserve training feature order and numeric types.
            row = row.reindex(columns=active_names).apply(pd.to_numeric, errors="coerce")
            head = active[meta["selected_model"]]
            score = phishing_score(head, row)
            flags = []
            # Independent defense-in-depth check for high-confidence brand typosquats.
            # This is not a substitute for a maintained reputation/threat-intelligence feed.
            brand_finding = inspect_brand_impersonation(url_text)
            is_phish = (score >= 0.5) or (brand_finding is not None)
            if brand_finding is not None:
                # Avoid presenting a contradictory low ML score as the final verdict.
                # Preserve and show the model score separately, but force the verdict
                # to phishing when a near-exact known-brand impersonation is detected.
                cls, title = "bad", "Likely phishing — brand lookalike detected"
                flags.append(brand_finding["reason"])
            else:
                cls, title = ("bad", "Likely phishing") if is_phish else ("ok", "Likely legitimate")
            color = "#ff7b8c" if is_phish else "#4fe0b0"
            if feats["is_ip"]: flags.append("The address uses a raw IP number instead of a domain name.")
            if feats["has_at"]: flags.append("The URL contains an @ symbol, which can hide the real destination.")
            if feats["has_punycode"]: flags.append("The domain uses punycode (xn--), which can disguise look-alike characters.")
            if feats["kw_hits"]: flags.append(f"The URL contains {feats['kw_hits']} sensitive keyword(s) such as login, secure or verify.")
            if feats["n_hyphen"] >= 2: flags.append(f"The domain contains {feats['n_hyphen']} hyphens.")
            if feats["n_digit"] >= 4: flags.append(f"The domain contains {feats['n_digit']} digits.")
            if feats["host_len"] >= 30: flags.append(f"The domain name is long ({feats['host_len']} characters).")
            if full_mode:
                if feats["HasPasswordField"]: flags.append("The page asks for a password.")
                if feats["HasExternalFormSubmit"]: flags.append("A form on the page sends its data to a different website.")
                if feats["NoOfiFrame"] >= 3: flags.append(f"The page embeds {feats['NoOfiFrame']} iframes.")
                if feats["NoOfURLRedirect"] and not feats["NoOfSelfRedirect"]: flags.append("The address redirects to a different website.")
                if feats["NoOfPopup"]: flags.append(f"The page's code opens {feats['NoOfPopup']} popup window(s).")
                if not feats["HasTitle"]: flags.append("The page has no title.")
            rows, n_flag = [], 0
            for name, clf in active.items():
                sc = phishing_score(clf, row)
                flagged = int(clf.predict(row)[0]) == 0
                n_flag += flagged
                c = "#ff7b8c" if flagged else "#4fe0b0"
                pill = f'<span class="pill {"bad" if flagged else "ok"}">{"Phishing" if flagged else "Legitimate"}</span>'
                acc = scores_meta[name]["accuracy"]
                rows.append(
                    f'<tr class="{"hl" if name == meta["selected_model"] else ""}"><td>{name}</td><td>{pill}</td>'
                    f'<td><span class="minibar"><div style="width:{sc*100:.0f}%;background:{c}"></div></span>{sc:.0%}</td>'
                    f'<td>{acc:.1%}</td></tr>'
                )
            model_html = (
                f'<p class="agree"><b>{n_flag} of {len(active)}</b> models flag this URL as phishing.</p>'
                '<table class="mp"><tr><th>Model</th><th>Prediction</th><th>Phishing score</th><th>Test accuracy</th></tr>'
                + "".join(rows) + "</table>"
            )
            flag_html = "<ul class='flags'>" + "".join(f"<li>{f}</li>" for f in flags) + "</ul>" if flags else ""
            status_html = f'<div class="status {"" if full_mode else "warn"}">{status}</div>'
            st.markdown(
                f'<div class="verdict {cls}"><div class="t">{title}</div>'
                f'<div class="s">{meta["selected_model"]} gives this URL a <b>{score:.0%}</b> phishing score. '
                + ("A separate brand-lookalike rule also flagged this domain." if brand_finding else "")
                + ' Treat this as decision support, not proof of safety.</div>'
                f'<div class="bar"><div style="width:{score*100:.0f}%;background:{color}"></div></div>'
                f"{status_html}{flag_html}{model_html}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Features extracted from this URL"):
                st.dataframe(
                    pd.DataFrame({"Feature": active_names, "Value": [round(float(feats[f]), 4) for f in active_names]}),
                    hide_index=True, width="stretch",
                )

# ---------------------------------------------------------------- performance
st.markdown(
    '<div class="sec">Model Performance</div>'
    f'<div class="sec-cap">Measured on {meta["n_test"]:,} URLs the model never saw during training '
    f'({round(meta["test_size"]*100)}% test split, random seed {meta["random_state"]}). Phishing is the positive class, '
    f"so precision and recall describe how well phishing URLs are caught. All {len(ALL_FEATURES)} dataset features are used, the same setup as the Colab notebook.</div>",
    unsafe_allow_html=True,
)


def card(label, value, desc):
    return f'<div class="card"><div class="k">{label}</div><div class="v">{value:.2%}</div><div class="d">{desc}</div></div>'


st.markdown(
    '<div class="cards">'
    + card("Accuracy", m["accuracy"], "Share of all test URLs classified correctly.")
    + card("Precision", m["precision"], "Of the URLs flagged as phishing, the share that really were.")
    + card("Recall", m["recall"], "Of the real phishing URLs, the share the model caught.")
    + card("F1-score", m["f1"], "Balance between precision and recall.")
    + "</div>",
    unsafe_allow_html=True,
)
st.write("")


def cell_color(v, lo, hi):
    t = 0 if hi == lo else (v - lo) / (hi - lo)
    a, b = (10, 20, 40), (77, 141, 255)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


vals = [tn, fp, fn, tp]
lo, hi = min(vals), max(vals)
cell = lambda v: f'<div class="c" style="background:{cell_color(v, lo, hi)}">{v:,}</div>'
cm_html = (
    '<div class="cm-wrap"><div class="cm">'
    '<div class="ylab">Actual class</div>'
    f'<div class="rl">Legitimate</div>{cell(tn)}{cell(fp)}'
    f'<div class="rl">Phishing</div>{cell(fn)}{cell(tp)}'
    '<div></div><div></div><div class="cl">Legitimate</div><div class="cl">Phishing</div>'
    '<div></div><div></div><div class="xlab">Predicted class</div>'
    "</div></div>"
)
read_html = (
    '<ul class="read">'
    f"<li><b>{tp:,}</b> phishing URLs correctly flagged</li>"
    f"<li><b>{tn:,}</b> legitimate URLs correctly passed</li>"
    f"<li><b>{fp:,}</b> legitimate URLs wrongly flagged as phishing (false positives)</li>"
    f"<li><b>{fn:,}</b> phishing URLs missed (false negatives)</li></ul>"
    '<div class="note">These scores come from one dataset. Real-world performance can be lower.</div>'
)
left, right = st.columns([1.55, 1], gap="medium")
with left:
    st.markdown('<div class="sub">Confusion matrix</div>' + cm_html, unsafe_allow_html=True)
with right:
    st.markdown('<div class="sub">How to read it</div>' + read_html, unsafe_allow_html=True)
st.markdown(
    '<div class="cap">Rows are actual outcomes and columns are model predictions. '
    "For this matrix, the displayed order is Legitimate then Phishing (equivalent to label 1 then label 0).</div>",
    unsafe_allow_html=True,
)

cmp_rows = "".join(
    f'<tr class="{"hl" if n == meta["selected_model"] else ""}"><td>{n}</td>'
    + "".join(f"<td>{v[k]:.2%}</td>" for k in ("accuracy", "precision", "recall", "f1", "roc_auc"))
    + "</tr>"
    for n, v in sorted(meta["models"].items(), key=lambda kv: -kv[1]["f1"])
)
st.markdown(
    '<div class="sub" style="margin-top:1.6rem">All models compared</div>'
    '<table class="cmp"><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>ROC-AUC</th></tr>'
    + cmp_rows + "</table>"
    f'<div class="cap">Same {meta["n_test"]:,} test URLs and {len(ALL_FEATURES)} features for every model, identical to the Colab notebook. The highlighted row is the one used for the main verdict.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- project information
st.markdown('<div class="sec">Project Information</div>', unsafe_allow_html=True)
st.markdown(
    '<table class="info">'
    f'<tr><td>Dataset</td><td>{meta["dataset"]} sample: {meta["n_rows"]:,} labelled URLs, roughly half phishing and half legitimate.</td></tr>'
    f'<tr><td>Model</td><td>{meta["selected_model"]} (200 trees) gives the main verdict; {len(models)} models are trained on the same {meta["n_train"]:,} URLs and shown side by side. ROC-AUC of the main model: {m["roc_auc"]:.3f}.</td></tr>'
    f'<tr><td>Inputs</td><td>{len(ALL_FEATURES)} features: {len(URL_FEATURES)} from the text of the domain name (length, dots, hyphens, digits, entropy, vowel ratio, IP address, @ symbol, punycode, keywords), '
    "26 from the page's HTML (size, title, favicon, forms, password field, iframes, images, scripts, internal and external links, redirects) and 1 top-level-domain frequency.</td></tr>"
    f'<tr><td>Fallback</td><td>If the page cannot be downloaded (offline, blocked or not HTML), the app uses {len(URL_FEATURES)} URL-text features instead. '
    f'That version reaches about {meta["url_only_models"][meta["selected_model"]]["accuracy"]:.0%} accuracy and the result says so.</td></tr>'
    "<tr><td>Limits</td><td>The scores above are exact for the dataset. The page features are rebuilt from raw HTML with reconstructed definitions, "
    "so live results are approximate. Many phishing pages are taken down quickly, and some sites block automated requests.</td></tr>"
    "</table>"
    '<div class="foot">Educational project. A prediction is decision support, not a guarantee that a website is safe. '
    "Always check the address and avoid entering credentials on links you did not expect.</div>",
    unsafe_allow_html=True,
)
