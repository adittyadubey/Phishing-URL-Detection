# Phishing URL Detection — Project Bundle

This folder contains the improved Google Colab/Jupyter notebook, its dataset, a local model-training script, a Streamlit interface, and dependency instructions.

## Files

- `Phishing_URL_Detection_Improved.ipynb` — EDA, cleaning, train/test split, model comparison and evaluation.
- `PhiUSIIL_10000_rows_clean.csv` — included engineered-feature dataset (10,000 rows, 45 columns in the supplied file).
- `phishing_features.py` — turns raw URL text into the 17 URL-text features (no network access).
- `page_features.py` — safely downloads a page's HTML and extracts the 26 page-content features.
- `train_model.py` — trains the four classifiers (Logistic Regression, SVM, Decision Tree, Random Forest) on all 44 features (identical to the Colab setup) plus a URL-only fallback set, and saves models, metrics and confusion matrices.
- `app.py` — Streamlit interface: paste a URL, see the verdict, model metrics, confusion matrix and project information. If `models/` is empty or incompatible with your scikit-learn version, the app retrains automatically on first launch.
- `.streamlit/config.toml` — dark theme.
- `requirements.txt` — Python dependencies.
- `models/` — trained artifacts are created here after running `train_model.py`.
- `outputs/` — comparison metrics and report are created here after training.

## Run locally on Windows

Open Command Prompt or PowerShell in this folder and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If `py` is not recognized, install Python from python.org and enable **Add Python to PATH**, then reopen the terminal.

## Run the notebook in Google Colab

1. Upload `Phishing_URL_Detection_Improved.ipynb` to Colab.
2. Upload `PhiUSIIL_10000_rows_clean.csv` when the notebook requests it.
3. Run notebook cells from top to bottom.

## How the app works

- Paste a URL. The app downloads the page's HTML once (never rendered, no JavaScript run; 2 MB cap, 8 s timeout, private/internal addresses refused) and builds all 44 features the notebook uses. Metrics on the page therefore match the Colab notebook exactly.
- If the page cannot be downloaded (offline, blocked, not HTML) the app falls back to the 17 URL-text features and says so; that version is about 75% accurate.
- `page_features.py` rebuilds the 26 page-content features from raw HTML. The original extraction code is not included, so these definitions are reconstructed and live results are approximate. `tld_freq` uses an exact value for `.com` and a typical value by TLD length for others, because the CSV has no TLD lookup table.

## Important limitations

- Dataset scores are exact; live predictions on arbitrary pages are approximate (see above).
- The notebook and scripts assume `label = 0` means phishing and `label = 1` means legitimate. Verify this mapping against the dataset documentation before submitting.
- Fetching a page contacts that site from your computer. Do not use the checker on links you are not comfortable contacting.
- The project is educational and does not guarantee that a website is safe.

## Validation and security notes (updated)

- **Label convention used by this project:** `0 = phishing`, `1 = legitimate`. Phishing is treated as the positive class for precision, recall and F1. Confusion matrices use rows and columns in `[legitimate, phishing]` order. The training script validates that labels are binary and records this mapping in model metadata.
- **Training protocol:** a stratified 70/30 split is made first. Hyperparameter search uses stratified 2-fold cross-validation on the training split only; the held-out 30% is used only for final metrics. The same four model families remain in place. Small search grids are used to keep retraining practical on a student laptop.
- **Duplicate/domain limitation:** exact duplicate rows are removed before splitting. The bundled CSV contains engineered features, not raw URL/domain identifiers, so domain-grouped splitting and reliable detection of related domains across splits are not possible with this file alone. Do not describe this as a domain-independent evaluation.
- **`tld_freq` limitation:** the source CSV already contains `tld_freq`, but does not include the original TLD key or feature-generation code. The runtime lookup is derived from training rows only, but the precomputed CSV feature may itself have been calculated globally before this project received the data. That potential upstream leakage cannot be conclusively ruled out from this CSV. For a leakage-free final study, obtain raw URLs and rebuild `tld_freq` from training-fold domains only.
- **Live features:** 17 URL features are extracted from the URL and 26 page features are reconstructed from fetched HTML. The original PhiUSIIL extraction implementation is not included, so the live page-feature definitions are approximations, not guaranteed exact matches to the dataset's definitions. Missing numeric values are handled by each model pipeline's median imputer.
- **Fetch security:** TLS certificate verification stays enabled; redirects are followed manually with a five-hop cap; each target is checked for public DNS addresses; proxy environment settings are ignored; connect/read timeouts and a 2 MB body limit are applied. DNS validation before a separate HTTP connection cannot completely eliminate DNS-rebinding/TOCTOU risk. Do not expose this educational app as a public URL-fetching service without stronger egress controls (for example, a restricted outbound proxy/firewall).
- **Two modes:** the full-feature model and URL-only fallback have separate metrics in `outputs/model_comparison.csv`. They are measured on the same held-out dataset split, not on a separate external set of newly collected URLs. Do not describe these results as independent live-URL validation.

## Run and test

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python train_model.py
streamlit run app.py
```

Model loading errors are now shown with a useful message instead of silently triggering retraining for every exception. Run `python train_model.py` intentionally when you want to regenerate model artifacts and metadata.


### Live-page response validation

The fetcher only sends a normal HTTP 200 HTML response to the full-feature model. HTTP 202 and other non-200 responses, common anti-bot/challenge pages, and very small HTML responses are treated as inconclusive and use the URL-only model instead. This avoids interpreting a bot-check page as the actual site's content. The URL-only model can still make mistakes; validate it against independent labelled URLs before relying on it.


## Brand lookalike protection

The live verdict includes a deterministic, defense-in-depth check for near-exact typosquatting of a curated list of frequently impersonated brands (including digit substitutions such as `micros0ft.com`). This can catch some false negatives missed by the ML model. It is not universal protection: new brands, novel lookalikes, compromised legitimate sites, and changing threat patterns require a maintained domain-reputation/threat-intelligence feed and continuous evaluation. Do not treat a domain as safe merely because this rule did not match.


## Desktop UI

The Streamlit interface uses wide layout with a responsive desktop-first content width (up to 1720px), larger typography and controls on large screens, and mobile breakpoints retained for smaller windows. Run with `python -m streamlit run app.py`.


## Quick start (working bundle)

The dataset is included under the exact filename expected by `train_model.py`.
The supplied notebook is installed as `Phishing_URL_Detection_Improved.ipynb`.
No trained model binaries are bundled; the first app launch trains models and writes
artifacts into `models/` and reports into `outputs/`. To train deliberately first:

```powershell
python -m pip install -r requirements.txt
python train_model.py
python -m pytest -q
python -m streamlit run app.py
```

For deployment, ensure the model artifacts are present in the repository or use a
deployment build step to generate them. First-time training can take several minutes.
