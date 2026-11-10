# PhishPulse

Time-aware calibration and selective detection for zero-day phishing URLs. This is the implementation of the plan in `analysis/`.
Everything here is free: open data, open-source libraries, and it runs on a CPU.

Each URL gets one of three outcomes: **PHISHING**, **BENIGN** or **UNCERTAIN / DEFER**. Every result comes with a calibrated probability, a confidence score, an uncertainty score and the SHAP features that drove it.

## Quick start
```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt   # or: python -m venv .venv && pip install -r requirements.txt
.venv/bin/python -m src.ingest            # downloads data (~20 min): PhishTank mirror + Common Crawl index samples
.venv/bin/python -m src.run_experiments   # full study -> reports/results.md, reports/tables, reports/figures, models/
.venv/bin/python -m pytest -q tests       # unit tests (canonicalisation, features, leakage, calibration, selective)
.venv/bin/uvicorn src.api:app --port 8000 # POST /score {"url": "..."}
.venv/bin/streamlit run app/streamlit_app.py
```
The trained models are already in `models/`, so the API and the demo work without retraining.

## Pipeline (`src/`)
| Step | File |
|---|---|
| Data ingest with provenance | `ingest.py` |
| Canonicalise, extract registered domain, deduplicate | `canonicalize.py` |
| Random, chronological and domain-disjoint splits | `split_temporal.py` |
| Lexical and structural features (~50 features, no network calls) | `features_lexical.py` |
| Past-only temporal features (domain/TLD phishing counts over 1, 7 and 30 days; burst; smoothed TLD ratio) | `features_temporal.py` |
| Logistic Regression, Random Forest, XGBoost, TF-IDF+LogReg/SVM, LightGBM, seed ensemble | `train.py` |
| Raw, Platt, isotonic, temperature and time-stratified calibration | `calibrate.py` |
| Confidence/entropy policy, ensemble disagreement, risk–coverage curves, AURC | `selective.py` |
| Metrics, ECE/ACE, bootstrap confidence intervals, plots | `evaluate.py` |
| SHAP summary, local waterfalls, importance drift | `explain.py` |
| FastAPI `/score` (URL-only fast path) | `api.py` |

## Headline results (future test period: 2026-06 to 2026-11)
These come from the run on 2026-11-10. Full tables are in [`reports/results.md`](reports/results.md).

| Question | Finding |
|---|---|
| RQ1: realism gap | LightGBM F1 is **0.953** on a random split, **0.772** on a chronological split and **0.638** on a domain-disjoint chronological split. |
| RQ2: temporal features | PR-AUC goes from 0.947 to **0.966** and F1 from 0.772 to **0.862**. The bootstrap confidence intervals do not overlap. |
| RQ3: calibration | Isotonic calibration lowers ECE from 0.063 to **0.021** and Brier score from 0.081 to 0.073. |
| RQ4: selective prediction | At **78.8%** coverage, the error on automated decisions falls from 10.7% to **3.3%**. Adding ensemble disagreement brings it to 2.4% at 74.2% coverage. |
| RQ5: time-stratified calibration | **Not supported.** Monthly calibrators did worse than one global calibrator: ECE 0.051 versus 0.021. |

Latency for the URL-only fast path is about 1 ms for feature extraction plus about 0.8 ms for the model (median). The model file is 0.75 MB.

## How it works, step by step

1. **Data collection (`ingest.py`).**
   - Phishing URLs come from the verified PhishTank feed. Each URL's timestamp is the time it was reported.
   - Benign URLs are real web pages sampled from Common Crawl. Each URL's timestamp is the time it was crawled.
   - Every row records where it came from (`source`) and when it was downloaded (`collection_ts`).
2. **Cleaning (`canonicalize.py`).**
   - URLs are normalised: lowercase host, `www.` and default ports removed, trailing `/` dropped.
   - Duplicates are removed, keeping the earliest sighting. If the same URL appears with both labels, it counts as phishing (the safer choice).
   - The registered domain (e.g. `example.co.uk`) and the TLD are extracted with an offline public-suffix list.
3. **Splits (`split_temporal.py`).**
   - **Chronological:** the oldest 65% of data trains the model, the next 15% is for validation and calibration, and the newest 20% is the "future" test set.
   - **Random:** a conventional random split, to show how optimistic it is.
   - **Domain-disjoint:** like chronological, but test domains never appear in training.
4. **Features.**
   - *Lexical/structural:* lengths, character ratios, symbol counts and entropy. Also flags for IP-address hosts, punycode, look-alike characters, suspicious words, suspicious TLDs, free-hosting platforms, brand names outside the real domain, redirect parameters, embedded URLs, subdomain count and path depth.
   - *Temporal:* how many phishing reports the same domain or TLD had in the previous 1, 7 and 30 days, a burst ratio, and the TLD's historical phishing rate. These use **only records strictly earlier** than the URL being scored; a unit test checks this.
5. **Models (`train.py`).**
   - Baselines: Logistic Regression, Random Forest, XGBoost, and character TF-IDF with Logistic Regression or a linear SVM.
   - Main model: LightGBM, tuned on the validation window by log loss.
   - A 5-model bootstrap LightGBM ensemble provides a disagreement-based uncertainty signal.
6. **Calibration (`calibrate.py`).** Raw, Platt (sigmoid), isotonic, temperature and time-stratified calibrators, all fitted on the validation window only. Isotonic gave the best results and is used in the final system.
7. **Selective decisions (`selective.py`).** For a calibrated probability `p`, confidence = max(p, 1−p) and uncertainty = binary entropy. The confidence threshold is chosen on validation as the lowest one that keeps error under 2%. Below it, the system answers **UNCERTAIN** so the URL goes to an analyst or sandbox. Optionally, it also defers when the ensemble models disagree.
8. **Evaluation (`evaluate.py`).**
   - Classification: PR-AUC, ROC-AUC, F1, precision, recall, FPR, FNR, MCC.
   - Calibration: ECE, ACE, Brier score, log loss.
   - Selective: coverage, selective risk, AURC.
   - Also: 95% bootstrap confidence intervals, reliability diagrams and risk–coverage curves.
9. **Explainability (`explain.py`).**
   - A global SHAP summary.
   - Local waterfall plots for one PHISHING, one BENIGN and one UNCERTAIN example.
   - A comparison of feature importance between the older training period and the future test period.
10. **Deployment.**
    - `api.py` (FastAPI `/score`) and `app/streamlit_app.py` serve the **URL-only** model. It never visits the URL.
    - It returns the decision, calibrated probability, confidence, uncertainty, the reason for deferring (if any), the top SHAP evidence and the model version.

## Repository layout
```
analysis/   original documentation (plan, models, reports, presentations)
configs/    experiment.yaml - every tunable setting
src/        pipeline code (see the table above)
tests/      unit tests
models/     phishpulse_urlonly.joblib (deployed), phishpulse_full.joblib (with temporal features)
reports/    results.md, tables/*.csv, figures/*.png, shap_examples.json (URLs defanged), run_meta.json
app/        Streamlit demo
```

## What could not be achieved
- **WHOIS/RDAP domain age and the "enriched" model.** No free source provides historical registration data as of each URL's first sighting, and live lookups would leak future information. Temporal history features take their place.
- **PhishTank's official download and OpenPhish.** PhishTank's site is rate-limited (HTTP 429), so a public GitHub mirror of the same feed was used. OpenPhish's free feed has no timestamps, so it can't support a time-based study.
- **Benign timestamps are crawl times, not true first-seen times.** Because of this, hour and day-of-week features are excluded from the main model; they would pick up the data source rather than phishing behaviour.
- **Time-stratified calibration (RQ5) was not supported.** Monthly calibrators did worse than one global calibrator on future data.
- **Optional extras were not built:** Character CNN, small Transformer, LIME/anchors and rolling-origin evaluation.
- **The deployed API is URL-only.** Temporal features need a live phishing-feed history, which the API doesn't maintain. The full model is saved for offline use.
- **There was no separate tuning window.** The validation window is used both for tuning and for calibration and thresholds, because there wasn't enough data for another window. The test set was used exactly once.
- **Results change from run to run** because the PhishTank feed changes daily, and it only lists URLs that are currently online.
- **Raw data is not published.** It contains live phishing links (an ethics rule in the plan). `python -m src.ingest` rebuilds it.

More detail on each point is in [`changes.md`](changes.md).
