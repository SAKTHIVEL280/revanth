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

See [`changes.md`](changes.md) for where this implementation differs from the documentation, and why.
