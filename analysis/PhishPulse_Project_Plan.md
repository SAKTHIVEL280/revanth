# PhishPulse: Time-Aware Uncertainty Calibration and Selective Detection for Zero-Day Phishing URLs

## Project overview

**PhishPulse** is a trustworthy phishing-URL detection system designed for real deployment conditions. Instead of only producing a phishing/benign label, it accounts for changing attacker behavior over time, calibrates its confidence estimates, and abstains on URLs for which a reliable automated decision cannot be made.

**Final objective:** build a lightweight, interpretable detector that returns one of three outcomes:

- **PHISHING** — high-confidence, calibrated phishing prediction
- **BENIGN** — high-confidence, calibrated benign prediction
- **UNCERTAIN / DEFER** — low-confidence or out-of-distribution URL routed to an analyst or a deeper inspection stage

## Research problem

Most phishing URL studies use random train/test splits. This can mix closely related campaigns, domains, URL templates, and time periods across training and testing data, producing overly optimistic results. In reality, phishing infrastructure and URL obfuscation change continuously; a model trained on old URLs can be overconfident on new campaigns.

The project addresses three connected questions:

1. How much does phishing URL performance degrade under a chronological test split compared with a random split?
2. Can temporal features and time-aware calibration make predicted phishing probabilities more reliable on newer data?
3. Can selective prediction reduce the error rate of automated decisions by abstaining on uncertain or likely zero-day URLs?

## Research gap and novelty

Existing work commonly uses lexical/structural features, standard ML or deep learning classifiers, and accuracy/F1 metrics. Some work considers temporal indicators, calibration, explainability, or uncertainty separately. PhishPulse combines them in one reproducible framework:

1. **Chronological zero-day evaluation:** training only on URLs observed before the validation and test periods.
2. **Time-aware feature layer:** domain-age and first-seen features, plus carefully leakage-controlled historical aggregation features.
3. **Time-stratified probability calibration:** compare global calibration with calibration models trained by time period.
4. **Selective prediction:** produce a defer decision for high-uncertainty samples and measure coverage-versus-risk.
5. **Interpretable lightweight deployment:** use LightGBM and SHAP to expose the evidence behind automated predictions.

Important wording for the paper: claim this as an **integrated framework and empirical study**, not as an absolute “first-ever” claim unless a final systematic literature review confirms it.

## Research questions and hypotheses

| ID | Question | Hypothesis |
|---|---|---|
| RQ1 | Does chronological testing expose lower generalization than random splitting? | Random-split performance will be materially higher than future-period performance. |
| RQ2 | Do temporal features improve future-period phishing detection? | Adding temporal features will improve AUC-PR/F1 or reduce error on the chronological test period. |
| RQ3 | Does calibration improve probability reliability under temporal shift? | Temperature/isotonic calibration will reduce ECE and Brier score versus raw model probabilities. |
| RQ4 | Does selective prediction reduce decision risk? | At a controlled coverage level, error among accepted predictions will be lower than full-coverage error. |
| RQ5 | Does time-stratified calibration outperform a single global calibrator? | Time-stratified calibration will have lower ECE on recent samples when sufficient validation data are available. |

## System architecture

```text
Raw URL + observed timestamp + optional passive DNS/WHOIS metadata
                |
                v
Data cleaning, canonicalization, deduplication, chronological split
                |
                v
Feature extraction
  lexical | structural | domain/registration | temporal/history
                |
                v
Base classifier (LightGBM)
                |
                v
Probability calibrator (global or time-stratified)
                |
                v
Uncertainty and selective-decision module
      |                          |
      v                          v
 PHISHING / BENIGN          UNCERTAIN / DEFER
      |
      v
SHAP explanation for accepted automated decisions
```

## Data plan

### Recommended sources

- **PhishTank:** phishing URLs with submission/verification timestamps where available
- **OpenPhish:** phishing feed; retain collection time and source provenance
- **Tranco or Cisco Umbrella top domains:** candidate benign domains; crawl only within ethical and legal constraints
- **Common Crawl:** optional benign URL source with crawl timestamps
- **UCI / Kaggle phishing datasets:** use mainly for baseline replication, not as the sole source for the temporal study

### Required fields

| Field | Purpose |
|---|---|
| `url_raw` | Original observed URL |
| `url_canonical` | Normalized URL used for deduplication |
| `label` | 1 phishing, 0 benign |
| `first_seen_ts` | Primary chronology field |
| `source` | Dataset/feed provenance |
| `domain` / `registered_domain` | Grouping and leakage control |
| `collection_ts` | Collection/provenance audit |
| `whois_created_ts` (optional) | Domain-age calculation |

### Ethical and reproducibility rules

- Never visit suspicious URLs in an ordinary browser; use passive data or an isolated sandbox.
- Do not publish live malicious links in a public demo; hash, defang, or redact them.
- Record the source, query date, label policy, and timestamp definition for every sample.
- Use only metadata available at or before prediction time. Do not use future labels, blacklist status obtained later, or future aggregates.

## Chronological evaluation protocol

### Recommended split

Use the actual timestamp distribution after collection rather than hard-coded years.

- **Training period:** earliest 60–70% of time-ordered data
- **Calibration/validation period:** next 10–20%
- **Future test period:** final 20–25%

Example only:

```text
Train:       2023-01 to 2024-06
Validation:  2024-07 to 2024-09
Future test: 2024-10 to 2025-03
```

### Leakage controls

- Deduplicate canonical URLs before splitting.
- Keep identical registered domains in one split when feasible, or report both URL-level and domain-disjoint temporal experiments.
- Build rolling counts using only samples with timestamps earlier than the current sample.
- Fit vectorizers, scalers, feature encoders, calibrators, and hyperparameters using training/validation data only.
- Use the final future test set exactly once for final reporting.

### Experimental settings

1. **Random split baseline:** establishes how optimistic conventional evaluation can be.
2. **Chronological split:** main realistic deployment evaluation.
3. **Domain-disjoint chronological split:** stronger robustness test; domains seen during training do not occur in test.
4. **Optional rolling-origin evaluation:** train through time T, test at T+1, then move forward; reports drift over several windows.

## Feature engineering

### A. Lexical features (URL string only)

- Total URL, host, path, query, and fragment length
- Digit ratio, letter ratio, punctuation ratio, uppercase ratio
- Number of dots, hyphens, underscores, slashes, equals signs, ampersands, percent encodings, and `@` symbols
- Shannon entropy of the URL, host, and path
- Number/length of query parameters
- Character repetition and maximum run length
- Suspicious-token indicators (for example: login, verify, secure, account, update)
- Presence of an IP address as host
- TLD category and suspicious-TLD flags
- Unicode/punycode and homoglyph-risk features
- Character n-gram features for host/path using TF-IDF or hashing vectorization

### B. Structural features

- Number of subdomains and host labels
- Path depth
- URL scheme (`http`/`https`)
- Explicit port presence
- Redirect-like parameters and embedded URL count
- Hostname/registered-domain length ratio
- Difference between display-like brand tokens and registered domain (only using a documented, fixed brand lexicon)

### C. Domain and registration metadata

These are optional because WHOIS can be incomplete, rate-limited, privacy-protected, or unavailable at inference.

- Domain age at observed time
- Days until expiry when historically available
- Registrar category
- DNS TTL, ASN, nameserver characteristics, or certificate metadata from passive sources

Create two deployment variants:

- **URL-only model:** no network or WHOIS dependency; fast and reproducible.
- **Enriched model:** adds passive domain metadata; may improve accuracy but can have lookup delay and availability bias.

### D. Time-aware features

- Age of domain at URL first-seen time
- Hour of day and day of week for first observation (cyclic encoding)
- Days/month since the start of dataset collection (for drift diagnostics, not as a shortcut)
- Historical count of phishing URLs from the same registered domain, registrar, ASN, or TLD in rolling 1-day, 7-day, and 30-day windows
- Historical phishing ratio for an entity, smoothed and computed strictly from earlier training observations
- Burst features: count/change rate compared with prior window

**Critical rule:** every rolling feature for sample `i` may use only records observed before `first_seen_ts(i)`.

## Model plan

### Primary production model

**LightGBM classifier** with engineered numeric/categorical features and sparse character n-gram representation where appropriate. Optimize log loss or binary cross-entropy, not only accuracy, because calibrated probabilities are needed downstream.

### Baselines

- Logistic Regression: interpretable linear URL-only baseline
- Random Forest: common tree ensemble baseline
- XGBoost: strong gradient-boosting baseline
- LightGBM (raw): primary model before calibration/selective decision
- Character TF-IDF + Linear SVM or Logistic Regression: strong inexpensive URL-text baseline
- Character CNN or small Transformer: optional deep-learning comparison, only if compute/time permits

### Ablation models

- LightGBM using lexical/structural features only
- LightGBM + domain metadata
- LightGBM + temporal features
- LightGBM + calibration
- LightGBM + calibration + selective prediction (full PhishPulse)

## Calibration methodology

### Goal

A probability of 0.80 should correspond approximately to an 80% observed phishing frequency among similarly scored URLs. Calibration must be assessed separately from discrimination (AUC/F1).

### Methods to compare

1. **No calibration:** raw LightGBM probabilities
2. **Platt scaling / logistic calibration:** simple baseline
3. **Isotonic regression:** flexible non-parametric calibrator; requires adequate validation data
4. **Temperature scaling:** primarily designed for logits; can be used when the base model exposes a score/logit representation
5. **Time-stratified calibration:** fit one calibrator per time bucket if each bucket has sufficient validation examples; otherwise use a global calibrator

### Calibration metrics

- Expected Calibration Error (ECE), reported with stated binning strategy
- Adaptive Calibration Error (optional)
- Brier score
- Negative log likelihood / log loss
- Reliability diagram

### Recommended implementation choice

Start with **CalibratedClassifierCV** using sigmoid and isotonic methods on a dedicated chronological validation window. Then evaluate a time-stratified variant such as separate monthly/quarterly calibrators with a fallback global calibrator for low-data windows.

## Uncertainty and selective prediction

### Uncertainty score

For a calibrated phishing probability `p`, calculate:

```text
confidence = max(p, 1 - p)
entropy = -p*log(p) - (1-p)*log(1-p)
```

High entropy / low confidence means the system should be cautious. A useful extension is **ensemble disagreement**: train several LightGBM models using different seeds or bootstrap samples and measure variance of their calibrated predictions.

### Selective policy

```text
If confidence >= threshold AND uncertainty <= uncertainty_limit:
    output PHISHING or BENIGN
Else:
    output UNCERTAIN / DEFER
```

The threshold must be tuned on the chronological validation set, not selected after inspecting test performance.

### Deferral handling

- Send deferred URLs to a human analyst, browser-isolation environment, reputation lookup, or more expensive multimodal classifier.
- Treat `UNCERTAIN` as a security action, not a classification mistake by itself.
- For phishing safety, use asymmetric costs: a missed phishing URL generally costs more than a false alert.

### Selective metrics

- **Coverage:** fraction of samples receiving an automated label
- **Selective risk:** error rate only among automated labels
- **Risk–coverage curve:** selective risk at many coverage levels
- **AURC:** area under the risk–coverage curve
- **Phishing recall at fixed coverage:** important for security operations
- **Deferred phishing proportion:** percentage of phishing URLs correctly routed for deeper inspection

## Explainability

Use SHAP TreeExplainer for the LightGBM component.

### Outputs

- Global SHAP summary plot: top features across future test data
- Per-prediction SHAP waterfall/bar chart: why an accepted URL was labeled phishing or benign
- Drift analysis: compare feature importance between older and future time periods
- Defer explanation: show the factors that made an input uncertain, while clearly distinguishing explanation of model score from a causal explanation

### Analyst-facing response example

```json
{
  "decision": "PHISHING",
  "calibrated_probability": 0.94,
  "confidence": 0.94,
  "uncertainty": 0.23,
  "top_evidence": [
    "domain_age_days: 1",
    "subdomain_depth: 4",
    "url_entropy: high",
    "punycode_present: true"
  ]
}
```

## Training workflow

1. Ingest and timestamp datasets; keep provenance.
2. Canonicalize and deduplicate URLs.
3. Create chronological train, calibration/validation, and future-test partitions.
4. Build strictly past-only temporal aggregates.
5. Extract URL-only, structural, domain, and temporal features.
6. Train baseline models on training data.
7. Select primary LightGBM hyperparameters with time-respecting validation.
8. Fit probability calibrators using validation data only.
9. Tune selective threshold for target coverage/risk on validation data.
10. Lock the final pipeline and evaluate once on the future test period.
11. Generate performance, calibration, risk–coverage, latency, and SHAP artifacts.
12. Run ablations and statistical confidence intervals (bootstrap confidence intervals recommended).

## Evaluation plan

### Classification metrics

- Precision, recall, F1 score
- False positive rate (FPR)
- False negative rate (FNR)
- ROC-AUC
- PR-AUC (especially useful for imbalance)
- Balanced accuracy / MCC (optional)

### Reliability metrics

- ECE
- Brier score
- Log loss
- Reliability plots

### Selective metrics

- Coverage
- Selective risk
- AURC
- Risk–coverage curves
- Class-wise recall after deferral policy

### Efficiency metrics

- Median and p95 inference latency per URL
- Feature extraction latency reported separately from model latency
- CPU, RAM, model size
- Availability rate for WHOIS/passive enrichment features

### Required ablation table

| Variant | Temporal features | Calibration | Selective abstention | SHAP | Expected role |
|---|---:|---:|---:|---:|---|
| Lexical baseline | No | No | No | Optional | Lower bound |
| Raw LightGBM | Optional | No | No | Yes | Main uncalibrated comparator |
| + Temporal features | Yes | No | No | Yes | Test RQ2 |
| + Global calibration | Yes | Yes | No | Yes | Test RQ3 |
| + Time-stratified calibration | Yes | Yes | No | Yes | Test RQ5 |
| Full PhishPulse | Yes | Yes | Yes | Yes | Final system |

## Suggested project structure

```text
phishpulse/
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/                 # never commit sensitive/live feeds
│   ├── processed/
│   └── external/
├── src/
│   ├── ingest.py
│   ├── canonicalize.py
│   ├── split_temporal.py
│   ├── features_lexical.py
│   ├── features_domain.py
│   ├── features_temporal.py
│   ├── train.py
│   ├── calibrate.py
│   ├── selective.py
│   ├── evaluate.py
│   ├── explain.py
│   └── api.py
├── notebooks/
├── tests/
├── models/
├── reports/
│   ├── figures/
│   └── tables/
└── app/
    └── streamlit_or_fastapi/
```

## Twelve-week implementation plan

| Week | Focus | Concrete deliverable |
|---|---|---|
| 1 | Scope and environment | Repository, environment, data schema, ethics/safety note |
| 2 | Data ingestion | Timestamped data table with provenance and labels |
| 3 | Cleaning and split | Canonicalization, deduplication, chronological split report |
| 4 | URL-only features | Lexical/structural extractor with unit tests |
| 5 | Baselines | Logistic Regression, RF, XGBoost/LightGBM benchmark |
| 6 | Temporal features | Leakage-safe rolling aggregates and domain-age pipeline |
| 7 | Main model | Tuned LightGBM plus first ablation results |
| 8 | Calibration | Global calibration, ECE/Brier/reliability diagrams |
| 9 | Time-stratified calibration | Bucketed calibration and comparison experiment |
| 10 | Selective detection | Threshold policy, risk–coverage curves, AURC |
| 11 | SHAP and latency | Explanation case studies, latency/memory profiling |
| 12 | Final study | Locked test evaluation, tables/figures, paper and demo |

## Minimum viable project (MVP)

Build this first:

- URL-only lexical/structural features
- Chronological train/validation/test split
- LightGBM classifier
- Isotonic and sigmoid calibration comparison
- Confidence threshold for `UNCERTAIN`
- ECE, Brier, PR-AUC, coverage, and selective risk
- SHAP summary plus three local examples

Then add WHOIS/passive domain features and rolling historical features only after the MVP is correct and leakage-free.

## Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Timestamp quality | Incorrect time ordering invalidates zero-day claims | Preserve raw source timestamps; document each timestamp meaning; exclude ambiguous records |
| Data leakage | Future reputation/aggregation can inflate results | Build features causally, fit all transforms on earlier data only, audit pipeline |
| Benign-data bias | Top-domain lists do not represent normal user traffic | Use multiple benign sources and report source-specific results |
| WHOIS incompleteness | Not always available at real-time inference | Maintain URL-only fallback model; report enriched-model coverage |
| Class imbalance | Accuracy can mislead | Report PR-AUC, class-wise recall, FPR/FNR, cost-aware thresholds |
| Repeated campaigns | Random URL split may memorize patterns | Use domain-disjoint chronological experiment |
| Overclaiming novelty | Similar components may exist separately | Position contribution as a unified, reproducible evaluation framework and validate with thorough literature review |

## Expected paper outputs

1. Dataset-statistics table: source, time range, class count, metadata availability.
2. Random versus chronological split table: demonstrates realism gap.
3. Main classification table: PR-AUC, ROC-AUC, F1, phishing recall, FPR.
4. Calibration table: ECE, Brier score, log loss for raw/global/time-stratified calibration.
5. Selective-detection table: coverage, selective risk, phishing recall, deferred phishing fraction.
6. Ablation table: contribution of temporal features, calibration, and abstention.
7. Reliability diagram: raw vs calibrated probabilities.
8. Risk–coverage plot: LightGBM and baselines.
9. SHAP summary plot and local explanation case studies.
10. Latency/resource table: URL-only and enriched versions.

## Deployment concept

A FastAPI service can expose `/score`:

```text
Input:  URL and optional observation timestamp
Output: decision, calibrated probability, confidence, uncertainty,
        defer reason, top SHAP factors, model/calibration version
```

Use two-speed processing:

- **Fast path:** URL-only features + calibrated LightGBM for immediate screening.
- **Deferred/enriched path:** passive DNS/WHOIS, sandbox or analyst review for uncertain cases.

## Final contribution statement

PhishPulse is an interpretable, lightweight framework for phishing URL detection under realistic temporal drift. It evaluates future-period generalization, calibrates model confidence, and adopts a selective decision policy so the system can defer uncertain likely zero-day URLs rather than making unsafe forced predictions.
