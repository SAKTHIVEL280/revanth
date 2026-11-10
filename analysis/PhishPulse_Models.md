# Models for PhishPulse

This document lists every model type used in the project, its role, and why it is included.

---

## Primary production model

### LightGBM (binary classification)

- **Type:** Gradient-boosted decision tree ensemble (tabular features)
- **Role:** Core phishing/benign classifier
- **Inputs:** Lexical, structural, domain/registration, and time-aware features; optional sparse character n-grams
- **Optimization target:** Binary cross-entropy / log loss (for better probability calibration)
- **Why LightGBM:**
  - Strong accuracy on tabular data
  - Very fast inference (sub-millisecond per URL)
  - Built-in feature importance and categorical support
  - Works well with SHAP TreeExplainer

This is the model that will be shipped in the final PhishPulse system.

---

## Baseline models (for comparison)

### 1. Logistic Regression

- **Type:** Linear model with L2 regularization
- **Role:** Simple, interpretable URL-only baseline
- **Inputs:** Numeric lexical/structural features; optionally TF-IDF character n-grams
- **Why:** Shows how much a linear model can achieve with engineered features; useful for interpretability and as a lower bound.

### 2. Random Forest

- **Type:** Bagged tree ensemble
- **Role:** Classic non-linear baseline
- **Inputs:** Same features as LightGBM
- **Why:** Commonly used in prior phishing studies; helps show the benefit of boosting and calibration.

### 3. XGBoost

- **Type:** Gradient-boosted decision trees
- **Role:** Strong boosting baseline
- **Inputs:** Same as LightGBM
- **Why:** Another state-of-the-art GBDT implementation; useful to confirm that improvements come from the framework, not just one library.

### 4. Character TF-IDF + Linear SVM / Logistic Regression

- **Type:** Sparse linear model on character n-grams
- **Role:** Text-based URL baseline
- **Inputs:** Character 3–5-grams from host/path/query
- **Why:** Strong inexpensive baseline that captures URL patterns without heavy feature engineering.

### 5. Character CNN (optional, if time allows)

- **Type:** 1D convolutional neural network over character sequences
- **Role:** Deep-learning URL-only baseline
- **Inputs:** Raw URL strings (character-level)
- **Why:** Represents modern deep approaches; used to contrast with lightweight GBDT in terms of latency, calibration, and interpretability.

### 6. Small Transformer / BERT-like model (optional, advanced)

- **Type:** Transformer encoder on URL tokens/characters
- **Role:** Upper-bound deep model for accuracy (not necessarily for deployment)
- **Inputs:** Tokenized URL sequences
- **Why:** Provides a strong but heavy baseline to highlight PhishPulse’s speed and calibration advantages.

---

## Calibration models (probability correction)

These are not standalone classifiers; they wrap the base classifier to improve probability quality.

### 1. Sigmoid (Platt) calibration

- **Type:** Logistic regression on model scores
- **Role:** Simple parametric calibration
- **Why:** Fast, robust when validation data are limited.

### 2. Isotonic regression

- **Type:** Non-parametric monotonic mapping
- **Role:** Flexible calibration when enough validation data exist
- **Why:** Often more accurate than sigmoid if data are sufficient; used as main calibrator in chronological validation.

### 3. Temperature scaling (if logits available)

- **Type:** Single scalar rescaling of logits
- **Role:** Simple calibration for deep models or models with logit outputs
- **Why:** Standard in deep learning; can be compared with isotonic/sigmoid for GBDT.

### 4. Time-stratified calibrators

- **Type:** Separate sigmoid/isotonic calibrators per time bucket (e.g., month or quarter)
- **Role:** Maintain calibration quality as attack behavior drifts
- **Why:** Core novelty of PhishPulse: calibration that adapts to time.

---

## Selective prediction / abstention logic

This is not a separate ML model but a decision policy.

### Confidence/entropy-based selector

- **Inputs:** Calibrated probability `p`, derived confidence and entropy
- **Policy:**
  - If `confidence >= threshold` and `entropy <= limit`: output PHISHING or BENIGN
  - Else: output UNCERTAIN / DEFER
- **Tuning:** Threshold chosen on chronological validation set to achieve target coverage and risk.

Optional extension:

### Ensemble disagreement selector

- **Models:** Multiple LightGBM instances with different seeds/bootstrap samples
- **Uncertainty:** Variance of calibrated predictions across ensemble
- **Policy:** Defer when ensemble variance is high, even if mean probability is extreme.

---

## Explainability components

### SHAP TreeExplainer for LightGBM

- **Type:** Shapley-value explainer specialized for tree ensembles
- **Role:** Explain why each automated decision was made
- **Outputs:**
  - Global feature importance
  - Per-URL SHAP values and waterfall plots

### Optional: LIME or anchor explanations

- **Role:** Alternative local explanations for comparison or robustness checks
- **Why:** Validate that SHAP is not the only explanation method showing similar evidence.

---

## Model inventory summary

| Model / component | Type | Primary role |
|---|---|---|
| LightGBM | GBDT | Main phishing classifier |
| Logistic Regression | Linear | Interpretable baseline |
| Random Forest | Bagged trees | Classic baseline |
| XGBoost | GBDT | Boosting baseline |
| TF-IDF + Linear SVM/LogReg | Sparse linear | Text-only baseline |
| Character CNN | Deep CNN | Deep-learning baseline (optional) |
| Small Transformer | Deep transformer | Strong but heavy baseline (optional) |
| Sigmoid calibrator | Logistic on scores | Probability calibration |
| Isotonic calibrator | Non-parametric mapping | Probability calibration |
| Temperature scaler | Scalar on logits | Calibration for deep models |
| Time-stratified calibrators | Multiple calibrators | Time-aware calibration |
| Confidence/entropy selector | Decision policy | Selective prediction / abstention |
| Ensemble disagreement | Variance across models | Optional uncertainty signal |
| SHAP TreeExplainer | Explainability | Per-decision explanations |

In the final paper and code, the **production PhishPulse system** will be:

- **LightGBM + time-aware features + isotonic (or sigmoid) calibration + confidence/entropy-based selective policy + SHAP explanations**, with optional time-stratified calibration and optional ensemble-based uncertainty.
