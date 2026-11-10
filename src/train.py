"""Model factory: baselines + LightGBM (primary) + seed/bootstrap ensemble."""
import numpy as np, lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def lgbm(params, seed=42):
    return lgb.LGBMClassifier(objective="binary", random_state=seed, verbose=-1, n_jobs=-1, **params)


def baselines(seed=42):
    """Name -> (estimator, input kind). 'num' = engineered features, 'text' = canonical URL string."""
    return {
        "LogisticRegression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)), "num"),
        "RandomForest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed), "num"),
        "XGBoost": (xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.8,
                                      colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=seed), "num"),
        "TFIDF-char+LogReg": (make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=300000,
                                                            sublinear_tf=True), LogisticRegression(max_iter=2000, C=4.0)), "text"),
        # LinearSVC has no probabilities; internal sigmoid on training folds only (train period) is used for scores
        "TFIDF-char+LinearSVM": (make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=300000,
                                                               sublinear_tf=True), CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)), "text"),
    }


class Ensemble:
    """Bootstrap + seed LightGBM ensemble for disagreement-based uncertainty."""
    def __init__(self, params, seeds):
        self.params, self.seeds = params, seeds

    def fit(self, X, y):
        self.models = []
        for s in self.seeds:
            i = np.random.default_rng(s).integers(0, len(y), len(y))
            self.models.append(lgbm(self.params, s).fit(X.iloc[i], y[i]))
        return self

    def predict_all(self, X):
        return np.stack([m.predict_proba(X)[:, 1] for m in self.models])
