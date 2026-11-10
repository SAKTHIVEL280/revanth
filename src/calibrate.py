"""Probability calibrators fitted on the chronological validation window only."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


class Identity:
    def fit(self, p, y): return self
    def predict(self, p): return np.asarray(p)


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6).fit(logit(p)[:, None], y); return self
    def predict(self, p): return self.m.predict_proba(logit(p)[:, None])[:, 1]


class Isotonic:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y); return self
    def predict(self, p): return self.m.predict(p)


class Temperature:
    """Single scalar T on the booster's raw logit (LightGBM raw_score = log-odds)."""
    def fit(self, p, y):
        z = logit(p)
        nll = lambda T: -np.mean(y * np.log(1 / (1 + np.exp(-z / T)) + EPS) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-z / T)) + EPS))
        self.T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x; return self
    def predict(self, p): return 1 / (1 + np.exp(-logit(p) / self.T))


class TimeStratified:
    """One calibrator per validation time bucket. Future samples (test) fall after every bucket, so they use the
    most recent bucket with >= min_n samples; buckets below min_n (and unseen buckets) fall back to the global one."""
    def __init__(self, base=Isotonic, freq="M", min_n=300):
        self.base, self.freq, self.min_n = base, freq, min_n

    def fit(self, p, y, ts):
        self.glob = self.base().fit(p, y)
        b = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_convert(None).dt.to_period(self.freq).to_numpy()
        self.cal = {k: self.base().fit(p[b == k], y[b == k]) for k in np.unique(b)
                    if (b == k).sum() >= self.min_n and 0 < y[b == k].mean() < 1}
        self.latest = max(self.cal) if self.cal else None
        return self

    def predict(self, p, ts):
        b = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_convert(None).dt.to_period(self.freq).to_numpy()
        out = self.glob.predict(p).astype(float)
        for k in np.unique(b):
            m = b == k
            c = self.cal.get(k) or (self.cal[self.latest] if self.latest is not None and k > self.latest else None)
            if c is not None:
                out[m] = c.predict(p[m])
        return out


CALIBRATORS = {"raw": Identity, "sigmoid": Platt, "isotonic": Isotonic, "temperature": Temperature}
