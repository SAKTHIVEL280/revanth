"""Selective prediction: confidence/entropy policy, ensemble disagreement, risk-coverage, AURC."""
import numpy as np

EPS = 1e-12


def confidence(p): return np.maximum(p, 1 - p)


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def risk_coverage(y, p, score=None):
    """Sort by confidence (or a custom score, higher = more trusted). Returns coverage, risk arrays and AURC."""
    s = confidence(p) if score is None else score
    o = np.argsort(-s, kind="stable")
    err = ((p[o] >= 0.5).astype(int) != y[o]).astype(float)
    n = np.arange(1, len(y) + 1)
    cov, risk = n / len(y), np.cumsum(err) / n
    return cov, risk, float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))


def tune_threshold(y, p, target_risk=0.02, min_coverage=0.5):
    """On validation: smallest confidence threshold whose accepted set has risk <= target (max coverage).
    If impossible, return the threshold giving min_coverage."""
    c = confidence(p)
    best = None
    for t in np.unique(np.round(c, 4)):
        acc = c >= t
        if acc.mean() < min_coverage:
            break
        risk = ((p[acc] >= 0.5) != y[acc]).mean()
        if risk <= target_risk:
            best = t; break
    if best is None:
        best = float(np.quantile(c, 1 - min_coverage))
    return float(best), float(entropy(np.array([best]))[0])  # entropy limit implied by the confidence threshold


class Policy:
    def __init__(self, conf_thr, ent_limit, std_limit=None):
        self.conf_thr, self.ent_limit, self.std_limit = conf_thr, ent_limit, std_limit

    def accept(self, p, std=None):
        a = (confidence(p) >= self.conf_thr) & (entropy(p) <= self.ent_limit + 1e-9)
        if self.std_limit is not None and std is not None:
            a &= std <= self.std_limit
        return a

    def decide(self, p, std=None):
        a = self.accept(p, std)
        return np.where(~a, "UNCERTAIN", np.where(p >= 0.5, "PHISHING", "BENIGN"))


def selective_metrics(y, p, accept):
    yhat = (p >= 0.5).astype(int)
    cov = accept.mean()
    risk = ((yhat != y) & accept).sum() / max(accept.sum(), 1)
    ph = y == 1
    return {"coverage": cov, "selective_risk": risk, "full_risk": (yhat != y).mean(),
            "phish_recall_accepted": ((yhat == 1) & accept & ph).sum() / max((accept & ph).sum(), 1),
            "phish_auto_caught": ((yhat == 1) & accept & ph).sum() / max(ph.sum(), 1),
            "phish_deferred": (~accept & ph).sum() / max(ph.sum(), 1),
            "phish_missed_auto": ((yhat == 0) & accept & ph).sum() / max(ph.sum(), 1),
            "benign_false_alarm_auto": ((yhat == 1) & accept & ~ph).sum() / max((~ph).sum(), 1)}
