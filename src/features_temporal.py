"""Time-aware features. Every history feature for sample i uses ONLY records with first_seen_ts < ts(i)."""
import numpy as np, pandas as pd

WINDOWS = {"1d": "1D", "7d": "7D", "30d": "30D"}


def cyclic(ts: pd.Series) -> pd.DataFrame:
    h, d = ts.dt.hour + ts.dt.minute / 60, ts.dt.dayofweek
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
                         "dow_sin": np.sin(2 * np.pi * d / 7), "dow_cos": np.cos(2 * np.pi * d / 7)}, index=ts.index)


def _past_ratio(df, key, prefix, alpha=10.0):
    """Smoothed historical phishing ratio of an entity using strictly earlier records."""
    d = df[[key, "first_seen_ts", "label"]].sort_values("first_seen_ts")
    prior = d.label.mean() if len(d) else 0.5
    g = d.groupby(key).label
    # cumulative sums shifted by one, then corrected for ties at identical timestamps
    cnt, pos = g.cumcount(), g.cumsum() - d.label
    same = d.groupby([key, "first_seen_ts"]).label
    cnt = cnt - same.cumcount(); pos = pos - (same.cumsum() - d.label)
    ratio = (pos + alpha * prior) / (cnt + alpha)
    return pd.DataFrame({f"{prefix}_phish_ratio": ratio, f"{prefix}_seen": cnt}, index=d.index).reindex(df.index)


def build(df: pd.DataFrame, t0) -> pd.DataFrame:
    """df must contain registered_domain, tld, first_seen_ts, label. Labels of *past* rows are assumed known
    at prediction time (feeds publish verdicts), matching an online deployment."""
    ts = df.first_seen_ts
    parts = [cyclic(ts),
             pd.DataFrame({"days_since_start": (ts - t0).dt.total_seconds() / 86400}, index=df.index),
             _rolling_safe(df, "registered_domain", "dom"), _rolling_safe(df, "tld", "tld"),
             _past_ratio(df, "tld", "tld")]
    return pd.concat(parts, axis=1).astype(np.float32)


def _rolling_safe(df, key, prefix):
    """Vectorised, strictly-past rolling phishing counts (searchsorted per entity)."""
    out = {f"{prefix}_phish_{n}": np.zeros(len(df)) for n in WINDOWS}
    naive = lambda s: (s.dt.tz_convert(None) if s.dt.tz is not None else s).to_numpy()
    ph_times = {k: np.sort(naive(g)) for k, g in df.loc[df.label == 1].groupby(key).first_seen_ts}
    groups, q = df.groupby(key).indices, df.first_seen_ts.values
    for k, times in ph_times.items():
        m = groups[k]
        hi = np.searchsorted(times, q[m], side="left")  # strictly earlier than current timestamp
        for n, w in WINDOWS.items():
            out[f"{prefix}_phish_{n}"][m] = hi - np.searchsorted(times, q[m] - pd.Timedelta(w).to_timedelta64(), side="left")
    o = pd.DataFrame(out, index=df.index)
    o[f"{prefix}_burst"] = o[f"{prefix}_phish_1d"] / (o[f"{prefix}_phish_7d"] / 7 + 1)
    return o
