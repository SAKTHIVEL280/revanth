"""Random, chronological and domain-disjoint chronological splits."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split


def chronological(df, train=0.65, val=0.15):
    """Cut by time quantiles of first_seen_ts (data-driven, no hard-coded years)."""
    t = df.first_seen_ts
    t1, t2 = t.quantile(train), t.quantile(train + val)
    return {"train": df.index[t < t1], "val": df.index[(t >= t1) & (t < t2)], "test": df.index[t >= t2]}, (t1, t2)


def random_split(df, train=0.65, val=0.15, seed=42):
    tr, rest = train_test_split(df.index, train_size=train, stratify=df.label, random_state=seed)
    va, te = train_test_split(rest, train_size=val / (1 - train), stratify=df.loc[rest, "label"], random_state=seed)
    return {"train": tr, "val": va, "test": te}


def domain_disjoint(df, split):
    """Chronological split but val/test rows whose registered domain occurs in train are removed."""
    seen = set(df.loc[split["train"], "registered_domain"])
    keep = lambda idx: idx[~df.loc[idx, "registered_domain"].isin(seen).values]
    return {"train": split["train"], "val": keep(split["val"]), "test": keep(split["test"])}


def report(df, split):
    rows = []
    for k, idx in split.items():
        d = df.loc[idx]
        rows.append({"split": k, "n": len(d), "phishing": int(d.label.sum()), "benign": int((1 - d.label).sum()),
                     "start": d.first_seen_ts.min(), "end": d.first_seen_ts.max(),
                     "domains": d.registered_domain.nunique()})
    return pd.DataFrame(rows)
