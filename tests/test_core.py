import numpy as np, pandas as pd
from src.canonicalize import canonical, registered_domain, clean
from src.features_lexical import extract_one
from src.features_temporal import build
from src.selective import tune_threshold, Policy, risk_coverage
from src.evaluate import ece
from src.calibrate import Isotonic, Temperature, TimeStratified


def test_canonical():
    assert canonical("HTTP://WWW.Example.com:80/a/") == "http://example.com/a"
    assert canonical("example.com") == "http://example.com/"
    assert registered_domain("http://a.b.example.co.uk/x") == "example.co.uk"


def test_lexical():
    f = extract_one("http://192.168.0.1/login/verify.php?redirect=http://x.com")
    assert f["is_ip"] == 1 and f["sus_tokens"] >= 2 and f["redirect_param"] == 1 and f["embedded_urls"] >= 1
    f = extract_one("https://xn--pypal-4ve.com/")
    assert f["punycode"] == 1 and f["https"] == 1
    assert extract_one("http://paypal.evil.xyz/")["brand_not_reg"] == 1


def test_temporal_strictly_past():
    ts = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-20", "2024-03-01"], utc=True)
    df = pd.DataFrame({"registered_domain": ["a.com"] * 5, "tld": ["com"] * 5, "first_seen_ts": ts, "label": [1, 1, 1, 0, 1]})
    t = build(df, ts.min())
    assert list(t.dom_phish_1d) == [0, 0, 2, 0, 0]          # same-timestamp rows never see each other
    assert list(t.dom_phish_30d) == [0, 0, 2, 3, 0]
    assert t.tld_seen.tolist() == [0, 0, 2, 3, 4]


def test_clean_dedup_conflict():
    df = pd.DataFrame({"url_raw": ["http://www.a.com/x/", "http://a.com/x"], "label": [0, 1],
                       "first_seen_ts": ["2024-01-01T00:00:00+00:00", "2024-02-01T00:00:00+00:00"], "source": ["s", "s"]})
    out = clean(df, "2023-01-01")
    assert len(out) == 1 and out.label.iloc[0] == 1


def test_calibration_and_selective():
    rng = np.random.default_rng(0); p = rng.uniform(size=5000); y = (rng.uniform(size=5000) < p ** 3).astype(int)
    assert ece(y, Isotonic().fit(p, y).predict(p)) < ece(y, p)
    assert Temperature().fit(p, y).T > 0
    ts = pd.Series(pd.date_range("2024-01-01", periods=5000, freq="h", tz="UTC"))
    q = TimeStratified(min_n=500).fit(p, y, ts).predict(p, ts + pd.Timedelta(days=365))
    assert q.shape == p.shape
    thr, ent = tune_threshold(y, p, 0.05, 0.1)
    acc = Policy(thr, ent).accept(p)
    assert 0 < acc.mean() <= 1
    cov, risk, aurc = risk_coverage(y, p)
    assert 0 <= aurc <= 1 and cov[-1] == 1
