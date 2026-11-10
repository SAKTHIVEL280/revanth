"""URL canonicalization, registered-domain extraction, deduplication."""
from urllib.parse import urlsplit, urlunsplit
import pandas as pd, tldextract

_EXT = tldextract.TLDExtract(suffix_list_urls=())  # bundled PSL snapshot, offline + reproducible


def canonical(url: str) -> str:
    u = str(url).strip()
    if "://" not in u:
        u = "http://" + u
    s = urlsplit(u)
    host = (s.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    port = "" if _bad_port(s) or s.port in (None, 80, 443) else f":{s.port}"
    path = s.path.rstrip("/") or "/"
    return urlunsplit((s.scheme.lower(), host + port, path, s.query, ""))


def _bad_port(s):
    try:
        s.port; return False
    except ValueError:
        return True


def registered_domain(url: str) -> str:
    e = _EXT(urlsplit(url if "://" in url else "http://" + url).hostname or "")
    return e.top_domain_under_public_suffix or e.domain or ""


def tld(url: str) -> str:
    return _EXT(urlsplit(url if "://" in url else "http://" + url).hostname or "").suffix or ""


def clean(df: pd.DataFrame, start: str) -> pd.DataFrame:
    df = df.dropna(subset=["url_raw"]).copy()
    df["first_seen_ts"] = pd.to_datetime(df.first_seen_ts, utc=True, errors="coerce", format="ISO8601")
    df = df[df.first_seen_ts >= pd.Timestamp(start, tz="UTC")]
    df["url_canonical"] = df.url_raw.map(canonical)
    df = df.sort_values("first_seen_ts")
    # conflicting labels for the same canonical URL -> keep phishing (security-conservative), earliest timestamp
    lab = df.groupby("url_canonical").label.max()
    df = df.drop_duplicates("url_canonical", keep="first").copy()
    df["label"] = df.url_canonical.map(lab)
    df["registered_domain"] = df.url_canonical.map(registered_domain)
    df["tld"] = df.url_canonical.map(tld)
    return df.reset_index(drop=True)
