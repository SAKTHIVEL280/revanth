"""URL-only lexical + structural features (no network access)."""
import ipaddress, math, re
from collections import Counter
from urllib.parse import urlsplit, parse_qsl
import numpy as np, pandas as pd
from .canonicalize import _EXT

SUS_TOKENS = ["login", "signin", "verify", "secure", "account", "update", "bank", "confirm", "password",
              "wallet", "support", "webscr", "billing", "unlock", "suspend", "auth", "recovery", "payment"]
SUS_TLDS = {"zip", "mov", "top", "xyz", "sbs", "cfd", "click", "icu", "buzz", "rest", "tk", "ml", "ga", "cf",
            "gq", "cyou", "monster", "shop", "online", "live", "info", "lol", "bond", "win", "support", "site"}
# fixed, documented brand lexicon for brand/registered-domain mismatch
BRANDS = ["paypal", "apple", "microsoft", "office", "outlook", "google", "amazon", "netflix", "facebook",
          "instagram", "whatsapp", "dhl", "fedex", "usps", "chase", "wellsfargo", "bankofamerica", "coinbase",
          "binance", "metamask", "allegro", "steam", "adobe", "dropbox", "linkedin", "att", "ebay", "docusign"]
FREE_HOSTS = {"blogspot.com", "weebly.com", "wixsite.com", "github.io", "000webhostapp.com", "firebaseapp.com",
              "web.app", "pages.dev", "vercel.app", "netlify.app", "glitch.me", "godaddysites.com", "sites.google.com",
              "ipfs.io", "workers.dev", "r2.dev", "webflow.io", "framer.app", "square.site", "duckdns.org"}
HOMO = set("аеорсухіјѕԁɡӏ０１")  # common Cyrillic/fullwidth look-alikes


def entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def max_run(s):
    return max((len(m.group()) for m in re.finditer(r"(.)\1*", s)), default=0)


def is_ip(h):
    try:
        ipaddress.ip_address(h.strip("[]")); return 1
    except ValueError:
        return int(bool(re.fullmatch(r"(0x[0-9a-f]+|\d+)", h)))  # hex / decimal-encoded IPs


def extract_one(url: str) -> dict:
    u = url if "://" in url else "http://" + url
    s = urlsplit(u)
    host = (s.hostname or "").lower()
    path, query, frag = s.path, s.query, s.fragment
    e = _EXT(host)
    reg = e.top_domain_under_public_suffix or host
    sub = e.subdomain
    low = u.lower()
    n = max(len(u), 1)
    try:
        port = int(s.port is not None)
    except ValueError:
        port = 1
    params = parse_qsl(query, keep_blank_values=True)
    brand_hits = [b for b in BRANDS if b in low]
    f = {
        "url_len": len(u), "host_len": len(host), "path_len": len(path), "query_len": len(query), "frag_len": len(frag),
        "reg_len": len(reg), "host_reg_ratio": len(host) / max(len(reg), 1),
        "digit_ratio": sum(ch.isdigit() for ch in u) / n, "letter_ratio": sum(ch.isalpha() for ch in u) / n,
        "upper_ratio": sum(ch.isupper() for ch in u) / n,
        "punct_ratio": sum(not ch.isalnum() for ch in u) / n,
        "host_digit_ratio": sum(ch.isdigit() for ch in host) / max(len(host), 1),
        **{f"n_{k}": u.count(c) for k, c in [("dot", "."), ("hyphen", "-"), ("underscore", "_"), ("slash", "/"),
                                              ("eq", "="), ("amp", "&"), ("pct", "%"), ("at", "@"), ("tilde", "~")]},
        "host_hyphens": host.count("-"),
        "ent_url": entropy(u), "ent_host": entropy(host), "ent_path": entropy(path),
        "n_params": len(params), "max_param_len": max((len(v) for _, v in params), default=0),
        "max_run": max_run(u),
        "sus_tokens": sum(t in low for t in SUS_TOKENS),
        "is_ip": is_ip(host), "sus_tld": int(e.suffix.split(".")[-1] in SUS_TLDS), "tld_len": len(e.suffix),
        "punycode": int("xn--" in host), "non_ascii": int(any(ord(ch) > 127 for ch in u)),
        "homoglyph": int(any(ch in HOMO for ch in host)),
        "n_subdomains": len(sub.split(".")) if sub else 0, "n_host_labels": host.count(".") + 1,
        "path_depth": len([p for p in path.split("/") if p]),
        "https": int(s.scheme == "https"), "port": port,
        "redirect_param": int(bool(re.search(r"(redirect|url|next|goto|return|dest|continue)=", low))),
        "embedded_urls": max(low.count("http") - 1, 0) + low.count("www.", len(s.scheme) + 3 + 4),
        "brand_in_url": len(brand_hits),
        "brand_not_reg": int(any(b not in reg for b in brand_hits)),
        "free_host": int(reg in FREE_HOSTS or any(host.endswith(fh) for fh in FREE_HOSTS)),
        "path_ext_exec": int(bool(re.search(r"\.(php|exe|zip|apk|js|scr|html?)$", path.lower()))),
        "long_host_label": max((len(x) for x in host.split(".")), default=0),
    }
    return f


def extract(urls) -> pd.DataFrame:
    return pd.DataFrame([extract_one(u) for u in urls]).astype(np.float32)
