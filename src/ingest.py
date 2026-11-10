"""Download free, timestamped data.
Phishing: PhishTank verified feed (GitHub mirror; PhishTank direct download is rate-limited).
Benign:   URLs sampled from Common Crawl CDX index shards (byte-range reads), timestamp = crawl capture time.
"""
import gzip, io, json, random, zlib
from datetime import datetime, timezone
import pandas as pd, requests
from .common import DATA, cfg

CC = "https://data.commoncrawl.org/cc-index/collections/{}/indexes/cdx-{:05d}.gz"


def phishing(c):
    df = pd.read_csv(c["data"]["phishtank_url"])
    df = df[df.verified.eq("yes")]
    return pd.DataFrame({"url_raw": df.url, "label": 1, "first_seen_ts": df.submission_time,
                         "source": "phishtank", "target": df.target})


def _gunzip_prefix(b):
    out, d = [], zlib.decompressobj(31)
    while b:
        try:
            out.append(d.decompress(b))
        except zlib.error:
            break
        if not d.eof:
            break
        b, d = d.unused_data, zlib.decompressobj(31)
    return b"".join(out).decode("utf-8", "ignore")


def crawls(start):
    info = requests.get("https://index.commoncrawl.org/collinfo.json", timeout=60).json()
    by_q = {}
    for x in info:  # one crawl per quarter, CC-MAIN-YYYY-WW
        parts = x["id"].split("-")
        if len(parts) != 4:
            continue
        y, w = int(parts[2]), int(parts[3])
        q = (y, (min(w, 52) - 1) // 13)
        if q >= (int(start[:4]), (int(start[5:7]) - 1) // 3) and q not in by_q:
            by_q[q] = x["id"]
    return sorted(by_q.values())


def benign(c):
    d, rng, rows = c["data"], random.Random(c["model"]["seed"]), []
    for crawl in crawls(d["start"]):
        for shard in rng.sample(range(300), d["cc_shards_per_crawl"]):
            try:
                r = requests.get(CC.format(crawl, shard), headers={"Range": f"bytes=0-{d['cc_bytes_per_shard']}"}, timeout=60)
                lines = _gunzip_prefix(r.content).splitlines()
            except Exception as e:
                print("skip", crawl, shard, e); continue
            rng.shuffle(lines); per_host, n = {}, 0
            for ln in lines:
                try:
                    _, ts, js = ln.split(" ", 2); j = json.loads(js)
                except ValueError:
                    continue
                if j.get("status") != "200" or j.get("mime") != "text/html":
                    continue
                host = ln.split(")", 1)[0]
                if per_host.get(host, 0) >= d["max_urls_per_host"]:
                    continue
                per_host[host] = per_host.get(host, 0) + 1
                rows.append((j["url"], 0, datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat(), f"commoncrawl:{crawl}", None))
                n += 1
                if n >= d["cc_urls_per_shard"]:
                    break
        print(crawl, len(rows))
    return pd.DataFrame(rows, columns=["url_raw", "label", "first_seen_ts", "source", "target"])


def main():
    c = cfg()
    df = pd.concat([phishing(c), benign(c)], ignore_index=True)
    df["collection_ts"] = datetime.now(timezone.utc).isoformat()
    df.to_csv(DATA / "raw" / "urls.csv.gz", index=False)
    print(df.label.value_counts())


if __name__ == "__main__":
    main()
