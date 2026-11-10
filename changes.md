# Changes from the documentation

This file lists what could not be done exactly as documented, and what was done instead.

1. **PhishTank direct download → GitHub mirror.** `data.phishtank.com` returned HTTP 429 (rate-limited; it now needs registration). We used the public, daily-updated mirror `Sky-ey/mirror-phishtank`, which carries the same verified feed with `submission_time`. The feed only holds URLs that are *currently online*, so older periods have fewer samples.
2. **OpenPhish not used.** The free feed has no timestamps, so it cannot support chronological evaluation.
3. **Benign data: Common Crawl instead of a Tranco crawl.** Tranco lists domains, not URLs, and has no first-seen times. Benign URLs were instead sampled from Common Crawl CDX index shards: one crawl per quarter, HTTP 200 HTML pages only, at most 2 URLs per host. Each URL's timestamp is its crawl capture time. The CDX API kept timing out, so the pipeline reads byte ranges of the index files directly.
4. **Study window starts 2023-07-01.** Both classes are restricted to this window so their timestamps cover the same period. Phishing goes back to 2012, but those older records are sparse.
5. **Hour and day-of-week features are excluded from the main model.** Benign timestamps are crawl times and phishing timestamps are report times, so these features reflect the data source, not phishing behaviour. They are still computed and reported in `cyclic_time_check`. `days_since_start` is kept for drift diagnostics only.
6. **WHOIS/RDAP domain age and the "enriched" model are not implemented.** No free source provides historical WHOIS at the time of each URL's first sighting, and live lookups would leak future information. Temporal history features (past phishing counts per domain and TLD) take their place. `features_domain.py` from the suggested structure is therefore omitted.
7. **Temporal features assume past labels are known.** History features for a URL use only earlier, already-labelled records. This matches an online system that receives feed verdicts as they arrive. The deployed API is URL-only because it has no live feed state.
8. **Time-stratified calibration on future data.** Test months come after every calibration month, so each test month uses the most recent validation month's calibrator, falling back to the global one. This did not beat global calibration (RQ5 not supported), and the report says so.
9. **Our own calibrator classes instead of `CalibratedClassifierCV(cv="prefit")`.** That option is deprecated in scikit-learn 1.6+. The classes in `calibrate.py` do the same thing: fit on the validation window only. Temperature scaling is applied to LightGBM's log-odds.
10. **Validation window used twice.** It is used for early stopping/tuning and also for fitting calibrators and the selective threshold. A separate window would be cleaner, but there was not enough data. The test set was used once.
11. **Optional models not built:** Character CNN, small Transformer, LIME/anchors and rolling-origin evaluation. Bootstrap confidence intervals are included.
12. **No raw data or live URLs in the repo.** Following the ethics rules in the plan, `data/` is git-ignored and is rebuilt with `python -m src.ingest`. URLs in `reports/shap_examples.json` are defanged.
13. **Results will differ between runs** because the PhishTank feed changes daily.
