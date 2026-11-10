"""End-to-end PhishPulse study: data -> features -> splits -> models -> calibration -> selective -> SHAP -> reports."""
import json, time
import joblib, numpy as np, pandas as pd, lightgbm as lgb
from . import calibrate as C, selective as S, evaluate as E, explain as X, split_temporal as SP
from .canonicalize import clean
from .common import DATA, MODELS, FIG, TAB, REPORTS, cfg
from .features_lexical import extract
from .features_temporal import build
from .train import lgbm, baselines, Ensemble

c = cfg(); SEED = c["model"]["seed"]; rows_md = []
md = lambda title, df: rows_md.append(f"\n### {title}\n\n" + df.to_markdown(floatfmt=".4f") + "\n")


def save(name, df, title):
    df.to_csv(TAB / f"{name}.csv"); md(title, df); print(title); print(df.round(4).to_string())


# ---------------- data ----------------
raw = pd.read_csv(DATA / "raw" / "urls.csv.gz")
df = clean(raw, c["data"]["start"])
df.to_csv(DATA / "processed" / "urls_clean.csv.gz", index=False)
t0 = df.first_seen_ts.min()
LEXF = extract(df.url_canonical); LEX = list(LEXF.columns)
TMPF = build(df, t0)
TEMP = [k for k in TMPF.columns if k.startswith(("dom_", "tld_"))]
CYC = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
F = pd.concat([LEXF, TMPF], axis=1); y_all = df.label.values

stats = df.groupby(["source"]).agg(n=("label", "size"), phishing=("label", "sum"), start=("first_seen_ts", "min"),
                                    end=("first_seen_ts", "max"), domains=("registered_domain", "nunique"))
stats.index = stats.index.str.replace(r"commoncrawl:.*", "commoncrawl", regex=True)
stats = stats.groupby(level=0).agg({"n": "sum", "phishing": "sum", "start": "min", "end": "max", "domains": "sum"})
save("dataset_stats", stats, "Dataset statistics (after canonicalisation + dedup)")

chrono, (t1, t2) = SP.chronological(df, c["split"]["train"], c["split"]["val"])
rand = SP.random_split(df, c["split"]["train"], c["split"]["val"], SEED)
disj = SP.domain_disjoint(df, chrono)
save("split_report", SP.report(df, chrono), "Chronological split")
save("split_report_domain_disjoint", SP.report(df, disj), "Domain-disjoint chronological split")


def fit_lgbm(split, cols, params=None, early=True):
    p = dict(params or c["model"]["lgbm"]); tr, va = split["train"], split["val"]
    m = lgbm(p, SEED)
    m.fit(F.loc[tr, cols], y_all[tr], eval_set=[(F.loc[va, cols], y_all[va])],
          callbacks=[lgb.early_stopping(50, verbose=False)] if early else None)
    return m


def proba(m, split, cols, part):
    return m.predict_proba(F.loc[split[part], cols])[:, 1]


# ---------------- hyper-parameter selection (time-respecting: validation window) ----------------
grid = []
for nl in (31, 63, 127):
    for lr in (0.03, 0.08):
        p = {**c["model"]["lgbm"], "num_leaves": nl, "learning_rate": lr, "n_estimators": 2000}
        m = fit_lgbm(chrono, LEX + TEMP, p)
        grid.append({"num_leaves": nl, "learning_rate": lr, "best_iter": m.best_iteration_,
                     "val_logloss": m.best_score_["valid_0"]["binary_logloss"]})
grid = pd.DataFrame(grid); save("lgbm_tuning", grid, "LightGBM tuning on validation window")
b = grid.loc[grid.val_logloss.idxmin()]
BEST = {**c["model"]["lgbm"], "num_leaves": int(b.num_leaves), "learning_rate": float(b.learning_rate),
        "n_estimators": int(b.best_iter)}

# ---------------- RQ1: realism gap ----------------
gap = {}
for name, sp in [("random", rand), ("chronological", chrono), ("domain-disjoint chronological", disj)]:
    m = fit_lgbm(sp, LEX, BEST, early=False)
    gap[name] = E.metrics(y_all[sp["test"]], proba(m, sp, LEX, "test"))
save("rq1_split_gap", pd.DataFrame(gap).T[["pr_auc", "roc_auc", "f1", "recall", "fpr", "ece", "brier"]],
     "RQ1 - Random vs chronological vs domain-disjoint (LightGBM, URL-only)")

# ---------------- baselines + RQ2 ----------------
yte, yva = y_all[chrono["test"]], y_all[chrono["val"]]
probs_test, probs_val, lat = {}, {}, {}
for name, (est, kind) in baselines(SEED).items():
    if kind == "num":
        est.fit(F.loc[chrono["train"], LEX], y_all[chrono["train"]])
        probs_test[name] = est.predict_proba(F.loc[chrono["test"], LEX])[:, 1]
    else:
        est.fit(df.loc[chrono["train"], "url_canonical"], y_all[chrono["train"]])
        probs_test[name] = est.predict_proba(df.loc[chrono["test"], "url_canonical"])[:, 1]
    print("trained", name)
m_lex = fit_lgbm(chrono, LEX, BEST, early=False)
m_full = fit_lgbm(chrono, LEX + TEMP, BEST, early=False)
m_cyc = fit_lgbm(chrono, LEX + TEMP + CYC, BEST, early=False)
probs_test["LightGBM (lexical)"] = proba(m_lex, chrono, LEX, "test")
probs_test["LightGBM (+temporal)"] = p_raw = proba(m_full, chrono, LEX + TEMP, "test")
pv_raw = proba(m_full, chrono, LEX + TEMP, "val")
main = pd.DataFrame({k: E.metrics(yte, v) for k, v in probs_test.items()}).T
save("main_classification", main[["pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "fnr", "mcc", "ece", "brier"]],
     "Main classification on future test period (raw probabilities, threshold 0.5)")
cyc = pd.DataFrame({"LEX+TEMP": E.metrics(yte, p_raw), "LEX+TEMP+hour/dow": E.metrics(yte, proba(m_cyc, chrono, LEX + TEMP + CYC, "test"))}).T
save("cyclic_time_check", cyc[["pr_auc", "roc_auc", "f1", "ece"]],
     "Hour/day-of-week features (excluded from main model - benign timestamps are crawl times, i.e. a source artifact)")
ci = E.bootstrap_ci(yte, p_raw)
ci_lex = E.bootstrap_ci(yte, probs_test["LightGBM (lexical)"])
save("bootstrap_ci", pd.concat({"lexical": ci_lex, "+temporal": ci}, axis=1), "95% bootstrap CIs (future test)")

# ---------------- RQ3 / RQ5: calibration ----------------
cal_p, cals = {}, {}
for name, cls in C.CALIBRATORS.items():
    cals[name] = cls().fit(pv_raw, yva); cal_p[name] = cals[name].predict(p_raw)
ts_va, ts_te = df.loc[chrono["val"], "first_seen_ts"].to_numpy(), df.loc[chrono["test"], "first_seen_ts"].to_numpy()
for base in ("isotonic", "sigmoid"):
    tsc = C.TimeStratified(C.Isotonic if base == "isotonic" else C.Platt, c["calibration"]["bucket"], c["calibration"]["min_bucket"])
    tsc.fit(pv_raw, yva, pd.Series(ts_va)); cal_p[f"time-stratified ({base})"] = tsc.predict(p_raw, pd.Series(ts_te))
    cals[f"time-stratified ({base})"] = tsc
calt = pd.DataFrame({k: E.metrics(yte, v) for k, v in cal_p.items()}).T
save("calibration", calt[["ece", "ace", "brier", "log_loss", "pr_auc", "f1"]], "RQ3/RQ5 - Calibration on future test")
recent = df.loc[chrono["test"], "first_seen_ts"] >= df.loc[chrono["test"], "first_seen_ts"].quantile(0.5)
recent = recent.values
save("calibration_recent", pd.DataFrame({k: {"ece_recent_half": E.ece(yte[recent], v[recent]),
                                             "brier_recent_half": E.metrics(yte[recent], v[recent])["brier"]}
                                         for k, v in cal_p.items()}).T, "Calibration on the most recent half of the test period")
E.reliability_plot(yte, {k: cal_p[k] for k in ["raw", "sigmoid", "isotonic", "temperature", "time-stratified (isotonic)"]},
                   FIG / "reliability.png")
CAL = "isotonic"
p_cal, pv_cal = cal_p[CAL], cals[CAL].predict(pv_raw)

# ---------------- RQ4: selective prediction ----------------
thr, ent = S.tune_threshold(yva, pv_cal, c["selective"]["target_risk"], c["selective"]["min_coverage"])
ens = Ensemble(BEST, c["model"]["ensemble_seeds"]).fit(F.loc[chrono["train"], LEX + TEMP], y_all[chrono["train"]])
sd_va = np.stack([cals[CAL].predict(r) for r in ens.predict_all(F.loc[chrono["val"], LEX + TEMP])]).std(0)
sd_te = np.stack([cals[CAL].predict(r) for r in ens.predict_all(F.loc[chrono["test"], LEX + TEMP])]).std(0)
std_lim = float(np.quantile(sd_va, 0.95))
pol, pol_e = S.Policy(thr, ent), S.Policy(thr, ent, std_lim)
sel = pd.DataFrame({"full coverage": S.selective_metrics(yte, p_cal, np.ones_like(yte, bool)),
                    "confidence/entropy policy": S.selective_metrics(yte, p_cal, pol.accept(p_cal)),
                    "+ ensemble disagreement": S.selective_metrics(yte, p_cal, pol_e.accept(p_cal, sd_te)),
                    "validation (policy)": S.selective_metrics(yva, pv_cal, pol.accept(pv_cal))}).T
save("selective", sel, f"RQ4 - Selective prediction (conf>={thr:.4f}, entropy<={ent:.4f}, ens-std<={std_lim:.4f}; tuned on validation)")
curves = {k: S.risk_coverage(yte, v) for k, v in probs_test.items()}
curves["PhishPulse (calibrated)"] = S.risk_coverage(yte, p_cal)
curves["PhishPulse (ens. score)"] = S.risk_coverage(yte, p_cal, S.confidence(p_cal) - sd_te)
E.rc_plot(curves, FIG / "risk_coverage.png")
aurc = pd.DataFrame({"AURC": {k: v[2] for k, v in curves.items()}})
for cv in (0.5, 0.8, 0.9):
    aurc[f"risk@{int(cv*100)}%cov"] = {k: float(v[1][int(cv * len(v[1])) - 1]) for k, v in curves.items()}
save("aurc", aurc, "Risk-coverage summary (future test)")

# ---------------- ablation ----------------
abl = {"Lexical baseline (LogReg)": (probs_test["LogisticRegression"], None),
       "Raw LightGBM (lexical)": (probs_test["LightGBM (lexical)"], None),
       "+ Temporal features": (p_raw, None), "+ Global calibration (isotonic)": (p_cal, None),
       "+ Time-stratified calibration": (cal_p["time-stratified (isotonic)"], None),
       "Full PhishPulse (+ selective)": (p_cal, pol.accept(p_cal))}
ab = {}
for k, (p, acc) in abl.items():
    m = E.metrics(yte, p); a = np.ones_like(yte, bool) if acc is None else acc
    s = S.selective_metrics(yte, p, a)
    ab[k] = {"pr_auc": m["pr_auc"], "f1": m["f1"], "ece": m["ece"], "brier": m["brier"],
             "coverage": s["coverage"], "error_on_automated": s["selective_risk"]}
save("ablation", pd.DataFrame(ab).T, "Ablation (future test)")

# ---------------- SHAP ----------------
Xte = F.loc[chrono["test"], LEX + TEMP].reset_index(drop=True)
samp = Xte.sample(min(3000, len(Xte)), random_state=SEED)
imp_test = X.summary_plot(m_full, samp, FIG / "shap_summary.png")
imp_train = pd.Series(np.abs(X.shap_values(m_full, F.loc[chrono["train"], LEX + TEMP].sample(3000, random_state=SEED))).mean(0),
                      index=LEX + TEMP)
drift = pd.DataFrame({"mean|SHAP| train period": imp_train, "mean|SHAP| future test": imp_test}).sort_values("mean|SHAP| future test", ascending=False).head(15)
save("shap_drift", drift, "SHAP importance drift: older (train) vs future (test) period")
dec = pol.decide(p_cal)
cases = {"phishing": np.flatnonzero((dec == "PHISHING") & (yte == 1)), "benign": np.flatnonzero((dec == "BENIGN") & (yte == 0)),
         "uncertain": np.flatnonzero(dec == "UNCERTAIN")}
examples = []
for k, idx in cases.items():
    if len(idx):
        r = int(idx[len(idx) // 2])
        X.waterfall(m_full, Xte, r, FIG / f"shap_local_{k}.png", f"{k.upper()} p={p_cal[r]:.3f}")
        url = df.loc[chrono["test"][r], "url_canonical"]
        examples.append({"case": k, "url_defanged": url.replace("http", "hxxp").replace(".", "[.]"), "p_cal": float(p_cal[r]),
                         "evidence": X.top_evidence(m_full, Xte.iloc[[r]])[0]})
json.dump(examples, open(REPORTS / "shap_examples.json", "w"), indent=2)

# ---------------- latency / footprint (deployed URL-only fast path) ----------------
from .features_lexical import extract_one
from .canonicalize import canonical
m_dep = fit_lgbm(chrono, LEX, BEST, early=False)
cal_dep = C.Isotonic().fit(proba(m_dep, chrono, LEX, "val"), yva)
pv_dep = cal_dep.predict(proba(m_dep, chrono, LEX, "val"))
thr_d, ent_d = S.tune_threshold(yva, pv_dep, c["selective"]["target_risk"], c["selective"]["min_coverage"])
urls = df.loc[chrono["test"], "url_raw"].sample(500, random_state=SEED).tolist()
fe, mo = [], []
for u in urls:
    t = time.perf_counter(); row = pd.DataFrame([extract_one(canonical(u))])[LEX].astype(np.float32); fe.append(time.perf_counter() - t)
    t = time.perf_counter(); cal_dep.predict(m_dep.predict_proba(row)[:, 1]); mo.append(time.perf_counter() - t)
art = {"model": m_dep, "calibrator": cal_dep, "features": LEX, "conf_thr": thr_d, "ent_limit": ent_d,
       "version": f"phishpulse-urlonly-{pd.Timestamp.now(tz='UTC'):%Y%m%d}", "calibration": "isotonic",
       "trained_through": str(t1)}
joblib.dump(art, MODELS / "phishpulse_urlonly.joblib")
joblib.dump({"model": m_full, "calibrator": cals[CAL], "features": LEX + TEMP, "conf_thr": thr, "ent_limit": ent},
            MODELS / "phishpulse_full.joblib")
import os, resource
latd = pd.DataFrame({"feature_extraction_ms": np.array(fe) * 1e3, "model+calibration_ms": np.array(mo) * 1e3}).describe(percentiles=[.5, .95]).T[["50%", "95%", "mean"]]
latd["model_size_MB"] = os.path.getsize(MODELS / "phishpulse_urlonly.joblib") / 1e6
latd["peak_RSS_MB(process)"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
save("latency", latd, "Latency per URL (single-row, CPU) and footprint - URL-only fast path")
dep = S.selective_metrics(yte, cal_dep.predict(proba(m_dep, chrono, LEX, "test")), S.Policy(thr_d, ent_d).accept(cal_dep.predict(proba(m_dep, chrono, LEX, "test"))))
save("deployed_urlonly", pd.DataFrame({"URL-only deployed model": dep}).T, "Deployed URL-only model, selective metrics (future test)")

meta = {"t_train_end": str(t1), "t_val_end": str(t2), "best_params": BEST, "calibrator": CAL,
        "conf_thr": thr, "entropy_limit": ent, "ens_std_limit": std_lim}
json.dump(meta, open(REPORTS / "run_meta.json", "w"), indent=2, default=str)
(REPORTS / "results.md").write_text("# PhishPulse - Results\n\nGenerated by `python -m src.run_experiments`.\n\n```json\n"
                                    + json.dumps(meta, indent=2, default=str) + "\n```\n" + "".join(rows_md))
print("done")
