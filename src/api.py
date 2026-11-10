"""FastAPI service: POST /score -> decision, calibrated probability, confidence, uncertainty, SHAP evidence.
Run: uvicorn src.api:app --port 8000"""
from datetime import datetime, timezone
from typing import Optional
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from .canonicalize import canonical
from .common import MODELS
from .features_lexical import extract_one
from .explain import top_evidence
from .selective import confidence, entropy

ART = joblib.load(MODELS / "phishpulse_urlonly.joblib")
app = FastAPI(title="PhishPulse", version=ART["version"])


class Req(BaseModel):
    url: str
    observed_ts: Optional[datetime] = None


def score(url: str, observed_ts=None, k=5) -> dict:
    X = pd.DataFrame([extract_one(canonical(url))])[ART["features"]].astype(np.float32)
    p = float(ART["calibrator"].predict(ART["model"].predict_proba(X)[:, 1])[0])
    conf, ent = float(confidence(np.array([p]))[0]), float(entropy(np.array([p]))[0])
    ok = conf >= ART["conf_thr"] and ent <= ART["ent_limit"] + 1e-9
    reason = None if ok else f"confidence {conf:.3f} < threshold {ART['conf_thr']:.3f} (route to analyst / sandbox / enrichment)"
    return {"decision": ("PHISHING" if p >= 0.5 else "BENIGN") if ok else "UNCERTAIN",
            "calibrated_probability": round(p, 4), "confidence": round(conf, 4), "uncertainty": round(ent, 4),
            "defer_reason": reason,
            "top_evidence": [f"{e['feature']}: {e['value']:g} ({e['direction']})" for e in top_evidence(ART["model"], X, k)[0]],
            "model_version": ART["version"], "calibration": ART["calibration"],
            "observed_ts": (observed_ts or datetime.now(timezone.utc)).isoformat()}


@app.post("/score")
def score_ep(r: Req):
    return score(r.url, r.observed_ts)


@app.get("/health")
def health():
    return {"status": "ok", "version": ART["version"]}
