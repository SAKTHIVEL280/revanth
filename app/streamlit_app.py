"""Analyst demo. Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.api import score, ART

st.set_page_config(page_title="PhishPulse", page_icon="🛡️")
st.title("PhishPulse")
st.caption(f"Calibrated, selective phishing-URL detection · {ART['version']} · URL-only fast path (no URL is visited)")
url = st.text_input("URL", "http://secure-login-paypal.com.verify-account.xyz/signin")
if url:
    r = score(url)
    color = {"PHISHING": "red", "BENIGN": "green", "UNCERTAIN": "orange"}[r["decision"]]
    st.markdown(f"## :{color}[{r['decision']}]")
    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated P(phishing)", f"{r['calibrated_probability']:.3f}")
    c2.metric("Confidence", f"{r['confidence']:.3f}")
    c3.metric("Uncertainty (entropy)", f"{r['uncertainty']:.3f}")
    if r["defer_reason"]:
        st.warning(r["defer_reason"])
    st.subheader("Top evidence (SHAP)")
    st.write(r["top_evidence"])
    st.json(r)
