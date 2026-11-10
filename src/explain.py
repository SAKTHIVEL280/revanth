"""SHAP TreeExplainer outputs for the LightGBM component."""
import numpy as np, pandas as pd, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def shap_values(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    return sv[1] if isinstance(sv, list) else sv


def top_evidence(model, X, k=5):
    """Per-row top-k features by |SHAP| with direction, for analyst-facing output."""
    sv = shap_values(model, X); out = []
    for r in range(len(X)):
        o = np.argsort(-np.abs(sv[r]))[:k]
        out.append([{"feature": X.columns[j], "value": float(X.iloc[r, j]), "shap": float(sv[r, j]),
                     "direction": "towards PHISHING" if sv[r, j] > 0 else "towards BENIGN"} for j in o])
    return out


def summary_plot(model, X, path, max_display=20):
    sv = shap_values(model, X)
    shap.summary_plot(sv, X, show=False, max_display=max_display)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)


def waterfall(model, X, row, path, title):
    e = shap.TreeExplainer(model)
    ex = e(X.iloc[[row]])
    if ex.values.ndim == 3:
        ex = ex[..., 1]
    shap.plots.waterfall(ex[0], show=False, max_display=12)
    plt.title(title, fontsize=9); plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
