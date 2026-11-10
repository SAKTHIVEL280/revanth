"""Classification + calibration metrics, bootstrap CIs, plots."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score, recall_score,
                             brier_score_loss, log_loss, matthews_corrcoef, balanced_accuracy_score, confusion_matrix)


def ece(y, p, bins=15):
    """Equal-width 15-bin ECE."""
    e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def ace(y, p, bins=15):
    """Adaptive (equal-mass) calibration error."""
    o = np.argsort(p); chunks = np.array_split(o, bins)
    return float(np.mean([abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)]))


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "f1": f1_score(y, yh),
            "precision": precision_score(y, yh, zero_division=0), "recall": recall_score(y, yh),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1), "mcc": matthews_corrcoef(y, yh),
            "bal_acc": balanced_accuracy_score(y, yh), "ece": ece(y, p), "ace": ace(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def bootstrap_ci(y, p, keys=("pr_auc", "roc_auc", "f1", "ece", "brier"), n=300, seed=0):
    rng, rows = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].mean() < 1:
            m = metrics(y[i], p[i]); rows.append({k: m[k] for k in keys})
    d = pd.DataFrame(rows)
    return pd.DataFrame({"low": d.quantile(0.025), "high": d.quantile(0.975)})


def reliability_plot(y, probs: dict, path, bins=15):
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--", lw=1)
    e = np.linspace(0, 1, bins + 1)
    for name, p in probs.items():
        idx = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() > 20]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() > 20]
        ax.plot(xs, ys, "o-", ms=3, label=f"{name} (ECE={ece(y, p):.3f})")
    ax.set(xlabel="Predicted phishing probability", ylabel="Observed phishing frequency", title="Reliability (future test)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def rc_plot(curves: dict, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (cov, risk, aurc) in curves.items():
        ax.plot(cov, risk, label=f"{name} (AURC={aurc:.4f})")
    ax.set(xlabel="Coverage", ylabel="Selective risk", title="Risk-coverage (future test)", ylim=(0, None))
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
