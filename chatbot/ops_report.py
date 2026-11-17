"""Operations report from the audit repository:  python -m chatbot.ops_report [path/to/audit.db]

Prints what an operator should look at every week: completion, time to complete (X), response time (Y),
re-asks per question, the decision mix, and drift of the model score against a reference sample (PSI).
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def psi(reference, current, bins=10):
    """Population Stability Index of `current` against `reference`, on quantile bins of the reference."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(reference, edges)[0] / len(reference)
    a = np.histogram(current, edges)[0] / len(current)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def report(db_path, reference_csv=os.path.join(ROOT, "outputs", "predictions_holdout.csv")):
    con = sqlite3.connect(db_path)
    s = pd.read_sql("SELECT * FROM sessions", con)
    t = pd.read_sql("SELECT * FROM turns", con)
    d = pd.read_sql("SELECT * FROM decisions", con)
    out = {"sessions": int(len(s)), "by_status": s["status"].value_counts().to_dict()}
    done = s[s["status"].isin(["completed", "completed_policy", "referred"])]
    out["completion_rate"] = round(len(done) / max(len(s), 1), 3)
    out["duration_s"] = {"p50": round(float(done["duration_s"].median()), 1), "p90": round(float(done["duration_s"].quantile(0.9)), 1)}
    out["questions_asked_median"] = float(done["questions_asked"].median())
    answers = t[~t["slot"].isin(["consent", "summary", "change", "mobile", "control"])]
    rejected = answers[(answers["valid"] == 0) & ~answers["error"].isin(["ambiguous - pick list shown", "confirmation requested"])]
    out["reask_rate_by_slot"] = (rejected.groupby("slot").size() / answers.groupby("slot").size()).fillna(0).round(3).to_dict()
    out["decisions"] = d["decision"].value_counts().to_dict()
    out["decision_sources"] = d["decision_source"].value_counts().to_dict()
    lat = d["decision_latency_ms"].dropna()
    out["decision_latency_ms"] = {"p50": round(float(lat.median()), 1), "p95": round(float(lat.quantile(0.95)), 1)}
    scores = d.loc[d["decision_source"] == "model", "score"].dropna().values
    if len(scores) >= 100 and os.path.exists(reference_csv):
        ref = pd.read_csv(reference_csv)["Probability_DECLINED"].values
        out["score_psi_vs_reference"] = round(psi(ref, scores), 4)
        out["mean_score"] = {"reference": round(float(ref.mean()), 4), "current": round(float(scores.mean()), 4)}
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("ABC_AUDIT_DIR") or os.path.join(ROOT, "logs"), "audit.db")
    for k, v in report(path).items():
        print(f"{k:26s} {v}")
