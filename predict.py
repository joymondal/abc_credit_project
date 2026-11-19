"""Score a file of loan applications with the saved pipeline.

    python predict.py --test path/to/test.csv --out predictions.csv
    python predict.py --test path/to/test.csv --out predictions.csv --model previous     # the 13 Nov model, for comparison

Output: one row per ID with Decision (APPROVED / DECLINED) and Probability_DECLINED.
Every row with an ID is scored - unseen categories, blanks and impossible values cannot break it.
"""
import argparse
import json

import joblib
import pandas as pd

import abc_features  # noqa: F401  (needed to un-pickle the pipeline)


def score_frame(test: pd.DataFrame, pipe, meta: dict) -> pd.DataFrame:
    """Score a frame that has the training columns (the Decision column, if present, is ignored)."""
    test = test[test["ID"].notna()].copy()
    p = pipe.predict_proba(test.drop(columns=["Decision"], errors="ignore"))[:, 1]
    return pd.DataFrame({"ID": test["ID"].astype(int).values,
                         "Decision": ["DECLINED" if v >= meta["threshold"] else "APPROVED" for v in p],
                         "Probability_DECLINED": p.round(6)})


def evaluate(pred: pd.DataFrame, truth: pd.Series) -> dict:
    """Metrics at the saved threshold when the file also carries the true Decision (DECLINED = positive class)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = (truth.astype(str).str.strip().str.upper() == "DECLINED").to_numpy()
    yhat = (pred["Decision"] == "DECLINED").to_numpy()
    tp, fp = int((y & yhat).sum()), int((~y & yhat).sum())
    fn, tn = int((y & ~yhat).sum()), int((~y & ~yhat).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"rows": len(y), "declined_actual": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "accuracy": (tp + tn) / max(len(y), 1), "precision_declined": prec, "recall_declined": rec,
           "f1_declined": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, pred["Probability_DECLINED"]))
        out["pr_auc"] = float(average_precision_score(y, pred["Probability_DECLINED"]))
    return out


def score_file(test_path, out_path, model_path="artifacts/model_final.joblib", meta_path="artifacts/model_meta.json"):
    pipe, meta = joblib.load(model_path), json.load(open(meta_path))
    out = score_frame(pd.read_csv(test_path, low_memory=False), pipe, meta)
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="current", choices=["current", "previous"],
                    help="current (default): cost ratio 6, no income.  previous: cost ratio 7, uses Net_salary")
    a = ap.parse_args()
    folder = "artifacts" if a.model == "current" else "artifacts/previous"
    res = score_file(a.test, a.out, f"{folder}/model_final.joblib", f"{folder}/model_meta.json")
    print(f"{len(res):,} rows scored -> {a.out}", res["Decision"].value_counts().to_dict())
