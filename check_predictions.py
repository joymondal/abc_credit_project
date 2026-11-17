"""Check a predictions file before it is submitted.

    python check_predictions.py --pred predictions.csv [--test test.csv]

Checks the column names, one row per ID, no missing values, the two allowed labels, probabilities in [0, 1],
that each Decision agrees with the saved threshold, and (with --test) that the IDs match the test file exactly.
Exits with status 1 if any check fails, so it can run in CI or before upload.
"""
import argparse, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def check(pred_path, test_path=None, meta_path=os.path.join(HERE, "artifacts", "model_meta.json")):
    t = json.load(open(meta_path))["threshold"]
    df = pd.read_csv(pred_path)
    res = []
    def ok(name, cond, detail=""): res.append((name, bool(cond), detail))
    ok("columns are ID, Decision, Probability_DECLINED", list(df.columns) == ["ID", "Decision", "Probability_DECLINED"], str(list(df.columns)))
    ok("no missing values", not df.isna().any().any(), f"{int(df.isna().sum().sum())} missing")
    ok("one row per ID", df["ID"].is_unique, f"{int(df['ID'].duplicated().sum())} duplicated IDs")
    ok("Decision is APPROVED or DECLINED", df["Decision"].isin(["APPROVED", "DECLINED"]).all(), str(sorted(df["Decision"].unique())))
    pr = pd.to_numeric(df["Probability_DECLINED"], errors="coerce")
    ok("probabilities lie in [0, 1]", pr.between(0, 1).all())
    agree = ((pr >= t) == (df["Decision"] == "DECLINED")).mean()
    ok(f"Decision agrees with threshold {t}", agree == 1.0, f"{agree:.4%} agree")
    if test_path:
        ids = pd.read_csv(test_path, usecols=["ID"])["ID"]
        ok("same number of rows as the test file", len(df) == len(ids), f"{len(df)} vs {len(ids)}")
        ok("same set of IDs as the test file", set(df["ID"]) == set(ids))
    return df, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--test")
    a = ap.parse_args()
    df, res = check(a.pred, a.test)
    for name, passed, detail in res:
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not passed else ""))
    print(f"{len(df):,} rows; {int((df['Decision'] == 'DECLINED').sum()):,} DECLINED ({(df['Decision'] == 'DECLINED').mean():.1%})")
    sys.exit(0 if all(p for _, p, _ in res) else 1)
