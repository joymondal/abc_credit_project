"""Checks of alternative modelling ideas against our own protocol.

Four questions, each answered with the same 80/20 split, the same 5 stratified folds and the
same seed as the notebook, so every number is comparable with Sections 7 to 12:

  families   Do XGBoost or CatBoost rank applicants better than the LightGBM we ship?
  features   Do extra hand-made flags and ratios (age flags, education order, income ratios,
             interaction flags, a composite risk count) add signal to the shipped feature set?
  monotone   Does forcing "a higher LTV can never look safer" cost accuracy, and how does the
             shipped model behave when the loan grows on the same vehicle?
  augment    What happens to the metrics when synthetic declined rows are added BEFORE the
             train/test split (a tempting but leaky protocol)?

    python peer_checks.py [families|features|monotone|augment|all]

Results are merged into outputs/peer_checks.json. Nothing here changes the saved model.
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

import abc_features as F

warnings.filterwarnings("ignore")
SEED = 42
OUT = "outputs/peer_checks.json"
META = json.load(open("artifacts/model_meta.json"))
KEEP, PARAMS = META["model_features"], META["lgbm_params"]


def load():
    X, y, _ = F.load_labelled("data/training_data.csv")
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def metrics(y, p):
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4)}


def oof(pipe, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def save(key, value):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[key] = value
    json.dump(res, open(OUT, "w"), indent=1)
    print(key, json.dumps(value)[:600])


# ---------------------------------------------------------------- 1. model families
def families():
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    X_tr, _, y_tr, _ = load()
    front = lambda: [("features", F.LoanFeatureBuilder(keep=KEEP)), ("encode", F.make_encoder(KEEP))]
    cands = {
        "LightGBM (shipped settings)": F.build_pipeline(keep=KEEP, **PARAMS),
        "XGBoost": Pipeline(front() + [("model", XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", eval_metric="aucpr",
            random_state=SEED, n_jobs=2, verbosity=0))]),
        "CatBoost": Pipeline(front() + [("model", CatBoostClassifier(
            iterations=700, learning_rate=0.05, depth=5, l2_leaf_reg=5.0, random_seed=SEED,
            thread_count=2, verbose=0))]),
    }
    rows = []
    for name, pipe in cands.items():
        t0 = time.time()
        p = oof(pipe, X_tr, y_tr)
        rows.append({"model": name, **metrics(y_tr, p), "mean_p": round(float(p.mean()), 4), "seconds": round(time.time() - t0)})
        print(rows[-1])
    save("families", rows)


# ---------------------------------------------------------------- 2. extra hand-made features
EDU = {"OTHERS": 1, "SSC": 2, "HSC": 3, "UG": 4, "GRAD": 5, "PG": 6}
EMP = {"STU": 0, "NREGI": 1, "AGR": 2, "SEP": 2, "SAL": 3}
EXTRA = ["loan_to_salary", "ltv_x_loan", "salary_per_age_year", "is_young", "is_senior", "young_high_ltv",
         "education_level", "employment_stability", "profile_score", "is_student", "is_informal",
         "low_education", "informal_low_edu", "has_past_loan", "past_loan_high_ltv", "past_loan_young",
         "new_borrower_informal", "repeat_risk", "is_metro", "is_rural", "loan_vs_tier_median", "composite_risk"]


class ExtraFlags(BaseEstimator, TransformerMixin):
    """The shipped features plus 22 flags, ordinal scores and ratios of the kind often added by hand."""

    def fit(self, X, y=None):
        self.base_ = F.LoanFeatureBuilder(keep=KEEP).fit(X)
        c = F.basic_clean(X)
        self.tier_median_ = c.groupby("tier_num")["Loan_Amount"].median().to_dict()
        self.loan_median_ = float(c["Loan_Amount"].median())
        return self

    def transform(self, X):
        out = self.base_.transform(X)
        c = F.basic_clean(X)
        hi_ltv = (c["LTV"] > 85).astype(float)
        young = (c["Age"] <= 25).astype(float)
        past = (c["PAST_LOANS_ACTIVE"] == "PAST_LOANS_ACTIVE").astype(float)
        informal = (c["Employment_Type"] == "NREGI").astype(float)
        low_edu = c["Qualifications"].isin(["OTHERS", "SSC"]).astype(float)
        lts = (c["Loan_Amount"] / c["Net_salary"].where(c["Net_salary"] > 0)).clip(upper=60)
        tier_med = c["tier_num"].map(self.tier_median_).fillna(self.loan_median_)
        e = pd.DataFrame({
            "loan_to_salary": lts, "ltv_x_loan": c["LTV"] * c["Loan_Amount"] / 1e5,
            "salary_per_age_year": c["Net_salary"] / c["Age"], "is_young": young,
            "is_senior": (c["Age"] > 55).astype(float), "young_high_ltv": young * hi_ltv,
            "education_level": c["Qualifications"].map(EDU).astype(float).fillna(0),
            "employment_stability": c["Employment_Type"].map(EMP).astype(float).fillna(0),
            "is_student": (c["Employment_Type"] == "STU").astype(float), "is_informal": informal,
            "low_education": low_edu, "informal_low_edu": informal * low_edu, "has_past_loan": past,
            "past_loan_high_ltv": past * hi_ltv, "past_loan_young": past * young,
            "new_borrower_informal": (1 - past) * informal, "repeat_risk": past + hi_ltv + young,
            "is_metro": (c["tier_num"] <= 3).astype(float), "is_rural": (c["tier_num"] >= 7).astype(float),
            "loan_vs_tier_median": c["Loan_Amount"] / tier_med}, index=c.index)
        e["profile_score"] = e["education_level"] + e["employment_stability"]
        e["composite_risk"] = (young + informal + low_edu + past + hi_ltv + e["is_rural"]
                               + (lts > 6).astype(float) + (lts > 9).astype(float)
                               + (e["loan_vs_tier_median"] > 1.3).astype(float))
        return pd.concat([out, e[EXTRA]], axis=1)


def features():
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    X_tr, _, y_tr, _ = load()
    base = oof(F.build_pipeline(keep=KEEP, **PARAMS), X_tr, y_tr)
    enc = F.make_encoder(KEEP)
    enc = ColumnTransformer(enc.transformers + [("extra", "passthrough", EXTRA)], verbose_feature_names_out=False)
    plus = Pipeline([("features", ExtraFlags()), ("encode", enc),
                     ("model", LGBMClassifier(**{**F.LGBM_DEFAULTS, **PARAMS}))])
    p = oof(plus, X_tr, y_tr)
    save("features", {"ours_shipped_features": metrics(y_tr, base), "ours_plus_22_handmade": metrics(y_tr, p), "n_shipped_features": len(KEEP),
                      "extra_features": EXTRA})


# ---------------------------------------------------------------- 3. monotonic behaviour
def monotone():
    X_tr, X_ho, y_tr, y_ho = load()
    # (a) constrained LightGBM: P(DECLINED) may not fall when LTV rises
    probe = Pipeline([("features", F.LoanFeatureBuilder(keep=KEEP)), ("encode", F.make_encoder(KEEP))]).fit(X_tr, y_tr)
    names = list(probe.named_steps["encode"].get_feature_names_out())
    cons = [1 if n == "LTV" else 0 for n in names]
    free = oof(F.build_pipeline(keep=KEEP, **PARAMS), X_tr, y_tr)
    tied = oof(F.build_pipeline(keep=KEEP, **PARAMS, monotone_constraints=cons,
                                monotone_constraints_method="advanced"), X_tr, y_tr)
    # (b) behaviour of the shipped model: same applicant, same vehicle, bigger loan
    model = joblib.load("artifacts/model_final.joblib")
    c = F.basic_clean(X_ho)
    ok = c["LTV"].notna() & c["Loan_Amount"].notna()
    sample = X_ho[ok].sample(2000, random_state=SEED)
    price = (c.loc[sample.index, "Loan_Amount"] / (c.loc[sample.index, "LTV"] / 100)).values
    grid = np.arange(50, 100, 5)                                     # LTV 50% ... 95%
    P = np.zeros((len(sample), len(grid)))
    for j, ltv in enumerate(grid):
        s = sample.copy()
        s["LTV"] = float(ltv)
        s["Loan_Amount"] = np.round(price * ltv / 100.0)
        P[:, j] = model.predict_proba(s)[:, 1]
    drop = np.clip(-(np.diff(P, axis=1)), 0, None)                   # falls in P(DECLINED) as the loan grows
    thr = META["threshold"]
    flips = ((P[:, :-1] >= thr) & (P[:, 1:] < thr)).any(axis=1)      # Decline at a lower LTV, Approve at a higher one
    save("monotone", {
        "cv_unconstrained": metrics(y_tr, free), "cv_ltv_constrained": metrics(y_tr, tied),
        "shipped_model_ltv_sweep": {
            "applicants": int(len(sample)), "ltv_grid": [int(g) for g in grid],
            "mean_p_declined_by_ltv": [round(float(v), 4) for v in P.mean(axis=0)],
            "share_with_any_fall_over_1pt": round(float((drop.max(axis=1) > 0.01).mean()), 4),
            "largest_fall": round(float(drop.max()), 4),
            "share_decline_to_approve_flip_as_loan_grows": round(float(flips.mean()), 4)}})


# ---------------------------------------------------------------- 4. synthetic rows before the split
def _synthetic_declines(X, y, per_row=4, k=5, seed=SEED):
    """Nearest-neighbour interpolation of DECLINED rows (the idea behind SMOTE-NC), on raw columns."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    d = X[y == 1].reset_index(drop=True)
    num = ["Age", "Loan_Amount", "LTV", "Net_salary"]
    cat = [c for c in X.columns if c not in num + ["ID"]]
    N = d[num].apply(pd.to_numeric, errors="coerce")
    N = N.fillna(N.median())
    Z = StandardScaler().fit_transform(N)
    dummies = pd.get_dummies(d[["Employment_Type", "Qualifications", "Product_Code", "PAST_LOANS_ACTIVE", "Final_Tier"]].astype(str)).values
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(np.hstack([Z, dummies])).kneighbors(np.hstack([Z, dummies]))
    base = np.repeat(np.arange(len(d)), per_row)
    nb = idx[base, rng.integers(1, k + 1, len(base))]
    gap = rng.random((len(base), len(num)))
    syn = pd.DataFrame(N.values[base] + gap * (N.values[nb] - N.values[base]), columns=num).round(0)
    take_nb = rng.random((len(base), len(cat))) < 0.5
    for j, col in enumerate(cat):
        syn[col] = np.where(take_nb[:, j], d[col].values[nb], d[col].values[base])
    syn["ID"] = 900000 + np.arange(len(syn))
    return syn[X.columns]


def augment():
    X, y, _ = F.load_labelled("data/training_data.csv")
    # Leaky protocol: synthesise declined rows from ALL declined rows, then split
    syn = _synthetic_declines(X, y)
    Xa = pd.concat([X, syn], ignore_index=True)
    ya = pd.concat([y, pd.Series(np.ones(len(syn), dtype=int))], ignore_index=True)
    is_syn = np.r_[np.zeros(len(X), bool), np.ones(len(syn), bool)]
    tr, te = train_test_split(np.arange(len(Xa)), test_size=0.2, stratify=ya, random_state=SEED)
    m = F.build_pipeline(keep=KEEP, **PARAMS).fit(Xa.iloc[tr], ya.iloc[tr])
    p = m.predict_proba(Xa.iloc[te])[:, 1]
    real = ~is_syn[te]
    # Clean protocol with the same synthetic idea: split first, synthesise from TRAINING declines only
    X_tr, X_ho, y_tr, y_ho = load()
    syn_tr = _synthetic_declines(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
    m2 = F.build_pipeline(keep=KEEP, **PARAMS).fit(pd.concat([X_tr, syn_tr], ignore_index=True),
                                                   pd.concat([y_tr, pd.Series(np.ones(len(syn_tr), dtype=int))], ignore_index=True))
    p2 = m2.predict_proba(X_ho)[:, 1]
    save("augment", {
        "synthetic_rows_added": int(len(syn)), "decline_share_after": round(float(ya.mean()), 4),
        "leaky_split_test_all_rows": {**metrics(ya.iloc[te], p), "rows": int(len(te)), "declined_share": round(float(ya.iloc[te].mean()), 4)},
        "leaky_split_test_real_rows_only": {**metrics(ya.iloc[te][real], p[real]), "rows": int(real.sum())},
        "split_first_then_synthesise_real_holdout": {**metrics(y_ho, p2), "mean_p": round(float(p2.mean()), 4),
                                                     "actual_rate": round(float(y_ho.mean()), 4)},
        "our_reported_holdout": {"roc_auc": round(META["holdout"]["roc_auc"], 4), "pr_auc": round(META["holdout"]["pr_auc"], 4)}})


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("families", families), ("features", features), ("monotone", monotone), ("augment", augment)]:
        if what in (name, "all"):
            fn()
