"""ABC Credit - cleaning, feature engineering and model pipeline.

One module shared by the training notebook, the batch scorer and the chatbot,
so every application is cleaned and encoded by exactly the same code
(no training-serving skew). All fitted steps live inside a scikit-learn Pipeline.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

SEED = 42
TARGET = "Decision"
POSITIVE = "DECLINED"            # positive class = the event we want to catch
RAW_COLUMNS = ["ID", "Qualifications", "Employment_Type", "Age", "Pincode", "Gender",
               "Product_Code", "Loan_Amount", "LTV", "Resident_Type", "Net_salary",
               "Final_Tier", "Make_Code", "Model_Description", "Model_Variant",
               "PAST_LOANS_ACTIVE"]
AGE_MIN, AGE_MAX = 18, 60        # policy range stated in the data dictionary
SALARY_CAP = 300_000             # ~99.9th percentile of monthly income

# Model features, grouped by encoder
CAT_LOW = ["Qualifications", "Employment_Type", "Resident_Type", "Product_Code",
           "Make_Code", "Model_Variant", "PAST_LOANS_ACTIVE"]          # one-hot
CAT_HIGH = ["Model_Description", "pin2", "pin3"]                        # target-encoded
NUMERIC = ["Age", "Loan_Amount", "LTV", "Net_salary", "salary_is_zero", "tier_num",
           "pin_zone", "vehicle_value", "down_payment", "price_vs_typical",
           "loan_to_annual_income", "engine_cc", "is_disc"]
ALL_FEATURES = CAT_LOW + CAT_HIGH + NUMERIC

# Which chatbot question produces which model features (used for question selection,
# reason codes and the audit log). Derived features need every listed question.
QUESTION_FEATURES = {
    "vehicle_model":   ["Model_Description", "Make_Code", "Model_Variant", "Product_Code",
                        "engine_cc", "is_disc"],
    "ltv":             ["LTV", "vehicle_value", "down_payment", "price_vs_typical"],
    "loan_amount":     ["Loan_Amount"],
    "pincode":         ["pin2", "pin3", "pin_zone", "tier_num"],
    "age":             ["Age"],
    "active_loans":    ["PAST_LOANS_ACTIVE"],
    "qualification":   ["Qualifications"],
    "employment_type": ["Employment_Type"],
    "net_salary":      ["Net_salary", "salary_is_zero", "loan_to_annual_income"],
    "residence_type":  ["Resident_Type"],
}

_MOJIBAKE = {"â€“": "-", "â€”": "-", "â€™": "'", "â€˜": "'", "Â": "", "–": "-", "—": "-"}


def _clean_text(s: pd.Series) -> pd.Series:
    """Upper-case, repair mis-decoded characters, collapse whitespace; blanks -> NA."""
    s = s.astype("string").str.upper()
    for bad, good in _MOJIBAKE.items():
        s = s.str.replace(bad.upper(), good, regex=False).str.replace(bad, good, regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.isin(["", "NAN", "NONE", "NULL", "NA"]))


def _normalise_variant(s: pd.Series) -> pd.Series:
    """'125CC DISC' and '125 CC DISC' are the same variant: one spelling per category."""
    s = _clean_text(s)
    return s.str.replace(r"(\d)\s*CC\b", r"\1 CC", regex=True)


def _engine_cc(variant: pd.Series, desc: pd.Series, product: pd.Series) -> pd.Series:
    """Engine size parsed from the variant, else the description; electric = 0."""
    text = variant.fillna("") + " | " + desc.fillna("")
    cc = pd.to_numeric(text.str.extract(r"\b(\d{2,3})\s*(?:CC|DISC|DRUM|4V|2V)", expand=False),
                       errors="coerce")
    alt = pd.to_numeric(text.str.extract(r"\b(100|110|125|150|160|180|200|225|310)\b",
                                         expand=False), errors="coerce")
    cc = cc.fillna(alt)
    cc = cc.where(~text.str.contains("RAIDER"), cc.fillna(125))
    cc = cc.where(~text.str.contains("RONIN"), cc.fillna(225))
    cc = cc.where(~text.str.contains("RADEON|SPORT|STAR CITY"), cc.fillna(110))
    cc = cc.where(~text.str.contains("SCOOTY|ZEST|PEP"), cc.fillna(110))
    return cc.mask(product.eq("EB") | text.str.contains("EBIKE|IQUBE"), 0).astype(float)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is safe for training rows, test rows and live chatbot input."""
    out = pd.DataFrame(df).copy()
    for c in RAW_COLUMNS:                                   # tolerate absent columns
        if c not in out.columns:
            out[c] = np.nan
    # 1. Age arrives as text (a blank string in one row) -> numeric, clipped to policy range
    out["Age"] = pd.to_numeric(out["Age"].astype("string").str.strip(), errors="coerce").astype(float)
    out["Age"] = out["Age"].clip(AGE_MIN, AGE_MAX)
    # 2. Numerics: impossible values become missing (LightGBM routes NaN natively)
    for c in ["Loan_Amount", "LTV", "Net_salary"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out.loc[(out["LTV"] <= 0) | (out["LTV"] > 100), "LTV"] = np.nan      # LTV of 0% is impossible
    out.loc[out["Loan_Amount"] <= 0, "Loan_Amount"] = np.nan
    out["salary_is_zero"] = (out["Net_salary"] == 0).astype(float)
    out["Net_salary"] = out["Net_salary"].clip(lower=0, upper=SALARY_CAP)
    # 3. Categorical text: one spelling per category
    for c in ["Qualifications", "Employment_Type", "Resident_Type", "Product_Code", "Make_Code",
              "PAST_LOANS_ACTIVE", "Final_Tier", "Gender", "Model_Description"]:
        out[c] = _clean_text(out[c])
    out["Model_Variant"] = _normalise_variant(out["Model_Variant"])
    # 4. Ordinal market tier: '01 MEGAPOLIS (A)' -> 1 ... '07 RURAL' -> 7
    out["tier_num"] = pd.to_numeric(out["Final_Tier"].str.extract(r"^(\d+)", expand=False),
                                    errors="coerce").astype(float)
    # 5. Pincode hierarchy (India Post): zone = 1st digit, sub-zone = 2, sorting district = 3
    pin = pd.to_numeric(out["Pincode"], errors="coerce")
    pin = pin.where((pin >= 100000) & (pin <= 999999))
    out["pin_zone"] = (pin // 100000).astype(float)
    out["pin2"] = (pin // 10000).astype("Int64").astype("string")
    out["pin3"] = (pin // 1000).astype("Int64").astype("string")
    # 6. Collateral and affordability
    out["vehicle_value"] = out["Loan_Amount"] / (out["LTV"] / 100.0)
    out["down_payment"] = out["vehicle_value"] - out["Loan_Amount"]
    annual = out["Net_salary"] * 12
    out["loan_to_annual_income"] = (out["Loan_Amount"] / annual.where(annual > 0)).clip(upper=10)
    # 7. Vehicle text -> structured attributes
    out["engine_cc"] = _engine_cc(out["Model_Variant"], out["Model_Description"], out["Product_Code"])
    out["is_disc"] = (out["Model_Description"].fillna("").str.contains("DISC")
                      | out["Model_Variant"].fillna("").str.contains("DISC")).astype(float)
    # 8. Missing categoricals become an explicit category (missingness can be informative)
    for c in CAT_LOW + CAT_HIGH:
        out[c] = out[c].astype("string").fillna("MISSING")
    return out


class LoanFeatureBuilder(BaseEstimator, TransformerMixin):
    """Cleaning + feature engineering as a scikit-learn step, so it is saved with the model.

    fit() learns only label-free reference values (typical on-road price per vehicle model).
    `keep` restricts the output to a chosen feature subset (question selection).
    """

    def __init__(self, keep=None):
        self.keep = keep

    def fit(self, X, y=None):
        c = basic_clean(X)
        self.typical_price_ = c.groupby("Model_Description")["vehicle_value"].median().dropna().to_dict()
        self.global_price_ = float(c["vehicle_value"].median())
        return self

    def transform(self, X):
        out = basic_clean(X)
        typical = out["Model_Description"].map(self.typical_price_).astype(float).fillna(self.global_price_)
        out["price_vs_typical"] = out["vehicle_value"] / typical
        cols = [c for c in ALL_FEATURES if self.keep is None or c in self.keep]
        out = out[cols].copy()
        for c in out.columns:                        # pandas 'string' -> object for sklearn encoders
            if str(out[c].dtype) == "string":
                out[c] = out[c].astype(object)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([c for c in ALL_FEATURES if self.keep is None or c in self.keep])


def make_encoder(keep=None) -> ColumnTransformer:
    """One-hot for small categoricals (rare/unseen -> 'infrequent'); cross-fitted, smoothed
    target encoding for high-cardinality ones (unseen -> global mean); numerics pass through."""
    use = lambda cols: [c for c in cols if keep is None or c in keep]
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)
    te = TargetEncoder(target_type="binary", smooth="auto", random_state=SEED,
                       cv=5, shuffle=True)
    parts = []
    if use(CAT_LOW):
        parts.append(("onehot", ohe, use(CAT_LOW)))
    if use(CAT_HIGH):
        parts.append(("target", te, use(CAT_HIGH)))
    if use(NUMERIC):
        parts.append(("num", "passthrough", use(NUMERIC)))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


LGBM_DEFAULTS = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=100,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                     random_state=SEED, n_jobs=2, verbose=-1)


def build_pipeline(keep=None, **lgbm_params) -> Pipeline:
    """features -> encode -> LightGBM. Returns P(DECLINED) via predict_proba()[:, 1]."""
    from lightgbm import LGBMClassifier
    params = {**LGBM_DEFAULTS, **lgbm_params}
    return Pipeline([("features", LoanFeatureBuilder(keep=keep)),
                     ("encode", make_encoder(keep)),
                     ("model", LGBMClassifier(**params))])


def load_labelled(path: str):
    """Read the labelled file, drop unusable rows, return X (raw columns), y (1 = DECLINED), log."""
    df = pd.read_csv(path, low_memory=False)
    log = {"rows_read": len(df)}
    df = df[df[TARGET].isin(["APPROVED", "DECLINED"])].copy()           # blank trailing row
    log["rows_without_label_dropped"] = log["rows_read"] - len(df)
    feat = [c for c in df.columns if c not in ("ID", TARGET)]
    dup = df.duplicated(subset=feat + [TARGET], keep="first")           # exact duplicates
    log["exact_duplicates_dropped"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    df["ID"] = df["ID"].astype(int)
    y = (df[TARGET] == POSITIVE).astype(int)
    log["rows_used"], log["declined"], log["decline_rate"] = len(df), int(y.sum()), float(y.mean())
    return df[RAW_COLUMNS], y, log
