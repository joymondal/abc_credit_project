"""Behaviour benchmark with made-up applicants:  python -m chatbot.personas

Six invented applicant profiles are each scored at four loan sizes on the same vehicle (60%, 80%,
90% and 96% of its price), plus two applicants outside the age policy: 26 personas in all. Every
persona goes through the same parsers, model, cut-off and audit log as a chat session.

The benchmark does not ask "is the model accurate" (the hold-out answers that). It checks that the
chatbot behaves the way a loan officer would expect:

  1. every persona gets a decision, with reasons, inside the Y-second target;
  2. a large down payment (60% loan-to-value) is approved for every profile;
  3. for the same person and vehicle, borrowing more never makes the application look safer;
  4. the chat engine's probability equals the saved pipeline's probability for the same input;
  5. applicants outside the 18-60 age policy are declined by the rule, without a model score.

No row of the training data is used: the profiles are invented and the vehicles, prices and
pincodes are public catalogue values.
"""
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chatbot.audit import AuditLog          # noqa: E402
from chatbot.engine import LoanChatbot      # noqa: E402

LTV_STEPS = [60, 80, 90, 96]
TOLERANCE = 0.01          # a fall in P(DECLINED) smaller than one point is treated as noise

# profile, vehicle (catalogue key), on-road price, pincode, age, employment, monthly income, active loan, qualification
# (monthly income is used only when the selected model asks for it; the current model does not)
PROFILES = [
    ("Salaried graduate, Bengaluru, scooter", "JUPITER 125 BSVI - DISC", 118500, "560037", 29, "Salaried", 45000, False, "Graduate"),
    ("Shop owner, Pune, commuter motorcycle", "TVS RADEON 110 ES MAG BSVI", 90000, "411014", 41, "Self-employed", 35000, True, "12th (HSC)"),
    ("Farmer, Jhansi district, moped", "TVS XL 100 HD ITS BSVI", 74500, "284001", 47, "Agriculture", 18000, False, "10th (SSC)"),
    ("First-time borrower aged 23, Lucknow, sports motorcycle", "APACHE RTR 160 2V RM DISC BSVI", 152000, "226010", 23, "Salaried", 28000, False, "Graduate"),
    ("Daily-wage worker, Indore, scooter", "NTORQ 125 DISC - RACE EDITION BSVI", 121000, "452001", 34, "Irregular income", 16000, False, "Other / below 10th"),
    ("Post-graduate professional, Hyderabad, electric scooter", "TVS IQUBE UG", 149000, "500081", 36, "Salaried", 80000, True, "Post-graduate"),
]
POLICY_CASES = [
    ("Retired applicant aged 65, Bengaluru, scooter", "JUPITER 125 BSVI - DISC", 118500, "560037", 65, "Pensioner", 30000, False, "Graduate"),
    ("Applicant aged 17, Pune, scooter", "JUPITER 125 BSVI - DISC", 118500, "411014", 17, "Student", 0, False, "12th (HSC)"),
]


def _answers(vehicle, price, pincode, age, employment, income, active, qualification, ltv):
    return {"vehicle_model": vehicle, "on_road_price": price, "loan_amount": round(price * ltv / 100, -2),
            "pincode": pincode, "area_type": "Town / urban", "age": age, "employment_type": employment,
            "net_salary": income, "active_loans": active, "qualification": qualification,
            "residence_type": "Owned"}


def run(audit_path: str | None = None, model: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Score the 26 personas with the chosen model (default: the current one).
    Returns (one row per persona, list of checks with pass / fail)."""
    audit_path = audit_path or os.path.join(tempfile.mkdtemp(prefix="abc_personas_"), "audit.db")
    bot = LoanChatbot(audit=AuditLog(audit_path), channel="personas", model=model)
    rows = []
    for i, (name, *spec) in enumerate(PROFILES, 1):
        for ltv in LTV_STEPS:
            r = bot.score_answers(_answers(*spec, ltv))
            batch = float(bot.pipe.predict_proba(pd.DataFrame([bot.model_input()]))[0, 1]) if r.get("ok") else None
            rows.append({"persona": f"P{i:02d}-{ltv}", "profile": name, "loan_to_value": ltv, **_flat(r), "p_saved_pipeline": batch})
    for j, (name, *spec) in enumerate(POLICY_CASES, 1):
        r = bot.score_answers(_answers(*spec, 70))
        rows.append({"persona": f"R{j:02d}", "profile": name, "loan_to_value": 70, **_flat(r), "p_saved_pipeline": None})
    df = pd.DataFrame(rows)

    model = df[df["decision_source"] == "model"]
    y_ms = 1000 * float(bot.meta.get("y_seconds", 2))
    ladders = model.pivot(index="profile", columns="loan_to_value", values="p_declined")[LTV_STEPS]
    falls = (-ladders.diff(axis=1)).max(axis=1)
    checks = [
        {"check": "Every persona received a decision", "passed": bool(df["ok"].all()), "detail": f"{int(df['ok'].sum())} of {len(df)}"},
        {"check": "Every model decision carries one to three reasons", "passed": bool(model["n_reasons"].between(1, 3).all()),
         "detail": f"reasons per decision: {int(model['n_reasons'].min())} to {int(model['n_reasons'].max())}"},
        {"check": f"Every decision arrived within Y = {y_ms / 1000:.0f} s", "passed": bool((df["latency_ms"] < y_ms).all()),
         "detail": f"slowest {df['latency_ms'].max():.0f} ms"},
        {"check": "A 60% loan-to-value is approved for every profile", "passed": bool((model[model.loan_to_value == 60]["decision"] == "APPROVED").all()),
         "detail": f"{int((model[model.loan_to_value == 60]['decision'] == 'APPROVED').sum())} of {len(PROFILES)} approved"},
        {"check": "Borrowing more on the same vehicle never looks safer", "passed": bool((falls <= TOLERANCE).all()),
         "detail": f"largest fall in P(DECLINED) between steps: {max(float(falls.max()), 0):.4f}"},
        {"check": "Chat engine and saved pipeline give the same probability", "passed": bool((model["p_declined"] - model["p_saved_pipeline"]).abs().max() < 1e-9),
         "detail": f"largest difference {(model['p_declined'] - model['p_saved_pipeline']).abs().max():.1e}"},
        {"check": "Applicants outside the 18-60 age policy are declined by the rule, not the model",
         "passed": bool(((df[df.persona.str.startswith('R')]["decision_source"] == "policy_rule") & (df[df.persona.str.startswith('R')]["decision"] == "DECLINED")).all()),
         "detail": ", ".join(df[df.persona.str.startswith('R')]["decision_source"])},
    ]
    return df, checks


def _flat(r: dict) -> dict:
    if not r.get("ok"):
        return {"ok": False, "decision": None, "decision_source": "invalid_input", "p_declined": None, "n_reasons": 0,
                "main_reason": str(r.get("errors")), "latency_ms": float("nan")}
    reasons = r.get("reasons") or []
    return {"ok": True, "decision": r["decision"], "decision_source": r["decision_source"],
            "p_declined": None if r.get("p_declined") is None else float(r["p_declined"]),
            "n_reasons": len(reasons), "main_reason": reasons[0] if reasons else "", "latency_ms": r["latency_ms"]}


def main():
    # python -m chatbot.personas [--model previous] [--keep-log]; the default model writes outputs/personas_benchmark.csv
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
    df, checks = run(os.path.join(ROOT, "logs", "personas_audit.db") if "--keep-log" in sys.argv else None, model=model)
    out = os.path.join(ROOT, "outputs", "personas_benchmark.csv" if model in (None, "current") else f"personas_benchmark_{model}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.drop(columns=["ok", "p_saved_pipeline"]).round({"p_declined": 4, "latency_ms": 1}).to_csv(out, index=False)
    model = df[df["decision_source"] == "model"]
    print(model.pivot(index="profile", columns="loan_to_value", values="p_declined").round(3).to_string())
    print("\nDeclined, by loan-to-value step:", model.groupby("loan_to_value")["decision"].apply(lambda s: int((s == "DECLINED").sum())).to_dict(),
          f"of {len(PROFILES)} profiles each")
    for c in checks:
        print(("PASS  " if c["passed"] else "FAIL  ") + c["check"] + "  [" + c["detail"] + "]")
    print(f"\n{len(df)} personas -> {os.path.relpath(out, ROOT)}")
    sys.exit(0 if all(c["passed"] for c in checks) else 1)


if __name__ == "__main__":
    main()
