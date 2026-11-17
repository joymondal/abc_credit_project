"""ABC Credit loan chatbot engine: a slot-filling state machine.

Four layers, one responsibility each
  1. Interaction  - templates + quick replies; optional LLM for messy answers / friendlier wording
  2. Validation   - deterministic parsers, range checks, master-data look-ups, confirmation
  3. Decision     - the trained pipeline returns P(DECLINED); code compares it with the threshold
  4. Governance   - PII masking and a complete audit trail of every turn and every decision

The credit decision is produced in exactly one place (`_decide`) by ordinary code. No text typed by
the applicant and no language-model output can reach it except as a validated field value.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from . import parsing as P
from . import pii
from .audit import AuditLog, now

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACTS = os.path.join(os.path.dirname(HERE), "artifacts")
NOTICE_VERSION = "ABC-notice-2026-11"

EMPLOYMENT = {  # code: (label, synonyms)
    "SAL": ("Salaried", ["salary", "job", "employee", "service", "private job", "govt job", "working"]),
    "SEP": ("Self-employed", ["self employed", "business", "shop", "own business", "trader", "freelance"]),
    "AGR": ("Agriculture", ["farmer", "farming", "agri", "kisan", "cultivator"]),
    "NREGI": ("Irregular income", ["daily wage", "labour", "labourer", "casual", "irregular", "non regular"]),
    "STU": ("Student", ["studying", "college"]),
    "PEN": ("Pensioner", ["pension", "retired"]),
    "NPP": ("On probation / not permanent", ["probation", "probationary", "contract", "temporary", "non permanent"]),
    "NONEARNMEM": ("Not earning", ["homemaker", "housewife", "unemployed", "non earning", "dependent"]),
}
QUALIFICATION = {
    "SSC": ("10th (SSC)", ["10th", "tenth", "matric", "sslc"]),
    "HSC": ("12th (HSC)", ["12th", "twelfth", "puc", "intermediate", "plus two"]),
    "UG": ("Undergraduate / diploma", ["diploma", "undergraduate", "iti", "pursuing degree"]),
    "GRAD": ("Graduate", ["degree", "graduation", "btech", "b.tech", "bcom", "b.com", "ba", "bsc", "be"]),
    "PG": ("Post-graduate", ["post graduate", "postgraduate", "masters", "mba", "mtech", "msc", "phd"]),
    "OTHERS": ("Other / below 10th", ["other", "others", "none", "below 10th", "no schooling"]),
}
RESIDENCE = {"O": ("Owned", ["own", "own house", "family house", "parents"]),
             "R": ("Rented", ["rent", "rental", "tenant", "pg"]),
             "L": ("Leased", ["lease"])}
TIERS = {"01 MEGAPOLIS (A)": ("Mega city (Delhi, Mumbai ...)", ["mega city"]),
         "03 LARGE METRO": ("Large metro", ["large metro", "big city"]),
         "04 METRO": ("Metro city", ["metro"]),
         "05 URBAN": ("Town / urban", ["urban", "town"]),
         "06 SEMI-URBAN": ("Semi-urban", ["semi urban", "small town"]),
         "07 RURAL": ("Village / rural", ["rural", "village"])}

# model feature -> customer-facing factor group (used for reason codes)
FACTOR_OF = {"LTV": "down_payment", "down_payment": "down_payment",
             "Loan_Amount": "loan_amount",
             "vehicle_value": "vehicle_price", "price_vs_typical": "vehicle_price",
             "Model_Description": "vehicle", "Make_Code": "vehicle", "Model_Variant": "vehicle",
             "Product_Code": "vehicle", "engine_cc": "vehicle", "is_disc": "vehicle",
             "pin2": "area", "pin3": "area", "pin_zone": "area", "tier_num": "area",
             "Age": "age", "Employment_Type": "employment", "Qualifications": "education",
             "Net_salary": "income", "salary_is_zero": "income", "loan_to_annual_income": "income",
             "PAST_LOANS_ACTIVE": "credit_history", "Resident_Type": "residence"}
REASON_TEXT = {  # shown when a factor pushed the application towards DECLINE
    "down_payment": "the loan covers a high share of the vehicle's price (loan-to-value {LTV:.0f}%)",
    "loan_amount": "the loan amount requested ({loan}) is on the higher side",
    "vehicle_price": "the on-road price of the vehicle in relation to the loan",
    "vehicle": "our lending experience with this vehicle model",
    "area": "our lending experience with applications from your pincode area",
    "age": "the applicant's age band",
    "employment": "the type of employment stated",
    "education": "the education level stated",
    "income": "the loan amount in relation to the income stated",
    "credit_history": "limited repayment track record with existing loans",
    "residence": "the type of residence stated",
}
STRENGTH_TEXT = {  # shown when a factor supported APPROVAL
    "down_payment": "a healthy down payment (loan-to-value {LTV:.0f}%)",
    "loan_amount": "a moderate loan amount ({loan})",
    "vehicle_price": "the vehicle's price in relation to the loan",
    "vehicle": "the vehicle model chosen",
    "area": "our lending experience in your area",
    "age": "your age band",
    "employment": "your employment type",
    "education": "your education level",
    "income": "the loan amount in relation to your income",
    "credit_history": "your existing loan track record",
    "residence": "your residence type",
}
NEXT_APPROVED = ["Keep your Aadhaar, PAN and address proof ready for e-KYC at the dealership.",
                 "You will receive a Key Fact Statement (interest rate, fees, EMI) before you sign anything.",
                 "This approval is in principle and is subject to verification of the details you gave."]
NEXT_DECLINED = ["A higher down payment (a smaller loan) can change the outcome - you are welcome to re-apply.",
                 "You can ask for a review of this decision by a loan officer at the dealership.",
                 "For any concern, write to our Grievance Redressal Officer (details on the ABC Credit website)."]


@dataclass
class Result:
    value: object = None
    error: str | None = None
    choices: list | None = None      # ask the user to pick one of these
    confirm: str | None = None       # ask the user to confirm an unusual value


@dataclass
class Slot:
    key: str
    label: str
    prompt: Callable[[dict], str]
    parse: Callable[[str, dict], Result]
    options: Callable[[dict], list] = lambda s: []
    ask_if: Callable[[dict], bool] = lambda s: True
    llm_hint: str = ""
    clears: tuple = ()               # slots that must be re-asked when this one changes


_ARTIFACT_CACHE: dict = {}


def load_artifacts(folder: str = DEFAULT_ARTIFACTS) -> dict:
    """Load the pipeline, metadata and master data once per process."""
    if folder not in _ARTIFACT_CACHE:
        model_path = os.path.join(folder, "model_final.joblib")
        with open(model_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        _ARTIFACT_CACHE[folder] = {
            "pipe": joblib.load(model_path),
            "meta": json.load(open(os.path.join(folder, "model_meta.json"))),
            "ref": json.load(open(os.path.join(folder, "reference_data.json"))),
            "sha256": sha,
        }
    return _ARTIFACT_CACHE[folder]


class LoanChatbot:
    def __init__(self, artifacts: str = DEFAULT_ARTIFACTS, audit: AuditLog | None = None,
                 llm=None, channel: str = "cli"):
        a = load_artifacts(artifacts)
        self.pipe, self.meta, self.ref, self.model_sha = a["pipe"], a["meta"], a["ref"], a["sha256"]
        self.threshold = float(self.meta["threshold"])
        self.catalogue = self.ref["vehicle_catalogue"]
        # ABC_AUDIT_DIR lets a deployment point the audit repository at a mounted volume (e.g. /app/data)
        audit_dir = os.environ.get("ABC_AUDIT_DIR") or os.path.join(os.path.dirname(artifacts.rstrip("/")), "logs")
        if audit is None:
            try:
                audit = AuditLog(os.path.join(audit_dir, "audit.db"))
            except OSError:                       # read-only host: keep logging, in the temp folder
                import tempfile
                audit = AuditLog(os.path.join(tempfile.gettempdir(), "abc_audit", "audit.db"))
        self.audit = audit
        self.llm, self.channel = llm, channel
        self.slots = [s for s in self._slot_definitions() if s.key in self.meta["chatbot_slots"]]
        self.stage = "new"

    # ------------------------------------------------------------------ slot definitions
    def _slot_definitions(self) -> list[Slot]:
        cat = self.catalogue
        opt = lambda d: (lambda s: [v[0] for v in d.values()])

        def p_vehicle(text, s):
            exact, cands = P.match_vehicle(text, cat)
            if exact:
                return Result(exact)
            if cands:
                return Result(choices=cands)
            return Result(error="I could not find that model. Please type the model name as on your "
                                "quotation, e.g. 'Jupiter 125 disc', 'Raider', 'Apache 160', 'XL 100', 'iQube'.")

        def p_price(text, s):
            typical = cat[s["vehicle_model"]]["typical_price"]
            if P.parse_yes_no(text) is True and not re.search(r"\d", text):
                return Result(int(typical))
            amt = P.parse_amount(text)
            if amt is None:
                return Result(error="Please enter the on-road price in rupees, e.g. 105000, or type 'ok'.")
            if not 20_000 <= amt <= 600_000:
                return Result(error="On-road prices are usually between Rs 20,000 and Rs 6,00,000. Please re-enter.")
            if not 0.6 * typical <= amt <= 1.6 * typical and s.get("_price_confirmed") != amt:
                s["_pending_flag"] = ("_price_confirmed", amt)
                return Result(amt, confirm=f"That is quite different from the usual price of this model "
                                           f"({P.inr(typical)}). Is {P.inr(amt)} correct? (Yes / No)")
            return Result(amt)

        def p_loan(text, s):
            amt = P.parse_amount(text)
            if amt is None:
                return Result(error="Please enter the loan amount in rupees, e.g. 75000.")
            price = s["on_road_price"]
            if amt < 10_000:
                return Result(error="The minimum loan is Rs 10,000. Please enter a larger amount.")
            if amt > price:
                return Result(error=f"The loan cannot be more than the on-road price ({P.inr(price)}). "
                                    "Please enter a smaller amount.")
            return Result(amt)

        def p_pin(text, s):
            pin = P.parse_pincode(text)
            if pin is None:
                return Result(error="Please enter a valid 6-digit pincode, e.g. 560037.")
            tier = self.ref["pincode_tier"].get(str(pin)) or self.ref["pin3_tier"].get(str(pin)[:3])
            s["_tier"], s["_need_area"] = tier, tier is None
            return Result(pin)

        def p_area(text, s):
            code = P.match_option(text, TIERS)
            if code is None:
                return Result(error="Please choose the option that best describes your area.")
            s["_tier"] = code
            return Result(code)

        def p_age(text, s):
            a = P.parse_int(text)
            if a is None or not 10 <= a <= 100:
                return Result(error="Please enter your age in completed years, e.g. 29.")
            lo, hi = self.meta.get("age_policy", [18, 60])
            if not lo <= a <= hi and s.get("_age_confirmed") != a:
                s["_pending_flag"] = ("_age_confirmed", a)
                return Result(a, confirm=f"Just to confirm, your age is {a}? (Yes / No)")
            return Result(a)

        def p_emp(text, s):
            code = P.match_option(text, EMPLOYMENT)
            return Result(code) if code else Result(error="Please choose one of the employment types shown.")

        def q_income(s):
            e = s.get("employment_type")
            if e in ("STU", "NONEARNMEM"):
                return "Do you have any monthly income of your own? Enter the amount in Rs, or 0 if none."
            if e == "PEN":
                return "What is your monthly pension (in Rs)?"
            if e in ("SEP", "AGR", "NREGI"):
                return "On average, how much do you earn in a month (in Rs)? An estimate is fine."
            return "What is your monthly take-home salary (in Rs)?"

        def p_income(text, s):
            if P.parse_yes_no(text) is False and not re.search(r"\d", text):
                return Result(0)
            amt = P.parse_amount(text)
            if amt is None:
                return Result(error="Please enter your monthly income in rupees, e.g. 30000 (0 if none).")
            if amt > 1_000_000:
                return Result(error="That looks too high for a monthly figure. Please enter your MONTHLY income.")
            if amt > 200_000 and s.get("_income_confirmed") != amt:
                s["_pending_flag"] = ("_income_confirmed", amt)
                return Result(amt, confirm=f"Is {P.inr(amt)} your income per MONTH (not per year)? (Yes / No)")
            return Result(amt)

        def p_yesno(text, s):
            v = P.parse_yes_no(text)
            return Result(v) if v is not None else Result(error="Please answer Yes or No.")

        def p_qual(text, s):
            code = P.match_option(text, QUALIFICATION)
            return Result(code) if code else Result(error="Please choose one of the education options shown.")

        def p_res(text, s):
            code = P.match_option(text, RESIDENCE)
            return Result(code) if code else Result(error="Please choose Owned, Rented or Leased.")

        return [
            Slot("vehicle_model", "Vehicle",
                 lambda s: "Which vehicle would you like to finance? Type the model name, e.g. 'Jupiter 125 disc'.",
                 p_vehicle, llm_hint="two-wheeler model name as text",
                 clears=("on_road_price", "loan_amount")),
            Slot("on_road_price", "On-road price",
                 lambda s: (f"The usual on-road price of {self._vname(s['vehicle_model'])} is about "
                            f"{P.inr(cat[s['vehicle_model']]['typical_price'])}. What is the on-road price on "
                            "your quotation? (Type 'ok' to use this price.)"),
                 p_price, options=lambda s: ["OK"], llm_hint="on-road price in rupees as a number",
                 clears=("loan_amount",)),
            Slot("loan_amount", "Loan amount",
                 lambda s: "How much loan do you need (in Rs)? The rest will be your down payment.",
                 p_loan, llm_hint="loan amount in rupees as a number"),
            Slot("pincode", "Pincode",
                 lambda s: "What is the 6-digit pincode of your residence?",
                 p_pin, llm_hint="6-digit Indian pincode", clears=("area_type",)),
            Slot("area_type", "Area type",
                 lambda s: "I don't have that pincode on record. Which best describes your area?",
                 p_area, options=opt(TIERS), ask_if=lambda s: bool(s.get("_need_area"))),
            Slot("age", "Age", lambda s: "How old are you (in completed years)?",
                 p_age, llm_hint="age in years as an integer"),
            Slot("employment_type", "Employment",
                 lambda s: "What is your employment type?", p_emp, options=opt(EMPLOYMENT),
                 llm_hint="one of: " + ", ".join(v[0] for v in EMPLOYMENT.values()),
                 clears=("net_salary",)),
            Slot("net_salary", "Monthly income", q_income, p_income,
                 llm_hint="monthly income in rupees as a number"),
            Slot("active_loans", "Other active loans",
                 lambda s: "Do you currently have any other active loan (home, personal, vehicle, gold ...)? (Yes / No)",
                 p_yesno, options=lambda s: ["Yes", "No"], llm_hint="'yes' or 'no'"),
            Slot("qualification", "Education",
                 lambda s: "What is your highest education?", p_qual, options=opt(QUALIFICATION),
                 llm_hint="one of: " + ", ".join(v[0] for v in QUALIFICATION.values())),
            Slot("residence_type", "Residence",
                 lambda s: "Is your home owned, rented or leased?", p_res, options=opt(RESIDENCE),
                 llm_hint="one of: Owned, Rented, Leased"),
        ]

    # ------------------------------------------------------------------ helpers
    def _vname(self, key: str) -> str:
        return self.catalogue[key].get("display", key.title())

    @staticmethod
    def _msg(text, options=None, **extra) -> dict:
        return {"text": text, "options": options or [], **extra}

    def _next_slot(self) -> Slot | None:
        for sl in self.slots:
            if sl.key not in self.s and sl.ask_if(self.s):
                return sl
        return None

    def _display(self, key, value):
        if key == "vehicle_model":
            return self._vname(value)
        if key in ("on_road_price", "loan_amount", "net_salary"):
            return P.inr(value)
        if key == "pincode":
            return pii.mask_pincode(value)
        if key == "employment_type":
            return EMPLOYMENT[value][0]
        if key == "qualification":
            return QUALIFICATION[value][0]
        if key == "residence_type":
            return RESIDENCE[value][0]
        if key == "area_type":
            return TIERS[value][0]
        if key == "active_loans":
            return "Yes" if value else "No"
        return str(value)

    def _masked_value(self, key, value):
        """What is written to the turn log for a parsed value."""
        if value is None:
            return None
        if key == "pincode":
            return pii.mask_pincode(value)
        if key == "mobile":
            return {"token": pii.token(str(value), "phone"), "masked": pii.mask_phone(value)}
        return value

    def _log_turn(self, slot, user_text, value, valid, error, source, t0):
        self.turn_no += 1
        self.audit.turn(session_id=self.session_id, turn_no=self.turn_no, ts=now(), slot=slot,
                        bot_prompt=pii.scrub_text(self.last_prompt),
                        user_input_masked=pii.scrub_text(user_text),
                        parsed_value_masked=json.dumps(self._masked_value(slot, value)),
                        valid=int(bool(valid)), error=error, parse_source=source,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    def _log_session(self, status, ended=False):
        self.audit.session(session_id=self.session_id, started_at=self.started_at,
                           ended_at=now() if ended else None, status=status, channel=self.channel,
                           consent=int(self.consent), notice_version=NOTICE_VERSION,
                           questions_asked=len(self.asked), reasks=self.reasks,
                           duration_s=round(time.time() - self.t_start, 2),
                           llm_enabled=int(bool(self.llm and self.llm.available)),
                           llm_model=getattr(self.llm, "model", None) if self.llm and self.llm.available else None)

    def _ask(self, slot: Slot, prefix: str = "") -> dict:
        self.current = slot
        self.asked.add(slot.key)
        self.last_prompt = slot.prompt(self.s)
        n = len([x for x in self.slots if x.key != "area_type"])
        done = len([k for k in self.s if not k.startswith("_") and k != "area_type"])
        return self._msg((prefix + " " if prefix else "") + self.last_prompt, slot.options(self.s),
                         slot=slot.key, progress=f"{min(done + 1, n)}/{n}")

    # ------------------------------------------------------------------ conversation
    def start(self) -> list[dict]:
        self.session_id = "S-" + uuid.uuid4().hex[:10].upper()
        self.s, self.stage, self.pending = {}, "consent", None
        self.turn_no, self.reasks, self.asked, self.consent = 0, 0, set(), False
        self.t_start, self.started_at = time.time(), now()
        self.result = None
        self.last_options = ["Yes, I agree", "No"]
        n = len([x for x in self.slots if x.key != "area_type"])
        mins = self.meta.get("x_minutes", 4)
        self.last_prompt = (
            f"Welcome to ABC Credit. I am an automated assistant and can give you a decision on a two-wheeler "
            f"loan in about {mins} minutes ({n} short questions). I will use your answers only to assess this "
            f"application; the decision is made by ABC Credit's credit model, and you can ask for a review by "
            f"a loan officer. Please do not share Aadhaar, PAN or bank details here. Do you agree to continue?")
        self._log_session("started")
        return [self._msg(self.last_prompt, ["Yes, I agree", "No"], slot="consent")]

    def handle(self, text: str) -> list[dict]:
        t0 = time.perf_counter()
        if self.stage == "new":
            return self.start()
        text = str(text or "").strip()
        low = text.lower()
        if low in {"restart", "start over", "reset"}:
            if self.stage != "done":
                self._log_session("abandoned", ended=True)
            return self.start()
        if self.stage == "done":
            return [self._msg("This application is complete. Type 'restart' to begin a new one.")]
        if low in {"stop", "quit", "exit", "cancel"}:
            self._log_turn("control", text, None, True, None, "rules", t0)
            self.stage = "done"
            self._log_session("abandoned", ended=True)
            return [self._msg("No problem - I have stopped this application. Nothing has been submitted.")]
        if re.fullmatch(r"[1-9]", text) and self.last_options and int(text) <= len(self.last_options):
            text = self.last_options[int(text) - 1]             # a bare digit picks the n-th quick reply
        out = getattr(self, "_on_" + self.stage)(text, t0)
        self.last_options = out[-1].get("options") or []
        return out

    def _on_consent(self, text, t0):
        yn = P.parse_yes_no(text)
        self._log_turn("consent", text, yn, yn is not None, None if yn is not None else "unclear", "rules", t0)
        if yn is None:
            return [self._msg("Please answer Yes to continue or No to stop.", ["Yes, I agree", "No"], slot="consent")]
        if not yn:
            self.stage = "done"
            self._log_session("no_consent", ended=True)
            return [self._msg("Understood - I will not continue. You can apply at any ABC Credit dealership.")]
        self.consent = True
        self.stage = "slots"
        return [self._ask(self._next_slot())]

    def _on_slots(self, text, t0):
        slot, s = self.current, self.s
        res, source = None, "rules"
        if self.pending and self.pending[0] == "choices":
            keys = self.pending[1]
            m = re.fullmatch(r"\s*(\d{1,2})\s*\.?", text)
            if m and 1 <= int(m.group(1)) <= len(keys):
                res = Result(keys[int(m.group(1)) - 1])
            else:
                hit = [k for k in keys if self._vname(k).lower() == text.lower() or k.lower() == text.lower()]
                res = Result(hit[0]) if hit else slot.parse(text, s)
        elif self.pending and self.pending[0] == "confirm":
            yn = P.parse_yes_no(text)
            if yn is True:
                flag, val = self.pending[2]
                s[flag] = val
                res = Result(self.pending[1])
            elif yn is False:
                self.pending = None
                self._log_turn(slot.key, text, None, False, "value not confirmed", "rules", t0)
                return [self._ask(slot, "No problem.")]
            else:
                res = slot.parse(text, s)            # they may have typed a corrected value
        else:
            res = slot.parse(text, s)
            llm_ok = bool(self.llm and self.llm.available and slot.llm_hint)
            if res.value is None and not res.choices and llm_ok:
                cand, _ = self.llm.extract(slot.key, slot.llm_hint, pii.scrub_text(text))   # LLM proposes ...
                if cand is not None:
                    res2 = slot.parse(str(cand), s)                                          # ... rules decide
                    if res2.value is not None or res2.choices:
                        res, source = res2, "llm+rules"
        self.pending = None

        if res.choices:
            self.pending = ("choices", res.choices)
            self._log_turn(slot.key, text, None, False, "ambiguous - pick list shown", source, t0)
            self.last_prompt = "I found a few matching models. Which one is yours?"
            return [self._msg(self.last_prompt, [self._vname(k) for k in res.choices], slot=slot.key)]
        if res.confirm:
            self.pending = ("confirm", res.value, s.pop("_pending_flag"))
            self._log_turn(slot.key, text, res.value, False, "confirmation requested", source, t0)
            self.last_prompt = res.confirm
            return [self._msg(res.confirm, ["Yes", "No"], slot=slot.key)]
        if res.value is None:
            self.reasks += 1
            off_topic = "?" in text or re.match(r"(?i)\s*(ignore|system|what|why|how|who|tell|approve|you are)\b", text)
            err = ("I can only help with the details of this application here. " if off_topic else "") + res.error
            self._log_turn(slot.key, text, None, False, res.error, source, t0)
            self.last_prompt = err
            return [self._msg(err, slot.options(s), slot=slot.key)]

        s[slot.key] = res.value
        self._log_turn(slot.key, text, res.value, True, None, source, t0)
        if slot.key == "age":
            lo, hi = self.meta.get("age_policy", [18, 60])
            if not lo <= res.value <= hi:
                return self._policy_decline(res.value, lo, hi, t0)
        nxt = self._next_slot()
        return [self._ask(nxt)] if nxt else [self._summary()]

    def _summary(self) -> dict:
        self.stage = "summary"
        s = self.s
        ltv = 100 * s["loan_amount"] / s["on_road_price"]
        lines = []
        for sl in self.slots:
            if sl.key in s:
                extra = (f" (down payment {P.inr(s['on_road_price'] - s['loan_amount'])}, LTV {ltv:.1f}%)"
                         if sl.key == "loan_amount" else "")
                lines.append(f"- {sl.label}: {self._display(sl.key, s[sl.key])}{extra}")
        self.last_prompt = "Please check your details:\n" + "\n".join(lines) + "\nShall I submit the application?"
        return self._msg(self.last_prompt, ["Yes, submit", "Change something"], slot="summary")

    def _on_summary(self, text, t0):
        yn = P.parse_yes_no(text)
        if yn is True:
            self._log_turn("summary", text, True, True, None, "rules", t0)
            return self._decide(t0)
        self._log_turn("summary", text, False, True, None, "rules", t0)
        self.stage = "change"
        labels = [sl.label for sl in self.slots if sl.key in self.s]
        self.last_prompt = "Which detail would you like to change?"
        return [self._msg(self.last_prompt, labels, slot="change")]

    def _on_change(self, text, t0):
        opts = {sl.key: (sl.label, [sl.key.replace("_", " ")]) for sl in self.slots if sl.key in self.s}
        key = P.match_option(text, opts)
        self._log_turn("change", text, key, key is not None, None if key else "unknown field", "rules", t0)
        if key is None:
            return [self._msg("Please pick one of the fields shown.", [v[0] for v in opts.values()], slot="change")]
        slot = next(sl for sl in self.slots if sl.key == key)
        for k in (key, *slot.clears):
            self.s.pop(k, None)
        for flag in [k for k in self.s if k.startswith("_") and k not in ("_tier", "_need_area")]:
            self.s.pop(flag)
        self.stage = "slots"
        return [self._ask(self._next_slot())]

    def _on_mobile(self, text, t0):
        self.stage = "done"
        m = pii.PATTERNS["PHONE"].search(text)
        if not m:
            self._log_turn("mobile", text, None, True, None, "rules", t0)
            return [self._msg("Thank you for applying with ABC Credit. Have a good day!")]
        self._log_turn("mobile", text, re.sub(r"\D", "", m.group())[-10:], True, None, "rules", t0)
        return [self._msg(f"Thank you! Our team will contact you on {pii.mask_phone(m.group())}.")]

    # ------------------------------------------------------------------ decision layer
    def model_input(self) -> dict:
        """Validated answers + master data -> the raw columns the pipeline was trained on.
        Gender is never collected; the pincode is coarsened to its sorting district."""
        s = self.s
        v = self.catalogue[s["vehicle_model"]]
        return {
            "ID": None, "Gender": None,
            "Qualifications": s.get("qualification"), "Employment_Type": s.get("employment_type"),
            "Age": s.get("age"), "Pincode": pii.coarsen_pincode(s["pincode"]),
            "Product_Code": v["Product_Code"], "Make_Code": v["Make_Code"],
            "Model_Description": s["vehicle_model"], "Model_Variant": v["Model_Variant"],
            "Loan_Amount": float(s["loan_amount"]),
            "LTV": round(100 * s["loan_amount"] / s["on_road_price"], 2),
            "Resident_Type": s.get("residence_type"), "Net_salary": s.get("net_salary"),
            "Final_Tier": s.get("_tier"),
            "PAST_LOANS_ACTIVE": (None if "active_loans" not in s else
                                  "PAST_LOANS_ACTIVE" if s["active_loans"] else "NO_PAST_LOANS"),
        }

    def score(self, x: dict) -> dict:
        """Features -> probability -> decision. Timed stage by stage."""
        t0 = time.perf_counter()
        feats = self.pipe.named_steps["features"].transform(pd.DataFrame([x]))
        Xt = np.asarray(self.pipe.named_steps["encode"].transform(feats), dtype=float)
        t1 = time.perf_counter()
        p = float(self.pipe.named_steps["model"].predict_proba(Xt)[0, 1])
        decision = "DECLINED" if p >= self.threshold else "APPROVED"      # <-- the ONLY decision point
        t2 = time.perf_counter()
        names = list(self.pipe.named_steps["encode"].get_feature_names_out())
        reasons = self._reason_codes(Xt, names, x, decision)
        return {"p": p, "decision": decision, "reasons": reasons,
                "engineered": {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                               for k, v in feats.iloc[0].items()},
                "vector": {n: round(float(v), 6) for n, v in zip(names, Xt[0])},
                "feature_ms": (t1 - t0) * 1000, "scoring_ms": (t2 - t1) * 1000}

    def _reason_codes(self, Xt, names, x, decision, k=3) -> list[dict]:
        """Per-applicant factor contributions (exact TreeSHAP values from LightGBM, in log-odds),
        summed by customer-facing factor. Faithful to the computation by construction."""
        try:
            contrib = self.pipe.named_steps["model"].booster_.predict(Xt, pred_contrib=True)[0][:-1]
        except Exception:
            return []
        groups: dict[str, float] = {}
        for n, c in zip(names, contrib):
            base = n if n in FACTOR_OF else next((col for col in FACTOR_OF if n.startswith(col + "_")), None)
            g = FACTOR_OF.get(base, "other")
            groups[g] = groups.get(g, 0.0) + float(c)
        sign = 1 if decision == "DECLINED" else -1
        texts = REASON_TEXT if decision == "DECLINED" else STRENGTH_TEXT
        top = [g for g, v in sorted(groups.items(), key=lambda kv: -sign * kv[1])
               if sign * v > 0.02 and g in texts][:k]
        fmt = {"LTV": x["LTV"], "loan": P.inr(x["Loan_Amount"])}
        return [{"factor": g, "text": texts[g].format(**fmt), "log_odds": round(groups[g], 4)} for g in top]

    def _decision_text(self, decision, reasons) -> tuple[str, str, list[str]]:
        r = "; ".join(x["text"] for x in reasons)
        if decision == "APPROVED":
            head = "Good news - your loan application is APPROVED in principle."
            body = (f"What helped: {r}.\n" if r else "") + "Next steps:\n" + "\n".join("- " + n for n in NEXT_APPROVED)
            return head, body, NEXT_APPROVED
        head = "We are unable to approve this loan application at present (DECLINED)."
        body = (f"The main factors were: {r}.\n" if r else "") + "What you can do:\n" + "\n".join("- " + n for n in NEXT_DECLINED)
        return head, body, NEXT_DECLINED

    @staticmethod
    def kill_switch_on() -> bool:
        """Operational safety flag: ABC_KILL_SWITCH=1 stops automated decisions without a redeploy."""
        return os.environ.get("ABC_KILL_SWITCH", "").strip().lower() in {"1", "true", "on", "yes"}

    def _refer(self, t0) -> list[dict]:
        """Kill switch engaged: record the application and hand it to a loan officer. No score is produced."""
        ms = (time.perf_counter() - t0) * 1000
        x = self.model_input()
        self.audit.decision(session_id=self.session_id, ts=now(), decision_source="kill_switch",
                            model_version=self.meta["model_version"], model_sha256=self.model_sha,
                            model_input_json=x, engineered_features_json={}, model_vector_json={}, score=None,
                            threshold=self.threshold, decision="REFERRED", reason_codes_json=[],
                            feature_ms=0, scoring_ms=0, explanation_ms=0, explanation_source="template",
                            decision_latency_ms=round(ms, 2))
        self.result = {"p": None, "decision": "REFERRED", "reasons": [], "decision_latency_ms": ms,
                       "model_input": x, "duration_s": time.time() - self.t_start}
        self.stage = "mobile"
        self._log_session("referred", ended=True)
        self.last_prompt = "Please share a mobile number so that the loan officer can reach you (or type 'skip')."
        return [self._msg("Thank you - your application has been recorded and will be reviewed by a loan officer, "
                          "who will contact you with the decision.", decision="REFERRED", latency_ms=round(ms, 1)),
                self._msg(self.last_prompt, ["Skip"], slot="mobile")]

    def score_answers(self, answers: dict) -> dict:
        """Headless channel (REST API, batch, another front end): one call with all answers.
        Every value passes through the same parsers and validators as the chat, the same model
        and threshold decide, and the same audit records are written."""
        t0 = time.perf_counter()
        self.start()
        self.consent, self.stage = True, "slots"
        errors = {}
        for sl in self.slots:
            if not sl.ask_if(self.s):
                continue
            if answers.get(sl.key) is None:
                errors[sl.key] = "missing"
                break
            raw = answers[sl.key]
            text = ("yes" if raw else "no") if isinstance(raw, bool) else str(raw)
            self.last_prompt = sl.prompt(self.s)
            self.asked.add(sl.key)
            res = sl.parse(text, self.s)
            self.s.pop("_pending_flag", None)                 # the calling system has already confirmed its values
            if res.choices:
                errors[sl.key] = {"ambiguous": res.choices}
            elif res.value is None:
                errors[sl.key] = res.error
            self._log_turn(sl.key, text, res.value, sl.key not in errors, None if sl.key not in errors else str(errors[sl.key]),
                           "rules", t0)
            if sl.key in errors:
                break                                         # later validators depend on earlier values
            self.s[sl.key] = res.value
        if errors:
            self.stage = "done"
            self._log_session("invalid_input", ended=True)
            return {"ok": False, "session_id": self.session_id, "errors": errors}
        lo, hi = self.meta.get("age_policy", [18, 60])
        if not lo <= self.s["age"] <= hi:
            self._policy_decline(self.s["age"], lo, hi, t0)
        else:
            self._decide(t0)
        self.stage = "done"
        r = self.result
        return {"ok": True, "session_id": self.session_id, "decision": r["decision"], "p_declined": r.get("p"),
                "threshold": self.threshold, "decision_source": ("policy_rule" if r.get("policy") else
                                                                 "kill_switch" if r["decision"] == "REFERRED" else "model"),
                "reasons": [x["text"] for x in r.get("reasons", [])], "latency_ms": round(r["decision_latency_ms"], 2),
                "model_version": self.meta["model_version"]}

    def _decide(self, t0) -> list[dict]:
        if self.kill_switch_on():
            return self._refer(t0)
        x = self.model_input()
        out = self.score(x)
        head, body, steps = self._decision_text(out["decision"], out["reasons"])
        decision_ms = (time.perf_counter() - t0) * 1000          # final input -> decision ready to display
        text, source, expl_ms = body, "template", 0.0
        if self.llm and self.llm.available:
            text, source, expl_ms = self.llm.explain(out["decision"], [r["text"] for r in out["reasons"]], steps, body)
        self.audit.decision(session_id=self.session_id, ts=now(), decision_source="model",
                            model_version=self.meta["model_version"], model_sha256=self.model_sha,
                            model_input_json=x, engineered_features_json=out["engineered"],
                            model_vector_json=out["vector"], score=round(out["p"], 6),
                            threshold=self.threshold, decision=out["decision"],
                            reason_codes_json=out["reasons"], feature_ms=round(out["feature_ms"], 2),
                            scoring_ms=round(out["scoring_ms"], 2), explanation_ms=round(expl_ms, 2),
                            explanation_source=source, decision_latency_ms=round(decision_ms, 2))
        self.result = {**out, "decision_latency_ms": decision_ms, "model_input": x,
                       "duration_s": time.time() - self.t_start}
        self.stage = "mobile"
        self._log_session("completed", ended=True)
        self.last_prompt = "If you would like our team to call you, please share your mobile number (or type 'skip')."
        return [self._msg(head, decision=out["decision"], latency_ms=round(decision_ms, 1)),
                self._msg(text, explanation_source=source),
                self._msg(self.last_prompt, ["Skip"], slot="mobile")]

    def _policy_decline(self, age, lo, hi, t0) -> list[dict]:
        """Eligibility rule from the product policy (ages 18-60). Deterministic, logged as such;
        the model is not called and no score is produced."""
        ms = (time.perf_counter() - t0) * 1000
        self.audit.decision(session_id=self.session_id, ts=now(), decision_source="policy_rule",
                            model_version=self.meta["model_version"], model_sha256=self.model_sha,
                            model_input_json={"Age": age}, engineered_features_json={}, model_vector_json={},
                            score=None, threshold=self.threshold, decision="DECLINED",
                            reason_codes_json=[{"factor": "age_policy",
                                                "text": f"this product is offered to applicants aged {lo} to {hi}"}],
                            feature_ms=0, scoring_ms=0, explanation_ms=0, explanation_source="template",
                            decision_latency_ms=round(ms, 2))
        self.result = {"p": None, "decision": "DECLINED", "decision_latency_ms": ms, "policy": True}
        self.stage = "done"
        self._log_session("completed_policy", ended=True)
        return [self._msg("We are unable to approve this application online (DECLINED).", decision="DECLINED",
                          latency_ms=round(ms, 1)),
                self._msg(f"This product is offered online to applicants aged {lo} to {hi}. This is an eligibility "
                          "rule, not an assessment of you. A co-applicant within this age range can apply, and a "
                          "loan officer at the dealership can review your case.")]
