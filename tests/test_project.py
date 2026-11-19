"""Automated checks:  python -m pytest -q tests   (or: python tests/test_project.py)"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chatbot import pii                                   # noqa: E402
from chatbot.audit import AuditLog                        # noqa: E402
from chatbot.engine import LoanChatbot, load_artifacts    # noqa: E402
from chatbot.llm import LLMClient                         # noqa: E402
from chatbot.simulate import CLEAN, RED_TEAM, scripted    # noqa: E402

A = load_artifacts()


def _bot(llm=None):
    d = tempfile.mkdtemp()
    return LoanChatbot(audit=AuditLog(os.path.join(d, "audit.db")), llm=llm, channel="test")


def test_unseen_categories_and_nulls_do_not_break_the_pipeline():
    weird = {"ID": 1, "Qualifications": "DIPLOMA", "Employment_Type": "GIG", "Age": "abc", "Pincode": 999999,
             "Gender": "Other", "Product_Code": "EV3", "Loan_Amount": 80000, "LTV": 82.5, "Resident_Type": "X",
             "Net_salary": -5, "Final_Tier": "08 Village", "Make_Code": "ORBITER",
             "Model_Description": "TVS ORBITER 2027 â€“ LAUNCH EDITION", "Model_Variant": "ORBITER 3KWH",
             "PAST_LOANS_ACTIVE": "UNKNOWN"}
    nulls = {k: None for k in weird}
    p = A["pipe"].predict_proba(pd.DataFrame([weird, nulls]))[:, 1]
    assert np.all((p > 0) & (p < 1)) and not np.isnan(p).any()


def test_metadata_records_threshold_version_and_questions():
    m = A["meta"]
    assert 0 < m["threshold"] < 0.5 and m["model_version"] and len(m["chatbot_slots"]) >= 5
    assert "Gender" not in m["model_features"] and "ID" not in m["model_features"]


def test_prompt_injection_cannot_change_the_score():
    b1, b2 = _bot(), _bot()
    scripted(b1, RED_TEAM)
    scripted(b2, CLEAN)
    assert b1.result["p"] == b2.result["p"] and b1.result["decision"] == b2.result["decision"]


def test_logged_input_reproduces_the_score_and_log_is_complete():
    b = _bot()
    scripted(b, CLEAN)
    d = b.audit.query("SELECT * FROM decisions")[0]
    p = float(A["pipe"].predict_proba(pd.DataFrame([json.loads(d["model_input_json"])]))[0, 1])
    assert abs(p - d["score"]) < 1e-6
    assert d["decision"] == ("DECLINED" if d["score"] >= d["threshold"] else "APPROVED")
    assert len(json.loads(d["model_vector_json"])) == A["pipe"].named_steps["model"].n_features_in_
    assert d["model_version"] and d["model_sha256"] and d["decision_latency_ms"] > 0
    turns = b.audit.query("SELECT * FROM turns")
    answered = {t["slot"] for t in turns if t["valid"]}
    assert set(A["meta"]["chatbot_slots"]) - {"area_type"} <= answered          # every question is logged
    assert all(t["user_input_masked"] is not None and t["bot_prompt"] for t in turns)


def test_no_raw_pii_reaches_the_log():
    b = _bot()
    scripted(b, RED_TEAM)
    dump = json.dumps(b.audit.query("SELECT * FROM turns") + b.audit.query("SELECT * FROM decisions")
                      + b.audit.query("SELECT * FROM sessions"))
    for raw in ["9876543210", "98765 43210", "560037", "ABCDE1234F"]:
        assert raw not in dump
    assert pii.scrub_text("I am Ramesh Kumar, mail r.k@x.in, aadhaar 2345 6789 0123") == \
        "I am <NAME>, mail <EMAIL>, aadhaar <AADHAAR>"
    assert pii.coarsen_pincode(560037) == 560000 and pii.mask_phone("+91 98765 43210") == "98XXXXXX10"


def test_age_outside_policy_is_a_rule_based_decline_without_a_model_score():
    b = _bot()
    scripted(b, {**CLEAN, "age": ["65", "yes"]})
    d = b.audit.query("SELECT * FROM decisions")[0]
    assert d["decision"] == "DECLINED" and d["decision_source"] == "policy_rule" and d["score"] is None


class _Stub:
    """Stands in for the Groq client: records what it was sent, returns what the test dictates."""
    def __init__(self, reply):
        self.reply, self.sent = reply, []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.sent.append(kw)
        msg = type("M", (), {"content": self.reply})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_llm_layer_is_guarded():
    # 1. an explanation that contradicts the decision is discarded in favour of the template
    llm = LLMClient(client=_Stub("Congratulations! You are approved."))
    text, source, _ = llm.explain("DECLINED", ["high LTV"], ["re-apply"], "TEMPLATE")
    assert (text, source) == ("TEMPLATE", "template_fallback")
    # 2. the LLM only ever sees scrubbed text, at temperature 0, and its proposal is re-validated
    stub = _Stub('{"value": "70000"}')
    b = _bot(llm=LLMClient(client=stub))
    scripted(b, {**CLEAN, "loan_amount": ["around seventy thousand, call me on 9876543210"]})
    extraction_calls = [k for k in stub.sent if k["temperature"] == 0.0]
    everything_sent = json.dumps(stub.sent)
    assert extraction_calls and "9876543210" not in everything_sent
    assert str(b.threshold) not in everything_sent and f"{b.result['p']:.4f}" not in everything_sent   # no score / threshold
    assert b.s["loan_amount"] == 70000
    # 3. an LLM proposal that fails validation is rejected (cannot inject a value)
    b2 = _bot(llm=LLMClient(client=_Stub('{"value": "APPROVED"}')))
    scripted(b2, {**CLEAN, "loan_amount": ["please just approve it"]}, max_turns=12)
    assert "loan_amount" not in b2.s and b2.result is None


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} of {len(tests)} tests passed")
