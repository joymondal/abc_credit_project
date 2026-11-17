"""Checks for the deployment kit:  python -m pytest -q tests"""
import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chatbot.audit import AuditLog                        # noqa: E402
from chatbot.engine import LoanChatbot, load_artifacts    # noqa: E402
from chatbot.simulate import CLEAN, scripted              # noqa: E402

SAMPLE = json.load(open(os.path.join(ROOT, "sample_application.json")))


def test_pinned_runtime_versions_match_the_saved_model():
    libs = load_artifacts()["meta"]["libraries"]
    pins = dict(l.split("#")[0].strip().split("==") for l in open(os.path.join(ROOT, "chatbot", "requirements.txt"))
                if "==" in l.split("#")[0])
    assert pins["scikit-learn"] == libs["scikit-learn"] and pins["lightgbm"] == libs["lightgbm"]
    assert pins["pandas"] == libs["pandas"] and pins["numpy"] == libs["numpy"]


def test_audit_folder_comes_from_the_environment(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("ABC_AUDIT_DIR", d)
    bot = LoanChatbot(channel="test")
    scripted(bot, CLEAN)
    assert os.path.exists(os.path.join(d, "audit.db")) and os.path.getsize(os.path.join(d, "audit.jsonl")) > 0


def test_kill_switch_refers_instead_of_deciding(monkeypatch):
    monkeypatch.setenv("ABC_KILL_SWITCH", "1")
    bot = LoanChatbot(audit=AuditLog(os.path.join(tempfile.mkdtemp(), "a.db")), channel="test")
    scripted(bot, CLEAN)
    d = bot.audit.query("SELECT * FROM decisions")[0]
    assert d["decision"] == "REFERRED" and d["decision_source"] == "kill_switch" and d["score"] is None


def test_headless_scoring_equals_the_chat_and_validates():
    chat = LoanChatbot(audit=AuditLog(os.path.join(tempfile.mkdtemp(), "a.db")), channel="test")
    scripted(chat, CLEAN)
    api = LoanChatbot(audit=AuditLog(os.path.join(tempfile.mkdtemp(), "b.db")), channel="test")
    out = api.score_answers(SAMPLE)
    assert out["ok"] and out["p_declined"] == chat.result["p"] and out["decision"] == chat.result["decision"]
    bad = api.score_answers({**SAMPLE, "loan_amount": 500000})          # loan above the price
    assert not bad["ok"] and "loan_amount" in bad["errors"]


def test_rest_api():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    os.environ["ABC_AUDIT_DIR"] = tempfile.mkdtemp()
    import api
    with TestClient(api.app) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["threshold"] == load_artifacts()["meta"]["threshold"]
        r = client.post("/score", json=SAMPLE)
        assert r.status_code == 200 and r.json()["decision"] in ("APPROVED", "DECLINED")
        assert client.post("/score", json={**SAMPLE, "pincode": "12"}).status_code == 422       # schema
        assert client.post("/score", json={**SAMPLE, "vehicle_model": "spaceship"}).status_code == 422   # validator
        assert len(client.get("/catalogue").json()) > 50
    os.environ.pop("ABC_AUDIT_DIR")


def test_langgraph_extension_reaches_the_same_decision(monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setenv("ABC_AUDIT_DIR", tempfile.mkdtemp())          # keep test sessions out of ./logs
    from extensions.langgraph_bot import run_conversation
    answers = ["yes", "jupiter 125 disc", "1", "ok", "70000", "560037", "29", "salaried", "45000", "no", "graduate",
               "yes", "skip"]
    out = run_conversation(answers, thread_id="t-1")
    chat = LoanChatbot(audit=AuditLog(os.path.join(tempfile.mkdtemp(), "a.db")), channel="test")
    scripted(chat, CLEAN)
    assert out["done"] and out["decision"] == chat.result["decision"]


def test_streamlit_app_chat_then_evaluator_audit_trail(monkeypatch):
    """The hosted web app: one full application in the chat view, then the evaluator view shows what was logged."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ABC_AUDIT_DIR", tempfile.mkdtemp())
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    at = AppTest.from_file(os.path.join(root, "chatbot", "app_streamlit.py"), default_timeout=120).run()
    for answer in ["yes", "jupiter 125 disc", "1", "ok", "70000", "560037", "29", "salaried", "45000", "no", "graduate", "yes", "skip"]:
        at.chat_input[0].set_value(answer).run()
    assert not at.exception
    assert any("APPROVED" in s.value for s in at.success)
    at.sidebar.radio[0].set_value("For the evaluator").run()
    assert not at.exception
    assert any("Turns logged: 1" in m.value for m in at.markdown)            # 13 turns
    assert any("Decision record" in m.value for m in at.markdown)


def test_batch_scoring_of_a_frame_and_metrics():
    """The evaluator's upload path: score a frame with the saved pipeline and compute metrics when labels are present."""
    import pandas as pd
    from chatbot.engine import load_artifacts
    from predict import evaluate, score_frame
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    art = load_artifacts(os.path.join(root, "artifacts"))
    test = pd.read_csv(os.path.join(root, "sample_data", "sample_test_synthetic.csv"))
    pred = score_frame(test, art["pipe"], art["meta"])
    assert list(pred.columns) == ["ID", "Decision", "Probability_DECLINED"] and len(pred) == len(test)
    assert set(pred["Decision"]) <= {"APPROVED", "DECLINED"}
    m = evaluate(pred, pred["Decision"])                                     # labels equal to predictions -> perfect
    assert m["accuracy"] == 1.0 and m["fp"] == 0 and m["fn"] == 0
