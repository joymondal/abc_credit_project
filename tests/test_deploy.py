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
    answers = ["yes", "jupiter 125 disc", "1", "ok", "70000", "560037", "29", "salaried", "no", "graduate",
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
    for answer in ["yes", "jupiter 125 disc", "1", "ok", "70000", "560037", "29", "salaried", "no", "graduate", "yes", "skip"]:
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


# ------------------------------------------------------------------ model selection (current by default, previous on request)
def test_current_model_is_the_default_and_the_previous_one_can_be_selected(monkeypatch):
    """Two models are installed. The default is the current one (cost ratio 6, no income question); the previous one
    (cost ratio 7, asks income) is used only when selected, and each conversation follows its own model's metadata."""
    import pandas as pd
    from chatbot.engine import DEFAULT_MODEL, available_models
    monkeypatch.delenv("ABC_MODEL", raising=False)
    models = {m["key"]: m for m in available_models()}
    assert list(models)[0] == DEFAULT_MODEL == "current" and "previous" in models
    assert models["current"]["version"] != models["previous"]["version"]
    assert not models["current"]["asks_income"] and models["previous"]["asks_income"]

    def run(model, answers):
        bot = LoanChatbot(audit=AuditLog(os.path.join(tempfile.mkdtemp(), "a.db")), channel="test", model=model)
        bot.start()
        for a in answers:
            bot.handle(a)
        assert bot.stage == "done"
        batch = float(bot.pipe.predict_proba(pd.DataFrame([bot.result["model_input"]]))[0, 1])
        logged = bot.audit.query("SELECT model_version, threshold FROM decisions WHERE session_id = ?", (bot.session_id,))[0]
        return bot, batch, logged

    base = ["yes", "jupiter 125 disc", "1", "ok", "70000", "560037", "29", "salaried"]
    new, p_new, log_new = run(None, base + ["no", "graduate", "yes", "skip"])                 # no model named -> default
    old, p_old, log_old = run("previous", base + ["45000", "no", "graduate", "yes", "skip"])  # one more answer: income
    assert new.model_key == "current" and "net_salary" not in [s.key for s in new.slots] and new.result["model_input"]["Net_salary"] is None
    assert old.model_key == "previous" and "net_salary" in [s.key for s in old.slots] and old.result["model_input"]["Net_salary"] == 45000
    assert new.threshold == models["current"]["threshold"] and old.threshold == models["previous"]["threshold"] != new.threshold
    assert new.result["p"] == pytest.approx(p_new) and old.result["p"] == pytest.approx(p_old) and p_new != p_old
    assert log_new["model_version"] == models["current"]["version"] and log_old["model_version"] == models["previous"]["version"]
    assert log_old["threshold"] == old.threshold                                              # the audit trail names the model that decided

    monkeypatch.setenv("ABC_MODEL", "previous")                                               # a deployment can change its default
    assert LoanChatbot(audit=new.audit, channel="test").model_key == "previous"
    with pytest.raises(ValueError):
        LoanChatbot(audit=new.audit, channel="test", model="no-such-model")


def test_streamlit_model_selector_switches_chat_and_evaluator(monkeypatch):
    """The sidebar selector: new model by default; choosing the older one restarts the application with nine
    questions and its own cut-off, and the evaluator view scores with the same selection."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ABC_AUDIT_DIR", tempfile.mkdtemp())
    monkeypatch.delenv("ABC_MODEL", raising=False)
    at = AppTest.from_file(os.path.join(ROOT, "chatbot", "app_streamlit.py"), default_timeout=120).run()
    selector = next(r for r in at.sidebar.radio if r.label == "Model")
    assert selector.value == "current" and "8 short questions" in at.chat_message[0].markdown[0].value
    at.chat_input[0].set_value("yes").run()                                   # an application is under way ...
    next(r for r in at.sidebar.radio if r.label == "Model").set_value("previous").run()
    assert not at.exception
    assert len(at.chat_message) == 1 and "9 short questions" in at.chat_message[0].markdown[0].value   # ... and is restarted
    for answer in ["yes", "jupiter 125 disc", "1", "ok", "70000", "560037", "29", "salaried", "45000", "no", "graduate", "yes", "skip"]:
        at.chat_input[0].set_value(answer).run()
    assert not at.exception and any("APPROVED" in s.value for s in at.success)
    next(r for r in at.sidebar.radio if r.label == "View").set_value("For the evaluator").run()
    assert not at.exception
    assert any("0.125" in m.value and "2026-11-13" in m.value for m in at.markdown)       # decision rule of the selected model
    next(r for r in at.sidebar.radio if r.label == "Model").set_value("current").run()
    assert not at.exception
    assert any("0.143" in m.value and "2026-11-19" in m.value for m in at.markdown)


def test_rest_api_and_batch_scoring_accept_a_model_choice(monkeypatch):
    """`/score?model=previous` and `predict.py --model previous` use the previous model; without it, the current one."""
    import pandas as pd
    from predict import score_frame
    monkeypatch.delenv("ABC_MODEL", raising=False)
    cur, prev = load_artifacts(os.path.join(ROOT, "artifacts")), load_artifacts(os.path.join(ROOT, "artifacts", "previous"))
    test = pd.read_csv(os.path.join(ROOT, "sample_data", "sample_test_synthetic.csv"))
    a, b = score_frame(test, cur["pipe"], cur["meta"]), score_frame(test, prev["pipe"], prev["meta"])
    assert len(a) == len(b) == len(test) and not a["Probability_DECLINED"].equals(b["Probability_DECLINED"])
    no_income = score_frame(test.drop(columns=["Net_salary"]), prev["pipe"], prev["meta"])   # still scored: income is treated as missing
    assert len(no_income) == len(test)
    assert score_frame(test.drop(columns=["Net_salary"]), cur["pipe"], cur["meta"]).equals(a)  # the current model ignores income
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ABC_AUDIT_DIR", tempfile.mkdtemp())
    import api
    with TestClient(api.app) as client:
        assert [m["key"] for m in client.get("/health").json()["models"]] == ["current", "previous"]
        r_new = client.post("/score", json=SAMPLE).json()
        r_old = client.post("/score?model=previous", json={**SAMPLE, "net_salary": 45000}).json()
        assert r_new["model_version"] == cur["meta"]["model_version"] and r_old["model_version"] == prev["meta"]["model_version"]
        assert r_new["threshold"] == cur["meta"]["threshold"] and r_old["threshold"] == prev["meta"]["threshold"]
        assert client.post("/score?model=previous", json=SAMPLE).status_code == 422       # the previous model needs income
        assert client.post("/score?model=nope", json=SAMPLE).status_code == 422
