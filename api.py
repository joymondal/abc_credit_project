"""Optional scoring service: the same engine behind a REST API, for channels other than the Streamlit chat.

    pip install fastapi uvicorn
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/score -H "Content-Type: application/json" -d @sample_application.json

One model, one threshold, one audit trail: /score runs every value through the chatbot's own validators
(LoanChatbot.score_answers), so a dealer portal or a WhatsApp webhook cannot bypass a rule the chat enforces.
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import abc_features  # noqa: F401  (must be importable before the pipeline is un-pickled)
from chatbot.audit import AuditLog
from chatbot.engine import DEFAULT_ARTIFACTS, LoanChatbot, load_artifacts
import os


class Application(BaseModel):
    """What a client system sends. Field names are the chatbot's slot names."""
    vehicle_model: str = Field(max_length=80, examples=["JUPITER 125 BSVI - DISC"])
    on_road_price: float = Field(gt=0, lt=1_000_000)
    loan_amount: float = Field(gt=0, lt=1_000_000)
    pincode: str = Field(pattern=r"^[1-9]\d{5}$")
    age: int = Field(ge=10, le=100)
    employment_type: str = Field(max_length=40, examples=["SAL"])
    net_salary: float = Field(ge=0)
    active_loans: bool
    qualification: str = Field(max_length=40, examples=["GRAD"])
    area_type: Optional[str] = Field(default=None, max_length=40, description="only for pincodes not in the master")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts(DEFAULT_ARTIFACTS)            # model loaded once per worker
    audit_dir = os.environ.get("ABC_AUDIT_DIR") or os.path.join(os.path.dirname(DEFAULT_ARTIFACTS), "logs")
    app.state.audit = AuditLog(os.path.join(audit_dir, "audit.db"))
    LoanChatbot(audit=app.state.audit, channel="api")                  # warm-up: first prediction is the slow one
    yield
    app.state.audit.close()


app = FastAPI(title="ABC Credit scoring service", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    a = app.state.artifacts
    return {"status": "ok", "model_version": a["meta"]["model_version"], "model_sha256": a["sha256"],
            "threshold": a["meta"]["threshold"], "kill_switch": LoanChatbot.kill_switch_on()}


@app.get("/catalogue")
def catalogue():
    """Vehicle models the service recognises, with the typical on-road price used for sanity checks."""
    cat = app.state.artifacts["ref"]["vehicle_catalogue"]
    return [{"vehicle_model": k, "display": v["display"], "typical_price": v["typical_price"]} for k, v in cat.items()]


@app.post("/score")
def score(a: Application):
    bot = LoanChatbot(audit=app.state.audit, channel="api")            # light object; artifacts are cached
    out = bot.score_answers(a.model_dump(exclude_none=True))
    if not out["ok"]:
        raise HTTPException(status_code=422, detail={"session_id": out["session_id"], "errors": out["errors"]})
    return out
