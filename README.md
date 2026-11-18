# ABC Credit Loan Approval Chatbot

PGPEM · Predictive and Generative AI · IIM Bangalore — course project.

**Group 10:** Sankalp Gharmalkar (2511228) · Vrushabh Nirwan (2511046) · Rushabh Bothara (2511341) · Amit Telang (2511488) ·
Yagneshkumar Ahir (2511333) · Joy Mondal (2516037) · Nitin Eshwar Harikant (2516039)

A trained model decides; a chatbot asks the questions and explains the outcome.

**Live app:** _paste your Streamlit link here after deploying_  ·  **Step-by-step guide:** `docs/ABC_Credit_Chatbot_Step_by_Step_Guide.pdf`

## Put it online in about 15 minutes (no coding)

1. Push this folder to a GitHub repository (GitHub Desktop: *Add Local Repository* → *Publish repository*).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → *Yup, I have an app*.
3. Repository: yours · Branch: `main` · Main file path: **`chatbot/app_streamlit.py`** → **Deploy**.
4. Open the link. The sidebar has two views: **Loan application chat** and **For the evaluator**
   (sample applicants, the audit trail of the session, and scoring of an uploaded test file).

Community Cloud installs `chatbot/requirements.txt` (pinned versions; tested on Python 3.12 and 3.13) and `packages.txt`.
The training data is not in the repository and is not needed by the app. Use made-up applicants only on a hosted demo.

## What is in this folder

| Path | Purpose |
|---|---|
| `ABC_Credit_Loan_Model.ipynb` | Executed notebook: data-quality audit, EDA, cleaning and feature pipeline, class-imbalance experiment, question selection, tuning, threshold, hold-out evaluation, saved artifacts, predictions |
| `abc_features.py` | Cleaning + feature engineering + encoding + model pipeline. Shared by the notebook, `predict.py` and the chatbot |
| `predict.py` | Batch scorer: `python predict.py --test <file.csv> --out predictions.csv` |
| `check_predictions.py` | Checks a predictions file before submission: `python check_predictions.py --pred predictions.csv --test <file.csv>` |
| `artifacts/` | `model_final.joblib` (pipeline), `model_meta.json` (threshold, version, questions, metrics), `reference_data.json` (vehicle catalogue, pincode → tier) |
| `chatbot/engine.py` | Slot-filling state machine: question flow, validation, model call, decision, reason codes |
| `chatbot/parsing.py` | Deterministic parsers ('70k', '1.2 lakh', fuzzy vehicle match, yes/no, options) |
| `chatbot/pii.py` | PII detection, masking, scrubbing, keyed pseudonymisation |
| `chatbot/audit.py` | Audit repository (SQLite + JSON Lines): sessions, turns, decisions |
| `chatbot/llm.py` | Optional Groq LLM layer: extraction fallback and explanation wording, with guardrails |
| `chatbot/cli.py`, `chatbot/app_streamlit.py` | Terminal chat and web chat |
| `chatbot/simulate.py` | Replays hold-out applications through the chatbot; latency; red-team script |
| `tests/test_project.py`, `tests/test_deploy.py` | 15 automated checks (model, chatbot, deployment kit) |
| `api.py` | Optional REST scoring service on the same engine |
| `extensions/langgraph_bot.py` | The chatbot driven by a LangGraph graph (course sessions 17-18) |
| `Dockerfile`, `docker-compose.yml`, `.env.example`, `.streamlit/config.toml`, `packages.txt`, `.github/workflows/ci.yml` | Deployment kit |
| `outputs/` | `results.json` (every number in the report), `predictions_holdout.csv`, simulation results, sample audit log |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r chatbot/requirements.txt                # pinned runtime: the versions the model was saved with
pip install -r requirements.txt                        # notebook, experiments and tests
pip install -r requirements-extras.txt                 # optional: REST API and LangGraph example
# put training_data.csv in data/
jupyter nbconvert --to notebook --execute --inplace ABC_Credit_Loan_Model.ipynb   # ~30 min on 2 CPUs; or run it in Jupyter

python -m chatbot.cli                       # terminal chat (templates only)
streamlit run chatbot/app_streamlit.py      # web chat on http://localhost:8501
python -m pytest -q tests                   # 15 checks (2 are skipped without the optional extras)
python -m chatbot.simulate 500              # consistency, latency and red-team evidence
python predict.py --test hidden_test.csv --out predictions.csv
python check_predictions.py --pred predictions.csv --test hidden_test.csv
```

## Configuration (environment variables, a `.env` file, or Streamlit secrets)

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Switches on the optional LLM interaction layer | unset: templates only |
| `ABC_LLM_MODEL` | Groq model ID for that layer | `openai/gpt-oss-20b` |
| `ABC_PII_KEY` | Secret for phone-number tokens (set it in any real deployment) | a demo key |
| `ABC_AUDIT_DIR` | Folder for the audit repository (a mounted volume in Docker) | `./logs` |
| `ABC_KILL_SWITCH` | `1` stops automated decisions and refers every application to a loan officer | off |

## Deploy

| Route | Files | Command |
|---|---|---|
| Streamlit Community Cloud | `chatbot/requirements.txt`, `packages.txt`, `.streamlit/config.toml` | Create app -> main file `chatbot/app_streamlit.py`, Python 3.13, paste secrets |
| Docker | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.env.example` | `docker compose up --build` |
| Render (fallback host) | `render.yaml`, `Dockerfile` | New → Blueprint → pick the repository (free plan sleeps after 15 minutes idle) |
| REST API | `api.py`, `sample_application.json` | `uvicorn api:app --port 8000` |
| LangGraph example | `extensions/langgraph_bot.py` | `python -m extensions.langgraph_bot` |

The step-by-step guide in `docs/` covers the hosted route click by click. The tutorial (`docs/ABC_Credit_Chatbot_Tutorial.pdf`) walks through each route in depth; the delivery guide (`docs/ABC_Credit_Chatbot_Delivery_Guide.pdf`) gives the stage-by-stage plan, gates, slide outline and demo script.

## Design rules

* The credit decision is computed in exactly one place (`LoanChatbot.score`): `DECLINED if p >= threshold else APPROVED`.
  The language model never sees the score or the threshold and cannot change the decision.
* Everything fitted (encoders, model) lives inside one scikit-learn `Pipeline`; cleaning is part of it.
* The class imbalance (5.3% DECLINED) is handled by stratified splitting, imbalance-aware metrics, an in-fold comparison of
  seven training strategies, calibrated probabilities and a cost-sensitive decision threshold.
* No name, Aadhaar, PAN or gender is collected. Free text is scrubbed before logging or any LLM call. The pincode is stored as its
  three-digit district. Logs contain everything needed to reproduce a decision and nothing that identifies the person by itself.
