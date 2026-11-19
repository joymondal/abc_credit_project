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
| `ABC_Credit_Loan_Model.ipynb` | Executed notebook: summary statistics and distributions, data-quality audit, EDA with statistical tests, cleaning and feature pipeline, class-imbalance experiment, question selection, tuning, threshold, hold-out evaluation, saved artifacts, predictions, checks against alternative approaches (Section 17) |
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
| `chatbot/personas.py` | Behaviour benchmark: 26 made-up applicants (six profiles at four loan sizes, two outside the age policy) and seven checks; also a tab in the evaluator view |
| `peer_checks.py` | Alternative ideas tested under our own split and folds: XGBoost and CatBoost, 22 extra hand-made features, monotonic constraints, synthetic rows before the split. Results in `outputs/peer_checks.json` and notebook Section 17 |
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
pip install -r requirements-extras.txt                 # optional: REST API, LangGraph example, XGBoost and CatBoost for peer_checks.py
# put training_data.csv in data/
jupyter nbconvert --to notebook --execute --inplace ABC_Credit_Loan_Model.ipynb   # ~30 min on 2 CPUs; or run it in Jupyter

python -m chatbot.cli                       # terminal chat (templates only)
streamlit run chatbot/app_streamlit.py      # web chat on http://localhost:8501
python -m pytest -q tests                   # 18 checks (3 are skipped without the optional extras)
python -m chatbot.simulate 500              # consistency, latency and red-team evidence
python -m chatbot.personas                  # 26 made-up applicants, 7 behaviour checks (no training data needed)
python peer_checks.py all                   # optional, ~4 min: needs data/training_data.csv and the extras (xgboost, catboost)
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

## Choosing the model (new by default, older on request)

Two models are installed. The **new** model is used everywhere unless another is chosen.

| | New (default) | Older |
|---|---|---|
| Key | `current` | `previous` |
| Version | `abc-credit-lgbm-2026-11-19` | `abc-credit-lgbm-2026-11-13` |
| Folder | `artifacts/` | `artifacts/previous/` |
| Cost ratio, cut-off | 6, 0.143 | 7, 0.125 |
| Questions | 8, no income | 9, asks monthly income |

| Where | How to choose |
|---|---|
| Web app, chat and evaluator views | **Model** selector in the sidebar. Switching starts a new application; every evaluator tab (samples, audit trail, persona check, test-file scoring) then uses the selected model |
| Default for a whole deployment | Environment variable or Streamlit secret `ABC_MODEL=previous` (unset = new model) |
| REST API | `POST /score?model=previous` (the body then needs `net_salary`); `GET /health` lists the models |
| Files | `python predict.py --test f.csv --out p.csv --model previous`, then `python check_predictions.py --pred p.csv --model previous` |
| Persona benchmark | `python -m chatbot.personas --model previous` |

The questions, the cut-off and the reasons come from the selected model's `model_meta.json`, and every decision is logged with the model version and file hash that made it. A test file without a `Net_salary` column is scored by both models: the new one ignores income, the older one treats it as missing. The report, the notebook and `predictions_holdout.csv` describe the new model only.

## Current model (19 Nov 2026)

`abc-credit-lgbm-2026-11-19`: the chatbot asks **eight** questions and the model uses **19** features. Stated monthly income is neither asked nor used:
in the question-selection test it added about +0.0001 PR-AUC, it is unverified, and it is the most sensitive question (notebook Section 9).
The `Net_salary` column may still be present in a test file; the pipeline ignores it. The cost ratio is **R = 6**, so the cut-off is 1 / (1 + 6) = **0.143**.
On the 20% hold-out: ROC-AUC 0.790, PR-AUC 0.186; at the cut-off, accuracy 90.0%, precision 20.7%, recall 31.7% on the DECLINED class.

## Checks against alternative approaches

Section 17 of the notebook tests four ideas another solution to the same brief might use, with the same split, folds and seed:

| Idea | Result | Decision |
|---|---|---|
| Add synthetic DECLINED rows before the train/test split | The same pipeline reports ROC-AUC 0.94 and PR-AUC 0.88 on the mixed test set, but 0.78 and 0.15 on its real rows (ours: 0.79 and 0.19) | Not used: the score is inflated by near-copies in the test set |
| XGBoost and CatBoost instead of LightGBM | ROC-AUC within 0.001; PR-AUC 0.178 and 0.174 against 0.179 | LightGBM kept |
| 22 extra hand-made flags and ratios (income ratios included) | PR-AUC 0.1785 against 0.1789 | Not added |
| Monotonic constraint on LTV | No loss (PR-AUC 0.1788 against 0.1789). Without it, borrowing more on the same vehicle turned a decline into an approval for 1 of 2,000 hold-out applicants | Model unchanged; add the constraint at the next retraining |

## Design rules

* The credit decision is computed in exactly one place (`LoanChatbot.score`): `DECLINED if p >= threshold else APPROVED`.
  The language model never sees the score or the threshold and cannot change the decision.
* Everything fitted (encoders, model) lives inside one scikit-learn `Pipeline`; cleaning is part of it.
* The class imbalance (5.3% DECLINED) is handled by stratified splitting, imbalance-aware metrics, an in-fold comparison of
  seven training strategies, calibrated probabilities and a cost-sensitive decision threshold.
* No name, Aadhaar, PAN or gender is collected. Free text is scrubbed before logging or any LLM call. The pincode is stored as its
  three-digit district. Logs contain everything needed to reproduce a decision and nothing that identifies the person by itself.
