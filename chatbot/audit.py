"""Audit repository: every input, answer, feature value and model score (objective d).

Two synchronised sinks: a SQLite database (queryable) and an append-only JSON-Lines file.
Only masked / scrubbed values are ever written here (see pii.py).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT, status TEXT, channel TEXT,
  consent INTEGER, notice_version TEXT, questions_asked INTEGER, reasks INTEGER,
  duration_s REAL, llm_enabled INTEGER, llm_model TEXT);
CREATE TABLE IF NOT EXISTS turns (
  session_id TEXT, turn_no INTEGER, ts TEXT, slot TEXT, bot_prompt TEXT,
  user_input_masked TEXT, parsed_value_masked TEXT, valid INTEGER, error TEXT,
  parse_source TEXT, latency_ms REAL);
CREATE TABLE IF NOT EXISTS decisions (
  session_id TEXT, ts TEXT, decision_source TEXT, model_version TEXT, model_sha256 TEXT,
  model_input_json TEXT, engineered_features_json TEXT, model_vector_json TEXT,
  score REAL, threshold REAL, decision TEXT, reason_codes_json TEXT,
  feature_ms REAL, scoring_ms REAL, explanation_ms REAL, explanation_source TEXT,
  decision_latency_ms REAL);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditLog:
    def __init__(self, db_path="logs/audit.db", jsonl_path=None):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.jsonl_path = jsonl_path or os.path.splitext(db_path)[0] + ".jsonl"
        self._lock = threading.Lock()
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.executescript(SCHEMA)
        self._con.commit()

    def _write(self, table: str, row: dict):
        cols = ",".join(row)
        with self._lock:
            self._con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({','.join('?' * len(row))})",
                              [json.dumps(v) if isinstance(v, (dict, list)) else v for v in row.values()])
            self._con.commit()
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"table": table, **row}, default=str) + "\n")

    def session(self, **row):
        self._write("sessions", row)

    def turn(self, **row):
        self._write("turns", row)

    def decision(self, **row):
        self._write("decisions", row)

    def query(self, sql: str, params=()):
        with self._lock:
            cur = self._con.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self):
        self._con.close()
