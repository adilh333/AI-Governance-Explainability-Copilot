"""
AI Governance and Explainability Copilot - API

Endpoints
---------
GET  /health                  Liveness check.
GET  /audit/summary           Global feature importance + fairness report
                               for the loaded sample model and dataset.
GET  /audit/instance/{idx}    SHAP explanation for a single row of the
                               holdout set, identified by its index.
POST /explain                 Send an audit payload (and optional reviewer
                               question) to the RAG pipeline and get back a
                               plain-English narrative grounded in the
                               governance knowledge base.
POST /feedback                Human-in-the-loop endpoint: store a reviewer's
                               flag/comment on a given explanation.

Run with:
    uvicorn main:app --reload --port 8000

Requires ANTHROPIC_API_KEY to be set for the /explain endpoint. The other
endpoints work without it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.explainability import ModelExplainer
from ml.fairness import fairness_report
from rag.rag_pipeline import GovernanceRAGPipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback_log.jsonl")

app = FastAPI(title="AI Governance and Explainability Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model and data once at startup -----------------------------------
_model = joblib.load(os.path.join(DATA_DIR, "model.pkl"))
_df = pd.read_csv(os.path.join(DATA_DIR, "loan_applications.csv"))
_explainer = ModelExplainer(_model)

# RAG pipeline is created lazily so the API still starts without an API key.
_rag_pipeline: GovernanceRAGPipeline | None = None


def get_rag_pipeline() -> GovernanceRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="ANTHROPIC_API_KEY is not set. See .env.example.",
            )
        _rag_pipeline = GovernanceRAGPipeline()
    return _rag_pipeline


# --- Schemas -----------------------------------------------------------------
class ExplainRequest(BaseModel):
    audit_data: dict
    question: str | None = None


class FeedbackRequest(BaseModel):
    explanation_id: str
    flagged: bool
    comment: str | None = None


# --- Endpoints ----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": len(_df)}


@app.get("/audit/summary")
def audit_summary():
    return {
        "global_importance": _explainer.global_importance(_df),
        "fairness": fairness_report(_model, _df),
        "n_rows": len(_df),
    }


@app.get("/audit/instance/{idx}")
def audit_instance(idx: int):
    if idx < 0 or idx >= len(_df):
        raise HTTPException(status_code=404, detail="Index out of range")
    row = _df.iloc[idx]
    explanation = _explainer.explain_instance(row)
    explanation["applicant_group"] = row["applicant_group"]
    explanation["actual_outcome"] = int(row["approved"])
    explanation["row_index"] = idx
    return explanation


@app.post("/explain")
def explain(req: ExplainRequest):
    pipeline = get_rag_pipeline()
    return pipeline.explain(req.audit_data, user_question=req.question)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "explanation_id": req.explanation_id,
        "flagged": req.flagged,
        "comment": req.comment,
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "recorded"}
