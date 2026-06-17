# AI Governance and Explainability Copilot

A small end-to-end system that audits a predictive model for explainability
and fairness, then uses a retrieval-augmented generation (RAG) pipeline to
turn those technical outputs into a plain-English narrative grounded in
summarised AI governance frameworks (EU AI Act, NIST AI RMF, and general
human-oversight principles).

## Motivation

Organisations deploying predictive models, particularly in regulated
contexts such as credit scoring, increasingly need to demonstrate not just
that a model is accurate, but that its behaviour is explainable, that it
does not produce materially different outcomes across protected groups, and
that a human reviewer can meaningfully oversee its decisions.

This project investigates how those three concerns, explainability, fairness
measurement, and governance-grounded communication, can be brought together
in a single tool. The model and dataset are synthetic (a loan-approval
classifier on generated data with a deliberately injected group-level bias),
but the explainability, fairness, and RAG components are written to operate
on any binary classifier and tabular dataset with minimal changes.

## Architecture

```
┌─────────────────┐      ┌──────────────────────────────────────┐
│   Streamlit      │      │              FastAPI backend           │
│   frontend        │◄────►│                                        │
│                   │ HTTP │  ┌────────────┐   ┌──────────────────┐ │
│  - Overview tab   │      │  │ SHAP        │   │ Fairlearn         │ │
│  - Applicant view │      │  │ explainer   │   │ fairness metrics  │ │
│  - Q&A tab        │      │  └────────────┘   └──────────────────┘ │
│  - feedback UI    │      │            │              │             │
└──────────────────┘      │            ▼              ▼             │
                            │     ┌─────────────────────────┐        │
                            │     │   RAG pipeline            │        │
                            │     │  - TF-IDF retrieval over   │        │
                            │     │    governance docs (.md)   │        │
                            │     │  - Claude generates         │        │
                            │     │    grounded narrative       │        │
                            │     └─────────────────────────┘        │
                            └──────────────────────────────────────┘
```

### Why these specific design choices

**TF-IDF retrieval instead of embeddings.** The knowledge base is a handful
of curated markdown documents. TF-IDF + cosine similarity needs no external
model downloads or API calls, is fully deterministic, and is easy to debug
by printing the matched chunks. The `GovernanceKnowledgeBase` class is the
only place that would need to change to swap this for sentence-transformer
embeddings and a FAISS index, the rest of the pipeline (chunking interface,
`retrieve()` signature) would stay the same.

**Why the prompt explicitly separates "audit data" from "retrieved context".**
A common failure mode for RAG over policy documents is the model blending
retrieved framing with the specific case being discussed, producing
authoritative-sounding claims that neither source actually supports. The
system prompt instructs the model to keep these separate and to flag its own
limitations, which matters more in a governance context than in a typical
RAG Q&A use case.

**Why fairness flags use illustrative thresholds, not "the law".** The
fairness module computes real metrics (demographic parity difference,
equalized odds difference) using Fairlearn, but deliberately does not claim
that crossing a threshold means a law has been broken. The governance
documents make this distinction explicit, and the system prompt enforces it.

**Why there's a feedback loop.** The NIST AI RMF "Manage" function and the
human-oversight notes both treat reviewer feedback as a measurement signal in
its own right. The `/feedback` endpoint and the Streamlit "flag this
explanation" control are a minimal implementation of that idea, in a larger
system this log would feed back into model or prompt review.

## Project layout

```
ai-governance-copilot/
├── backend/
│   ├── main.py                  FastAPI app and endpoints
│   ├── ml/
│   │   ├── explainability.py     SHAP wrapper
│   │   └── fairness.py           Fairlearn-based fairness report
│   ├── rag/
│   │   ├── knowledge_base.py     TF-IDF retrieval over governance docs
│   │   ├── rag_pipeline.py       Claude-based narrative generation
│   │   └── documents/            Governance framework summaries (.md)
│   └── data/
│       ├── generate_sample_data.py  Synthetic dataset + model training
│       ├── loan_applications.csv    Generated dataset
│       └── model.pkl                 Trained RandomForestClassifier
├── frontend/
│   └── app.py                   Streamlit dashboard
├── .env.example
└── README.md
```

## Running locally

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Regenerate the sample dataset and model (already included, but
# re-running is useful if you tweak generate_sample_data.py)
python data/generate_sample_data.py

cp ../.env.example ../.env   # then edit .env and add your ANTHROPIC_API_KEY
export $(grep -v '^#' ../.env | xargs)

uvicorn main:app --reload --port 8000
```

### 2. Frontend

In a second terminal:

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501` and will talk to
the backend at `http://localhost:8000`.

Without an `ANTHROPIC_API_KEY` set, the SHAP and fairness analysis, charts,
and applicant-level views all work normally; only the "explain in plain
English" narrative generation requires the key.

## Adapting this to a real dataset and model

1. Replace `data/loan_applications.csv` and `data/model.pkl` with your own
   data and a trained `scikit-learn`-compatible classifier (anything with
   `.predict` and `.predict_proba`).
2. Update `FEATURE_COLS` and the protected attribute name in
   `ml/explainability.py` and `ml/fairness.py`.
3. Add or edit the markdown files in `rag/documents/` to reflect the
   governance frameworks relevant to your domain.

## Possible extensions

- Swap TF-IDF retrieval for sentence-transformer embeddings + FAISS for a
  larger knowledge base.
- Add authentication and per-reviewer feedback history.
- Containerise with Docker and deploy the backend and frontend separately
  (e.g. Render, Railway, or AWS).
- Extend the fairness module to support multiple protected attributes
  simultaneously and intersectional analysis.
