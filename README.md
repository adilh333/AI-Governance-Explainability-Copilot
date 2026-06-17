# AI Governance and Explainability Copilot

An end-to-end tool that audits a machine learning model for fairness and explainability, then explains the results in plain English. It is built around a loan-approval model, but the underlying components work on any binary classifier and tabular dataset.

**Live demo:** https://ai-governance-explainability-copilot-jkhdniamzdjm9xunzcnum5.streamlit.app

## What problem this solves

Organisations that use AI to make decisions about people, such as approving loans or screening applications, increasingly need to demonstrate three things: that they can explain why the model made a given decision, that the model does not treat protected groups unfairly, and that a human can meaningfully review its behaviour. Regulations such as the EU AI Act now require this for high-risk systems.

This tool automates that audit. It examines a model, measures whether it is fair across groups, explains what drives its decisions, and produces a plain-English narrative grounded in real governance frameworks so that a non-technical reviewer can understand and act on the findings.

## Two views

The interface has a sidebar toggle between two views, so it serves both audiences:

- **Simple view** is written for non-technical reviewers and stakeholders. It opens with a clear verdict (for example, "This model has been flagged for a potential fairness concern"), translates every metric into plain language ("Group A is approved 16% more often than group B"), and avoids technical jargon.
- **Technical view** shows the underlying SHAP values, the exact Fairlearn metrics, and the governance passages retrieved for each narrative, for users who want the detail.

## How it works

The tool combines four components:

**1. Explainability (SHAP).** Calculates how much each feature (income, credit score, debt-to-income ratio, years employed) influences the model's decisions, both globally and for an individual applicant.

**2. Fairness (Fairlearn).** Measures whether the model treats two applicant groups equally, using demographic parity difference (are approval rates similar?) and equalized odds difference (are error rates similar?). Either metric exceeding a threshold raises a flag.

**3. Retrieval (RAG).** A set of plain-English summaries of governance frameworks (EU AI Act, NIST AI Risk Management Framework, and human-oversight principles) are searched using TF-IDF and cosine similarity to find the passages most relevant to the current findings.

**4. Generation (Claude).** The audit findings and the retrieved governance passages are sent to the Anthropic Claude API, which writes a clear narrative for a non-technical reviewer. The prompt explicitly separates the audit data from the retrieved context so the model explains findings without inventing legal conclusions.

A human-in-the-loop step lets a reviewer flag or approve generated explanations, reflecting how governance frameworks treat human oversight as an ongoing signal.

## Architecture

```
Streamlit frontend (streamlit_app.py)
        |
        | imports directly
        v
Backend modules (backend/)
  - ml/explainability.py    SHAP wrapper
  - ml/fairness.py          Fairlearn metrics
  - rag/knowledge_base.py   TF-IDF retrieval over governance documents
  - rag/rag_pipeline.py     Claude-based narrative generation
  - rag/documents/          Governance framework summaries (markdown)
  - data/                   Synthetic dataset and trained model
```

A FastAPI backend (`backend/main.py`) is retained as a reference implementation of the same logic exposed as a REST API. For the live deployment the logic runs in a single Streamlit process, since Streamlit Community Cloud runs one service.

## The dataset

The model and data are synthetic. The dataset is a generated loan-approval set with a deliberately injected bias against one group, so that the fairness component has a meaningful pattern to detect. This is a demonstration of the audit pipeline rather than a clinically or commercially validated system, and the fairness thresholds used are illustrative rather than legal determinations.

## Running locally

```bash
git clone https://github.com/adilh333/AI-Governance-Explainability-Copilot.git
cd AI-Governance-Explainability-Copilot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Regenerate the synthetic dataset and model (optional, already included)
python backend/data/generate_sample_data.py

streamlit run streamlit_app.py
```

Narrative generation requires an Anthropic API key. Set it as an environment variable or, when deploying to Streamlit Cloud, add it via the app's Secrets manager:

```
ANTHROPIC_API_KEY = "your-key-here"
```

The SHAP feature importance, fairness metrics, and per-applicant explanations all work without an API key; only the plain-English narrative generation requires it.

## Tech stack

Python, Streamlit, FastAPI, scikit-learn, SHAP, Fairlearn, Anthropic Claude API, pandas, NumPy.

## Possible extensions

- Swap TF-IDF retrieval for sentence-transformer embeddings and a vector index for a larger knowledge base.
- Support multiple protected attributes and intersectional fairness analysis.
- Allow users to upload their own dataset and trained model.
- Persist human-in-the-loop feedback and surface patterns over time.
