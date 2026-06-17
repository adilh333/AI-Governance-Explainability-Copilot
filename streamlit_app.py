"""
AI Governance and Explainability Copilot - Streamlit App

This is the deployment entry point. It imports the explainability, fairness,
and RAG modules directly rather than calling the FastAPI backend, so the whole
application runs in a single process on Streamlit Community Cloud.

The FastAPI backend (backend/main.py) is retained in the repository as a
reference implementation of the same logic exposed as a REST API.

Run locally:
    streamlit run streamlit_app.py

Requires ANTHROPIC_API_KEY for narrative generation. On Streamlit Cloud this
is set via the app's Secrets manager. The SHAP and fairness analysis work
without it.
"""

import os
import sys

import pandas as pd
import streamlit as st

# Make the backend modules importable.
BACKEND = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, BACKEND)

import joblib
from ml.explainability import ModelExplainer
from ml.fairness import fairness_report
from rag.knowledge_base import GovernanceKnowledgeBase

# The RAG pipeline is imported lazily inside the function that needs it, so the
# app still loads if the anthropic package or API key is unavailable.

DATA_DIR = os.path.join(BACKEND, "data")

st.set_page_config(page_title="AI Governance Copilot", layout="wide")


# --- API key handling --------------------------------------------------------
def get_api_key():
    """Read the Anthropic API key from Streamlit secrets or environment."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


# --- Cached loaders -----------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(DATA_DIR, "model.pkl"))
    df = pd.read_csv(os.path.join(DATA_DIR, "loan_applications.csv"))
    explainer = ModelExplainer(model)
    kb = GovernanceKnowledgeBase()
    return model, df, explainer, kb


def generate_narrative(audit_data, question=None):
    """Call the RAG pipeline. Returns None if no API key is configured."""
    api_key = get_api_key()
    if not api_key:
        return None
    from rag.rag_pipeline import GovernanceRAGPipeline
    pipeline = GovernanceRAGPipeline(api_key=api_key)
    return pipeline.explain(audit_data, user_question=question)


# --- App ----------------------------------------------------------------------
st.title("AI Governance and Explainability Copilot")
st.caption(
    "Audits a loan-approval model using SHAP explainability and Fairlearn "
    "fairness metrics, then generates a plain-English governance narrative "
    "grounded in summarised AI governance frameworks (EU AI Act, NIST AI RMF) "
    "via a retrieval-augmented generation pipeline."
)

try:
    model, df, explainer, kb = load_artifacts()
except Exception as e:
    st.error(f"Could not load model or data: {e}")
    st.stop()

if not get_api_key():
    st.info(
        "Narrative generation is currently disabled because no Anthropic API "
        "key is configured. The SHAP feature importance, fairness metrics, and "
        "per-applicant explanations below are fully functional without it."
    )

tab1, tab2, tab3 = st.tabs(["Overview", "Applicant View", "Ask a Question"])

# --- Tab 1: Overview ---------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global feature importance")
        gi = explainer.global_importance(df)
        gi_df = pd.DataFrame(gi)
        st.bar_chart(gi_df.set_index("feature")["mean_abs_shap"])
        st.caption(
            "Mean absolute SHAP value per feature. Higher means the feature "
            "has more influence on the model's predictions overall."
        )

    with col2:
        st.subheader("Fairness metrics by group")
        fairness = fairness_report(model, df)
        st.dataframe(pd.DataFrame(fairness["by_group"]), use_container_width=True)
        st.metric("Demographic parity difference",
                  f"{fairness['demographic_parity_difference']:.3f}")
        st.metric("Equalized odds difference",
                  f"{fairness['equalized_odds_difference']:.3f}")
        for flag in fairness["flags"]:
            if "exceeds" in flag:
                st.warning(flag)
            else:
                st.success(flag)

    st.divider()
    st.subheader("Audit narrative")
    if st.button("Generate audit summary narrative"):
        with st.spinner("Generating narrative..."):
            result = generate_narrative({
                "global_importance": gi,
                "fairness": fairness,
            })
        if result:
            st.write(result["narrative"])
            with st.expander("Retrieved governance context"):
                for c in result["retrieved_context"]:
                    st.markdown(f"- **{c['source']}** - {c['heading']}")
        else:
            st.warning("Narrative generation requires an Anthropic API key.")

# --- Tab 2: Applicant view ----------------------------------------------------
with tab2:
    st.subheader("Inspect an individual applicant")
    idx = st.number_input("Applicant row index", min_value=0,
                          max_value=len(df) - 1, value=0, step=1)

    if st.button("Load applicant"):
        instance = explainer.explain_instance(df.iloc[int(idx)])
        instance["applicant_group"] = df.iloc[int(idx)]["applicant_group"]
        instance["actual_outcome"] = int(df.iloc[int(idx)]["approved"])
        st.session_state["instance"] = instance
        st.session_state["instance_idx"] = int(idx)

    instance = st.session_state.get("instance")
    if instance:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model decision",
                  "Approved" if instance["prediction"] else "Declined")
        c2.metric("Approval probability",
                  f"{instance['probability_approved']:.1%}")
        c3.metric("Actual outcome",
                  "Approved" if instance["actual_outcome"] else "Declined")

        st.write(f"Applicant group: **{instance['applicant_group']}**")

        contrib_df = pd.DataFrame(instance["contributions"])
        st.dataframe(contrib_df, use_container_width=True)
        st.bar_chart(contrib_df.set_index("feature")["shap_contribution"])
        st.caption(
            "Positive values pushed the prediction towards approval; negative "
            "values pushed it towards decline, for this applicant specifically."
        )

        st.divider()
        if st.button("Explain this decision in plain English"):
            with st.spinner("Generating explanation..."):
                result = generate_narrative({"instance": instance})
            if result:
                st.write(result["narrative"])
                with st.expander("Retrieved governance context"):
                    for c in result["retrieved_context"]:
                        st.markdown(f"- **{c['source']}** - {c['heading']}")
            else:
                st.warning("Explanation requires an Anthropic API key.")

# --- Tab 3: Free-text question ------------------------------------------------
with tab3:
    st.subheader("Ask a question about this audit")
    st.caption(
        "Questions are answered using the current audit summary and the "
        "governance knowledge base, not a general-purpose chatbot."
    )
    question = st.text_input(
        "Your question",
        placeholder="e.g. Why might the demographic parity difference matter here?",
    )
    if st.button("Ask") and question:
        gi = explainer.global_importance(df)
        fairness = fairness_report(model, df)
        with st.spinner("Thinking..."):
            result = generate_narrative(
                {"global_importance": gi, "fairness": fairness},
                question=question,
            )
        if result:
            st.write(result["narrative"])
            with st.expander("Retrieved governance context"):
                for c in result["retrieved_context"]:
                    st.markdown(f"- **{c['source']}** - {c['heading']}")
        else:
            st.warning("Answering questions requires an Anthropic API key.")
