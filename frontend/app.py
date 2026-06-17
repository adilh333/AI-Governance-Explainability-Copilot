"""
AI Governance and Explainability Copilot - Frontend

A Streamlit dashboard that talks to the FastAPI backend. Three tabs:

1. Overview        - global feature importance and fairness metrics, with
                      an LLM-generated narrative summary.
2. Applicant View   - per-applicant SHAP explanation, with an LLM narrative
                      and a human-in-the-loop feedback control.
3. Ask a Question   - free-text question answered by the RAG pipeline,
                      grounded in the current audit data.

Run with:
    streamlit run app.py

Expects the backend to be running at BACKEND_URL (default localhost:8000).
"""

import os
import uuid

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Governance Copilot", layout="wide")
st.title("AI Governance and Explainability Copilot")
st.caption(
    "Demo project: explains a loan-approval model's behaviour using SHAP and "
    "fairness metrics, then generates a plain-English audit narrative "
    "grounded in summarised AI governance frameworks via a RAG pipeline."
)


@st.cache_data(ttl=60)
def get_summary():
    r = requests.get(f"{BACKEND_URL}/audit/summary", timeout=30)
    r.raise_for_status()
    return r.json()


def get_instance(idx: int):
    r = requests.get(f"{BACKEND_URL}/audit/instance/{idx}", timeout=30)
    r.raise_for_status()
    return r.json()


def explain(audit_data: dict, question: str | None = None):
    r = requests.post(
        f"{BACKEND_URL}/explain",
        json={"audit_data": audit_data, "question": question},
        timeout=60,
    )
    if r.status_code == 503:
        st.warning(
            "ANTHROPIC_API_KEY is not set on the backend, narrative "
            "generation is unavailable. The structured audit data above "
            "is still fully functional without it."
        )
        return None
    r.raise_for_status()
    return r.json()


def send_feedback(explanation_id: str, flagged: bool, comment: str):
    requests.post(
        f"{BACKEND_URL}/feedback",
        json={"explanation_id": explanation_id, "flagged": flagged, "comment": comment},
        timeout=10,
    )


tab1, tab2, tab3 = st.tabs(["Overview", "Applicant View", "Ask a Question"])

# --- Tab 1: Overview ---------------------------------------------------------
with tab1:
    try:
        summary = get_summary()
    except requests.RequestException as e:
        st.error(f"Could not reach backend at {BACKEND_URL}. Is it running? ({e})")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global feature importance")
        gi_df = pd.DataFrame(summary["global_importance"])
        st.bar_chart(gi_df.set_index("feature")["mean_abs_shap"])
        st.caption(
            "Mean absolute SHAP value per feature across a sample of "
            "applicants. Higher means the feature has more influence on "
            "the model's predictions, in either direction."
        )

    with col2:
        st.subheader("Fairness metrics by group")
        fairness = summary["fairness"]
        by_group_df = pd.DataFrame(fairness["by_group"])
        st.dataframe(by_group_df, use_container_width=True)
        st.metric(
            "Demographic parity difference",
            f"{fairness['demographic_parity_difference']:.3f}",
        )
        st.metric(
            "Equalized odds difference",
            f"{fairness['equalized_odds_difference']:.3f}",
        )
        for flag in fairness["flags"]:
            if "exceeds" in flag:
                st.warning(flag)
            else:
                st.success(flag)

    st.divider()
    st.subheader("Audit narrative")
    if st.button("Generate audit summary narrative"):
        with st.spinner("Generating narrative..."):
            result = explain(
                {
                    "global_importance": summary["global_importance"],
                    "fairness": summary["fairness"],
                }
            )
        if result:
            st.write(result["narrative"])
            with st.expander("Retrieved governance context"):
                for c in result["retrieved_context"]:
                    st.markdown(f"- **{c['source']}** — {c['heading']}")

# --- Tab 2: Applicant view ----------------------------------------------------
with tab2:
    st.subheader("Inspect an individual applicant")
    idx = st.number_input("Applicant row index", min_value=0, max_value=1999, value=0, step=1)

    if st.button("Load applicant"):
        instance = get_instance(int(idx))
        st.session_state["instance"] = instance

    instance = st.session_state.get("instance")
    if instance:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model decision", "Approved" if instance["prediction"] else "Declined")
        c2.metric("Approval probability", f"{instance['probability_approved']:.2%}")
        c3.metric("Actual outcome", "Approved" if instance["actual_outcome"] else "Declined")

        st.write(f"Applicant group: **{instance['applicant_group']}**")

        contrib_df = pd.DataFrame(instance["contributions"])
        st.dataframe(contrib_df, use_container_width=True)
        st.bar_chart(contrib_df.set_index("feature")["shap_contribution"])
        st.caption(
            "Positive values pushed the prediction towards approval; "
            "negative values pushed it towards decline, for this "
            "applicant specifically."
        )

        st.divider()
        if st.button("Explain this decision in plain English"):
            with st.spinner("Generating explanation..."):
                result = explain({"instance": instance})
            if result:
                st.session_state["last_explanation"] = result
                st.write(result["narrative"])
                with st.expander("Retrieved governance context"):
                    for c in result["retrieved_context"]:
                        st.markdown(f"- **{c['source']}** — {c['heading']}")

        # Human-in-the-loop feedback
        if "last_explanation" in st.session_state:
            st.divider()
            st.subheader("Reviewer feedback")
            comment = st.text_area("Comment (optional)")
            fcol1, fcol2 = st.columns(2)
            if fcol1.button("Looks correct"):
                send_feedback(str(uuid.uuid4()), flagged=False, comment=comment)
                st.success("Feedback recorded. Thank you.")
            if fcol2.button("Flag this explanation"):
                send_feedback(str(uuid.uuid4()), flagged=True, comment=comment)
                st.success("Flag recorded. Thank you.")

# --- Tab 3: Free-text question -------------------------------------------------
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
        try:
            summary = get_summary()
        except requests.RequestException as e:
            st.error(f"Could not reach backend: {e}")
        else:
            with st.spinner("Thinking..."):
                result = explain(
                    {
                        "global_importance": summary["global_importance"],
                        "fairness": summary["fairness"],
                    },
                    question=question,
                )
            if result:
                st.write(result["narrative"])
                with st.expander("Retrieved governance context"):
                    for c in result["retrieved_context"]:
                        st.markdown(f"- **{c['source']}** — {c['heading']}")
