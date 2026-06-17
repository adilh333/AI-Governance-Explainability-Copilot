"""
AI Governance and Explainability Copilot - Streamlit App

Deployment entry point. Imports the explainability, fairness, and RAG modules
directly so the whole application runs in a single process on Streamlit Cloud.

Offers two views via a sidebar toggle:
  - Simple view: plain-language verdict and translations for non-technical
    reviewers and stakeholders.
  - Technical view: the underlying SHAP values and fairness metrics.

Requires ANTHROPIC_API_KEY (set via Streamlit secrets) for narrative
generation. SHAP and fairness analysis work without it.
"""

import os
import sys

import pandas as pd
import streamlit as st

BACKEND = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, BACKEND)

import joblib
from ml.explainability import ModelExplainer
from ml.fairness import fairness_report
from rag.knowledge_base import GovernanceKnowledgeBase

DATA_DIR = os.path.join(BACKEND, "data")

st.set_page_config(page_title="AI Governance Copilot", layout="wide")

FEATURE_LABELS = {
    "income": "Income",
    "credit_score": "Credit score",
    "debt_to_income": "Debt-to-income ratio",
    "employment_years": "Years employed",
}

FAIRNESS_THRESHOLD = 0.10


def get_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(DATA_DIR, "model.pkl"))
    df = pd.read_csv(os.path.join(DATA_DIR, "loan_applications.csv"))
    explainer = ModelExplainer(model)
    kb = GovernanceKnowledgeBase()
    return model, df, explainer, kb


def generate_narrative(audit_data, question=None):
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        from rag.rag_pipeline import GovernanceRAGPipeline
        pipeline = GovernanceRAGPipeline(api_key=api_key)
        return pipeline.explain(audit_data, user_question=question)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def fairness_verdict(fairness):
    dpd = fairness["demographic_parity_difference"]
    eod = fairness["equalized_odds_difference"]
    flagged = abs(dpd) > FAIRNESS_THRESHOLD or abs(eod) > FAIRNESS_THRESHOLD
    if flagged:
        return (
            "warning", "ORANGE",
            "This model has been flagged for a potential fairness concern.",
            "The model treats the two applicant groups noticeably differently. "
            "A human reviewer should examine this before the model is relied on "
            "for decisions.",
        )
    return (
        "ok", "GREEN",
        "No fairness concerns were flagged.",
        "The two applicant groups are treated similarly within the threshold "
        "set for this review.",
    )


def plain_fairness_lines(fairness):
    lines = []
    by_group = {g["group"]: g for g in fairness["by_group"]}
    groups = list(by_group.keys())
    if len(groups) == 2:
        g1, g2 = groups
        r1 = by_group[g1]["selection_rate"]
        r2 = by_group[g2]["selection_rate"]
        higher, lower = (g1, g2) if r1 >= r2 else (g2, g1)
        gap = abs(r1 - r2)
        lines.append(
            f"Group {higher} is approved {gap:.0%} more often than group {lower}. "
            f"A gap above {FAIRNESS_THRESHOLD:.0%} is flagged for review."
        )
    eod = fairness["equalized_odds_difference"]
    lines.append(
        f"The model's error rates differ by {abs(eod):.0%} between groups. "
        f"A difference above {FAIRNESS_THRESHOLD:.0%} is flagged for review."
    )
    return lines


st.title("AI Governance and Explainability Copilot")

try:
    model, df, explainer, kb = load_artifacts()
except Exception as e:
    st.error(f"Could not load model or data: {e}")
    st.stop()

with st.sidebar:
    st.header("View")
    view = st.radio(
        "Choose how much detail to show:",
        ["Simple view", "Technical view"],
        help="Simple view explains everything in plain language. "
             "Technical view shows the underlying SHAP and fairness metrics.",
    )
    st.caption(
        "This tool audits a loan-approval model for fairness and "
        "explainability, then explains the results."
    )

simple = view == "Simple view"

if simple:
    st.caption(
        "This tool checks an automated loan-approval model: is it fair to "
        "different groups of people, and can we explain the decisions it makes? "
        "Everything below is written in plain language."
    )
else:
    st.caption(
        "Audits a loan-approval model using SHAP explainability and Fairlearn "
        "fairness metrics, with a retrieval-augmented generation pipeline "
        "grounding plain-English narratives in governance frameworks "
        "(EU AI Act, NIST AI RMF)."
    )

if not get_api_key():
    st.info(
        "Plain-English narrative generation is currently disabled (no API key "
        "configured). All charts and metrics below still work."
    )

tab1, tab2, tab3 = st.tabs(["Overview", "Applicant View", "Ask a Question"])

with tab1:
    fairness = fairness_report(model, df)
    gi = explainer.global_importance(df)

    if simple:
        status, icon, headline, detail = fairness_verdict(fairness)
        badge = "[FLAGGED]" if status == "warning" else "[OK]"
        st.subheader(f"{badge} {headline}")
        st.write(detail)
        st.divider()

        st.markdown("### Is the model fair?")
        for line in plain_fairness_lines(fairness):
            if status == "warning":
                st.warning(line)
            else:
                st.success(line)
        st.caption(
            "Fairness here means: does the model approve people from different "
            "groups at similar rates, and does it make mistakes at similar rates?"
        )

        st.markdown("### What matters most to this model?")
        gi_df = pd.DataFrame(gi)
        gi_df["Factor"] = gi_df["feature"].map(FEATURE_LABELS).fillna(gi_df["feature"])
        top = gi_df.iloc[0]
        st.write(
            f"The single biggest factor in this model's decisions is "
            f"**{top['Factor']}**. The chart below ranks every factor by how "
            f"much it influences whether someone is approved."
        )
        st.bar_chart(gi_df.set_index("Factor")["mean_abs_shap"])
        st.caption(
            "Longer bars mean the factor has more influence on the model's "
            "decisions, in either direction."
        )

        st.divider()
        st.markdown("### What should a reviewer do about this?")
        if st.button("Generate plain-English summary"):
            with st.spinner("Writing summary..."):
                result = generate_narrative({"global_importance": gi, "fairness": fairness})
            if result and "error" in result:
                st.error(f"Summary generation failed - {result['error']}")
            elif result:
                st.write(result["narrative"])
            else:
                st.warning("This feature needs an Anthropic API key.")

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Global feature importance (SHAP)")
            gi_df = pd.DataFrame(gi)
            st.bar_chart(gi_df.set_index("feature")["mean_abs_shap"])
            st.caption("Mean absolute SHAP value per feature across a sample.")
        with col2:
            st.subheader("Fairness metrics (Fairlearn)")
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
                result = generate_narrative({"global_importance": gi, "fairness": fairness})
            if result and "error" in result:
                st.error(f"Narrative generation failed - {result['error']}")
            elif result:
                st.write(result["narrative"])
                with st.expander("Retrieved governance context"):
                    for c in result["retrieved_context"]:
                        st.markdown(f"- **{c['source']}** - {c['heading']}")
            else:
                st.warning("Narrative generation requires an Anthropic API key.")

with tab2:
    st.subheader("Inspect an individual applicant")
    idx = st.number_input("Applicant row number", min_value=0,
                          max_value=len(df) - 1, value=0, step=1)

    if st.button("Load applicant"):
        instance = explainer.explain_instance(df.iloc[int(idx)])
        instance["applicant_group"] = df.iloc[int(idx)]["applicant_group"]
        instance["actual_outcome"] = int(df.iloc[int(idx)]["approved"])
        st.session_state["instance"] = instance

    instance = st.session_state.get("instance")
    if instance:
        decision = "Approved" if instance["prediction"] else "Declined"
        if simple:
            st.markdown(f"### This applicant was: **{decision}**")
            st.write(
                f"The model was {instance['probability_approved']:.0%} confident "
                f"this applicant should be approved."
            )
            st.markdown("#### Why?")
            contrib = sorted(instance["contributions"],
                             key=lambda c: abs(c["shap_contribution"]), reverse=True)
            for c in contrib:
                factor = FEATURE_LABELS.get(c["feature"], c["feature"])
                direction = "towards approval" if c["shap_contribution"] > 0 else "towards decline"
                st.write(f"- **{factor}** pushed the decision {direction}.")
            st.caption(
                "These are the factors that most influenced this specific "
                "decision, in order of importance."
            )
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Model decision", decision)
            c2.metric("Approval probability", f"{instance['probability_approved']:.1%}")
            c3.metric("Actual outcome", "Approved" if instance["actual_outcome"] else "Declined")
            st.write(f"Applicant group: **{instance['applicant_group']}**")
            contrib_df = pd.DataFrame(instance["contributions"])
            st.dataframe(contrib_df, use_container_width=True)
            st.bar_chart(contrib_df.set_index("feature")["shap_contribution"])
            st.caption("Positive values pushed towards approval; negative towards decline.")

        st.divider()
        label = "Explain this decision" if simple else "Explain this decision in plain English"
        if st.button(label):
            with st.spinner("Generating explanation..."):
                result = generate_narrative({"instance": instance})
            if result and "error" in result:
                st.error(f"Explanation failed - {result['error']}")
            elif result:
                st.write(result["narrative"])
                if not simple:
                    with st.expander("Retrieved governance context"):
                        for c in result["retrieved_context"]:
                            st.markdown(f"- **{c['source']}** - {c['heading']}")
            else:
                st.warning("This feature needs an Anthropic API key.")

with tab3:
    st.subheader("Ask a question about this audit")
    if simple:
        st.caption("Ask anything about the model and its fairness in plain language.")
    else:
        st.caption("Answered using the current audit summary and the governance "
                   "knowledge base, not a general-purpose chatbot.")
    question = st.text_input(
        "Your question",
        placeholder="e.g. Why does it matter if the two groups are approved at different rates?",
    )
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            result = generate_narrative(
                {"global_importance": gi, "fairness": fairness}, question=question)
        if result and "error" in result:
            st.error(f"Question answering failed - {result['error']}")
        elif result:
            st.write(result["narrative"])
            if not simple:
                with st.expander("Retrieved governance context"):
                    for c in result["retrieved_context"]:
                        st.markdown(f"- **{c['source']}** - {c['heading']}")
        else:
            st.warning("This feature needs an Anthropic API key.")
