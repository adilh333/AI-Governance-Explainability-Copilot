"""
RAG generation pipeline.

Takes structured outputs from the explainability and fairness modules,
retrieves relevant governance context, and prompts an LLM (Anthropic Claude)
to produce a plain-English audit narrative. The prompt explicitly instructs
the model to ground its explanation in the retrieved context and to
distinguish between the model's quantitative outputs and the governance
framework's framing of those outputs, the goal is to avoid the LLM inventing
authoritative-sounding claims that go beyond either source.
"""

from __future__ import annotations

import json
import os

from anthropic import Anthropic

from .knowledge_base import GovernanceKnowledgeBase

SYSTEM_PROMPT = """\
You are an AI governance assistant. You help a human reviewer understand the \
outputs of a model explainability and fairness audit tool.

Rules:
1. Base your explanation only on the provided AUDIT DATA (model outputs) and \
RETRIEVED CONTEXT (governance framework excerpts). Do not invent statistics, \
legal conclusions, or facts not present in either source.
2. Clearly separate (a) what the audit data shows, from (b) how the retrieved \
governance context frames that kind of finding. Do not present (b) as a \
definitive legal judgement about this specific model.
3. Write for a non-technical reviewer. Avoid raw jargon like "SHAP value" \
without a one-line explanation of what it represents.
4. If the audit data shows no flagged issues, say so plainly rather than \
inventing a concern.
5. End with one or two suggested next steps for the human reviewer, framed as \
suggestions, not instructions.
6. Keep the response under 300 words.
"""


class GovernanceRAGPipeline:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.kb = GovernanceKnowledgeBase()
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def _build_query(self, audit_data: dict) -> str:
        """Turn the audit data into a retrieval query string."""
        terms = []
        fairness = audit_data.get("fairness")
        if fairness:
            terms.append("fairness demographic parity equalized odds protected group")
            if any("exceeds" in f for f in fairness.get("flags", [])):
                terms.append("bias non-discrimination risk")
        if audit_data.get("instance"):
            terms.append("individual explanation human oversight feature attribution")
        if audit_data.get("global_importance"):
            terms.append("feature importance transparency credit scoring")
        return " ".join(terms) or "AI governance explainability"

    def explain(self, audit_data: dict, user_question: str | None = None) -> dict:
        query = user_question or self._build_query(audit_data)
        retrieved = self.kb.retrieve(query, top_k=3)

        context_text = "\n\n".join(
            f"[{c.source} - {c.heading}]\n{c.text}" for c in retrieved
        )

        user_message = (
            f"AUDIT DATA:\n{json.dumps(audit_data, indent=2)}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
        )
        if user_question:
            user_message += f"REVIEWER QUESTION: {user_question}\n"
        else:
            user_message += (
                "Produce a summary audit narrative covering the points above.\n"
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        narrative = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return {
            "narrative": narrative,
            "retrieved_context": [
                {"source": c.source, "heading": c.heading} for c in retrieved
            ],
        }
