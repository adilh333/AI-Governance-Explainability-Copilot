"""
Fairness module.

Computes group-level fairness metrics across a protected attribute using
Fairlearn. The results feed directly into the RAG pipeline, which grounds
the LLM's explanation of *why* a metric might be flagged in the relevant
governance framework (e.g. EU AI Act risk categories, ICO guidance on
discrimination in automated decision-making).
"""

from __future__ import annotations

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score

FEATURE_COLS = ["income", "credit_score", "debt_to_income", "employment_years"]


def fairness_report(model, df: pd.DataFrame, protected_col: str = "applicant_group") -> dict:
    X = df[FEATURE_COLS]
    y_true = df["approved"]
    y_pred = model.predict(X)
    sensitive = df[protected_col]

    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)

    by_group = mf.by_group.reset_index().to_dict(orient="records")

    return {
        "protected_attribute": protected_col,
        "by_group": [
            {
                "group": row[protected_col],
                "selection_rate": round(float(row["selection_rate"]), 4),
                "accuracy": round(float(row["accuracy"]), 4),
            }
            for row in by_group
        ],
        "demographic_parity_difference": round(float(dpd), 4),
        "equalized_odds_difference": round(float(eod), 4),
        "flags": _flag_issues(dpd, eod),
    }


def _flag_issues(dpd: float, eod: float, threshold: float = 0.10) -> list[str]:
    """Simple rule-based flags. Thresholds are illustrative, not regulatory
    determinations, the LLM layer is responsible for contextualising these
    against actual governance frameworks."""
    flags = []
    if abs(dpd) > threshold:
        flags.append(
            f"Demographic parity difference of {dpd:.3f} exceeds the "
            f"illustrative threshold of {threshold}. Approval rates differ "
            f"meaningfully between groups."
        )
    if abs(eod) > threshold:
        flags.append(
            f"Equalized odds difference of {eod:.3f} exceeds the "
            f"illustrative threshold of {threshold}. Error rates differ "
            f"meaningfully between groups."
        )
    if not flags:
        flags.append("No fairness metric exceeded the illustrative threshold.")
    return flags
