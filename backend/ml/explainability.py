"""
Explainability module.

Wraps SHAP's TreeExplainer to produce both global feature-importance
summaries and per-instance explanations for a single prediction. The
output format is deliberately simple (lists of dicts) so it can be
serialised directly to JSON for the API layer and consumed easily by
the RAG pipeline as context for the LLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


FEATURE_COLS = ["income", "credit_score", "debt_to_income", "employment_years"]


class ModelExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def global_importance(self, X: pd.DataFrame, max_samples: int = 200) -> list[dict]:
        """Mean absolute SHAP value per feature, across a sample of rows."""
        sample = X[FEATURE_COLS].sample(
            n=min(max_samples, len(X)), random_state=42
        )
        shap_values = self.explainer.shap_values(sample)
        values = _select_class1(shap_values)

        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]

        return [
            {"feature": FEATURE_COLS[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
            for i in order
        ]

    def explain_instance(self, row: pd.Series) -> dict:
        """Per-feature SHAP contribution for a single applicant."""
        x = row[FEATURE_COLS].to_frame().T
        shap_values = self.explainer.shap_values(x)
        values = _select_class1(shap_values)[0]

        ev = self.explainer.expected_value
        base_value = ev[1] if hasattr(ev, "__len__") else ev

        prediction = int(self.model.predict(x)[0])
        probability = float(self.model.predict_proba(x)[0][1])

        contributions = [
            {
                "feature": FEATURE_COLS[i],
                "value": float(x.iloc[0, i]),
                "shap_contribution": round(float(values[i]), 4),
            }
            for i in range(len(FEATURE_COLS))
        ]
        contributions.sort(key=lambda c: abs(c["shap_contribution"]), reverse=True)

        return {
            "prediction": prediction,
            "probability_approved": round(probability, 4),
            "base_value": round(float(base_value), 4),
            "contributions": contributions,
        }


def _select_class1(shap_values):
    """Normalise SHAP output across library versions to a 2D array of
    shape (n_samples, n_features) representing the positive class.

    Depending on the SHAP/sklearn version, TreeExplainer.shap_values may
    return:
      - a list [class0_array, class1_array], each (n_samples, n_features)
      - a single 3D array (n_samples, n_features, n_classes)
      - a single 2D array (n_samples, n_features) for binary-only output
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr
