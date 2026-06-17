"""
Generates a synthetic loan-approval dataset and trains a baseline classifier.

This stands in for a real-world dataset. In a production setting, this would
be replaced by an uploaded CSV and a user-provided or user-trained model.
The synthetic data deliberately includes a mild correlation between a
protected attribute (applicant_group) and the outcome, so that the fairness
module has something meaningful to detect.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

RANDOM_STATE = 42


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    income = rng.normal(35000, 12000, n_samples).clip(8000, 120000)
    credit_score = rng.normal(650, 80, n_samples).clip(300, 850)
    debt_to_income = rng.normal(0.35, 0.15, n_samples).clip(0, 1.2)
    employment_years = rng.gamma(2, 2, n_samples).clip(0, 40)

    # Protected attribute: two groups, A and B.
    # Group B is deliberately given a slightly less favourable income
    # distribution to simulate the kind of historical bias a fairness
    # audit is designed to surface.
    applicant_group = rng.choice(["A", "B"], size=n_samples, p=[0.6, 0.4])
    income = np.where(
        applicant_group == "B",
        income * rng.normal(0.80, 0.05, n_samples),
        income,
    )

    # Outcome driven mainly by legitimate factors, with a small additional
    # group-based effect baked in to make the audit meaningful.
    score = (
        0.00004 * income
        + 0.004 * credit_score
        - 1.2 * debt_to_income
        + 0.05 * employment_years
        + rng.normal(0, 0.5, n_samples)
    )
    group_adjustment = np.where(applicant_group == "B", -0.55, 0.0)
    prob_approved = 1 / (1 + np.exp(-(score - 3 + group_adjustment)))
    approved = (rng.random(n_samples) < prob_approved).astype(int)

    df = pd.DataFrame(
        {
            "income": income.round(2),
            "credit_score": credit_score.round(0),
            "debt_to_income": debt_to_income.round(3),
            "employment_years": employment_years.round(1),
            "applicant_group": applicant_group,
            "approved": approved,
        }
    )
    return df


def train_model(df: pd.DataFrame):
    feature_cols = ["income", "credit_score", "debt_to_income", "employment_years"]
    X = df[feature_cols]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def main():
    out_dir = os.path.join(os.path.dirname(__file__))
    df = generate_dataset()
    model, X_train, X_test, y_train, y_test = train_model(df)

    df.to_csv(os.path.join(out_dir, "loan_applications.csv"), index=False)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    X_test.assign(applicant_group=df.loc[X_test.index, "applicant_group"], approved=y_test).to_csv(
        os.path.join(out_dir, "holdout.csv"), index=False
    )

    print(f"Generated {len(df)} rows.")
    print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")
    print("Saved: loan_applications.csv, model.pkl, holdout.csv")


if __name__ == "__main__":
    main()
