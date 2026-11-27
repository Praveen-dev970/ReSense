"""
ReSense - ABIDE II KKI candidate model coefficient review.

Fits the frozen exploratory candidate (AGE_AT_SCAN + SEX + FIQ) on the
148-person training split only. The 38-person holdout is not loaded for
fitting, scoring, or interpretation.

This is a descriptive coefficient review, not clinical validation.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SOURCE_CSV = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI"
    r"\original_files\ABIDEII-KKI_1.csv"
)
OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki"
) / "candidate_model_review"

REPORT_PATH = OUTPUT_DIR / "abide_ii_kki_age_sex_fiq_coefficient_report.json"
COEFFICIENTS_PATH = OUTPUT_DIR / "abide_ii_kki_age_sex_fiq_coefficients.csv"

AGE_MIN = 8.0
AGE_MAX_EXCLUSIVE = 12.0
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42
TARGET = "DX_GROUP"
ID_COLUMN = "SUB_ID"
NUMERIC_FEATURES = ["AGE_AT_SCAN", "FIQ"]
CATEGORICAL_FEATURES = ["SEX"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError("Original source CSV not found: {}".format(SOURCE_CSV))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(SOURCE_CSV)

    data = pd.read_csv(SOURCE_CSV)
    data.columns = [str(column).strip() for column in data.columns]

    required = [
        ID_COLUMN, "AGE_AT_SCAN", "SEX", "FIQ", TARGET
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Required columns missing: {}".format(missing))

    for column in ["AGE_AT_SCAN", "SEX", "FIQ", TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    selected = data.loc[
        (data["AGE_AT_SCAN"] >= AGE_MIN)
        & (data["AGE_AT_SCAN"] < AGE_MAX_EXCLUSIVE)
    ].copy()

    if selected[ID_COLUMN].isna().any() or selected[ID_COLUMN].duplicated().any():
        raise ValueError("Participant IDs are missing or duplicated in selected subset.")
    if selected[TARGET].isna().any():
        raise ValueError("Target labels are missing.")
    selected[TARGET] = selected[TARGET].astype(int)

    if set(selected[TARGET].unique()) != {1, 2}:
        raise ValueError("Unexpected DX_GROUP coding; expected 1 and 2.")
    if selected["SEX"].isna().any() or not set(selected["SEX"].unique()).issubset({1, 2}):
        raise ValueError("Unexpected or missing SEX values.")
    if int(selected["FIQ"].isna().sum()) != 1:
        raise ValueError("FIQ missingness differs from the previously audited source.")

    # Recreate the previously established split exactly.
    train_indices, holdout_indices = train_test_split(
        np.arange(len(selected)),
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=selected[TARGET],
    )
    train_data = selected.iloc[train_indices].copy()

    if len(train_data) != 148 or len(holdout_indices) != 38:
        raise ValueError("Expected 148 training and 38 holdout participants.")

    y_train = train_data[TARGET].astype(int)
    class_counts = {
        str(int(label)): int(count)
        for label, count in y_train.value_counts().sort_index().items()
    }
    if class_counts != {"1": 37, "2": 111}:
        raise ValueError("Training class counts differ from the established split.")

    X_train = train_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
        )),
    ])

    model.fit(X_train, y_train)

    fitted_preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]
    encoder = fitted_preprocessing.named_transformers_["categorical"].named_steps["onehot"]

    feature_names = (
        NUMERIC_FEATURES
        + ["SEX_{}".format(value) for value in encoder.categories_[0]]
    )

    # In binary scikit-learn LogisticRegression, coef_[0] corresponds to
    # classes_[1]. DX_GROUP 1=Autism and 2=Control in this dataset.
    fitted_positive_class = int(classifier.classes_[1])
    if fitted_positive_class != 2:
        raise ValueError("Unexpected classifier class order: {}".format(classifier.classes_))

    coefficient_rows = []
    for feature_name, coefficient in zip(feature_names, classifier.coef_[0]):
        coefficient = float(coefficient)
        coefficient_rows.append({
            "transformed_feature": feature_name,
            "coefficient_for_control_vs_autism": coefficient,
            "odds_ratio_control_vs_autism": float(np.exp(coefficient)),
            "interpretation": (
                "Positive coefficient shifts the model toward Control (DX_GROUP=2); "
                "negative coefficient shifts it toward Autism (DX_GROUP=1)."
            ),
        })

    coefficient_frame = pd.DataFrame(coefficient_rows)
    coefficient_frame.to_csv(COEFFICIENTS_PATH, index=False)

    source_hash_after = sha256_file(SOURCE_CSV)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Original source file changed during execution.")

    report = {
        "project": "ReSense",
        "dataset": "ABIDE II KKI Phenotypic Data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(SOURCE_CSV),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "age_filter": "8 <= AGE_AT_SCAN < 12",
        "selected_records": int(len(selected)),
        "training_records": int(len(train_data)),
        "holdout_records": int(len(holdout_indices)),
        "holdout_used": False,
        "training_class_counts": class_counts,
        "candidate_model": {
            "features": ["AGE_AT_SCAN", "SEX", "FIQ"],
            "numeric_preprocessing": "Median imputation followed by StandardScaler",
            "categorical_preprocessing": "Most-frequent imputation followed by OneHotEncoder",
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
            "fitted_classes": [int(value) for value in classifier.classes_],
            "coefficient_direction": (
                "Coefficients are for class 2 (Control) versus class 1 (Autism). "
                "Positive values favor Control; negative values favor Autism."
            ),
            "numeric_coefficient_scale": (
                "AGE_AT_SCAN and FIQ coefficients correspond to a one-standard-deviation "
                "increase after training-set scaling."
            ),
            "sex_encoding_note": (
                "Both SEX categories are one-hot encoded. Interpret their coefficients "
                "jointly; they are not independent effects against a single omitted reference."
            ),
        },
        "coefficients": coefficient_rows,
        "intercept_for_control_vs_autism": float(classifier.intercept_[0]),
        "limitations": [
            "Coefficients describe this fitted model and are not causal effects.",
            "The candidate was selected after exploratory cross-validation; these coefficients are descriptive.",
            "The training cohort is small, with 37 Autism participants.",
            "FIQ may not be practical or appropriate for a preliminary screening workflow.",
            "The target is ABIDE II cohort grouping (DX_GROUP), not a ReSense clinical outcome.",
            "No clinical validation of ReSense is performed.",
        ],
        "model_training_performed": True,
        "clinical_validation_performed": False,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)

    print("=" * 68)
    print("ABIDE II KKI - AGE + SEX + FIQ COEFFICIENT REVIEW")
    print("=" * 68)
    print("Selected participants:", len(selected))
    print("Training participants:", len(train_data))
    print("Holdout participants:", len(holdout_indices))
    print("Holdout used: NO")
    print("Training class counts (1=Autism, 2=Control):", class_counts)
    print("\nCoefficients (positive favors Control; negative favors Autism):")
    print(coefficient_frame.to_string(index=False))
    print("\nIntercept (Control vs Autism):", float(classifier.intercept_[0]))
    print("\nSource unchanged:", source_hash_before == source_hash_after)
    print("scikit-learn:", sklearn.__version__)
    print("Report saved:", REPORT_PATH)
    print("Coefficient CSV saved:", COEFFICIENTS_PATH)


if __name__ == "__main__":
    main()
