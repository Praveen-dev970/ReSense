"""
ReSense - ABIDE II KKI demographic baseline evaluation
-------------------------------------------------------
Input:
  C:\\ReSenseResearchData\\questionnaire\\processed\\abide_ii_kki\\
  abide_ii_kki_demographic_baseline.csv

Output directory:
  C:\\ReSenseResearchData\\questionnaire\\processed\\abide_ii_kki\\evaluation\\

This script:
  1. Performs a reproducible stratified 80:20 train/test split.
  2. Compares a majority-class DummyClassifier with Logistic Regression.
  3. Uses AGE_AT_SCAN and SEX only.
  4. Reports class-sensitive metrics and a confusion matrix.
  5. Does not tune on or repeatedly inspect the test set.

Important:
  - This predicts ABIDE II cohort group labels, not clinical diagnosis.
  - The test set is small; results will have substantial uncertainty.
  - No claims of clinical validity are made.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki"
)
INPUT_CSV = BASE_DIR / "abide_ii_kki_demographic_baseline.csv"
OUTPUT_DIR = BASE_DIR / "evaluation"

FEATURES = ["AGE_AT_SCAN", "SEX"]
TARGET = "DX_GROUP"
CLASS_LABELS = [1, 2]  # 1=Autism, 2=Control
TEST_SIZE = 0.20
RANDOM_STATE = 42


def json_safe_report(y_true, y_pred):
    """Build a classification report with explicit class labels."""
    return classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=["Autism (1)", "Control (2)"],
        output_dict=True,
        zero_division=0,
    )


def evaluate_predictions(y_true, y_pred, positive_scores=None):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "autism_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "autism_recall_sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "autism_f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix_labels_1_then_2": confusion_matrix(
            y_true, y_pred, labels=CLASS_LABELS
        ).tolist(),
        "classification_report": json_safe_report(y_true, y_pred),
    }

    if positive_scores is not None and len(set(y_true)) == 2:
        result["autism_roc_auc"] = float(
            roc_auc_score((pd.Series(y_true) == 1).astype(int), positive_scores)
        )

    return result


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            "Model-ready CSV not found: {}\n"
            "Run abide_ii_kki_preprocess.py first.".format(INPUT_CSV)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_CSV)
    expected_columns = FEATURES + [TARGET]

    if list(data.columns) != expected_columns:
        raise ValueError(
            "Unexpected columns. Expected {}, found {}.".format(
                expected_columns, list(data.columns)
            )
        )

    if data.isna().any().any():
        raise ValueError("Missing values found. Stop before splitting.")

    if not set(data[TARGET].unique()).issubset(set(CLASS_LABELS)):
        raise ValueError("Unexpected target values found.")

    if set(data[TARGET].unique()) != set(CLASS_LABELS):
        raise ValueError("Both target classes must be present.")

    # The preprocessing audit confirmed unique participant IDs in the
    # selected source subset. Each row therefore represents one participant.
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Save split membership/data for reproducibility. The test set is not
    # used for fitting or hyperparameter selection.
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    train_path = OUTPUT_DIR / "abide_ii_kki_train_split.csv"
    test_path = OUTPUT_DIR / "abide_ii_kki_test_split.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Baseline 1: predict the most frequent class in training data.
    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(X_train, y_train)
    majority_predictions = majority_model.predict(X_test)
    majority_results = evaluate_predictions(
        y_test, majority_predictions
    )

    # Baseline 2: Logistic Regression with standardized inputs and
    # class_weight='balanced' to account for the target imbalance.
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )),
    ])
    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)

    classifier = logistic_model.named_steps["classifier"]
    autism_column_index = list(classifier.classes_).index(1)
    autism_probabilities = logistic_model.predict_proba(X_test)[
        :, autism_column_index
    ]

    logistic_results = evaluate_predictions(
        y_test,
        logistic_predictions,
        positive_scores=autism_probabilities,
    )

    prediction_df = X_test.copy()
    prediction_df["actual_DX_GROUP"] = y_test
    prediction_df["majority_prediction"] = majority_predictions
    prediction_df["logistic_prediction"] = logistic_predictions
    prediction_df["logistic_autism_probability"] = autism_probabilities

    predictions_path = OUTPUT_DIR / "abide_ii_kki_test_predictions.csv"
    prediction_df.to_csv(predictions_path, index=False)

    report = {
        "project": "ReSense",
        "dataset": "ABIDE II KKI Phenotypic Data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "target": {
            "column": TARGET,
            "coding": {"1": "Autism", "2": "Control"},
        },
        "features": FEATURES,
        "split": {
            "method": "train_test_split with stratify=y",
            "train_fraction": 0.80,
            "test_fraction": 0.20,
            "random_state": RANDOM_STATE,
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "train_class_distribution": {
                str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
            },
            "test_class_distribution": {
                str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
            },
            "participant_level_split": (
                "The source-subset audit confirmed one unique SUB_ID per row; "
                "SUB_ID is excluded from model predictors."
            ),
        },
        "models": {
            "majority_class_dummy": {
                "strategy": "most_frequent",
                "test_metrics": majority_results,
            },
            "logistic_regression": {
                "standardization": "StandardScaler fitted on training data only",
                "class_weight": "balanced",
                "max_iter": 1000,
                "test_metrics": logistic_results,
            },
        },
        "artifacts": {
            "train_split": str(train_path),
            "test_split": str(test_path),
            "test_predictions": str(predictions_path),
        },
        "evaluation_cautions": [
            "The test set contains a small number of Autism participants.",
            "Metrics are estimates for this selected ABIDE II KKI subgroup.",
            "The test set must not be used for model tuning.",
            "This experiment predicts cohort group labels and does not establish clinical diagnostic performance.",
        ],
        "model_training_performed": True,
        "clinical_validation_performed": False,
    }

    report_path = OUTPUT_DIR / "abide_ii_kki_baseline_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=4, ensure_ascii=False)

    print("ABIDE II KKI baseline evaluation completed.")
    print("scikit-learn version:", sklearn.__version__)
    print("Train records:", len(X_train))
    print("Test records:", len(X_test))
    print("Train target distribution:", report["split"]["train_class_distribution"])
    print("Test target distribution:", report["split"]["test_class_distribution"])
    print("\nMajority-class baseline:")
    print(json.dumps(majority_results, indent=2))
    print("\nLogistic Regression:")
    print(json.dumps(logistic_results, indent=2))
    print("\nSaved report:", report_path)
    print("Saved train split:", train_path)
    print("Saved test split:", test_path)
    print("Saved test predictions:", predictions_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc)
        sys.exit(1)
