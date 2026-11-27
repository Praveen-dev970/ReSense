"""
ReSense - ABIDE II KKI frozen candidate secondary holdout comparison.

Compares:
  A. Previously evaluated baseline: AGE_AT_SCAN + SEX, both standardized.
  B. Frozen exploratory candidate: AGE_AT_SCAN + SEX + FIQ,
     with median imputation/scaling for numeric variables and one-hot SEX.

The exact age-filtered cohort and stratified 80:20 split are reconstructed
using random_state=42. Both models are fitted on the same 148 training
participants and evaluated on the same 38-person holdout.

IMPORTANT:
  The holdout was already evaluated for the AGE+SEX baseline earlier.
  This is a secondary comparison, not a pristine independent test.
  Do not tune the model based on these results.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SOURCE_CSV = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI"
    r"\original_files\ABIDEII-KKI_1.csv"
)
BASE_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki"
)
OUTPUT_DIR = BASE_DIR / "candidate_model_review"

AGE_MIN = 8.0
AGE_MAX_EXCLUSIVE = 12.0
TARGET = "DX_GROUP"
ID_COLUMN = "SUB_ID"
CLASS_LABELS = [1, 2]  # 1=Autism, 2=Control
TEST_SIZE = 0.20
RANDOM_STATE = 42


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(y_true, y_pred, autism_probabilities=None):
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
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=["Autism (1)", "Control (2)"],
            output_dict=True,
            zero_division=0,
        ),
    }
    if autism_probabilities is not None:
        result["autism_roc_auc"] = float(
            roc_auc_score((pd.Series(y_true) == 1).astype(int), autism_probabilities)
        )
    return result


def make_candidate_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, ["AGE_AT_SCAN", "FIQ"]),
        ("categorical", categorical_pipeline, ["SEX"]),
    ], remainder="drop")

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )),
    ])


def main():
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError("Original ABIDE II KKI CSV not found: {}".format(SOURCE_CSV))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(SOURCE_CSV)

    data = pd.read_csv(SOURCE_CSV)
    data.columns = [str(column).strip() for column in data.columns]

    required = [ID_COLUMN, "AGE_AT_SCAN", "SEX", "FIQ", TARGET]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError("Missing required source columns: {}".format(missing_columns))

    for column in ["AGE_AT_SCAN", "SEX", "FIQ", TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    selected = data.loc[
        (data["AGE_AT_SCAN"] >= AGE_MIN)
        & (data["AGE_AT_SCAN"] < AGE_MAX_EXCLUSIVE)
    ].copy()

    if len(selected) != 186:
        raise ValueError("Expected 186 age-selected participants; found {}".format(len(selected)))
    if selected[ID_COLUMN].isna().any() or selected[ID_COLUMN].duplicated().any():
        raise ValueError("Missing or duplicate participant IDs in age-selected cohort.")
    if selected[TARGET].isna().any() or set(selected[TARGET].unique()) != {1, 2}:
        raise ValueError("Unexpected target values.")
    if selected["SEX"].isna().any() or not set(selected["SEX"].unique()).issubset({1, 2}):
        raise ValueError("Unexpected or missing SEX values.")
    if int(selected["FIQ"].isna().sum()) != 1:
        raise ValueError("Expected exactly one missing FIQ value in selected cohort.")

    # Recreate the original baseline's split method and ordering.
    # SUB_ID is used only for audit/manifests and never as a predictor.
    X_split = selected[["AGE_AT_SCAN", "SEX"]].copy()
    y = selected[TARGET].astype(int).copy()
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_split,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_ids = selected.loc[X_train_base.index, ID_COLUMN].astype(str)
    test_ids = selected.loc[X_test_base.index, ID_COLUMN].astype(str)

    if len(X_train_base) != 148 or len(X_test_base) != 38:
        raise ValueError("Expected train=148 and holdout=38.")
    if set(train_ids).intersection(set(test_ids)):
        raise ValueError("Participant overlap between training and holdout.")
    if y_train.value_counts().sort_index().to_dict() != {1: 37, 2: 111}:
        raise ValueError("Training class distribution differs from established split.")
    if y_test.value_counts().sort_index().to_dict() != {1: 9, 2: 29}:
        raise ValueError("Holdout class distribution differs from established split.")

    # Majority class reference.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_base, y_train)
    dummy_pred = dummy.predict(X_test_base)
    dummy_result = evaluate(y_test, dummy_pred)

    # Refit the exact earlier age+sex baseline architecture on the same split.
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )),
    ])
    baseline.fit(X_train_base, y_train)
    baseline_pred = baseline.predict(X_test_base)
    baseline_classifier = baseline.named_steps["classifier"]
    baseline_autism_index = list(baseline_classifier.classes_).index(1)
    baseline_autism_prob = baseline.predict_proba(X_test_base)[:, baseline_autism_index]
    baseline_result = evaluate(y_test, baseline_pred, baseline_autism_prob)

    # Frozen candidate: age + sex + FIQ.
    candidate_features = ["AGE_AT_SCAN", "SEX", "FIQ"]
    X_candidate = selected[candidate_features].copy()
    X_train_candidate = X_candidate.loc[X_train_base.index]
    X_test_candidate = X_candidate.loc[X_test_base.index]

    candidate = make_candidate_pipeline()
    candidate.fit(X_train_candidate, y_train)
    candidate_pred = candidate.predict(X_test_candidate)
    candidate_classifier = candidate.named_steps["classifier"]
    candidate_autism_index = list(candidate_classifier.classes_).index(1)
    candidate_autism_prob = candidate.predict_proba(X_test_candidate)[:, candidate_autism_index]
    candidate_result = evaluate(y_test, candidate_pred, candidate_autism_prob)

    # Participant-level prediction ledger. IDs are included only in this
    # restricted research artifact, not used as predictors.
    prediction_ledger = pd.DataFrame({
        "SUB_ID": test_ids.values,
        "actual_DX_GROUP": y_test.values,
        "majority_prediction": dummy_pred,
        "age_sex_baseline_prediction": baseline_pred,
        "age_sex_baseline_autism_probability": baseline_autism_prob,
        "age_sex_fiq_candidate_prediction": candidate_pred,
        "age_sex_fiq_candidate_autism_probability": candidate_autism_prob,
    })
    prediction_path = OUTPUT_DIR / "abide_ii_kki_secondary_holdout_predictions.csv"
    prediction_ledger.to_csv(prediction_path, index=False)

    source_hash_after = sha256_file(SOURCE_CSV)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Original source CSV changed during execution.")

    report = {
        "project": "ReSense",
        "dataset": "ABIDE II KKI Phenotypic Data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "source_file": str(SOURCE_CSV),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "age_filter": "8 <= AGE_AT_SCAN < 12",
        "selected_records": int(len(selected)),
        "split": {
            "method": "Stratified train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "training_records": int(len(X_train_base)),
            "holdout_records": int(len(X_test_base)),
            "training_class_counts": {"1_Autism": 37, "2_Control": 111},
            "holdout_class_counts": {"1_Autism": 9, "2_Control": 29},
            "participant_level_separation_verified": True,
            "SUB_ID_used_as_predictor": False,
        },
        "models": {
            "majority_class_dummy": {
                "strategy": "most_frequent",
                "metrics": dummy_result,
            },
            "previous_age_sex_baseline": {
                "features": ["AGE_AT_SCAN", "SEX"],
                "preprocessing": "StandardScaler",
                "classifier": "LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)",
                "metrics": baseline_result,
            },
            "frozen_age_sex_fiq_candidate": {
                "features": candidate_features,
                "numeric_preprocessing": "Median imputation + StandardScaler for AGE_AT_SCAN and FIQ",
                "categorical_preprocessing": "Most-frequent imputation + OneHotEncoder for SEX",
                "classifier": "LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)",
                "metrics": candidate_result,
            },
        },
        "comparison_note": (
            "The candidate is compared with the previously evaluated AGE+SEX baseline "
            "on the same reconstructed holdout split. The holdout was already used for "
            "the earlier baseline evaluation, so this is a secondary comparison, not a "
            "pristine independent test."
        ),
        "holdout_used_for_training_or_tuning": False,
        "model_selection_note": (
            "AGE+SEX+FIQ was selected after exploratory training-only cross-validation. "
            "The reported cross-validation results are exploratory and subject to model-selection optimism."
        ),
        "clinical_interpretation": (
            "These models predict ABIDE II KKI cohort group labels (DX_GROUP), not a "
            "ReSense clinical outcome. Results do not establish clinical diagnostic performance."
        ),
        "artifacts": {
            "holdout_prediction_ledger": str(prediction_path),
        },
        "clinical_validation_performed": False,
        "model_training_performed": True,
    }

    report_path = OUTPUT_DIR / "abide_ii_kki_secondary_holdout_comparison_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)

    print("=" * 68)
    print("ABIDE II KKI - SECONDARY HOLDOUT COMPARISON")
    print("=" * 68)
    print("Selected participants:", len(selected))
    print("Training participants:", len(X_train_base))
    print("Holdout participants:", len(X_test_base))
    print("Training class counts:", report["split"]["training_class_counts"])
    print("Holdout class counts:", report["split"]["holdout_class_counts"])
    print("\nMajority-class reference:")
    print(json.dumps(dummy_result, indent=2))
    print("\nPrevious AGE + SEX baseline:")
    print(json.dumps(baseline_result, indent=2))
    print("\nFrozen AGE + SEX + FIQ candidate:")
    print(json.dumps(candidate_result, indent=2))
    print("\nSource unchanged:", source_hash_before == source_hash_after)
    print("Holdout used for fitting/tuning: NO")
    print("\nReport saved:", report_path)
    print("Prediction ledger saved:", prediction_path)
    print("\nCAUTION: holdout had already been evaluated for the AGE+SEX baseline.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc)
        sys.exit(1)
