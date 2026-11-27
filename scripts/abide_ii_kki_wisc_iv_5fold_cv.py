#!/usr/bin/env python
"""
Controlled 5-fold CV for the ABIDE II KKI WISC-IV-compatible cohort.

Features: AGE_AT_SCAN, SEX, FIQ
Target: DX_GROUP (1=Autism, 2=Control)
Models: majority-class DummyClassifier and balanced LogisticRegression.

Python 3.9 / scikit-learn 0.24 compatible.
No ABIDE I data is read. No holdout test set is used.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_COHORT = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki_wisc_iv_development\abide_ii_kki_wisc_iv_development_cohort.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki_wisc_iv_development\evaluation"
)

FEATURES = ["AGE_AT_SCAN", "SEX", "FIQ"]
TARGET = "DX_GROUP"
ID_COLUMN = "SUB_ID"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_logistic_pipeline():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, ["AGE_AT_SCAN", "FIQ"]),
        ("categorical", categorical, ["SEX"]),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
            solver="liblinear",
        )),
    ])


def build_dummy_pipeline():
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), ["AGE_AT_SCAN", "FIQ"]),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), ["SEX"]),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])


def calculate_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "autism_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "autism_sensitivity_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "autism_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "autism_roc_auc": float(roc_auc_score((np.asarray(y_true) == 1).astype(int), y_score)),
        "confusion_matrix_labels_1_autism_2_control": cm.tolist(),
    }


def summarize(fold_records, model_name):
    metric_names = [
        "accuracy", "balanced_accuracy", "autism_precision",
        "autism_sensitivity_recall", "autism_f1", "autism_roc_auc",
    ]
    summary = {}
    for metric in metric_names:
        values = np.asarray([record[metric] for record in fold_records], dtype=float)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)),
            "mean_percent": float(np.mean(values) * 100),
            "sample_std_percentage_points": float(np.std(values, ddof=1) * 100),
        }
    return {"model": model_name, "metrics": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled 5-fold CV on the ABIDE II WISC-IV cohort."
    )
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cohort_path = args.cohort_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not cohort_path.is_file():
        raise FileNotFoundError("Development cohort not found: {}".format(cohort_path))

    source_hash_before = sha256_file(cohort_path)
    df = pd.read_csv(cohort_path)

    required = FEATURES + [TARGET, ID_COLUMN, "FIQ_TEST_TYPE"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Cohort CSV is missing required columns: {}".format(missing))

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in required features, target, ID, or FIQ test type.")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate SUB_ID values found.")
    if set(df[TARGET].unique()) != {1, 2}:
        raise ValueError("Target must contain exactly DX_GROUP codes 1 and 2.")
    if not df["FIQ_TEST_TYPE"].astype(str).str.strip().str.upper().isin(
        ["WISC-IV", "WISC_IV", "WISC_IV_FULL"]
    ).all():
        raise ValueError("Non-WISC-IV-compatible FIQ test type found.")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    participant_ids = df[ID_COLUMN].astype(str).copy()

    if len(df) < 5 or y.value_counts().min() < 5:
        raise ValueError("Insufficient class counts for stratified 5-fold CV.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows = []
    prediction_rows = []

    for fold_number, (train_index, validation_index) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_val = y.iloc[train_index], y.iloc[validation_index]

        models = {
            "majority_baseline": build_dummy_pipeline(),
            "balanced_logistic_regression": build_logistic_pipeline(),
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            if model_name == "balanced_logistic_regression":
                classes = list(model.named_steps["classifier"].classes_)
                autism_index = classes.index(1)
                scores = model.predict_proba(X_val)[:, autism_index]
            else:
                # DummyClassifier's predicted class probabilities are valid scores,
                # though its majority-only behavior is intentionally uninformative.
                classes = list(model.named_steps["classifier"].classes_)
                autism_index = classes.index(1)
                scores = model.predict_proba(X_val)[:, autism_index]

            metrics = calculate_metrics(y_val, predictions, scores)
            fold_rows.append({
                "fold": fold_number,
                "model": model_name,
                "train_records": int(len(train_index)),
                "validation_records": int(len(validation_index)),
                "train_autism": int((y_train == 1).sum()),
                "train_control": int((y_train == 2).sum()),
                "validation_autism": int((y_val == 1).sum()),
                "validation_control": int((y_val == 2).sum()),
                **metrics,
            })

            for local_index, original_index in enumerate(validation_index):
                prediction_rows.append({
                    "fold": fold_number,
                    "model": model_name,
                    "SUB_ID": participant_ids.iloc[original_index],
                    "actual_DX_GROUP": int(y.iloc[original_index]),
                    "predicted_DX_GROUP": int(predictions[local_index]),
                    "autism_probability_or_score": float(scores[local_index]),
                })

    logistic_folds = [r for r in fold_rows if r["model"] == "balanced_logistic_regression"]
    dummy_folds = [r for r in fold_rows if r["model"] == "majority_baseline"]

    # Fit a final candidate on all 182 development participants for later frozen
    # external evaluation. This fit is not used to calculate CV metrics.
    final_model = build_logistic_pipeline()
    final_model.fit(X, y)

    source_hash_after = sha256_file(cohort_path)
    source_unchanged = source_hash_before == source_hash_after

    report = {
        "report_title": "ABIDE II KKI WISC-IV Controlled 5-Fold Cross-Validation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cohort_file": str(cohort_path),
        "cohort_sha256_before": source_hash_before,
        "cohort_sha256_after": source_hash_after,
        "cohort_unchanged": source_unchanged,
        "cohort_records": int(len(df)),
        "class_counts": {
            "1_Autism": int((y == 1).sum()),
            "2_Control": int((y == 2).sum()),
        },
        "features": FEATURES,
        "target": TARGET,
        "positive_class": "1 = Autism",
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": 5,
            "shuffle": True,
            "random_state": 42,
            "preprocessing_fitted_inside_each_fold": True,
            "external_dataset_used": False,
            "holdout_test_set_used": False,
        },
        "models": [
            summarize(dummy_folds, "majority_baseline"),
            summarize(logistic_folds, "balanced_logistic_regression"),
        ],
        "fold_results": fold_rows,
        "model_configuration": {
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
            "numeric_features": ["AGE_AT_SCAN", "FIQ"],
            "numeric_preprocessing": ["median imputation", "StandardScaler"],
            "categorical_features": ["SEX"],
            "categorical_preprocessing": ["most_frequent imputation", "OneHotEncoder"],
        },
        "final_candidate_fit": {
            "fitted_on_all_development_records": True,
            "purpose": "Frozen candidate for a later external evaluation; not used for CV scoring.",
            "saved_model_file": "abide_ii_kki_wisc_iv_logistic_candidate.joblib",
        },
        "validation_checks": {
            "required_columns_present": True,
            "no_missing_required_values": True,
            "unique_participant_ids": True,
            "both_target_classes_present": True,
            "wisc_iv_compatible_test_types_only": True,
            "source_cohort_unchanged": source_unchanged,
        },
        "all_validation_checks_passed": source_unchanged,
        "limitations": [
            "Cross-validation estimates are internal development results, not external validation.",
            "The cohort is small and imbalanced; fold-level metrics may vary substantially.",
            "FIQ compatibility across ABIDE I and ABIDE II still requires documentation-level confirmation.",
            "DX_GROUP is a research cohort label, not a ReSense clinical diagnosis.",
            "No clinical diagnostic performance is established.",
        ],
        "external_validation_performed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "abide_ii_kki_wisc_iv_5fold_results.csv"
    prediction_path = output_dir / "abide_ii_kki_wisc_iv_cv_predictions.csv"
    report_path = output_dir / "abide_ii_kki_wisc_iv_5fold_cv_report.json"
    model_path = output_dir / "abide_ii_kki_wisc_iv_logistic_candidate.joblib"

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False, encoding="utf-8-sig")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)
    joblib.dump(final_model, model_path)

    print("\nABIDE II KKI WISC-IV CONTROLLED 5-FOLD CV")
    print("=" * 52)
    print("Records:", len(df))
    print("Class counts:", report["class_counts"])
    print("scikit-learn:", sklearn.__version__)
    for model_summary in report["models"]:
        print("\n" + model_summary["model"])
        for metric, values in model_summary["metrics"].items():
            print(
                "  {}: {:.2f}% +/- {:.2f} pp".format(
                    metric,
                    values["mean_percent"],
                    values["sample_std_percentage_points"],
                )
            )
    print("\nAll validation checks passed:", report["all_validation_checks_passed"])
    print("Source cohort unchanged:", source_unchanged)
    print("\nFold report:", fold_path)
    print("CV predictions:", prediction_path)
    print("JSON report:", report_path)
    print("Frozen candidate model:", model_path)
    print("\nNo ABIDE I data used. No holdout test set used for CV scoring.")


if __name__ == "__main__":
    main()
