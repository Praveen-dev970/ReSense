"""
ReSense - ABIDE II KKI feature comparison using training-only 5-fold CV.

Source:
  C:\\ReSenseResearchData\\questionnaire\\raw\\ABIDE_II_KKI\\original_files\\ABIDEII-KKI_1.csv

This script:
- Normalizes whitespace in CSV column names (does not modify source file).
- Selects participants with 8 <= AGE_AT_SCAN < 12.
- Recreates the original stratified 80:20 split (random_state=42).
- Preserves SUB_ID in a split-membership manifest, but NEVER uses it as a predictor.
- Runs 5-fold stratified CV ONLY on the 80% training participants.
- Compares:
    Baseline: Age + Sex
    Experiment 1: Age + Sex + Handedness
    Experiment 2: Age + Sex + FIQ
    Experiment 3: Age + Sex + Handedness + FIQ
- Uses fold-specific imputation, scaling, and one-hot encoding inside pipelines.
- Does not fit, score, tune, or inspect model predictions on the holdout test set.
- Does not overwrite previous evaluation outputs.

Interpretation boundary:
This predicts ABIDE II cohort group labels (DX_GROUP), not clinical diagnosis
and does not establish clinical validity for ReSense.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki"
)
SOURCE_CSV = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI"
    r"\original_files\ABIDEII-KKI_1.csv"
)
OUTPUT_DIR = BASE_DIR / "feature_comparison"
REPORT_PATH = OUTPUT_DIR / "abide_ii_kki_feature_comparison_cv_report.json"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "abide_ii_kki_participant_split_manifest.csv"

AGE_MIN = 8.0
AGE_MAX_EXCLUSIVE = 12.0
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42
CV_FOLDS = 5
CV_RANDOM_STATE = 42
TARGET = "DX_GROUP"
ID_COLUMN = "SUB_ID"
POSITIVE_LABEL = 1  # 1=Autism, 2=Control

EXPERIMENTS = {
    "baseline_age_sex": {
        "numeric": ["AGE_AT_SCAN"],
        "categorical": ["SEX"],
    },
    "experiment_1_age_sex_handedness": {
        "numeric": ["AGE_AT_SCAN"],
        "categorical": ["SEX", "HANDEDNESS_CATEGORY"],
    },
    "experiment_2_age_sex_fiq": {
        "numeric": ["AGE_AT_SCAN", "FIQ"],
        "categorical": ["SEX"],
    },
    "experiment_3_age_sex_handedness_fiq": {
        "numeric": ["AGE_AT_SCAN", "FIQ"],
        "categorical": ["SEX", "HANDEDNESS_CATEGORY"],
    },
}

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "autism_precision",
    "autism_sensitivity_recall",
    "autism_f1",
    "autism_roc_auc",
]


def make_model(numeric_features, categorical_features):
    """Create leakage-safe preprocessing + Logistic Regression pipeline."""
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=CV_RANDOM_STATE,
            max_iter=1000,
        )),
    ])


def evaluate_fold(y_true, y_pred, autism_scores):
    y_binary = (np.asarray(y_true) == POSITIVE_LABEL).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "autism_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "autism_sensitivity_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "autism_f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "autism_roc_auc": float(roc_auc_score(y_binary, autism_scores)),
        "confusion_matrix_labels_1_then_2": confusion_matrix(
            y_true, y_pred, labels=[1, 2]
        ).tolist(),
    }


def summarize(fold_results):
    summary = {}
    for metric in METRICS:
        values = [item[metric] for item in fold_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "fold_values": [float(value) for value in values],
        }
    return summary


def main():
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            "Original ABIDE II KKI CSV not found:\n{}\n"
            "Check SOURCE_CSV at the top of this script.".format(SOURCE_CSV)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read-only source handling; normalize column-name whitespace in memory.
    data = pd.read_csv(SOURCE_CSV)
    data.columns = [str(column).strip() for column in data.columns]

    required = [
        ID_COLUMN, "AGE_AT_SCAN", "SEX", "HANDEDNESS_CATEGORY",
        "FIQ", "DX_GROUP",
    ]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError("Required source columns missing: {}".format(missing_columns))

    source_hash_before = None
    try:
        import hashlib
        def sha256_file(path):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        source_hash_before = sha256_file(SOURCE_CSV)
    except OSError:
        raise

    # Convert expected numeric fields safely.
    for column in ["AGE_AT_SCAN", "SEX", "HANDEDNESS_CATEGORY", "FIQ", "DX_GROUP"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    selected = data.loc[
        (data["AGE_AT_SCAN"] >= AGE_MIN)
        & (data["AGE_AT_SCAN"] < AGE_MAX_EXCLUSIVE)
    ].copy()

    if selected[ID_COLUMN].isna().any():
        raise ValueError("Missing SUB_ID in age-selected records.")
    if selected[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate SUB_ID in age-selected records.")
    if selected[TARGET].isna().any():
        raise ValueError("Missing DX_GROUP in age-selected records.")
    selected[TARGET] = selected[TARGET].astype(int)

    if set(selected[TARGET].unique()) != {1, 2}:
        raise ValueError("Expected DX_GROUP codes 1=Autism and 2=Control.")

    # Confirm predictor coding. Missing FIQ is allowed and imputed within folds.
    if selected["SEX"].isna().any():
        raise ValueError("Missing SEX values; investigate before proceeding.")
    if not set(selected["SEX"].dropna().unique()).issubset({1, 2}):
        raise ValueError("Unexpected SEX coding.")
    if selected["HANDEDNESS_CATEGORY"].isna().any():
        raise ValueError("Missing handedness category; investigate before proceeding.")
    if not set(selected["HANDEDNESS_CATEGORY"].dropna().unique()).issubset({1, 2, 3}):
        raise ValueError("Unexpected handedness coding.")
    if selected["FIQ"].isna().sum() > 1:
        raise ValueError("FIQ missingness differs from the audited source; investigate.")

    # Recreate the same stratified split from the age-filtered source order.
    # This mirrors the original baseline script's split parameters.
    X_split = selected[["AGE_AT_SCAN", "SEX"]].copy()
    y = selected[TARGET].copy()

    train_indices, test_indices = train_test_split(
        np.arange(len(selected)),
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y,
    )

    train_data = selected.iloc[train_indices].copy()
    test_data = selected.iloc[test_indices].copy()

    # Save only participant membership and target labels; no test predictions.
    manifest = pd.DataFrame({
        ID_COLUMN: selected[ID_COLUMN].values,
        "split": [
            "train" if index in set(train_indices) else "holdout_test"
            for index in range(len(selected))
        ],
        TARGET: selected[TARGET].values,
    })
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)

    X_train = train_data.drop(columns=[ID_COLUMN, TARGET])
    y_train = train_data[TARGET].astype(int)

    if len(train_data) != 148 or len(test_data) != 38:
        raise ValueError(
            "Recreated split size differs from the verified baseline "
            "(expected 148 train / 38 test; found {} / {}).".format(
                len(train_data), len(test_data)
            )
        )

    expected_train_counts = {1: 37, 2: 111}
    actual_train_counts = {
        int(key): int(value)
        for key, value in y_train.value_counts().sort_index().items()
    }
    if actual_train_counts != expected_train_counts:
        raise ValueError(
            "Training class distribution differs from the verified baseline: "
            "{}".format(actual_train_counts)
        )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    experiment_reports = {}

    print("=" * 72)
    print("ABIDE II KKI - FEATURE COMPARISON / TRAINING-ONLY 5-FOLD CV")
    print("=" * 72)
    print("Age-selected participants:", len(selected))
    print("Training participants:", len(train_data))
    print("Untouched holdout participants:", len(test_data))
    print("Training class counts:", actual_train_counts)
    print("Features never include SUB_ID or diagnostic assessment variables.")
    print("Holdout test set: NOT USED for fitting, scoring, or tuning.")
    print("scikit-learn:", sklearn.__version__)

    for experiment_name, feature_config in EXPERIMENTS.items():
        feature_columns = (
            feature_config["numeric"] + feature_config["categorical"]
        )
        X = X_train[feature_columns].copy()

        fold_results = []
        dummy_results = []

        print("\n" + "-" * 72)
        print(experiment_name)
        print("Features:", feature_columns)

        for fold_number, (fit_idx, validation_idx) in enumerate(
            cv.split(X, y_train), start=1
        ):
            X_fit = X.iloc[fit_idx]
            X_validation = X.iloc[validation_idx]
            y_fit = y_train.iloc[fit_idx]
            y_validation = y_train.iloc[validation_idx]

            # Majority baseline, evaluated on the same folds.
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_fit, y_fit)
            dummy_prediction = dummy.predict(X_validation)
            dummy_results.append({
                "accuracy": float(accuracy_score(y_validation, dummy_prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_validation, dummy_prediction)
                ),
                "autism_sensitivity_recall": float(
                    recall_score(
                        y_validation, dummy_prediction, pos_label=1, zero_division=0
                    )
                ),
            })

            model = make_model(
                feature_config["numeric"],
                feature_config["categorical"],
            )
            model.fit(X_fit, y_fit)
            predictions = model.predict(X_validation)

            classifier = model.named_steps["classifier"]
            autism_index = list(classifier.classes_).index(POSITIVE_LABEL)
            autism_scores = model.predict_proba(X_validation)[:, autism_index]

            metrics = evaluate_fold(
                y_validation, predictions, autism_scores
            )
            fold_results.append(metrics)

            print(
                "Fold {}: balanced_accuracy={:.4f}, sensitivity={:.4f}, "
                "precision={:.4f}, F1={:.4f}, ROC-AUC={:.4f}".format(
                    fold_number,
                    metrics["balanced_accuracy"],
                    metrics["autism_sensitivity_recall"],
                    metrics["autism_precision"],
                    metrics["autism_f1"],
                    metrics["autism_roc_auc"],
                )
            )

        experiment_reports[experiment_name] = {
            "features": feature_columns,
            "numeric_features": feature_config["numeric"],
            "categorical_features": feature_config["categorical"],
            "missing_fiq_values_in_training_data": int(
                train_data["FIQ"].isna().sum()
            ),
            "logistic_regression": {
                "class_weight": "balanced",
                "max_iter": 1000,
                "fold_results": fold_results,
                "summary": summarize(fold_results),
            },
            "majority_baseline": {
                "strategy": "most_frequent",
                "fold_results": dummy_results,
                "summary": {
                    metric: {
                        "mean": float(np.mean([item[metric] for item in dummy_results])),
                        "std": float(np.std([item[metric] for item in dummy_results], ddof=1)),
                        "fold_values": [float(item[metric]) for item in dummy_results],
                    }
                    for metric in [
                        "accuracy", "balanced_accuracy",
                        "autism_sensitivity_recall",
                    ]
                },
            },
        }

    source_hash_after = sha256_file(SOURCE_CSV)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source CSV hash changed during execution.")

    report = {
        "project": "ReSense",
        "dataset": "ABIDE II KKI Phenotypic Data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(SOURCE_CSV),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "age_filter": "8 <= AGE_AT_SCAN < 12",
        "age_selected_records": int(len(selected)),
        "age_selected_class_counts": {
            str(int(k)): int(v)
            for k, v in selected[TARGET].value_counts().sort_index().items()
        },
        "participant_id": {
            "column": ID_COLUMN,
            "unique_in_selected_subset": bool(selected[ID_COLUMN].is_unique),
            "used_as_predictor": False,
            "split_manifest": str(SPLIT_MANIFEST_PATH),
        },
        "target": {
            "column": TARGET,
            "coding": {"1": "Autism", "2": "Control"},
        },
        "split": {
            "method": "stratified train_test_split recreated from original source order",
            "train_fraction": 0.80,
            "test_fraction": 0.20,
            "random_state": SPLIT_RANDOM_STATE,
            "train_records": int(len(train_data)),
            "holdout_test_records": int(len(test_data)),
            "train_class_counts": {
                str(int(k)): int(v)
                for k, v in y_train.value_counts().sort_index().items()
            },
            "holdout_test_used_for_modeling": False,
            "holdout_test_used_for_feature_selection": False,
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": CV_FOLDS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
            "performed_only_on_training_participants": True,
        },
        "preprocessing": {
            "numeric": "Median imputation and StandardScaler fitted within each fold",
            "categorical": "Most-frequent imputation and OneHotEncoder fitted within each fold",
            "pipeline_prevents_fold_preprocessing_leakage": True,
        },
        "experiments": experiment_reports,
        "software": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Small selected cohort; fold metrics may be unstable.",
            "The cohort is imbalanced (46 Autism and 140 Control participants).",
            "FIQ has one missing value in the age-selected subset; imputation is fitted inside each training fold.",
            "FIQ, VIQ, and PIQ are correlated cognitive measures; this script tests FIQ only.",
            "Handedness categories 2 and 3 have small counts.",
            "The three cognitive measures and handedness are exploratory cohort features, not established screening predictors.",
            "The evaluation predicts the ABIDE II cohort grouping label, not clinical diagnosis.",
            "No clinical validation of ReSense is performed.",
        ],
        "model_training_performed": True,
        "clinical_validation_performed": False,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("MEAN +/- SAMPLE STANDARD DEVIATION")
    print("=" * 72)
    for name, details in experiment_reports.items():
        summary = details["logistic_regression"]["summary"]
        print("\n{}".format(name))
        for metric in METRICS:
            print("  {:30s} {:.4f} +/- {:.4f}".format(
                metric,
                summary[metric]["mean"],
                summary[metric]["std"],
            ))

    print("\nReport saved:", REPORT_PATH)
    print("Participant split manifest saved:", SPLIT_MANIFEST_PATH)
    print("Source unchanged:", source_hash_before == source_hash_after)
    print("Holdout test set was not used.")


if __name__ == "__main__":
    main()
