"""
ReSense - ABIDE II KKI cognitive-feature comparison: FIQ vs VIQ vs PIQ.

Experiments (same participant split and CV folds):
  Reference: Age + Sex + FIQ
  Experiment 4: Age + Sex + VIQ
  Experiment 5: Age + Sex + PIQ

Rules:
- Reads the original ABIDE II KKI CSV without modifying it.
- Selects 8 <= AGE_AT_SCAN < 12.
- Recreates the original stratified 80:20 split (random_state=42).
- Uses only the 148 training participants for 5-fold CV.
- Keeps SUB_ID for participant tracking only; never a predictor.
- Imputation, scaling, and categorical encoding occur inside each fold.
- Does not access holdout test labels or evaluate holdout predictions.
- Saves to a new output folder; does not overwrite earlier reports.

This predicts ABIDE II cohort grouping (DX_GROUP), not clinical diagnosis.
It does not establish clinical validity for ReSense.
"""

import json
import hashlib
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


SOURCE_CSV = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI"
    r"\original_files\ABIDEII-KKI_1.csv"
)
BASE_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki"
)
OUTPUT_DIR = BASE_DIR / "cognitive_feature_comparison"
REPORT_PATH = OUTPUT_DIR / "abide_ii_kki_fiq_viq_piq_cv_report.json"
SPLIT_MANIFEST_PATH = OUTPUT_DIR / "abide_ii_kki_participant_split_manifest.csv"

ID_COLUMN = "SUB_ID"
TARGET = "DX_GROUP"
POSITIVE_LABEL = 1
AGE_MIN = 8.0
AGE_MAX_EXCLUSIVE = 12.0
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42
CV_FOLDS = 5
CV_RANDOM_STATE = 42

EXPERIMENTS = {
    "reference_age_sex_fiq": {
        "numeric": ["AGE_AT_SCAN", "FIQ"],
        "categorical": ["SEX"],
    },
    "experiment_4_age_sex_viq": {
        "numeric": ["AGE_AT_SCAN", "VIQ"],
        "categorical": ["SEX"],
    },
    "experiment_5_age_sex_piq": {
        "numeric": ["AGE_AT_SCAN", "PIQ"],
        "categorical": ["SEX"],
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(numeric_features, categorical_features):
    """All preprocessing is fitted independently inside each CV fold."""
    transformers = []

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers.append(("numeric", numeric_pipeline, numeric_features))

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
    output = {}
    for metric in METRICS:
        values = [row[metric] for row in fold_results]
        output[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "fold_values": [float(value) for value in values],
        }
    return output


def main():
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            "Original ABIDE II KKI CSV not found:\n{}\n"
            "Check SOURCE_CSV at the top of this script.".format(SOURCE_CSV)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(SOURCE_CSV)

    data = pd.read_csv(SOURCE_CSV)
    data.columns = [str(column).strip() for column in data.columns]

    required = [
        ID_COLUMN, "AGE_AT_SCAN", "SEX", "FIQ", "VIQ", "PIQ", TARGET
    ]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError("Required source columns missing: {}".format(missing_columns))

    for column in ["AGE_AT_SCAN", "SEX", "FIQ", "VIQ", "PIQ", TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    selected = data.loc[
        (data["AGE_AT_SCAN"] >= AGE_MIN)
        & (data["AGE_AT_SCAN"] < AGE_MAX_EXCLUSIVE)
    ].copy()

    if selected[ID_COLUMN].isna().any() or selected[ID_COLUMN].duplicated().any():
        raise ValueError("Selected subset has missing or duplicate participant IDs.")
    if selected[TARGET].isna().any():
        raise ValueError("Missing target labels in selected subset.")

    selected[TARGET] = selected[TARGET].astype(int)
    if set(selected[TARGET].unique()) != {1, 2}:
        raise ValueError("Expected DX_GROUP codes 1=Autism and 2=Control.")

    # The source audit reported no missing VIQ or PIQ and one missing FIQ.
    # Stop if source values differ from that audited condition.
    cognitive_missing = {
        feature: int(selected[feature].isna().sum())
        for feature in ["FIQ", "VIQ", "PIQ"]
    }
    if cognitive_missing != {"FIQ": 1, "VIQ": 0, "PIQ": 0}:
        raise ValueError(
            "Cognitive missingness differs from the audited source: {}".format(
                cognitive_missing
            )
        )

    if selected["SEX"].isna().any() or not set(
        selected["SEX"].dropna().unique()
    ).issubset({1, 2}):
        raise ValueError("SEX values are missing or contain unexpected codes.")

    # Recreate original baseline split from the age-filtered source order.
    train_indices, test_indices = train_test_split(
        np.arange(len(selected)),
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=selected[TARGET],
    )
    train_data = selected.iloc[train_indices].copy()
    test_data = selected.iloc[test_indices].copy()

    if len(train_data) != 148 or len(test_data) != 38:
        raise ValueError(
            "Expected 148 training and 38 holdout participants; found {} and {}."
            .format(len(train_data), len(test_data))
        )

    y_train = train_data[TARGET].astype(int)
    train_counts = {
        int(k): int(v)
        for k, v in y_train.value_counts().sort_index().items()
    }
    if train_counts != {1: 37, 2: 111}:
        raise ValueError(
            "Training class counts differ from original baseline: {}".format(
                train_counts
            )
        )

    manifest = pd.DataFrame({
        ID_COLUMN: selected[ID_COLUMN].values,
        "split": [
            "train" if i in set(train_indices) else "holdout_test"
            for i in range(len(selected))
        ],
        TARGET: selected[TARGET].values,
    })
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    experiment_reports = {}

    print("=" * 72)
    print("ABIDE II KKI - FIQ / VIQ / PIQ COMPARISON")
    print("=" * 72)
    print("Age-selected participants:", len(selected))
    print("Training participants:", len(train_data))
    print("Untouched holdout participants:", len(test_data))
    print("Training class counts (1=Autism, 2=Control):", train_counts)
    print("Cognitive missing values:", cognitive_missing)
    print("Holdout test set: NOT USED")
    print("scikit-learn:", sklearn.__version__)

    for experiment_name, config in EXPERIMENTS.items():
        features = config["numeric"] + config["categorical"]
        X = train_data[features].copy()
        fold_results = []
        dummy_results = []

        print("\n" + "-" * 72)
        print(experiment_name)
        print("Features:", features)

        for fold_number, (fit_idx, validation_idx) in enumerate(
            cv.split(X, y_train), start=1
        ):
            X_fit = X.iloc[fit_idx]
            X_validation = X.iloc[validation_idx]
            y_fit = y_train.iloc[fit_idx]
            y_validation = y_train.iloc[validation_idx]

            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_fit, y_fit)
            dummy_predictions = dummy.predict(X_validation)
            dummy_results.append({
                "accuracy": float(accuracy_score(y_validation, dummy_predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_validation, dummy_predictions)
                ),
                "autism_sensitivity_recall": float(
                    recall_score(
                        y_validation, dummy_predictions, pos_label=1,
                        zero_division=0
                    )
                ),
            })

            model = make_model(config["numeric"], config["categorical"])
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
            "features": features,
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
                        "mean": float(np.mean([row[metric] for row in dummy_results])),
                        "std": float(np.std([row[metric] for row in dummy_results], ddof=1)),
                        "fold_values": [float(row[metric]) for row in dummy_results],
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
        raise RuntimeError("Original source CSV changed during execution.")

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
            "unique": bool(selected[ID_COLUMN].is_unique),
            "used_as_predictor": False,
            "split_manifest": str(SPLIT_MANIFEST_PATH),
        },
        "target": {
            "column": TARGET,
            "coding": {"1": "Autism", "2": "Control"},
        },
        "split": {
            "method": "Stratified train_test_split recreated from original source order",
            "random_state": SPLIT_RANDOM_STATE,
            "train_records": int(len(train_data)),
            "holdout_test_records": int(len(test_data)),
            "train_class_counts": {str(k): v for k, v in train_counts.items()},
            "holdout_test_used": False,
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": CV_FOLDS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
            "training_only": True,
        },
        "preprocessing": {
            "numeric": "Median imputation and StandardScaler fitted within each fold",
            "categorical": "Most-frequent imputation and OneHotEncoder fitted within each fold",
        },
        "cognitive_missing_values_in_age_selected_subset": cognitive_missing,
        "experiments": experiment_reports,
        "software": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Small cohort and only 37 Autism participants in the training split.",
            "Fold-level estimates may be unstable.",
            "FIQ has one missing value; VIQ and PIQ have none in the selected subset.",
            "FIQ, VIQ, and PIQ are related cognitive measures.",
            "Cognitive measures may not be practical or appropriate inputs for a preliminary screening workflow.",
            "This predicts ABIDE II cohort grouping labels, not clinical diagnosis.",
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
        print("\n" + name)
        for metric in METRICS:
            values = details["logistic_regression"]["summary"][metric]
            print("  {:30s} {:.4f} +/- {:.4f}".format(
                metric, values["mean"], values["std"]
            ))

    print("\nReport saved:", REPORT_PATH)
    print("Participant split manifest saved:", SPLIT_MANIFEST_PATH)
    print("Source unchanged:", source_hash_before == source_hash_after)
    print("Holdout test set was not used.")


if __name__ == "__main__":
    main()
