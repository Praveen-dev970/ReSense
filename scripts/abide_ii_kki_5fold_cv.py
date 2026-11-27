"""
ABIDE II KKI: 5-fold stratified cross-validation on the existing training split.

Important:
- Uses ONLY the previously saved training split.
- Does NOT read or evaluate the held-out test split.
- Fits scaling inside each fold via Pipeline to prevent preprocessing leakage.
- Compares a majority-class DummyClassifier with balanced Logistic Regression.
- This is exploratory research, not clinical validation or diagnostic performance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
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
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki"
)
EVALUATION_DIR = DATA_DIR / "evaluation"
TRAIN_FILE = EVALUATION_DIR / "abide_ii_kki_train_split.csv"
REPORT_FILE = EVALUATION_DIR / "abide_ii_kki_5fold_cv_report.json"

FEATURES = ["AGE_AT_SCAN", "SEX"]
TARGET = "DX_GROUP"
POSITIVE_LABEL = 1  # Official coding: 1 = Autism, 2 = Control
N_SPLITS = 5
RANDOM_STATE = 42


def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate metrics with Autism (DX_GROUP=1) treated as the positive class."""
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "autism_precision": float(
            precision_score(
                y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "autism_sensitivity_recall": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "autism_f1": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "confusion_matrix_labels_1_then_2": confusion_matrix(
            y_true, y_pred, labels=[1, 2]
        ).tolist(),
    }

    if y_score is not None:
        result["autism_roc_auc"] = float(
            roc_auc_score((np.asarray(y_true) == POSITIVE_LABEL).astype(int), y_score)
        )

    return result


def summarize_metric(fold_results, metric_name):
    values = [fold[metric_name] for fold in fold_results]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "fold_values": [float(value) for value in values],
    }


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            "Training split not found:\n{}\n"
            "Run the baseline evaluation script first.".format(TRAIN_FILE)
        )

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(TRAIN_FILE)

    missing_columns = [column for column in FEATURES + [TARGET] if column not in data.columns]
    if missing_columns:
        raise ValueError("Missing required columns: {}".format(missing_columns))

    if data[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Missing values found in training split. Stop and investigate.")

    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    observed_labels = sorted(data[TARGET].unique().tolist())
    if observed_labels != [1, 2]:
        raise ValueError(
            "Expected target labels [1, 2], found: {}".format(observed_labels)
        )

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    class_counts = {str(int(label)): int(count) for label, count in y.value_counts().items()}
    if min(y.value_counts()) < N_SPLITS:
        raise ValueError(
            "Each class must have at least {} records for {}-fold CV.".format(
                N_SPLITS, N_SPLITS
            )
        )

    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )

    dummy_model = DummyClassifier(strategy="most_frequent")

    logistic_folds = []
    dummy_folds = []

    print("=" * 68)
    print("ABIDE II KKI - 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 68)
    print("Training records:", len(data))
    print("Training class counts (1=Autism, 2=Control):", class_counts)
    print("Features:", FEATURES)
    print("Held-out test split: NOT USED")
    print("Scikit-learn version:", sklearn.__version__)
    print("-" * 68)

    for fold_number, (train_indices, validation_indices) in enumerate(
        cv.split(X, y), start=1
    ):
        X_fold_train = X.iloc[train_indices]
        X_fold_validation = X.iloc[validation_indices]
        y_fold_train = y.iloc[train_indices]
        y_fold_validation = y.iloc[validation_indices]

        # Majority-class baseline
        dummy_model.fit(X_fold_train, y_fold_train)
        dummy_predictions = dummy_model.predict(X_fold_validation)
        dummy_metrics = calculate_metrics(
            y_fold_validation, dummy_predictions
        )
        dummy_folds.append(dummy_metrics)

        # Logistic Regression; scaler is fitted only on this fold's training data
        logistic_model.fit(X_fold_train, y_fold_train)
        logistic_predictions = logistic_model.predict(X_fold_validation)

        classifier = logistic_model.named_steps["classifier"]
        autism_column = list(classifier.classes_).index(POSITIVE_LABEL)
        autism_probabilities = logistic_model.predict_proba(X_fold_validation)[
            :, autism_column
        ]

        logistic_metrics = calculate_metrics(
            y_fold_validation,
            logistic_predictions,
            autism_probabilities,
        )
        logistic_folds.append(logistic_metrics)

        print("\nFold {} (validation records: {})".format(
            fold_number, len(validation_indices)
        ))
        print("  Dummy balanced accuracy: {:.4f}".format(
            dummy_metrics["balanced_accuracy"]
        ))
        print("  Logistic accuracy:       {:.4f}".format(
            logistic_metrics["accuracy"]
        ))
        print("  Logistic balanced acc.:  {:.4f}".format(
            logistic_metrics["balanced_accuracy"]
        ))
        print("  Autism precision:        {:.4f}".format(
            logistic_metrics["autism_precision"]
        ))
        print("  Autism sensitivity:      {:.4f}".format(
            logistic_metrics["autism_sensitivity_recall"]
        ))
        print("  Autism F1:               {:.4f}".format(
            logistic_metrics["autism_f1"]
        ))
        print("  Autism ROC-AUC:          {:.4f}".format(
            logistic_metrics["autism_roc_auc"]
        ))

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "autism_precision",
        "autism_sensitivity_recall",
        "autism_f1",
        "autism_roc_auc",
    ]

    report = {
        "study": "ABIDE II KKI demographic baseline - 5-fold stratified CV",
        "purpose": (
            "Exploratory assessment of training-fold consistency; "
            "not clinical validation or diagnostic performance."
        ),
        "input_file": str(TRAIN_FILE),
        "held_out_test_split_used": False,
        "model_training_performed": True,
        "features": FEATURES,
        "target": TARGET,
        "positive_class": {"code": 1, "meaning": "Autism"},
        "negative_class": {"code": 2, "meaning": "Control"},
        "training_records": int(len(data)),
        "training_class_counts": class_counts,
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "preprocessing": (
            "StandardScaler fitted within each fold using Pipeline; "
            "no validation-fold scaling information used for fitting."
        ),
        "models": {
            "majority_baseline": {
                "type": "DummyClassifier",
                "strategy": "most_frequent",
                "fold_results": dummy_folds,
                "summary": {
                    metric: summarize_metric(dummy_folds, metric)
                    for metric in metric_names if metric in dummy_folds[0]
                },
            },
            "logistic_regression": {
                "type": "StandardScaler + LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "fold_results": logistic_folds,
                "summary": {
                    metric: summarize_metric(logistic_folds, metric)
                    for metric in metric_names
                },
            },
        },
        "software": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "The training split contains a relatively small number of participants.",
            "Fold metrics may vary substantially because of the small sample.",
            "Only age and sex are used as predictors.",
            "DX_GROUP represents the source cohort grouping and does not validate ReSense.",
            "This evaluation does not establish clinical diagnostic performance.",
            "The held-out test split was not used in this cross-validation step.",
        ],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    print("\n" + "=" * 68)
    print("CROSS-VALIDATION SUMMARY (mean +/- sample standard deviation)")
    print("=" * 68)

    for model_name in ["majority_baseline", "logistic_regression"]:
        print("\n{}".format(model_name.replace("_", " ").title()))
        for metric in metric_names:
            summary = report["models"][model_name]["summary"].get(metric)
            if summary:
                print("  {:30s} {:.4f} +/- {:.4f}".format(
                    metric, summary["mean"], summary["std"]
                ))

    print("\nReport saved to:")
    print(REPORT_FILE)
    print("\nCompleted. The held-out test split was not accessed.")


if __name__ == "__main__":
    main()
