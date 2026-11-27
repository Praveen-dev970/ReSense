"""
ReSense — Frozen ABIDE I KKI external evaluation

Two-stage workflow:
1) --prepare-protocol writes a frozen protocol file and performs NO predictions.
2) Review the protocol file. Then run --evaluate --confirm-frozen-protocol.

The evaluation stage:
- Loads the existing fitted ABIDE II WISC-IV Logistic Regression pipeline.
- Does not fit, tune, or overwrite the model.
- Does not alter source CSV files.
- Uses the fixed positive class (1=Autism) and probability threshold (0.50).
- Saves participant-level predictions, metrics, and confusion-matrix CSV.
- Uses the majority class from the ABIDE II development cohort as the baseline.

Compatible with the project's older scikit-learn 0.24.2 environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


FEATURES = ["AGE_AT_SCAN", "SEX", "FIQ"]
TARGET = "DX_GROUP"
ID_COLUMN = "SUB_ID"
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 2
THRESHOLD = 0.50
PROTOCOL_VERSION = "1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    required = FEATURES + [TARGET, ID_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {missing}")

    if frame[ID_COLUMN].isna().any():
        raise ValueError(f"{label} contains missing participant IDs.")
    if frame[ID_COLUMN].astype(str).str.strip().duplicated().any():
        raise ValueError(f"{label} contains duplicate participant IDs.")

    for column in FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[FEATURES + [TARGET]].isna().any().any():
        missing_counts = frame[FEATURES + [TARGET]].isna().sum().to_dict()
        raise ValueError(f"{label} has missing/non-numeric required values: {missing_counts}")

    if not frame["AGE_AT_SCAN"].between(8, 12, inclusive="left").all():
        raise ValueError(f"{label} contains ages outside 8 <= age < 12.")

    if not set(frame["SEX"].astype(int).unique()).issubset({1, 2}):
        raise ValueError(f"{label} contains unexpected SEX codes.")
    if not set(frame[TARGET].astype(int).unique()).issubset({1, 2}):
        raise ValueError(f"{label} contains unexpected DX_GROUP codes.")

    if not np.allclose(frame["SEX"], frame["SEX"].astype(int)):
        raise ValueError(f"{label} contains non-integer SEX codes.")
    if not np.allclose(frame[TARGET], frame[TARGET].astype(int)):
        raise ValueError(f"{label} contains non-integer target codes.")

    frame["SEX"] = frame["SEX"].astype(int)
    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def validate_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("Saved model is not a scikit-learn Pipeline.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("Saved pipeline does not expose predict_proba().")

    final_estimator = model.steps[-1][1]
    check_is_fitted(final_estimator)

    classes = list(getattr(final_estimator, "classes_", []))
    if sorted(int(value) for value in classes) != [1, 2]:
        raise ValueError(f"Unexpected fitted model classes: {classes}")

    return model, classes


def prepare_protocol(args):
    development = load_and_validate_csv(args.development, "Development cohort")
    external = load_and_validate_csv(args.external, "External cohort")

    development_ids = set(development[ID_COLUMN].astype(str).str.strip())
    external_ids = set(external[ID_COLUMN].astype(str).str.strip())
    overlap = sorted(development_ids.intersection(external_ids))
    if overlap:
        raise ValueError(
            f"Participant overlap detected ({len(overlap)} IDs). "
            "External evaluation is stopped."
        )

    model, classes = validate_model(args.model)
    del model

    development_counts = {
        str(int(key)): int(value)
        for key, value in development[TARGET].value_counts().sort_index().items()
    }
    baseline_class = int(
        development[TARGET].value_counts().sort_values(ascending=False).index[0]
    )

    protocol = {
        "protocol_name": "ReSense ABIDE I KKI frozen external evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREPARED_FOR_REVIEW_NOT_YET_EXECUTED",
        "purpose": (
            "Evaluate the already-fitted ABIDE II KKI WISC-IV Logistic Regression "
            "candidate on non-overlapping ABIDE I KKI participants."
        ),
        "cohorts": {
            "development": {
                "path": str(args.development),
                "sha256": sha256_file(args.development),
                "n": int(len(development)),
                "target_counts": development_counts,
            },
            "external": {
                "path": str(args.external),
                "sha256": sha256_file(args.external),
                "n": int(len(external)),
                "target_counts": {
                    str(int(key)): int(value)
                    for key, value in external[TARGET].value_counts().sort_index().items()
                },
                "age_rule": "8 <= AGE_AT_SCAN < 12",
            },
            "participant_overlap_count": 0,
        },
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "classes": [int(value) for value in classes],
            "features_in_order": FEATURES,
            "model_type": "Fitted scikit-learn Pipeline ending in LogisticRegression",
            "training_or_tuning_during_external_evaluation": False,
        },
        "prediction_rule": {
            "positive_class": POSITIVE_CLASS,
            "positive_class_name": "Autism",
            "negative_class": NEGATIVE_CLASS,
            "negative_class_name": "Control",
            "probability_threshold": THRESHOLD,
            "rule": "Predict Autism (class 1) when P(class 1) >= 0.50; otherwise Control (class 2).",
            "threshold_changes_permitted": False,
        },
        "baseline": {
            "method": "Always predict the majority class in the ABIDE II development cohort",
            "predicted_class": baseline_class,
            "predicted_class_name": "Autism" if baseline_class == 1 else "Control",
            "external_target_distribution_not_used_to_select_baseline": True,
        },
        "metrics": [
            "accuracy",
            "balanced_accuracy",
            "autism_sensitivity_recall",
            "autism_specificity",
            "autism_precision",
            "autism_f1",
            "roc_auc_for_autism_class",
            "confusion_matrix_with_labels_1_then_2",
        ],
        "constraints": [
            "No model fitting or hyperparameter tuning.",
            "No threshold adjustment after inspecting external results.",
            "No removal of external participants after inspecting predictions.",
            "No modification of the saved model or source cohort CSV files.",
            "Report all metrics as research-cohort evaluation, not clinical diagnostic performance.",
            "Both cohorts are from KKI; this is not independent-site validation.",
        ],
        "documentation_items_to_retain": [
            "ABIDE data-use terms, non-commercial conditions, attribution, and institutional approval.",
            "Rationale for WISC-IV FIQ comparability; retain original test-type labels.",
            "Confirmation that the exact external cohort extract and model hashes match this protocol at evaluation time.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "abide_external_evaluation_frozen_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Protocol prepared. NO PREDICTIONS WERE GENERATED.")
    print(f"Protocol file: {protocol_path}")
    print(f"Development cohort: n={len(development)}; majority baseline class={baseline_class}")
    print(f"External cohort: n={len(external)}")
    print("Review the JSON protocol before proceeding to evaluation.")
    return protocol_path


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def calculate_metrics(y_true, y_pred, positive_prob):
    # Matrix order is explicit: Autism (1), then Control (2).
    matrix = confusion_matrix(y_true, y_pred, labels=[1, 2])
    tp, fn = int(matrix[0, 0]), int(matrix[0, 1])
    fp, tn = int(matrix[1, 0]), int(matrix[1, 1])

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "autism_sensitivity_recall": safe_ratio(tp, tp + fn),
        "autism_specificity": safe_ratio(tn, tn + fp),
        "autism_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "autism_f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc_for_autism_class": float(roc_auc_score(y_true == 1, positive_prob)),
        "confusion_matrix_labels_1_autism_then_2_control": matrix.tolist(),
        "counts": {
            "true_autism_predicted_autism": tp,
            "true_autism_predicted_control": fn,
            "true_control_predicted_autism": fp,
            "true_control_predicted_control": tn,
        },
    }


def evaluate(args):
    protocol_path = args.protocol
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Frozen protocol file not found: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    if protocol.get("status") != "PREPARED_FOR_REVIEW_NOT_YET_EXECUTED":
        raise ValueError("Protocol status is unexpected; refusing to evaluate.")
    if not args.confirm_frozen_protocol:
        raise ValueError("Add --confirm-frozen-protocol only after reviewing the protocol JSON.")

    model_path = Path(protocol["model"]["path"])
    development_path = Path(protocol["cohorts"]["development"]["path"])
    external_path = Path(protocol["cohorts"]["external"]["path"])

    for path, expected_hash, label in [
        (model_path, protocol["model"]["sha256"], "Model"),
        (development_path, protocol["cohorts"]["development"]["sha256"], "Development cohort"),
        (external_path, protocol["cohorts"]["external"]["sha256"], "External cohort"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} file no longer exists: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{label} SHA256 changed since protocol preparation. "
                "Refusing evaluation to protect the frozen protocol."
            )

    development = load_and_validate_csv(development_path, "Development cohort")
    external = load_and_validate_csv(external_path, "External cohort")

    dev_ids = set(development[ID_COLUMN].astype(str).str.strip())
    ext_ids = set(external[ID_COLUMN].astype(str).str.strip())
    if dev_ids.intersection(ext_ids):
        raise ValueError("Participant overlap found at evaluation time. Stopping.")

    model, classes = validate_model(model_path)
    if [int(value) for value in classes] != protocol["model"]["classes"]:
        raise ValueError("Model classes differ from the frozen protocol.")
    if protocol["model"]["features_in_order"] != FEATURES:
        raise ValueError("Feature order differs from the frozen protocol.")

    # Only inference occurs below. No fit/fit_transform calls are made.
    X_external = external[FEATURES].copy()
    y_true = external[TARGET].astype(int).to_numpy()
    probabilities = model.predict_proba(X_external)

    classes_array = np.asarray(classes)
    positive_indices = np.where(classes_array == POSITIVE_CLASS)[0]
    if len(positive_indices) != 1:
        raise ValueError("Could not uniquely locate Autism class probability.")
    positive_prob = probabilities[:, int(positive_indices[0])]

    threshold = float(protocol["prediction_rule"]["probability_threshold"])
    y_pred = np.where(
        positive_prob >= threshold, POSITIVE_CLASS, NEGATIVE_CLASS
    ).astype(int)

    model_metrics = calculate_metrics(y_true, y_pred, positive_prob)

    baseline_class = int(protocol["baseline"]["predicted_class"])
    baseline_pred = np.full(len(y_true), baseline_class, dtype=int)
    baseline_prob = np.full(len(y_true), 1.0 if baseline_class == 1 else 0.0)
    baseline_metrics = calculate_metrics(y_true, baseline_pred, baseline_prob)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        ID_COLUMN: external[ID_COLUMN].astype(str),
        "true_DX_GROUP": y_true,
        "predicted_DX_GROUP": y_pred,
        "probability_Autism_class_1": positive_prob,
        "decision_threshold": threshold,
    })
    predictions_path = output_dir / "abide_i_kki_external_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    matrix = np.asarray(model_metrics["confusion_matrix_labels_1_autism_then_2_control"])
    matrix_frame = pd.DataFrame(
        matrix,
        index=["Actual Autism (1)", "Actual Control (2)"],
        columns=["Predicted Autism (1)", "Predicted Control (2)"],
    )
    matrix_path = output_dir / "abide_i_kki_external_confusion_matrix.csv"
    matrix_frame.to_csv(matrix_path)

    report = {
        "evaluation_name": "ReSense ABIDE I KKI frozen external evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_version": protocol.get("protocol_version"),
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "classes": [int(value) for value in classes],
            "features": FEATURES,
            "training_performed": False,
            "tuning_performed": False,
            "threshold_adjusted": False,
        },
        "cohorts": {
            "development_n": int(len(development)),
            "external_n": int(len(external)),
            "external_target_counts": {
                str(int(key)): int(value)
                for key, value in external[TARGET].value_counts().sort_index().items()
            },
            "participant_overlap_count": 0,
        },
        "protocol": {
            "positive_class": POSITIVE_CLASS,
            "threshold": threshold,
            "baseline_class_from_development": baseline_class,
        },
        "results": {
            "logistic_regression": model_metrics,
            "majority_class_baseline": baseline_metrics,
        },
        "output_files": {
            "predictions_csv": str(predictions_path),
            "confusion_matrix_csv": str(matrix_path),
        },
        "limitations": [
            "This is a small same-site, independent-participant cohort evaluation; it is not independent-site validation.",
            "The ABIDE cohort labels are research cohort labels and do not establish ReSense clinical diagnostic performance.",
            "Metrics may be imprecise because the external sample is small.",
            "FIQ comparability and all applicable data-use conditions must be documented in the project records.",
            "Results must not be generalized to population screening or clinical diagnosis.",
        ],
    }

    report_path = output_dir / "abide_i_kki_external_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== ReSense ABIDE I KKI External Evaluation ===")
    print("Inference completed using the frozen model. No fitting or tuning performed.")
    print(f"External participants: {len(external)}")
    print(f"Report: {report_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {matrix_path}")
    print("\nMetric                         Logistic       Baseline")
    for key in [
        "accuracy", "balanced_accuracy", "autism_sensitivity_recall",
        "autism_specificity", "autism_precision", "autism_f1",
        "roc_auc_for_autism_class",
    ]:
        a = model_metrics.get(key)
        b = baseline_metrics.get(key)
        av = "N/A" if a is None else f"{a:.4f}"
        bv = "N/A" if b is None else f"{b:.4f}"
        print(f"{key:30s} {av:12s} {bv:12s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-protocol", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--confirm-frozen-protocol", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.prepare_protocol == args.evaluate:
        parser.error("Choose exactly one: --prepare-protocol OR --evaluate.")

    if args.prepare_protocol:
        for name in ["model", "development", "external"]:
            if getattr(args, name) is None:
                parser.error(f"--{name} is required with --prepare-protocol.")
        prepare_protocol(args)
    else:
        if args.protocol is None:
            parser.error("--protocol is required with --evaluate.")
        if not args.confirm_frozen_protocol:
            parser.error(
                "Evaluation requires --confirm-frozen-protocol after you review the protocol."
            )
        evaluate(args)


if __name__ == "__main__":
    main()
