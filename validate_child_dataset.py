"""
ReSense - Child Dataset Final Validation
Read-only validation of the original ARFF and preprocessing outputs.
Run: python validate_child_dataset.py
"""

import json
import math
from pathlib import Path

import pandas as pd
from scipy.io import arff

SOURCE_FILE = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\UCI_Children"
    r"\original_files\Autism-Child-Data.arff"
)
OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\child"
)
PROCESSED_CSV = OUTPUT_DIR / "child_processed.csv"
MODEL_READY_CSV = OUTPUT_DIR / "child_model_ready.csv"
DATA_DICTIONARY_JSON = OUTPUT_DIR / "child_data_dictionary.json"
PREPROCESSING_REPORT_JSON = OUTPUT_DIR / "child_preprocessing_report.json"
VALIDATION_REPORT_JSON = OUTPUT_DIR / "child_validation_report.json"

QUESTION_COLUMNS = ["A{}_Score".format(i) for i in range(1, 11)]
SCORE_COLUMN = "result"
TARGET_COLUMN = "Class/ASD"
EXPECTED_COLUMNS = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score",
    "A5_Score", "A6_Score", "A7_Score", "A8_Score",
    "A9_Score", "A10_Score", "age", "gender",
    "ethnicity", "jundice", "austim", "contry_of_res",
    "used_app_before", "result", "age_desc", "relation",
    "Class/ASD"
]
CHILD_LABEL_THRESHOLD = 6


def load_arff_dataset(path):
    raw_data, _metadata = arff.loadarff(str(path))
    frame = pd.DataFrame(raw_data)
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].apply(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes) else value
        )
    return frame.replace("?", pd.NA)


def is_missing(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def values_equal(left, right):
    if is_missing(left) and is_missing(right):
        return True
    if is_missing(left) or is_missing(right):
        return False
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1e-9, abs_tol=1e-9
        )
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def frames_equal(left, right):
    if list(left.columns) != list(right.columns):
        return False, "Column names or order differ."
    if len(left) != len(right):
        return False, "Record counts differ."
    for row_index in range(len(left)):
        for column in left.columns:
            if not values_equal(
                left.iloc[row_index][column],
                right.iloc[row_index][column]
            ):
                return False, "Difference at row {}, column '{}'.".format(
                    row_index + 1, column
                )
    return True, "All values match in the same row order."


def add_check(checks, errors, passed, message):
    checks.append({"passed": bool(passed), "message": message})
    if not passed:
        errors.append(message)


def write_report(checks, errors, warnings, summary=None):
    report = {
        "project": "ReSense",
        "validation": "UCI child questionnaire dataset",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "summary": summary
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\nRESENSE CHILD DATASET VALIDATION")
    print("=" * 48)
    for item in checks:
        print("[{}] {}".format(
            "PASS" if item["passed"] else "FAIL", item["message"]
        ))
    if summary:
        print("\nRecords:", summary["original_records"])
        print("Score mismatches:", summary["score_mismatch_count"])
        print("Score/label mismatches:",
              summary["score_label_mismatch_count"])
        print("Checks passed:", summary["passed_checks"])
        print("Checks failed:", summary["failed_checks"])
        print("Validation passed:", summary["validation_passed"])
    for warning in warnings:
        print("WARNING:", warning)
    for error in errors:
        print("ERROR:", error)
    print("\nValidation report:", VALIDATION_REPORT_JSON)


def main():
    checks, errors, warnings = [], [], []
    required = [
        SOURCE_FILE, PROCESSED_CSV, MODEL_READY_CSV,
        DATA_DICTIONARY_JSON, PREPROCESSING_REPORT_JSON
    ]
    missing = [str(path) for path in required if not path.is_file()]
    add_check(
        checks, errors, not missing,
        "All source and preprocessing output files exist."
    )
    if missing:
        for path in missing:
            errors.append("Missing file: " + path)
        write_report(checks, errors, warnings)
        return

    original = load_arff_dataset(SOURCE_FILE)
    processed = pd.read_csv(PROCESSED_CSV)
    model_ready = pd.read_csv(MODEL_READY_CSV)
    with open(DATA_DICTIONARY_JSON, "r", encoding="utf-8") as file:
        data_dictionary = json.load(file)
    with open(PREPROCESSING_REPORT_JSON, "r", encoding="utf-8") as file:
        old_report = json.load(file)

    add_check(
        checks, errors, list(original.columns) == EXPECTED_COLUMNS,
        "Original ARFF has the expected 21 attributes in order."
    )
    add_check(
        checks, errors, len(original) == 292,
        "Original ARFF contains 292 records."
    )
    add_check(
        checks, errors, len(processed) == len(original),
        "Processed CSV retains every original record."
    )
    add_check(
        checks, errors, len(model_ready) == len(original),
        "Model-ready CSV retains every original record."
    )

    same_source, source_message = frames_equal(original, processed)
    add_check(
        checks, errors, same_source,
        "Processed CSV matches original ARFF: " + source_message
    )

    expected_columns = [
        col for col in processed.columns if col != SCORE_COLUMN
    ]
    add_check(
        checks, errors,
        SCORE_COLUMN in processed.columns
        and SCORE_COLUMN not in model_ready.columns,
        "'result' is present in processed data and excluded from model-ready data."
    )
    add_check(
        checks, errors, TARGET_COLUMN in model_ready.columns,
        "Class/ASD target is retained in model-ready data."
    )
    add_check(
        checks, errors, list(model_ready.columns) == expected_columns,
        "Model-ready columns equal processed columns minus 'result'."
    )
    expected_model = processed.drop(columns=[SCORE_COLUMN])
    same_model, model_message = frames_equal(expected_model, model_ready)
    add_check(
        checks, errors, same_model,
        "Model-ready CSV matches expected data: " + model_message
    )

    questions = processed[QUESTION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_questions = (
        questions.notna().all(axis=1)
        & questions.isin([0, 1]).all(axis=1)
    )
    add_check(
        checks, errors, bool(valid_questions.all()),
        "A1-A10 responses are present and binary (0 or 1)."
    )

    calculated = questions.sum(axis=1, min_count=10)
    stored = pd.to_numeric(processed[SCORE_COLUMN], errors="coerce")
    score_valid = stored.notna() & calculated.eq(stored)
    score_mismatches = int((~score_valid).sum())
    add_check(
        checks, errors, score_mismatches == 0,
        "Stored result equals the A1-A10 sum for every record."
    )

    labels = (
        processed[TARGET_COLUMN].astype("string").str.strip().str.upper()
    )
    valid_labels = labels.notna() & labels.isin({"YES", "NO"})
    add_check(
        checks, errors, bool(valid_labels.all()),
        "All Class/ASD labels are YES or NO."
    )
    expected_labels = pd.Series(
        ["YES" if score > CHILD_LABEL_THRESHOLD else "NO"
         for score in calculated],
        index=processed.index, dtype="string"
    )
    label_mismatches = int((~labels.eq(expected_labels)).sum())
    add_check(
        checks, errors, label_mismatches == 0,
        "Class/ASD matches the dataset threshold check: score > 6 => YES; "
        "score <= 6 => NO."
    )

    original_missing = original.isna().sum()
    processed_missing = processed.isna().sum()
    missing_preserved = (
        list(original_missing.index) == list(processed_missing.index)
        and original_missing.equals(processed_missing)
    )
    add_check(
        checks, errors, bool(missing_preserved),
        "Missing-value counts are preserved for every source attribute."
    )

    add_check(
        checks, errors,
        isinstance(data_dictionary, dict)
        and SCORE_COLUMN in data_dictionary
        and TARGET_COLUMN in data_dictionary,
        "Data dictionary defines result and Class/ASD."
    )
    counts_match = (
        old_report.get("original_records") == len(original)
        and old_report.get("processed_records") == len(processed)
        and old_report.get("model_ready_records") == len(model_ready)
    )
    add_check(
        checks, errors, counts_match,
        "Preprocessing report counts match the actual files."
    )
    add_check(
        checks, errors,
        old_report.get("original_source_modified") is False,
        "Preprocessing report says the original source was not modified."
    )
    add_check(
        checks, errors,
        old_report.get("machine_learning_training_performed") is False,
        "Preprocessing report confirms model training was not performed."
    )

    warnings.append(
        "The score/label threshold check validates dataset-label consistency; "
        "it does not establish clinical diagnostic validity."
    )
    if label_mismatches:
        warnings.append(
            "Investigate score/label mismatches before considering model training."
        )

    summary = {
        "original_records": int(len(original)),
        "processed_records": int(len(processed)),
        "model_ready_records": int(len(model_ready)),
        "score_mismatch_count": score_mismatches,
        "score_label_mismatch_count": label_mismatches,
        "target_distribution": {
            str(key): int(value)
            for key, value in labels.value_counts(dropna=False).items()
        },
        "missing_values_by_column": {
            str(key): int(value)
            for key, value in processed.isna().sum().items()
        },
        "total_checks": len(checks),
        "passed_checks": sum(1 for item in checks if item["passed"]),
        "failed_checks": sum(1 for item in checks if not item["passed"]),
        "validation_passed": len(errors) == 0
    }
    write_report(checks, errors, warnings, summary)


if __name__ == "__main__":
    main()
