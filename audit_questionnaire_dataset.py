
"""
ReSense questionnaire dataset audit and preprocessing.

- Never modifies the original source CSV.
- Retains only documented Yes/No labels.
- Validates questionnaire responses and stored scores.
- Checks binary labels against the documented score threshold.
- Excludes all records belonging to conflicting duplicate groups.
- Generates a clean subset, exclusion records, and an audit report.
- Does not train a machine-learning model.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


QUESTION_COLUMNS = ["A{}".format(i) for i in range(1, 11)]
SCORE_COLUMN = "Qchat-10-Score"
TARGET_COLUMN = "Class/ASD Traits "

EXPECTED_LABELS = {"Yes", "No", "High", "Middle"}
BINARY_LABELS = {"Yes", "No"}

DUPLICATE_COLUMNS = (
    QUESTION_COLUMNS
    + ["Age_Mons", "Sex", "Jaundice", "Family_mem_with_ASD"]
)


def audit_dataset(input_path, output_dir):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            "Input CSV was not found: {}".format(input_path)
        )

    if output_dir == input_path.parent:
        raise ValueError(
            "Output directory must be separate from the source directory."
        )

    df = pd.read_csv(input_path)

    required_columns = (
        QUESTION_COLUMNS
        + [
            SCORE_COLUMN,
            TARGET_COLUMN,
            "Age_Mons",
            "Sex",
            "Jaundice",
            "Family_mem_with_ASD"
        ]
    )

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: {}".format(missing_columns)
        )

    # Normalize labels for auditing without modifying the source data.
    labels = df[TARGET_COLUMN].astype("string").str.strip()

    label_counts = labels.value_counts(dropna=False).to_dict()
    unknown_labels = sorted(
        str(value)
        for value in labels.dropna().unique()
        if value not in EXPECTED_LABELS
    )

    # Validate questionnaire values.
    numeric_questions = df[QUESTION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    question_values_valid = bool(
        numeric_questions.notna().all().all()
        and numeric_questions.isin([0, 1]).all().all()
    )

    invalid_question_rows = (
        numeric_questions.isna().any(axis=1)
        | ~numeric_questions.isin([0, 1]).all(axis=1)
    )

    # Calculate score only when all ten responses are valid.
    calculated_score = numeric_questions.sum(axis=1)
    valid_question_rows = ~invalid_question_rows

    stored_score = pd.to_numeric(
        df[SCORE_COLUMN], errors="coerce"
    )

    score_valid_rows = (
        valid_question_rows
        & stored_score.notna()
        & calculated_score.eq(stored_score)
    )

    score_mismatch_rows = valid_question_rows & (
        stored_score.isna()
        | calculated_score.ne(stored_score)
    )

    # Check binary labels against the documented threshold:
    # score > 3 means Yes; otherwise No.
    expected_label = calculated_score.gt(3).map({
        True: "Yes",
        False: "No"
    })

    binary_label_rows = labels.isin(BINARY_LABELS)

    label_score_mismatch_rows = (
        binary_label_rows
        & score_valid_rows
        & labels.ne(expected_label)
    )

    # Identify all duplicate input groups with conflicting labels.
    # Case_No is deliberately excluded because it is an identifier.
    duplicate_group_sizes = df.groupby(
        DUPLICATE_COLUMNS,
        dropna=False,
        sort=False
    ).size()

    duplicate_group_label_counts = df.assign(
        _audit_label=labels
    ).groupby(
        DUPLICATE_COLUMNS,
        dropna=False,
        sort=False
    )["_audit_label"].nunique()

    conflicting_group_keys = duplicate_group_label_counts[
        duplicate_group_label_counts > 1
    ].index

    # Mark every record belonging to a conflicting group.
    row_group_index = pd.MultiIndex.from_frame(
        df[DUPLICATE_COLUMNS]
    )

    conflicting_duplicate_rows = row_group_index.isin(
        conflicting_group_keys
    )

    conflicting_group_count = int(
        (duplicate_group_label_counts > 1).sum()
    )

    # Build explicit exclusion reasons for every source record.
    exclusion_reasons = []

    for i in range(len(df)):
        reasons = []

        if not binary_label_rows.iloc[i]:
            reasons.append("Non-binary or unknown target label")

        if invalid_question_rows.iloc[i]:
            reasons.append("Invalid or missing questionnaire response")

        if score_mismatch_rows.iloc[i]:
            reasons.append("Stored score does not match calculated score")

        if label_score_mismatch_rows.iloc[i]:
            reasons.append("Binary label does not match score threshold")

        if conflicting_duplicate_rows[i]:
            reasons.append("Belongs to a conflicting duplicate group")

        exclusion_reasons.append("; ".join(reasons))

    exclusion_mask = pd.Series(
        [bool(reason) for reason in exclusion_reasons],
        index=df.index
    )

    # A record is retained only if it passes every validation rule.
    clean_mask = ~exclusion_mask

    clean_df = df.loc[clean_mask].copy()

    excluded_df = df.loc[exclusion_mask].copy()
    excluded_df["Audit_Exclusion_Reason"] = [
        exclusion_reasons[i]
        for i in excluded_df.index
    ]

    # Output files.
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "questionnaire_binary_subset.csv"
    excluded_csv = output_dir / "questionnaire_excluded_records.csv"
    output_json = output_dir / "questionnaire_audit_report.json"

    clean_df.to_csv(output_csv, index=False)
    excluded_df.to_csv(excluded_csv, index=False)

    # Prepare the audit report.
    report = {
        "source_file": str(input_path),
        "original_source_modified": False,
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),

        "original_target_label_counts": {
            str(k): int(v) for k, v in label_counts.items()
        },
        "unknown_target_labels": unknown_labels,

        "question_columns": QUESTION_COLUMNS,
        "question_values_valid": question_values_valid,
        "invalid_question_response_records": int(
            invalid_question_rows.sum()
        ),

        "score_mismatch_count": int(score_mismatch_rows.sum()),
        "stored_score_matches_calculated_score": bool(
            score_mismatch_rows.sum() == 0
        ),

        "binary_label_score_mismatch_count": int(
            label_score_mismatch_rows.sum()
        ),

        "conflicting_duplicate_groups": conflicting_group_count,
        "records_in_conflicting_duplicate_groups": int(
            conflicting_duplicate_rows.sum()
        ),

        "binary_label_records_before_filtering": int(
            binary_label_rows.sum()
        ),

        "clean_subset_records": int(clean_mask.sum()),
        "excluded_records": int(exclusion_mask.sum()),

        "clean_subset_validation_passed": bool(
            clean_mask.sum() + exclusion_mask.sum() == len(df)
            and clean_mask.sum() > 0
            and not invalid_question_rows.loc[clean_mask].any()
            and score_valid_rows.loc[clean_mask].all()
            and not label_score_mismatch_rows.loc[clean_mask].any()
            and not conflicting_duplicate_rows[clean_mask.to_numpy()].any()
            and labels.loc[clean_mask].isin(BINARY_LABELS).all()
        ),

        "processed_csv": str(output_csv),
        "excluded_records_csv": str(excluded_csv),

        "notes": [
            "Original source CSV was not modified.",
            "Only Yes and No target labels are retained.",
            "High and Middle labels are excluded without reinterpretation.",
            "Stored scores must match the sum of A1-A10.",
            "Binary labels must match the documented score > 3 rule.",
            "All records belonging to conflicting duplicate groups are excluded.",
            "Qchat-10-Score must not be used as a predictor because it was used to assign the original class label.",
            "Using A1-A10 to predict these labels would reproduce the questionnaire scoring process, not establish clinical validity.",
            "Dataset reuse permission remains to be verified."
        ]
    }

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    # Display results.
    print("\nRESENSE QUESTIONNAIRE DATASET AUDIT")
    print("=" * 48)

    print("Source records:", len(df))
    print("\nOriginal label counts:")
    print(pd.Series(label_counts).to_string())

    print("\nVALIDATION RESULTS")
    print("-" * 48)
    print("Question values valid:", question_values_valid)
    print(
        "Invalid questionnaire records:",
        int(invalid_question_rows.sum())
    )
    print("Score mismatches:", int(score_mismatch_rows.sum()))
    print(
        "Binary label/score mismatches:",
        int(label_score_mismatch_rows.sum())
    )
    print("Conflicting duplicate groups:", conflicting_group_count)
    print(
        "Records in conflicting groups:",
        int(conflicting_duplicate_rows.sum())
    )

    print("\nFINAL DATASET")
    print("-" * 48)
    print("Binary records before filtering:", int(binary_label_rows.sum()))
    print("Clean subset records:", int(clean_mask.sum()))
    print("Excluded records:", int(exclusion_mask.sum()))
    print(
        "Clean subset validation passed:",
        report["clean_subset_validation_passed"]
    )

    print("\nOUTPUT FILES")
    print("-" * 48)
    print("Clean CSV:", output_csv)
    print("Excluded records:", excluded_csv)
    print("Audit report:", output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit and preprocess the ReSense questionnaire dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original questionnaire CSV."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Separate output directory for processed files."
    )

    args = parser.parse_args()

    audit_dataset(args.input, args.output)
