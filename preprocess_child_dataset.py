
"""
ReSense - Child ARFF Dataset Preprocessing

Source:
Autism-Child-Data.arff

Rules:
- Never modify the original ARFF file.
- Preserve all source records and target labels.
- Preserve missing demographic values.
- Validate A1-A10 and result score consistency.
- Exclude result from model predictors to prevent score leakage.
- Do not train a machine-learning model.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.io import arff


INPUT_FILE = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\UCI_Children"
    r"\original_files\Autism-Child-Data.arff"
)

OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\child"
)

QUESTION_COLUMNS = [
    "A{}_Score".format(i) for i in range(1, 11)
]

TARGET_COLUMN = "Class/ASD"
SCORE_COLUMN = "result"

EXPECTED_COLUMNS = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score",
    "A5_Score", "A6_Score", "A7_Score", "A8_Score",
    "A9_Score", "A10_Score", "age", "gender",
    "ethnicity", "jundice", "austim", "contry_of_res",
    "used_app_before", "result", "age_desc", "relation",
    "Class/ASD"
]


def load_arff_dataset(input_file):
    raw_data, metadata = arff.loadarff(str(input_file))
    df = pd.DataFrame(raw_data)

    # Decode byte-string values.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].apply(
            lambda value: (
                value.decode("utf-8")
                if isinstance(value, bytes)
                else value
            )
        )

    # Convert ARFF missing markers to pandas missing values.
    df = df.replace("?", pd.NA)

    return df


def preprocess_dataset():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            "Original ARFF file not found: {}".format(INPUT_FILE)
        )

    if OUTPUT_DIR.resolve() == INPUT_FILE.parent.resolve():
        raise ValueError(
            "Output directory must be separate from the source directory."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_arff_dataset(INPUT_FILE)

    # Verify expected attributes.
    missing_columns = [
        col for col in EXPECTED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Expected attributes are missing: {}".format(missing_columns)
        )

    if len(df.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            "Unexpected number of attributes. Review the ARFF structure."
        )

    # Validate questionnaire responses.
    numeric_questions = df[QUESTION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_questions = (
        numeric_questions.notna().all(axis=1)
        & numeric_questions.isin([0, 1]).all(axis=1)
    )

    if not valid_questions.all():
        raise ValueError(
            "Invalid or missing questionnaire responses detected."
        )

    # Validate stored score against the sum of A1-A10.
    calculated_score = numeric_questions.sum(
        axis=1, min_count=10
    )

    stored_score = pd.to_numeric(
        df[SCORE_COLUMN], errors="coerce"
    )

    score_mismatches = (
        stored_score.isna()
        | calculated_score.ne(stored_score)
    )

    if score_mismatches.any():
        raise ValueError(
            "Score mismatches detected. Preprocessing stopped. "
            "Inspect the original records before proceeding."
        )

    # Validate target labels.
    labels = (
        df[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    allowed_labels = {"YES", "NO"}

    if not labels.isin(allowed_labels).all():
        raise ValueError(
            "Unexpected target labels detected: {}".format(
                sorted(labels.dropna().unique().tolist())
            )
        )

    # Preserve original target values and all records.
    processed_df = df.copy()

    # Model-ready dataset: retain target, exclude result.
    model_ready_df = processed_df.drop(
        columns=[SCORE_COLUMN]
    ).copy()

    # Output paths.
    processed_csv = OUTPUT_DIR / "child_processed.csv"
    model_ready_csv = OUTPUT_DIR / "child_model_ready.csv"
    dictionary_json = OUTPUT_DIR / "child_data_dictionary.json"
    report_json = OUTPUT_DIR / "child_preprocessing_report.json"

    # Save outputs.
    processed_df.to_csv(processed_csv, index=False)
    model_ready_df.to_csv(model_ready_csv, index=False)

    # Data dictionary.
    data_dictionary = {
        "questionnaire": {
            column: "Binary questionnaire response (0 or 1)"
            for column in QUESTION_COLUMNS
        },
        "age": "Age attribute from the original dataset",
        "gender": "Gender attribute",
        "ethnicity": "Ethnicity attribute",
        "jundice": "Jaundice attribute (original spelling)",
        "austim": "Autism-related attribute (original spelling)",
        "contry_of_res": "Country of residence (original spelling)",
        "used_app_before": "Previous application usage",
        "result": "Sum of the ten questionnaire responses",
        "age_desc": "Age description",
        "relation": "Relationship of questionnaire respondent",
        TARGET_COLUMN: "Original target label (YES or NO)"
    }

    with open(dictionary_json, "w", encoding="utf-8") as file:
        json.dump(
            data_dictionary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Report missing values.
    missing_values = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }

    label_counts = {
        str(label): int(count)
        for label, count in labels.value_counts().items()
    }

    report = {
        "project": "ReSense",
        "dataset": "Autism-Child-Data",
        "source_file": str(INPUT_FILE),
        "original_source_modified": False,

        "original_records": int(len(df)),
        "processed_records": int(len(processed_df)),
        "model_ready_records": int(len(model_ready_df)),

        "original_attributes": int(len(df.columns)),
        "model_ready_attributes": int(len(model_ready_df.columns)),

        "target_column": TARGET_COLUMN,
        "target_distribution": label_counts,

        "questionnaire_validation_passed": bool(
            valid_questions.all()
        ),
        "score_validation_passed": bool(
            not score_mismatches.any()
        ),
        "score_mismatch_count": int(
            score_mismatches.sum()
        ),

        "missing_values_by_column": missing_values,
        "total_missing_values": int(df.isna().sum().sum()),

        "result_excluded_from_model_predictors": True,
        "target_preserved_in_model_ready_dataset": True,

        "records_removed": 0,
        "machine_learning_training_performed": False,

        "processed_csv": str(processed_csv),
        "model_ready_csv": str(model_ready_csv),
        "data_dictionary": str(dictionary_json),

        "notes": [
            "All original records were retained.",
            "Original ARFF file was not modified.",
            "Missing demographic values were preserved.",
            "A1-A10 responses were validated as binary values.",
            "Stored result scores were verified against A1-A10.",
            "The result attribute was excluded from model predictors.",
            "The original Class/ASD target was preserved.",
            "This dataset provides screening labels, not independent clinical diagnoses.",
            "Dataset licensing and reuse permissions must be verified."
        ]
    }

    with open(report_json, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Console summary.
    print("\nRESENSE CHILD DATASET PREPROCESSING")
    print("=" * 48)

    print("Original records:", len(df))
    print("Processed records:", len(processed_df))
    print("Model-ready records:", len(model_ready_df))
    print("Records removed:", len(df) - len(processed_df))

    print("\nTarget distribution:")
    print(pd.Series(label_counts).to_string())

    print("\nQuestionnaire validation:", valid_questions.all())
    print("Score validation:", not score_mismatches.any())
    print("Score mismatches:", int(score_mismatches.sum()))

    print("\nMissing values:")
    print(pd.Series(missing_values).to_string())

    print("\nOUTPUT FILES")
    print("-" * 48)
    print("Processed CSV:", processed_csv)
    print("Model-ready CSV:", model_ready_csv)
    print("Data dictionary:", dictionary_json)
    print("Preprocessing report:", report_json)


if __name__ == "__main__":
    preprocess_dataset()
