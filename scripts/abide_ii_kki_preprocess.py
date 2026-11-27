"""
ReSense - ABIDE II KKI demographic baseline preprocessing
----------------------------------------------------------
Purpose:
    Create a separate, leakage-conscious model-ready CSV for a future
    demographic baseline experiment.

Source:
    C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI\original_files\ABIDEII-KKI_1.csv

Output:
    C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki\
        abide_ii_kki_demographic_baseline.csv
        abide_ii_kki_preprocessing_report.json

Important:
    - The original source CSV is read only and never modified.
    - This script does NOT train a model.
    - The target is DX_GROUP (1=Autism, 2=Control).
    - Initial predictors are AGE_AT_SCAN and SEX only.
    - Rows are restricted to age >= 8 and < 12 years.
"""

from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd


SOURCE_FILE = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI"
    r"\original_files\ABIDEII-KKI_1.csv"
)

OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki"
)

OUTPUT_CSV = OUTPUT_DIR / "abide_ii_kki_demographic_baseline.csv"
REPORT_JSON = OUTPUT_DIR / "abide_ii_kki_preprocessing_report.json"

AGE_MIN_INCLUSIVE = 8.0
AGE_MAX_EXCLUSIVE = 12.0

TARGET_COLUMN = "DX_GROUP"
PREDICTOR_COLUMNS = ["AGE_AT_SCAN", "SEX"]
OUTPUT_COLUMNS = PREDICTOR_COLUMNS + [TARGET_COLUMN]

EXPECTED_DX_CODES = {1, 2}
EXPECTED_SEX_CODES = {1, 2}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_counts_as_strings(series: pd.Series) -> dict:
    """JSON-safe frequency distribution with stable string keys."""
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def main() -> int:
    if not SOURCE_FILE.is_file():
        print("ERROR: Source CSV was not found:")
        print(SOURCE_FILE)
        print("Check the path. Do not move or edit the original dataset.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_hash_before = sha256_file(SOURCE_FILE)

    # Read source data. Normalize header whitespace because AGE_AT_SCAN
    # is stored with a trailing space in the supplied CSV.
    source_df = pd.read_csv(SOURCE_FILE)
    original_row_count = int(len(source_df))
    original_column_count = int(len(source_df.columns))

    normalized_columns = [str(column).strip() for column in source_df.columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError(
            "Column names are not unique after trimming whitespace. "
            "Stop and inspect the source file before proceeding."
        )
    source_df.columns = normalized_columns

    required_columns = [
        "SUB_ID",
        "AGE_AT_SCAN",
        "SEX",
        "DX_GROUP",
    ]
    missing_columns = [
        column for column in required_columns
        if column not in source_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Required columns are missing: {}".format(missing_columns)
        )

    # Convert only the age field needed for filtering.
    source_df["AGE_AT_SCAN"] = pd.to_numeric(
        source_df["AGE_AT_SCAN"], errors="coerce"
    )

    # Age range: 8 years inclusive through under 12 years.
    age_mask = (
        source_df["AGE_AT_SCAN"].ge(AGE_MIN_INCLUSIVE)
        & source_df["AGE_AT_SCAN"].lt(AGE_MAX_EXCLUSIVE)
    )
    age_subset = source_df.loc[age_mask].copy()

    # Select only the approved initial baseline columns.
    model_df = age_subset[OUTPUT_COLUMNS].copy()

    # Convert codes to numeric for strict validation.
    model_df["SEX"] = pd.to_numeric(model_df["SEX"], errors="coerce")
    model_df["DX_GROUP"] = pd.to_numeric(
        model_df["DX_GROUP"], errors="coerce"
    )

    # ---------------------------
    # Validation checks
    # ---------------------------
    checks = {}

    checks["age_filter_applied"] = bool(
        model_df["AGE_AT_SCAN"].ge(AGE_MIN_INCLUSIVE).all()
        and model_df["AGE_AT_SCAN"].lt(AGE_MAX_EXCLUSIVE).all()
    )
    checks["expected_output_columns_only"] = (
        list(model_df.columns) == OUTPUT_COLUMNS
    )
    checks["no_missing_values_in_model_columns"] = (
        not model_df.isna().any().any()
    )
    checks["sex_codes_valid"] = set(
        model_df["SEX"].astype(int).unique()
    ).issubset(EXPECTED_SEX_CODES)
    checks["target_codes_valid"] = set(
        model_df["DX_GROUP"].astype(int).unique()
    ).issubset(EXPECTED_DX_CODES)
    checks["participant_ids_unique_in_source_subset"] = (
        not age_subset["SUB_ID"].duplicated().any()
    )
    checks["nonempty_output"] = len(model_df) > 0
    checks["target_has_both_classes"] = (
        model_df["DX_GROUP"].nunique() == 2
    )

    # Check that only expected numeric codes are present, not fractional
    # values such as 1.5.
    checks["sex_codes_are_integer_values"] = bool(
        (model_df["SEX"] % 1 == 0).all()
    )
    checks["target_codes_are_integer_values"] = bool(
        (model_df["DX_GROUP"] % 1 == 0).all()
    )

    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(
            "Validation failed: {}. No model-ready CSV was written.".format(
                ", ".join(failed_checks)
            )
        )

    # Save a separate processed file. The source CSV is never overwritten.
    model_df.to_csv(OUTPUT_CSV, index=False)

    # Verify saved output by reading it back.
    saved_df = pd.read_csv(OUTPUT_CSV)
    checks["saved_file_readable"] = True
    checks["saved_file_shape_matches"] = (
        saved_df.shape == model_df.shape
    )
    checks["saved_file_columns_match"] = (
        list(saved_df.columns) == OUTPUT_COLUMNS
    )
    checks["saved_file_has_no_missing_values"] = (
        not saved_df.isna().any().any()
    )

    # Confirm original source checksum is unchanged.
    source_hash_after = sha256_file(SOURCE_FILE)
    checks["original_source_unchanged"] = (
        source_hash_before == source_hash_after
    )

    failed_after_save = [
        name for name, passed in checks.items() if not passed
    ]
    if failed_after_save:
        raise ValueError(
            "Post-save validation failed: {}".format(failed_after_save)
        )

    report = {
        "project": "ReSense",
        "dataset": "ABIDE II KKI Phenotypic Data",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": "1.0",
        "source_file": str(SOURCE_FILE),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_file_unchanged": source_hash_before == source_hash_after,
        "original_records": original_row_count,
        "original_columns": original_column_count,
        "age_filter": {
            "column": "AGE_AT_SCAN",
            "minimum_inclusive_years": AGE_MIN_INCLUSIVE,
            "maximum_exclusive_years": AGE_MAX_EXCLUSIVE,
            "description": "8 years <= age < 12 years",
        },
        "age_selected_records": int(len(age_subset)),
        "age_selected_records_excluded_by_filter": int(
            original_row_count - len(age_subset)
        ),
        "output_file": str(OUTPUT_CSV),
        "output_records": int(len(saved_df)),
        "output_columns": list(saved_df.columns),
        "target": {
            "column": TARGET_COLUMN,
            "coding": {"1": "Autism", "2": "Control"},
            "distribution": value_counts_as_strings(saved_df[TARGET_COLUMN]),
        },
        "predictors": {
            "AGE_AT_SCAN": {
                "role": "initial demographic predictor",
                "minimum": float(saved_df["AGE_AT_SCAN"].min()),
                "maximum": float(saved_df["AGE_AT_SCAN"].max()),
                "mean": float(saved_df["AGE_AT_SCAN"].mean()),
            },
            "SEX": {
                "role": "initial demographic predictor",
                "coding": {"1": "Male", "2": "Female"},
                "distribution": value_counts_as_strings(saved_df["SEX"]),
            },
        },
        "excluded_from_initial_baseline": [
            "SUB_ID (participant identifier)",
            "HANDEDNESS_CATEGORY (reserved for a separate exploratory analysis)",
            "HANDEDNESS_SCORES (reserved pending redundancy review)",
            "PDD_DSM_IV_TR and ASD_DSM_5 (diagnostic fields)",
            "ADI-R, ADOS, SRS, SCQ, RBS-R and related assessment fields",
            "FIQ, VIQ and PIQ (cognitive assessment fields reserved for separate review)",
            "All other source columns not explicitly approved as predictors",
        ],
        "validation_checks": checks,
        "validation_passed": all(checks.values()),
        "model_training_performed": False,
        "clinical_interpretation": (
            "This dataset supports research analysis of cohort group labels. "
            "It does not establish clinical diagnostic performance."
        ),
        "notes": [
            "The original source file was read only.",
            "No participant records were intentionally removed within the selected age range.",
            "No model was trained.",
            "The output is a demographic baseline dataset, not a clinical diagnostic dataset.",
        ],
    }

    with REPORT_JSON.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4, ensure_ascii=False)

    print("ABIDE II KKI preprocessing completed successfully.")
    print("Original records:", original_row_count)
    print("Age-selected records:", len(age_subset))
    print("Output shape:", saved_df.shape)
    print("Target distribution:", report["target"]["distribution"])
    print("SEX distribution:", report["predictors"]["SEX"]["distribution"])
    print("Validation passed:", report["validation_passed"])
    print("Original source unchanged:", checks["original_source_unchanged"])
    print("Model trained: No")
    print("Output CSV:", OUTPUT_CSV)
    print("Audit report:", REPORT_JSON)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR:", error)
        sys.exit(1)
