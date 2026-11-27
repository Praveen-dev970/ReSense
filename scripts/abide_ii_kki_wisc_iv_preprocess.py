#!/usr/bin/env python
"""
Prepare a WISC-IV-compatible ABIDE II KKI development cohort.

Read-only with respect to the original ABIDE II CSV.
Creates a model-development dataset and provenance/validation reports.
Does NOT train a model.

Python 3.9 compatible.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI\original_files\ABIDEII-KKI_1.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki_wisc_iv_development"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_columns(df):
    result = df.copy()
    result.columns = [str(c).strip() for c in result.columns]
    return result


def value_counts(series):
    return {str(k): int(v) for k, v in series.fillna("<MISSING>").astype(str).value_counts().items()}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WISC-IV-compatible ABIDE II KKI development cohort."
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_path = args.source_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError("ABIDE II KKI source CSV not found: {}".format(source_path))

    source_hash_before = sha256_file(source_path)
    raw = pd.read_csv(source_path)
    df = normalize_columns(raw)

    required = [
        "SUB_ID", "AGE_AT_SCAN", "SEX", "DX_GROUP",
        "FIQ", "FIQ_TEST_TYPE",
    ]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise ValueError("Missing required columns: {}".format(missing_required))

    for column in ["AGE_AT_SCAN", "SEX", "DX_GROUP", "FIQ"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep source values and IDs for traceability. IDs are not model predictors.
    age_mask = (df["AGE_AT_SCAN"] >= 8) & (df["AGE_AT_SCAN"] < 12)
    age_subset = df.loc[age_mask].copy()

    test_type = age_subset["FIQ_TEST_TYPE"].astype("string").str.strip().str.upper()
    wisc_iv_mask = test_type.isin(["WISC-IV", "WISC_IV", "WISC_IV_FULL"])
    selected = age_subset.loc[wisc_iv_mask].copy()

    # Require complete target and candidate predictors. No imputation here;
    # the selected WISC-IV FIQ field is expected to be complete.
    complete_mask = (
        selected["SUB_ID"].notna()
        & selected["AGE_AT_SCAN"].notna()
        & selected["SEX"].notna()
        & selected["DX_GROUP"].notna()
        & selected["FIQ"].notna()
    )
    model_ready = selected.loc[complete_mask].copy()

    invalid_sex = sorted(set(model_ready["SEX"].dropna().unique()) - {1, 2})
    invalid_dx = sorted(set(model_ready["DX_GROUP"].dropna().unique()) - {1, 2})
    if invalid_sex:
        raise ValueError("Unexpected SEX codes: {}".format(invalid_sex))
    if invalid_dx:
        raise ValueError("Unexpected DX_GROUP codes: {}".format(invalid_dx))

    if model_ready["SUB_ID"].duplicated().any():
        raise ValueError("Duplicate participant IDs found in selected cohort.")
    if model_ready["FIQ"].isna().any():
        raise ValueError("Missing FIQ remains after complete-case selection.")

    # Explicit, compact modeling table. SUB_ID is retained only for audit/split
    # tracking; the downstream training script must exclude it from predictors.
    model_table = model_ready[
        ["SUB_ID", "AGE_AT_SCAN", "SEX", "FIQ", "DX_GROUP", "FIQ_TEST_TYPE"]
    ].copy()
    model_table["DX_GROUP"] = model_table["DX_GROUP"].astype(int)
    model_table["SEX"] = model_table["SEX"].astype(int)
    model_table["DX_GROUP_LABEL"] = model_table["DX_GROUP"].map(
        {1: "Autism", 2: "Control"}
    )

    source_hash_after = sha256_file(source_path)
    source_unchanged = source_hash_before == source_hash_after

    checks = {
        "required_columns_present": not missing_required,
        "age_filter_applied": bool(((model_table["AGE_AT_SCAN"] >= 8) & (model_table["AGE_AT_SCAN"] < 12)).all()),
        "wisc_iv_test_type_only": bool(
            model_table["FIQ_TEST_TYPE"].astype(str).str.strip().str.upper()
            .isin(["WISC-IV", "WISC_IV", "WISC_IV_FULL"]).all()
        ),
        "participant_ids_complete": bool(model_table["SUB_ID"].notna().all()),
        "participant_ids_unique": not bool(model_table["SUB_ID"].duplicated().any()),
        "age_complete": bool(model_table["AGE_AT_SCAN"].notna().all()),
        "sex_complete_and_valid": bool(model_table["SEX"].notna().all() and set(model_table["SEX"].unique()).issubset({1, 2})),
        "target_complete_and_valid": bool(model_table["DX_GROUP"].notna().all() and set(model_table["DX_GROUP"].unique()).issubset({1, 2})),
        "fiq_complete": bool(model_table["FIQ"].notna().all()),
        "both_target_classes_present": set(model_table["DX_GROUP"].unique()) == {1, 2},
        "source_unchanged": source_unchanged,
    }

    report = {
        "report_title": "ABIDE II KKI WISC-IV-Compatible Development Cohort Preparation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(source_path),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_unchanged,
        "selection": {
            "original_records": int(len(df)),
            "age_rule": "8 <= AGE_AT_SCAN < 12",
            "age_selected_records": int(len(age_subset)),
            "age_selected_fiq_test_type_counts": value_counts(age_subset["FIQ_TEST_TYPE"]),
            "wisc_iv_test_type_rule": ["WISC-IV", "WISC_IV", "WISC_IV_FULL"],
            "wisc_iv_selected_records": int(len(selected)),
            "complete_case_records": int(len(model_ready)),
            "excluded_age_records": int(len(df) - len(age_subset)),
            "excluded_non_wisc_iv_records": int(len(age_subset) - len(selected)),
            "excluded_incomplete_selected_records": int(len(selected) - len(model_ready)),
        },
        "development_cohort": {
            "records": int(len(model_table)),
            "columns": list(model_table.columns),
            "target_counts": value_counts(model_table["DX_GROUP"]),
            "target_labels": {"1": "Autism", "2": "Control"},
            "sex_counts": value_counts(model_table["SEX"]),
            "age_min": float(model_table["AGE_AT_SCAN"].min()) if len(model_table) else None,
            "age_max": float(model_table["AGE_AT_SCAN"].max()) if len(model_table) else None,
            "age_mean": float(model_table["AGE_AT_SCAN"].mean()) if len(model_table) else None,
            "fiq_min": float(model_table["FIQ"].min()) if len(model_table) else None,
            "fiq_max": float(model_table["FIQ"].max()) if len(model_table) else None,
            "fiq_mean": float(model_table["FIQ"].mean()) if len(model_table) else None,
        },
        "predictor_policy": {
            "candidate_predictors": ["AGE_AT_SCAN", "SEX", "FIQ"],
            "target": "DX_GROUP",
            "participant_identifier": "SUB_ID (tracking only; exclude from predictors)",
            "fiq_test_type": "FIQ_TEST_TYPE (eligibility/provenance only; exclude from predictors)",
            "excluded": [
                "DSM_IV_TR", "ASD_DSM_5", "PDD_DSM_IV_TR",
                "ADI-R variables", "ADOS variables", "SRS variables",
                "SCQ variables", "VIQ", "PIQ", "handedness",
            ],
        },
        "validation_checks": checks,
        "all_validation_checks_passed": all(checks.values()),
        "model_training_performed": False,
        "external_abide_i_data_used": False,
        "original_source_modified": False,
        "limitations": [
            "This is cohort preparation, not model training or external validation.",
            "FIQ test-version equivalence must be documented from source metadata/protocol before final external evaluation.",
            "The selected cohort may be small; performance estimates will be uncertain.",
            "DX_GROUP is a research cohort label, not a ReSense clinical diagnosis.",
            "Use is subject to the source dataset's access and licensing conditions.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "abide_ii_kki_wisc_iv_development_cohort.csv"
    report_path = output_dir / "abide_ii_kki_wisc_iv_preprocessing_report.json"

    model_table.to_csv(cohort_path, index=False, encoding="utf-8-sig")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)

    print("\nABIDE II KKI WISC-IV DEVELOPMENT COHORT")
    print("=" * 48)
    print("Original records:", len(df))
    print("Age 8 to under 12:", len(age_subset))
    print("WISC-IV-compatible records:", len(selected))
    print("Complete development records:", len(model_table))
    print("Target counts:", value_counts(model_table["DX_GROUP"]))
    print("Sex counts:", value_counts(model_table["SEX"]))
    print("All validation checks passed:", all(checks.values()))
    print("Source unchanged:", source_unchanged)
    print("\nCohort:", cohort_path)
    print("Report:", report_path)
    print("\nNo model was trained. ABIDE I was not used.")


if __name__ == "__main__":
    main()
