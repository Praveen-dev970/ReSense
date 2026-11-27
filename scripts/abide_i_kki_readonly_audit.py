#!/usr/bin/env python
"""
Read-only audit of ABIDE I KKI phenotypic data and its composite CSV copy.

Python 3.9 compatible. Reads original CSV files without modifying them.
Creates only audit outputs in the selected output directory.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_KKI = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_I_KKI\original_files\phenotypic_KKI.csv"
)
DEFAULT_COMPOSITE = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_I_KKI\original_files\Phenotypic_V1_0b_preprocessed1.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_i_kki_audit"
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


def clean_value_counts(series):
    counts = series.fillna("<MISSING>").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def missing_summary(df, columns):
    output = {}
    for col in columns:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            output[col] = {
                "missing": missing,
                "non_missing": int(len(df) - missing),
                "missing_percent": round((missing / len(df) * 100), 3) if len(df) else None,
            }
    return output


def compare_kki_rows(kki_df, composite_df):
    result = {
        "composite_has_site_id": "SITE_ID" in composite_df.columns,
        "composite_kki_rows": 0,
        "common_columns_compared": [],
        "participant_id_sets_match": False,
        "row_value_comparison": "not_performed",
        "mismatched_cells": None,
        "notes": [],
    }

    if "SITE_ID" not in composite_df.columns or "SUB_ID" not in kki_df.columns or "SUB_ID" not in composite_df.columns:
        result["notes"].append("Required SITE_ID or SUB_ID column is missing; comparison could not be completed.")
        return result

    comp_kki = composite_df[
        composite_df["SITE_ID"].astype(str).str.strip().str.upper() == "KKI"
    ].copy()
    result["composite_kki_rows"] = int(len(comp_kki))

    ids_a = set(kki_df["SUB_ID"].dropna().astype(str).str.strip())
    ids_b = set(comp_kki["SUB_ID"].dropna().astype(str).str.strip())
    result["participant_id_sets_match"] = ids_a == ids_b
    result["kki_only_ids"] = sorted(ids_a - ids_b)
    result["composite_only_ids"] = sorted(ids_b - ids_a)

    common = [c for c in kki_df.columns if c in comp_kki.columns]
    result["common_columns_compared"] = common

    if not result["participant_id_sets_match"]:
        result["notes"].append("Participant ID sets differ; cell-level comparison is not reliable.")
        return result

    # Align rows by participant ID and compare shared fields. Pandas treats
    # paired missing values as equal for this audit.
    left = kki_df[common].copy()
    right = comp_kki[common].copy()
    left["SUB_ID"] = left["SUB_ID"].astype(str).str.strip()
    right["SUB_ID"] = right["SUB_ID"].astype(str).str.strip()
    left = left.set_index("SUB_ID").sort_index()
    right = right.set_index("SUB_ID").sort_index()

    shared = [c for c in common if c != "SUB_ID"]
    mismatches = {}
    total_mismatched_cells = 0
    for col in shared:
        a = left[col]
        b = right[col]
        # Compare numerically when both columns are numeric-compatible; this
        # avoids false mismatches such as 98 versus 98.0. Otherwise compare
        # trimmed strings. Paired missing values count as equal.
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        numeric_mask = a.notna() & b.notna()
        if numeric_mask.any() and a_num[numeric_mask].notna().all() and b_num[numeric_mask].notna().all():
            equal = pd.Series(True, index=a.index)
            equal.loc[numeric_mask] = (a_num.loc[numeric_mask] - b_num.loc[numeric_mask]).abs() <= 1e-9
            equal.loc[a.isna() ^ b.isna()] = False
            diff = ~equal
        else:
            a_norm = a.astype("string").fillna("<NA>").str.strip()
            b_norm = b.astype("string").fillna("<NA>").str.strip()
            diff = a_norm != b_norm
        count = int(diff.sum())
        if count:
            mismatches[col] = count
            total_mismatched_cells += count

    result["row_value_comparison"] = "completed"
    result["mismatched_cells"] = total_mismatched_cells
    result["mismatched_columns"] = mismatches
    result["all_shared_values_match"] = total_mismatched_cells == 0
    return result


def main():
    parser = argparse.ArgumentParser(description="Read-only ABIDE I KKI phenotypic data audit.")
    parser.add_argument("--kki-csv", type=Path, default=DEFAULT_KKI)
    parser.add_argument("--composite-csv", type=Path, default=DEFAULT_COMPOSITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    kki_path = args.kki_csv.expanduser().resolve()
    composite_path = args.composite_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not kki_path.is_file():
        raise FileNotFoundError("KKI CSV not found: {}".format(kki_path))
    if not composite_path.is_file():
        raise FileNotFoundError("Composite CSV not found: {}".format(composite_path))

    source_hashes_before = {
        "kki_csv_sha256": sha256_file(kki_path),
        "composite_csv_sha256": sha256_file(composite_path),
    }

    kki_raw = pd.read_csv(kki_path)
    composite_raw = pd.read_csv(composite_path)
    kki = normalize_columns(kki_raw)
    composite = normalize_columns(composite_raw)

    required = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ"]
    missing_required = [c for c in required if c not in kki.columns]
    if missing_required:
        raise ValueError("KKI CSV is missing required columns: {}".format(missing_required))

    for col in ["AGE_AT_SCAN", "DX_GROUP", "SEX", "FIQ", "VIQ", "PIQ", "HANDEDNESS_SCORES"]:
        if col in kki.columns:
            kki[col] = pd.to_numeric(kki[col], errors="coerce")

    age = kki["AGE_AT_SCAN"]
    age_subset = kki[(age >= 8) & (age < 12)].copy()

    # Validate expected categorical codes against the uploaded ABIDE legend.
    dx_values = sorted(kki["DX_GROUP"].dropna().unique().tolist())
    sex_values = sorted(kki["SEX"].dropna().unique().tolist())
    handedness_values = clean_value_counts(kki["HANDEDNESS_CATEGORY"]) if "HANDEDNESS_CATEGORY" in kki.columns else {}

    duplicate_id_rows = int(kki["SUB_ID"].duplicated(keep=False).sum())
    missing_id_rows = int(kki["SUB_ID"].isna().sum())
    duplicate_ids = sorted(
        kki.loc[kki["SUB_ID"].duplicated(keep=False), "SUB_ID"].dropna().astype(str).unique().tolist()
    )

    age_group = {}
    if len(age_subset):
        age_group = {
            "records": int(len(age_subset)),
            "age_min": float(age_subset["AGE_AT_SCAN"].min()),
            "age_max": float(age_subset["AGE_AT_SCAN"].max()),
            "age_mean": round(float(age_subset["AGE_AT_SCAN"].mean()), 4),
            "dx_group_counts": clean_value_counts(age_subset["DX_GROUP"]),
            "sex_counts": clean_value_counts(age_subset["SEX"]),
            "handedness_category_counts": clean_value_counts(age_subset["HANDEDNESS_CATEGORY"])
                if "HANDEDNESS_CATEGORY" in age_subset.columns else {},
            "fiq_non_missing": int(age_subset["FIQ"].notna().sum()),
            "fiq_missing": int(age_subset["FIQ"].isna().sum()),
            "fiq_test_type_counts": clean_value_counts(age_subset["FIQ_TEST_TYPE"])
                if "FIQ_TEST_TYPE" in age_subset.columns else {},
        }

    key_columns = [
        "SUB_ID", "DX_GROUP", "DSM_IV_TR", "AGE_AT_SCAN", "SEX",
        "HANDEDNESS_CATEGORY", "HANDEDNESS_SCORES", "FIQ", "VIQ", "PIQ",
        "FIQ_TEST_TYPE", "VIQ_TEST_TYPE", "PIQ_TEST_TYPE",
        "ADI_R_SOCIAL_TOTAL_A", "ADI_R_VERBAL_TOTAL_BV", "ADI_RRB_TOTAL_C",
        "ADI_R_ONSET_TOTAL_D", "ADI_R_RSRCH_RELIABLE", "ADOS_MODULE",
        "ADOS_TOTAL", "ADOS_RSRCH_RELIABLE", "SRS_VERSION", "SRS_RAW_TOTAL",
        "SCQ_TOTAL", "AQ_TOTAL"
    ]

    comparison = compare_kki_rows(kki, composite)

    # Capture source hashes after reading to detect accidental modification.
    source_hashes_after = {
        "kki_csv_sha256": sha256_file(kki_path),
        "composite_csv_sha256": sha256_file(composite_path),
    }

    checks = {
        "required_columns_present": len(missing_required) == 0,
        "participant_ids_complete": missing_id_rows == 0,
        "participant_ids_unique": duplicate_id_rows == 0,
        "age_values_complete": int(kki["AGE_AT_SCAN"].isna().sum()) == 0,
        "diagnostic_labels_complete": int(kki["DX_GROUP"].isna().sum()) == 0,
        "diagnostic_codes_within_legend": set(dx_values).issubset({1, 2}),
        "sex_codes_within_legend": set(sex_values).issubset({1, 2}),
        "age_filtered_records_exist": len(age_subset) > 0,
        "source_files_unchanged": source_hashes_before == source_hashes_after,
        "composite_kki_participant_ids_match": comparison.get("participant_id_sets_match", False),
        "composite_shared_values_match": comparison.get("all_shared_values_match", False),
    }

    report = {
        "report_title": "ABIDE I KKI Phenotypic Dataset Read-Only Audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audit_scope": "Read-only data quality and source consistency checks; no model training.",
        "source_files": {
            "kki_csv": str(kki_path),
            "composite_csv": str(composite_path),
            "hashes_before": source_hashes_before,
            "hashes_after": source_hashes_after,
        },
        "source_dataset": {
            "kki_records": int(len(kki)),
            "kki_columns": int(len(kki.columns)),
            "composite_records": int(len(composite)),
            "composite_columns": int(len(composite.columns)),
            "kki_site_values": clean_value_counts(kki["SITE_ID"]) if "SITE_ID" in kki.columns else {},
        },
        "participant_quality": {
            "unique_participant_ids": int(kki["SUB_ID"].nunique(dropna=True)),
            "missing_participant_id_rows": missing_id_rows,
            "duplicate_participant_id_rows": duplicate_id_rows,
            "duplicate_participant_ids": duplicate_ids,
        },
        "diagnostic_group": {
            "legend_interpretation": {"1": "Autism", "2": "Control"},
            "observed_counts": clean_value_counts(kki["DX_GROUP"]),
            "observed_codes": dx_values,
            "dsm_iv_tr_counts": clean_value_counts(kki["DSM_IV_TR"]) if "DSM_IV_TR" in kki.columns else {},
        },
        "age": {
            "column": "AGE_AT_SCAN",
            "unit": "years",
            "full_cohort_min": float(kki["AGE_AT_SCAN"].min()),
            "full_cohort_max": float(kki["AGE_AT_SCAN"].max()),
            "selected_rule": "8 <= AGE_AT_SCAN < 12",
            "selected_subgroup": age_group,
            "records_outside_selected_age_range": int(len(kki) - len(age_subset)),
        },
        "feature_availability_full_cohort": missing_summary(kki, key_columns),
        "feature_availability_age_selected_subgroup": missing_summary(age_subset, key_columns),
        "category_distributions": {
            "sex_legend": {"1": "Male", "2": "Female"},
            "sex_counts_full_cohort": clean_value_counts(kki["SEX"]),
            "sex_counts_age_selected": clean_value_counts(age_subset["SEX"]),
            "handedness_legend": {
                "R": "Right", "L": "Left", "M": "Mixed", "Ambi": "Ambidextrous"
            },
            "handedness_counts_full_cohort": handedness_values,
        },
        "composite_file_cross_check": comparison,
        "validation_checks": checks,
        "all_validation_checks_passed": all(checks.values()),
        "important_limitations": [
            "This audit checks data structure and consistency, not clinical validity.",
            "DX_GROUP is a cohort grouping label, not a ReSense clinical outcome.",
            "Diagnostic assessment variables must not be used as predictors when DX_GROUP is the target.",
            "A public file URL does not by itself establish permission to use or redistribute the data.",
            "The 8-to-under-12 age subgroup is small and should be treated as exploratory.",
        ],
        "model_training_performed": False,
        "original_files_modified": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "abide_i_kki_audit_report.json"
    subgroup_path = output_dir / "abide_i_kki_age_8_to_under_12_audit_extract.csv"

    # The extract is an audit convenience copy, not a model-ready dataset.
    age_subset.to_csv(subgroup_path, index=False, encoding="utf-8-sig")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, ensure_ascii=False)

    print("\nABIDE I KKI READ-ONLY AUDIT")
    print("=" * 45)
    print("KKI records:", len(kki))
    print("KKI columns:", len(kki.columns))
    print("Age 8 to under 12:", len(age_subset))
    print("Age-filtered diagnostic groups:", age_group.get("dx_group_counts", {}))
    print("Age-filtered sex counts:", age_group.get("sex_counts", {}))
    print("Age-filtered FIQ missing:", age_group.get("fiq_missing", "N/A"))
    print("Composite KKI rows:", comparison.get("composite_kki_rows"))
    print("Composite participant IDs match:", comparison.get("participant_id_sets_match"))
    print("Composite shared values match:", comparison.get("all_shared_values_match"))
    print("All validation checks passed:", all(checks.values()))
    print("Original files unchanged:", checks["source_files_unchanged"])
    print("\nReport:", report_path)
    print("Age-filtered audit extract:", subgroup_path)
    print("\nNo model was trained. Original source files were not modified.")


if __name__ == "__main__":
    main()
