"""
ReSense - ABIDE II KKI Phenotypic Metadata Audit
------------------------------------------------
Purpose:
    Audit the official KKI phenotypic CSV and the 8 <= age < 12 subgroup.

This script:
    - Does NOT modify the original CSV.
    - Does NOT train a machine-learning model.
    - Does NOT create a modelling dataset.
    - Writes a JSON audit report to the separate processed directory.

Run from PowerShell:
    python audit_abide_kki.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# Update SOURCE_FILE only if your downloaded file has a different name/location.
SOURCE_FILE = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI"
    r"\original_files\ABIDEII-KKI_1.csv"
)

OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\ABIDE_II_KKI"
)

AGE_MIN = 8.0
AGE_MAX_EXCLUSIVE = 12.0

ID_COLUMN = "SUB_ID"
TARGET_COLUMN = "DX_GROUP"
AGE_COLUMN_CANDIDATE = "AGE_AT_SCAN"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_column(columns, requested_name):
    """Find a column while ignoring accidental leading/trailing spaces."""
    matches = [column for column in columns if str(column).strip() == requested_name]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one column named {!r} after trimming spaces; found: {}"
            .format(requested_name, matches)
        )
    return matches[0]


def json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            "Source CSV not found:\n{}\n"
            "Check SOURCE_FILE at the top of this script.".format(SOURCE_FILE)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the original column names intact; resolve known names safely.
    data = pd.read_csv(SOURCE_FILE)
    id_col = find_column(data.columns, ID_COLUMN)
    target_col = find_column(data.columns, TARGET_COLUMN)
    age_col = find_column(data.columns, AGE_COLUMN_CANDIDATE)

    data[age_col] = pd.to_numeric(data[age_col], errors="coerce")

    child = data.loc[
        data[age_col].ge(AGE_MIN) & data[age_col].lt(AGE_MAX_EXCLUSIVE)
    ].copy()

    if child.empty:
        raise ValueError("No participants found in the selected age range.")

    # Group counts and validation
    group_counts = {
        str(key): int(value)
        for key, value in child[target_col].value_counts(dropna=False).items()
    }

    # Legend coding: DX_GROUP 1=Autism; 2=Control.
    unexpected_groups = sorted(
        str(value)
        for value in child[target_col].dropna().unique()
        if value not in (1, 2, 1.0, 2.0)
    )

    # Missingness across every column in the selected subgroup.
    missing_counts = child.isna().sum()
    missing_summary = [
        {
            "column": str(column),
            "missing_count": int(count),
            "missing_percent": round(float(count) / len(child) * 100, 2),
        }
        for column, count in missing_counts.items()
        if count > 0
    ]
    missing_summary.sort(
        key=lambda item: (-item["missing_count"], item["column"])
    )

    # Identify diagnostic/assessment fields for human review.
    # These are risk flags, not a claim that every field caused the label.
    leakage_terms = (
        "ADI_R", "ADOS", "SRS", "SCQ", "AQ_TOTAL",
        "CASI_AUTISTIC", "CSI_AUTISTIC", "PDD_DSM",
        "ASD_DSM", "DX_GROUP"
    )
    flagged_columns = [
        str(column)
        for column in data.columns
        if any(term in str(column).upper() for term in leakage_terms)
    ]

    # Important identity/label/age checks.
    duplicate_ids = int(child[id_col].duplicated(keep=False).sum())
    missing_ids = int(child[id_col].isna().sum())
    missing_target = int(child[target_col].isna().sum())

    age_summary = {
        "count": int(child[age_col].count()),
        "minimum": float(child[age_col].min()),
        "maximum": float(child[age_col].max()),
        "mean": round(float(child[age_col].mean()), 4),
    }

    report = {
        "project": "ReSense",
        "audit_name": "ABIDE II KKI phenotypic metadata audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(SOURCE_FILE),
        "source_sha256": sha256_file(SOURCE_FILE),
        "source_file_size_bytes": int(SOURCE_FILE.stat().st_size),
        "source_records": int(len(data)),
        "source_columns": int(len(data.columns)),
        "selected_age_range": {
            "minimum_inclusive_years": AGE_MIN,
            "maximum_exclusive_years": AGE_MAX_EXCLUSIVE,
        },
        "selected_records": int(len(child)),
        "selected_columns": int(len(child.columns)),
        "resolved_column_names": {
            "participant_id": str(id_col),
            "diagnostic_group": str(target_col),
            "age": str(age_col),
        },
        "age_summary_years": age_summary,
        "diagnostic_group_coding_from_legend": {
            "1": "Autism",
            "2": "Control",
        },
        "diagnostic_group_counts": group_counts,
        "unexpected_diagnostic_group_values": unexpected_groups,
        "data_quality_checks": {
            "missing_participant_ids": missing_ids,
            "participant_id_duplicate_rows": duplicate_ids,
            "missing_diagnostic_group_labels": missing_target,
            "age_missing_in_source": int(data[age_col].isna().sum()),
            "selected_subgroup_not_empty": bool(len(child) > 0),
        },
        "missing_values_in_selected_subgroup": missing_summary,
        "diagnostic_or_assessment_fields_for_leakage_review": flagged_columns,
        "modelling_status": "No model trained; no modelling dataset created.",
        "source_integrity": "Original CSV read only; not modified by this script.",
        "interpretation_warning": (
            "DX_GROUP is the repository's diagnostic group field. "
            "Assessment variables must be reviewed for target leakage and "
            "their role in diagnostic assignment before predictor selection. "
            "This audit does not establish clinical validity or authorize "
            "model training."
        ),
    }

    output_file = OUTPUT_DIR / "abide_ii_kki_metadata_audit.json"
    with output_file.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=4, ensure_ascii=False, default=json_safe)

    print("\nRESENSE ABIDE II KKI METADATA AUDIT")
    print("=" * 48)
    print("Source records: {}".format(len(data)))
    print("Source columns: {}".format(len(data.columns)))
    print("Selected age range: 8 <= age < 12 years")
    print("Selected participants: {}".format(len(child)))
    print("\nDiagnostic group counts (DX_GROUP):")
    for code, count in sorted(group_counts.items()):
        label = {"1": "Autism", "2": "Control"}.get(code, "Unexpected/missing")
        print("  {} ({}): {}".format(label, code, count))

    print("\nData quality:")
    print("  Missing participant IDs: {}".format(missing_ids))
    print("  Duplicate-ID rows: {}".format(duplicate_ids))
    print("  Missing group labels: {}".format(missing_target))
    print("  Missing age values in source: {}".format(data[age_col].isna().sum()))
    print("  Assessment/diagnostic fields flagged: {}".format(len(flagged_columns)))
    print("\nAudit report saved to:")
    print(output_file)
    print("\nNo original data was modified. No model was trained.")


if __name__ == "__main__":
    main()
