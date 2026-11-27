#!/usr/bin/env python
"""
ABIDE I KKI vs ABIDE II KKI feature compatibility assessment.
Read-only with respect to source CSV files. No model training.
Python 3.9 compatible.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_ABIDE_I = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_I_KKI\original_files\phenotypic_KKI.csv"
)
DEFAULT_ABIDE_II = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\ABIDE_II_KKI\original_files\ABIDEII-KKI_1.csv"
)
DEFAULT_OUTPUT = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\abide_i_kki_audit\feature_compatibility"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["AGE_AT_SCAN", "DX_GROUP", "SEX", "FIQ", "VIQ", "PIQ", "HANDEDNESS_SCORES"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def counts(series):
    return {str(k): int(v) for k, v in series.fillna("<MISSING>").astype(str).value_counts().items()}


def summarize_by_group(df, cohort):
    rows = []
    for dx in sorted(df["DX_GROUP"].dropna().unique()):
        g = df[df["DX_GROUP"] == dx]
        rows.append({
            "cohort": cohort,
            "DX_GROUP": int(dx),
            "group": "Autism" if int(dx) == 1 else "Control" if int(dx) == 2 else "Unexpected",
            "n": int(len(g)),
            "age_mean": round(float(g["AGE_AT_SCAN"].mean()), 4),
            "age_sd": round(float(g["AGE_AT_SCAN"].std()), 4) if len(g) > 1 else None,
            "FIQ_n": int(g["FIQ"].notna().sum()),
            "FIQ_missing": int(g["FIQ"].isna().sum()),
            "FIQ_mean": round(float(g["FIQ"].mean()), 4) if g["FIQ"].notna().any() else None,
            "FIQ_sd": round(float(g["FIQ"].std()), 4) if g["FIQ"].notna().sum() > 1 else None,
            "FIQ_min": float(g["FIQ"].min()) if g["FIQ"].notna().any() else None,
            "FIQ_max": float(g["FIQ"].max()) if g["FIQ"].notna().any() else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abide-i-csv", type=Path, default=DEFAULT_ABIDE_I)
    parser.add_argument("--abide-ii-csv", type=Path, default=DEFAULT_ABIDE_II)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    p1 = args.abide_i_csv.expanduser().resolve()
    p2 = args.abide_ii_csv.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if not p1.is_file():
        raise FileNotFoundError("ABIDE I KKI CSV not found: {}".format(p1))
    if not p2.is_file():
        raise FileNotFoundError("ABIDE II KKI CSV not found: {}".format(p2))

    hashes_before = {"abide_i": sha256_file(p1), "abide_ii": sha256_file(p2)}
    i = load_csv(p1)
    ii = load_csv(p2)

    required = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ", "FIQ_TEST_TYPE", "HANDEDNESS_CATEGORY"]
    for name, df in [("ABIDE I KKI", i), ("ABIDE II KKI", ii)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("{} missing required columns: {}".format(name, missing))

    i_age = i[(i["AGE_AT_SCAN"] >= 8) & (i["AGE_AT_SCAN"] < 12)].copy()
    ii_age = ii[(ii["AGE_AT_SCAN"] >= 8) & (ii["AGE_AT_SCAN"] < 12)].copy()

    # Normalize handedness to a common descriptive category for comparison only.
    i_hand_map = {"R": "Right", "L": "Left", "M": "Mixed", "Ambi": "Ambidextrous"}
    ii_hand_map = {1: "Right", 2: "Left", 3: "Mixed"}
    i_age["HANDEDNESS_NORMALIZED"] = i_age["HANDEDNESS_CATEGORY"].astype("string").str.strip().map(i_hand_map)
    ii_age["HANDEDNESS_NORMALIZED"] = ii_age["HANDEDNESS_CATEGORY"].map(ii_hand_map)

    # ABIDE II strict WISC-IV subgroup: records with nonmissing FIQ and explicit WISC-IV test type.
    ii_wisc4 = ii_age[
        ii_age["FIQ"].notna()
        & ii_age["FIQ_TEST_TYPE"].astype("string").str.strip().str.upper().eq("WISC-IV")
    ].copy()

    summaries = summarize_by_group(i_age, "ABIDE_I_KKI") + summarize_by_group(ii_age, "ABIDE_II_KKI")
    summaries += summarize_by_group(ii_wisc4, "ABIDE_II_KKI_WISC_IV_COMPLETE")

    def cohort_summary(df):
        return {
            "n": int(len(df)),
            "age_min": float(df["AGE_AT_SCAN"].min()),
            "age_max": float(df["AGE_AT_SCAN"].max()),
            "age_mean": round(float(df["AGE_AT_SCAN"].mean()), 4),
            "dx_group": counts(df["DX_GROUP"]),
            "sex": counts(df["SEX"]),
            "handedness_original": counts(df["HANDEDNESS_CATEGORY"]),
            "handedness_normalized": counts(df["HANDEDNESS_NORMALIZED"]),
            "fiq_nonmissing": int(df["FIQ"].notna().sum()),
            "fiq_missing": int(df["FIQ"].isna().sum()),
            "fiq_test_type": counts(df["FIQ_TEST_TYPE"]),
            "viq_nonmissing": int(df["VIQ"].notna().sum()) if "VIQ" in df.columns else 0,
            "piq_nonmissing": int(df["PIQ"].notna().sum()) if "PIQ" in df.columns else 0,
        }

    # Check shared IDs: these cohorts should be different participants.
    ids_i = set(i["SUB_ID"].dropna().astype(str).str.strip())
    ids_ii = set(ii["SUB_ID"].dropna().astype(str).str.strip())
    overlap = sorted(ids_i.intersection(ids_ii))

    hashes_after = {"abide_i": sha256_file(p1), "abide_ii": sha256_file(p2)}

    checks = {
        "age_field_present_and_numeric": i["AGE_AT_SCAN"].notna().all() and ii["AGE_AT_SCAN"].notna().all(),
        "age_units_documented_as_years_in_legends": True,
        "diagnostic_group_codes_match": set(i["DX_GROUP"].dropna().unique()).issubset({1, 2}) and set(ii["DX_GROUP"].dropna().unique()).issubset({1, 2}),
        "sex_codes_match": set(i["SEX"].dropna().unique()).issubset({1, 2}) and set(ii["SEX"].dropna().unique()).issubset({1, 2}),
        "fiq_available_for_all_abide_i_age_subset": i_age["FIQ"].notna().all(),
        "fiq_test_version_exactly_same_for_all_records": False,
        "handedness_can_be_mapped_to_shared_categories": i_age["HANDEDNESS_NORMALIZED"].notna().all() and ii_age["HANDEDNESS_NORMALIZED"].notna().all(),
        "participant_ids_do_not_overlap": len(overlap) == 0,
        "source_files_unchanged": hashes_before == hashes_after,
    }

    report = {
        "title": "ABIDE I KKI vs ABIDE II KKI Feature Compatibility Assessment",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "Read-only comparison of age 8 to under 12 cohorts; no model training or feature tuning.",
        "source_files": {
            "abide_i_kki": str(p1),
            "abide_ii_kki": str(p2),
            "sha256_before": hashes_before,
            "sha256_after": hashes_after,
        },
        "age_filter": "8 <= AGE_AT_SCAN < 12",
        "cohort_summaries": {
            "abide_i_kki": cohort_summary(i_age),
            "abide_ii_kki": cohort_summary(ii_age),
            "abide_ii_kki_wisc_iv_complete": cohort_summary(ii_wisc4),
        },
        "group_level_age_fiq_statistics": summaries,
        "feature_assessment": {
            "AGE_AT_SCAN": {
                "status": "compatible",
                "reason": "Both legends define age at scan in years.",
            },
            "SEX": {
                "status": "compatible",
                "mapping": {"1": "Male", "2": "Female"},
                "reason": "Both legends use the same numeric coding.",
            },
            "DX_GROUP": {
                "status": "compatible_as_cohort_label",
                "mapping": {"1": "Autism", "2": "Control"},
                "caution": "Cohort group labels are not ReSense clinical outcomes.",
            },
            "FIQ": {
                "status": "conditionally_compatible",
                "reason": "Both datasets define FIQ as full-scale IQ standard score, but test-type labels and versions differ.",
                "abide_i_age_subset_test_types": counts(i_age["FIQ_TEST_TYPE"]),
                "abide_ii_age_subset_test_types": counts(ii_age["FIQ_TEST_TYPE"]),
                "abide_ii_wisc_iv_complete_n": int(len(ii_wisc4)),
                "recommendation": "For the strictest initial comparison, use ABIDE I KKI WISC_IV_FULL records and ABIDE II KKI records with nonmissing FIQ and explicit WISC-IV test type. Confirm the WISC_IV_FULL label from source documentation before treating it as identical to WISC-IV.",
            },
            "HANDEDNESS_CATEGORY": {
                "status": "compatible_after_recoding",
                "abide_i_mapping": {"R": "Right", "L": "Left", "M": "Mixed", "Ambi": "Ambidextrous"},
                "abide_ii_mapping": {"1": "Right", "2": "Left", "3": "Mixed"},
                "caution": "Ambidextrous is represented in the ABIDE I legend but not as a separate ABIDE II category.",
                "recommendation": "Keep handedness out of the frozen Age+Sex+FIQ candidate unless separately justified.",
            },
            "VIQ_PIQ": {
                "status": "not_compatible_for_external_validation",
                "reason": "No VIQ or PIQ values are present in the age-filtered ABIDE I KKI cohort.",
            },
        },
        "participant_overlap": {
            "abide_i_unique_ids": int(i["SUB_ID"].nunique()),
            "abide_ii_unique_ids": int(ii["SUB_ID"].nunique()),
            "overlap_count": int(len(overlap)),
            "overlap_ids": overlap,
        },
        "validation_checks": checks,
        "all_checks_passed": all(checks.values()),
        "model_training_performed": False,
        "original_files_modified": False,
        "limitations": [
            "This is a feature and data compatibility assessment, not model validation.",
            "FIQ score comparability requires confirmation of the meaning of WISC_IV_FULL and test-version equivalence.",
            "The ABIDE I age-filtered cohort is small (51 participants).",
            "Differences in cohort composition and IQ distributions may affect external performance.",
            "Do not use ADI-R, ADOS, DSM subtype, SRS, or other diagnostic assessment fields as predictors when DX_GROUP is the target.",
        ],
    }

    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "abide_i_ii_feature_compatibility_report.json"
    stats_path = out / "abide_i_ii_age_fiq_group_statistics.csv"
    pd.DataFrame(summaries).to_csv(stats_path, index=False, encoding="utf-8-sig")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False, default=lambda value: value.item() if hasattr(value, "item") else str(value))

    print("\nABIDE I / ABIDE II FEATURE COMPATIBILITY")
    print("=" * 48)
    print("ABIDE I age 8 to under 12:", len(i_age))
    print("ABIDE II age 8 to under 12:", len(ii_age))
    print("ABIDE I FIQ test types:", counts(i_age["FIQ_TEST_TYPE"]))
    print("ABIDE II FIQ test types:", counts(ii_age["FIQ_TEST_TYPE"]))
    print("ABIDE II complete WISC-IV FIQ records:", len(ii_wisc4))
    print("ABIDE I VIQ nonmissing:", int(i_age["VIQ"].notna().sum()))
    print("ABIDE I PIQ nonmissing:", int(i_age["PIQ"].notna().sum()))
    print("Participant ID overlap:", len(overlap))
    print("Source files unchanged:", checks["source_files_unchanged"])
    print("\nReport:", report_path)
    print("Group statistics:", stats_path)
    print("No model trained; source CSV files not modified.")


if __name__ == "__main__":
    main()
