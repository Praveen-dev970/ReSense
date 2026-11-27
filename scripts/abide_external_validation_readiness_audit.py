"""
ReSense — ABIDE I / ABIDE II external-validation readiness audit

READ-ONLY with respect to source datasets and the saved model:
- Does NOT train or tune a model.
- Does NOT call predict(), predict_proba(), or generate predictions.
- Does NOT modify source CSVs or the saved model.
- Writes only a JSON audit report to the selected output directory.

Run from PowerShell (inside the ReSense virtual environment):
python scripts\abide_external_validation_readiness_audit.py `
  --model "C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki_wisc_iv\evaluation\abide_ii_kki_wisc_iv_logistic_candidate.joblib" `
  --development "C:\ReSenseResearchData\questionnaire\processed\abide_ii_kki_wisc_iv\abide_ii_kki_wisc_iv_development_cohort.csv" `
  --external "C:\ReSenseResearchData\questionnaire\processed\abide_i_kki_audit\abide_i_kki_age_8_to_under_12_audit_extract.csv" `
  --output "C:\ReSenseResearchData\questionnaire\processed\external_validation_readiness"

If your files are in different locations, pass their actual paths with the same arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


FEATURES = ["AGE_AT_SCAN", "SEX", "FIQ"]
TARGET = "DX_GROUP"
ID_COLUMN = "SUB_ID"
EXPECTED_CLASSES = [1, 2]  # ABIDE coding: 1=Autism, 2=Control


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if pd.isna(value):
        return None
    return value


def inspect_csv(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}")

    frame = pd.read_csv(path)
    # Normalize surrounding whitespace in headers only; do not alter the file.
    original_columns = list(frame.columns)
    frame.columns = [str(column).strip() for column in frame.columns]

    info = {
        "path": str(path),
        "sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_column_names_after_whitespace_normalization": (
            pd.Index(frame.columns).duplicated().any()
        ),
        "headers_had_surrounding_whitespace": original_columns != list(frame.columns),
        "required_columns_present": {
            column: column in frame.columns
            for column in [*FEATURES, TARGET, ID_COLUMN]
        },
    }
    return frame, info


def values_as_int_list(series: pd.Series) -> list[int] | None:
    values = pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    if not values:
        return None
    return sorted(int(value) for value in values)


def inspect_dataset(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label}
    required = [*FEATURES, TARGET, ID_COLUMN]
    missing_columns = [column for column in required if column not in frame.columns]
    result["missing_required_columns"] = missing_columns
    if missing_columns:
        return result

    numeric = frame.copy()
    for column in [*FEATURES, TARGET]:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    result.update({
        "rows": int(len(frame)),
        "missing_values_in_required_fields": {
            column: int(numeric[column].isna().sum())
            for column in [*FEATURES, TARGET, ID_COLUMN]
        },
        "duplicate_participant_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "age_min": float(numeric["AGE_AT_SCAN"].min()) if numeric["AGE_AT_SCAN"].notna().any() else None,
        "age_max": float(numeric["AGE_AT_SCAN"].max()) if numeric["AGE_AT_SCAN"].notna().any() else None,
        "sex_codes": values_as_int_list(numeric["SEX"]),
        "target_codes": values_as_int_list(numeric[TARGET]),
        "target_counts": {
            str(key): int(value)
            for key, value in numeric[TARGET].value_counts(dropna=False).sort_index().items()
        },
        "fiq_min": float(numeric["FIQ"].min()) if numeric["FIQ"].notna().any() else None,
        "fiq_max": float(numeric["FIQ"].max()) if numeric["FIQ"].notna().any() else None,
        "feature_dtypes_as_read": {
            column: str(frame[column].dtype) for column in FEATURES
        },
        "fiq_test_type_columns": [
            column for column in frame.columns
            if "FIQ" in column.upper() and ("TYPE" in column.upper() or "TEST" in column.upper())
        ],
    })

    if result["fiq_test_type_columns"]:
        result["fiq_test_type_counts"] = {}
        for column in result["fiq_test_type_columns"]:
            result["fiq_test_type_counts"][column] = {
                str(key): int(value)
                for key, value in frame[column].value_counts(dropna=False).items()
            }

    result["age_within_expected_8_to_under_12"] = bool(
        numeric["AGE_AT_SCAN"].notna().all()
        and numeric["AGE_AT_SCAN"].ge(8).all()
        and numeric["AGE_AT_SCAN"].lt(12).all()
    )
    result["sex_codes_are_abide_1_or_2"] = bool(
        numeric["SEX"].notna().all()
        and set(values_as_int_list(numeric["SEX"]) or []).issubset({1, 2})
    )
    result["target_codes_are_abide_1_or_2"] = bool(
        numeric[TARGET].notna().all()
        and set(values_as_int_list(numeric[TARGET]) or []).issubset({1, 2})
    )
    result["required_fields_complete"] = bool(
        all(numeric[column].notna().all() for column in [*FEATURES, TARGET])
        and frame[ID_COLUMN].notna().all()
    )
    return result



def inspect_estimator(model):
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    details = {
        "object_type": f"{type(model).__module__}.{type(model).__name__}",
        "is_pipeline": isinstance(model, Pipeline),
        "check_is_fitted": False,
        "pipeline_steps": [],
    }

    if isinstance(model, Pipeline):
        for name, step in model.steps:
            details["pipeline_steps"].append({
                "name": name,
                "class": f"{type(step).__module__}.{type(step).__name__}",
            })

        final_estimator = model.steps[-1][1]

        try:
            check_is_fitted(final_estimator)
            details["check_is_fitted"] = True
        except Exception as exc:
            details["fitted_check_error"] = str(exc)

        details["model_classes"] = (
            getattr(final_estimator, "classes_", None).tolist()
            if hasattr(final_estimator, "classes_")
            else None
        )

        details["model_n_features_in"] = getattr(
            final_estimator, "n_features_in_", None
        )

        details["model_feature_names_in"] = getattr(
            final_estimator, "feature_names_in_", None
        )

    else:
        try:
            check_is_fitted(model)
            details["check_is_fitted"] = True
        except Exception as exc:
            details["fitted_check_error"] = str(exc)

        details["model_classes"] = (
            getattr(model, "classes_", None).tolist()
            if hasattr(model, "classes_")
            else None
        )

    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    for path, name in [
        (args.model, "Saved model"),
        (args.development, "Development cohort"),
        (args.external, "External cohort"),
    ]:
        if not path.is_file():
            parser.error(f"{name} file does not exist: {path}")

    report: dict[str, Any] = {
        "audit_name": "ABIDE I / ABIDE II external-validation readiness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model_training": False,
            "model_tuning": False,
            "predictions_generated": False,
            "source_files_modified": False,
            "model_file_modified": False,
            "only_output_written": "JSON audit report",
        },
        "important_interpretation": (
            "This is a technical readiness audit only. Passing checks do not establish "
            "clinical validity, diagnostic accuracy, or permission to use/share data."
        ),
        "paths": {
            "model": str(args.model),
            "development_cohort": str(args.development),
            "external_cohort": str(args.external),
            "output_directory": str(args.output),
        },
        "runtime": {
            "python_version": sys.version,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "checks": {},
        "unresolved_items": [],
    }

    try:
        dev_frame, dev_file_info = inspect_csv(args.development, "Development")
        ext_frame, ext_file_info = inspect_csv(args.external, "External")
        report["files"] = {
            "development": dev_file_info,
            "external": ext_file_info,
            "model": {
                "path": str(args.model),
                "sha256": sha256_file(args.model),
                "file_size_bytes": args.model.stat().st_size,
            },
        }
        report["datasets"] = {
            "development": inspect_dataset(dev_frame, "ABIDE II KKI WISC-IV development"),
            "external": inspect_dataset(ext_frame, "ABIDE I KKI external candidate"),
        }

        model = joblib.load(args.model)
        report["model_inspection"] = inspect_estimator(model)

        # Cross-dataset participant overlap check.
        overlap = None
        if ID_COLUMN in dev_frame.columns and ID_COLUMN in ext_frame.columns:
            dev_ids = set(dev_frame[ID_COLUMN].dropna().astype(str).str.strip())
            ext_ids = set(ext_frame[ID_COLUMN].dropna().astype(str).str.strip())
            overlap = sorted(dev_ids.intersection(ext_ids))
        report["participant_overlap"] = {
            "checked": overlap is not None,
            "overlap_count": len(overlap) if overlap is not None else None,
            "overlap_ids": overlap if overlap else [],
            "no_participant_overlap": bool(overlap == []) if overlap is not None else False,
        }

        dev = report["datasets"]["development"]
        ext = report["datasets"]["external"]
        model_info = report["model_inspection"]

        report["checks"] = {
            "development_has_required_fields": not dev.get("missing_required_columns"),
            "external_has_required_fields": not ext.get("missing_required_columns"),
            "development_required_fields_complete": dev.get("required_fields_complete", False),
            "external_required_fields_complete": ext.get("required_fields_complete", False),
            "development_age_range_valid": dev.get("age_within_expected_8_to_under_12", False),
            "external_age_range_valid": ext.get("age_within_expected_8_to_under_12", False),
            "development_sex_codes_valid": dev.get("sex_codes_are_abide_1_or_2", False),
            "external_sex_codes_valid": ext.get("sex_codes_are_abide_1_or_2", False),
            "development_target_codes_valid": dev.get("target_codes_are_abide_1_or_2", False),
            "external_target_codes_valid": ext.get("target_codes_are_abide_1_or_2", False),
            "development_ids_unique": dev.get("duplicate_participant_ids") == 0,
            "external_ids_unique": ext.get("duplicate_participant_ids") == 0,
            "no_participant_overlap": report["participant_overlap"]["no_participant_overlap"],
            "saved_model_is_fitted": model_info.get("check_is_fitted", False),
            "saved_model_is_pipeline": model_info.get("is_pipeline", False),
            "saved_model_classes_match_abide_codes": (
                model_info.get("model_classes") is not None
                and sorted(int(x) for x in model_info["model_classes"]) == EXPECTED_CLASSES
            ),
        }

        # The two FIQ test-type labels are not assumed equivalent by this script.
        report["fiq_compatibility"] = {
            "development_test_type_counts": dev.get("fiq_test_type_counts"),
            "external_test_type_counts": ext.get("fiq_test_type_counts"),
            "status": "DOCUMENTATION_REVIEW_REQUIRED",
            "note": (
                "ABIDE I may label the measure WISC_IV_FULL while ABIDE II uses WISC-IV. "
                "This script does not assert equivalence. Confirm the official cohort "
                "documentation/test administration metadata before external evaluation."
            ),
        }
        report["unresolved_items"].append(
            "Confirm from authoritative ABIDE documentation that the ABIDE I FIQ "
            "measure is sufficiently comparable to the ABIDE II WISC-IV FIQ measure."
        )
        report["unresolved_items"].append(
            "Confirm applicable ABIDE usage agreement, non-commercial research terms, "
            "attribution, institutional approval, and any publication requirements."
        )
        report["unresolved_items"].append(
            "Freeze and document the external-evaluation protocol and decision threshold "
            "before generating any external-cohort predictions."
        )
        report["unresolved_items"].append(
            "The audit checks schema and fitted metadata only; it does not establish "
            "that the external cohort is an independent-site validation cohort. Both "
            "cohorts are from KKI."
        )

        passed = sum(value is True for value in report["checks"].values())
        total = len(report["checks"])
        report["summary"] = {
            "checks_passed": passed,
            "checks_total": total,
            "all_technical_checks_passed": passed == total,
            "readiness_status": (
                "READY_FOR_PROTOCOL_REVIEW_ONLY"
                if passed == total
                else "NOT_READY_FIX_FAILED_CHECKS_FIRST"
            ),
            "warning": (
                "Even if all technical checks pass, do not generate external predictions "
                "until FIQ comparability, data-use conditions, and a frozen evaluation "
                "protocol are documented."
            ),
        }

    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        report["summary"] = {
            "all_technical_checks_passed": False,
            "readiness_status": "AUDIT_FAILED",
        }

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "abide_external_validation_readiness_report.json"
    report_path.write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== ReSense External-Validation Readiness Audit ===")
    print(f"Report: {report_path}")
    print(f"Status: {report.get('summary', {}).get('readiness_status')}")
    if "checks" in report:
        for name, passed in report["checks"].items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if report.get("unresolved_items"):
        print("\nItems requiring documentation/review:")
        for item in report["unresolved_items"]:
            print(f"- {item}")
    if report.get("fatal_error"):
        print(f"\nERROR: {report['fatal_error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
