
import json
from pathlib import Path

import pandas as pd
from scipy.io import arff


INPUT_FILE = Path(
    r"C:\ReSenseResearchData\questionnaire\raw\UCI_Children"
    r"\original_files\Autism-Child-Data.arff"
)

QUESTION_COLUMNS = [
    "A{}_Score".format(i) for i in range(1, 11)
]

TARGET_COLUMN = "Class/ASD"
SCORE_COLUMN = "result"

DUPLICATE_COLUMNS = (
    QUESTION_COLUMNS
    + ["age", "gender", "jundice", "austim"]
)


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(str(INPUT_FILE))

    raw_data, metadata = arff.loadarff(str(INPUT_FILE))
    df = pd.DataFrame(raw_data)

    # Decode ARFF byte-string values.
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

    numeric_questions = df[QUESTION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    calculated_score = numeric_questions.sum(
        axis=1, min_count=10
    )

    stored_score = pd.to_numeric(
        df[SCORE_COLUMN], errors="coerce"
    )

    valid_questions = (
        numeric_questions.notna().all(axis=1)
        & numeric_questions.isin([0, 1]).all(axis=1)
    )

    score_mismatch = (
        valid_questions
        & stored_score.notna()
        & calculated_score.ne(stored_score)
    )

    labels = df[TARGET_COLUMN].astype("string").str.strip().str.upper()

    duplicate_groups = df.groupby(
        DUPLICATE_COLUMNS,
        dropna=False
    )[TARGET_COLUMN].nunique()

    conflicting_groups = duplicate_groups[
        duplicate_groups > 1
    ]

    report = {
        "source_file": str(INPUT_FILE),
        "records": int(len(df)),
        "attributes": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_values": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
        },
        "label_distribution": {
            str(k): int(v)
            for k, v in labels.value_counts(dropna=False).items()
        },
        "valid_question_records": int(valid_questions.sum()),
        "invalid_question_records": int((~valid_questions).sum()),
        "score_mismatch_records": int(score_mismatch.sum()),
        "missing_stored_scores": int(stored_score.isna().sum()),
        "conflicting_duplicate_groups": int(len(conflicting_groups)),
        "duplicate_group_count": int(
            df.duplicated(
                subset=DUPLICATE_COLUMNS,
                keep=False
            ).groupby(
                [df[col] for col in DUPLICATE_COLUMNS],
                dropna=False
            ).ngroups
        )
    }

    output_file = INPUT_FILE.parent / "child_arff_audit_report.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\nCHILD ARFF DATASET AUDIT")
    print("=" * 40)
    print("Records:", report["records"])
    print("Attributes:", report["attributes"])

    print("\nLabel distribution:")
    print(pd.Series(report["label_distribution"]).to_string())

    print("\nMissing values:")
    print(pd.Series(report["missing_values"]).to_string())

    print("\nValid questionnaire records:", valid_questions.sum())
    print("Invalid questionnaire records:", (~valid_questions).sum())
    print("Score mismatches:", score_mismatch.sum())
    print("Missing stored scores:", stored_score.isna().sum())
    print("Conflicting duplicate groups:", len(conflicting_groups))

    print("\nAudit report:", output_file)


if __name__ == "__main__":
    main()
