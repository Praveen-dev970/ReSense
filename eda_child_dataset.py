"""
ReSense - UCI Child Dataset Exploratory Data Analysis (EDA)

Reads child_processed.csv only. Does not modify the original ARFF or
preprocessing outputs and does not train a machine-learning model.

Run from PowerShell:
    python eda_child_dataset.py

Outputs are written to:
    C:\\ReSenseResearchData\\questionnaire\\processed\\child\\eda
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\child"
    r"\child_processed.csv"
)
OUTPUT_DIR = Path(
    r"C:\ReSenseResearchData\questionnaire\processed\child\eda"
)

QUESTION_COLUMNS = ["A{}_Score".format(i) for i in range(1, 11)]
TARGET_COLUMN = "Class/ASD"
SCORE_COLUMN = "result"

# Original ARFF spellings are intentionally retained.
CATEGORICAL_COLUMNS = [
    "gender", "ethnicity", "jundice", "austim",
    "contry_of_res", "used_app_before", "age_desc", "relation"
]


def save_bar_chart(series, title, xlabel, ylabel, filename, rotate=0):
    counts = series.fillna("Missing").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=rotate)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    return counts


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            "Processed child CSV not found: {}".format(INPUT_CSV)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    required = QUESTION_COLUMNS + [
        "age", TARGET_COLUMN, SCORE_COLUMN
    ]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Required columns missing from processed CSV: {}".format(
                missing_columns
            )
        )

    # Normalize labels for descriptive summaries only.
    labels = df[TARGET_COLUMN].astype("string").str.strip().str.upper()

    # Numeric conversion for numerical summaries.
    age = pd.to_numeric(df["age"], errors="coerce")
    score = pd.to_numeric(df[SCORE_COLUMN], errors="coerce")
    questions = df[QUESTION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    # 1. General dataset profile.
    missing_by_column = df.isna().sum()
    profile = {
        "dataset": "Autism-Child-Data",
        "records": int(len(df)),
        "attributes": int(len(df.columns)),
        "target_column": TARGET_COLUMN,
        "target_distribution": {
            str(k): int(v)
            for k, v in labels.value_counts(dropna=False).items()
        },
        "missing_values_by_column": {
            str(k): int(v) for k, v in missing_by_column.items()
        },
        "total_missing_values": int(missing_by_column.sum()),
        "age_summary_years": {
            "count": int(age.count()),
            "missing": int(age.isna().sum()),
            "min": float(age.min()) if age.notna().any() else None,
            "median": float(age.median()) if age.notna().any() else None,
            "mean": float(age.mean()) if age.notna().any() else None,
            "max": float(age.max()) if age.notna().any() else None
        },
        "screening_score_summary": {
            "count": int(score.count()),
            "min": float(score.min()) if score.notna().any() else None,
            "mean": float(score.mean()) if score.notna().any() else None,
            "median": float(score.median()) if score.notna().any() else None,
            "max": float(score.max()) if score.notna().any() else None
        },
        "notes": [
            "This is descriptive exploratory analysis only.",
            "No machine-learning model was trained.",
            "Class/ASD is a screening-generated label, not an independent clinical diagnosis.",
            "Questionnaire-target associations are not independent evidence of diagnostic performance."
        ]
    }

    with open(
        OUTPUT_DIR / "child_eda_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(profile, file, indent=4, ensure_ascii=False)

    # 2. Missing-value summary.
    missing_table = pd.DataFrame({
        "column": missing_by_column.index,
        "missing_count": missing_by_column.values,
        "missing_percent": (
            missing_by_column.values / len(df) * 100
        ).round(2)
    })
    missing_table.to_csv(
        OUTPUT_DIR / "missing_values_summary.csv", index=False
    )

    # 3. Target distribution.
    target_counts = save_bar_chart(
        labels, "Child Dataset Target Distribution",
        "Class/ASD label", "Number of records",
        "target_distribution.png"
    )

    # 4. Age distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    age.dropna().plot(kind="hist", bins=15, ax=ax, edgecolor="black")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "age_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Individual question response distributions.
    question_counts = {}
    for column in QUESTION_COLUMNS:
        counts = questions[column].value_counts().reindex(
            [0, 1], fill_value=0
        )
        question_counts[column] = {
            str(k): int(v) for k, v in counts.items()
        }

    question_table = pd.DataFrame(question_counts).T
    question_table.columns = ["response_0", "response_1"]
    question_table.index.name = "question"
    question_table.to_csv(
        OUTPUT_DIR / "question_response_distribution.csv"
    )

    ax = question_table.plot(
        kind="bar", figsize=(11, 6), rot=0
    )
    ax.set_title("A1-A10 Response Distribution")
    ax.set_xlabel("Question")
    ax.set_ylabel("Number of records")
    ax.legend(title="Response")
    ax.figure.tight_layout()
    ax.figure.savefig(
        OUTPUT_DIR / "question_response_distribution.png", dpi=150
    )
    plt.close(ax.figure)

    # 6. Response-1 rate by target. Descriptive only: target is
    # generated from the screening questionnaire scoring process.
    response_rate = questions.assign(target=labels).groupby(
        "target", dropna=False
    )[QUESTION_COLUMNS].mean().T
    response_rate.index.name = "question"
    response_rate.to_csv(
        OUTPUT_DIR / "question_response_rate_by_target.csv"
    )

    ax = response_rate.plot(
        kind="bar", figsize=(12, 6), rot=0
    )
    ax.set_title(
        "Mean A1-A10 Response by Screening Label "
        "(Descriptive; Not Independent Validation)"
    )
    ax.set_xlabel("Question")
    ax.set_ylabel("Mean response (0-1)")
    ax.legend(title="Class/ASD")
    ax.figure.tight_layout()
    ax.figure.savefig(
        OUTPUT_DIR / "question_response_rate_by_target.png", dpi=150
    )
    plt.close(ax.figure)

    # 7. Score distribution by target (documents label construction).
    score_frame = pd.DataFrame({
        "score": score,
        "target": labels
    }).dropna()
    score_frequency = pd.crosstab(
        score_frame["score"], score_frame["target"]
    ).sort_index()
    score_frequency.to_csv(
        OUTPUT_DIR / "score_frequency_by_target.csv"
    )

    ax = score_frequency.plot(
        kind="bar", figsize=(10, 6), rot=0
    )
    ax.set_title(
        "Screening Score by Class/ASD Label "
        "(Label-Construction Relationship)"
    )
    ax.set_xlabel("Screening score")
    ax.set_ylabel("Number of records")
    ax.legend(title="Class/ASD")
    ax.figure.tight_layout()
    ax.figure.savefig(
        OUTPUT_DIR / "score_by_target.png", dpi=150
    )
    plt.close(ax.figure)

    # 8. Demographic category frequencies.
    demographic_summaries = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue
        counts = save_bar_chart(
            df[column], "{} Distribution".format(column),
            column, "Number of records",
            "demographic_{}.png".format(column),
            rotate=45
        )
        demographic_summaries[column] = {
            str(k): int(v) for k, v in counts.items()
        }

    with open(
        OUTPUT_DIR / "demographic_distributions.json",
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            demographic_summaries, file,
            indent=4, ensure_ascii=False
        )

    # 9. Numeric descriptive statistics.
    numeric_columns = ["age"] + QUESTION_COLUMNS + [SCORE_COLUMN]
    numeric_summary = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    ).describe().T
    numeric_summary.to_csv(
        OUTPUT_DIR / "numeric_descriptive_statistics.csv"
    )

    print("\nRESENSE CHILD DATASET EDA")
    print("=" * 48)
    print("Records:", len(df))
    print("Attributes:", len(df.columns))
    print("\nTarget distribution:")
    print(target_counts.to_string())
    print("\nMissing values:")
    print(missing_table.to_string(index=False))
    print("\nAge summary:")
    print(numeric_summary.loc["age"].to_string())
    print("\nScreening score by target:")
    print(score_frequency.to_string())
    print("\nEDA outputs saved to:")
    print(OUTPUT_DIR)
    print("\nNo machine-learning model was trained.")


if __name__ == "__main__":
    main()
