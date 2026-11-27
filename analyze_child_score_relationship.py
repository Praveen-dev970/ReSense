
import pandas as pd

FILE = (
    r"C:\ReSenseResearchData\questionnaire"
    r"\processed\child\child_processed.csv"
)

df = pd.read_csv(FILE)

questions = ["A{}_Score".format(i) for i in range(1, 11)]

df["Calculated_Score"] = df[questions].sum(axis=1)

print("\nRESENSE CHILD DATASET: SCORE AND TARGET ANALYSIS")
print("=" * 55)

print("\n1. SCORE DISTRIBUTION BY TARGET")

print(
    df.groupby("Class/ASD")["Calculated_Score"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(3)
    .to_string()
)

print("\n2. SCORE FREQUENCY BY TARGET")

print(
    pd.crosstab(
        df["Calculated_Score"],
        df["Class/ASD"]
    ).to_string()
)

print("\n3. TARGET DISTRIBUTION BY SCORE")

print(
    pd.crosstab(
        df["Calculated_Score"],
        df["Class/ASD"],
        normalize="index"
    ).round(3).to_string()
)

print("\n4. EXACT SCORE-TARGET AGREEMENT WITH THRESHOLD > 3")

expected = df["Calculated_Score"].gt(3).map({
    True: "YES",
    False: "NO"
})

actual = df["Class/ASD"].astype(str).str.strip().str.upper()

print("Matching records:", int((expected == actual).sum()))
print("Mismatching records:", int((expected != actual).sum()))
print("Total records:", len(df))

print("\nAnalysis completed.")
