
import pandas as pd

DATA_FILE = (
    r"C:\ReSenseResearchData\questionnaire"
    r"\processed\questionnaire_binary_subset.csv"
)

df = pd.read_csv(DATA_FILE)

questions = ["A{}".format(i) for i in range(1, 11)]
target = "Class/ASD Traits "

print("\nRESENSE TODDLER QUESTIONNAIRE FEATURE ANALYSIS")
print("=" * 55)

# Target distribution
print("\n1. TARGET DISTRIBUTION")
print(df[target].value_counts().to_string())

# Response distribution
print("\n2. QUESTION RESPONSE DISTRIBUTION")

for question in questions:
    print("\n{}".format(question))
    print(df[question].value_counts(dropna=False).sort_index().to_string())

# Mean response by target
print("\n3. MEAN QUESTION RESPONSE BY TARGET")

print(
    df.groupby(target)[questions]
    .mean()
    .round(3)
    .to_string()
)

# Mean total score by target
df["Calculated_Score"] = df[questions].sum(axis=1)

print("\n4. CALCULATED SCORE BY TARGET")

print(
    df.groupby(target)["Calculated_Score"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(3)
    .to_string()
)

# Demographic summary
print("\n5. DEMOGRAPHIC SUMMARY")

for column in [
    "Age_Mons",
    "Sex",
    "Ethnicity",
    "Jaundice",
    "Family_mem_with_ASD"
]:
    if column in df.columns:
        print("\n{}".format(column))
        print(df[column].value_counts(dropna=False).to_string())

print("\nFeature analysis completed.")
