# ============================================================
# STEP 1: SYNTHETIC BANK LOAN DATASET GENERATION
# ============================================================
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. SET RANDOM SEED
# ------------------------------------------------------------
np.random.seed(42)
# Number of customer records
NUMBER_OF_RECORDS = 1000

# ------------------------------------------------------------
# 2. GENERATE CUSTOMER INFORMATION
# ------------------------------------------------------------
age = np.random.randint(21, 66, NUMBER_OF_RECORDS)
income = np.random.lognormal(
    mean=11.0,
    sigma=0.45,
    size=NUMBER_OF_RECORDS
)

employment_type = np.random.choice(
    ["Salaried", "Self-Employed", "Business", "Unemployed"],
    size=NUMBER_OF_RECORDS,
    p=[0.50, 0.25, 0.15, 0.10]
)

education_level = np.random.choice(
    ["High School", "Graduate", "Postgraduate", "Doctorate"],
    size=NUMBER_OF_RECORDS,
    p=[0.25, 0.40, 0.25, 0.10]
)

credit_score = np.clip(
    np.random.normal(680, 65, NUMBER_OF_RECORDS),
    300,
    850
).round().astype(int)

# ------------------------------------------------------------
# 3. GENERATE LOAN INFORMATION
# ------------------------------------------------------------
loan_term = np.random.choice(
    [12, 24, 36, 48, 60],
    size=NUMBER_OF_RECORDS
)

interest_rate = np.clip(
    14 - (credit_score - 650) * 0.025
    + np.random.normal(0, 1.5, NUMBER_OF_RECORDS),
    5,
    25
).round(2)

existing_debt = (
    income * np.random.uniform(0.05, 0.45, NUMBER_OF_RECORDS)
).round(2)

debt_to_income = (
    existing_debt / income
).round(3)

# Generate requested loan amount.
# Loan amount depends partly on income and credit score.
loan_multiplier = np.random.uniform(
    0.4, 2.0, NUMBER_OF_RECORDS
)

loan_amount = (
    income
    * loan_multiplier
    * (credit_score / 700)
    * np.random.uniform(0.85, 1.15, NUMBER_OF_RECORDS)
).round(2)

# ------------------------------------------------------------
# 4. GENERATE LOAN DATES
# ------------------------------------------------------------
loan_date = (
    pd.Timestamp("2024-01-01")
    + pd.to_timedelta(
        np.random.randint(0, 730, NUMBER_OF_RECORDS),
        unit="D"
    )
)

# ------------------------------------------------------------
# 5. GENERATE LOAN DEFAULT TARGET
# ------------------------------------------------------------
# Default probability is influenced by:
# - Credit score
# - Debt-to-income ratio
# - Interest rate
# - Employment status
unemployment_risk = (
    employment_type == "Unemployed"
).astype(int)

default_logit = (
    -2.6
    + ((700 - credit_score) / 100) * 0.8
    + debt_to_income * 2.0
    + (interest_rate - 10) * 0.12
    + unemployment_risk * 0.5
)

default_probability = (
    1 / (1 + np.exp(-default_logit))
)

loan_default = np.random.binomial(
    1,
    default_probability
)

# ------------------------------------------------------------
# 6. CREATE THE DATAFRAME
# ------------------------------------------------------------
loan_data = pd.DataFrame({
    "age": age,
    "income": income.round(2),
    "employment_type": employment_type,
    "education_level": education_level,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "interest_rate": interest_rate,
    "loan_term": loan_term,
    "loan_date": loan_date,
    "existing_debt": existing_debt,
    "debt_to_income": debt_to_income,
    "loan_default": loan_default
})

# ------------------------------------------------------------
# 7. INTRODUCE MISSING VALUES
# ------------------------------------------------------------
# Missing values are introduced only in input features.
# Target columns remain complete.
missing_columns = [
    "age",
    "income",
    "employment_type",
    "education_level",
    "credit_score",
    "interest_rate",
    "existing_debt"
]

for column in missing_columns:
    missing_indices = loan_data.sample(
        frac=0.05,
        random_state=42
    ).index

    loan_data.loc[missing_indices, column] = np.nan

# ------------------------------------------------------------
# 8. SAVE THE DATASET
# ------------------------------------------------------------
project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

data_folder.mkdir(parents=True, exist_ok=True)

file_path = data_folder / "loan_data.csv"

loan_data.to_csv(file_path, index=False)

# ------------------------------------------------------------
# 9. DISPLAY DATASET INFORMATION
# ------------------------------------------------------------
print("\n========== DATASET CREATED SUCCESSFULLY ==========")

print("\nDataset Shape:")
print(loan_data.shape)

print("\nFirst 5 Records:")
print(loan_data.head())

print("\nColumn Information:")
loan_data.info()

print("\nMissing Values:")
print(loan_data.isnull().sum())

print("\nLoan Default Distribution:")
print(loan_data["loan_default"].value_counts())

print("\nLoan Default Percentage:")
print(
    (loan_data["loan_default"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nLoan Amount Statistics:")
print(loan_data["loan_amount"].describe())

print("\nDataset saved successfully at:")
print(file_path)