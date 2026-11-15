"""Data validation step of the CI pipeline - fails on invalid data."""
import sys

import pandas as pd

DATA_PATH = "data/student_placement.csv"
FEATURES = ["CGPA", "Attendance", "CodingScore", "Projects", "Internship"]
TARGET = "Placement"


def validate(path=DATA_PATH):
    df = pd.read_csv(path)
    errors = []
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    else:
        if df.isnull().values.any():
            errors.append("Dataset contains missing values")
        if not df["CGPA"].between(0, 10).all():
            errors.append("CGPA must be between 0 and 10")
        if not df["Attendance"].between(0, 100).all():
            errors.append("Attendance must be between 0 and 100")
        if not df["CodingScore"].between(0, 100).all():
            errors.append("CodingScore must be between 0 and 100")
        if not df["Internship"].isin([0, 1]).all():
            errors.append("Internship must be 0 or 1")
        if not df[TARGET].isin([0, 1]).all():
            errors.append("Placement must be 0 or 1")
    return df, errors


if __name__ == "__main__":
    df, errors = validate()
    if errors:
        for e in errors:
            print("VALIDATION ERROR:", e)
        sys.exit(1)
    print(f"Data validation passed: {len(df)} records, {len(df.columns)} columns")