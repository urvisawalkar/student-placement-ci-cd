"""Trains the Student Placement model and applies the accuracy quality gate."""
import json
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from validate_data import FEATURES, TARGET, validate

MODEL_PATH = "placement_model.pkl"
ACCURACY_THRESHOLD = 0.80


def train():
    df, errors = validate()
    if errors:
        raise ValueError(f"Invalid dataset: {errors}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_PATH)
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy, "threshold": ACCURACY_THRESHOLD}, f, indent=2)
    return model, accuracy


if __name__ == "__main__":
    model, accuracy = train()
    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Model saved as {MODEL_PATH}")
    if accuracy < ACCURACY_THRESHOLD:
        print(f"QUALITY GATE FAILED: accuracy below {ACCURACY_THRESHOLD:.0%}")
        sys.exit(1)
    print(f"QUALITY GATE PASSED: accuracy >= {ACCURACY_THRESHOLD:.0%}")