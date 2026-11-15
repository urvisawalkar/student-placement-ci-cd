"""Loads the saved model and predicts placement for a student."""
import joblib
import pandas as pd

from validate_data import FEATURES

MODEL_PATH = "placement_model.pkl"


def predict(cgpa, attendance, coding_score, projects, internship):
    model = joblib.load(MODEL_PATH)
    student = pd.DataFrame(
        [[cgpa, attendance, coding_score, projects, internship]], columns=FEATURES
    )
    return int(model.predict(student)[0])


if __name__ == "__main__":
    samples = [(8.5, 92, 85, 3, 1), (6.0, 62, 45, 1, 0)]
    for s in samples:
        result = "PLACED" if predict(*s) == 1 else "NOT PLACED"
        print(f"Student {s} -> {result}")