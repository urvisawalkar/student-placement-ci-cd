"""Automated tests executed by GitHub Actions using pytest."""
import os

import pandas as pd

from predict import predict
from train import ACCURACY_THRESHOLD, MODEL_PATH, train
from validate_data import validate


def test_dataset_is_valid():
    df, errors = validate()
    assert not df.empty
    assert errors == []


def test_validation_detects_bad_data(tmp_path):
    bad = pd.read_csv("data/student_placement.csv")
    bad.loc[0, "CGPA"] = 12  # invalid value
    bad_file = tmp_path / "bad.csv"
    bad.to_csv(bad_file, index=False)
    _, errors = validate(bad_file)
    assert len(errors) > 0


def test_model_meets_accuracy_threshold():
    _, accuracy = train()
    assert accuracy >= ACCURACY_THRESHOLD


def test_model_file_created():
    train()
    assert os.path.exists(MODEL_PATH)


def test_predictions_are_valid():
    train()
    assert predict(8.5, 92, 85, 3, 1) == 1   # strong profile -> placed
    assert predict(5.8, 60, 40, 1, 0) == 0   # weak profile -> not placed