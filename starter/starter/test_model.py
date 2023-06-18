"""
Unit test scripts for the model functions
Author: Moises Gonzalez
Date: 15/Jun/2023
"""

import pytest
import sys
import logging
import pandas as pd
from typing import Tuple
from pathlib import Path


MODEL_PATH = Path(__name__).resolve().parent.parent / "model"
DATA_PATH = Path(__name__).resolve().parent.parent / "data"
DATA_FILE = DATA_PATH / "census.csv"
TRAIN_DATA_FILE = DATA_PATH / "train_df.csv"
TEST_DATA_FILE = DATA_PATH / "test_df.csv"
ENCODER_FILE = MODEL_PATH / "encoder.pkl"
LB_FILE = MODEL_PATH / "label_binarizer.pkl"
MODEL_FILE = MODEL_PATH / "trained_model.pkl"

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def df():
    """
    Fixture to load dataset
    Returns
    -------
    Pandas dataframe
    """
    df = pd.read_csv(DATA_FILE)

    yield df


@pytest.fixture(scope="session")
def train_and_test_df() -> Tuple:
    """
    Fixture to load train and test data set
    Returns
    -------
    Train and test dataset
    """
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    return train_df, test_df


def test_imported_data():
    """
    Test that the data file is imported
    """
    assert DATA_FILE.exists(), f"File not found -> {DATA_FILE}"


def test_data_shape(df):
    """
    Test that the imported data file is not empty
    """
    assert df.shape[0] > 0, f"Number of rows must be greater than 0:{df.shape[0]}"
    assert df.shape[1] > 0, f"Number of column must be greater than 0:{df.shape[1]}"


def test_features_in_data(df):
    """
    Test that the features are present in the data
    """
    features = ["workclass", "education", "marital_status", "occupation",
                "relationship", "race", "sex", "native_country"]
    assert sorted(set(df.columns).intersection(features)) == sorted(features)


def test_model_files():
    """
    Test that the model files are generated and saved
    """

    assert ENCODER_FILE.exists(), f"Encoder file not found -> {ENCODER_FILE}"
    assert LB_FILE.exists(), f"Label binarizer file not found -> {LB_FILE}"
    assert MODEL_FILE.exists(), f"Model file not found -> {MODEL_FILE}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-vv", str(Path.cwd())]))
