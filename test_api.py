"""
Unit test scripts for the API GET/POST methods
Author: Moises Gonzalez
Date: 21/Jun/2023
"""

import pytest
import logging
import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

OK_STATUS_CODE = 200


def test_welcome():
    """
    Test that the welcome message is received
    """

    request = client.get("/")
    assert request.status_code == OK_STATUS_CODE
    assert request.json()["message"] == "Welcome!"


def test_prediction_greater():
    """
    Test a model prediction
    """
    sample = {
        "age": 31,
        "workclass": "Local-gov",
        "fnlgt": 446358,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Never-married",
        "occupation": "Prof-speciality",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "Mexico"
    }

    request = client.post("/predictions/", data=json.dumps(sample))

    assert request.status_code == OK_STATUS_CODE
    assert request.json()["prediction"] == ">50k"


def test_prediction_lower():
    sample = {
        "age": 32,
        "workclass": "Private",
        "fnlgt": 27882,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Other-relative",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 2205,
        "hours_per_week": 40,
        "native_country": "Holand-Netherlands"
    }

    request = client.post("/predictions/", data=json.dumps(sample))

    assert request.status_code == OK_STATUS_CODE
    assert request.json()["prediction"] == "<=50k"
