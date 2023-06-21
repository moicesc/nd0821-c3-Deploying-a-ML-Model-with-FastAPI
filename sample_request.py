"""
Script to create a valid request to the Census API
Author: Moises Gonzalez
Date: 21/Jun/2023
"""

import requests
import json
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

url = "http://127.0.0.1:7000/predictions/"

sample = {
    "age": 35,
    "workclass": "Private",
    "fnlgt": 7777,
    "education": "Some-college",
    "education_num": 10,
    "marital_status": "Never-married",
    "occupation": "Armed-Forces",
    "relationship": "Not-in-family",
    "race": "Other",
    "sex": "Male",
    "capital_gain": 2000,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "Holand-Netherlands"
  }

response = requests.post(url, data=json.dumps(sample))

logger.info(f"Sample prediction -> {response.json()}")
