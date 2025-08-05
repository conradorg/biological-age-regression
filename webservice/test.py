import sys
import requests

# from src.data_utils import load_preprocessed_data_parquet
# print(load_preprocessed_data_parquet("data/prepared/2017.parquet").iloc[0])

# Example input
sample = {
    "RIDAGEYR": 66.0,
    "RIAGENDR": "female",
    "BMXHT": 158.3,
    "BMXWAIST": 101.8,
    "BMXWT": 79.5,
    "PAD680": 300.0,
    "SMQ020": "yes",
    "ALQ130": 1,
    "SLD012": 8.0,
}
# "phenoage":    65.060415,

url = "http://localhost:9696/predict"
response = requests.post(url, json=sample)
print(response.json())
