import sys
import requests

# from src.data_utils import load_preprocessed_data_parquet
# print(load_preprocessed_data_parquet("data/prepared/2017.parquet").iloc[0])

# Example input
sample = {
    "RIDAGEYR": 66.0,  # age
    "RIAGENDR": "female",  # gender
    "BMXHT": 158.3,  # body height
    "BMXWAIST": 101.8,  # body waist
    "BMXWT": 79.5,  # weight
    "PAD680": 300.0,  # minutes sitting
    "SMQ020": "yes",  # smoked 100 cigaretes in life
    "ALQ130": 1,  # average number of alcoholoc drinks per day
    "SLD012": 8.0,  # number of hours sleep on weekdays/workdays
}
# "phenoage":    65.060415,  # ground truth calculated from biomarkers (also invasive ones e.g. by blood samples)

url = "http://localhost:9696/predict"
response = requests.post(url, json=sample)
print(response.json())
