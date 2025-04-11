# %%
import pandas as pd
from model import test
from typing import List
import requests
import json

# %%
# prepare inputs first
data_path = f"../data/process/test.csv"
test_df = pd.read_csv(data_path)
inputs: List[str] = test_df['mention'].to_list()
test_inputs = inputs[:100]

# # execute model only
# def test_model(test_inputs):
#     # test model execution
#     model_result = test(test_inputs)
#     # check the output
#     return model_result
# 
# print(test_model(test_inputs))
# %%
# execute model via api endpoint
def test_api(test_inputs):
    # note: you must start the fastapi server: fastapi dev predict.py
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "texts": test_inputs
    }

    response = requests.post(url, json=payload)

    print("Status Code:", response.status_code)
    data = response.json()
    pretty_dump = json.dumps(data, indent=4)

    return pretty_dump


print(test_api(test_inputs))