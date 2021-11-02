#!/usr/bin/env python
# coding: utf-8

import random
import pandas as pd
import requests
import json

random_state = 1234
random.seed(random_state)

url = 'http://localhost:9696/attrition_score'
stop = False

data_file = 'attrition_test_data.csv'
df_test = pd.read_csv(data_file)
df_test = df_test.reset_index(drop=True)
test_json = df_test.apply(lambda x: json.loads(x.to_json()), axis=1)

while not stop:
	idx = random.randint(0, len(test_json))
	employee = test_json.values[idx]

	response = requests.post(url, json=employee)
	result = response.json()
	print(f"employee: {employee}\n")
	print(f"result: {result}\n")

	print("Stop? (y/n)")
	stop = input() == 'y'

print("Stopped.")