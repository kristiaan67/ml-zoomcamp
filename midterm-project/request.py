#!/usr/bin/env python
# coding: utf-8

import sys, getopt
import random
import pandas as pd
import requests
import json

random_state = 1234
random.seed(random_state)


def main(argv):
    mode = 'local'

    try:
        opts, args = getopt.getopt(argv,"",["mode="])
    except getopt.GetoptError:
        print("INFO: python request.py --mode=local/aws)")
        sys.exit()

    for opt, arg in opts:
        if opt == '--mode':
            mode = arg

    if mode == 'local':
        url = 'http://localhost:9696/attrition_score'
    elif mode == 'aws':
        url = 'http://attrition-predict-env.eba-smk4cc3d.us-west-2.elasticbeanstalk.com/attrition_score'
    else:
        print("INFO: python request.py --mode=local/aws)")
        sys.exit()

    print("Using mode '%s' with endpoint '%s'" % (mode, url))
    data_file = 'attrition_test_data.csv'
    df_test = pd.read_csv(data_file)
    df_test = df_test.reset_index(drop=True)
    test_json = df_test.apply(lambda x: json.loads(x.to_json()), axis=1)

    stop = False
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


if __name__ == "__main__":
   main(sys.argv[1:])