#!/usr/bin/env python
# coding: utf-8


## Imports

import pickle

from flask import Flask, request, jsonify


app = Flask('Attrition App')

## Load ML model
model_file = '/output/attrition-model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
print(f"Machine Learning Model loaded from '%s'" % model_file)



def predict_attrition(employee):
    X = dv.transform([employee])
    y_pred = model.predict_proba(X)[:, 1]

    result = {
        'attrition_score': y_pred[0],
        'attrition': bool(y_pred[0] >= 0.5)
    }
    return(result)


@app.route('/attrition_score', methods=['POST'])
def attrition_score():
    employee = request.get_json()
    print(f"Received employee: {employee}")
    attrition_score = predict_attrition(employee)
    print("Attrition score: %s" % attrition_score)
    return(jsonify(attrition_score))

# Test
employee = {
    "age":48,
    "businesstravel":"Travel_Rarely",
    "dailyrate":969,
    "department":"Research & Development",
    "distancefromhome":2,
    "education":"College",
    "educationfield":"Technical Degree",
    "environmentsatisfaction":"very high",
    "gender":"Male",
    "hourlyrate":76,
    "jobinvolvement":"very high",
    "joblevel":1,
    "jobrole":"Laboratory Technician",
    "jobsatisfaction":"medium",
    "maritalstatus":"Single",
    "monthlyincome":2559,
    "monthlyrate":16620,
    "numcompaniesworked":5,
    "overtime":0,
    "percentsalaryhike":11,
    "performancerating":"excellent",
    "relationshipsatisfaction":"high",
    "stockoptionlevel":0,
    "totalworkingyears":7,
    "trainingtimeslastyear":4,
    "worklifebalance":"good",
    "yearsatcompany":1,
    "yearsincurrentrole":0,
    "yearssincelastpromotion":0,
    "yearswithcurrmanager":0
}
attrition_score = predict_attrition(employee)
print(f"Attrition score: {attrition_score}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
